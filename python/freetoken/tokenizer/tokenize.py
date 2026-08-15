from __future__ import annotations

import importlib.util
import json
import os
from types import ModuleType
from typing import Any, List

import torch
from freetoken.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase

#: Reasoning-effort values accepted by the DeepSeek-V4 encoder. Anything else
#: (notably OpenAI's default ``"medium"``) makes ``encoding_dsv4.render_message``
#: raise an assertion, so unsupported values are normalized to ``None``.
VALID_REASONING_EFFORTS = ("max", "high")


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


def normalize_reasoning_effort(value: Any | None) -> str | None:
    """Drop reasoning-effort values the dsv4 encoder cannot accept (-> ``None``)."""
    return value if value in VALID_REASONING_EFFORTS else None


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            add_special_tokens = True
            if isinstance(msg.text, list):
                chat_template_kwargs = msg.chat_template_kwargs or {}
                if self._dsv4_encoder is not None:
                    prompt = _apply_dsv4_chat_encoder(
                        self._dsv4_encoder,
                        msg.text,
                        msg.tools,
                        chat_template_kwargs,
                    )
                else:
                    if msg.tools is not None:
                        chat_template_kwargs = {**chat_template_kwargs, "tools": msg.tools}
                    prompt = self.tokenizer.apply_chat_template(
                        msg.text,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                    assert isinstance(prompt, str)
                    # The template owns every special token (HF's apply_chat_template
                    # tokenizes with add_special_tokens=False for the same reason):
                    # tokenizers that auto-add bos (muse-glimmer's, llama's) would
                    # otherwise double it -- the template already rendered one.
                    add_special_tokens = False
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=add_special_tokens
                )
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
) -> str:
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=normalize_reasoning_effort(chat_template_kwargs.get("reasoning_effort")),
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
