"""MiniMax-M3 reasoning parser (<mm:think>, 3 thinking modes) and tool-call
detector (namespace-delimited recursive XML) -- one-shot and streaming."""

from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import (
    Function,
    MiniMaxM3Detector,
    Tool,
)
from freetoken.server.reasoning_parser import MiniMaxM3ReasoningParser, ReasoningParser

NS = "]<]minimax[>["


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------
def test_reasoning_registered():
    assert ReasoningParser.ReasoningParserEnum["minimax_m3"] is MiniMaxM3ReasoningParser


def test_reasoning_adaptive_with_tags():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("<mm:think>chain of thought</mm:think>The answer is 4.")
    assert r.reasoning_text == "chain of thought"
    assert r.normal_text == "The answer is 4."


def test_reasoning_adaptive_without_tags():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    r = p.detect_and_parse("Plain answer, the model chose not to think.")
    assert r.reasoning_text == ""
    assert r.normal_text == "Plain answer, the model chose not to think."


def test_reasoning_enabled_implicit_open():
    # thinking_mode=enabled: the template pre-opens <mm:think>, the model emits
    # only the closing tag -> the caller constructs with force_reasoning=True.
    p = MiniMaxM3ReasoningParser(force_reasoning=True)
    r = p.detect_and_parse("thinking hard</mm:think>final answer")
    assert r.reasoning_text == "thinking hard"
    assert r.normal_text == "final answer"


def test_reasoning_streaming_split_marker():
    p = MiniMaxM3ReasoningParser(force_reasoning=False)
    reasoning, content = [], []
    for chunk in ["<mm:th", "ink>let me ", "think</mm:t", "hink>done: 42"]:
        r = p.parse_streaming_increment(chunk)
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = p.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    assert "".join(reasoning) == "let me think"
    assert "".join(content) == "done: 42"


def test_reasoning_tool_block_ends_missing_close():
    # A malformed turn skipping </mm:think> straight into a tool block: the block
    # must land in normal_text for the tool parser (dsv4 precedent).
    p = MiniMaxM3ReasoningParser(force_reasoning=True)
    text = f"half a thought{NS}<tool_call>...{NS}</tool_call>"
    r = p.detect_and_parse(text)
    assert r.reasoning_text == "half a thought"
    assert r.normal_text.startswith(f"{NS}<tool_call>")


# ---------------------------------------------------------------------------
# Tool-call detector
# ---------------------------------------------------------------------------
def _tools():
    return [
        Tool(
            function=Function(
                name="create_order",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "note": {"type": "string"},
                        "shipping": {"type": "object"},
                        "items": {"type": "array"},
                    },
                },
            )
        ),
        Tool(function=Function(name="get_weather", parameters={
            "type": "object", "properties": {"city": {"type": "string"}},
        })),
    ]


def _block(inner: str) -> str:
    return f"{NS}<tool_call>\n{inner}{NS}</tool_call>"


_NESTED_INVOKE = (
    f'{NS}<invoke name="create_order">'
    f"{NS}<user_id>42{NS}</user_id>"
    f"{NS}<note>two books{NS}</note>"
    f"{NS}<shipping>"
    f"{NS}<city>Singapore{NS}</city>"
    f"{NS}<zip>018956{NS}</zip>"
    f"{NS}</shipping>"
    f"{NS}<items>"
    f"{NS}<item>"
    f"{NS}<sku>book-001{NS}</sku>"
    f"{NS}<qty>2{NS}</qty>"
    f"{NS}</item>"
    f"{NS}</items>"
    f"{NS}</invoke>\n"
)


def test_detect_and_parse_recursive_args():
    det = MiniMaxM3Detector()
    text = "I'll place the order." + _block(_NESTED_INVOKE)
    assert det.has_tool_call(text)
    res = det.detect_and_parse(text, _tools())
    assert res.normal_text == "I'll place the order."
    assert len(res.calls) == 1
    call = res.calls[0]
    assert call.name == "create_order"
    args = json.loads(call.parameters)
    assert args["user_id"] == 42  # schema-typed integer
    assert args["note"] == "two books"  # schema-typed string
    # nested leaves are loose-JSON typed: numbers parse, but a leading-zero token
    # like "018956" is not valid JSON and stays a string (never silently mangled)
    assert args["shipping"] == {"city": "Singapore", "zip": "018956"}
    assert args["items"] == [{"sku": "book-001", "qty": 2}]


def test_detect_and_parse_multiple_invokes():
    det = MiniMaxM3Detector()
    two = (
        _NESTED_INVOKE
        + f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n'
    )
    res = det.detect_and_parse(_block(two), _tools())
    assert [c.name for c in res.calls] == ["create_order", "get_weather"]
    assert json.loads(res.calls[1].parameters) == {"city": "Paris"}


def test_detect_and_parse_unknown_tool_follows_forwarding_policy():
    # FreeToken forwards unknown tool names by default (FORWARD_UNKNOWN_TOOLS);
    # the detector must honor the same policy switch as every other family.
    import freetoken.server.function_call_parser as fcp

    det = MiniMaxM3Detector()
    text = _block(f'{NS}<invoke name="nope">{NS}<a>1{NS}</a>{NS}</invoke>\n')
    res = det.detect_and_parse(text, _tools())
    assert [c.name for c in res.calls] == ["nope"]  # default: forwarded

    orig = fcp.FORWARD_UNKNOWN_TOOLS
    try:
        fcp.FORWARD_UNKNOWN_TOOLS = False
        res = MiniMaxM3Detector().detect_and_parse(text, _tools())
        assert res.calls == []  # strict mode: dropped with a warning
    finally:
        fcp.FORWARD_UNKNOWN_TOOLS = orig


def test_streaming_text_then_call():
    det = MiniMaxM3Detector()
    full = "Let me check." + _block(
        f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n'
    )
    # Chop into small chunks that split the namespace marker mid-token.
    chunks = [full[i : i + 7] for i in range(0, len(full), 7)]
    normal, calls = "", []
    for ch in chunks:
        r = det.parse_streaming_increment(ch, _tools())
        normal += r.normal_text
        calls.extend(r.calls)
    assert normal == "Let me check."
    named = [c for c in calls if c.name]
    frags = [c for c in calls if c.name is None and c.parameters]
    assert [c.name for c in named] == ["get_weather"]
    assert json.loads("".join(c.parameters for c in frags)) == {"city": "Paris"}
    # ledgers the serving layer reads at stream end
    assert det.prev_tool_call_arr[0]["name"] == "get_weather"
    assert det.prev_tool_call_arr[0]["arguments"] == {"city": "Paris"}


def test_streaming_multiple_invokes_and_indices():
    det = MiniMaxM3Detector()
    two = (
        _NESTED_INVOKE
        + f'{NS}<invoke name="get_weather">{NS}<city>Paris{NS}</city>{NS}</invoke>\n'
    )
    full = _block(two)
    chunks = [full[i : i + 13] for i in range(0, len(full), 13)]
    calls = []
    for ch in chunks:
        calls.extend(det.parse_streaming_increment(ch, _tools()).calls)
    named = [(c.tool_index, c.name) for c in calls if c.name]
    assert named == [(0, "create_order"), (1, "get_weather")]


def test_streaming_plain_text_never_held():
    det = MiniMaxM3Detector()
    r = det.parse_streaming_increment("Just a normal answer, no tools.", _tools())
    assert r.normal_text == "Just a normal answer, no tools."
    assert r.calls == []


def test_auto_selection_picks_minimax_m3():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {
                "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
                "model_type": "minimax_m3_vl",
                "text_config": {"model_type": "minimax_m2"},
            }

    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/anon"])
    assert args.tool_call_parser == "minimax_m3"
    assert args.reasoning_parser == "minimax_m3"


@pytest.mark.parametrize(
    "gear,mode", [("off", "disabled"), ("adaptive", "adaptive"), ("on", "enabled")]
)
def test_think_gears(gear, mode):
    from freetoken.server.model_meta import think_chat_template_kwargs, think_spec

    gears, default = think_spec("minimax_m3")
    assert gears == ("off", "adaptive", "on") and default == "adaptive"
    assert think_chat_template_kwargs("minimax_m3", gear) == {"thinking_mode": mode}
