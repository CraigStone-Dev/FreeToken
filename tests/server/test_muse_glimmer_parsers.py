"""Muse Glimmer ATEM reasoning parser (to=self / to=user / tool channels) and
tool-call detector (<atem:function_calls> invoke/parameter XML) -- one-shot and
streaming."""

from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import (
    Function,
    FunctionCallParser,
    MuseGlimmerDetector,
    Tool,
)
from freetoken.server.reasoning_parser import MuseGlimmerReasoningParser, ReasoningParser


def _tools():
    return [
        Tool(function=Function(name="weather.get", parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "object"},
            },
        })),
        Tool(function=Function(name="fs.write", parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        })),
    ]


def _atem(name: str, params: dict[str, str]) -> str:
    body = "".join(
        f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params.items()
    )
    return (
        f'<atem:function_calls>\n<atem:invoke name="{name}">\n{body}'
        f"</atem:invoke>\n</atem:function_calls>"
    )


def _tool_channel(name: str, params: dict[str, str], *, closer: str = "<|eot|>") -> str:
    return f"<|start|>assistant to={name}<|message|>{_atem(name, params)}{closer}"


def _stream(parser, text: str, step: int = 3):
    reasoning, content = [], []
    for i in range(0, len(text), step):
        r = parser.parse_streaming_increment(text[i : i + step])
        reasoning.append(r.reasoning_text)
        content.append(r.normal_text)
    r = parser.flush()
    reasoning.append(r.reasoning_text)
    content.append(r.normal_text)
    return "".join(reasoning), "".join(content)


def _stream_detect(det, text: str, tools, step: int = 3):
    normal, calls = [], []
    for i in range(0, len(text), step):
        r = det.parse_streaming_increment(text[i : i + step], tools)
        normal.append(r.normal_text)
        calls.extend(r.calls)
    while True:
        r = det.parse_streaming_increment("", tools)
        if not r.normal_text and not r.calls:
            break
        normal.append(r.normal_text)
        calls.extend(r.calls)
    normal.append(det.finish_streaming())
    return "".join(normal), calls


def _assemble(calls):
    """(name, joined-args-json) per streamed call, in order."""
    out: list[list] = []
    for c in calls:
        if c.name is not None:
            out.append([c.name, ""])
        if c.parameters:
            out[-1][1] += c.parameters
    return [(name, json.loads(args or "{}")) for name, args in out]


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------
def test_reasoning_registered():
    assert ReasoningParser.ReasoningParserEnum["muse_glimmer"] is MuseGlimmerReasoningParser


def test_reasoning_bare_first_segment_self_then_user():
    text = (
        " to=self<|message|>Let me think about this.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "Let me think about this."
    assert r.normal_text == "The answer is 42."


def test_reasoning_user_only_turn():
    r = MuseGlimmerReasoningParser().detect_and_parse(" to=user<|message|>Hi!<|eot|>")
    assert r.reasoning_text == "" and r.normal_text == "Hi!"


def test_reasoning_recipientless_header_is_content():
    r = MuseGlimmerReasoningParser().detect_and_parse("assistant<|message|>Plain.<|eot|>")
    assert r.reasoning_text == "" and r.normal_text == "Plain."


def test_reasoning_plain_text_passthrough():
    # Raw, non-templated output carries no ATEM markers and must pass through.
    text = "Just a plain answer, no channels."
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "" and r.normal_text == text
    p = MuseGlimmerReasoningParser()
    reasoning, content = _stream(p, text)
    assert reasoning == "" and content == text


def test_reasoning_tool_channel_preserved_verbatim_and_normalized():
    text = (
        " to=self<|message|>check weather<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        + _atem("weather.get", {"city": "Paris"})
        + "<|eot|>"
    )
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "check weather"
    assert r.normal_text.startswith("<|start|>assistant to=weather.get<|message|>")
    assert "<atem:function_calls>" in r.normal_text and r.normal_text.endswith("<|eot|>")

    # A tool call as the FIRST (bare) segment gets the opener prepended.
    bare = " to=weather.get<|message|>" + _atem("weather.get", {"city": "Rome"}) + "<|eot|>"
    r2 = MuseGlimmerReasoningParser().detect_and_parse(bare)
    assert r2.normal_text.startswith("<|start|>assistant to=weather.get<|message|>")


def test_reasoning_streaming_matches_one_shot():
    text = (
        " to=self<|message|>step one\nstep two<|eom|>"
        "<|start|>assistant to=user<|message|>Done: 42.<|eot|>"
    )
    one = MuseGlimmerReasoningParser().detect_and_parse(text)
    reasoning, content = _stream(MuseGlimmerReasoningParser(), text)
    assert reasoning == one.reasoning_text == "step one\nstep two"
    assert content == one.normal_text == "Done: 42."


def test_reasoning_streaming_holds_partial_markers():
    p = MuseGlimmerReasoningParser()
    out = p.parse_streaming_increment(" to=user<|message|>Hello <|eo")
    assert out.normal_text == "Hello "  # the partial closer is held, not leaked
    out = p.parse_streaming_increment("t|>")
    assert out.normal_text == ""
    r = p.flush()
    assert r.normal_text == "" and r.reasoning_text == ""


def test_reasoning_truncated_header_dropped_at_flush():
    p = MuseGlimmerReasoningParser()
    assert p.parse_streaming_increment(" to=se").normal_text == ""
    r = p.flush()
    assert r.normal_text == "" and r.reasoning_text == ""


def test_reasoning_multiple_self_segments_concatenate():
    text = (
        " to=self<|message|>alpha<|eom|>"
        "<|start|>assistant to=self<|message|>beta<|eom|>"
        "<|start|>assistant to=user<|message|>done<|eot|>"
    )
    r = MuseGlimmerReasoningParser().detect_and_parse(text)
    assert r.reasoning_text == "alphabeta"
    assert r.normal_text == "done"


# ---------------------------------------------------------------------------
# Tool-call detector
# ---------------------------------------------------------------------------
def test_detect_and_parse_typed_args():
    det = MuseGlimmerDetector()
    text = "One sec. " + _tool_channel(
        "weather.get", {"city": "Paris", "days": "3", "units": '{"temp": "C"}'}
    )
    assert det.has_tool_call(text)
    res = det.detect_and_parse(text, _tools())
    assert res.normal_text == "One sec."
    assert len(res.calls) == 1
    call = res.calls[0]
    assert call.name == "weather.get"
    args = json.loads(call.parameters)
    assert args == {"city": "Paris", "days": 3, "units": {"temp": "C"}}


def test_detect_and_parse_multiple_invokes_in_one_block():
    det = MuseGlimmerDetector()
    block = (
        "<atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls>"
    )
    res = det.detect_and_parse(block, _tools())
    assert [json.loads(c.parameters)["city"] for c in res.calls] == ["Paris", "Tokyo"]
    assert [c.tool_index for c in res.calls] == [0, 1]


def test_string_values_kept_verbatim():
    # The template's contract: "spaces for string values are not stripped".
    det = MuseGlimmerDetector()
    content = "line1\nline2\n"
    text = _tool_channel("fs.write", {"path": " /tmp/x ", "content": content})
    res = det.detect_and_parse(text, _tools())
    assert json.loads(res.calls[0].parameters) == {"path": " /tmp/x ", "content": content}

    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert _assemble(calls) == [("fs.write", {"path": " /tmp/x ", "content": content})]
    assert normal.strip() == ""


def test_streaming_matches_one_shot_with_channel_markup():
    text = "Checking. " + _tool_channel("weather.get", {"city": "Paris", "days": "3"})
    one = MuseGlimmerDetector().detect_and_parse(text, _tools())

    det = MuseGlimmerDetector()
    normal, calls = _stream_detect(det, text, _tools())
    assert normal.strip() == one.normal_text.strip() == "Checking."
    assert _assemble(calls) == [
        (c.name, json.loads(c.parameters)) for c in one.calls
    ] == [("weather.get", {"city": "Paris", "days": 3})]
    # ledgers the serving layer reads at stream end
    assert det.prev_tool_call_arr[0]["name"] == "weather.get"
    assert det.prev_tool_call_arr[0]["arguments"] == {"city": "Paris", "days": 3}


def test_streaming_bare_block_without_channel_header():
    normal, calls = _stream_detect(
        MuseGlimmerDetector(), "Look: " + _atem("weather.get", {"city": "Oslo"}), _tools()
    )
    assert normal.strip() == "Look:"
    assert _assemble(calls) == [("weather.get", {"city": "Oslo"})]


def test_streaming_bare_first_segment_tool_channel():
    # No reasoning parser upstream: generation starts with the header continuation.
    text = " to=weather.get<|message|>" + _atem("weather.get", {"city": "Lima"}) + "<|eot|>"
    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert normal.strip() == ""
    assert _assemble(calls) == [("weather.get", {"city": "Lima"})]


def test_streaming_self_channel_swallowed_without_reasoning_parser():
    text = (
        " to=self<|message|>secret chain of thought<|eom|>"
        "<|start|>assistant to=user<|message|>Visible.<|eot|>"
    )
    normal, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assert calls == []
    assert "secret" not in normal
    assert normal.strip() == "Visible."


def test_streaming_two_tool_channels_sequential_indices():
    text = (
        _tool_channel("weather.get", {"city": "Paris"}, closer="<|eom|>")
        + _tool_channel("weather.get", {"city": "Rome"})
    )
    _, calls = _stream_detect(MuseGlimmerDetector(), text, _tools())
    assembled = _assemble(calls)
    assert assembled == [
        ("weather.get", {"city": "Paris"}),
        ("weather.get", {"city": "Rome"}),
    ]
    named = [c.tool_index for c in calls if c.name is not None]
    assert named == [0, 1]


def test_streaming_trailing_text_defers_until_after_call():
    det = MuseGlimmerDetector()
    chunk = "Pre. " + _tool_channel("weather.get", {"city": "Paris"}) + " Post."
    r1 = det.parse_streaming_increment(chunk, _tools())
    assert r1.normal_text == "Pre. "
    assert [c.name for c in r1.calls if c.name] == ["weather.get"]
    r2 = det.parse_streaming_increment("", _tools())
    assert r2.normal_text == " Post." and r2.calls == []


def test_streaming_plain_text_never_held():
    det = MuseGlimmerDetector()
    r = det.parse_streaming_increment("Just a normal answer, no tools.", _tools())
    assert r.normal_text == "Just a normal answer, no tools."
    assert r.calls == []


def test_truncated_call_suppressed_and_recovered():
    # Generation cut mid-invoke (max_tokens). The call already started streaming
    # (its Start was emitted at the invoke open), so the serving layer closes it
    # from unstreamed_arguments (the detector's partial-parse ledger), and
    # finish_streaming must not leak the raw markup.
    parser = FunctionCallParser(_tools(), "muse_glimmer")
    det = parser.detector
    truncated = (
        "Ordering. <|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="days">4'
    )
    normal, calls = "", []
    for i in range(0, len(truncated), 11):
        r = det.parse_streaming_increment(truncated[i : i + 11], _tools())
        normal += r.normal_text
        calls.extend(r.calls)
    assert normal == "Ordering. "
    assert [c.name for c in calls if c.name] == ["weather.get"]
    # the completed parameter survives in the ledger; the truncated one is dropped
    assert json.loads(parser.unstreamed_arguments(0)) == {"city": "Paris"}
    assert det.finish_streaming() == ""


def test_unknown_tool_follows_forwarding_policy():
    import freetoken.server.function_call_parser as fcp

    text = _tool_channel("nope.call", {"a": "1"})
    res = MuseGlimmerDetector().detect_and_parse(text, _tools())
    assert [c.name for c in res.calls] == ["nope.call"]  # default: forwarded

    orig = fcp.FORWARD_UNKNOWN_TOOLS
    try:
        fcp.FORWARD_UNKNOWN_TOOLS = False
        res = MuseGlimmerDetector().detect_and_parse(text, _tools())
        assert res.calls == []
    finally:
        fcp.FORWARD_UNKNOWN_TOOLS = orig


def test_empty_parameters_invoke():
    det = MuseGlimmerDetector()
    block = (
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    res = det.detect_and_parse(block, _tools())
    assert json.loads(res.calls[0].parameters) == {}
    _, calls = _stream_detect(MuseGlimmerDetector(), block, _tools())
    assert _assemble(calls) == [("weather.get", {})]


# ---------------------------------------------------------------------------
# Wiring: auto-selection and thinking gears
# ---------------------------------------------------------------------------
def test_auto_selection_picks_muse_glimmer():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {
                "architectures": ["MuseGlimmerForConditionalGeneration"],
                "model_type": "muse_glimmer",
                "text_config": {"model_type": "muse_glimmer_text"},
            }

    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", "/models/anon"])
    assert args.tool_call_parser == "muse_glimmer"
    assert args.reasoning_parser == "muse_glimmer"


@pytest.mark.parametrize("gear", ["low", "medium", "high", "xhigh"])
def test_think_gears(gear):
    from freetoken.server.model_meta import think_chat_template_kwargs, think_spec

    gears, default = think_spec("muse_glimmer")
    assert gears == ("low", "medium", "high", "xhigh") and default == "high"
    assert think_chat_template_kwargs("muse_glimmer", gear) == {"reasoning_strength": gear}


def test_reasoning_effort_maps_to_reasoning_strength():
    from freetoken.server.model_meta import effort_toggle_kwargs

    assert effort_toggle_kwargs("muse_glimmer", "xhigh", None) == {"reasoning_strength": "xhigh"}
    assert effort_toggle_kwargs("muse_glimmer", "minimal", None) == {"reasoning_strength": "low"}
    # an explicit template kwarg wins wholesale
    assert effort_toggle_kwargs("muse_glimmer", "low", {"reasoning_strength": "high"}) == {
        "reasoning_strength": "high"
    }
    # muse has no off gear: effort "none" maps to no toggle at all
    assert effort_toggle_kwargs("muse_glimmer", "none", None) == {}
