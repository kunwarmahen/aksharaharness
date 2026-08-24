"""End-to-end integration: real adapters, real SSE fixtures, real loop.

Phase 6's proof. A MockTransport serves the streamed tool-call fixture on
request 1 and the plain-text fixture on request 2, so the ENTIRE pipeline
runs without a network:

    wire bytes -> iter_sse_lines -> parse_events -> StreamEvents -> collect()
    -> Agent loop -> real tools -> history -> wire encoding of round 2

Everything upstream of here was tested piecewise; this pins the seams
BETWEEN the pieces -- especially that the tool results produced by
execution are encoded back onto each wire correctly (the formats diverge
exactly there: tool_result blocks vs fan-out role:"tool" messages).
"""

from __future__ import annotations

import json

import httpx
import pytest

from akshara.agent import Agent, ToolExecuted, TurnEnd
from akshara.permissions import allow_read_only
from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.base import ProviderSettings
from akshara.providers.openai import OpenAIProvider
from akshara.tools import default_registry

from conftest import Recorder, load_fixture

README_TEXT = "# demo project\n\nhello from the sandbox\n"

EXPECTED = {
    "anthropic": {
        "cls": AnthropicProvider,
        "model": "claude-sonnet-4-5-20250929",
        "call_ids": ["toolu_01ABC"],           # fixture makes ONE call
        "result_id_key": "tool_use_id",
    },
    "openai": {
        "cls": OpenAIProvider,
        "model": "gpt-4o-mini",
        "call_ids": ["call_001", "call_002"],  # ...and TWO (interleaved)
        "result_id_key": "tool_call_id",
    },
}


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "README.md").write_text(README_TEXT)
    (tmp_path / "src").mkdir()
    return tmp_path


def _make_agent(provider_name: str, tmp_path) -> tuple[Agent, Recorder]:
    spec = EXPECTED[provider_name]
    script = [load_fixture(f"{provider_name}_tool.sse"),      # iteration 1
              load_fixture(f"{provider_name}_text.sse")]      # iteration 2
    recorder = Recorder(lambda req: httpx.Response(
        200, content=script.pop(0),
        headers={"content-type": "text/event-stream"},
    ))
    provider = spec["cls"](ProviderSettings(api_key="k", base_url="http://mock.local"),
                           transport=httpx.MockTransport(recorder))
    agent = Agent(provider, model=spec["model"], system=None,
                  tools=default_registry(), cwd=tmp_path,
                  permissions=allow_read_only)
    return agent, recorder


@pytest.mark.parametrize("provider_name", sorted(EXPECTED))
def test_full_streamed_tool_turn(provider_name, sandbox):
    spec = EXPECTED[provider_name]
    agent, recorder = _make_agent(provider_name, sandbox)

    events = list(agent.run_streaming("what's in README.md?"))

    # ---- exactly two model calls, then the script would be empty ----
    assert len(recorder.calls) == 2

    # ---- fragment reassembly survived the REAL adapters ----
    executed = [e for e in events if isinstance(e, ToolExecuted)]
    assert [e.call.id for e in executed] == spec["call_ids"]
    read_result = next(e.result for e in executed if e.call.name == "read_file")
    assert read_result.is_error is False
    assert "hello from the sandbox" in read_result.content  # real file, really read
    for e in executed:
        assert e.call.arguments == {"path": "README.md"} or \
               e.call.arguments == {"path": "src"}

    # ---- history shape: user / assistant / user(results) / assistant ----
    roles = [m.role for m in agent.history]
    assert roles == ["user", "assistant", "user", "assistant"]
    final = agent.history[3]
    assert final.text() == "Hello there"                    # canonical fixture

    # ---- the turn ended properly, usage accumulated across iterations ----
    end = events[-1]
    assert isinstance(end, TurnEnd) and end.reason == "end_turn"
    # tool fixture reports 42in/55out, text fixture 17in/9out -- identical
    # totals for both providers, which is itself a nice normalization check
    assert agent.total_usage.input_tokens == 59
    assert agent.total_usage.output_tokens == 64


@pytest.mark.parametrize("provider_name", sorted(EXPECTED))
def test_results_encoded_back_in_native_wire_shape(provider_name, sandbox):
    """THE divergence this whole project normalizes away: how executed
    results travel BACK to the provider."""
    spec = EXPECTED[provider_name]
    agent, recorder = _make_agent(provider_name, sandbox)
    list(agent.run_streaming("go"))

    body = json.loads(recorder.calls[1].content)

    match provider_name:
        case "anthropic":
            # results ride INSIDE the next user message as blocks
            last = body["messages"][-1]
            assert last["role"] == "user"
            result_blocks = [b for b in last["content"]
                             if b.get("type") == "tool_result"]
            assert [b["tool_use_id"] for b in result_blocks] == spec["call_ids"]
            assert any("hello from the sandbox" in b["content"] for b in result_blocks)
            # and the assistant turn before it carried tool_use blocks
            prev = body["messages"][-2]
            assert any(b.get("type") == "tool_use" for b in prev["content"])
        case "openai":
            # results fan out into one role:"tool" message PER call
            tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
            assert [m[spec["result_id_key"]] for m in tool_msgs] == spec["call_ids"]
            assert any("hello from the sandbox" in m["content"] for m in tool_msgs)
            # preceded by an assistant message carrying the tool_calls array
            idx = body["messages"].index(tool_msgs[0]) - 1
            assistant = body["messages"][idx]
            assert assistant["role"] == "assistant"
            assert [tc["id"] for tc in assistant["tool_calls"]] == spec["call_ids"]
