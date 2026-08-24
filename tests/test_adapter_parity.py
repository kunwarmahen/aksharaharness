"""Parity: the property that makes the provider abstraction real.

1. Within an adapter: collect(stream()) must equal complete() -- the
   event vocabulary is lossless.
2. Across adapters: equivalent wire exchanges normalize to equivalent
   ModelResponses -- the loop never needs to know which dialect spoke.

The fixtures encode the SAME conversation ("Hello there", 17 in / 9 out)
in both dialects; only the model names differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.base import ProviderSettings, collect
from akshara.providers.openai import OpenAIProvider
from akshara.providers.responses import ResponsesProvider
from akshara.types import Message, TextBlock, ToolCall, ToolResult, Usage

FIXTURES = Path(__file__).parent / "fixtures"

PROVIDERS = {
    "anthropic": (AnthropicProvider, "anthropic_settings"),
    "openai": (OpenAIProvider, "openai_settings"),
    "responses": (ResponsesProvider, "responses_settings"),
}

DIALECTS = list(PROVIDERS)


def _make(provider_name: str, settings, responder):
    cls = PROVIDERS[provider_name][0]
    return cls(settings, transport=httpx.MockTransport(responder))


def _sse_responder(body: bytes):
    return lambda request: httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"}
    )


def _json_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize("provider_name", DIALECTS)
def test_collect_stream_equals_complete(provider_name, request):
    """Within one adapter: streaming and non-streaming agree."""
    settings = request.getfixturevalue(PROVIDERS[provider_name][1])
    sse_body = (FIXTURES / f"{provider_name}_text.sse").read_bytes()
    json_body = _json_fixture(f"{provider_name}_text.json")

    streamed = collect(
        _make(provider_name, settings, _sse_responder(sse_body)).stream(
            messages=[Message("user", [TextBlock("hi")])],
            system=None, tools=[], model="m", max_tokens=100,
        )
    )
    direct = _make(provider_name, settings, lambda r: httpx.Response(200, json=json_body)).complete(
        messages=[Message("user", [TextBlock("hi")])],
        system=None, tools=[], model="m", max_tokens=100,
    )

    assert streamed.message == direct.message  # same blocks, same order
    assert streamed.stop_reason == direct.stop_reason == "end_turn"
    assert streamed.usage == direct.usage == _usage(17, 9)


def _usage(inp: int, out: int):
    return Usage(input_tokens=inp, output_tokens=out)


def test_cross_provider_normalization_equivalence(request):
    """Equivalent exchanges through ALL adapters -> equivalent replies.

    This is what lets the agent loop be written exactly once.
    """
    normalized = {}
    for provider_name in DIALECTS:
        settings = request.getfixturevalue(PROVIDERS[provider_name][1])
        sse_body = (FIXTURES / f"{provider_name}_text.sse").read_bytes()
        response = collect(
            _make(provider_name, settings, _sse_responder(sse_body)).stream(
                messages=[Message("user", [TextBlock("hi")])],
                system=None, tools=[], model="m", max_tokens=100,
            )
        )
        normalized[provider_name] = response

    expected = Usage(input_tokens=17, output_tokens=9)
    for provider_name, response in normalized.items():
        assert response.message.text() == "Hello there", provider_name
        assert response.stop_reason == "end_turn", provider_name
        assert response.usage == expected, provider_name


def test_tool_round_trip_request_shapes_diverge_predictably(request):
    """Same internal history -> dialect-appropriate tool result encoding."""
    history = [
        Message("user", [TextBlock("read it")]),
        Message("assistant", [ToolCall(id="t1", name="read_file",
                                       arguments={"path": "README.md"})]),
        Message("user", [ToolResult(tool_call_id="t1", content="hello file")]),
    ]

    bodies = {}
    # Minimal valid completion bodies -- we only care about the REQUEST.
    minimal_ok = {
        "anthropic": {"id": "msg_1", "role": "assistant", "model": "m",
                      "content": [{"type": "text", "text": "ok"}],
                      "stop_reason": "end_turn", "usage":
                      {"input_tokens": 1, "output_tokens": 1}},
        "openai": {"choices": [{"message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop"}], "usage": {}},
        "responses": {"id": "resp_1", "object": "response", "status": "completed",
                      "model": "m",
                      "output": [{"type": "message", "id": "msg_1",
                                  "status": "completed", "role": "assistant",
                                  "content": [{"type": "output_text",
                                               "text": "ok"}]}],
                      "usage": {"input_tokens": 1, "output_tokens": 1}},
    }
    for provider_name in DIALECTS:
        settings = request.getfixturevalue(PROVIDERS[provider_name][1])
        sent: list[httpx.Request] = []

        def handler(r: httpx.Request) -> httpx.Response:
            sent.append(r)
            return httpx.Response(200, json=minimal_ok[provider_name])

        _make(provider_name, settings, handler).complete(
            messages=history, system=None, tools=[], model="m", max_tokens=100,
        )
        bodies[provider_name] = json.loads(sent[0].content)

    # Anthropic: tool_result is a BLOCK inside a user message
    anthropic_last = bodies["anthropic"]["messages"][-1]
    assert anthropic_last["role"] == "user"
    assert anthropic_last["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "hello file"}
    ]
    # OpenAI: it became its own role:"tool" message
    assert bodies["openai"]["messages"][-1] == {
        "role": "tool", "tool_call_id": "t1", "content": "hello file",
    }
    # Responses: it became its own function_call_output INPUT ITEM
    assert bodies["responses"]["input"][-1] == {
        "type": "function_call_output", "call_id": "t1", "output": "hello file",
    }
