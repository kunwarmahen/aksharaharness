"""The OpenAI adapter against canned wire responses -- zero network.

Mirror of test_anthropic_adapter.py: same assertions, different dialect.
Reading the two side by side IS the lesson -- every difference between
these files is a cell in notes/02-wire-formats.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from akshara.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
)
from akshara.providers.base import ProviderSettings, collect
from akshara.providers.openai import OpenAIProvider
from akshara.providers.retry import RetryPolicy
from akshara.types import (
    EndEvent,
    Message,
    StartEvent,
    TextBlock,
    ThinkingBlock,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _provider(settings: ProviderSettings, responder, *, retry=None) -> tuple[OpenAIProvider, list]:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return responder(request)

    kwargs = {"retry": retry} if retry is not None else {}
    provider = OpenAIProvider(settings, transport=httpx.MockTransport(handler),
                              **kwargs)
    return provider, sent


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_anatomy(self, openai_settings):
        provider, sent = _provider(openai_settings, lambda r: httpx.Response(200, json=_fixture("openai_text.json")))

        response = provider.complete(
            messages=[Message("user", [TextBlock("hi")])],
            system="Be terse.",
            tools=[],
            model="test-model",
            max_tokens=1234,
        )

        request = sent[0]
        assert request.url.path == "/v1/chat/completions"  # base_url includes /v1
        assert request.headers["authorization"] == "Bearer test-key"
        assert "anthropic-version" not in request.headers  # different auth scheme

        body = json.loads(request.content)
        # system prompt is a MESSAGE here, not a top-level field
        assert body["messages"][0] == {"role": "system", "content": "Be terse."}
        assert body["messages"][1] == {"role": "user", "content": "hi"}
        assert body["max_tokens"] == 1234
        assert body["stream"] is False
        assert "stream_options" not in body  # only when streaming

    def test_streaming_sends_include_usage(self, openai_settings):
        provider, sent = _provider(
            openai_settings,
            lambda r: httpx.Response(200, content=(FIXTURES / "openai_text.sse").read_bytes(),
                                     headers={"content-type": "text/event-stream"}),
        )
        list(provider.stream(messages=[Message("user", [TextBlock("hi")])], system=None,
                             tools=[], model="m", max_tokens=100))
        assert json.loads(sent[0].content)["stream_options"] == {"include_usage": True}

    def test_tool_results_fan_out_into_role_tool_messages(self, openai_settings):
        """THE structural difference from Anthropic: one internal user
        Message becomes SEVERAL wire messages."""
        provider, sent = _provider(openai_settings, lambda r: httpx.Response(200, json=_fixture("openai_text.json")))

        messages = [
            Message("assistant", [
                TextBlock("Checking two things."),
                ToolCall(id="call_001", name="read_file", arguments={"path": "a.txt"}),
                ToolCall(id="call_002", name="list_dir", arguments={"path": "."}),
            ]),
            Message("user", [
                ToolResult(tool_call_id="call_001", content="contents"),
                ToolResult(tool_call_id="call_002", content="nope", is_error=True),
            ]),
        ]
        provider.complete(messages=messages, system=None, tools=[], model="m", max_tokens=100)

        wire = json.loads(sent[0].content)["messages"]
        assert wire[0]["role"] == "assistant"
        assert wire[0]["tool_calls"] == [
            {"id": "call_001", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}},
            {"id": "call_002", "type": "function",
             "function": {"name": "list_dir", "arguments": "{\"path\": \".\"}"}},
        ]
        # arguments were dicts internally -> JSON STRINGS on the wire
        # each ToolResult became its own role:"tool" message
        assert wire[1] == {"role": "tool", "tool_call_id": "call_001", "content": "contents"}
        assert wire[2] == {"role": "tool", "tool_call_id": "call_002",
                           "content": "ERROR: nope"}  # no is_error flag: marked in text

    def test_tools_encode_wrapped_in_function(self, openai_settings):
        provider, sent = _provider(openai_settings, lambda r: httpx.Response(200, json=_fixture("openai_text.json")))
        spec = ToolSpec(name="read_file", description="Read a file.",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}})
        provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                          tools=[spec], model="m", max_tokens=100)
        assert json.loads(sent[0].content)["tools"] == [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_text_response(self, openai_settings):
        provider, _ = _provider(openai_settings, lambda r: httpx.Response(200, json=_fixture("openai_text.json")))

        response = provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                                     tools=[], model="m", max_tokens=100)

        assert response.message.text() == "Hello there"
        assert response.stop_reason == "end_turn"  # finish_reason "stop"
        assert (response.usage.input_tokens, response.usage.output_tokens) == (17, 9)

    def test_null_content_plus_tool_calls(self, openai_settings):
        provider, _ = _provider(openai_settings, lambda r: httpx.Response(200, json=_fixture("openai_tool_use.json")))

        response = provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                                     tools=[], model="m", max_tokens=100)

        calls = response.message.tool_calls()
        assert [(c.id, c.name) for c in calls] == [("call_001", "read_file"), ("call_002", "list_dir")]
        assert calls[0].arguments == {"path": "README.md"}  # parsed dict, not string
        assert response.stop_reason == "tool_use"

    def test_unparseable_arguments_become_visible_sentinel(self, openai_settings):
        fixture = {
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": "call_9", "type": "function",
                                            "function": {"name": "f", "arguments": "{oops"}}]},
                "finish_reason": "tool_calls",
            }],
        }
        provider, _ = _provider(openai_settings, lambda r: httpx.Response(200, json=fixture))

        response = provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                                     tools=[], model="m", max_tokens=100)
        (call,) = response.message.tool_calls()
        assert call.arguments == {"_unparseable_json": "{oops"}

    def test_empty_arguments_string_parses_as_empty_dict(self, openai_settings):
        fixture = {
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": "call_9", "type": "function",
                                            "function": {"name": "ping", "arguments": ""}}]},
                "finish_reason": "tool_calls",
            }],
        }
        provider, _ = _provider(openai_settings, lambda r: httpx.Response(200, json=fixture))

        response = provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                                     tools=[], model="m", max_tokens=100)
        assert response.message.tool_calls()[0].arguments == {}


# ---------------------------------------------------------------------------
# Errors + streaming
# ---------------------------------------------------------------------------


class TestErrorsAndStreaming:
    @pytest.mark.parametrize(
        ("status", "payload", "expected"),
        [
            (401, {"error": {"message": "bad key", "type": "invalid_request_error"}}, AuthError),
            (429, {"error": {"message": "quota", "type": "rate_limit_error"}}, RateLimitError),
            (503, {"error": {"message": "down", "type": "server_error"}}, ProviderError),
        ],
    )
    def test_http_status_maps_to_exception_family(self, openai_settings, status, payload, expected):
        provider, _ = _provider(
            openai_settings,
            lambda r: httpx.Response(status, json=payload),
            retry=RetryPolicy(max_attempts=1),  # taxonomy test, not backoff test
        )
        with pytest.raises(expected):
            provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                              tools=[], model="m", max_tokens=100)

    def test_context_overflow_detected(self, openai_settings):
        payload = {"error": {"message": "This model's maximum context length is 128000 tokens",
                             "type": "invalid_request_error"}}
        provider, _ = _provider(openai_settings, lambda r: httpx.Response(400, json=payload))
        with pytest.raises(ContextOverflowError):
            provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                              tools=[], model="m", max_tokens=100)

    def _sse(self, settings, body: bytes) -> OpenAIProvider:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return OpenAIProvider(settings, transport=httpx.MockTransport(handler))

    def _args(self):
        return dict(messages=[Message("user", [TextBlock("hi")])], system=None,
                    tools=[], model="m", max_tokens=100)

    def test_text_stream_event_sequence(self, openai_settings):
        provider = self._sse(openai_settings, (FIXTURES / "openai_text.sse").read_bytes())
        events = list(provider.stream(**self._args()))
        assert events == [
            StartEvent(model="gpt-4o-mini"),
            TextDelta(text="Hello"),
            TextDelta(text=" there"),
            EndEvent(stop_reason="end_turn", usage=Usage(input_tokens=17, output_tokens=9)),
        ]

    def test_parallel_tool_calls_reassemble_from_interleaved_fragments(
        self, openai_settings
    ):
        """The hardest streaming case on this wire format: fragments for
        index 0 and index 1 interleave; id/name appear only once each."""
        provider = self._sse(openai_settings, (FIXTURES / "openai_tool.sse").read_bytes())

        response = collect(provider.stream(**self._args()))

        first, second = response.message.tool_calls()
        assert (first.id, first.name, first.arguments) == (
            "call_001", "read_file", {"path": "README.md"},
        )
        assert (second.id, second.name, second.arguments) == (
            "call_002", "list_dir", {"path": "src"},
        )
        assert response.stop_reason == "tool_use"

    def test_missing_done_sentinel_still_ends_cleanly(self, openai_settings):
        body = (
            b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"hey"}}]}\n\n'
        )
        provider = self._sse(openai_settings, body)
        events = list(provider.stream(**self._args()))
        assert isinstance(events[-1], EndEvent)


def test_usage_null_counters_decode_to_zero(openai_settings):
    body = {"id": "c1", "object": "chat.completion", "model": "gw/m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": 5}}
    provider = OpenAIProvider(
        openai_settings,
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)),
    )
    response = provider.complete(messages=[Message("user", [TextBlock("q")])],
                                 system=None, tools=[], model="m")
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 5


def test_reasoning_field_becomes_thinking_block(openai_settings):
    """Gateway convention: reasoning arrives as a message field (not typed
    blocks) and is display-only -- it must never reappear on requests."""
    body = {"id": "c1", "object": "chat.completion", "model": "gw/m",
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "reasoning": "pondering...",
                                     "content": "the answer"},
                         "finish_reason": "stop"}],
            "usage": {}}
    provider = OpenAIProvider(
        openai_settings,
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)),
    )
    response = provider.complete(messages=[Message("user", [TextBlock("q")])],
                                 system=None, tools=[], model="m")
    assert response.message.text() == "the answer"
    assert isinstance(response.message.content[0], ThinkingBlock)
    assert response.message.content[0].thinking == "pondering..."

    # ...and the encode side drops it silently (no wire representation):
    out = provider.build_request_body(messages=[response.message],
                                      system=None, tools=[], model="m",
                                      max_tokens=10)
    assert out["messages"] == [{"role": "assistant", "content": "the answer"}]
