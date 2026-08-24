"""The Responses API adapter against canned wire responses -- zero network.

Third dialect, same discipline: every difference between this file,
test_anthropic_adapter.py, and test_openai_adapter.py is a cell in the
wire cheat-sheet (README / notes/19).
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
from akshara.providers.responses import (
    REASONING_INDEX,
    ResponsesProvider,
    ResponsesStreamRouter,
)
from akshara.types import (
    EndEvent,
    ImageBlock,
    Message,
    StartEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCall,
    ToolCallStart,
    ToolResult,
    ToolSpec,
    Usage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _provider(settings: ProviderSettings, responder, *, retry=None):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return responder(request)

    kwargs = {"retry": retry} if retry is not None else {}
    return (ResponsesProvider(settings, transport=httpx.MockTransport(handler),
                              **kwargs),
            sent)


def _sse_bytes() -> bytes:
    return (FIXTURES / "responses_text.sse").read_bytes()


def _streaming_responder(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_sse_bytes(),
                          headers={"content-type": "text/event-stream"})


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_anatomy(self, responses_settings):
        provider, sent = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")))

        provider.complete(
            messages=[Message("user", [TextBlock("hi")])],
            system="Be terse.",
            tools=[],
            model="test-model",
            max_tokens=1234,
        )

        request = sent[0]
        assert request.url.path == "/v1/responses"  # base_url includes /v1
        assert request.headers["authorization"] == "Bearer test-key"

        body = json.loads(request.content)
        # system prompt rides top-level as `instructions` (like Anthropic),
        # NOT as a system-role message (chat-completions style).
        assert body["instructions"] == "Be terse."
        assert body["input"] == [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ]
        # chat-completions' max_tokens renamed on this wire.
        assert body["max_output_tokens"] == 1234
        assert body["stream"] is False
        assert "store" not in body and "previous_response_id" not in body

    def test_tools_are_flat_definitions(self, responses_settings):
        """No {"type":"function","function":{...}} wrapper here."""
        spec = ToolSpec(name="get_weather", description="Weather.",
                        parameters={"type": "object", "properties": {}})
        provider, sent = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")))

        provider.complete(messages=[Message("user", [TextBlock("hi")])],
                          system=None, tools=[spec], model="m", max_tokens=100)

        body = json.loads(sent[0].content)
        assert body["tools"] == [
            {"type": "function", "name": "get_weather",
             "description": "Weather.",
             "parameters": {"type": "object", "properties": {}}},
        ]

    def test_tool_round_trip_fans_out_into_items(self, responses_settings):
        """Assistant calls become function_call items; each result becomes
        its own function_call_output item keyed by the SAME call_id."""
        provider, sent = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")))

        messages = [
            Message("assistant", [
                TextBlock("Checking two things."),
                ToolCall(id="call_001", name="read_file",
                         arguments={"path": "a.txt"}),
                ToolCall(id="call_002", name="list_dir", arguments={"path": "."}),
            ]),
            Message("user", [
                ToolResult(tool_call_id="call_001", content="contents"),
                ToolResult(tool_call_id="call_002", content="nope", is_error=True),
            ]),
        ]
        provider.complete(messages=messages, system=None, tools=[],
                          model="m", max_tokens=100)

        items = json.loads(sent[0].content)["input"]
        # Assistant turn -> one message item + one function_call item per call
        assert items[0] == {
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "Checking two things."}],
        }
        assert items[1] == {
            "type": "function_call", "call_id": "call_001", "name": "read_file",
            "arguments": "{\"path\": \"a.txt\"}",  # JSON string on the wire
        }
        assert items[2]["call_id"] == "call_002"
        # Results -> function_call_output items; errors marked in the TEXT
        # (this wire has no is_error flag either).
        assert items[3] == {
            "type": "function_call_output", "call_id": "call_001",
            "output": "contents",
        }
        assert items[4] == {
            "type": "function_call_output", "call_id": "call_002",
            "output": "ERROR: nope",
        }

    def test_images_become_input_image_parts(self, responses_settings):
        provider, sent = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")))

        messages = [Message("user", [
            TextBlock("what is this?"),
            ImageBlock(media_type="image/png", data="aGk="),
        ])]
        provider.complete(messages=messages, system=None, tools=[],
                          model="m", max_tokens=100)

        content = json.loads(sent[0].content)["input"][0]["content"]
        assert content == [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,aGk="},
        ]

    def test_thinking_blocks_dropped_on_encode(self, responses_settings):
        """Display-only reasoning: nothing to send it back through."""
        from akshara.types import ThinkingBlock
        provider, sent = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")))

        messages = [Message("assistant", [
            ThinkingBlock("hmm", signature="sig"),
            TextBlock("answer"),
        ])]
        provider.complete(messages=messages, system=None, tools=[],
                          model="m", max_tokens=100)

        items = json.loads(sent[0].content)["input"]
        assert len(items) == 1  # only the message item survived
        assert items[0]["content"][0]["text"] == "answer"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_fixture(self, responses_settings):
        provider, _ = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")))
        resp = provider.complete(messages=[Message("user", [TextBlock("hi")])],
                                 system=None, tools=[], model="m", max_tokens=100)
        assert resp.message.text() == "Hello there"
        assert resp.stop_reason == "end_turn"
        assert resp.usage == Usage(input_tokens=17, output_tokens=9)
        assert resp.model == "gpt-4o-mini"

    def test_function_call_and_reasoning(self, responses_settings):
        raw = {
            "id": "resp_1", "status": "completed", "model": "o4-mini",
            "output": [
                {"type": "reasoning", "id": "rs_1",
                 "summary": [{"type": "summary_text", "text": "Need weather."}]},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "status": "completed",
                 "content": [{"type": "output_text", "text": "Checking."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_9",
                 "name": "get_weather",
                 "arguments": "{\"city\": \"SF\"}"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5,
                      "input_tokens_details": {"cached_tokens": 4}},
        }
        provider, _ = _provider(responses_settings,
                                lambda r: httpx.Response(200, json=raw))
        resp = provider.complete(messages=[Message("user", [TextBlock("hi")])],
                                 system=None, tools=[], model="m", max_tokens=100)

        blocks = resp.message.content
        assert isinstance(blocks[0], ThinkingBlock)  # display-only summary
        assert blocks[0].thinking == "Need weather."
        assert isinstance(blocks[1], TextBlock)
        # Arguments arrive as a JSON STRING -> parsed exactly once, here.
        assert blocks[2] == ToolCall(id="call_9", name="get_weather",
                                     arguments={"city": "SF"})
        assert resp.stop_reason == "tool_use"  # any function_call => tool_use
        assert resp.usage.cache_read_tokens == 4

    def test_incomplete_maps_to_max_tokens(self, responses_settings):
        raw = {"id": "resp_1", "status": "incomplete", "model": "m",
               "output": [],
               "incomplete_details": {"reason": "max_output_tokens"},
               "usage": {}}
        provider, _ = _provider(responses_settings,
                                lambda r: httpx.Response(200, json=raw))
        resp = provider.complete(messages=[Message("user", [TextBlock("hi")])],
                                 system=None, tools=[], model="m", max_tokens=100)
        assert resp.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_text_stream_event_sequence(self, responses_settings):
        provider, sent = _provider(responses_settings, _streaming_responder)
        events = list(provider.stream(
            messages=[Message("user", [TextBlock("hi")])],
            system=None, tools=[], model="m", max_tokens=100))

        assert sent[0].url.path == "/v1/responses"
        body = json.loads(sent[0].content)
        assert body["stream"] is True
        # No stream_options on this wire: usage rides the terminal event.
        assert "stream_options" not in body

        assert isinstance(events[0], StartEvent)
        assert events[0].model == "gpt-4o-mini"
        texts = [e.text for e in events if isinstance(e, TextDelta)]
        assert "".join(texts) == "Hello there"
        end = events[-1]
        assert isinstance(end, EndEvent)
        assert end.stop_reason == "end_turn"
        assert end.usage == Usage(input_tokens=17, output_tokens=9)

    def test_parity_with_fixture(self, responses_settings):
        """collect(stream()) == complete() against the SAME exchange."""
        streamed = collect(_provider(responses_settings, _streaming_responder)[0]
                           .stream(messages=[Message("user", [TextBlock("hi")])],
                                   system=None, tools=[], model="m",
                                   max_tokens=100))
        direct = _provider(
            responses_settings,
            lambda r: httpx.Response(200, json=_fixture("responses_text.json")),
        )[0].complete(messages=[Message("user", [TextBlock("hi")])],
                      system=None, tools=[], model="m", max_tokens=100)
        assert streamed.message == direct.message
        assert streamed.stop_reason == direct.stop_reason
        assert streamed.usage == direct.usage


class TestStreamRouter:
    """Unit tests over the router itself: gateway-variance edge cases."""

    @staticmethod
    def _events(pairs: list[tuple[str | None, str]]) -> list:
        router = ResponsesStreamRouter()
        out = []
        for name, data in pairs:
            out.extend(router.feed(name, data))
        out.extend(router.finish())
        return out

    def test_function_call_stream(self):
        events = self._events([
            ("response.created",
             '{"type":"response.created","response":{"id":"r1","model":"o4-mini"}}'),
            ("response.output_item.added",
             '{"type":"response.output_item.added","output_index":1,'
             '"item":{"type":"function_call","id":"fc_1","call_id":"call_7",'
             '"name":"bash","arguments":""}}'),
            ("response.function_call_arguments.delta",
             '{"type":"response.function_call_arguments.delta",'
             '"output_index":1,"delta":"{\\"command\\""}'),
            ("response.function_call_arguments.delta",
             '{"type":"response.function_call_arguments.delta",'
             '"output_index":1,"delta":": \\"ls\\"}"}'),
            ("response.completed",
             '{"type":"response.completed","response":{"id":"r1",'
             '"status":"completed","output":[{"type":"function_call",'
             '"call_id":"call_7","name":"bash","arguments":"{}"}],'
             '"usage":{"input_tokens":8,"output_tokens":3}}}'),
            ("[DONE]", "[DONE]"),
        ])
        start = next(e for e in events if isinstance(e, StartEvent))
        assert start.model == "o4-mini"
        starts = [e for e in events if isinstance(e, ToolCallStart)]
        assert [(s.index, s.id, s.name) for s in starts] == \
            [(1, "call_7", "bash")]
        end = events[-1]
        assert isinstance(end, EndEvent)
        assert end.stop_reason == "tool_use"
        assert end.usage == Usage(input_tokens=8, output_tokens=3)

    def test_arguments_done_without_deltas_synthesizes_one_delta(self):
        """Some backends send NO argument fragments -- just the final
        `.done`. The whole payload must still reach collect()."""
        events = self._events([
            ("response.output_item.added",
             '{"type":"response.output_item.added","output_index":0,'
             '"item":{"type":"function_call","call_id":"c1","name":"grep"}}'),
            ("response.function_call_arguments.done",
             '{"type":"response.function_call_arguments.done",'
             '"output_index":0,"arguments":"{\\"pattern\\": \\"x\\"}"}'),
            ("response.completed",
             '{"type":"response.completed","response":{"status":"completed",'
             '"output":[],"usage":{}}}'),
        ])
        deltas = [e for e in events if hasattr(e, "partial_json")]
        assert len(deltas) == 1
        assert json.loads(deltas[0].partial_json) == {"pattern": "x"}

    def test_arguments_embedded_in_output_item_done(self):
        """Last-resort delivery: minimal gateways put arguments only on
        the item-done envelope."""
        events = self._events([
            ("response.output_item.added",
             '{"type":"response.output_item.added","output_index":0,'
             '"item":{"type":"function_call","call_id":"c1","name":"ls"}}'),
            ("response.output_item.done",
             '{"type":"response.output_item.done","output_index":0,'
             '"item":{"type":"function_call","call_id":"c1","name":"ls",'
             '"arguments":"{\\"path\\": \\".\\"}"}}'),
            ("response.completed",
             '{"type":"response.completed","response":{"status":"completed",'
             '"output":[],"usage":{}}}'),
        ])
        deltas = [e for e in events if hasattr(e, "partial_json")]
        assert json.loads(deltas[0].partial_json) == {"path": "."}

    def test_openrouter_aliases_accepted(self):
        """OpenRouter documents response.done as the terminal event and
        content_part.delta for text -- both must work."""
        events = self._events([
            ("response.created",
             '{"type":"response.created","response":{"model":"m"}}'),
            ("response.content_part.delta",
             '{"type":"response.content_part.delta","delta":"hi"}'),
            ("response.done",
             '{"type":"response.done","response":{"status":"completed",'
             '"output":[],"usage":{"input_tokens":2,"output_tokens":1}}}'),
        ])
        assert any(isinstance(e, TextDelta) and e.text == "hi" for e in events)
        assert isinstance(events[-1], EndEvent)
        assert events[-1].usage.input_tokens == 2

    def test_missing_sse_name_falls_back_to_payload_type(self):
        events = self._events([
            (None, '{"type":"response.created","response":{"model":"m"}}'),
            (None, '{"type":"response.output_text.delta","delta":"yo"}'),
        ])
        assert isinstance(events[0], StartEvent)
        assert isinstance(events[1], TextDelta)

    def test_reasoning_summary_delta_uses_reserved_index(self):
        events = self._events([
            ("response.created",
             '{"type":"response.created","response":{"model":"m"}}'),
            ("response.reasoning_summary_text.delta",
             '{"type":"response.reasoning_summary_text.delta","delta":"hmm"}'),
        ])
        delta = events[1]
        assert isinstance(delta, ThinkingDelta)
        assert delta.index == REASONING_INDEX
        assert delta.text == "hmm"

    def test_error_event_raises_midstream(self):
        router = ResponsesStreamRouter()
        with pytest.raises(ProviderError, match="quota"):
            router.feed("error",
                        '{"type":"error","code":"insufficient_quota",'
                        '"message":"You exceeded your quota"}')

    def test_malformed_json_tolerated(self):
        router = ResponsesStreamRouter()
        assert router.feed(None, "not json at all") == []

    def test_truncated_stream_still_ends(self):
        router = ResponsesStreamRouter()
        first = router.feed("response.created",
                            '{"type":"response.created",'
                            '"response":{"model":"m"}}')
        assert isinstance(first[0], StartEvent)
        tail = router.finish()
        assert len(tail) == 1
        assert isinstance(tail[0], EndEvent)


# ---------------------------------------------------------------------------
# Error mapping + retry wiring
# ---------------------------------------------------------------------------


class TestErrors:
    def test_auth_error(self, responses_settings):
        body = {"error": {"message": "Invalid key",
                          "type": "invalid_request_error"}}
        provider, _ = _provider(
            responses_settings,
            lambda r: httpx.Response(401, json=body))
        with pytest.raises(AuthError):
            provider.complete(messages=[Message("user", [TextBlock("hi")])],
                              system=None, tools=[], model="m", max_tokens=100)

    def test_rate_limit_carries_retry_after(self, responses_settings):
        from akshara.providers.retry import RetryPolicy
        # max_attempts=1: raise immediately -- we are testing the ERROR's
        # payload here, not the retry loop (which would honor retry-after
        # and sleep for real).
        provider, _ = _provider(
            responses_settings,
            lambda r: httpx.Response(429, json={"error": {"message": "slow down"}},
                                     headers={"retry-after": "7"}),
            retry=RetryPolicy(max_attempts=1))
        with pytest.raises(RateLimitError) as exc_info:
            provider.complete(messages=[Message("user", [TextBlock("hi")])],
                              system=None, tools=[], model="m", max_tokens=100)
        assert exc_info.value.retry_after == 7.0

    def test_context_overflow_detected(self, responses_settings):
        provider, _ = _provider(
            responses_settings,
            lambda r: httpx.Response(
                400, json={"error": {"message":
                                     "context length exceeded"}}))
        with pytest.raises(ContextOverflowError):
            provider.complete(messages=[Message("user", [TextBlock("hi")])],
                              system=None, tools=[], model="m", max_tokens=100)

    def test_opening_5xx_retries_then_succeeds(self, responses_settings):
        """The retry skeleton is wired: a 500 before anything is delivered
        re-opens under the policy and succeeds."""
        attempts = {"n": 0}

        def flaky(_: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 2:
                return httpx.Response(500, json={"error": {"message": "boom"}})
            return httpx.Response(200, json=_fixture("responses_text.json"))

        from akshara.providers.retry import RetryPolicy
        provider, _ = _provider(
            responses_settings, flaky,
            retry=RetryPolicy(max_attempts=3, base_delay=0.0))
        resp = provider.complete(messages=[Message("user", [TextBlock("hi")])],
                                 system=None, tools=[], model="m", max_tokens=100)
        assert resp.message.text() == "Hello there"
        assert attempts["n"] == 2
