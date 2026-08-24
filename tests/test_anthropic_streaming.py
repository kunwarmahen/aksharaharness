"""The streaming half of the Anthropic adapter, against canned SSE bodies.

The event-sequence assertion is the normalization contract: it pins the
exact StreamEvents a given wire body must produce. collect() then folds
that sequence into the same ModelResponse complete() would return.
"""

from __future__ import annotations

import httpx
import pytest

from akshara.errors import AuthError, ProviderError
from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.base import ProviderSettings, collect
from akshara.types import (
    ThinkingBlock,
    EndEvent,
    Message,
    StartEvent,
    TextBlock,
    TextDelta,
    Usage,
)

from conftest import load_fixture


def _sse_provider(settings: ProviderSettings, body: bytes) -> AnthropicProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    return AnthropicProvider(settings, transport=httpx.MockTransport(handler))


def _stream(provider: AnthropicProvider):
    return provider.stream(
        messages=[Message("user", [TextBlock("hi")])],
        system=None,
        tools=[],
        model="m",
        max_tokens=100,
    )


class TestStreaming:
    def test_text_stream_event_sequence(self, anthropic_settings):
        provider = _sse_provider(anthropic_settings, load_fixture("anthropic_text.sse"))

        events = list(_stream(provider))

        assert events == [
            StartEvent(model="claude-sonnet-4-5-20250929"),
            TextDelta(text="Hello"),
            TextDelta(text=" there"),
            EndEvent(stop_reason="end_turn",
                     usage=Usage(input_tokens=17, output_tokens=9)),
        ]

    def test_collect_folds_to_model_response(self, anthropic_settings):
        provider = _sse_provider(anthropic_settings, load_fixture("anthropic_text.sse"))

        response = collect(_stream(provider))

        assert response.model == "claude-sonnet-4-5-20250929"
        assert response.message.text() == "Hello there"
        assert response.stop_reason == "end_turn"
        assert response.usage == Usage(input_tokens=17, output_tokens=9)

    def test_tool_arguments_accumulate_across_fragments(self, anthropic_settings):
        # Fixture splits {"path": "README.md"} mid-key: '{"pa' + 'th": ...'
        provider = _sse_provider(anthropic_settings, load_fixture("anthropic_tool.sse"))

        response = collect(_stream(provider))

        (call,) = response.message.tool_calls()
        assert (call.id, call.name) == ("toolu_01ABC", "read_file")
        assert call.arguments == {"path": "README.md"}
        assert response.message.text() == "Let me read that file."
        assert response.stop_reason == "tool_use"

    def test_error_status_before_stream_maps_like_complete(self, anthropic_settings):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"type": "error",
                      "error": {"type": "authentication_error", "message": "bad key"}},
            )

        provider = AnthropicProvider(anthropic_settings,
                                     transport=httpx.MockTransport(handler))
        with pytest.raises(AuthError):
            list(_stream(provider))

    def test_mid_stream_error_event_raises_provider_error(self, anthropic_settings):
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
            b"event: error\n"
            b'data: {"type":"error","error":{"type":"overloaded_error","message":"try again"}}\n'
            b"\n"
        )
        provider = _sse_provider(anthropic_settings, body)

        with pytest.raises(ProviderError, match="overloaded_error"):
            list(_stream(provider))

    def test_thinking_fragments_reassemble_with_signature(self, anthropic_settings):
        """thinking_delta prose + signature_delta must fold into ONE
        ThinkingBlock -- signature included -- so it can round-trip."""
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"thinking","thinking":""}}\n'
            b"\n"
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"thinking_delta","thinking":"think"}}\n'
            b"\n"
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"thinking_delta","thinking":"ing"}}\n'
            b"\n"
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"signature_delta","signature":"sig-1"}}\n'
            b"\n"
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":0}\n'
            b"\n"
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":1,'
            b'"content_block":{"type":"text","text":""}}\n'
            b"\n"
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"text_delta","text":"answer"}}\n'
            b"\n"
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{}}\n'
            b"\n"
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n'
            b"\n"
        )
        provider = _sse_provider(anthropic_settings, body)

        response = collect(_stream(provider))

        assert response.message.text() == "answer"
        (thought,) = [b for b in response.message.content
                      if isinstance(b, ThinkingBlock)]
        assert thought.thinking == "thinking"
        assert thought.signature == "sig-1"

    def test_redacted_thinking_arrives_whole_and_survives_collect(self, anthropic_settings):
        """redacted_thinking has NO deltas -- its ciphertext payload lands
        complete inside content_block_start. collect() must still record it:
        it round-trips verbatim like a signed block, and dropping it here
        would 400 the next request of the same tool loop."""
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"redacted_thinking",'
            b'"data":"RXhhbXBsZUNpcGhlcnRleHQ="}}\n'
            b"\n"
            b"event: content_block_stop\n"
            b'data: {"type":"content_block_stop","index":0}\n'
            b"\n"
            b"event: content_block_start\n"
            b'data: {"type":"content_block_start","index":1,'
            b'"content_block":{"type":"text","text":""}}\n'
            b"\n"
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"text_delta","text":"answer"}}\n'
            b"\n"
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{}}\n'
            b"\n"
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n'
            b"\n"
        )
        provider = _sse_provider(anthropic_settings, body)

        response = collect(_stream(provider))

        # arrival order preserved: ciphertext block first, then the text
        assert [type(b).__name__ for b in response.message.content] == [
            "RedactedThinkingBlock", "TextBlock"]
        assert response.message.content[0].data == "RXhhbXBsZUNpcGhlcnRleHQ="
        assert response.message.text() == "answer"
