"""The Anthropic adapter against canned wire responses -- zero network.

These tests ARE the cheat-sheet: each assertion pins one cell of the
wire-format table (endpoint path, auth headers, mandatory max_tokens,
block encoding, stop-reason mapping, error taxonomy).
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
from akshara.providers.anthropic import API_VERSION, AnthropicProvider
from akshara.providers.base import ProviderSettings
from akshara.providers.retry import RetryPolicy
from akshara.types import (
    Message, RedactedThinkingBlock, TextBlock, ThinkingBlock,
    ToolCall, ToolResult, Usage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _text_fixture() -> dict:
    return json.loads((FIXTURES / "anthropic_text.json").read_text())


def _provider(settings: ProviderSettings, responder, *, retry=None) -> tuple[AnthropicProvider, list]:
    """Provider on MockTransport + the recorded requests (for assertions).

    ``retry`` passes a RetryPolicy straight through -- error-taxonomy
    tests use RetryPolicy(max_attempts=1) so retryable statuses raise
    immediately instead of exercising the backoff loop.
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return responder(request)

    kwargs = {"retry": retry} if retry is not None else {}
    provider = AnthropicProvider(settings, transport=httpx.MockTransport(handler),
                                 **kwargs)
    return provider, sent


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_anatomy(self, anthropic_settings):
        provider, sent = _provider(anthropic_settings, lambda r: httpx.Response(200, json=_text_fixture()))

        response = provider.complete(
            messages=[Message("user", [TextBlock("hi")])],
            system=None,
            tools=[],
            model="test-model",
            max_tokens=1234,
        )

        assert len(sent) == 1
        request = sent[0]
        # endpoint + auth headers are the adapter's contract with the wire
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == API_VERSION

        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 1234  # MANDATORY on this wire format
        assert body["stream"] is False
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]

        # normalized reply
        assert response.stop_reason == "end_turn"
        assert response.model == "claude-sonnet-4-5-20250929"
        assert response.message.text() == "Hello there"
        assert response.usage.input_tokens == 17
        assert response.usage.output_tokens == 9
        assert response.raw is not None and response.raw["id"].startswith("msg_")

    def test_system_prompt_is_toplevel_not_a_message(self, anthropic_settings):
        provider, sent = _provider(anthropic_settings, lambda r: httpx.Response(200, json=_text_fixture()))

        provider.complete(
            messages=[Message("user", [TextBlock("hi")])],
            system="Be terse.",
            tools=[],
            model="m",
            max_tokens=100,
        )

        body = json.loads(sent[0].content)
        assert body["system"] == "Be terse."
        assert all(m["role"] in ("user", "assistant") for m in body["messages"])

    def test_tool_blocks_encode_anthropically(self, anthropic_settings):
        provider, sent = _provider(anthropic_settings, lambda r: httpx.Response(200, json=_text_fixture()))

        messages = [
            Message(
                "assistant",
                [ToolCall(id="toolu_01", name="read_file",
                          arguments={"path": "a.txt", "limit": 5})],
            ),
            Message(
                "user",
                [
                    ToolResult(tool_call_id="toolu_01", content="file contents"),
                    ToolResult(tool_call_id="toolu_02", content="nope", is_error=True),
                    TextBlock("now summarize"),
                ],
            ),
        ]
        provider.complete(messages=messages, system=None, tools=[], model="m", max_tokens=100)

        wire = json.loads(sent[0].content)["messages"]
        assert wire[0]["content"] == [
            {"type": "tool_use", "id": "toolu_01", "name": "read_file",
             "input": {"path": "a.txt", "limit": 5}},
        ]
        # tool_result blocks live INSIDE a user message; is_error only when true
        assert wire[1]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_01", "content": "file contents"},
            {"type": "tool_result", "tool_use_id": "toolu_02", "content": "nope",
             "is_error": True},
            {"type": "text", "text": "now summarize"},
        ]

    def test_tools_encode_with_input_schema(self, anthropic_settings):
        from akshara.types import ToolSpec

        provider, sent = _provider(anthropic_settings, lambda r: httpx.Response(200, json=_text_fixture()))
        spec = ToolSpec(
            name="read_file",
            description="Read a file.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                          tools=[spec], model="m", max_tokens=100)

        body = json.loads(sent[0].content)
        assert body["tools"] == [{
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }]


# ---------------------------------------------------------------------------
# Response parsing edge cases
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_thinking_block_decodes_and_preserves_signature(self, anthropic_settings):
        """Thinking is a FIRST-CLASS block now: decoded with its signature
        (needed to send it back during tool loops), kept OUT of .text()."""
        fixture = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "thinking", "thinking": "internal reasoning",
                 "signature": "sig-abc"},
                {"type": "text", "text": "the answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        provider, _ = _provider(anthropic_settings, lambda r: httpx.Response(200, json=fixture))

        response = provider.complete(
            messages=[Message("user", [TextBlock("x")])], system=None,
            tools=[], model="m", max_tokens=100,
        )

        assert response.message.text() == "the answer"  # no placeholder pollution
        blocks = response.message.content
        assert isinstance(blocks[0], ThinkingBlock)
        assert blocks[0].thinking == "internal reasoning"
        assert blocks[0].signature == "sig-abc"

    def test_thinking_block_round_trips_onto_the_wire(self, anthropic_settings):
        """Encode side: signed blocks come back verbatim -- and UNSIGNED
        blocks still carry an explicit empty signature field. Gateways
        reject key-absent thinking blocks even when they emit unsigned
        ones themselves (differential-probed: "" => 200, absent => 400)."""
        message = Message("assistant", [
            ThinkingBlock("because...", signature="sig-xyz"),
            ThinkingBlock("unsigned upstream"),
            ToolCall("t1", "echo", {}),
        ])
        body = AnthropicProvider(anthropic_settings).build_request_body(
            messages=[message], system=None, tools=[], model="m", max_tokens=10,
        )
        blocks = [b for b in body["messages"][0]["content"]
                  if b.get("type") == "thinking"]
        assert blocks[0] == {"type": "thinking", "thinking": "because...",
                             "signature": "sig-xyz"}
        assert blocks[1] == {"type": "thinking", "thinking": "unsigned upstream",
                             "signature": ""}  # key present, ALWAYS

    def test_redacted_thinking_decodes_and_round_trips(self, anthropic_settings, capsys):
        """``redacted_thinking`` is encrypted reasoning with a CONTRACT:
        the ``data`` ciphertext must survive verbatim (the provider
        validates it on the next request). First-class decode + encode --
        NOT the unknown-block placeholder path."""
        fixture = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "redacted_thinking", "data":"Q0lQSEVSVEVYVA=="},
                {"type": "text", "text": "the answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        provider, _ = _provider(anthropic_settings,
                                lambda r: httpx.Response(200, json=fixture))

        response = provider.complete(
            messages=[Message("user", [TextBlock("x")])], system=None,
            tools=[], model="m", max_tokens=100,
        )

        assert response.message.text() == "the answer"  # no placeholder pollution
        (block,) = [b for b in response.message.content
                    if isinstance(b, RedactedThinkingBlock)]
        assert block.data == "Q0lQSEVSVEVYVA=="  # ciphertext untouched
        assert "unsupported" not in capsys.readouterr().err  # not a placeholder

        # ...and back onto the wire byte-verbatim
        body = AnthropicProvider(anthropic_settings).build_request_body(
            messages=[response.message], system=None, tools=[],
            model="m", max_tokens=10,
        )
        assert body["messages"][0]["content"][0] == {
            "type": "redacted_thinking", "data": block.data,
        }

    def test_unknown_block_becomes_visible_placeholder(self, anthropic_settings, capsys):
        fixture = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "server_tool_use", "input": {}},  # genuinely unknown
                {"type": "text", "text": "the answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        provider, _ = _provider(anthropic_settings, lambda r: httpx.Response(200, json=fixture))

        response = provider.complete(
            messages=[Message("user", [TextBlock("x")])], system=None,
            tools=[], model="m", max_tokens=100,
        )

        text = response.message.text()
        assert "[unsupported block: server_tool_use]" in text  # surfaced, not hidden
        assert text.endswith("the answer")
        assert "unsupported content block" in capsys.readouterr().err

    def test_missing_usage_decodes_as_zeros(self, anthropic_settings):
        fixture = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        }
        provider, _ = _provider(anthropic_settings, lambda r: httpx.Response(200, json=fixture))

        response = provider.complete(
            messages=[Message("user", [TextBlock("x")])], system=None,
            tools=[], model="m", max_tokens=100,
        )
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "payload", "expected"),
        [
            (401, {"type": "error", "error": {"type": "authentication_error",
                                              "message": "invalid x-api-key"}}, AuthError),
            (403, {"type": "error", "error": {"type": "permission_error",
                                              "message": "not allowed"}}, AuthError),
            (500, {"type": "error", "error": {"type": "api_error",
                                              "message": "boom"}}, ProviderError),
        ],
    )
    def test_http_status_maps_to_exception_family(self, anthropic_settings, status, payload, expected):
        provider, _ = _provider(
            anthropic_settings,
            lambda r: httpx.Response(status, json=payload),
            retry=RetryPolicy(max_attempts=1),  # taxonomy test, not backoff test
        )

        with pytest.raises(expected) as excinfo:
            provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                              tools=[], model="m", max_tokens=100)
        assert excinfo.value.status == status

    def test_rate_limit_carries_retry_after(self, anthropic_settings):
        payload = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        provider, _ = _provider(
            anthropic_settings,
            lambda r: httpx.Response(429, json=payload, headers={"retry-after": "7"}),
            retry=RetryPolicy(max_attempts=1),
        )

        with pytest.raises(RateLimitError) as excinfo:
            provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                              tools=[], model="m", max_tokens=100)
        assert excinfo.value.retry_after == 7.0

    def test_context_overflow_is_detected_and_typed(self, anthropic_settings):
        payload = {
            "type": "error",
            "error": {"type": "invalid_request_error",
                      "message": "prompt is too long: 250000 tokens > 200000 maximum"},
        }
        provider, _ = _provider(anthropic_settings, lambda r: httpx.Response(400, json=payload))

        with pytest.raises(ContextOverflowError):
            provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                              tools=[], model="m", max_tokens=100)

    def test_plain_bad_request_stays_provider_error(self, anthropic_settings):
        payload = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad field"},
        }
        provider, _ = _provider(anthropic_settings, lambda r: httpx.Response(400, json=payload))

        with pytest.raises(ProviderError) as excinfo:
            provider.complete(messages=[Message("user", [TextBlock("x")])], system=None,
                              tools=[], model="m", max_tokens=100)
        assert not isinstance(excinfo.value, ContextOverflowError)


class TestGatewayNulls:
    def test_usage_null_counters_decode_to_zero(self, anthropic_settings):
        """OpenRouter sends explicit nulls where the real API omits keys.
        None must never reach Usage fields -- it would poison .add()."""
        body = {
            "id": "msg_x", "type": "message", "role": "assistant",
            "model": "gateway/whatever",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4,
                      "cache_read_input_tokens": None,
                      "cache_creation_input_tokens": None},
        }
        provider = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json=body)),
        )
        response = provider.complete(messages=[Message("user", [TextBlock("q")])],
                                     system=None, tools=[], model="m")
        assert response.usage.cache_read_tokens == 0
        assert response.usage.cache_write_tokens == 0
        response.usage.add(Usage(input_tokens=1, output_tokens=1))  # must not raise
