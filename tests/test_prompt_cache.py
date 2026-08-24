"""Prompt caching: request-side breakpoints + response-side usage.

The wire rules under test (see notes/13-caching.md):

* Anthropic caching is OPT-IN per provider and expressed as up to four
  ``cache_control`` breakpoints on the request PREFIX. Our placement:
  last tool, system, last message block -- the stable-then-growing
  shape of an agent loop.
* OpenAI dialect has NOTHING to send: upstream caching is automatic,
  and hits surface only in ``usage.prompt_tokens_details.cached_tokens``.
* The flag must be invisible when off -- byte-identical requests to
  before this feature existed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.base import ProviderSettings, acollect, collect
from akshara.providers.openai import OpenAIProvider
from akshara.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolSpec,
    Usage,
)

TOOLS = [ToolSpec(name="echo", description="repeat", parameters={"type": "object"})]
MESSAGES = [
    Message("user", [TextBlock("old question")]),
    Message("assistant", [TextBlock("old answer")]),
    Message("user", [TextBlock("new question")]),
]


def _anthropic(cache_control: bool) -> tuple[AnthropicProvider, list[dict]]:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "model": "m", "role": "assistant", "type": "message",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    provider = AnthropicProvider(
        ProviderSettings(api_key="k", base_url="https://x.test"),
        transport=httpx.MockTransport(handler),
        cache_control=cache_control,
    )
    return provider, captured


def _complete(provider):
    return provider.complete(messages=MESSAGES, system="be brief",
                             tools=TOOLS, model="m", max_tokens=10)


def _breakpoints(body: dict) -> list:
    """Every cache_control marker in a request body, in wire order."""
    found = []
    for tool in body.get("tools") or []:
        if "cache_control" in tool:
            found.append(("tools", tool["name"]))
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if "cache_control" in block:
                found.append(("system", block.get("text", "")[:12]))
    for message in body.get("messages") or []:
        content = message.get("content") or []
        if isinstance(content, list):  # string content carries no markers
            for block in content:
                if "cache_control" in block:
                    found.append((message["role"], block.get("text", "?")[:12]))
    return found


class TestAnthropicRequestShape:
    def test_off_by_default_is_byte_identical_to_before(self):
        provider, captured = _anthropic(cache_control=False)
        _complete(provider)
        assert _breakpoints(captured[0]) == []
        assert isinstance(captured[0]["system"], str)  # not promoted to blocks

    def test_on_marks_tools_system_and_last_message(self):
        provider, captured = _anthropic(cache_control=True)
        _complete(provider)
        body = captured[0]
        # exactly our three placements, in prefix order
        assert _breakpoints(body) == [
            ("tools", "echo"),
            ("system", "be brief"),
            ("user", "new question"),
        ]
        # every marker is the one flavor the API offers
        for tool in body["tools"]:
            assert tool["cache_control"] == {"type": "ephemeral"}
        # within budget even after a fourth would be legal
        assert len(_breakpoints(body)) <= 4

    def test_system_promoted_to_block_array_only_when_caching(self):
        provider, captured = _anthropic(cache_control=True)
        _complete(provider)
        assert captured[0]["system"] == [{
            "type": "text", "text": "be brief",
            "cache_control": {"type": "ephemeral"},
        }]

    def test_optional_parts_shrink_the_breakpoint_set(self):
        provider, captured = _anthropic(cache_control=True)

        def _run(**kwargs):
            provider.complete(model="m", max_tokens=10,
                              messages=kwargs.pop("messages", MESSAGES),
                              **kwargs)

        _run(system=None, tools=[])
        body = captured[0]
        assert _breakpoints(body) == [("user", "new question")]
        assert "tools" not in body and "system" not in body

    def test_streaming_request_marked_the_same_way(self):
        sse = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":1}}\n'
            b"\n"
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        )
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, content=sse, headers={
                "content-type": "text/event-stream"})

        provider = AnthropicProvider(
            ProviderSettings(api_key="k", base_url="https://x.test"),
            transport=httpx.MockTransport(handler), cache_control=True)
        events = collect(provider.stream(messages=MESSAGES, system=None,
                                         tools=[], model="m", max_tokens=5))
        assert events.stop_reason == "end_turn"
        assert len(_breakpoints(captured[0])) == 1  # just the last message


class TestOpenAIDialect:
    def test_flag_is_a_no_op_there_is_nothing_to_send(self):
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "x", "model": "m", "choices": [{"index": 0,
                                                      "finish_reason": "stop",
                                                      "message": {"role": "assistant",
                                                                  "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })

        provider = OpenAIProvider(
            ProviderSettings(api_key="k", base_url="https://x.test/v1"),
            transport=httpx.MockTransport(handler), cache_control=True)
        provider.complete(messages=MESSAGES, system="s", tools=[],
                          model="m", max_tokens=10)
        blob = json.dumps(captured[0])
        assert "cache_control" not in blob

    def test_cached_tokens_surface_from_prompt_tokens_details(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "x", "model": "m", "choices": [{"index": 0,
                                                      "finish_reason": "stop",
                                                      "message": {"role": "assistant",
                                                                  "content": "ok"}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 2,
                          "prompt_tokens_details": {"cached_tokens": 256}},
            })

        provider = OpenAIProvider(
            ProviderSettings(api_key="k", base_url="https://x.test/v1"),
            transport=httpx.MockTransport(handler))
        response = provider.complete(messages=MESSAGES, system=None, tools=[],
                                     model="m", max_tokens=10)
        assert response.usage.cache_read_tokens == 256
        assert response.usage.cache_write_tokens == 0  # no write concept there

    def test_null_details_stay_zero_not_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "id": "x", "model": "m", "choices": [{"index": 0,
                                                      "finish_reason": "stop",
                                                      "message": {"role": "assistant",
                                                                  "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                          "prompt_tokens_details": None},
            })

        provider = OpenAIProvider(
            ProviderSettings(api_key="k", base_url="https://x.test/v1"),
            transport=httpx.MockTransport(handler))
        response = provider.complete(messages=MESSAGES, system=None, tools=[],
                                     model="m", max_tokens=10)
        assert response.usage.cache_read_tokens == 0


class TestAnthropicUsageFold:
    CACHE_USAGE = {"input_tokens": 100, "output_tokens": 7,
                   "cache_creation_input_tokens": 2117,
                   "cache_read_input_tokens": 88004}

    @staticmethod
    def _provider(extra_usage: dict, *, mode: str) -> AnthropicProvider:
        # one canned response per dialect skin -- a streaming body cannot
        # be served to complete() (JSON parse) or vice versa (see the
        # parity-test lesson in test_async_providers.py)
        if mode == "stream":
            sse = (
                b"event: message_start\n"
                b'data: {"type":"message_start","message":{"model":"m","usage":'
                + json.dumps(extra_usage).encode() + b'}}\n'
                b"\n"
                b"event: message_delta\n"
                b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                b'"usage":{"output_tokens":7}}\n'
                b"\n"
                b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
            )

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, content=sse, headers={
                    "content-type": "text/event-stream"})
        else:

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={
                    "id": "x", "model": "m", "role": "assistant",
                    "type": "message",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn", "usage": extra_usage})

        return AnthropicProvider(
            ProviderSettings(api_key="k", base_url="https://x.test"),
            transport=httpx.MockTransport(handler))

    @pytest.mark.parametrize("mode", ["stream", "complete"])
    def test_cache_counters_fold_through_both_skins(self, mode):
        provider = self._provider(dict(self.CACHE_USAGE), mode=mode)
        kwargs = dict(messages=MESSAGES, system=None, tools=[], model="m",
                      max_tokens=10)
        response = (collect(provider.stream(**kwargs)) if mode == "stream"
                    else provider.complete(**kwargs))
        assert response.usage.input_tokens == 100      # only the uncached part
        assert response.usage.output_tokens == 7
        assert response.usage.cache_write_tokens == 2117
        assert response.usage.cache_read_tokens == 88004

    def test_async_skin_folds_identically(self):
        import asyncio
        provider = self._provider({"input_tokens": 5, "output_tokens": 2,
                                   "cache_read_input_tokens": 42},
                                  mode="stream")

        async def _drain():
            return await acollect(provider.astream(
                messages=MESSAGES, system=None, tools=[], model="m",
                max_tokens=10))

        response = asyncio.run(_drain())
        assert response.usage.cache_read_tokens == 42
