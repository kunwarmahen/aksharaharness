"""The async provider layer: same wire logic, awaited plumbing.

Every test here has a sync twin elsewhere in the suite on purpose -- the
claim under test is NOT "async works" but "the async surface produces
BYTE-IDENTICAL protocol behavior with zero duplicated rules". Fixtures
are shared verbatim from the sync tests via conftest.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from akshara.errors import AuthError, ContextOverflowError, ProviderError, RateLimitError
from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.base import (
    ProviderSettings,
    acollect,
    collect,
)
from akshara.providers.fallback import FallbackProvider
from akshara.providers.openai import OpenAIProvider
from akshara.providers.retry import RetryPolicy
from akshara.providers.sse import aiter_sse_lines, aparse_events, iter_sse_lines, parse_events
from akshara.types import (
    EndEvent,
    Message,
    StartEvent,
    TextBlock,
    TextDelta,
    Usage,
)

from conftest import load_fixture


def _chunks(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


# ---------------------------------------------------------------------------
# SSE twins: async framing == sync framing, including hostile chunking
# ---------------------------------------------------------------------------


class TestAsyncSse:
    FIXTURES = ["anthropic_text.sse", "openai_tool.sse"]

    @pytest.mark.parametrize("name", FIXTURES)
    @pytest.mark.parametrize("size", [1, 3, 4096])  # 1 = mid-multibyte splits
    def test_async_lines_and_events_match_sync_exactly(self, name, size):
        data = load_fixture(name)

        async def achunks():
            for chunk in _chunks(data, size):
                yield chunk

        async def _run():
            lines = [line async for line in aiter_sse_lines(achunks())]
            events = [pair async for pair in aparse_events(aiter_sse_lines(achunks()))]
            return lines, events
        lines, events = asyncio.run(_run())

        assert lines == list(iter_sse_lines(_chunks(data, size)))
        assert events == list(parse_events(iter_sse_lines(_chunks(data, size))))


# ---------------------------------------------------------------------------
# Adapters: event parity + error mapping through the awaited transport
# ---------------------------------------------------------------------------


def _sse_handler(body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"})
    return handler


def _call(provider, *, tools=None):
    return provider.astream(
        messages=[Message("user", [TextBlock("hi")])],
        system=None,
        tools=tools or [],
        model="m",
        max_tokens=100,
    )


async def _drain(agen):
    return [event async for event in agen]


class TestAsyncAnthropic:
    def test_event_sequence_matches_the_sync_contract(self, anthropic_settings):
        provider = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(_sse_handler(load_fixture("anthropic_text.sse"))))
        events = asyncio.run(_drain(_call(provider)))

        assert events == [
            StartEvent(model="claude-sonnet-4-5-20250929"),
            TextDelta(text="Hello"),
            TextDelta(text=" there"),
            EndEvent(stop_reason="end_turn",
                     usage=Usage(input_tokens=17, output_tokens=9)),
        ]

    def test_acollect_folds_to_the_same_response_acomplete_returns(
            self, anthropic_settings):
        """The lossless-vocabulary proof, async edition."""
        import json as _json
        streaming = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(_sse_handler(load_fixture("anthropic_text.sse"))))
        nonstreaming = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(lambda req: httpx.Response(
                200, json=_json.loads(load_fixture("anthropic_text.json")))))

        async def _both():
            folded = await acollect(_call(streaming))
            direct = await nonstreaming.acomplete(
                messages=[Message("user", [TextBlock("hi")])],
                system=None, tools=[], model="m", max_tokens=100)
            return folded, direct

        folded, direct = asyncio.run(_both())
        assert folded.message.text() == direct.message.text() == "Hello there"
        assert folded.stop_reason == direct.stop_reason == "end_turn"
        assert folded.usage == direct.usage

    def test_tool_arguments_accumulate_across_fragments(self, anthropic_settings):
        provider = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(_sse_handler(load_fixture("anthropic_tool.sse"))))
        response = asyncio.run(acollect(_call(provider)))

        (call,) = response.message.tool_calls()
        assert call.arguments == {"path": "README.md"}  # split mid-key upstream
        assert response.stop_reason == "tool_use"

    def test_401_before_stream_maps_to_auth_error(self, anthropic_settings):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"type": "error",
                      "error": {"type": "authentication_error", "message": "bad key"}})

        provider = AnthropicProvider(anthropic_settings,
                                     transport=httpx.MockTransport(handler))
        with pytest.raises(AuthError):
            asyncio.run(_drain(_call(provider)))

    def test_mid_stream_error_event_raises_through_the_async_skin(
            self, anthropic_settings):
        body = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
            b"event: error\n"
            b'data: {"type":"error","error":{"type":"overloaded_error","message":"try again"}}\n'
            b"\n"
        )
        provider = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(_sse_handler(body)))
        with pytest.raises(ProviderError, match="overloaded_error"):
            asyncio.run(_drain(_call(provider)))


class TestAsyncOpenAI:
    def test_acollect_parity_with_acomplete(self, openai_settings):
        import json as _json
        streaming = OpenAIProvider(
            openai_settings,
            transport=httpx.MockTransport(_sse_handler(load_fixture("openai_text.sse"))))
        nonstreaming = OpenAIProvider(
            openai_settings,
            transport=httpx.MockTransport(lambda req: httpx.Response(
                200, json=_json.loads(load_fixture("openai_text.json")))))

        async def _both():
            folded = await acollect(_call(streaming))
            direct = await nonstreaming.acomplete(
                messages=[Message("user", [TextBlock("hi")])],
                system=None, tools=[], model="m", max_tokens=100)
            return folded, direct

        folded, direct = asyncio.run(_both())
        assert folded.message.text() == direct.message.text()
        assert folded.stop_reason == direct.stop_reason
        assert folded.usage == direct.usage

    def test_parallel_tool_calls_fold_through_the_async_skin(self, openai_settings):
        provider = OpenAIProvider(
            openai_settings,
            transport=httpx.MockTransport(_sse_handler(load_fixture("openai_tool.sse"))))

        response = asyncio.run(acollect(_call(provider)))

        first, second = response.message.tool_calls()
        assert (first.id, first.name, first.arguments) == (
            "call_001", "read_file", {"path": "README.md"})
        assert (second.id, second.name, second.arguments) == (
            "call_002", "list_dir", {"path": "src"})
        assert response.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# Async retry: budgets and waits, sleeps intercepted like the sync tests
# ---------------------------------------------------------------------------


class TestAsyncRetry:
    def test_429_then_success_retries_on_the_loop(self, anthropic_settings,
                                                  monkeypatch):
        # Unlike the sync tests' plain list.append, the interceptor itself
        # must be awaitable -- the retry loop AWAITS its wait.
        waited: list[float] = []

        async def _record(delay):
            waited.append(delay)

        monkeypatch.setattr("akshara.providers.retry._asleep", _record)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": {}})
            return httpx.Response(
                200, content=load_fixture("anthropic_text.sse"),
                headers={"content-type": "text/event-stream"})

        provider = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(handler),
            retry=RetryPolicy(max_attempts=3, base_delay=2.0))
        response = asyncio.run(acollect(_call(provider)))
        assert response.message.text() == "Hello there"
        assert calls["n"] == 2                       # retried once...
        assert 2.0 <= waited[0] <= 4.0               # ...after backoff+U(0,base)

    def test_budget_exhaustion_raises_the_last_error(self, anthropic_settings,
                                                     monkeypatch):
        async def _nosleep(_delay):
            return None

        monkeypatch.setattr("akshara.providers.retry._asleep", _nosleep)

        provider = AnthropicProvider(
            anthropic_settings,
            transport=httpx.MockTransport(lambda req: httpx.Response(500, text="down")),
            retry=RetryPolicy(max_attempts=3))
        with pytest.raises(ProviderError):
            asyncio.run(_drain(_call(provider)))


# ---------------------------------------------------------------------------
# Async fallback: commit point and taxonomy survive the await
# ---------------------------------------------------------------------------


class FakeAsyncProvider:
    """Duck-typed provider for failover tests -- no HTTP at all."""

    def __init__(self, name: str, *, opening_error: Exception | None = None,
                 events: tuple = (), die_after_first: Exception | None = None,
                 complete_error: Exception | None = None):
        self.name = name
        self._opening_error = opening_error
        self._events = events
        self._die_after_first = die_after_first
        self._complete_error = complete_error

    def astream(self, **kwargs):
        if self._opening_error is not None:
            raise self._opening_error
        return self._gen()

    async def _gen(self):
        first = True
        for event in self._events:
            yield event
            if first and self._die_after_first is not None:
                raise self._die_after_first
            first = False

    async def acomplete(self, **kwargs):
        if self._complete_error is not None:
            raise self._complete_error
        from akshara.types import ModelResponse
        return ModelResponse(message=Message("assistant", [TextBlock(f"from {self.name}")]),
                             stop_reason="end_turn", usage=Usage())

    def stream(self, **kwargs):  # pragma: no cover - sync path unused here
        raise AssertionError("sync path unused in async tests")

    def complete(self, **kwargs):  # pragma: no cover
        raise AssertionError("sync path unused in async tests")


EVENTS = (StartEvent(model="fb"), TextDelta(text="ok"),
          EndEvent(stop_reason="end_turn", usage=Usage()))


class TestAsyncFallback:
    async def _gather(self, agen):
        return [event async for event in agen]

    def test_opening_failure_fails_over_and_serves_from_fallback(self):
        primary = FakeAsyncProvider("dead", opening_error=RateLimitError("429"))
        secondary = FakeAsyncProvider("live", events=EVENTS)
        composite = FallbackProvider(primary, secondary)

        events = asyncio.run(self._gather(composite.astream(
            messages=[], system=None, tools=[], model="m")))

        assert [type(e).__name__ for e in events] == [
            "StartEvent", "TextDelta", "EndEvent"]

    def test_commit_point_no_replay_after_first_event(self):
        primary = FakeAsyncProvider("flaky", events=(StartEvent(model="p"),),
                                    die_after_first=ProviderError("midstream"))
        secondary = FakeAsyncProvider("would-duplicate", events=EVENTS)
        composite = FallbackProvider(primary, secondary)

        with pytest.raises(ProviderError, match="midstream"):
            asyncio.run(self._gather(composite.astream(
                messages=[], system=None, tools=[], model="m")))

    def test_request_shaped_failure_never_fails_over(self):
        overflow = ContextOverflowError("400 context overflow: prompt too long")
        primary = FakeAsyncProvider("a", opening_error=overflow)
        secondary = FakeAsyncProvider("b", events=EVENTS)
        composite = FallbackProvider(primary, secondary)

        with pytest.raises(ContextOverflowError):
            asyncio.run(self._gather(composite.astream(
                messages=[], system=None, tools=[], model="m")))

    def test_acomplete_fails_over_like_astream(self):
        primary = FakeAsyncProvider("dead", complete_error=RateLimitError("429"))
        secondary = FakeAsyncProvider("live")
        composite = FallbackProvider(primary, secondary)

        async def _run():
            healthy = await secondary.acomplete(messages=[], system=None,
                                                tools=[], model="m")
            failed_over = await composite.acomplete(messages=[], system=None,
                                                    tools=[], model="m")
            return healthy, failed_over

        healthy, failed_over = asyncio.run(_run())
        assert healthy.message.text() == "from live"      # sanity: secondary fine
        assert failed_over.message.text() == "from live"  # composite landed there
