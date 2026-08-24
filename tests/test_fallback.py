"""FallbackProvider: failover across SPACE with retry's TIME discipline.

The one rule everything here pins: failover happens ONLY before the
first event crosses the boundary. Past that, the exchange is committed
-- replaying it means duplicate tokens -- so failures propagate no
matter how healthy the next provider looks.
"""

from __future__ import annotations

import pytest
from conftest import ScriptedProvider, assistant_text

from akshara.agent import Agent
from akshara.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
)
from akshara.permissions import yolo
from akshara.providers.base import collect
from akshara.providers.fallback import FallbackProvider
from akshara.types import StartEvent, TextDelta, Usage


def healthy(text: str = "ok", **kw) -> ScriptedProvider:
    return ScriptedProvider([assistant_text(text, **kw)])


def failing(exc: Exception) -> ScriptedProvider:
    """A provider that dies during the OPENING (before any event)."""
    provider = ScriptedProvider([])

    def stream(**kwargs):
        raise exc
        yield  # pragma: no cover -- makes this a generator

    provider.stream = stream  # type: ignore[method-assign]
    provider.complete = lambda **kwargs: (_ for _ in ()).throw(exc)
    return provider


class DiesMidStream(ScriptedProvider):
    """Delivers real events, THEN dies -- the committed case."""

    def __init__(self) -> None:
        super().__init__([])

    def stream(self, **kwargs):
        yield StartEvent(model="m")
        yield TextDelta("partial ")
        raise ProviderError("connection reset mid-stream")


def make(primary, secondary) -> FallbackProvider:
    return FallbackProvider(primary, secondary)


class TestOpeningOnly:
    def test_healthy_primary_never_touches_the_fallback(self):
        primary, secondary = healthy("primary answer"), healthy()
        fb = make(primary, secondary)

        response = collect(fb.stream(messages=[], system=None, tools=[],
                                     model="m"))

        assert "primary answer" in response.message.text()
        assert secondary.requests == []

    @pytest.mark.parametrize("exc", [
        RateLimitError("429 slow down", status=429, retry_after=0),
        AuthError("401 bad key", status=401),
        ProviderError("503 upstream dead", status=503),
    ])
    def test_provider_side_opening_failure_fails_over(self, exc):
        primary, secondary = failing(exc), healthy("backup answer")
        fb = make(primary, secondary)

        response = collect(fb.stream(messages=[], system=None, tools=[],
                                     model="m"))

        # auth can't be FIXED by waiting but CAN by another venue --
        # that is why fallback exists next to retry at all
        assert "backup answer" in response.message.text()
        assert len(secondary.requests) == 1

    def test_midstream_failure_propagates_no_replay(self):
        secondary = healthy("backup")
        fb = make(DiesMidStream(), secondary)

        with pytest.raises(ProviderError, match="mid-stream"):
            collect(fb.stream(messages=[], system=None, tools=[], model="m"))

        assert secondary.requests == []  # commit point passed: no failover

    @pytest.mark.parametrize("exc", [
        ContextOverflowError("400 prompt is too long", status=400),
        ProviderError("400 bad tool schema", status=400),
    ])
    def test_request_shaped_failures_fail_fast(self, exc):
        # the SAME bytes go to every backend -- changing venue cannot
        # heal them, so don't pay to find out
        secondary = healthy("backup")
        fb = make(failing(exc), secondary)

        with pytest.raises(type(exc)):
            collect(fb.stream(messages=[], system=None, tools=[], model="m"))

        assert secondary.requests == []


class TestExhaustion:
    def test_all_failed_aggregates_every_reason_last_wins_as_cause(self):
        first = failing(RateLimitError("429: primary throttled",
                                       status=429, retry_after=0))
        second = failing(AuthError("401: fallback key invalid", status=401))
        fb = make(first, second)

        with pytest.raises(ProviderError) as info:
            collect(fb.stream(messages=[], system=None, tools=[], model="m"))

        message = str(info.value)
        assert "all providers failed" in message
        assert "primary throttled" in message   # earlier attempts ride along
        assert "fallback key invalid" in message
        assert isinstance(info.value.__cause__, AuthError)  # last error wins

    def test_complete_falls_through_too(self):
        fb = make(failing(ProviderError("502", status=502)),
                  healthy("complete from backup"))

        response = fb.complete(messages=[], system=None, tools=[], model="m")

        assert "backup" in response.message.text()

    def test_needs_two_providers(self):
        with pytest.raises(ValueError, match="at least two"):
            FallbackProvider(healthy())


class TestShape:
    def test_name_describes_the_chain(self):
        assert make(healthy(), healthy()).name == "scripted -> scripted"

    def test_agent_treats_it_like_any_provider(self):
        agent = Agent(make(failing(RateLimitError("429", status=429,
                                                  retry_after=0)),
                           healthy(usage=Usage(input_tokens=3,
                                               output_tokens=2))),
                      model="m", permissions=yolo)
        response = agent.run("q")
        assert "ok" in response.message.text()
