"""Retry-policy tests: budgets, backoff shape, and THE STREAMING RULE.

The rule under test: retries cover only the OPENING of a conversation.
A flaky connection before status-line -> retried. A failure after events
have reached the caller -> propagates untouched, because replaying a
half-delivered stream would duplicate output.

Sleeps are intercepted (akshara.providers.retry._sleep) so the suite
stays offline-fast while still asserting real delay decisions.
"""

from __future__ import annotations

import httpx
import pytest

from conftest import load_fixture

from akshara.errors import AuthError, ProviderError, RateLimitError
from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.base import ProviderSettings
from akshara.providers.retry import RetryPolicy, connect_with_retries, delay_for
from akshara.types import Message, TextBlock


def _provider(settings: ProviderSettings, responder) -> tuple[AnthropicProvider, list]:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return responder(request)

    provider = AnthropicProvider(
        settings,
        transport=httpx.MockTransport(handler),
        retry=RetryPolicy(max_attempts=5, base_delay=1.0),
    )
    return provider, sent


def _complete(provider: AnthropicProvider) -> None:
    provider.complete(messages=[Message("user", [TextBlock("hi")])], system=None,
                      tools=[], model="m", max_tokens=100)


@pytest.fixture
def sleeps(monkeypatch):
    """Capture backoff waits instead of taking them."""
    waited: list[float] = []
    monkeypatch.setattr("akshara.providers.retry._sleep", waited.append)
    return waited


# ---- policy unit -----------------------------------------------------------


class TestDelayFor:
    def test_terminal_statuses_never_retry(self):
        assert delay_for(AuthError("no", status=401), 1, RetryPolicy()) is None
        assert delay_for(ProviderError("bad", status=400), 1, RetryPolicy()) is None

    def test_retry_after_wins_over_backoff(self):
        err = RateLimitError("slow down", status=429, retry_after=7.0)
        assert delay_for(err, 1, RetryPolicy()) == 7.0

    def test_transport_errors_are_retryable(self):
        assert delay_for(httpx.ConnectError("refused"), 1, RetryPolicy()) is not None

    def test_exponential_growth_with_jitter_bounded(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
        # attempt n backs off base*2**(n-1) plus U(0, base) jitter
        for attempt, low in ((1, 1.0), (2, 2.0), (3, 4.0)):
            d = delay_for(ProviderError("down", status=503), attempt, policy)
            assert low <= d <= low + 1.0
        # ...and never past the single-wait ceiling
        d = delay_for(ProviderError("down", status=503), 10, policy)
        assert d <= 30.0


# ---- integration: the opening is retried -------------------------------------


class TestOpeningRetried:
    def test_two_500s_then_success(self, anthropic_settings, sleeps):
        responses = iter([
            httpx.Response(500, json={"type": "error", "error": {
                "type": "api_error", "message": "boom"}}),
            httpx.Response(500, json={"type": "error", "error": {
                "type": "api_error", "message": "boom"}}),
            httpx.Response(200, json={"model": "m", "stop_reason": "end_turn",
                                      "content": [{"type": "text", "text": "ok"}],
                                      "usage": {}}),
        ])

        def responder(request):
            return next(responses)

        provider, sent = _provider(anthropic_settings, responder)
        _complete(provider)

        assert len(sent) == 3          # two failures, one success
        assert len(sleeps) == 2        # backed off between them

    def test_401_raises_immediately_without_sleeping(self, anthropic_settings, sleeps):
        provider, sent = _provider(
            anthropic_settings,
            lambda r: httpx.Response(401, json={"type": "error", "error": {
                "type": "authentication_error", "message": "bad key"}}),
        )
        with pytest.raises(AuthError):
            _complete(provider)
        assert len(sent) == 1 and sleeps == []

    def test_retry_after_header_is_honored_verbatim(self, anthropic_settings, sleeps):
        payload = {"type": "error", "error": {"type": "rate_limit_error",
                                              "message": "slow down"}}
        responses = iter([
            httpx.Response(429, json=payload, headers={"retry-after": "7"}),
            httpx.Response(200, json={"model": "m", "stop_reason": "end_turn",
                                      "content": [{"type": "text", "text": "ok"}],
                                      "usage": {}}),
        ])
        provider, sent = _provider(anthropic_settings, lambda r: next(responses))

        _complete(provider)

        assert sleeps == [7.0]         # server's hint, not our formula
        assert len(sent) == 2

    def test_attempt_budget_exhaustion_raises_last_error(self, anthropic_settings, sleeps):
        provider, sent = _provider(
            anthropic_settings,
            lambda r: httpx.Response(503, json={"type": "error", "error": {
                "type": "api_error", "message": "down"}}),
        )
        with pytest.raises(ProviderError) as excinfo:
            _complete(provider)
        assert excinfo.value.status == 503
        assert len(sent) == 5          # max_attempts counts TOTAL tries
        assert len(sleeps) == 4        # ...so only attempts-1 waits


# ---- THE STREAMING RULE ------------------------------------------------------


class TestStreamBodyNeverRetried:
    def test_mid_stream_failure_propagates_untouched(self, anthropic_settings, sleeps):
        """Status 200 arrives, THEN the stream errors. Retrying would replay
        already-delivered tokens -- so this must propagate on attempt 1."""
        ok_prefix = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
        )
        error_tail = (
            b"event: error\n"
            b'data: {"type":"error","error":{"type":"overloaded_error",'
            b'"message":"mid-stream boom"}}\n'
            b"\n"
        )
        calls = {"n": 0}

        def responder(request):
            calls["n"] += 1
            return httpx.Response(200, content=ok_prefix + error_tail,
                                  headers={"content-type": "text/event-stream"})

        provider, sent = _provider(anthropic_settings, responder)

        with pytest.raises(ProviderError, match="mid-stream boom"):
            list(provider.stream(messages=[Message("user", [TextBlock("hi")])],
                                 system=None, tools=[], model="m", max_tokens=100))
        assert calls["n"] == 1 and sleeps == []


class TestStreamOpeningRetried:
    def test_flaky_status_then_sse_success(self, anthropic_settings, sleeps):
        """The pre-yield opening participates in the same policy as
        complete(): a 503 before any event is safely retryable."""
        good = load_fixture("anthropic_text.sse")
        responses = iter([
            httpx.Response(503, json={"type": "error", "error": {
                "type": "api_error", "message": "overloaded"}}),
            httpx.Response(200, content=good,
                           headers={"content-type": "text/event-stream"}),
        ])
        provider, sent = _provider(anthropic_settings, lambda r: next(responses))

        events = list(provider.stream(messages=[Message("user", [TextBlock("hi")])],
                                      system=None, tools=[], model="m",
                                      max_tokens=100))

        assert len(sent) == 2
        assert len(events) >= 2        # StartEvent ... EndEvent arrived once


class TestPreFirstEventReconnect:
    """The PRE-FIRST-EVENT notch: a 200 whose body dies before its first
    event is indistinguishable from a failed opening -- re-opened under
    the same budgets, invisible to the caller. After ANY event has been
    forwarded the old rule returns: propagate, never replay."""

    @staticmethod
    def _sse_response(body) -> httpx.Response:
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    def test_reset_before_first_event_reopens_invisibly(self, anthropic_settings,
                                                        sleeps):
        good = load_fixture("anthropic_text.sse")
        calls = {"n": 0}

        def dying_body():
            # Status line + headers arrive; then the connection resets
            # before a single SSE block does.
            yield b""
            raise httpx.ReadError("connection reset by peer")

        def responder(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return self._sse_response(dying_body())
            return self._sse_response(good)

        provider, sent = _provider(anthropic_settings, responder)
        events = list(provider.stream(messages=[Message("user", [TextBlock("hi")])],
                                      system=None, tools=[], model="m",
                                      max_tokens=100))

        assert calls["n"] == 2         # re-opened exactly once
        assert len(sleeps) == 1        # ...under the same policy budgets
        assert len(sent) == 2
        assert sum(type(e).__name__ == "EndEvent" for e in events) == 1

    def test_reset_after_first_event_propagates_untouched(self, anthropic_settings,
                                                          sleeps):
        prefix = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"model":"m","usage":{}}}\n'
            b"\n"
        )

        def flaky_body():
            yield prefix               # one real event reaches the caller...
            raise httpx.ReadError("reset mid-stream")

        provider, sent = _provider(
            anthropic_settings,
            lambda r: self._sse_response(flaky_body()))

        collected = []
        with pytest.raises(httpx.ReadError):
            for event in provider.stream(messages=[Message("user", [TextBlock("hi")])],
                                         system=None, tools=[], model="m",
                                         max_tokens=100):
                collected.append(event)

        assert len(collected) == 1     # exactly what was forwarded -- no replay
        assert len(sent) == 1          # NEVER re-opened
        assert sleeps == []            # no budget burned on an unreplayable stream
