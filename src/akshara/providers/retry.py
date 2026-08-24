"""Retry policy for transient provider failures.

Book-inspired (agent-harness-book ch5), adapted to sync generators:

* RETRY:      429 / 500 / 502 / 503 / 504, plus connection-level failures
              (httpx.TransportError -- DNS, connect, reset, timeout).
* NEVER:      400 / 401 / 403 / 404. A malformed request will not fix
              itself; neither will a wrong key.
* BACKOFF:    exponential with jitter --
                  wait = min(max_delay, base * 2**(attempt-1) + U(0, base))
              Jitter is load-bearing: without it every throttled client
              retries on the same schedule and re-creates the herd that
              caused the throttle.
* Retry-After wins when the provider sends it (429); overriding the
  server's own hint invites harder throttling.

THE STREAMING RULE this module exists to enforce, in two notches:

* OPENING: request -> status line is freely retryable (429/5xx/
  connection errors) -- nothing reached the caller.
* PRE-FIRST-EVENT: a stream that dies AFTER a 200 but BEFORE its first
  event is indistinguishable from a failed opening -- nobody has seen
  anything, so re-opening under the same budgets replays nothing. The
  adapters implement this notch with the same budget arithmetic
  (:func:`budgeted_delay`).
* POST-FIRST-EVENT: never. Replaying would duplicate tokens into
  someone's terminal; failures propagate (durable recovery is a
  checkpointer's job, not a retry loop's).

Budgets are hard ceilings on purpose: attempts AND total wall-clock.
Unbounded retry is how agents run up silent four-figure bills.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from akshara.errors import ProviderError, RateLimitError

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Module-level so tests can intercept without real sleeping.
_sleep = time.sleep
_asleep = asyncio.sleep


@dataclass(frozen=True)
class RetryPolicy:
    """Hard budgets, not suggestions."""

    max_attempts: int = 5          # TOTAL tries, including the first
    base_delay: float = 1.0        # seconds; also the jitter width
    max_delay: float = 30.0        # single-wait ceiling
    max_total_seconds: float = 120.0  # wall-clock ceiling across waits


def delay_for(error: Exception, attempt: int, policy: RetryPolicy) -> float | None:
    """How long to wait before ``attempt``+1, or None = do not retry."""
    status = getattr(error, "status", None)
    retryable = isinstance(error, httpx.TransportError) or status in RETRYABLE_STATUSES
    if not retryable:
        return None
    if isinstance(error, RateLimitError) and error.retry_after:
        return float(error.retry_after)
    backoff = policy.base_delay * 2 ** (attempt - 1)
    return min(policy.max_delay, backoff + random.uniform(0, policy.base_delay))


def budgeted_delay(error: Exception, attempt: int, policy: RetryPolicy,
                   total_waited: float, *,
                   blocked: bool = False) -> float | None:
    """delay_for + the hard-budget check, shared by every retry loop.

    Returns the seconds to wait, or None when giving up is final: the
    error isn't retryable, attempts are exhausted, the wall-clock budget
    can't cover the wait -- or ``blocked`` says a caller-level rule
    forbids replay (the stream guard's any-event-forwarded rule).
    """
    delay = delay_for(error, attempt, policy)
    if blocked or delay is None or attempt >= policy.max_attempts \
            or delay > policy.max_total_seconds - total_waited:
        return None
    return delay


def connect_with_retries(
    send: Callable[[], httpx.Response],
    *,
    classify: Callable[[httpx.Response], Exception],
    policy: RetryPolicy,
) -> httpx.Response:
    """Run ``send()`` until it yields a 200 response or failure is final.

    ``classify`` turns a non-200 response into the ProviderError to raise
    (and MUST consume the response body first -- streaming responses are
    closed right after classification).

    Connection-level exceptions raised by ``send()`` participate in the
    same policy via ``delay_for``. The last error wins when budgets are
    exhausted -- the caller sees exactly why the provider gave up.
    """
    total_waited = 0.0
    error: Exception
    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = send()
        except httpx.TransportError as exc:
            response, error = None, exc
        else:
            if response.status_code == 200:
                return response
            error = classify(response)
            response.close()

        delay = delay_for(error, attempt, policy)
        out_of_budget = (
            delay is None
            or attempt >= policy.max_attempts
            or delay > policy.max_total_seconds - total_waited
        )
        if out_of_budget:
            raise error
        _sleep(delay)
        total_waited += delay
    raise AssertionError("unreachable: the loop always returns or raises")


async def aconnect_with_retries(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    classify: Callable[[httpx.Response], Awaitable[Exception]],
    policy: RetryPolicy,
) -> httpx.Response:
    """Async twin of :func:`connect_with_retries`.

    Identical budget arithmetic and the identical streaming rule -- only
    the send is awaited and the wait is ``asyncio.sleep`` (which yields
    to the event loop instead of freezing it, the quiet reason async
    retry loops must never use ``time.sleep``).

    ``classify`` is awaited because even READING an error body is async
    on this surface (``response.aread()``).
    """
    total_waited = 0.0
    error: Exception
    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = await send()
        except httpx.TransportError as exc:
            response, error = None, exc
        else:
            if response.status_code == 200:
                return response
            error = await classify(response)
            await response.aclose()

        delay = delay_for(error, attempt, policy)
        out_of_budget = (
            delay is None
            or attempt >= policy.max_attempts
            or delay > policy.max_total_seconds - total_waited
        )
        if out_of_budget:
            raise error
        await _asleep(delay)
        total_waited += delay
    raise AssertionError("unreachable: the loop always returns or raises")
