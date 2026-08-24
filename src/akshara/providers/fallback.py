"""Failover across providers -- opening-only, like retry, but across SPACE.

Retry (``retry.py``) fixes TIME problems: same place, later moment.
Fallback fixes PLACE problems: same moment, different backend. They
compose (each adapter retries its own opening first; only after a
backend exhausts its budget does failover even look at the next one),
and neither duplicates output, because both obey THE STREAMING RULE:

    once a single event has reached the caller, the exchange is
    COMMITTED -- replaying it means duplicate tokens in someone's
    terminal. Failover therefore happens only when a provider failed
    BEFORE delivering its first event.

What counts as "worth failing over":

* YES -- provider-side trouble where ANOTHER PLACE may genuinely differ:
  AuthError (the next key may be valid), RateLimitError, 5xx
  ProviderError, connection-level httpx.TransportError.
* NO  -- request-shaped failures: the SAME bytes are about to be sent to
  the next backend too. ContextOverflowError and ordinary 400s will not
  heal by changing venue; fail fast and let the caller see it.

Deliberately NOT a Provider subclass: it has no settings, no base URL,
no client of its own -- it IS a list of those. Duck-typed to the two
methods the agent actually calls. Library-level on purpose: session
persistence stores a provider NAME and rebuilds it via the factory, and
a composite cannot round-trip through a name yet.

All-fail surfacing follows retry.py's honesty rule, upgraded: the LAST
error is the raised exception (its ``__cause__``), and every earlier
attempt's reason rides along in the message -- "primary throttled, then
fallback key invalid" is a diagnosis; either error alone is a riddle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx

from akshara.errors import AuthError, ContextOverflowError, ProviderError, RateLimitError
from akshara.providers.base import ModelResponse, StreamEvent
from akshara.types import Message, ToolSpec

# Exceptions where a DIFFERENT backend plausibly changes the outcome.
_FAILOVERABLE = (AuthError, RateLimitError, ProviderError, httpx.TransportError)


def _can_fail_over(error: Exception) -> bool:
    if isinstance(error, ContextOverflowError):
        return False  # request-shaped: every venue gets the same history
    if isinstance(error, ProviderError) and getattr(error, "status", None) == 400:
        return False  # same argument as above, minus the overflow marker
    return isinstance(error, _FAILOVERABLE)


class FallbackProvider:
    """Tries providers left to right; commits to the first that opens.

    ``FallbackProvider(primary, secondary)`` behaves exactly like
    ``primary`` while primary is healthy -- including streaming -- and
    degrades to ``secondary`` only on an opening-phase failure. After
    the first event crosses the boundary, failures propagate untouched.
    """

    def __init__(self, *providers) -> None:
        if len(providers) < 2:
            raise ValueError(
                "FallbackProvider needs at least two providers "
                "(a primary and someone to fall back TO)")
        self.providers = list(providers)
        self.name = " -> ".join(p.name for p in providers)

    def stream(
        self,
        *,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 16384,
        temperature: float | None = None,
    ) -> Iterator[StreamEvent]:
        errors: list[tuple[str, Exception]] = []
        for provider in self.providers:
            try:
                gen = provider.stream(messages=messages, system=system,
                                      tools=tools, model=model,
                                      max_tokens=max_tokens,
                                      temperature=temperature)
                first = next(gen)  # the commit point: past here, no replay
            except StopIteration:
                return  # degenerate empty stream; nothing to hand on
            except Exception as exc:
                errors.append((provider.name, exc))
                if not _can_fail_over(exc):
                    raise
                continue  # nothing delivered yet -- safe to try the next
            yield first
            yield from gen
            return

        _, last_error = errors[-1]
        raise ProviderError(
            f"all providers failed ({self.name}): "
            + "; ".join(f"{n}: {e}" for n, e in errors),
            status=getattr(last_error, "status", None),
        ) from last_error

    async def astream(
        self,
        *,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 16384,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async twin of stream(): identical commit point, identical
        fail-over taxonomy -- only the await differs."""
        errors: list[tuple[str, Exception]] = []
        for provider in self.providers:
            try:
                # Constructing the generator is INSIDE the try on purpose:
                # a duck-typed provider may raise eagerly, and any opening
                # failure must look identical regardless of when it fires.
                agen = provider.astream(messages=messages, system=system,
                                        tools=tools, model=model,
                                        max_tokens=max_tokens,
                                        temperature=temperature)
                first = await agen.__anext__()  # the commit point
            except StopAsyncIteration:
                return  # degenerate empty stream; nothing to hand on
            except Exception as exc:
                errors.append((provider.name, exc))
                if not _can_fail_over(exc):
                    raise
                continue  # nothing delivered yet -- safe to try the next
            yield first
            async for event in agen:
                yield event
            return

        _, last_error = errors[-1]
        raise ProviderError(
            f"all providers failed ({self.name}): "
            + "; ".join(f"{n}: {e}" for n, e in errors),
            status=getattr(last_error, "status", None),
        ) from last_error

    def complete(
        self,
        *,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 16384,
        temperature: float | None = None,
    ) -> ModelResponse:
        errors: list[tuple[str, Exception]] = []
        for provider in self.providers:
            try:
                return provider.complete(messages=messages, system=system,
                                         tools=tools, model=model,
                                         max_tokens=max_tokens,
                                         temperature=temperature)
            except Exception as exc:
                errors.append((provider.name, exc))
                if not _can_fail_over(exc):
                    raise
        _, last_error = errors[-1]
        raise ProviderError(
            f"all providers failed ({self.name}): "
            + "; ".join(f"{n}: {e}" for n, e in errors),
            status=getattr(last_error, "status", None),
        ) from last_error

    async def acomplete(
        self,
        *,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 16384,
        temperature: float | None = None,
    ) -> ModelResponse:
        errors: list[tuple[str, Exception]] = []
        for provider in self.providers:
            try:
                return await provider.acomplete(
                    messages=messages, system=system, tools=tools,
                    model=model, max_tokens=max_tokens,
                    temperature=temperature)
            except Exception as exc:
                errors.append((provider.name, exc))
                if not _can_fail_over(exc):
                    raise
        _, last_error = errors[-1]
        raise ProviderError(
            f"all providers failed ({self.name}): "
            + "; ".join(f"{n}: {e}" for n, e in errors),
            status=getattr(last_error, "status", None),
        ) from last_error
