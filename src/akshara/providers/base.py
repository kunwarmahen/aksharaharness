"""The provider interface -- the seam where wire formats are normalized.

Each adapter implements two translations:

* request building:   internal types -> provider JSON
* response parsing:   provider JSON (or SSE events) -> internal types

Everything else in the harness speaks ONLY the internal types from
``akshara.types``. That is what will let the agent switch providers
mid-session: history is stored in internal types, so re-pointing it at a
different adapter requires zero translation of anything.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

import httpx

from akshara.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
)
from akshara.providers.retry import RetryPolicy
from akshara.types import (
    Block,
    EndEvent,
    Message,
    ModelResponse,
    RedactedThinking,
    RedactedThinkingBlock,
    StartEvent,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolSpec,
    Usage,
)


@dataclass(frozen=True)
class ProviderSettings:
    """Everything an adapter needs to reach its API."""

    api_key: str
    base_url: str


# Phrases (lowercased) whose presence in a 400 marks a context-window
# overflow rather than an ordinary bad request. Anthropic says "prompt
# is too long"; OpenAI-style APIs say "context length".
_OVERFLOW_MARKERS = ("prompt is too long", "context length", "context_length")


def provider_error_for(
    status: int,
    detail: str,
    body: str,
    retry_after: float | None = None,
) -> ProviderError:
    """Map one failed HTTP response onto the exception taxonomy.

    Shared by both adapters: error BODY shapes differ per provider
    (each adapter parses its own into ``detail``), but the
    status-to-exception mapping is identical.
    """
    if status == 400 and any(m in detail.lower() for m in _OVERFLOW_MARKERS):
        return ContextOverflowError(f"400 context overflow: {detail}", status=status, body=body)
    if status in (401, 403):
        return AuthError(f"{status}: {detail}", status=status, body=body)
    if status == 429:
        return RateLimitError(f"429: {detail}", status=status, body=body, retry_after=retry_after)
    return ProviderError(f"{status}: {detail}", status=status, body=body)


class Provider(ABC):
    """A chat-model backend, in BOTH dialects of control flow.

    The sync surface (``stream``/``complete``) is the original harness
    API. The async twins (``astream``/``acomplete``) exist because the
    place async genuinely pays is ACROSS conversations -- many turns
    driven concurrently by one event loop -- not inside one (a single
    conversation is sequential at the model boundary). Both surfaces
    share every line of protocol logic; only the HTTP plumbing differs.
    """

    name: str  # "anthropic" | "openai" -- set by subclasses

    def __init__(
        self,
        settings: ProviderSettings,
        transport: httpx.BaseTransport | None = None,
        retry: RetryPolicy | None = None,
        cache_control: bool = False,
    ) -> None:
        self.settings = settings
        self.retry = retry or RetryPolicy()
        # Wire-level prompt-cache opt-in. Adapters that support explicit
        # cache breakpoints honor it (anthropic stamps cache_control onto
        # the request prefix); dialects whose upstream caching is automatic
        # ignore it (openai -- nothing to send, hits just show up in usage).
        # Off by default: it is a billing-relevant choice, not a free win.
        self.cache_control = cache_control
        # One connection pool per provider. `transport` is the test seam:
        # tests pass httpx.MockTransport(handler), which serves canned
        # responses from pure functions while exercising the REAL
        # request-building code path. No network, no patching.
        self.client = httpx.Client(
            base_url=settings.base_url,
            transport=transport,
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
        # The async pool. MockTransport implements BOTH handler methods,
        # so when the test seam is one it can serve both clients; real
        # sync transports cannot drive an AsyncClient, so production gets
        # a fresh default pool there.
        shared_async_transport = (
            transport if isinstance(transport, httpx.MockTransport) else None
        )
        self.aclient = httpx.AsyncClient(
            base_url=settings.base_url,
            transport=shared_async_transport,
            timeout=httpx.Timeout(600.0, connect=10.0),
        )

    @abstractmethod
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
        """Yield normalized StreamEvents; fold them with collect()."""

    @abstractmethod
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
        """Non-streaming twin of stream(); kept for parity testing."""

    @abstractmethod
    def astream(
        self,
        *,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec],
        model: str,
        max_tokens: int = 16384,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async twin of stream(): same events, awaited transport."""

    @abstractmethod
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
        """Non-streaming twin of astream()."""


class ResponseFolder:
    """The event-accumulation state machine behind collect()/acollect().

    The accumulation rules live HERE, not in the adapters:

    * Text fragments append to the trailing TextBlock (or start one).
    * Tool-call argument JSON fragments accumulate per stream index and
      are parsed exactly ONCE, at finish(). Never parse per-fragment:
      fragments can split an escape sequence mid-token (``{"pa`` +
      ``th": ...``).
    * Thinking fragments accumulate per stream index too -- prose and
      signature are separate fragment kinds joined into one block.
    * Redacted thinking needs NO accumulation: it arrives complete at
      its start event and is appended verbatim.
    * Blocks appear in arrival order, so text-then-tools reads back
      exactly as the model emitted it.
    """

    def __init__(self) -> None:
        self.model = ""
        self.stop_reason: StopReason = "other"
        self.usage = Usage()
        self.ordered: list[Block] = []  # final message content, arrival order
        self.bufs: dict[int, str] = {}  # stream index -> accumulated args JSON
        self.calls: dict[int, ToolCall] = {}  # stream index -> accumulating call
        self.thinkers: dict[int, ThinkingBlock] = {}  # index -> accumulating block

    def feed(self, event: StreamEvent) -> None:
        match event:
            case StartEvent(model=name):
                self.model = name
            case TextDelta(text=fragment):
                ordered = self.ordered
                if ordered and isinstance(ordered[-1], TextBlock):
                    ordered[-1].text += fragment
                else:
                    ordered.append(TextBlock(fragment))
            case ThinkingDelta(index=i, text=fragment, signature=sig):
                block = self.thinkers.get(i)
                if block is None:  # no start event needed: nothing to know up front
                    block = ThinkingBlock("")
                    self.thinkers[i] = block
                    self.ordered.append(block)  # arrival order, like everything else
                if fragment:
                    block.thinking += fragment
                if sig:
                    block.signature += sig
            case RedactedThinking(index=_, data=payload):
                # No accumulation: the whole ciphertext arrived at once.
                self.ordered.append(RedactedThinkingBlock(payload))
            case ToolCallStart(index=i, id=cid, name=name):
                call = ToolCall(id=cid, name=name, arguments={})
                self.calls[i] = call  # stream index -> object, no guessing
                self.ordered.append(call)
                self.bufs[i] = ""
            case ToolCallDelta(index=i, partial_json=piece):
                self.bufs[i] = self.bufs.get(i, "") + piece
            case EndEvent(stop_reason=sr, usage=u):
                self.stop_reason = sr
                self.usage = u

    def finish(self) -> ModelResponse:
        # Parse each call's arguments once, now that fragments are complete.
        for i, raw in self.bufs.items():
            call = self.calls.get(i)
            if call is None:
                continue
            try:
                call.arguments = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                # Malformed arguments from the model: keep them visible as a
                # dict the executor will reject, instead of crashing the loop.
                call.arguments = {"_unparseable_json": raw}

        return ModelResponse(
            message=Message("assistant", self.ordered),
            stop_reason=self.stop_reason,
            usage=self.usage,
            model=self.model,
        )


def collect(events: Iterable[StreamEvent]) -> ModelResponse:
    """Fold a StreamEvent stream into the ModelResponse that complete()
    would have returned for the same exchange.

    Shared by every adapter -- this is the proof that the event
    vocabulary is lossless: parity tests hold each adapter to
    collect(stream(...)) == complete(...).
    """
    folder = ResponseFolder()
    for event in events:
        folder.feed(event)
    return folder.finish()


async def acollect(events: AsyncIterator[StreamEvent]) -> ModelResponse:
    """Async twin of :func:`collect` -- same folder, zero new rules."""
    folder = ResponseFolder()
    async for event in events:
        folder.feed(event)
    return folder.finish()
