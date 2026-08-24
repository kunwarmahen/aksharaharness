"""Adapter for the Anthropic Messages API wire format.

Endpoint:  POST {base}/v1/messages
Auth:      x-api-key header + anthropic-version header
Reference: https://docs.anthropic.com/en/api/messages

Request shape (the parts we use):

    {
      "model": "...",
      "max_tokens": 1024,           # REQUIRED -- requests fail without it
      "system": "..." | null,       # top-level, NOT inside messages
      "messages": [
        {"role": "user" | "assistant",
         "content": [ {"type": "text", "text"},
                      {"type": "tool_use", "id", "name", "input"},
                      {"type": "tool_result", "tool_use_id", "content"} ]}
      ],
      "tools": [ {"name", "description", "input_schema": {...}} ],
      "stream": false
    }

Responses carry content as a list of typed blocks plus a stop_reason
string -- both map almost 1:1 onto our internal types, which is exactly
why the internal vocabulary mirrors Anthropic's shape.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

import httpx

from akshara.errors import ProviderError
from akshara.providers.base import Provider, provider_error_for
from akshara.providers import retry as _retry  # sleeps go through the module attr
from akshara.providers.retry import (
    aconnect_with_retries,
    budgeted_delay,
    connect_with_retries,
)
from akshara.providers.sse import (
    aparse_events,
    aiter_sse_lines,
    iter_sse_lines,
    parse_events,
)
from akshara.types import (
    Block,
    EndEvent,
    ImageBlock,
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
    ToolResult,
    ToolSpec,
    Usage,
)

API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    # ---- request building -------------------------------------------------

    def _url(self) -> str:
        return "/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.settings.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def build_request_body(
        self,
        *,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSpec],
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Internal types -> Messages-API JSON.

        Public (not underscore) on purpose: examples print exactly what
        goes over the wire before sending it.
        """
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [_encode_message(m) for m in messages],
            "stream": stream,
        }
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = [_encode_tool(t) for t in tools]
        if self.cache_control:
            _mark_cache_breakpoints(body)
        return body

    # ---- non-streaming ----------------------------------------------------

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
        body = self.build_request_body(
            messages=messages,
            system=system,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        # Non-streaming: the whole call is pre-first-byte, so it participates
        # in the retry policy like any other connection opening.
        response = connect_with_retries(
            lambda: self.client.post(self._url(), headers=self._headers(), json=body),
            classify=self._error_for,
            policy=self.retry,
        )
        return self._parse_response(response.json())

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
        body = self.build_request_body(
            messages=messages,
            system=system,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        async def _classify(response: httpx.Response) -> Exception:
            return self._error_for(response)

        response = await aconnect_with_retries(
            lambda: self.aclient.post(self._url(), headers=self._headers(), json=body),
            classify=_classify,
            policy=self.retry,
        )
        return self._parse_response(response.json())

    # ---- streaming --------------------------------------------------------

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
        body = self.build_request_body(
            messages=messages,
            system=system,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        def _open() -> httpx.Response:
            # Fresh Request per attempt: the body is already built, so this
            # is cheap, and there is never a question of replaying consumed
            # request content.
            request = self.client.build_request(
                "POST", self._url(), headers=self._headers(), json=body
            )
            return self.client.send(request, stream=True)

        def _classify(response: httpx.Response) -> Exception:
            response.read()  # an error body arrives as a stream too
            return self._error_for(response)

        # THE STREAMING RULE (retry.py), both notches. The opening gets
        # full retries via connect_with_retries; the attempt LOOP below
        # re-opens when a 200 stream dies BEFORE its first event (same
        # budgets -- nothing was forwarded, so nothing can be duplicated).
        # Once one event reaches the caller, any failure propagates.
        attempt = 0
        total_waited = 0.0
        while True:
            attempt += 1
            response = connect_with_retries(
                _open, classify=_classify, policy=self.retry)
            forwarded = 0
            try:
                for event in _stream_events(response.iter_bytes()):
                    forwarded += 1
                    yield event
                return
            except Exception as exc:
                delay = budgeted_delay(exc, attempt, self.retry,
                                       total_waited, blocked=forwarded > 0)
                if delay is None:
                    raise
                _retry._sleep(delay)
                total_waited += delay
            finally:
                # Closing the generator early (Ctrl-C) unwinds into this
                # finally and releases the socket -- every attempt's socket.
                response.close()

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
        body = self.build_request_body(
            messages=messages,
            system=system,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async def _open() -> httpx.Response:
            request = self.aclient.build_request(
                "POST", self._url(), headers=self._headers(), json=body
            )
            return await self.aclient.send(request, stream=True)

        async def _classify(response: httpx.Response) -> Exception:
            await response.aread()  # an error body arrives as a stream too
            return self._error_for(response)

        # The same two-notch streaming rule as stream() (sync twin above):
        # opening retries via aconnect_with_retries; pre-first-event deaths
        # re-open under shared budgets; after that, failures propagate.
        attempt = 0
        total_waited = 0.0
        while True:
            attempt += 1
            response = await aconnect_with_retries(
                _open, classify=_classify, policy=self.retry)
            forwarded = 0
            try:
                async for event in _astream_events(response.aiter_bytes()):
                    forwarded += 1
                    yield event
                return
            except Exception as exc:
                delay = budgeted_delay(exc, attempt, self.retry,
                                       total_waited, blocked=forwarded > 0)
                if delay is None:
                    raise
                await _retry._asleep(delay)
                total_waited += delay
            finally:
                # Cancelling the consuming task unwinds here and releases
                # the socket -- the async twin of GeneratorExit.
                await response.aclose()

    # ---- response parsing ---------------------------------------------------

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            message=Message("assistant", _decode_blocks(data.get("content", []))),
            stop_reason=_map_stop_reason(data.get("stop_reason")),
            usage=_decode_usage(data.get("usage") or {}),
            model=data.get("model", ""),
            raw=data,
        )

    def _error_for(self, response: httpx.Response) -> ProviderError:
        """Parse the Anthropic error body shape, then share the mapping.

        Anthropic errors look like:
            {"type": "error", "error": {"type": "...", "message": "..."}}
        and can ALSO arrive mid-stream as ``event: error``.
        """
        try:
            err = response.json().get("error") or {}
            detail = f"{err.get('type', 'unknown')}: {err.get('message', '')}"
        except Exception:
            detail = response.text[:500]

        retry_after: float | None = None
        if response.status_code == 429:
            raw = response.headers.get("retry-after")
            try:
                retry_after = float(raw) if raw else None
            except ValueError:
                retry_after = None

        return provider_error_for(response.status_code, detail, response.text, retry_after)


# ---------------------------------------------------------------------------
# Encoding: internal -> wire
# ---------------------------------------------------------------------------

CACHE_CONTROL = {"type": "ephemeral"}  # the only flavor the API offers


def _mark_cache_breakpoints(body: dict[str, Any]) -> None:
    """Stamp up to three cache breakpoints onto the request prefix.

    Prompt caching is PREFIX caching: everything up to and including a
    block carrying ``cache_control`` becomes a cache entry (5-minute TTL,
    refreshed on every hit; writes bill at 1.25x input, reads at 0.1x).
    An agent loop has exactly the stable-then-growing shape this was
    designed for -- [tools] + [system] + [...history...] -- so we mark
    the LAST of each and every later turn reuses the earlier prefix,
    paying full price only for the newest messages. That spends 3 of the
    API's 4-breakpoint budget. Blocks shorter than ~1024 tokens never
    cache (the API ignores them silently), so leaving the flag on for
    short conversations is harmless.
    """
    tools = body.get("tools")
    if tools:
        tools[-1]["cache_control"] = dict(CACHE_CONTROL)
    if isinstance(body.get("system"), str):
        body["system"] = [
            {"type": "text", "text": body["system"],
             "cache_control": dict(CACHE_CONTROL)},
        ]
    messages = body.get("messages")
    if messages:
        content = messages[-1].get("content") or []
        if content:
            content[-1]["cache_control"] = dict(CACHE_CONTROL)


def _encode_message(message: Message) -> dict[str, Any]:
    """Encode one Message. Role-agnostic: the same block types appear on
    both sides (tool_use in assistant turns, tool_result in user turns)."""
    content: list[dict[str, Any]] = []
    for block in message.content:
        match block:
            case TextBlock(text=text):
                content.append({"type": "text", "text": text})
            case ToolCall(id=cid, name=name, arguments=args):
                content.append(
                    {"type": "tool_use", "id": cid, "name": name, "input": args}
                )
            case ToolResult(tool_call_id=cid, content=out, is_error=is_err):
                entry: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": cid,
                    "content": out,
                }
                if is_err:
                    entry["is_error"] = True
                content.append(entry)
            case ThinkingBlock(thinking=text, signature=sig):
                # Round-tripped verbatim: continuing this turn with tool
                # results requires the thinking blocks back. ALWAYS include
                # the signature key -- gateways may emit unsigned blocks,
                # but their upstreams still reject the request if the
                # FIELD is absent ("" passes, omission 400s). Found by
                # differential probe against OpenRouter.
                content.append(
                    {"type": "thinking", "thinking": text, "signature": sig}
                )
            case RedactedThinkingBlock(data=payload):
                content.append({"type": "redacted_thinking", "data": payload})
            case ImageBlock(media_type=mime, data=b64):
                content.append(
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime,
                                "data": b64}}
                )
    return {"role": message.role, "content": content}


def _encode_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    }


# ---------------------------------------------------------------------------
# Decoding: wire -> internal
# ---------------------------------------------------------------------------


def _decode_blocks(raw_blocks: list[dict[str, Any]]) -> list[Block]:
    out: list[Block] = []
    for raw in raw_blocks:
        kind = raw.get("type")
        if kind == "text":
            out.append(TextBlock(raw.get("text", "")))
        elif kind == "tool_use":
            args = raw.get("input")
            out.append(
                ToolCall(raw["id"], raw["name"], args if isinstance(args, dict) else {})
            )
        elif kind == "thinking":
            out.append(ThinkingBlock(
                raw.get("thinking", ""),
                signature=raw.get("signature", ""),
            ))
        elif kind == "redacted_thinking":
            out.append(RedactedThinkingBlock(raw.get("data", "")))
        else:
            # server_tool_use, ...: surface them as a visible placeholder --
            # silently dropping content hides bugs; crashing on unknown
            # blocks makes forward-compatibility awful.
            print(
                f"[akshara] unsupported content block type {kind!r}; "
                "rendered as placeholder text",
                file=sys.stderr,
            )
            out.append(TextBlock(f"[unsupported block: {kind}]"))
    return out


def _map_stop_reason(raw: str | None) -> StopReason:
    if raw in ("end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"):
        return raw  # type: ignore[return-value]
    return "other"


def _decode_usage(usage: dict[str, Any]) -> Usage:
    def _count(key: str) -> int:
        # .get's default only fires on a MISSING key; gateways (OpenRouter)
        # send explicit nulls for absent counters. None would poison
        # Usage.add() arithmetic three iterations later.
        return usage.get(key) or 0

    return Usage(
        input_tokens=_count("input_tokens"),
        output_tokens=_count("output_tokens"),
        cache_read_tokens=_count("cache_read_input_tokens"),
        cache_write_tokens=_count("cache_creation_input_tokens"),
    )


# ---------------------------------------------------------------------------
# Streaming: SSE bytes -> normalized StreamEvents
# ---------------------------------------------------------------------------


class AnthropicStreamRouter:
    """Stateful wire-event -> StreamEvent router, fed one SSE event at a
    time by BOTH the sync and async transport loops.

    Deliberately near-stateless: this class only RE-ROUTEs wire events;
    accumulating argument fragments and building the final message is
    collect()'s job (shared with the OpenAI adapter).

    Wire sequence for reference:

        message_start          -> StartEvent + input-token usage
        ping                   -> ignored
        content_block_start    -> ToolCallStart (when type == tool_use)
                                  RedactedThinking (when type ==
                                  redacted_thinking -- arrives whole, no deltas)
        content_block_delta    -> TextDelta | ThinkingDelta | ToolCallDelta
                                  (by delta.type)
        content_block_stop     -> nothing to do (parsing happens in collect)
        message_delta          -> stop_reason + cumulative output usage
        message_stop           -> EndEvent
        error                  -> ProviderError raised mid-stream

    Thinking blocks have no start event on our side: unlike tool calls,
    nothing about them needs to be known up front, so collect() lazily
    creates the block on first ThinkingDelta for an index.
    """

    def __init__(self) -> None:
        self.stop_reason: StopReason = "other"
        self.usage = Usage()

    def feed(self, name: str | None, data: str) -> list[StreamEvent]:
        try:
            payload = json.loads(data) if data else {}
        except json.JSONDecodeError:
            return []  # tolerate a malformed keep-alive rather than dying

        kind = name or payload.get("type")

        if kind == "error":
            err = payload.get("error") or {}
            raise ProviderError(
                "mid-stream error: "
                f"{err.get('type', 'unknown')}: {err.get('message', '')}"
            )
        if kind == "ping":
            return []

        if kind == "message_start":
            message = payload.get("message") or {}
            model = message.get("model", "")
            self.usage = _decode_usage(message.get("usage") or {})
            return [StartEvent(model=model)]

        if kind == "content_block_start":
            block = payload.get("content_block") or {}
            if block.get("type") == "tool_use":
                return [ToolCallStart(
                    index=payload["index"],
                    id=block["id"],
                    name=block["name"],
                )]
            if block.get("type") == "redacted_thinking":
                # Arrives COMPLETE at start -- no deltas follow, because the
                # ciphertext payload has nothing human-shaped to stream. Must
                # still reach collect(): it round-trips verbatim like a signed
                # thinking block, and dropping it 400s the next request.
                return [RedactedThinking(
                    index=payload["index"], data=block.get("data", "")
                )]
            if block.get("type") not in ("text", "thinking"):
                print(
                    f"[akshara] unsupported streamed block type "
                    f"{block.get('type')!r}; deltas ignored",
                    file=sys.stderr,
                )
            return []

        if kind == "content_block_delta":
            delta = payload.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                return [TextDelta(delta.get("text", ""))]
            if dtype == "thinking_delta":
                return [ThinkingDelta(index=payload["index"],
                                      text=delta.get("thinking", ""))]
            if dtype == "signature_delta":
                return [ThinkingDelta(index=payload["index"],
                                      signature=delta.get("signature", ""))]
            if dtype == "input_json_delta":
                return [ToolCallDelta(
                    index=payload["index"],
                    partial_json=delta.get("partial_json", ""),
                )]
            return []  # other delta types: ignored

        if kind == "message_delta":
            delta = payload.get("delta") or {}
            self.stop_reason = _map_stop_reason(delta.get("stop_reason"))
            out = (payload.get("usage") or {}).get("output_tokens")
            if out is not None:
                self.usage.output_tokens = out  # cumulative, replaces not adds
            return []

        if kind == "message_stop":
            return [EndEvent(stop_reason=self.stop_reason, usage=self.usage)]

        return []


def _stream_events(chunks: Iterable[bytes]) -> Iterator[StreamEvent]:
    """Sync skin over the router: SSE bytes -> normalized StreamEvents."""
    router = AnthropicStreamRouter()
    for name, data in parse_events(iter_sse_lines(chunks)):
        yield from router.feed(name, data)


async def _astream_events(achunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
    """Async skin over the SAME router -- zero duplicated routing rules."""
    router = AnthropicStreamRouter()
    async for name, data in aparse_events(aiter_sse_lines(achunks)):
        for event in router.feed(name, data):
            yield event
