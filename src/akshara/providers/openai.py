"""Adapter for the OpenAI chat-completions wire format.

Endpoint:  POST {base}/chat/completions   (base INCLUDES /v1)
Auth:      Authorization: Bearer header
Reference: https://platform.openai.com/docs/api-reference/chat

Same interface as AnthropicProvider -- that's the point. The dialect
differences live here and nowhere else (see notes/02-wire-formats.md):

* system prompt is messages[0], not a top-level field
* tool definitions are wrapped: {"type":"function", "function": {...}}
* tool calls come back as message.tool_calls[] with arguments as a
  JSON STRING (we parse it once, here)
* tool results go back as one role:"tool" message PER RESULT -- there
  are no block-shaped user messages, so our user Message fans out into
  several wire messages
* streams are anonymous chunks ending with a literal ``data: [DONE]``
* usage only arrives when stream_options.include_usage is sent, and
  some upstreams ignore even then
* reasoning arrives as ``reasoning`` / ``reasoning_content`` fields (the
  de-facto gateway convention), NOT typed blocks -- display-only, since
  this wire has no way to send reasoning back
"""

from __future__ import annotations

import json
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
    EndEvent,
    ImageBlock,
    Message,
    ModelResponse,
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

#: chat-completions has ONE reasoning channel, not indexed blocks -- give
#: it an index that can never collide with real tool-call indices.
REASONING_INDEX = -1


class OpenAIProvider(Provider):
    name = "openai"

    # ---- request building -------------------------------------------------

    def _url(self) -> str:
        return "/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
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
        """Internal types -> chat-completions JSON. Public for examples."""
        wire_messages: list[dict[str, Any]] = []
        if system is not None:
            wire_messages.append({"role": "system", "content": system})
        for m in messages:
            wire_messages.extend(_encode_message(m))

        body: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            # Newer OpenAI models prefer max_completion_tokens; OpenAI-
            # compatible gateways (incl. OpenRouter) universally accept
            # max_tokens. Switch here if you target OpenAI directly.
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = [_encode_tool(t) for t in tools]
        if stream:
            # Without this the stream never reports usage. Some upstreams
            # ignore it anyway -- our Usage stays possibly-zero.
            body["stream_options"] = {"include_usage": True}
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
            # Fresh Request per attempt (body already built; no replayed content).
            request = self.client.build_request(
                "POST", self._url(), headers=self._headers(), json=body
            )
            return self.client.send(request, stream=True)

        def _classify(response: httpx.Response) -> Exception:
            response.read()  # an error body arrives as a stream too
            return self._error_for(response)

        # THE STREAMING RULE (retry.py), both notches -- identical loop to
        # the anthropic adapter's: full opening retries; a 200 that dies
        # before its first event re-opens under the same budgets (nothing
        # forwarded, nothing duplicated); after that, failures propagate.
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

        # The same two-notch streaming rule as stream() (sync twin above).
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
                await response.aclose()

    # ---- response parsing ---------------------------------------------------

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        choice = data["choices"][0]
        raw_message = choice.get("message") or {}

        blocks: list[Any] = []
        reasoning = raw_message.get("reasoning") or raw_message.get("reasoning_content")
        if reasoning:
            # Gateway convention (OpenRouter, DeepSeek-style). Display-only:
            # there is no field to send it back through.
            blocks.append(ThinkingBlock(reasoning))
        if raw_message.get("content"):
            blocks.append(TextBlock(raw_message["content"]))
        for tc in raw_message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            blocks.append(ToolCall(tc["id"], fn.get("name", ""), _parse_arguments(fn)))

        usage_raw = data.get("usage") or {}
        # OpenAI caches automatically upstream -- nothing to request; hits
        # are only visible here, in prompt_tokens_details. Wire semantics:
        # prompt_tokens INCLUDES those hits. The internal convention
        # (types.Usage) counts input_tokens as FULL-RATE tokens only --
        # subtract, or every downstream cost computation bills cached
        # tokens twice.
        cached = ((usage_raw.get("prompt_tokens_details")
                   or {}).get("cached_tokens")) or 0
        prompt = usage_raw.get("prompt_tokens") or 0
        usage = Usage(
            # `or 0` not .get-default: gateways send explicit nulls for
            # absent counters, and None would poison Usage.add() later.
            input_tokens=max(0, prompt - cached),
            output_tokens=usage_raw.get("completion_tokens") or 0,
            cache_read_tokens=cached,
        )
        return ModelResponse(
            message=Message("assistant", blocks),
            stop_reason=_map_finish_reason(choice.get("finish_reason")),
            usage=usage,
            model=data.get("model", ""),
            raw=data,
        )

    def _error_for(self, response: httpx.Response) -> ProviderError:
        # Shared with the Responses adapter: both OpenAI-family dialects
        # speak the same error body shape.
        return openai_error_for(response)


def openai_error_for(response: httpx.Response) -> ProviderError:
    """Parse the OpenAI error body shape, then share the mapping.

    OpenAI errors look like:
        {"error": {"message": "...", "type": "...", "code": "..."}}

    Module-level so providers/responses.py (same error shape) reuses
    this verbatim instead of duplicating it.
    """
    try:
        err = response.json().get("error") or {}
        kind = err.get("type") or err.get("code") or "unknown"
        detail = f"{kind}: {err.get('message', '')}"
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
#
# The structural heart of this adapter: ONE internal Message can become
# SEVERAL wire messages (a user turn holding ToolResult blocks fans out
# into role:"tool" messages). The Anthropic adapter never does this --
# its wire format is block-shaped like ours.
# ---------------------------------------------------------------------------


def _encode_message(message: Message) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if message.role == "assistant":
        # ThinkingBlocks and RedactedThinkingBlocks are dropped here
        # DELIBERATELY: chat-completions has no field to carry reasoning
        # back, and servers regenerate it. (Contrast the Anthropic
        # adapter, which MUST round-trip both -- signature/ciphertext
        # included -- during tool loops.)
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        calls = [
            {
                "id": call.id,
                "type": "function",
                # arguments travel as a JSON STRING on this wire
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls()
        ]
        entry: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            entry["tool_calls"] = calls
        out.append(entry)
        return out

    # user role: each ToolResult becomes its own role:"tool" message,
    # and they go FIRST -- this wire requires tool results to directly
    # follow the assistant turn that made the calls. Any remaining
    # text/images then ride ONE trailing user message; that trailing
    # position is how a tool-produced image (read_image) reaches the
    # model at all, since role:"tool" cannot carry an image on this wire.
    # No is_error flag exists either -- the convention is to mark it in
    # the content text.
    for block in message.content:
        match block:
            case ToolResult(tool_call_id=cid, content=out_text, is_error=is_err):
                content = f"ERROR: {out_text}" if is_err else out_text
                out.append(
                    {"role": "tool", "tool_call_id": cid, "content": content}
                )
    if any(isinstance(b, ImageBlock) for b in message.content):
        # Multimodal message: chat-completions carries text+images as an
        # ordered ARRAY of typed parts (images as data: URLs). Only used
        # when an image is present -- plain-text requests keep the plain
        # string shape, byte-identical to before vision existed.
        parts: list[dict[str, Any]] = []
        for block in message.content:
            match block:
                case TextBlock(text=t) if t:
                    parts.append({"type": "text", "text": t})
                case ImageBlock(media_type=mime, data=b64):
                    parts.append({"type": "image_url",
                                  "image_url":
                                      {"url": f"data:{mime};base64,{b64}"}})
        if parts:
            out.append({"role": "user", "content": parts})
    else:
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        if text:
            out.append({"role": "user", "content": text})
    return out


def _encode_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


# ---------------------------------------------------------------------------
# Decoding: wire -> internal
# ---------------------------------------------------------------------------


def _parse_arguments(fn: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string; parse exactly once."""
    raw = fn.get("arguments") or ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Malformed arguments from the model: keep them visible as a dict
        # the executor will reject, instead of crashing the loop.
        return {"_unparseable_json": raw}
    return parsed if isinstance(parsed, dict) else {"_unparseable_json": raw}


_FINISH_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _map_finish_reason(raw: str | None) -> StopReason:
    if raw is None:
        return "other"
    return _FINISH_MAP.get(raw, "other")


# ---------------------------------------------------------------------------
# Streaming: SSE bytes -> normalized StreamEvents
# ---------------------------------------------------------------------------


class OpenAIStreamRouter:
    """Stateful wire-chunk -> StreamEvent router, fed one SSE event at a
    time by BOTH the sync and async transport loops.

    Wire sequence for reference:

        first chunk (delta.role)      -> StartEvent
        delta.reasoning fragments     -> ThinkingDelta (index REASONING_INDEX)
        delta.content fragments       -> TextDelta
        delta.tool_calls fragments    -> ToolCallStart (first fragment per
                                         index; id/name usually appear ONLY
                                         there) + ToolCallDelta per fragment
        finish_reason chunk           -> remembered, becomes EndEvent
        empty-choices chunk w/ usage  -> captured, no event
        data: [DONE]                  -> EndEvent, end of stream

    Parallel tool calls interleave their fragments by index, so state is
    keyed by index -- never by arrival order.
    """

    def __init__(self) -> None:
        self.stop_reason: StopReason = "other"
        self.usage = Usage()
        self.started = False
        self.seen_tools: set[int] = set()

    def feed(self, _name: str | None, data: str) -> list[StreamEvent]:
        # The sentinel is NOT JSON -- check before parsing.
        if data.strip() == "[DONE]":
            return [EndEvent(stop_reason=self.stop_reason, usage=self.usage)]

        try:
            chunk = json.loads(data) if data else {}
        except json.JSONDecodeError:
            return []  # tolerate malformed keep-alives

        out: list[StreamEvent] = []
        if chunk.get("model"):
            if not self.started:
                self.started = True
                out.append(StartEvent(model=chunk["model"]))

        reported_usage = chunk.get("usage")
        if reported_usage:
            # Same convention as the non-streaming path: cached hits ride
            # INSIDE prompt_tokens on the wire, so they are subtracted out
            # of input_tokens and stored solely as cache_read_tokens.
            # `or 0` not .get-default: gateways send explicit nulls here too.
            cached = ((reported_usage.get("prompt_tokens_details")
                       or {}).get("cached_tokens")) or 0
            self.usage.input_tokens = max(
                0, (reported_usage.get("prompt_tokens") or 0) - cached)
            self.usage.output_tokens = (
                reported_usage.get("completion_tokens") or 0)
            self.usage.cache_read_tokens = cached

        choices = chunk.get("choices") or []
        if not choices:
            return out  # usage-only chunk or keep-alive

        choice = choices[0]
        delta = choice.get("delta") or {}

        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if reasoning:
            out.append(ThinkingDelta(index=REASONING_INDEX, text=reasoning))

        if delta.get("content"):
            out.append(TextDelta(delta["content"]))

        for tc in delta.get("tool_calls") or []:
            index = tc.get("index", 0)
            fn = tc.get("function") or {}
            if index not in self.seen_tools:
                self.seen_tools.add(index)
                out.append(ToolCallStart(
                    index=index,
                    id=tc.get("id") or f"call_{index}",
                    name=fn.get("name") or "",
                ))
            fragment = fn.get("arguments")
            if fragment:
                out.append(ToolCallDelta(index=index, partial_json=fragment))

        if choice.get("finish_reason"):
            self.stop_reason = _map_finish_reason(choice["finish_reason"])

        return out

    def finish(self) -> list[StreamEvent]:
        """Stream ended without [DONE] (some gateways truncate) -- still end."""
        if self.started:
            return [EndEvent(stop_reason=self.stop_reason, usage=self.usage)]
        return []


def _stream_events(chunks: Iterable[bytes]) -> Iterator[StreamEvent]:
    """Sync skin over the router: SSE bytes -> normalized StreamEvents."""
    router = OpenAIStreamRouter()
    for _name, data in parse_events(iter_sse_lines(chunks)):
        events = router.feed(_name, data)
        if events and isinstance(events[0], EndEvent):
            yield events[0]
            return  # [DONE]: stop consuming immediately
        yield from events
    for event in router.finish():
        yield event


async def _astream_events(achunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
    """Async skin over the SAME router -- zero duplicated routing rules."""
    router = OpenAIStreamRouter()
    async for _name, data in aparse_events(aiter_sse_lines(achunks)):
        events = router.feed(_name, data)
        if events and isinstance(events[0], EndEvent):
            yield events[0]
            return
        for event in events:
            yield event
    for event in router.finish():
        yield event
