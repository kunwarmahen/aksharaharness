"""Adapter for the OpenAI Responses API wire format.

Endpoint:  POST {base}/responses   (base INCLUDES /v1, same contract as openai)
Auth:      Authorization: Bearer header
Reference: https://platform.openai.com/docs/api-reference/responses
           https://openrouter.ai/docs/api_reference/responses/overview

The third dialect behind the identical Provider interface (chat-completions'
successor). What actually changes vs the openai adapter -- and what doesn't:

* history is ONE flat ``input`` array of TYPED ITEMS, not ``messages`` --
  but a user turn holding ToolResults still fans out into multiple items
  (``function_call_output``), exactly as it fanned out into role:"tool"
  messages on chat-completions
* the system prompt rides top-level as ``instructions``
* tool definitions are FLAT ({name, description, parameters} at top level)
  -- no {"type":"function","function":{...}} wrapper
* tool calls come back as ``function_call`` items with arguments as a JSON
  STRING, keyed by ``call_id``; results return keyed by the same call_id
* max_tokens is named ``max_output_tokens``
* streams are NAMED SSE events (Anthropic-style!) ending with the same
  literal ``data: [DONE]`` sentinel (OpenAI-style) -- sse.py's grouper
  already yields ``(event_name, data)`` pairs, so this adapter routes on
  the name and falls back to the payload's own ``type`` field
* usage arrives ONLY on the terminal event's response object; caching is
  automatic upstream, hits visible at input_tokens_details.cached_tokens
* reasoning arrives as ``reasoning`` items / summary deltas -- display-only;
  like chat-completions there is no way to send reasoning back

Statelessness note: we never send ``store``/``previous_response_id``, so
every request carries full history -- which is what our internal-types
history already does.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

import httpx

from akshara.errors import ProviderError
from akshara.providers.base import Provider, provider_error_for
from akshara.providers import retry as _retry  # sleeps go through the module attr
from akshara.providers.openai import _parse_arguments, openai_error_for
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

#: Reasoning summaries stream on their own channel, not indexed items --
#: an index that can never collide with real output_index values.
REASONING_INDEX = -2


class ResponsesProvider(Provider):
    name = "responses"

    # ---- request building -------------------------------------------------

    def _url(self) -> str:
        return "/responses"

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
        """Internal types -> Responses API JSON. Public for examples."""
        input_items: list[dict[str, Any]] = []
        for m in messages:
            input_items.extend(_encode_message(m))

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            # Named differently from every other dialect; same meaning.
            "max_output_tokens": max_tokens,
            "stream": stream,
        }
        if system is not None:
            body["instructions"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            # FLAT definitions: name/description/parameters at top level.
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
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
            request = self.client.build_request(
                "POST", self._url(), headers=self._headers(), json=body
            )
            return self.client.send(request, stream=True)

        def _classify(response: httpx.Response) -> Exception:
            response.read()  # an error body arrives as a stream too
            return self._error_for(response)

        # THE STREAMING RULE (retry.py), both notches -- identical loop to
        # the other adapters': full opening retries; a 200 that dies before
        # its first event re-opens under the same budgets (nothing
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
        blocks: list[Any] = []
        for item in data.get("output") or []:
            match item.get("type"):
                case "reasoning":
                    # Display-only summary text (there is no field to send
                    # reasoning back through on this dialect either).
                    text = "".join(part.get("text", "")
                                   for part in item.get("summary") or [])
                    if text:
                        blocks.append(ThinkingBlock(text))
                case "message":
                    for part in item.get("content") or []:
                        if part.get("type") == "output_text" and part.get("text"):
                            blocks.append(TextBlock(part["text"]))
                        elif part.get("type") == "refusal" and part.get("refusal"):
                            # Refusals surface as plain text -- the loop
                            # treats them as content, because they are.
                            blocks.append(TextBlock(part["refusal"]))
                case "function_call":
                    blocks.append(ToolCall(
                        id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name", ""),
                        arguments=_parse_arguments(item),
                    ))
                case _:
                    pass  # unknown item types (web_search_call, ...) ignored

        usage_raw = data.get("usage") or {}
        # Automatic upstream caching; hits are only visible here, riding
        # INSIDE input_tokens on the wire -- subtract them out so the
        # stored counter means full-rate tokens only (see types.Usage).
        cached = ((usage_raw.get("input_tokens_details")
                   or {}).get("cached_tokens")) or 0
        usage = Usage(
            # `or 0` not .get-default: explicit nulls would poison Usage.add().
            input_tokens=max(0, (usage_raw.get("input_tokens") or 0) - cached),
            output_tokens=usage_raw.get("output_tokens") or 0,
            cache_read_tokens=cached,
        )
        return ModelResponse(
            message=Message("assistant", blocks),
            stop_reason=_map_status(data),
            usage=usage,
            model=data.get("model", ""),
            raw=data,
        )

    def _error_for(self, response: httpx.Response) -> ProviderError:
        # OpenAI-family error bodies -- shared verbatim with openai.py.
        return openai_error_for(response)


# ---------------------------------------------------------------------------
# Encoding: internal -> wire
#
# One internal user message can become SEVERAL input items (each
# ToolResult its own function_call_output), mirroring chat-completions'
# fan-out into role:"tool" messages. Assistant turns become one message
# item PLUS one function_call item per call.
# ---------------------------------------------------------------------------


def _encode_message(message: Message) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if message.role == "assistant":
        # ThinkingBlocks dropped DELIBERATELY (see openai.py): this wire
        # has no way to carry reasoning back either.
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        if text:
            out.append({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            })
        for call in message.tool_calls():
            out.append({
                "type": "function_call",
                # call_id is the linkage key; results echo it back.
                "call_id": call.id,
                "name": call.name,
                # arguments travel as a JSON STRING on this wire too
                "arguments": json.dumps(call.arguments),
            })
        return out

    # user role: text/images become ONE user message item (typed parts);
    # each ToolResult becomes its own function_call_output item. No
    # is_error flag exists here either -- mark it in the content text.
    parts: list[dict[str, Any]] = []
    for block in message.content:
        match block:
            case TextBlock(text=t) if t:
                parts.append({"type": "input_text", "text": t})
            case ImageBlock(media_type=mime, data=b64):
                parts.append({"type": "input_image",
                              "image_url": f"data:{mime};base64,{b64}"})
    if parts:
        out.append({"type": "message", "role": "user", "content": parts})
    for block in message.content:
        match block:
            case ToolResult(tool_call_id=cid, content=result_text, is_error=is_err):
                content = f"ERROR: {result_text}" if is_err else result_text
                out.append({
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": content,
                })
    return out


# ---------------------------------------------------------------------------
# Decoding: status -> StopReason
# ---------------------------------------------------------------------------


def _map_status(data: dict[str, Any]) -> StopReason:
    """Responses reports WHY IT STOPPED via status + incomplete_details,
    plus implicit tool_use when any function_call item exists."""
    status = data.get("status")
    if status == "completed":
        calls = any(item.get("type") == "function_call"
                    for item in data.get("output") or [])
        return "tool_use" if calls else "end_turn"
    if status == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            return "max_tokens"
        return "other"
    if status == "failed":
        return "other"
    return "other"


def _extract_usage(data: dict[str, Any]) -> Usage:
    usage_raw = data.get("usage") or {}
    # Same convention as _response_to_internal: cached hits ride inside
    # input_tokens on the wire and are subtracted back out here, so the
    # stored counter means full-rate tokens only (see types.Usage).
    cached = ((usage_raw.get("input_tokens_details")
               or {}).get("cached_tokens")) or 0
    return Usage(
        input_tokens=max(0, (usage_raw.get("input_tokens") or 0) - cached),
        output_tokens=usage_raw.get("output_tokens") or 0,
        cache_read_tokens=cached,
    )


# ---------------------------------------------------------------------------
# Streaming: named SSE events -> normalized StreamEvents
# ---------------------------------------------------------------------------

#: Terminal events -- any of these ends the stream (OpenRouter documents
#: ``response.done`` where OpenAI says ``response.completed``).
_TERMINAL_EVENTS = frozenset({
    "response.completed", "response.done",
    "response.incomplete", "response.failed",
})


class ResponsesStreamRouter:
    """Stateful SSE-event -> StreamEvent router, fed one named event at a
    time by BOTH transport loops.

    Wire sequence for reference:

        response.created                     -> StartEvent
        reasoning summary deltas             -> ThinkingDelta (REASONING_INDEX)
        output_item.added (function_call)    -> ToolCallStart (id/name HERE)
        function_call_arguments.delta        -> ToolCallDelta per fragment
        output_text.delta                    -> TextDelta
        terminal event w/ full response      -> EndEvent (stop_reason + usage)
        data: [DONE]                         -> EndEvent, end of stream

    Two robustness rules learned from gateway variance:

    * the SSE ``event:`` name wins, but a missing name falls back to the
      payload's own ``type`` field (they always match when both exist)
    * argument fragments MAY arrive only as a single ``.done`` event, or
      even just embedded in ``output_item.done`` -- whichever fires first
      for an index delivers that call's arguments as one delta, so
      collect() parses complete JSON no matter how fragmented the feed
    """

    def __init__(self) -> None:
        self.stop_reason: StopReason = "other"
        self.usage = Usage()
        self.started = False
        self.finished = False
        self.arg_seen: set[int] = set()  # indices that got >=1 args fragment

    def feed(self, name: str | None, data: str) -> list[StreamEvent]:
        # The sentinel is NOT JSON -- check before parsing.
        if data.strip() == "[DONE]":
            self.finished = True
            return [EndEvent(stop_reason=self.stop_reason, usage=self.usage)]

        try:
            chunk = json.loads(data) if data else {}
        except json.JSONDecodeError:
            return []  # tolerate malformed keep-alives

        etype = name or chunk.get("type")

        if etype == "response.created":
            if not self.started:
                self.started = True
                model = (chunk.get("response") or {}).get("model", "")
                return [StartEvent(model=model)]
            return []

        if etype == "response.output_text.delta":
            if delta := chunk.get("delta"):
                return [TextDelta(delta)]
            return []
        if etype == "response.content_part.delta":
            # OpenRouter alias for output_text.delta (documented shape:
            # the delta string sits under "delta").
            if delta := chunk.get("delta"):
                return [TextDelta(delta)]
            return []

        if etype in ("response.reasoning_summary_text.delta",
                     "response.reasoning_text.delta"):
            if delta := chunk.get("delta"):
                return [ThinkingDelta(index=REASONING_INDEX, text=delta)]
            return []

        if etype == "response.output_item.added":
            item = chunk.get("item") or {}
            if item.get("type") == "function_call":
                index = chunk.get("output_index", 0)
                return [ToolCallStart(
                    index=index,
                    id=item.get("call_id") or item.get("id") or "",
                    name=item.get("name", ""),
                )]
            return []  # message/reasoning items need no start event

        if etype == "response.function_call_arguments.delta":
            index = chunk.get("output_index", 0)
            if piece := chunk.get("delta"):
                self.arg_seen.add(index)
                return [ToolCallDelta(index=index, partial_json=piece)]
            return []

        if etype == "response.function_call_arguments.done":
            index = chunk.get("output_index", 0)
            arguments = chunk.get("arguments") or ""
            if index not in self.arg_seen and arguments:
                # No fragments ever arrived: the whole payload IS the
                # arguments. One synthetic delta keeps collect()'s
                # parse-once-at-end contract intact.
                self.arg_seen.add(index)
                return [ToolCallDelta(index=index, partial_json=arguments)]
            return []

        if etype == "response.output_item.done":
            item = chunk.get("item") or {}
            if item.get("type") == "function_call":
                index = chunk.get("output_index", 0)
                arguments = item.get("arguments") or ""
                if index not in self.arg_seen and arguments:
                    # Last-resort delivery path (minimal gateways).
                    self.arg_seen.add(index)
                    return [ToolCallDelta(index=index, partial_json=arguments)]
            return []

        if etype in _TERMINAL_EVENTS:
            response = chunk.get("response") or {}
            self.stop_reason = _map_status(response)
            self.usage = _extract_usage(response)
            self.finished = True
            return [EndEvent(stop_reason=self.stop_reason, usage=self.usage)]

        if etype == "error":
            # Mid-stream failure (fields: code/message/param).
            detail = chunk.get("message") or json.dumps(chunk)[:200]
            raise ProviderError(f"mid-stream error event: {detail}")

        return []  # response.in_progress, content_part.added/done, pings...

    def finish(self) -> list[StreamEvent]:
        """Stream ended without a terminal event (gateways truncate)."""
        if self.started and not self.finished:
            return [EndEvent(stop_reason=self.stop_reason, usage=self.usage)]
        return []


def _stream_events(chunks: Iterable[bytes]) -> Iterator[StreamEvent]:
    """Sync skin over the router: SSE bytes -> normalized StreamEvents."""
    router = ResponsesStreamRouter()
    for name, data in parse_events(iter_sse_lines(chunks)):
        events = router.feed(name, data)
        if events and isinstance(events[0], EndEvent):
            yield events[0]
            return  # terminal event or [DONE]: stop consuming immediately
        yield from events
    for event in router.finish():
        yield event


async def _astream_events(achunks: AsyncIterator[bytes]) -> AsyncIterator[StreamEvent]:
    """Async skin over the SAME router -- zero duplicated routing rules."""
    router = ResponsesStreamRouter()
    async for name, data in aparse_events(aiter_sse_lines(achunks)):
        events = router.feed(name, data)
        if events and isinstance(events[0], EndEvent):
            yield events[0]
            return
        for event in events:
            yield event
    for event in router.finish():
        yield event
