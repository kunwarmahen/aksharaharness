# Wire formats — the two dialects (reference)

> This is the spec our first two adapters implement. Keep it open while reading
> `providers/anthropic.py` and `providers/openai.py`. (A third dialect landed
> later — OpenAI's Responses API, [notes/19](19-responses-api.md) — behind the
> same interface.)

## Non-streaming

| Concern | Anthropic Messages | OpenAI chat/completions |
|---|---|---|
| Endpoint | `POST {base}/v1/messages` | `POST {base}/chat/completions` |
| Base URL (direct) | `https://api.anthropic.com` | `https://api.openai.com/v1` |
| Base URL (OpenRouter) | `https://openrouter.ai/api` | `https://openrouter.ai/api/v1` |
| Auth | `x-api-key: <key>` **plus** `anthropic-version: 2023-06-01` | `Authorization: Bearer <key>` |
| System prompt | top-level `"system"` field — NOT in messages | `messages[0] = {"role":"system", ...}` |
| `max_tokens` | **required** (400 without it) | optional |
| Tool definition | `{"name", "description", "input_schema": {...}}` | `{"type":"function", "function": {"name", "description", "parameters": {...}}}` |
| Assistant reply content | `content: [ {"type":"text"...}, {"type":"tool_use","id","name","input":{obj}} ]` | `message.content` (string or null) + `message.tool_calls: [{"id","type":"function","function":{"name","arguments":"<JSON STRING>"}}]` |
| Sending tool results back | next message `role:"user"` with `content: [{"type":"tool_result","tool_use_id","content","is_error"}]`, one block per call | one message per call: `{"role":"tool","tool_call_id","content":"<string>"}` |
| Stop signal | `stop_reason`: `end_turn` \| `tool_use` \| `max_tokens` \| `stop_sequence` \| `refusal` | `choices[0].finish_reason`: `stop` \| `length` \| `tool_calls` \| `content_filter` |
| Stop-reason mapping | `end_turn↔stop`, `tool_use↔tool_calls`, `max_tokens↔length`, `content_filter→refusal`, else `other` | (same, read right-to-left) |
| Usage | `usage.input_tokens` / `output_tokens` (+ cache fields) | `usage.prompt_tokens` / `completion_tokens` |
| Error body | `{"type":"error","error":{"type","message"}}` — can also arrive MID-STREAM as `event: error` | `{"error":{"message","type","code"}}` |

## Streaming shapes

### Anthropic — named events, blank-line delimited

```
event: message_start        → StartEvent + input usage (usage.input_tokens)
event: ping                 → ignored
event: content_block_start  → ToolCallStart (when content_block.type == "tool_use")
event: content_block_delta  → TextDelta (delta.type == "text_delta")
                              ToolCallDelta (delta.type == "input_json_delta",
                              fragments in delta.partial_json)
event: content_block_stop   → nothing to do; parsing happens in collect()
event: message_delta        → stop_reason + CUMULATIVE usage.output_tokens
event: message_stop         → EndEvent
event: error                → raise ProviderError mid-stream
```

Rules: dispatch on the `event:` name; tool arguments arrive as
`partial_json` fragments that must be concatenated and parsed ONCE
(fragments split escape sequences mid-token: `{"pa` + `th": ...`).

### OpenAI — anonymous chunks, `[DONE]` sentinel

```
data: {..., "choices":[{"delta":{"role":"assistant","content":""}}]}     first chunk
data: {..., "choices":[{"delta":{"content":"Hello"}}]}                   text
data: {..., "choices":[{"delta":{"tool_calls":[{index:0, id, function:{name, arguments:""}}]}}]}
data: {..., "choices":[{"delta":{"tool_calls":[{index:0, function:{arguments:"{\"pa"}}]}}]}  fragments
data: {..., "choices":[{"delta":{},"finish_reason":"tool_calls"}]}       terminal choice
data: {"choices":[],"usage":{...}}                                       usage (empty choices!)
data: [DONE]                                                             end sentinel
```

Rules:
- No event names — terminate on `data: [DONE]` (check BEFORE json.loads).
- Tool-call fragments carry an `index` (position in `tool_calls`); `id`/`name`
  usually appear ONLY on the first fragment for each index.
- Argument pieces concatenate per index, parsed once at `finish_reason`.
- Chunks with EMPTY `choices` carry usage or are keep-alives — skip safely.
- Usage only arrives if the request sends `stream_options: {"include_usage": true}`;
  some upstreams ignore it — model usage as possibly-zero, never crash.

## What normalization buys us

The agent loop ([05](05-agent-loop.md)) never learns any of the above. It sees:

```
provider.stream(...) -> StartEvent | TextDelta | ToolCallStart | ToolCallDelta | EndEvent
provider.complete(...) -> ModelResponse
```

Both adapters translate. That's the whole deal.

## Reading the two adapters side by side

`tests/test_adapter_parity.py` pins three properties worth internalizing:

1. **Within one adapter**: `collect(stream()) == complete()`. The event
   vocabulary is lossless -- streaming adds latency behavior, not shape.
2. **Across adapters**: equivalent wire exchanges normalize to equivalent
   ModelResponses (same text, stop_reason, usage). This is WHY the loop,
   tools, and CLI get written exactly once.
3. **The round trip diverges predictably**: send the same internal history
   through both adapters and inspect the request bodies. Anthropic ships
   tool results as blocks inside a user message; OpenAI fans them out into
   `role:"tool"` messages. Same meaning, different encoding -- absorbed at
   the adapter boundary either way.

Other things worth noticing in `providers/openai.py`:

- The auth header changes (`Authorization: Bearer`) AND the endpoint base
  convention flips: OpenAI-style base URLs INCLUDE `/v1`.
- Tool arguments travel as JSON **strings** on this wire; we parse once in
  `_parse_arguments`, and malformed model output becomes a visible
  `_unparseable_json` sentinel instead of a crash.
- There is no `is_error` flag on tool results here -- the convention is to
  mark it in the content text (`ERROR: ...`).

## Lessons from going live: gateways are a third dialect of their own

Found while running the suite live through OpenRouter (one key, both
dialects):

* **Explicit nulls vs absent keys.** `dict.get(key, 0)` only defaults on a
  MISSING key; OpenRouter's Messages-dialect emulation sends
  `"cache_creation_input_tokens": null`, so None sailed into Usage fields
  and would have crashed `Usage.add()` on any multi-iteration turn. Rule:
  when decoding counters from third-party-compatible APIs, coerce with
  `value or 0`. Regression tests pin this per adapter.
* **Thinking passthrough is NOT optional when tools are involved.**
  Reasoning arrives as ordinary content blocks (`type: "thinking"`,
  carrying an opaque `signature`). During a tool loop the FULL assistant
  message -- thinking blocks included -- must be sent back verbatim on
  the next request, or the upstream rejects it. That forced
  `ThinkingBlock` into types.py and `ThinkingDelta` into the stream
  vocabulary: preserved, not displayed. OpenAI's dialect has no such
  round-trip: its `reasoning` / `reasoning_content` fields are
  display-only, so that adapter decodes them for the UI and drops them
  on encode.
* **Gateways are a third dialect of the third dialect.** Ours emits
  *unsigned* thinking blocks yet still rejects a request where the
  `signature` FIELD is absent. Differential probe (`/tmp/think_debug.py`
  variants against the live endpoint): `"signature": ""` -> 200,
  key omitted -> 400. Lesson: for opaque provider metadata, send the
  field even when empty; presence and value have separate validation.
* **One key, two dialects, same model** (a Claude-class frontier model
  served under both endpoints): the normalization claim is testable for
  real -- identical internal types came back from both endpoints.

## redacted_thinking: an unknown block with a contract

Anthropic can return `type: "redacted_thinking"` -- reasoning the safety
system encrypted. The payload (`data`) is opaque ciphertext. The obvious
move is our unknown-block placeholder; the CORRECT move is a first-class
type, because unlike `server_tool_use` and friends this one has a
CONTRACT: it must round-trip verbatim like a signed thinking block, or
the next request of the same tool loop fails validation.

Two implementation wrinkles worth keeping:

* **Streaming shape is inverted.** Every other block arrives as deltas
  after its start event. Redacted reasoning arrives COMPLETE inside
  `content_block_start` (ciphertext has nothing human-shaped to stream)
  and then goes silent. If collect() only listened to deltas, the block
  would silently vanish in streamed turns and reappear as a 400 two
  requests later -- so a whole new StreamEvent (`RedactedThinking`)
  exists just to carry it at start time.
* **Every layer must agree it exists.** types, both encoders (verbatim
  for Anthropic; dropped for OpenAI alongside plain thinking), session
  dump/load (`kind: "redacted_thinking"`), token estimation (the
  ciphertext still spends real window), and every renderer (show THAT it
  happened, never WHAT).

Rule of thumb the project now follows: *placeholder-ize unknown blocks,
but promote any block the provider will validate on a later request.*

## Local models (Ollama): a dialect profile, not a dialect

Adding Ollama cost ~15 lines of wire code -- because Ollama serves an
OpenAI-compatible `/v1/chat/completions`, so `OllamaProvider` is just
`OpenAIProvider` with `name = "ollama"`. That is the payoff of
normalizing at one layer: **a provider = dialect + profile**. The
profile carries what actually differs:

* **Auth**: none locally; we still send `Bearer ollama` so the header
  exists (proxies in front of Ollama may want a real key).
* **Context window**: auto-compaction does its math against a window
  assumption. Cloud default is 200k; an 8k local pull under that
  assumption would never compact before the server rejected the
  request. Per-provider defaults (`OLLAMA_CONTEXT_WINDOW`) fix the
  guess.
* **Identity**: session checkpoints store the provider NAME, so a
  distinct class (not `get_provider("openai", ...)` with tweaked env)
  is what makes `--resume` rebuild *Ollama* and not plain openai.

Same pattern would absorb vLLM / LM Studio / llama.cpp servers: find
the OpenAI-compatible endpoint, write a profile.
