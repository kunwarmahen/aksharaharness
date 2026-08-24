# 19 — The third dialect: OpenAI's Responses API

*The adapter exercise again, one newer wire this time: a fresh
adapter behind the identical Provider interface, and the internal types
absorb it in one file.*

## Why this wire exists

Chat-completions grew by accretion: tool calls bolted onto `message`,
reasoning squeezed into unofficial fields (`reasoning_content`), usage
trickling in on empty-choices chunks. The Responses API is OpenAI's
rewrite — one flat `input[]` array of TYPED ITEMS instead of
role-messages, named SSE events instead of anonymous chunks, and a
terminal event that finally carries usage deterministically.
OpenAI-compatible surfaces converged on it: OpenRouter serves a
stateless beta at the same path, and Ollama has served `/v1/responses`
since 0.13.3.

## The cheat-sheet row that matters

| Concern | chat-completions | Responses |
|---|---|---|
| history | role messages; user turns fan out into `role:"tool"` messages | ONE flat `input[]`: `message`, `function_call`, `function_call_output` items |
| tools | `{type:"function", function:{...}}` wrapper | FLAT `{type:"function", name, description, parameters}` |
| system | `messages[0] role:"system"` | top-level `"instructions"` |
| budget | `max_tokens` | `max_output_tokens` |
| stop | `finish_reason` per-choice | top-level `status` + any `function_call` item ⇒ `tool_use` |
| stream | anonymous chunks + `[DONE]` | NAMED events (`response.output_text.delta`, ...) + `[DONE]` |

The funny part: **the new dialect is Anthropic-shaped in three places**
(top-level system field, typed item array like content blocks, named
SSE events) while staying OpenAI-shaped in the ones that hurt (arguments
still a JSON string, results still fan out one-item-per-result, errors
still `{"error": {...}}`). A third column teaches what neither pair of
columns could: which wire traits are *OpenAI-vs-Anthropic* and which are
just *old-vs-new*.

## What reused, what didn't

Reused untouched: `sse.py` (its `(event_name, data)` pairs were built
for Anthropic's named events — now they carry Responses' names too),
`provider_error_for` + the whole retry skeleton (opening retries,
pre-first-event reconnect), `_parse_arguments` (arguments-as-string is
inherited from chat-completions), the error-body parser (extracted from
`openai.py` into module-level `openai_error_for` so both OpenAI-family
adapters share it).

New: `ResponsesStreamRouter`. Its one genuinely tricky job is gateway
variance around argument fragments:

* official streams send `function_call_arguments.delta` fragments;
* some backends send ONLY `function_call_arguments.done`;
* minimal ones embed `arguments` solely in `output_item.done`.

Rule: whichever arrives FIRST for an index delivers the arguments as one
synthetic delta — `collect()` keeps parsing exactly once at the end no
matter how fragmented (or not) the feed was. Same tolerance for event
names: OpenRouter documents `response.done` / `content_part.delta`
where OpenAI says `response.completed` / `output_text.delta`; the router
accepts both, and falls back to the payload's own `type` field when an
SSE `event:` name is missing.

Reasoning stays display-only (like chat-completions): summary deltas map
to ThinkingDelta on reserved index -2, and reasoning items decode to
ThinkingBlocks but are dropped on encode — there is still no way to send
reasoning back through this wire. (Ollama cheekily fills
`encrypted_content` with PLAINTEXT summary text — treated as opaque,
because the contract is opacity.)

Statelessness is assumed, never sent: we transmit neither `store` nor
`previous_response_id`, so full history rides every request — which our
internal-types history already guarantees. OpenRouter rejects stateful
requests with a 400; against them we'd fail fast either way.

## Test map

`tests/test_responses_adapter.py`: request anatomy (path/auth/
instructions/flat tools/`max_output_tokens`) · tool round-trip fan-out
into items with call_id linkage + ERROR-prefix convention · image parts ·
thinking dropped on encode · fixture parse (text/reasoning/function_call/
cached tokens) · status→StopReason incl. incomplete/max_output_tokens ·
router units (fragmented vs `.done`-only vs embedded-only arguments,
aliases, missing SSE names, error events raise mid-stream, malformed
JSON tolerated, truncated-stream finish) · auth/429+retry-after/overflow
mapping · opening-5xx retry wiring.

Parity grew its third column: `tests/fixtures/responses_text.{sse,json}`
encode the same canonical exchange ("Hello there", 17 in / 9 out) as the
other two dialects, `collect(stream()) == complete()` holds within the
adapter, and cross-dialect equivalence now loops all three.

## Live receipts

Local first (`qwen3.8` on Ollama's `/v1/responses`): one-shot "PONG"
with the reasoning summary arriving as thinking deltas, then a tool
loop — `read_file` on README.md answered with its first heading on
the second iteration, usage accumulating across both requests. Then
the cloud leg, one tiny call against OpenRouter's stateless beta
(`openai/gpt-4o-mini`), also "PONG".

One key, three dialects, zero adapter-visible differences above the
boundary.
