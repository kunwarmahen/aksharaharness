# 03 · SSE parsing and the fold

> A Server-Sent-Events parser written from the spec, a streaming path
> for the Anthropic adapter, and `collect()` — the function that folds
> a stream of events back into one ModelResponse.

## SSE in one paragraph

An SSE body is a text stream of "events" separated by blank lines. Each
line is `field: value` (strip exactly ONE leading space). `data:` lines
accumulate; `event:` names the event (Anthropic sends names, OpenAI never
does); lines starting with `:` are keep-alive comments; a blank line
dispatches. OpenAI ends its stream with a literal `data: [DONE]`.

## The two bugs networks cause (and our tests catch)

1. **Multibyte characters split across TCP chunks.** `中` is bytes
   `\xe4\xb8\xad` — the network can deliver `\xe4` in one chunk and
   `\xb8\xad` in the next. Decoding each chunk separately corrupts it.
   Fix: `codecs.getincrementaldecoder("utf-8")` buffers incomplete
   sequences. See `test_multibyte_char_split_across_chunks`.

2. **A `\r\n` split across chunks.** If a chunk ends in `\r`, is that a
   lone-CR terminator (spec-legal!) or the first half of `\r\n`? Decide
   too early and you emit a spurious empty line — which would terminate
   an SSE event prematurely. Fix: when a lone trailing `\r` sits at the
   end of the buffer, wait for more data. See
   `test_cr_at_chunk_boundary_is_not_two_lines`.

## The state machine that isn't

The Anthropic streaming adapter (`_stream_events`) is deliberately
near-stateless: it only RE-ROUTES wire events into our five event kinds.
It accumulates nothing. All accumulation — text fragments, tool-argument
JSON fragments — lives in `collect()` in `providers/base.py`, which is
shared with the OpenAI adapter.

Why: there is exactly one place that turns fragments into objects, so
both providers get identical semantics for free, and parity tests
(`collect(stream()) == complete()`) can hold each adapter to it.

## The accumulation rules (collect)

- Text fragments append to the trailing TextBlock, or start a new one —
  blocks stay in arrival order, so text-then-tools reads back exactly as
  the model emitted it.
- Tool-call argument JSON accumulates per stream index and is parsed
  exactly ONCE at the end. Never parse per-fragment: a fragment can cut
  an escape mid-token (`{"pa` + `th": ...`).
- Stream index → ToolCall object is a dict lookup, not a positional
  guess. Anthropic block indices count TEXT blocks too (tool at index 1
  is the FIRST tool call); getting this wrong silently yields `{}` args.
- Malformed argument JSON becomes `{"_unparseable_json": raw}` — visible
  to the model as an error later, never a crash here.

## Cancellation is free with generators

`stream()` is a generator wrapping `client.send(request, stream=True)`.
When the consumer closes it early (Ctrl-C), GeneratorExit propagates at
the current `yield`, the `finally` closes the response, the socket is
released. No callbacks, no asyncio, no cleanup plumbing.
