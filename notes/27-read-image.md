# 27 — read_image: the loop grows eyes, and one wire can't carry them

*Half of vision shipped with notes/15: the HUMAN could attach a
picture. But "what's in screenshot.png?" and "does the chart I just
made look right?" bounced off read_file's binary refusal. The model
could be shown pictures; it couldn't choose to look. Closing that loop
took ~40 lines of tool and a surprising amount of plumbing — because
one of the three wires physically cannot do the obvious thing.*

## The tool is thin; the hand-off is the design

`load_image_block` (akshara/images.py) already owned validation:
extension allowlist, 5 MB cap, base64 encoding before any request
exists. Both adapters already carried ImageBlocks in user messages.
The only new machinery is the hand-off from TOOL RESULT to HISTORY,
and it exists because of a hard constraint:

**The OpenAI-family wires have no way to put an image inside a tool
result.** Chat-completions' `role:"tool"` message and Responses'
`function_call_output` item both carry text, full stop. Anthropic's
`tool_result` CAN nest images — but routing Anthropic through its rich
path and OpenAI through a workaround would fork the transcript shapes
across dialects.

So one path for everyone: **the loop hoists.**

1. A tool's `run()` may return a `ToolOutput` — text plus images.
   Plain str stays the norm; ONE capability shouldn't drag every tool
   through a richer contract.
2. `_run_one` splits it: text truncates into the ordinary ToolResult;
   images ride an (adapter-invisible) field on it.
3. When the batch appends to history, `_batch_message` writes results
   FIRST, then hoisted ImageBlocks AFTER — one user message, same shape
   on every dialect, replayed faithfully by `_answer_outstanding` on
   Ctrl-C too.

## The fan-out adapters had to learn ordering

Anthropic's encoder is block-shaped; results-then-images just works.

The two OpenAI-dialect encoders fan one internal user message into
several wire messages, and they previously emitted any text/images
BEFORE the tool payloads. For mixed messages that order violates both
wires: tool outputs must directly follow the assistant turn that made
the calls. Both encoders now emit function outputs first, THEN a
trailing user message carrying whatever text/images remain — which
trailing message is exactly how the image reaches the model. Pure-text
and pure-results messages encode byte-identically to before; only the
newly-possible mixed shape changed.

## The silent bug this flushed out

Session checkpoints DROPPED ImageBlocks without error — `_dump_message`
had no case for them, so any conversation containing a picture silently
lost it across save/load (user attachments included; nobody had ever
checkpointed one). Now `kind:"image"` round-trips byte-exact — base64
in, base64 out, no re-encode. Lesson repeated from notes/05: a match
statement with a fall-through default isn't exhaustive handling, it's
exhaustive SILENCE. (Serialization here raises on unknown KINDS when
loading but skipped unknown blocks when dumping — the fix makes both
directions explicit for images.)

## What the tests pin

- tool: ToolOutput returned, media type right, missing/bad-extension/
  sandbox-escape all ToolError
- BOTH loop twins (sync + ScriptedProvider async): history[2] holds
  ToolResult(call_1) then ImageBlock, bytes matching the file
- anthropic wire: tool_result block followed by image blocks, in order
- openai wire: role:"tool" message(s) FIRST, then user parts carrying
  data:-URLs; responses wire: function_call_output then input_image
- plain text / pure-results messages still encode EXACTLY as before
  (parametrized over both fan-out adapters)
- checkpoint round-trip preserves result+images byte-exact, including
  plain user attachments

## Receipts

Offline suite green.

Live receipt (Ollama `gemma4:12b` — the same vision model notes/15
used, one-shot `--yolo`): a 64×64 solid-red PNG generated on the fly;
the model called read_image, the panel showed `image loaded: pic.png
(image/png, 180 bytes)` with `[1 image(s) attached to this result]`,
and its answer came back *"The dominant color of the image is red."*
— pixels traveled tool → loop → history → wire → eyes, end to end.
`2293 in / 64 out · 3 iteration(s)`.
