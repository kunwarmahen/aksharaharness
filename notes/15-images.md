# 15 — Images: one block, two dialects, zero loop changes

*Companion to every adapter note; the third feature (after thinking
blocks and caching) that lives entirely at the type boundary.*

## The shape

`ImageBlock(media_type, data)` in `types.py` — base64 payload WITHOUT
the `data:` prefix. Input-side only: models answer in text/thinking/
tool calls, never with images, so there is no decode half. That makes
images the simplest Block: the adapters translate it on the way OUT and
nothing on the way IN.

## Three seams

| Seam | Job | File |
|---|---|---|
| load | file → validated ImageBlock, before any turn | `akshara/images.py` |
| encode | ImageBlock → wire dialect | both adapters' `_encode_*` |
| attach | text first, then images, ONE user message | `Agent.run_streaming` / async twin |

The loader validates up front — missing file, unsupported extension,
oversize — because a usage error should exit 2 with a clear message,
never pollute history or become turn data. Deliberately shallow
validation (extension→MIME + size cap; no magic-byte sniffing): a wrong
extension earns a provider-side 400 the user sees verbatim, which
teaches more than silently second-guessing them. Limits mirror
Anthropic's documented image constraints (5 MB per image, measured on
RAW bytes — providers size limits in decoded bytes even though the wire
carries base64).

## What each dialect does with it

| Concern | Anthropic | OpenAI |
|---|---|---|
| encoding | content block `{type:"image", source:{type:"base64", media_type, data}}` | part `{type:"image_url", image_url:{url:"data:<mime>;base64,<data>"}}` |
| message shape | blocks are always an array — nothing changes | **only** multimodal messages become a parts array; image-free requests keep the plain string, byte-identical to before vision existed |

That last row is this feature's real lesson: adding a capability must
not perturb requests that don't use it. `test_plain_messages_keep_the_string_shape`
pins it.

## Loop contract

`run(user_input, images=[...])` appends images AFTER the text block in
one user message (both dialects preserve order). No new events, no new
stop reasons, no loop edits — same as hooks. Compaction was already
block-polymorphic via `estimate_tokens`; images bill by decoded size
(`len(b64) * 3 // 4`, generous on purpose — overestimating triggers
compaction early, underestimating risks overflow 400s).

CLI: `--image PATH` is repeatable and fails fast — flag without a
prompt, or any loader error, exits 2 before MCP sessions open or a turn
starts.

Interactive mode stages with `/image PATH...`: validated at attach time
(a typo errors immediately, not after three more turns), held in
`Repl._pending_images`, consumed by the NEXT message — so you can
attach first and think while composing. Staging survives slash
commands (`/help` between attach and send loses nothing);
`/image` alone shows the stage, `/image clear` unstages. Quoted paths
survive via shlex. Same seam as the flag — nothing new reaches the
adapters.

## Twins discipline

Sync `Agent` and `AsyncAgent` grew the identical parameter; tests pin
that both put `[TextBlock, ImageBlock]` into the scripted provider's
request untouched.

## Test map

`tests/test_images.py`: loader round-trip / MIME aliases / missing file /
unsupported ext / size cap (monkeypatched small) · Anthropic
source-object encoding · OpenAI parts-array + string-shape preservation ·
sync + async attach · context-estimate formula · CLI wiring incl.
exit-code paths. `tests/test_repl.py::TestImageCommand`: staging rides
the next turn / attach-time errors / survives slash commands in between /
bare-command report / clear / quoted paths.

## Try it yourself

Same path cloud or local — file → loader → dialect → gateway → answer:

```bash
# needs a key in .env:
uv run akshara --yolo --image dot.png "One word: what color dominates this image?"

# fully local, no key:
printf '/image dot.png\nOne word: dominant color?\n/quit\n' \
  | uv run akshara --provider ollama --model gemma4:12b --yolo
```

The REPL route (`/image`) stages onto your *next* message — staging,
consumption by the following turn, data-URL parts array, a *local*
vision model answering in one shot — while one-shot `--image` skips
the staging entirely.
