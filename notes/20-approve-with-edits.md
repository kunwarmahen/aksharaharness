# 20 — Approve-with-edits: the gate that talks back

*The permission prompt upgrades from binary y/n to y/n/**e** — and it
cost almost no code, because every piece it needed was already in place.*

## The whole feature is one mutability decision

`PermissionRequest` was a frozen dataclass. Unfreezing `arguments` (one
field — everything else stays loop-owned) turns approval from a rubber
stamp into a review step: **a gate may REPLACE `request.arguments`
before answering True**, and the agent loop notices via an identity
check:

```python
if allowed and request.arguments is not call.arguments:
    call.arguments = request.arguments   # what runs is what was approved
```

Two lines in each `_gate` (sync + async), and suddenly editing needs no
callback protocol, no edit-result type, no second round-trip through the
loop. The edited dict rides the normal path: execution sees it, hooks see
it, history records it. That last part matters most — *what runs is what's
kept*, so the transcript never lies about which command executed.

## The supporting seam: `summarize`

An editor changes args; the preview must change with them. But the gate
doesn't know what tool it's judging. So the loop pre-binds
`tool.summary(args, ctx)` into the request as `summarize` — any UI can
re-render for amended args without importing a single tool. Missing →
raw JSON fallback (honest, just less pretty).

## The CLI prompt

`Confirm.ask` became `Prompt.ask("run it?", choices=["y","n","e"],
default="n")`. Choosing `e` opens the injected editor on the current
args; success swaps + re-previews tagged *(edited)* + re-asks. Bad JSON
prints the parse error and re-asks; a cancelled edit returns to the
prompt — **a failed edit never silently denies**. The default editor:
pretty-printed JSON in a tempfile through `$EDITOR` when stdin/stdout
are TTYs, else one inline `input()` line (which is how piped receipts
work). Read-only tools still never prompt; sandboxed bash still skips
via `trust_sandbox` — auto-approval has no edit surface by design.

## What the tests pin

- Gate edits → Agent executes the EDITED dict; history records the edited
  form (`EchoTool` echoes its args back, so the assertion reads "ran
  {"amended": true}" — no mocking of internals).
- Edit-then-deny executes nothing (edits are proposals until True).
- `summarize` is supplied by the loop and rebinds previews.
- REPL flows with fake editors: plain-y never opens the editor;
  edit→approve shows *(edited)* + new summary; bad JSON → "bad edit" →
  explicit n; cancelled edit → re-prompt; JSON fallback summary.

## Live receipt

Ollama `qwen3.8`, real REPL: panel previewed the command, `e` opened
the inline editor, the human amended it, the panel re-previewed tagged
*(edited)* via the bash tool's own `summary()`, and `y` ran exactly
the amended form. Best detail: the model then told the user its
original command had been changed before running — the edit reached
execution, history, AND the model's own awareness.
