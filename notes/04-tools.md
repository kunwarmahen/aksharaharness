# 04 · Tools: schemas, sandboxes, and honest summaries

> Files: `akshara/tools/*`. See also `05-agent-loop.md`
> for how the loop turns tool failures into data.

## A tool is three functions glued to a schema

```python
class Tool(ABC):
    name: str; description: str
    parameters: dict        # hand-written JSON Schema
    read_only: bool         # drives auto-approval
    def spec(self) -> ToolSpec
    def summary(self, args, ctx) -> str   # for the PERMISSION PROMPT
    def run(self, args, ctx) -> str       # returns output as text
```

**Lesson: write the schema by hand, with real descriptions.** Vague
schemas produce vague tool calls — the description strings are the model's
only documentation of the tool. `additionalProperties` and explicit
`required` matter more than they look.

## `summary()` takes the same `(args, ctx)` as `run()`

First version took only `args` and resolved paths against the *import-time*
cwd — previews showed different files than execution would touch. A
permission prompt you can't trust is worse than none. Fix: same context,
same resolution, so **what you approve is exactly what runs**.

Also learned defensively: a crashing `summary()` must never take down the
approval flow itself — `_execute` falls back to a generic preview.

## The sandbox is convenience confinement, NOT security

```python
def resolve_in_sandbox(ctx, raw):
    path = (ctx.cwd / raw).resolve()
    if not path.is_relative_to(ctx.cwd.resolve()):
        raise ToolError("path escapes sandbox")
```

This covers fs tools only. `bash` can `cd /` trivially — which is precisely
why bash is permission-gated rather than sandboxed. Say this honestly in
the docs instead of implying safety that isn't there. (Since this note was
written, a real containment layer arrived behind the same tool: the
pluggable `ToolSandbox`, bubblewrap when available —
[16-sandboxing.md](16-sandboxing.md).)

Two gotchas hit while writing it:

* `Path.resolve()` follows symlinks — do containment checks AFTER resolving,
  or `/tmp/link -> /etc/passwd` slips through.
* Tests for escape rejection must create the target file first if they also
  assert a successful sibling read (`is_relative_to` doesn't care about
  existence, humans reading the test do).

## Every cap exists because of one bad afternoon

| Cap | Value | Prevents |
|---|---|---|
| read_file | 256 KB / 2000 lines | one `cat huge.log` eating the context window |
| list_dir | 500 entries | dumping `/usr/lib` into history |
| grep | 200 matches | a broad regex flooding the transcript |
| bash output | 10k chars, head+tail | same, but errors live at the END |

Truncation style matters: keep **both ends** (`head + "\n[... N chars omitted
...]\n" + tail`). Tracebacks and error summaries are at the bottom; losing
them to head-only truncation hides the reason the command failed.

## bash timeout salvage

On `TimeoutExpired`: `os.killpg(pid, SIGKILL)` (the whole process group —
`start_new_session=True` made one), then call `process.communicate()` a
**second** time. subprocess docs guarantee retrying communication loses no
output; reading `process.stdout` directly races its reader threads and
returns nothing. Partial output from a killed command is still evidence the
model can use.

**Ctrl-C needs the same treatment.** Because the child is in its own
session, an interrupt during `communicate()` does NOT reach it — without
cleanup, `sleep 25` keeps running as an orphan after the turn is
cancelled. Fix found in live testing: a `except BaseException:` arm that
kills the group, reaps via `communicate()`, and re-raises (cancellation is
not a tool failure). Found by running the mid-turn Ctrl-C scenario against
a real model that happily requested `sleep 25`.

## Argument validation is part of the tool

Model-supplied arguments are untrusted input. `require_str` /
`require_int` raise `ToolError` with a message phrased for the MODEL to
read and correct ("argument 'offset' must be an integer, got str") — not
an assert, not a traceback.

## grep grew a ripgrep backend (one contract, two engines)

`search.py` now picks its engine per call:

1. **ripgrep subprocess** when an `rg` binary is on PATH -- production
   speed on big trees. Output parsed from `--json` events, streamed line
   by line so the 200-match cap stops reading the tree (`proc.kill()`)
   instead of slurping it.
2. **pure-Python `os.walk` fallback** -- unchanged original walker; also
   used whenever rg is unusable.

The subtle part is *when the fallback fires*. Not just "binary absent":

* rust's regex dialect lacks Python features (lookaheads, backrefs) --
  rg exits nonzero -> fall back rather than half-answer;
* any spawn/pipe failure -> same.

An empty result list and a failed backend are DIFFERENT values here:
`[]` means "genuinely no matches", `None` means "this engine couldn't
run -- try the other one".

Semantic parity was forced, not hoped for. The walker ignores
`.gitignore` (deterministic across machines), so ripgrep gets
`--no-ignore --hidden` plus `-g !dir` exclusions mirroring `SKIP_DIRS`
-- otherwise the two backends would silently disagree about which files
exist. Tests pin identical output format ('path:line: text'), the cap,
include-filtering, and skip rules on BOTH sides: existing grep tests run
engine-agnostic, new ones pin each backend (a fake `rg` script makes the
subprocess path hermetic). Verified live against real rg 14.1.1: both
engines return byte-identical matches on the same tree.

## memory.py: scratchpad + retrieval (notes that outlive compaction)

The context window is working memory; `write_note` / `recall_notes` are
long-term memory. The design pressure is compaction itself: masking and
summarizing DELIBERATELY discard old tool results, so anything expensive
to learn must be written down or it stops existing after an auto-compact.

* **One JSON file, not a note directory** (`.akshara/memory.json`,
  atomic tmp+replace writes). The model should never have to remember
  WHERE its memory lives — only WHAT it called things. Topics are keys;
  rewriting one is an upsert.
* **Retrieval is ranked substring match**, topics weighted above bodies.
  Vector search would add a dependency and a mystery for zero teaching
  value; "explainable and deterministic" beats "smart" for infrastructure
  a model has to debug its own use of.
* **recall with no query returns an index** (topic + preview) rather than
  dumping every full body: browse-then-fetch keeps the tool's output size
  proportional to what was actually asked.
* Empty memory answers with GUIDANCE ("no notes yet — write_note
  persists...") instead of empty string or error — same philosophy as
  edit_file's ambiguity message: the next caller is a model deciding
  what to do.
* Honest limits stated in the docstring: single-process (two harness
  processes sharing a sandbox can lose updates) and literal substring
  matching, which is exactly as smart as it sounds.

Verified live across PROCESSES: one session saved a deploy checklist via
write_note; a brand-new process recalled it by query — persistence is
the feature, so the verification crosses the boundary it claims to.

## ask_user: a tool can hold a service, not just do work

Every tool before this one closed over *state* (memory.json's path) or
*capabilities* (a sandbox). `ask_user` closes over a **service** — a
`UserChannel` with one blocking method, `ask()`, that returns whatever
the human said. The REPL injects stdin, the web UI injects a websocket
round-trip ([22-web-ui.md](22-web-ui.md)), tests inject a canned
answer; the tool object never knows which.

Three contract details that earned their keep:

* `read_only=True` — asking costs nothing and touches nothing, so it
  never triggers the permission gate in any frontend.
* Choices are capped at 6 with a ToolError beyond — an option list is a
  hint, not a menu, and the user can always type something else.
* The blocking is honest. Tools already run on worker threads, so
  "wait for a human" is just another slow IO — bounded by the human's
  patience rather than a timeout, which is right for a question worth
  stopping work over.

The headless case gets its own exception family, not an error result:
with `channel=None` (piped stdin, cron, evals) a call raises
`UserUnavailable` — a deliberate `BaseException` so the loop's
convert-failures-to-data rule cannot hand the model a "no user found"
note it would try to explain away with a guess. Nobody home is not the
model's problem to fix.
