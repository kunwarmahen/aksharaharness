# 12 · The real-world test: an agent that builds a project

> Every prior note verified a FEATURE. This one verifies the PRODUCT:
> the harness doing Claude Code's actual job — take a spec, build a
> project, test it, run it. File: `examples/builder_demo.py`.

## The shape of the job

One `Agent`, one conversation, an empty workspace:

```
 spec ──► [write files] ──► [run acceptance commands] ──► failures?
              ▲                        │                     │
              └────────── fix ◄────────┘ ◄───────────────────┘
                                       │ all pass
                                       ▼
                          final answer: what was built
```

Nothing new had to be BUILT for this to work — that is the finding.
The loop, tools, sandbox, errors-as-data, and iteration cap from phases
4–7 were sufficient. What makes it feel like a coding agent is just
which tools are registered (`write_file`/`edit_file`/`bash`/`read_file`)
and what the spec asks for.

## Design decisions worth stealing

* **The demo trusts nothing the model says.** After the turn ends,
  `verify()` re-runs every acceptance command ITSELF — exit codes and
  stdout checked against expectations. "It said the tests pass" is not
  evidence; `subprocess.run(...) == 0` is. The process exits 0 only if
  independent verification passes, so the demo doubles as a CI gate.
* **Acceptance criteria are runnable commands**, not prose. "exit code 2
  on unknown unit" is verifiable by both the agent (mid-loop) and the
  demo (after). Vague criteria ("make it nice") cannot steer a loop.
* **Sandboxing is layered, stated honestly.** fs tools confine paths to
  `ToolContext.cwd` (the fresh tempdir) by construction; bash is NOT
  sandboxable, so the autonomous run uses the same trust decision as
  `--yolo` — printed plainly in the header rather than hidden.
* **A fresh workspace per run** (`tempfile.mkdtemp`), kept only with
  `--keep` — repeat runs are reproducible and never clobber anything.

## What the live runs showed

Every preset went BUILD GREEN on its first live attempt — the
greenfield builds and the repair job alike, the last with its
test-file checksum verified untouched afterwards.

## The repair job: reading tracebacks for a living

The `repair` preset inverts the build: the workspace starts with a
working test suite and a BROKEN implementation
([examples/broken_projects/textstats](../examples/broken_projects/) —
5 planted bugs of different flavors: whitespace splitting, integer
division, case-normalization drift, sort direction + tie-breaking,
boundary off-by-one). Offline-verified before any model saw it: 6
failures as shipped, 14/14 after the intended fixes.

Two design points make this a real test of the harness:

* **The checksum guard.** The demo hashes every `test_*.py` before the
  turn and compares after. "Make the tests pass" has a degenerate
  solution — weaken the tests — so the demo removes it and reports
  `FAIL: tests were MODIFIED` if the model takes the shortcut.
* **Docstrings-as-contract.** Each buggy function's docstring states
  correct behavior that disagrees with its code; the spec tells the
  agent where code and docstring disagree, the tests define truth.

The live run behaved like a careful engineer: read the contract tests
FIRST, then five surgical `edit_file` calls (one per bug, zero blind
rewrites), one green test run, and a diagnosis per bug matching every
planted defect. Errors-as-data carried the loop end to end — failing
tests arrived as tool output to read, not exceptions to crash on.

## Details from the build runs

Details that mattered more than the green:

* **Tests before code.** In the unitconv run the model wrote
  `test_conversions.py` FIRST, then implementation, then CLI — then ran
  everything. Nobody told it to; the spec's DEFINITION OF DONE section
  made verification salient and ordering followed.
* **Errors-as-data carried the loop.** Every bash invocation returned as
  a ToolResult; had a test failed, the failure text would have ridden
  back exactly like success output. (First runs happened not to need a
  fix cycle — the honest way to exercise one is a harder preset or a
  deliberately buggy spec; the mechanism is identical to the denial and
  timeout recoveries already tested offline.)
* **Defensive code nobody asked for.** The generated `todo.py` checks
  `isinstance(data, list)` after JSON decode; the tests isolate state by
  backing up/restoring `todo.json` and cover the CLI via subprocess.
  The model over-delivered because the summary line said "iterate until
  everything passes" — acceptance pressure shapes output quality.
* **Usage accounting honesty:** OpenRouter reported input tokens as null
  on these streaming calls, so totals show `0 in` (the known gateway
  quirk from [notes/02](notes/02-wire-formats.md), tolerated as zero).

## Cost & scope

~10–20 tool calls per run; at typical prices, single-digit cents. The
point is LOOP SHAPE, not scale — but the scale levers are already in the
codebase: sub-agents for delegation ([notes/08](notes/08-sub-agents.md)),
auto-compaction when transcripts grow ([notes/07](notes/07-reliability-and-scale.md)),
and async for N builds at once ([notes/11](notes/11-async.md)).

> This demo later graduated into the product proper —
> `akshara/builder.py` with `--build` and REPL `/build`, independent
> verification plus a bounded repair round that feeds failures back into
> the conversation. See [18-builder-mode.md](18-builder-mode.md).
