# 10 · Evals: measuring right behavior, not just working code

> Book ch19. Files: [evals.py](../src/akshara/evals.py),
> [`examples/run_evals.py`](../examples/run_evals.py) (the live gate),
> `tests/test_evals.py`.

## Evals are not tests

Tests: binary, free, deterministic, every commit — they protect
MECHANICS (scripted providers prove the loop, adapters, gates). Evals:
probabilistic pass RATES over real models, cost money, and protect
BEHAVIOR — completion, tool discipline, budgets — so they run on a
merge/nightly cadence and before model upgrades. Thesis (Hamel Husain,
via ch19): agent complexity is only justified when you can define
precise task-success criteria; without evals, features are debt.

## Four metric classes, checked per case, failures ACCUMULATED

* **completion** — did it finish? Crashes and provider errors become
  failed RESULTS with `error=` set; one broken case never aborts the
  suite.
* **correctness** — deterministic `check_answer` wherever a function
  will do; `judge()` (below) only for genuinely subjective criteria.
* **process validity** — `required_tools` / `forbidden_tools`, checked
  against what ACTUALLY EXECUTED.
* **cost** — token and iteration ceilings. A correct answer at 50K
  tokens is worse than the same answer at 5K.

A case can fail five ways at once and reports ALL of them — first-fail
short-circuiting throws away exactly the information you diagnose with.

## The recording trick

The runner wraps every tool in `_RecordingTool`, which appends each
real execution to a shared list before delegating. Wrapping at
`run()` (not registry lookup) means it records executions — including
work done by sub-agents through the same tool instances (see the
setup seam below for how spawn-capable cases get wired).

Fresh Agent per case — golden trajectories are independent, no history
bleed (test-pinned: every request in a two-case run carries exactly one
user message).

## LLM-as-judge, deliberately boring

Reply 'PASS' or 'FAIL' plus one sentence; parse by prefix;
anything ambiguous FAILS CLOSED. Two limits from the chapter worth
internalizing: judge and candidate sharing weights correlates their
errors (prefer a different model for judging), and no judge grades past
its own ceiling. Composition, not special-casing:
`check_answer=lambda a: judge(provider2, "haiku", ...)[0]`.

## Live-run lessons (the suite teaching its own lesson)

First real run: **4/5**, exit 1 — the gate worked immediately.
`notes-count` failed "required tool not used: list_dir"; the report
showed `[bash]`: the model counted via shell instead. Correct answer,
"wrong" process — which exposed a CASE bug, not a model bug: *a
neutrally-worded question must not carry a tool requirement.* Reshaped
into two cases with different intents:

* `notes-count` — path-neutral (`check_answer` + budget): measures cost
  and completion, any sane route.
* `list-dir-discipline` — prompt explicitly demands the listing tool;
  required+forbidden now measure INSTRUCTION-FOLLOWING.

Second run after reshaping: **6/6, exit 0**. Also added mid-flight:
reports print executed tools per line (`[bash,bash]`) — an eval report
that can't answer "what did it actually do" isn't diagnosable.

Other cases earn their place: `multiply-via-bash` forces actual
execution (required bash); `grep-discipline` forbids bash while asking
a grep-shaped question; `premature-finalization-trap` demands all five
squares across five separate calls (its live trace shows
`[bash,bash,bash,bash,bash]` — the process check sees each one).

## The per-case setup seam (`EvalCase.setup`, `spawn_setup()`)

The runner builds a fresh Agent per case; some cases want EXTRA machinery
on that agent — the seam is one field:

```python
EvalCase(..., required_tools=["spawn_subagent", "list_dir"],
         setup=spawn_setup())          # this case's agent gets a spawner
```

`run_case` applies `setup(agent)` after construction, then re-wraps the
registry idempotently so setup-registered tools are recorded too — an
execution lands in the seen-list exactly once no matter how many proxy
layers it sits under.

What this buys for sub-agents specifically: `required_tools=
["spawn_subagent"]` converts the narrate-instead-of-delegate failure
mode (child described in prose, never invoked) into a visible process
failure. And because sub-agent catalogs copy the SAME wrapped tool
instances, the child's executions flow into the parent case's record —
documented conflation (delegation is the parent's doing), which is
exactly what lets a case assert "the child really called list_dir".
Child budgets are eval-tight by default (3 spawns / 10 iterations) so a
runaway delegator can't torch the suite's cost ceiling.

## The async twin (`AsyncEvalRunner`, `--async`)

Golden trajectories are independent, so the natural scale lever is
running them at once. `AsyncEvalRunner` is the twin of `EvalRunner`:
same cases, same grading — literally the same `_result()` method —
one event loop driving several trajectories concurrently.

Two deliberate differences from the sync runner:

* **Per-case seen-lists.** The sync runner shares one list and clears
  it per case (fine sequentially); concurrent cases would scribble on
  each other's records. Each async case gets its own.
* **A semaphore bounds in-flight cases** (default 4). Independent
  trajectories, shared provider: N-at-once multiplies request rate,
  so the cap is a courtesy to rate limits, not a correctness need.

Everything else carries over unchanged: crashes become failed results
even under `gather` (one broken case can't sink the batch), results
come back in SUBMISSION order regardless of finish order (same
contract as parallel tool batches), and `spawn_setup()` works as-is —
the spawner reads only attributes `AsyncAgent` mirrors, and its child
is a sync `Agent` run via the spawn tool's `to_thread` default
`arun`, off the loop ([notes/11](11-async.md)).

Run with `--async` and the runner measures itself: same suite, both
orders, verdicts printed per case — the wall-clock win is visible live.

```bash
uv run --env-file .env python examples/run_evals.py --async
```

Offline tests pin the contract: field-for-field parity with sync
scoring, submission-order results under reversed finish order,
seen-list isolation between concurrent cases, crash-becomes-result
under gather, and the semaphore actually bounding in-flight cases.

## Fossils: production failures become regression cases

`case_from_trace(trace_id, reason, message, tokens_used=...)` builds a
case whose budget is observed-cost × 1.5 — the fix must not cost more
than the bug. Workflow: monitoring flags failure → human confirms
regression-worthiness → fossil committed → CI blocks recurrence.
"Every real failure in production leaves a fossil in the suite."

## Exit code as gate

`run_evals.py` exits 1 on any failure — wire into CI as pre-merge /
pre-model-upgrade / nightly. Not per commit: cost and flakiness are
real, and probabilistic red doesn't belong on every push.
