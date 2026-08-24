# 18 — Builder mode: verify, don't trust

*Companion to [notes/12](12-builder.md), which proved an agent COULD
build a project. This note is the promotion of that demo into a
product: `akshara/builder.py`, `--build`, `/build`.*

## What changed when it became a library

The demo had one shape; the library has a contract:

```python
spec = BuildSpec(task="...", checks=default_checks(),   # unittest discover gate
                 seed_dir=None, checksum_tests=True)
result = run_build(agent_factory, spec, workspace,
                   on_event=..., max_iterations=30)
```

`agent_factory(workspace) -> Agent` — the builder never constructs
agents itself, so callers control provider, model, permissions, and
tools (the CLI passes a yolo/trust_sandbox factory; tests pass a
ScriptedProvider one). `run_build` returns a frozen `BuildResult(ok,
elapsed_seconds, files, response_text, checks, tampered_tests, usage…)`
— exit-code-as-CI-gate falls straight out of `ok`.

## The verification contract

**Trust nothing the model claims.** After every agent turn the builder
re-runs each acceptance check ITSELF via `subprocess.run`:

* expected exit code matched exactly (non-zero expectations are legal:
  "this program should reject bad input" is a spec like any other);
* expected substring matched against stdout on a clean exit, stdout+stderr
  otherwise;
* a check that exceeds ITS OWN 120s budget fails with what was salvaged.

Because verification is independent, a model that *says* "all tests
pass" over broken code still fails the build
(`test_lying_model_fails_verification`). That test is the feature.

## The checksum contract

With `checksum_tests=True`, every `test_*.py` present after seeding is
hashed; any byte change during the build marks `tampered_tests` and
fails the build EVEN IF THE SUITE THEN PASSES. This closes repair-job
cheating: the cheapest way to make failing tests pass is to weaken the
tests. Legitimate new files are fine — only pre-existing checksummed
files are watched.

## Where it lives in the product

| Surface | Shape |
|---|---|
| `uv run akshara --build "SPEC"` | one-shot: workspace defaults to `.akshara/builds/<ts>/` (override with `--cwd`); prints per-check PASS/FAIL + BUILD GREEN/RED; **exit 0/1** so CI can gate on it |
| REPL `/build TASK` | child Agent via the SubagentSpawner pattern: same provider/model, BUILD_SYSTEM prompt, fresh workspace under the session's cwd, parent history untouched; streams through the normal renderer |
| `examples/builder_demo.py` | now a thin wrapper over `run_build` with its presets (`unitconv`, `todo`, `repair`) as BuildSpec data |

Gate policy is honest about containment: a confined sandbox
([notes/16](16-sandboxing.md)) gets `trust_sandbox(confirm_gate)` and
runs unattended; unconfined runs get yolo WITH A LOUD WARNING printed —
the current stance, not a hidden default.

## Iteration loop

`run_build` drives `Agent.run_streaming`, then verifies independently.
On RED it does what Claude Code does: feeds the failures back into the
SAME conversation (`_failure_report` — exact commands, actual vs
expected exits, output tails) and lets the agent fix its work, up to
`spec.max_repair_rounds` times (default 2), re-verifying after each.
The feedback is conversation data like everything else — history
invariant untouched, same agent, full context of what it built.

Two deliberate sharp edges:

* **Tampering is never repaired.** A checksummed test that changed ends
  the build immediately — "please un-weaken my tests" is not a repair
  conversation worth having.
* **A provider dying mid-repair keeps the last verdict** (bounded
  effort, honest RED); only KeyboardInterrupt propagates.

`max_iterations=30` bounds the agent's own loop; `on_event` receives
the raw stream events across ALL rounds, which is how the CLI renderer
and REPL renderer reuse one code path.
