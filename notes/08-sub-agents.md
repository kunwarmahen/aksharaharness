# 08 · Sub-agents: agent-as-tool

> Book ch15 adapted to the sync-generator harness.
> Files: `subagent.py` (~230 lines incl. docstring), `tests/test_subagent.py`.

## The shape

The parent gets ONE new tool, `spawn_subagent`. Calling it builds a
FRESH child `Agent` — empty history, generated system prompt, filtered
tool catalog — runs it to completion synchronously, and returns ONLY
the child's final answer (+ cost metadata) as the tool-result string.

```
parent turn ──> model wants spawn_subagent(objective=…)
                     │
                     ▼
              SubagentSpawner.spawn()
              ├── validate (objective / format / justification / tools_allowed)
              ├── budget check + increment
              ├── ToolRegistry(only tools_allowed)      <-- scope = CATALOG
              ├── Agent(provider, system=CHILD_PROMPT, permissions=PARENT'S)
              └── child.run(objective)   # fresh history, silent streaming
                     │
                     ▼
        ToolResult("…final answer…\n[sub-agent · 5 iteration(s) · …]")
```

Why this is cheap here: the loop was already a library class with
injectable everything. A sub-agent is ~30 lines of composition, not a
new runtime. That is the dividend of "the harness IS the Agent class".

## Three constraints, enforced in code

1. **One level deep.** `tools_allowed` containing `spawn_subagent` is
   rejected outright. Nested delegation compounds failure rates —
   three 85%-reliable agents in series ≈ 61% end-to-end. Same choice
   Claude Code makes.
2. **Bounded.** Per-session budget (default 5) checked BEFORE the run,
   incremented when ATTEMPTED (failed spawns count — otherwise models
   probe for free retries), plus a mandatory non-empty `justification`.
   Spawning feels like doing work; the friction makes over-delegation
   visible to whoever reads the transcript.
3. **Compact results.** The parent's context inflates by what the child
   CONCLUDED (`summary + [sub-agent · N iteration(s) · M call(s) · Xin/Mout]`),
   never by its transcript. A 40-iteration research run arrives as one
   result block.

## Fresh context is the feature, not a limitation

Nothing from the parent conversation crosses into the child except the
objective string you typed into the tool call. Anthropic's multi-agent
research finding: independent context windows are most of the VALUE of
the pattern — each worker thinks at full signal-to-noise instead of
wading through the coordinator's whole history.

Corollary the schema makes explicit: `objective` is operationally
load-bearing. The child cannot ask the parent anything; a vague
objective is indistinguishable from no objective. `output_format` is
required for the same reason (vague requests produce rambling).

## Scope restriction lives in the catalog, not the prompt

The child PHYSICALLY has no other tools registered — filtering happens
at `ToolRegistry` construction. "Please only use X" in a system prompt
is a wish; an empty registry is a fact.

Permissions are inherited from the parent unchanged: *a sub-agent
cannot escalate privilege by being a sub-agent*. Under `allow_read_only`
a spawn whose children would write still prompts the human (spawn is
marked `read_only=False` — conservative because children can do
whatever their tools can do). Our tests tripped over exactly this on
day one: under `allow_read_only`, every spawn was denied and the tests
silently exercised denial-as-data instead of spawning. The gate worked;
the test premise didn't.

## Failure modes worth their weight

* **Narrate-instead-of-execute.** Small/local models may DESCRIBE what
  they would do and return confident fabricated numbers. Signature:
  one iteration, zero tool calls. Guard: the child system prompt's
  mandatory-execute clause ("describing a call in prose without
  invoking it is a failure"); evals later assert the count stays zero.
* **Provider errors inside the child** (rate limit, dead upstream):
  caught and returned as DATA to the parent — which can decide to
  retry later or continue without — instead of crashing the parent's
  turn. Errors-as-data holds at every layer of the stack.
* **Iteration cap without an answer**: `child.run()` raises; we salvage
  the last assistant prose and mark the result `[INCOMPLETE -- …]`
  rather than returning nothing or lying.

## Verified live

A coordinator asked to delegate "find SubagentSpawner" produced a
read-only child that researched the question across several real
iterations and returned a one-sentence report relayed verbatim by the
parent — with the cost metadata attached to the spawn automatically.

## The stream tee (`--subagents`)

Originally shelved as a nice-to-have, then built once the CLI grew a
flag to turn sub-agents on at all — silent children turned out to feel
like a hang on long research spawns.

Design in one sentence: `SubagentSpawner` takes an optional
`on_child_event(spawn_number, StreamEvent)` callback; `spawn()` sets the
child's existing `on_stream_event` hook to forward into it, and the
CLI's `SubagentTee` routes pairs to per-child views that draw nested
blocks (`┌ child 1 · model … └ child 1 · end_turn · X in / Y out`).

Three choices worth remembering:

* **No new event vocabulary.** The `StreamEvent` union is untouched —
  the callback's second parameter is typed `Any` on purpose. Child
  events are a UI concern with different provenance, not new loop
  semantics; only StreamEvents flow (ToolExecuted/TurnEnd belong to the
  generator the child consumes internally).
* **Framing derives from the events themselves.** StartEvent opens a
  block, EndEvent closes it — one block PER MODEL CALL. A multi-iteration
  child shows several blocks; that is the truthful unit, not one big box
  pretending the run was atomic.
* **Per-child rendering state.** Each `ChildStreamView` owns its own
  thinking-block tracking instead of sharing the parent Renderer's —
  two agents mutating one renderer's state is the bug that only
  appears mid-turn.

Teeing changes nothing about the contract the tests already pinned:
default stays silent, the parent's context still inflates by the report
only, and a spawn with an observer attached produces byte-identical
history.

## Deliberately not built

Handoffs (permanent control transfer — hard to observe/reason about,
and simulable with agent-as-tool); parallel fan-out of siblings (our
parallel machinery would allow it — spawn inside a concurrent batch —
but sequential-first keeps gates sane; revisit with ch17).
