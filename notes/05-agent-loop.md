# 05 · The agent loop, cancellation, and the history invariant

> Files: `akshara/agent.py`, `akshara/permissions.py`.
> This is the note `errors.py` points at from `ContextOverflowError`.

## The loop

```
append user Message
for iteration in 1..max_iterations:
    response = collect(provider.stream(history, system, tools, model, max_tokens))
    history.append(response.message); total_usage += response.usage
    if stop_reason != "tool_use": yield TurnEnd(response, "end_turn"); return
    for call in calls: result = _execute(call); yield ToolExecuted(call, result)
    history.append(Message("user", [results...]))       # results as BLOCKS
yield TurnEnd(None, "max_iterations")
```

Everything the model ever sees comes back through this shape — which is
why `/provider` switching mid-session is free: history is stored in
internal types, never wire format.

## Errors are data — four funnels, one choke point

`_execute()` **never raises**. Every failure becomes an `is_error`
ToolResult the model reads and can plan around:

| Failure | Result content |
|---|---|
| unknown tool | `no such tool: 'nope'` |
| permission denied | `Permission denied by user.` |
| tool raised ToolError | its message (phrased for the model) |
| tool crashed otherwise | `ValueError: boom` (type + message) |
| gate itself crashed | `permission gate failed: ...` |

The loop physically cannot be crashed by a tool. KeyboardInterrupt is the
one deliberate exception — cancellation is not a tool failure, so it
re-raises.

## THE INVARIANT (the #1 source of 400s in hand-rolled harnesses)

> Every tool_call id that appears in an assistant message must have a
> matching ToolResult in a LATER user message before the next request.

Anthropic rejects violations with a 400 ("tool_use ids without
tool_result blocks"); OpenAI-style APIs misbehave more quietly. The
dangerous paths are all *abnormal exits*:

* **iteration cap** — subtler than expected: at the cap the LAST batch was
  already appended, so there are usually no outstanding ids. `_answer_outstanding({})`
  after the loop is defensive; it no-ops when trailing message is `user`.
* **Ctrl-C / generator.close()** — the real case. See below.
* **denied/timed-out calls** — handled inline by making denial a result.

`_answer_outstanding(executed)` closes any gap on every exit path:
replay already-executed results faithfully (preserving their original
`is_error`!), synthesize `INTERRUPTED_MESSAGE` errors for never-run calls.

## Generator cancellation mechanics

The whole program is sync generators, which makes Ctrl-C almost free:

1. REPL catches `KeyboardInterrupt` (or the consumer closes the generator)
2. `stream.close()` → `GeneratorExit` thrown at the suspension point
3. `except BaseException:` in `run_streaming` runs the invariant cleanup,
   then **re-raises** (cleanup must not swallow cancellation)
4. closing also unwinds into `provider.stream()`'s `finally` → socket freed

One surprise worth remembering: if the interrupt lands DURING the model
pull, `collect()` never returns — so the assistant message never lands in
history and there is nothing outstanding. History = just the user turn.
Trivially resumable. The interesting cases are all "interrupt between
execute and append".

## Permission gates are plain callables

```python
PermissionFn = Callable[[PermissionRequest], bool]
```

No class hierarchy. CLI ships a rich Confirm-prompt factory showing the
tool's own `summary()`; tests pass lambdas; `allow_read_only` /
`yolo` / `deny_all` are three one-line functions. Whatever the gate
answers becomes data (see funnel table) — a denial says "the human said
no" to the model, which can then pick a different approach instead of
the session dying. The prompt has since grown a third answer: **e**
edits the pending arguments before approval, and the loop adopts the
edited form ([notes/20](20-approve-with-edits.md)).

## Testing a loop without a network

`ScriptedProvider` (tests/conftest.py) scripts canned ModelResponses;
its `stream()` re-synthesizes the event vocabulary from each one, so the
loop folds them with the REAL `collect()`. Consequence: every loop test
silently re-verifies collect()'s round-trip fidelity — if it dropped a
field, scripted-vs-rebuilt comparisons would fail.

The invariant gets a dedicated checker (`assert_history_resumable`) run
across happy path, cap, close-mid-batch, and interrupt tests. Encode the
invariant once as code rather than restating per-test expectations.

## Truncation

Tool output > 20k chars → middle-truncated (head+tail). Rationale in
[04-tools.md](04-tools.md): errors live at the end; head-only truncation
hides them.

## Two channels out of a turn (found by the model's answer going missing)

`run_streaming()` yields only ToolExecuted / TurnEnd. Raw StreamEvents
(text/thinking deltas) are consumed INSIDE collect(), and you cannot
yield outward from a pull-based drain -- so for the first live sessions
the CLI showed tool panels and footers but NEVER the model's prose, and
nothing failed. The fix is push vs pull:

```python
agent.on_stream_event = renderer      # PUSHED as each response streams
for event in agent.run_streaming(prompt):   # YIELDED execution events
    ...
```

`Agent._tee` forwards every StreamEvent to the subscriber while collect()
drains it. Rule of thumb: **consumers pull, UIs subscribe.**

## Rendering note

ThinkingDelta renders dim-italic under a lazy `· thinking` header; the
first TextDelta after it closes the line. Argument fragments stay
invisible in the REPL (the result panel shows the parsed dict) but the
loop demo paints them raw -- seeing `'{"pa' '+th": ...'` arrive is
what makes "parse once at the end" click.

## Parallel batches changed the close story

Tool batches now gate sequentially, then execute in a thread pool, and
only THEN yield ToolExecuted events (submission order preserved).
Consequences the invariant tests pin:

* Closing after the first panel can no longer catch siblings
  mid-execution -- every result already exists, so `_answer_outstanding`
  replays REAL results for the whole batch (the old test's synthesized
  interrupt for an un-run sibling became a denial/real result instead).
* Ctrl-C while WAITING on the pool: workers don't receive SIGINT, the
  `with` block joins them as they finish, and the batch appends its REAL
  results before propagating -- faithful beats synthetic when the work
  actually happened.
* Record-into-`executed{}` happens BEFORE the first yield, so a consumer
  that closes at panel one keeps panel two's data resumable.
