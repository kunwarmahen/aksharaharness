# 14 · Hooks: watching without touching

> Files: [agent.py](../src/akshara/agent.py) +
> [async_agent.py](../src/akshara/async_agent.py) (`on_before_tool`,
> `on_after_tool`), [`examples/hooks_demo.py`](../examples/hooks_demo.py),
> `tests/test_hooks.py`.

## The problem they solve

Sooner or later every harness author asks: *how do I log time every
tool call? Add an audit trail? Emit metrics?* The tempting answer is
to edit the loop, or wrap tools in proxies. Both scale badly — the
loop is the one file you promised not to churn, and proxies must be
re-applied everywhere a registry gets copied (sub-agent catalogs,
eval runners…).

Hooks are the boring answer: two optional callbacks on the agent,
fired by the loop itself.

```python
agent = Agent(...,
    on_before_tool=lambda call: log.info("start %s", call.name),
    on_after_tool=lambda call, r: metrics.record(call.name, r.is_error))
```

`before` fires as an approved call's execution starts; `after` fires
with the wrapped `ToolResult` — errors included, because an error
result is still data by then. They mirror `on_stream_event`: optional
push callbacks, no new types to learn.

## Gates decide, hooks watch

The doctrine that keeps this from turning into "middleware":

* **A hook cannot veto anything.** There is no return value checked,
  no way to say no. Denial belongs to the permission gate — one
  concept, one job. If you find yourself wanting to block from a
  hook, what you actually want is a gate.
* **Denied calls and unknown tools are never observed.** Hooks
  bracket real EXECUTIONS; nothing ran, nothing to watch. (Tests pin
  this — it is the visible consequence of the doctrine.)
* **A raising hook crashes the turn. Loudly.** This is the exact
  opposite of tools-as-data, and deliberate: tools are UNTRUSTED
  input whose failures the model must read and recover from; hooks
  are YOUR infrastructure (logging, audit). A broken audit trail must
  not fail silently into a turn that LOOKS fine. Fail fast, fix the
  hook. Even then the crash path honors the history invariant —
  outstanding calls get synthesized results, so the session survives
  (regression-tested on both twins).

## Where they fire

Inside `_run_one`, bracketing `tool.run()`:

```
gate approves ──► before(call) ──► tool.run() ──► wrap result ──► after(call, result)
   (decides)        WATCH            the work       errors=data      WATCH
```

In batches, sync workers fire hooks from pool threads (hook authors
own their thread-safety — same deal as any logging call); the async
twin fires them inline on the loop thread. Same contract either way,
which is the point of twins: `tests/test_hooks.py` asserts identical
bracket order and payloads through `Agent` and `AsyncAgent`.

## Relation to the other observers

The harness now has three ways to see a turn, at three grains:

| Channel | Grain | Delivery |
|---|---|---|
| `on_stream_event` | tokens/thinking deltas | pushed during each response |
| `ToolExecuted` events | finished tool calls | pulled from `run_streaming()` |
| `on_before_tool`/`on_after_tool` | execution start + outcome | pushed from inside `_run_one` |

The eval runner keeps its own recording proxies — those exist to
SCORE trajectories and must survive registry re-wrapping; hooks exist
to OBSERVE them. Different jobs, different mechanisms.

## Live receipt

`examples/hooks_demo.py` against a real provider: one read-only
question, hooks printed

```
  ▶ grep {"path": "README.md", "pattern": "^# "}
  ◀ grep -> ok, 297 chars
  ▶ read_file {"path": "README.md"}
  ◀ read_file -> ok, 20030 chars
```

Two executions, four observations, zero loop edits.
