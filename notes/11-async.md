# 11 · Async: one event loop, many conversations

> The final phase converts the harness to run natively under asyncio —
> not by rewriting it, but by *splitting every module along its single
> await point*. Files: `providers/sse.py`, `providers/retry.py`,
> `providers/base.py`, `providers/{anthropic,openai}.py` (+ `fallback.py`),
> `context.py`, `tools/base.py`, `async_agent.py`; tests
> `tests/test_async_providers.py`, `tests/test_async_agent.py`;
> demo `examples/async_demo.py`.

## The decision, and what it bought

The sync loop is sequential at the model boundary no matter what color
its functions are: one conversation cannot overlap with itself. So a
full conversion buys nothing for ONE conversation — what it buys is
MANY conversations: **one event loop driving K independent AsyncAgents
concurrently**, each holding its own history. That is the shape batch
evals and multi-session servers actually want, and it is what the live
demo measures.

What deliberately stayed sync: the CLI, REPL, renderer, MCP client,
and eval runner. They are interactive or subprocess-shaped; converting
them adds color without adding capability. The async surface is a
library surface — importable, composable, testable without a terminal.

## The pattern: protocol cores, colored skins

The whole phase rests on one structural idea. Every wire rule already
lived in *incremental* functions — feed a chunk, get events out. Those
become stateful classes whose methods are plain sync calls:

* `SSELineSplitter` / `SSEEventGrouper` — framing as feed()/flush()
* `AnthropicStreamRouter` / `OpenAIStreamRouter` — event decoding as
  feed(name, data), raising ProviderError on error events
* `ResponseFolder` — folding events into a ModelResponse as feed(event)

Sync and async are then THIN SKINS over the same core:

```python
def collect(...):                 # sync skin
    folder = ResponseFolder()
    for event in parse_events(iter_sse_lines(chunks)):
        folder.feed(event)
    return folder.finish()

async def acollect(...):          # async twin -- same two rules
    folder = ResponseFolder()
    async for event in aparse_events(aiter_sse_lines(achunks)):
        folder.feed(event)
    return folder.finish()
```

Zero duplicated rules: chunk boundaries, CRLF handling, UTF-8 across
splits, argument-fragment accumulation, usage extraction — written
once, exercised by both worlds. The parity tests prove it directly:
the async SSE suite replays the same fixtures at chunk sizes 1, 3, and
4096 and asserts byte-exact equality with the sync output.

Two gotchas worth remembering:

* `yield from` is ILLEGAL inside async generators (`SyntaxError`) —
  flatten to explicit `for ... : yield` loops.
* Error classification needs the BODY before it can decide, so the
  async retry helper's classify callback is awaited:
  `aconnect_with_retries(send, classify=<async>)` — the handler must
  `await response.aread()` before matching status/body. The budget
  arithmetic itself is identical to sync, down to `_asleep =
  asyncio.sleep` being a swappable module global the tests intercept
  (with an ASYNC interceptor — a plain list.append explodes with
  "object NoneType can't be used in 'await' expression").

MockTransport serves both dialects too: `Provider.__init__` hands the
same transport to `AsyncClient` when it is an `httpx.MockTransport`,
so fixture-driven tests cover both colors with zero new fixtures.

## Tools: honest concurrency at both layers

`Tool.arun()` has one default implementation:

```python
async def arun(self, args, ctx) -> str:
    return await asyncio.to_thread(self.run, args, ctx)
```

That is deliberate, not a stopgap. Our tools do blocking syscall IO
(open/subprocess/os.replace); declaring blocking syscalls `async`
without a true async backend would block the event loop while lying
about it. One blocking implementation per tool; the default pushes it
to a worker thread. A tool with a genuinely non-blocking backend
overrides `arun` (see `SlowAsyncTool` in the tests).

## The async loop: a visible TWIN, not a clever unification

`async_agent.py` visibly repeats `agent.py`'s shape instead of hiding
both behind a dunder-protocol driver. Reason: sync and async control
flow cannot share a generator's spine, and the duplication IS the
lesson — you can see where the await points are and how the invariant
survives cancellation in each world. Leaf logic (event folding,
compaction arithmetic, truncation, gating) is shared verbatim.

Rules carried over unchanged:

* **Errors are data** — tool failures become `is_error` results.
* **History invariant on every exit path** — including iteration caps,
  generator close (`aclose()` unwinds like generator `.close()`), and
  task cancellation mid-turn or mid-batch.
* **Gates sequential, execution parallel** — permission prompts may own
  the terminal, so they run one at a time; approved calls fan out via
  tasks and results return in SUBMISSION order regardless of completion
  order.
* **Worker ceiling, chosen not inherited** — the sync loop's
  `MAX_PARALLEL_TOOLS` bounded thread *cost*; `asyncio` tasks don't pay
  it, so an equivalent cap has to exist by choice: `max_parallel_tools`
  (default 8) caps EXECUTION, not task creation — see "the width cap"
  at the bottom of this note.

Compaction crosses the boundary through one seam: masking, red-zone
arithmetic, and cut-point validation are pure and shared; only
`summary_text = await asummarize(render_segment(segment))` differs from
sync. `Agent._before_model_call` became async for exactly this — the
utilization check stays sync, actual summarization suspends.

## Cancellation: the lesson that cost a test failure

Threads cannot be interrupted. When Ctrl-C lands mid-batch in the sync
loop, the ThreadPoolExecutor join waits for running workers and records
their REAL results ("the work happened"). The async twin must reproduce
that — and the naive translation fails in an instructive way:

```python
try:
    await asyncio.gather(*tasks.values())     # WRONG
except BaseException:
    ...harvest...
```

Cancelling the outer task makes `gather` **forward cancellation into
every child** — each worker task dies of CancelledError at its await
point EVEN THOUGH its `to_thread` thread finishes the work. The harvest
then collects CancelledError from all of them and records
"lost during cancellation" — precisely the loss the design forbids.

The fix is one call:

```python
inner = asyncio.gather(*tasks.values())
try:
    await asyncio.shield(inner)   # cancel lands HERE, children unaffected
except BaseException as exc:
    cancelled = exc
outcomes = await inner            # join: real outcomes, threads finished
```

`shield` decouples the JOIN from the JOINED work: the caller gets their
CancelledError immediately while the batch keeps running to completion;
the second await joins it and records real results. Regression-tested
by cancelling mid-batch and asserting both real outputs land in history
with no error flags.

Same family, different mechanism: closing the async generator early
(`aclose()`) raises GeneratorExit at the current yield — outstanding
calls get synthesized answers via the same `_answer_outstanding` path
the sync twin uses, replaying REAL results where they exist and
`(interrupted)` placeholders where they don't.

## What the live run showed

`examples/async_demo.py --conversations 4` measures it directly:
first tokens arriving interleaved from ~zero seconds in, completion
order diverging from submission order, the speedup printed live. The
honest caveat it prints alongside: this is wall-clock, not token
throughput — the provider serves roughly the same total tokens either
way; concurrency hides each request's latency behind other sessions'
waits instead of making any request faster.

## Honest limitations

* No async REPL/CLI: interactive input() is inherently blocking; a true
  async terminal would need a different input architecture entirely.
* MCP sessions remain sync (subprocess + reader thread). An asyncio
  MCP transport would be a new session class, not a conversion.
* The async loop trusts `provider.astream`; there is no automatic
  fallback from an async-incapable provider object (all built-ins
  implement both).
* `to_thread` uses the interpreter's default pool; thousands of truly
  simultaneous blocking tools would queue there (as they would in any
  thread-based design).

## The width cap: `max_parallel_tools`

Everything above was about WIDTH being free. One honest correction: it
is free for the *event loop*, but a 10-call bash-heavy batch running
10-wide can still saturate the machine (or trip a provider-side
concurrency limit on API-shaped tools). Leases prevent races, not
saturation.

`AsyncAgent(max_parallel_tools=8)` now wraps execution in an
`asyncio.Semaphore` acquired INSIDE each worker coroutine:

```python
sem = asyncio.Semaphore(self.max_parallel_tools)
async def _bounded(call):
    async with sem:
        return await self._run_one(call)
tasks = {i: asyncio.ensure_future(_bounded(c)) for i, c in pending}
```

All tasks are still created at once (gather semantics preserved), but
at most N execute concurrently; the submission-order harvest is
untouched because ordering was always decided by the result dict, never
by completion time. The cap counts EXECUTION, not queueing — a queued
call hasn't started its tool, so leases aren't held while waiting.
Pinned by a test that tracks peak concurrency across a 4-call batch
under a cap of 2.
