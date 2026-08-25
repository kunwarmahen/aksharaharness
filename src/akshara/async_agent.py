"""The agent loop, event-loop edition -- the async twin of ``agent.py``.

    user message -> model -> [tool calls? -> execute -> results] -> ... -> final answer

Why this file exists (and why it is a TWIN, not a rewrite): the sync
loop is sequential at the model boundary no matter what color its
functions are, so converting it buys nothing for ONE conversation. What
it buys is MANY conversations: one event loop driving K independent
AsyncAgents concurrently, each holding its own history -- the shape
batch evals and multi-session servers actually want.

The rules are byte-identical to the sync loop; only the plumbing is
awaited:

* ERRORS ARE DATA. Tool failures become ``is_error`` results the model
  reads and recovers from; the loop cannot be crashed by a tool.
* THE HISTORY INVARIANT: every tool_call id gets a matching result
  before the next request -- on EVERY exit path, including iteration
  caps and task cancellation mid-turn or mid-batch.
* GATES SEQUENTIAL, EXECUTION PARALLEL. Permission gates may prompt y/n
  on the one shared terminal, so they run one at a time; approved calls
  then fan out via ``gather`` and yield in submission order -- capped at
  ``max_parallel_tools`` RUNNING tasks. The semaphore lives INSIDE each
  worker coroutine: every call still gets a task immediately (gather
  starts them all), only simultaneous EXECUTION is bounded. Tasks don't
  pay thread cost, but a 10-wide bash batch saturates the machine just
  the same -- leases prevent races, the width cap prevents saturation.
* CANCELLATION IS NOT LOSS. Cancelling mid-batch mirrors the sync
  Ctrl-C story exactly: ``to_thread`` workers cannot be interrupted, so
  the batch is joined, REAL results recorded ("the work happened"), and
  the exception propagates with history left resumable.

Deliberate duplication, documented: the loop body visibly repeats
``Agent.run_streaming``'s shape instead of unifying both behind a clever
driver. Sync and async control flow cannot share a generator's spine,
and a dunder-protocol abstraction would hide the two things this
project exists to teach -- where the await points ARE, and how the
invariant survives cancellation in each world. The leaf logic (event
folding, compaction arithmetic, truncation) IS shared.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from akshara.agent import (
    INTERRUPTED_MESSAGE,
    MAX_PARALLEL_TOOLS,
    AgentEvent,
    ToolExecuted,
    TurnEnd,
    _batch_message,
    _truncate_middle,
)
from akshara.context import RED, SUMMARY_PROMPT, acompact_history, estimate_history
from akshara.errors import ToolError
from akshara.permissions import PermissionFn, PermissionRequest, allow_read_only
from akshara.providers.base import Provider, acollect
from akshara.tools.base import ToolContext, ToolOutput, ToolRegistry
from akshara.tools.selector import ToolCatalog, query_from_transcript
from akshara.types import (
    Block,
    ImageBlock,
    Message,
    ModelResponse,
    StreamEvent,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)


class AsyncAgent:
    """A conversation with tools, driven by an event loop.

    Same constructor surface as :class:`akshara.agent.Agent`; the provider
    must implement the async surface (all built-in adapters do). One
    AsyncAgent == one session's history; drive several from one loop for
    real concurrency.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        system: str | None = None,
        tools: ToolRegistry | None = None,
        cwd: Path | None = None,
        max_tokens: int = 16384,
        max_iterations: int = 25,
        permissions: PermissionFn | None = None,
        on_stream_event: Callable[[StreamEvent], None] | None = None,
        on_before_tool: Callable[[ToolCall], None] | None = None,
        on_after_tool: Callable[[ToolCall, ToolResult], None] | None = None,
        context_window: int = 200_000,
        auto_compact: bool = True,
        tool_catalog: ToolCatalog | None = None,
        tools_per_turn: int = 7,
        max_parallel_tools: int = MAX_PARALLEL_TOOLS,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system = system
        self.registry = tools if tools is not None else ToolRegistry()
        # Dynamic tool loading -- identical contract to the sync twin.
        self.tool_catalog = tool_catalog
        self.tools_per_turn = tools_per_turn
        # Concurrent-execution ceiling for one batch (sync twin bounds its
        # thread pool the same way).
        self.max_parallel_tools = max_parallel_tools
        self._turn_tools: list | None = None
        self.ctx = ToolContext(cwd=(cwd or Path.cwd()).resolve())
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.auto_compact = auto_compact
        self.permissions = permissions or allow_read_only
        # Push-based stream tap -- identical contract to the sync Agent's.
        self.on_stream_event = on_stream_event or (lambda event: None)
        # Observational tool hooks -- identical contract to the sync
        # Agent's (gates decide, hooks watch; a raising hook crashes the
        # turn loudly). Hooks stay SYNC callables fired inline on the
        # loop thread; they observe executions, they never veto them.
        self.on_before_tool = on_before_tool or (lambda call: None)
        self.on_after_tool = on_after_tool or (lambda call, result: None)
        self.history: list[Message] = []
        self.total_usage = Usage()
        # Same totals, bucketed per model slug -- session cost is a
        # per-model sum because a /model switch mid-session changes the
        # price sheet mid-stream (see akshara.pricing).
        self.usage_by_model: dict[str, Usage] = {}
        # Provider-reported size of the last request (all four usage
        # counters summed -- see Usage.window_tokens).
        self.last_context_tokens: int = 0
        self.last_compaction: dict | None = None

    # ---- public entry points ----------------------------------------------

    async def run(self, user_input: str,
                  *, images: list[ImageBlock] | None = None) -> ModelResponse:
        """Library convenience: run a turn to completion, return the reply.

        ``images`` are appended after the text block in one user message
        -- identical contract to the sync twin.
        """
        async for event in self.run_streaming(user_input, images=images):
            if isinstance(event, TurnEnd):
                if event.response is not None:
                    return event.response
                raise RuntimeError(
                    f"turn ended without a response ({event.reason} after "
                    f"{event.iterations} iterations)"
                )
        raise RuntimeError("unreachable")

    async def run_streaming(self, user_input: str,
                            *, images: list[ImageBlock] | None = None,
                            ) -> AsyncIterator[AgentEvent]:
        """Run one turn, yielding stream events + tool results + TurnEnd.

        ``images`` ride in the same user message as the text, exactly as
        in the sync twin.

        Cancel-safe like the sync twin: cancelling the consuming task
        (or closing this async generator) leaves history RESUMABLE --
        outstanding tool calls get synthesized error results first.
        """
        blocks: list[Block] = [TextBlock(user_input)]
        if images:
            blocks.extend(images)
        self.history.append(Message("user", blocks))
        executed: dict[str, ToolResult] = {}  # current batch's completed results
        try:
            for iteration in range(1, self.max_iterations + 1):
                self._begin_iteration()
                response = await acollect(
                    self._atee(
                        self.provider.astream(
                            messages=await self._before_model_call(self.history),
                            system=self.system,
                            tools=self._specs_for_request(),
                            model=self.model,
                            max_tokens=self.max_tokens,
                        )
                    )
                )
                self.history.append(response.message)
                self.total_usage.add(response.usage)
                bucket = self.usage_by_model.setdefault(
                    response.model or self.model, Usage())
                bucket.add(response.usage)
                # Window footprint, not just fresh input: a cached request
                # bills ~nothing fresh yet still fills the window.
                self.last_context_tokens = response.usage.window_tokens()

                calls = response.message.tool_calls()
                if response.stop_reason != "tool_use" or not calls:
                    yield TurnEnd(response=response, reason="end_turn",
                                  iterations=iteration)
                    return

                executed.clear()
                results = await self._execute_batch(calls)
                batch: list[ToolResult] = []
                # Record EVERYTHING before yielding anything: a consumer
                # abandoning the turn after the first panel must not lose
                # its batch-mates' real (already computed) results.
                for call, result in zip(calls, results):
                    batch.append(result)
                    executed[call.id] = result
                for call, result in zip(calls, results):
                    yield ToolExecuted(call=call, result=result)
                self.history.append(_batch_message(batch))

            # Ran out of iterations while the model still wanted tools.
            self._answer_outstanding({})
            yield TurnEnd(response=None, reason="max_iterations",
                          iterations=self.max_iterations)
        except BaseException:
            # CancelledError lands here exactly where KeyboardInterrupt
            # does in the sync loop: leave history resumable, re-raise.
            self._answer_outstanding(executed)
            raise

    # ---- context management ------------------------------------------------

    def _begin_iteration(self) -> None:
        """Re-pick this model call's visible tool set -- deliberate twin of
        ``Agent._begin_iteration`` (see there for the reasoning)."""
        if self.tool_catalog is None:
            self._turn_tools = None
            return
        self._turn_tools = self.tool_catalog.select(
            query_from_transcript(self.history), k=self.tools_per_turn)

    def _specs_for_request(self) -> list:
        if self._turn_tools is None:
            return self.registry.specs()
        return [t.spec() for t in self._turn_tools]

    def _get_visible_tool(self, name: str):
        """registry.get under selection -- with SOFT ADMISSION, the
        deliberate twin of ``Agent._get_visible_tool`` (see there for
        why calling an existing tool by exact name loads it on the spot
        instead of erroring)."""
        if self._turn_tools is not None:
            for tool in self._turn_tools:
                if tool.name == name:
                    return tool
            tool = self.tool_catalog.get(name)
            if tool is not None:
                self._turn_tools.append(tool)
                return tool
            raise KeyError(f"no such tool: {name!r}") from None
        return self.registry.get(name)

    async def _before_model_call(self, history: list[Message]) -> list[Message]:
        """The compaction seam: fires automatically in the red zone.

        Utilization arithmetic is cheap and stays sync; only actual LLM
        summarization suspends."""
        if self.auto_compact:
            ratio = self.utilization()
            if ratio is not None and ratio >= RED:
                await self.compact()
        return history

    def utilization(self) -> float | None:
        """Fraction of the usable window consumed by the next request.
        Prefers the provider's own last reported footprint; falls back to
        the chars/4 heuristic before any response has arrived."""
        usable = max(self.context_window - self.max_tokens, 1)
        used = self.last_context_tokens or estimate_history(self.history)
        if not self.history:
            return None
        return min(used / usable, 1.0)

    async def compact(self) -> dict:
        """Force two-layer compaction now. Returns stats for the UI."""
        before = len(self.history)
        self.history[:], stats = await acompact_history(
            self.history,
            asummarize=self._asummarize_segment,
            context_window=self.context_window,
            max_tokens=self.max_tokens,
        )
        stats["messages_before"] = before
        stats["messages_after"] = len(self.history)
        self.last_compaction = stats
        return stats

    async def _asummarize_segment(self, rendered: str) -> str:
        """The lossy layer: one plain completion through the same provider."""
        response = await self.provider.acomplete(
            messages=[Message("user", [TextBlock(SUMMARY_PROMPT + rendered)])],
            system=None,
            tools=[],
            model=self.model,
            max_tokens=2000,
        )
        return response.message.text().strip()

    async def _atee(self, events) -> AsyncIterator[StreamEvent]:
        """Forward each StreamEvent to on_stream_event while acollect drains.

        Push, not yield -- same reasoning as the sync _tee: collect() owns
        the pull, so a live renderer subscribes via the callback.
        """
        async for event in events:
            self.on_stream_event(event)
            yield event

    # ---- execution ---------------------------------------------------------

    async def _execute_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run a whole batch; returned list aligns with ``calls`` positionally.

        Gates SEQUENTIAL (they may own the terminal), execution CONCURRENT
        via gather, results in SUBMISSION order -- the exact contract of
        the sync ThreadPoolExecutor path, minus the pool.
        """
        results: list[ToolResult | None] = [self._gate(c) for c in calls]
        pending = [(i, c) for i, c in enumerate(calls) if results[i] is None]

        if len(pending) <= 1:  # fast path: nothing to parallelize
            for i, call in pending:
                results[i] = await self._run_one(call)
            return results  # type: ignore[return-value]

        # Width cap: the semaphore is acquired INSIDE the worker, so every
        # call becomes a task at once but at most max_parallel_tools run
        # simultaneously. Submission-order harvest below is untouched.
        sem = asyncio.Semaphore(self.max_parallel_tools)

        async def _bounded(call: ToolCall) -> ToolResult:
            async with sem:
                return await self._run_one(call)

        tasks = {i: asyncio.ensure_future(_bounded(call))
                 for i, call in pending}
        inner = asyncio.gather(*tasks.values())
        cancelled: BaseException | None = None
        try:
            # SHIELDED: a bare `await gather(...)` would forward the cancel
            # into every child task -- each would die of CancelledError at
            # its await point EVEN THOUGH its thread finishes the work.
            # Shield keeps the join alive so the harvest below can collect
            # the real outcomes, exactly like the sync pool's join-on-Ctrl-C.
            await asyncio.shield(inner)
        except BaseException as exc:
            cancelled = exc

        if cancelled is None:
            for i, task in tasks.items():
                results[i] = task.result()
            return results  # type: ignore[return-value]

        # Threads cannot be interrupted: every to_thread worker keeps
        # running, so join them all and record the REAL outcomes --
        # "lost during cancellation" only for workers that themselves
        # failed. The work HAPPENED; history says so.
        calls_by_index = dict(pending)
        outcomes = await inner
        for i, outcome in zip(tasks, outcomes):
            if isinstance(outcome, BaseException):
                results[i] = ToolResult(
                    calls_by_index[i].id,
                    f"lost during cancellation: "
                    f"{type(outcome).__name__}: {outcome}",
                    is_error=True,
                )
            else:
                results[i] = outcome
        self.history.append(
            _batch_message([r for r in results if r is not None]))
        raise cancelled

    def _gate(self, call: ToolCall) -> ToolResult | None:
        """Permission stage. Returns an error ToolResult to block the call,
        or None when approved. Never runs the tool."""
        try:
            tool = self._get_visible_tool(call.name)
        except KeyError as exc:
            return ToolResult(call.id, str(exc), is_error=True)

        # Build the human-facing summary defensively: a broken summary()
        # must never block the approval flow itself.
        try:
            summary = tool.summary(call.arguments, self.ctx)
        except Exception:
            summary = f"{call.name}({json.dumps(call.arguments)})"

        request = PermissionRequest(
            tool_name=call.name,
            arguments=call.arguments,
            summary=summary,
            read_only=tool.read_only,
            # Closed over so an edit-and-reapprove UI can re-render the
            # preview for amended args (approve-with-edits).
            summarize=lambda args: tool.summary(args, self.ctx),
        )
        try:
            allowed = self.permissions(request)
        except Exception as exc:
            return ToolResult(call.id, f"permission gate failed: {exc}", is_error=True)
        if not allowed:
            return ToolResult(call.id, "Permission denied by user.", is_error=True)
        if request.arguments is not call.arguments:
            # The gate EDITED the arguments before approving (identity
            # check: same dict means untouched). Adoption happens here --
            # execution, hooks, and history all see the approved form.
            call.arguments = request.arguments
        return None

    async def _run_one(self, call: ToolCall) -> ToolResult:
        """Execution stage: run the tool, wrap every failure as data.

        Hooks bracket the real execution exactly as in the sync twin --
        sync callables fired inline (they observe; they don't suspend and
        they never veto). A raising hook propagates loudly on purpose.
        """
        self.on_before_tool(call)
        tool = self._get_visible_tool(call.name)
        try:
            output = await tool.arun(call.arguments, self.ctx)
        except ToolError as exc:
            result = ToolResult(call.id, str(exc), is_error=True)
        except asyncio.CancelledError:
            raise  # cancellation is not a tool failure
        except Exception as exc:
            result = ToolResult(call.id, f"{type(exc).__name__}: {exc}",
                                is_error=True)
        else:
            if isinstance(output, ToolOutput):
                # Rich result: text truncates like any other; images ride
                # the result for _batch_message to hoist into history.
                result = ToolResult(call.id, _truncate_middle(output.text),
                                    images=output.images)
            else:
                result = ToolResult(call.id, _truncate_middle(output))
        self.on_after_tool(call, result)
        return result

    # ---- the invariant -----------------------------------------------------

    def _answer_outstanding(self, executed: dict[str, ToolResult]) -> None:
        """Guarantee every tool_call in the trailing assistant message has
        a matching ToolResult in history. Called on abnormal exits only.

        Without this, the NEXT request references a tool_use id that never
        got a result -- and providers reject it with a 400. This helper is
        why cancelling a turn here never corrupts the session."""
        if not self.history or self.history[-1].role != "assistant":
            return
        blocks = []
        images = []
        for call in self.history[-1].tool_calls():
            if call.id in executed:
                # Real result already produced (and shown to the user),
                # just never appended -- replay it faithfully.
                done = executed[call.id]
                blocks.append(ToolResult(call.id, done.content, done.is_error))
                images.extend(done.images)
            else:
                blocks.append(ToolResult(call.id, INTERRUPTED_MESSAGE, is_error=True))
        if blocks:
            self.history.append(Message("user", [*blocks, *images]))
