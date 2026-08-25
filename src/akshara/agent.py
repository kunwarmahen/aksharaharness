"""The agent loop -- the reason this project exists.

    user message -> model -> [tool calls? -> execute -> results] -> ... -> final answer

The rules that make it robust (see notes/05-agent-loop.md):

* ERRORS ARE DATA. A tool that doesn't exist, bad arguments, a crash, a
  denial -- all become ``is_error`` ToolResults the model reads and can
  recover from. The loop physically cannot be crashed by a tool.
* THE HISTORY INVARIANT: every tool_call id must have a matching result
  in history before the next request. Violating this is the #1 source
  of 400s in hand-rolled harnesses. We maintain it on EVERY exit path,
  including iteration caps and Ctrl-C mid-turn.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from akshara.context import (
    RED,
    SUMMARY_PROMPT,
    compact_history,
    estimate_history,
)
from akshara.errors import ToolError
from akshara.permissions import PermissionFn, PermissionRequest, allow_read_only
from akshara.providers.base import Provider, collect
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

MAX_TOOL_RESULT_CHARS = 20_000
MAX_PARALLEL_TOOLS = 8  # per-batch worker ceiling; batches are usually small
INTERRUPTED_MESSAGE = "[turn interrupted before completion; no result was produced]"


@dataclass(slots=True)
class ToolExecuted:
    """A tool call finished (successfully or not)."""

    call: ToolCall
    result: ToolResult


@dataclass(slots=True)
class TurnEnd:
    """Terminal event of one turn."""

    response: ModelResponse | None  # None when capped/cancelled
    reason: Literal["end_turn", "max_iterations", "cancelled"]
    iterations: int


AgentEvent = StreamEvent | ToolExecuted | TurnEnd


def _truncate_middle(text: str, cap: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Keep both ends of oversized tool output; errors live at the end."""
    if len(text) <= cap:
        return text
    keep = cap // 2
    omitted = len(text) - cap
    return f"{text[:keep]}\n[... {omitted} chars omitted ...]\n{text[-keep:]}"


def _batch_message(batch: list[ToolResult]) -> Message:
    """A finished batch as ONE user message -- the shape the invariant wants.

    Tool results come first (the OpenAI-family wires require them to
    directly follow the assistant's tool calls); any images a tool
    produced are hoisted after them, because those wires cannot carry an
    image inside a role:"tool" payload. Anthropic takes the same block
    order as-is. See ToolResult.images.
    """
    blocks: list[Block] = [*batch]
    for result in batch:
        blocks.extend(result.images)
    return Message("user", blocks)


class Agent:
    """A conversation with tools. One Agent == one session's history."""

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
        interrupt_check: Callable[[], bool] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system = system
        self.registry = tools if tools is not None else ToolRegistry()
        # Dynamic tool loading (book ch12): when a ToolCatalog is set, each
        # ITERATION sends only the catalog's top-K picks for the current
        # conversation (see _begin_iteration). Selection caps what gets
        # SENT, not what can EXECUTE: calling an existing tool by exact
        # name soft-admits it (see _get_visible_tool), so the only
        # model-visible failure left is a genuinely unknown name.
        self.tool_catalog = tool_catalog
        self.tools_per_turn = tools_per_turn
        self.ctx = ToolContext(cwd=(cwd or Path.cwd()).resolve())
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.auto_compact = auto_compact
        self.permissions = permissions or allow_read_only
        # Raw StreamEvents cannot be YIELDED from run_streaming -- they are
        # consumed inside collect(), and you cannot yield outward from a
        # pull-based drain. Instead they are PUSHED here as they arrive,
        # which is exactly what a live renderer wants. UIs set this:
        #     agent.on_stream_event = renderer
        self.on_stream_event = on_stream_event or (lambda event: None)
        # Observational tool hooks (the push-channel analog for execution):
        # before fires per APPROVED call right as its execution starts,
        # after fires with the wrapped result -- success or error, both are
        # data by then. GATES DECIDE, HOOKS WATCH: there is no way to veto
        # from a hook (that is permissions' job), and a raising hook crashes
        # the turn LOUDLY -- hooks are developer infrastructure (logging,
        # metrics, audit), not untrusted input, so their failures must not
        # be laundered into model-visible data like tool failures are.
        # Denied calls and unknown tools never execute -> never observed.
        self.on_before_tool = on_before_tool or (lambda call: None)
        self.on_after_tool = on_after_tool or (lambda call, result: None)
        self.history: list[Message] = []
        self.total_usage = Usage()
        # Same totals, bucketed per model slug -- session cost is a
        # per-model sum because a /model switch mid-session changes the
        # price sheet mid-stream (see akshara.pricing).
        self.usage_by_model: dict[str, Usage] = {}
        # This iteration's visible tool set; None = whole registry.
        self._turn_tools: list | None = None
        # Provider-reported size of the last request (all four usage
        # counters summed -- see Usage.window_tokens) -- the honest context
        # signal (the chars/4 heuristic in context.py is only a proxy).
        self.last_context_tokens: int = 0
        self.last_compaction: dict | None = None
        # Cooperative cancellation for hosts where the loop runs on a worker
        # thread that no signal can reach (the web UI's cancel button). When
        # set, it is polled between stream events -- so a Stop click lands
        # MID-MODEL-CALL, within one SSE event, not just at the checkpoints
        # a terminal Ctrl-C already owns. Raising KeyboardInterrupt keeps the
        # resumable-history cleanup identical to every other exit path.
        self.interrupt_check = interrupt_check

    def _interrupted(self) -> bool:
        """Poll the host's cancel flag, if one is wired."""
        check = self.interrupt_check
        return bool(check and check())

    # ---- public entry points ----------------------------------------------

    def run(self, user_input: str,
            *, images: list[ImageBlock] | None = None) -> ModelResponse:
        """Library convenience: run a turn to completion, return the reply.

        ``images`` are appended after the text block in one user message
        (see akshara.images for loading files into ImageBlocks).
        """
        for event in self.run_streaming(user_input, images=images):
            if isinstance(event, TurnEnd):
                if event.response is not None:
                    return event.response
                raise RuntimeError(
                    f"turn ended without a response ({event.reason} after "
                    f"{event.iterations} iterations)"
                )
        raise RuntimeError("unreachable")

    def run_streaming(self, user_input: str,
                      *, images: list[ImageBlock] | None = None,
                      ) -> Iterator[AgentEvent]:
        """Run one turn, yielding stream events + tool results + TurnEnd.

        ``images`` ride in the same user message as the text (text first,
        then images -- both dialects preserve block order).

        Cancel-safe: closing this generator early (or KeyboardInterrupt
        while pulling) leaves history RESUMABLE -- outstanding tool calls
        get synthesized error results so the next request stays valid.
        """
        blocks: list[Block] = [TextBlock(user_input)]
        if images:
            blocks.extend(images)
        self.history.append(Message("user", blocks))
        executed: dict[str, ToolResult] = {}  # current batch's completed results
        try:
            for iteration in range(1, self.max_iterations + 1):
                self._begin_iteration()
                response = collect(
                    self._tee(
                        self.provider.stream(
                            messages=self._before_model_call(self.history),
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
                results = self._execute_batch(calls)
                batch: list[ToolResult] = []
                # Record EVERYTHING before yielding anything: a consumer
                # closing the generator after the first panel must not
                # lose its batch-mates' real (already computed) results.
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
            # KeyboardInterrupt or generator.close(): leave history resumable.
            self._answer_outstanding(executed)
            raise

    # ---- context management ------------------------------------------------

    def _begin_iteration(self) -> None:
        """Re-pick the visible tool set for this model call.

        With no catalog this is a no-op (full registry, as always). With
        one, the query comes from the transcript -- so the selection
        TRACKS the conversation: naming a missing tool puts its name in
        history, which is exactly what surfaces it next iteration.
        """
        if self.tool_catalog is None:
            self._turn_tools = None
            return
        self._turn_tools = self.tool_catalog.select(
            query_from_transcript(self.history), k=self.tools_per_turn)

    def _specs_for_request(self) -> list:
        if self._turn_tools is None:
            return self.registry.specs()
        # A tool selected earlier this turn can be disabled mid-turn; it
        # must not ride the NEXT request of the same turn.
        return [t.spec() for t in self._turn_tools
                if not self.registry.is_disabled(t.name)]

    def _get_visible_tool(self, name: str):
        """registry.get under selection -- with SOFT ADMISSION.

        The K-cap governs what gets SENT (context economy), not what can
        execute: a model that names an existing tool has already
        discovered it (a prior listing, resumed history, or the name is
        simply right), and punishing exact knowledge with a failed turn
        taught nothing. So an unselected-but-real call is ADMITTED into
        _turn_tools on the spot and runs this same turn -- permission
        gating unchanged, and its name in history means BM25 keeps it
        selected from here on (convergence without the punishment lap).
        Only a genuinely unknown name still errors, as data.

        Operator-disabled tools are refused HERE -- the single choke
        point every path flows through (selected or soft-admitted) -- so
        a mid-turn disable takes effect on the very next call.
        """
        if self.registry.is_disabled(name):
            raise KeyError(f"tool {name!r} is disabled by the operator")
        if self._turn_tools is not None:
            for tool in self._turn_tools:
                if tool.name == name:
                    return tool
            tool = self.tool_catalog.get(name)
            if tool is not None:
                # List.append is atomic under the GIL; batch workers may
                # admit concurrently, but nothing ever removes from here.
                self._turn_tools.append(tool)
                return tool
            raise KeyError(f"no such tool: {name!r}") from None
        return self.registry.get(name)

    def _before_model_call(self, history: list[Message]) -> list[Message]:
        """The compaction seam: fires automatically in the red zone.

        Called with the full history right before every request. When
        utilization crosses RED, compact in place and send the shrunken
        version -- the model never needs to know this happened.
        """
        if self.auto_compact:
            ratio = self.utilization()
            if ratio is not None and ratio >= RED:
                self.compact()
        return history

    def utilization(self) -> float | None:
        """Fraction of the usable window consumed by the next request.
        Prefers the provider's own last reported footprint; falls back to
        the chars/4 heuristic before any response has arrived.

        The reply budget is capped at HALF the window when computing
        headroom: a small local model (8k window) configured with the
        cloud-default 16k reply budget would otherwise make
        ``window - max_tokens`` hit zero and peg this reading at 100%
        forever -- the "pressure is always full" display bug."""
        headroom = min(self.max_tokens, self.context_window // 2)
        usable = max(self.context_window - headroom, 1)
        used = self.last_context_tokens or estimate_history(self.history)
        if not self.history:
            return None
        return min(used / usable, 1.0)

    def compact(self) -> dict:
        """Force two-layer compaction now. Returns stats for the UI."""
        before = len(self.history)
        self.history[:], stats = compact_history(
            self.history,
            summarize=self._summarize_segment,
            context_window=self.context_window,
            max_tokens=self.max_tokens,
        )
        stats["messages_before"] = before
        stats["messages_after"] = len(self.history)
        self.last_compaction = stats
        return stats

    def _summarize_segment(self, rendered: str) -> str:
        """The lossy layer: one plain completion through the same provider
        (no tools). A cheaper model would do; same model keeps it simple."""
        response = self.provider.complete(
            messages=[Message("user", [TextBlock(SUMMARY_PROMPT + rendered)])],
            system=None,
            tools=[],
            model=self.model,
            max_tokens=2000,
        )
        return response.message.text().strip()

    def _tee(self, events: Iterable[StreamEvent]) -> Iterator[StreamEvent]:
        """Forward each StreamEvent to on_stream_event while collect() drains.

        Push, not yield: collect() owns the pull, so the only way a live
        renderer sees text/thinking deltas as they arrive is this callback
        -- yielding them from run_streaming would interleave badly with
        ToolExecuted (they'd all arrive AFTER the whole response was folded).

        This is also the interrupt checkpoint INSIDE a model call: polled
        per event, a host cancel flag aborts the drain mid-stream. Nothing
        has been appended to history yet at that point, so unwinding here
        is trivially resumable (see run_streaming's BaseException handler).
        """
        for event in events:
            if self._interrupted():
                raise KeyboardInterrupt
            self.on_stream_event(event)
            yield event

    # ---- execution --------------------------------------------------------

    def _execute(self, call: ToolCall) -> ToolResult:
        """Gate + run one tool call. NEVER raises (except cancellation)."""
        denied = self._gate(call)
        if denied is not None:
            return denied
        return self._run_one(call)

    def _execute_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run a whole batch; the returned list aligns with ``calls`` by position.

        Gates run SEQUENTIALLY first -- they may prompt y/n on the shared
        terminal and must never interleave. Approved calls then execute
        CONCURRENTLY (everything-parallel policy), but results are yielded
        in submission order regardless of completion order, so panels stay
        deterministic.
        """
        results: list[ToolResult | None] = [self._gate(c) for c in calls]
        pending = [(i, c) for i, c in enumerate(calls) if results[i] is None]

        if len(pending) <= 1:  # fast path: nothing to parallelize
            for i, call in pending:
                if self._interrupted():  # stop BEFORE the next call starts
                    raise KeyboardInterrupt
                results[i] = self._run_one(call)
            return results  # type: ignore[return-value]

        with ThreadPoolExecutor(
            max_workers=min(len(pending), MAX_PARALLEL_TOOLS)
        ) as pool:
            futures = [(i, pool.submit(self._run_one, call)) for i, call in pending]
            cancelled: BaseException | None = None
            try:
                for i, fut in futures:  # submission order == calls order
                    if self._interrupted():
                        # Cancel while a batch is in flight: the except
                        # below still records every finished worker's REAL
                        # result before propagating.
                        raise KeyboardInterrupt
                    results[i] = fut.result()
            except BaseException as exc:
                cancelled = exc

        if cancelled is not None:
            # Workers never receive SIGINT, so the `with` above has already
            # joined them (bounded by tool timeouts) and every future is
            # done. The work HAPPENED -- record the real results before
            # propagating, so history stays faithful AND resumable.
            for i, fut in futures:
                if results[i] is None:
                    try:
                        results[i] = fut.result()
                    except BaseException as exc:  # e.g. a worker hit Ctrl-C
                        results[i] = ToolResult(
                            calls[i].id,
                            f"lost during cancellation: {type(exc).__name__}: {exc}",
                            is_error=True,
                        )
            self.history.append(
                _batch_message([r for r in results if r is not None])
            )
            raise cancelled

        return results  # type: ignore[return-value]

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

    def _run_one(self, call: ToolCall) -> ToolResult:
        """Execution stage: run the tool, wrap every failure as data.

        The hooks bracket the real execution (before = starting, after =
        outcome, errors included). They are NOT wrapped in try/except: a
        raising hook propagates and crashes the turn -- deliberate, see
        the constructor comment. (May fire on a pool worker thread in
        batches; hook authors own their thread-safety.)
        """
        self.on_before_tool(call)
        tool = self._get_visible_tool(call.name)
        try:
            output = tool.run(call.arguments, self.ctx)
        except ToolError as exc:
            result = ToolResult(call.id, str(exc), is_error=True)
        except KeyboardInterrupt:
            raise  # cancellation is not a tool failure
        except Exception as exc:
            result = ToolResult(call.id, f"{type(exc).__name__}: {exc}",
                                is_error=True)
        else:
            if isinstance(output, ToolOutput):
                # The rich-result path: text truncates like any other,
                # images ride along on the result for the batch append to
                # hoist into history (see _batch_message).
                result = ToolResult(call.id, _truncate_middle(output.text),
                                    images=output.images)
            else:
                result = ToolResult(call.id, _truncate_middle(output))
        self.on_after_tool(call, result)
        return result

    # ---- the invariant ----------------------------------------------------

    def _answer_outstanding(self, executed: dict[str, ToolResult]) -> None:
        """Guarantee every tool_call in the trailing assistant message has
        a matching ToolResult in history. Called on abnormal exits only.

        Without this, the NEXT request references a tool_use id that never
        got a result -- and Anthropic rejects it with a 400. This helper
        is why cancelling a turn here never corrupts the session.
        """
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
