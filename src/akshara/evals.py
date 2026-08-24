"""Evals: trajectory-level checks that the harness produces RIGHT BEHAVIOR.

Unit tests pin mechanics with scripted providers (deterministic, free,
every commit). Evals ask a different question -- did a REAL model, over
a whole trajectory, finish the task, use the right tools, stay in
budget? -- so they cost money, are probabilistic, and belong on a
merge/nightly cadence, not per-commit. "Agent complexity is only
justified when you can define precise task-success criteria" (Hamel
Husain, via book ch19); without evals, agent features are debt.

Four metric classes, all checked per case and ACCUMULATED (a case can
fail for five reasons at once -- seeing all five is how you fix it):

* completion   -- it finished (crashes and iteration caps become
                 failed results, never raised exceptions)
* correctness  -- task-specific, deterministic where possible
                 (``check_answer``); reserve LLM judging (``judge``)
                 for genuinely subjective criteria
* process      -- required tools used, forbidden tools avoided
* cost         -- token and iteration ceilings; a correct answer at
                 50K tokens is worse than the same answer at 5K

The recording trick: the runner wraps every tool in a proxy that
appends each ACTUAL execution to a shared list before delegating, so
process checks see what really ran -- including work a sub-agent did
through the same tool instances (documented conflation: delegation is
the parent's doing).

Every real failure in production should leave a fossil in the suite:
``case_from_trace`` turns an observed failure into a regression case
whose budget is observed-cost x 1.5.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from akshara.agent import Agent
from akshara.async_agent import AsyncAgent
from akshara.providers.base import Provider, collect
from akshara.subagent import SpawnSubagent, SubagentSpawner
from akshara.tools.base import Tool, ToolContext, ToolRegistry
from akshara.types import Message, TextBlock


@dataclass(slots=True)
class EvalCase:
    """A golden trajectory: task + expected outcomes + scorer."""

    id: str
    description: str
    user_message: str
    system: str | None = None
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    check_answer: Callable[[str], bool] | None = None
    max_tokens: int | None = None       # input+output ceiling for the turn
    max_iterations: int | None = None   # model round-trips ceiling
    setup: Callable[[Agent], None] | None = None  # per-case agent wiring
                                      # (e.g. spawn_setup() -> sub-agents)


@dataclass(slots=True)
class EvalResult:
    case_id: str
    passed: bool
    failures: list[str]
    final_answer: str
    tokens_used: int
    iterations_used: int
    tool_calls_seen: list[str]
    duration_seconds: float
    error: str | None = None            # crash description, if any


class _RecordingTool(Tool):
    """Delegating proxy that records every real execution of a tool.

    Instance attributes shadow the ABC's ClassVars, so the wrapped
    tool's identity (name, schema, read_only) is preserved for spec(),
    permission gates, and sub-agent catalog copies alike.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, Any]] = {}

    def __init__(self, inner: Tool, seen: list[str]) -> None:
        self._inner = inner
        self._seen = seen
        self.name = inner.name
        self.description = inner.description
        self.parameters = inner.parameters
        self.read_only = inner.read_only

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return self._inner.summary(args, ctx)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        self._seen.append(self._inner.name)  # execution, not merely lookup
        return self._inner.run(args, ctx)


def recording_registry(base: ToolRegistry, seen: list[str]) -> ToolRegistry:
    """Wrap every tool in a recording proxy. IDEMPOTENT: a registry that
    already holds proxies (re-wrapping after case.setup registered extras)
    unwraps to the real tool first -- an execution is recorded exactly
    once no matter how many layers it sits under."""
    reg = ToolRegistry()
    for name in base.names():
        tool = base.get(name)
        inner = getattr(tool, "_inner", None)
        reg.register(_RecordingTool(inner if inner is not None else tool, seen))
    return reg


class EvalRunner:
    """Runs cases sequentially against a live provider.

    Sequential is deliberate for 20-50 cases: deterministic ordering,
    easy rate-limiting, readable logs. Fresh Agent per case -- golden
    trajectories are independent, no history bleed.
    """

    def __init__(self, provider: Provider, model: str, *, tools: ToolRegistry,
                 permissions: Callable[[Any], bool], max_iterations: int = 25,
                 context_window: int | None = None,
                 cwd: Any = None) -> None:
        self.provider = provider
        self.model = model
        self.base_tools = tools
        self.permissions = permissions
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.cwd = cwd  # sandbox root for the cases' tool calls
        self.seen: list[str] = []  # shared with the proxies; cleared per case

    def run_case(self, case: EvalCase) -> EvalResult:
        self.seen.clear()
        # Agent.context_window has a concrete default (200k) that its
        # compaction math requires -- only override when actually given
        kwargs: dict[str, Any] = {}
        if self.context_window is not None:
            kwargs["context_window"] = self.context_window
        if self.cwd is not None:
            kwargs["cwd"] = self.cwd
        agent = Agent(
            self.provider, model=self.model,
            system=case.system,
            tools=recording_registry(self.base_tools, self.seen),
            max_iterations=self.max_iterations,
            permissions=self.permissions,
            **kwargs,
        )
        if case.setup is not None:
            # per-case wiring seam: the runner builds a fresh Agent, the
            # case decides what EXTRA machinery it gets. Re-wrap after so
            # setup-registered tools are recorded too (idempotent for the
            # originals -- no double-counting).
            case.setup(agent)
            agent.registry = recording_registry(agent.registry, self.seen)
        start = time.monotonic()
        try:
            response = agent.run(case.user_message)
        except Exception as exc:
            # completion failure is a RESULT, not a raise: one broken
            # case must not abort the suite (ProviderError, iteration
            # cap RuntimeError, anything unexpected)
            return EvalResult(
                case_id=case.id, passed=False,
                failures=[f"crashed: {type(exc).__name__}: {exc}"],
                final_answer="", tokens_used=agent.total_usage.input_tokens
                + agent.total_usage.output_tokens,
                iterations_used=self._count_iterations(agent),
                tool_calls_seen=list(self.seen),
                duration_seconds=time.monotonic() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        duration = time.monotonic() - start
        answer = response.message.text().strip()
        tokens = agent.total_usage.input_tokens + agent.total_usage.output_tokens
        return self._result(case, agent, list(self.seen), answer, tokens,
                            duration)

    def run_all(self, cases: list[EvalCase]) -> list[EvalResult]:
        return [self.run_case(case) for case in cases]

    def _result(self, case: EvalCase, agent: Agent, seen: list[str],
                answer: str, tokens: int, duration: float) -> EvalResult:
        """Score a finished trajectory. SHARED with AsyncEvalRunner --
        identical grading rules are the whole point of the twin."""
        iterations = EvalRunner._count_iterations(agent)
        failures: list[str] = []
        if case.check_answer is not None and not case.check_answer(answer):
            failures.append("answer check failed")
        for name in case.required_tools:
            if name not in seen:
                failures.append(f"required tool not used: {name}")
        for name in case.forbidden_tools:
            if name in seen:
                failures.append(f"forbidden tool used: {name}")
        if case.max_tokens is not None and tokens > case.max_tokens:
            failures.append(f"over token budget: {tokens} > {case.max_tokens}")
        if case.max_iterations is not None and iterations > case.max_iterations:
            failures.append(
                f"over iteration budget: {iterations} > {case.max_iterations}")
        return EvalResult(
            case_id=case.id, passed=not failures, failures=failures,
            final_answer=answer, tokens_used=tokens,
            iterations_used=iterations, tool_calls_seen=seen,
            duration_seconds=duration,
        )

    @staticmethod
    def _count_iterations(agent: Agent) -> int:
        return sum(1 for m in agent.history if m.role == "assistant")


class AsyncEvalRunner:
    """The async twin of EvalRunner: same cases, same scoring, one event
    loop driving several trajectories at once.

    Grading is SHARED (``EvalRunner._result``) -- identical rules is the
    whole point; only the driving differs. Two deliberate differences:

    * each case gets its OWN seen-list. The sync runner shares one and
      clears it per case (fine sequentially); concurrent cases would
      scribble on each other.
    * ``run_all`` bounds in-flight cases with a semaphore -- golden
      trajectories are independent but the provider behind them is not;
      N-at-once multiplies request rate.

    The delegate case runs here unchanged: SubagentSpawner reads only
    attributes AsyncAgent mirrors (registry/provider/model/permissions/...
    ), and its child is a sync Agent whose blocking run() goes through
    the spawn tool's to_thread default arun -- off the loop, per the
    tools doctrine in [notes/11](notes/11-async.md).
    """

    def __init__(self, provider: Provider, model: str, *, tools: ToolRegistry,
                 permissions: Callable[[Any], bool], max_iterations: int = 25,
                 context_window: int | None = None, cwd: Any = None,
                 concurrency: int = 4) -> None:
        self.provider = provider
        self.model = model
        self.base_tools = tools
        self.permissions = permissions
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.cwd = cwd
        self.concurrency = concurrency

    async def run_case(self, case: EvalCase) -> EvalResult:
        seen: list[str] = []  # per case: concurrent cases must not share
        kwargs: dict[str, Any] = {}
        if self.context_window is not None:
            kwargs["context_window"] = self.context_window
        if self.cwd is not None:
            kwargs["cwd"] = self.cwd
        agent = AsyncAgent(
            self.provider, model=self.model,
            system=case.system,
            tools=recording_registry(self.base_tools, seen),
            max_iterations=self.max_iterations,
            permissions=self.permissions,
            **kwargs,
        )
        if case.setup is not None:
            case.setup(agent)
            agent.registry = recording_registry(agent.registry, seen)
        start = time.monotonic()
        try:
            response = await agent.run(case.user_message)
        except Exception as exc:
            return EvalResult(
                case_id=case.id, passed=False,
                failures=[f"crashed: {type(exc).__name__}: {exc}"],
                final_answer="",
                tokens_used=agent.total_usage.input_tokens
                + agent.total_usage.output_tokens,
                iterations_used=EvalRunner._count_iterations(agent),
                tool_calls_seen=list(seen),
                duration_seconds=time.monotonic() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
        answer = response.message.text().strip()
        tokens = agent.total_usage.input_tokens + agent.total_usage.output_tokens
        # grading rules are the SYNC runner's, verbatim
        return EvalRunner._result(
            self, case, agent, list(seen), answer,
            tokens, time.monotonic() - start)

    async def run_all(self, cases: list[EvalCase]) -> list[EvalResult]:
        gate = asyncio.Semaphore(self.concurrency)

        async def _bounded(case: EvalCase) -> EvalResult:
            async with gate:
                return await self.run_case(case)

        # gather preserves SUBMISSION order: results align with `cases`
        # no matter who finishes first (same contract as tool batches).
        return await asyncio.gather(*(_bounded(c) for c in cases))


def judge(provider: Provider, model: str, *, question: str, answer: str,
          reference: str | None = None,
          criteria: str | None = None) -> tuple[bool, str]:
    """LLM-as-judge for genuinely SUBJECTIVE criteria.

    Deliberately boring: the judge replies 'PASS' or 'FAIL' plus one
    sentence; anything not starting with PASS counts as FAIL (a confused
    judge must fail closed). Two known limits (book ch19): judge and
    candidate sharing a brain correlates their errors -- prefer a
    different provider/model for judging -- and a judge can't grade
    past its own capability ceiling. A deterministic check_answer beats
    this whenever a function would do; wire it in as
    ``check_answer=lambda a: judge(...)[0]``.
    """
    system = ("You are an impartial grader of AI answers. Reply with "
              "'PASS' or 'FAIL' followed by a one-sentence reason. "
              "Nothing else.")
    parts = [f"QUESTION:\n{question}", f"CANDIDATE ANSWER:\n{answer}"]
    if reference:
        parts.append(f"REFERENCE ANSWER:\n{reference}")
    parts.append("CRITERIA:\n" + (criteria or
                "accuracy, completeness, and relevance to the question"))
    response = collect(provider.stream(
        messages=[Message(role="user",
                          content=[TextBlock(text="\n\n".join(parts))])],
        system=system, model=model,
        tools=[],  # the grader runs with no tools -- by contract, not omission
    ))
    text = response.message.text().strip()
    return text.upper().startswith("PASS"), text


def case_from_trace(trace_id: str, failure_reason: str, user_message: str, *,
                    tokens_used: int, required_tools: list[str] | None = None,
                    **case_kwargs: Any) -> EvalCase:
    """Turn a real failure into a regression case (the fossil rule).

    Budget heuristic: whatever the failing run SPENT becomes the ceiling
    (x1.5 headroom) -- the fix must not cost more than the bug.
    """
    return EvalCase(
        id=f"trace-{trace_id[:8]}",
        description=failure_reason,
        user_message=user_message,
        max_tokens=int(tokens_used * 1.5) or None,
        required_tools=required_tools or [],
        **case_kwargs,
    )


def spawn_setup(*, max_per_session: int = 3,
                default_max_iterations: int = 10) -> Callable[[Agent], None]:
    """Case.setup factory: opt ONE case's agent into sub-agents.

    The runner builds a fresh Agent per case, and the spawn budget lives
    on the spawner -- so every spawn-capable case needs its own wiring;
    this attaches spawner + tool to that case's agent only. Eval-tight
    child budgets (3 spawns / 10 iterations) keep a runaway delegator
    from torching the suite.

    With this in place, ``required_tools=["spawn_subagent"]`` asserts
    delegation actually HAPPENED (the narrate-instead-of-delegate failure
    mode shows up as "required tool not used"), and work the CHILD does
    through shared tool instances lands in the same seen-list --
    documented conflation: delegation is the parent's doing.
    """
    def setup(agent: Agent) -> None:
        spawner = SubagentSpawner(agent, max_per_session=max_per_session,
                                  default_max_iterations=default_max_iterations)
        agent.registry.register(SpawnSubagent(spawner))
    return setup


def summarize(results: list[EvalResult]) -> str:
    """Human-readable report: one line per case + the aggregate."""
    lines = []
    for r in results:
        mark = "✓" if r.passed else "✗"
        tools = ",".join(r.tool_calls_seen) if r.tool_calls_seen else "-"
        line = (f"{mark} {r.case_id}: {r.tokens_used}tok · "
                f"{r.iterations_used}it · {r.duration_seconds:.1f}s · "
                f"[{tools}]")
        if r.failures:
            line += f" -- {'; '.join(r.failures)}"
        lines.append(line)
    passed = sum(1 for r in results if r.passed)
    lines.append(f"{passed}/{len(results)} passed")
    return "\n".join(lines)
