"""Evals: the METADATA machinery (completion / process / cost checks,
recording proxies, fossil conversion, reporting) is deterministic and
gets unit tests here. Only live-model correctness is left to real runs
(examples/run_evals.py) -- exactly the tests-vs-evals split ch19 draws.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar

from conftest import ScriptedProvider, assistant_text, assistant_tool_call

from akshara.evals import (
    AsyncEvalRunner,
    EvalCase,
    EvalResult,
    EvalRunner,
    case_from_trace,
    judge,
    spawn_setup,
    summarize,
)
from akshara.errors import ProviderError
from akshara.permissions import yolo
from akshara.subagent import SPAWN_TOOL_NAME
from akshara.tools.base import Tool, ToolRegistry
from akshara.types import Usage


class EchoTool(Tool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "returns its text argument"
    parameters: ClassVar[dict] = {"type": "object",
                                  "properties": {"text": {"type": "string"}},
                                  "required": ["text"]}
    read_only: ClassVar[bool] = True

    def summary(self, args: dict[str, Any], ctx) -> str:
        return f"echo {args.get('text', '')!r}"

    def run(self, args: dict[str, Any], ctx) -> str:
        return str(args.get("text", ""))


def make_runner(script: list) -> tuple[EvalRunner, ScriptedProvider]:
    provider = ScriptedProvider(script)
    registry = ToolRegistry()
    registry.register(EchoTool())
    return (EvalRunner(provider, "m", tools=registry, permissions=yolo),
            provider)


def case(**overrides: Any) -> EvalCase:
    base = dict(id="c1", description="d", user_message="do the thing",
                required_tools=[], forbidden_tools=[])
    return EvalCase(**{**base, **overrides})


class TestChecks:
    def test_happy_trajectory_passes_and_records(self):
        runner, _ = make_runner([
            assistant_tool_call("t1", "echo", {"text": "hi"}),
            assistant_text("all done",
                           usage=Usage(input_tokens=10, output_tokens=5)),
        ])
        result = runner.run_case(case(check_answer=lambda a: "done" in a))
        assert result.passed and result.failures == []
        assert result.tool_calls_seen == ["echo"]
        assert result.tokens_used == 15
        assert result.iterations_used == 2

    def test_failures_accumulate_instead_of_first_fail(self):
        runner, _ = make_runner([assistant_text(
            "no tools here", usage=Usage(input_tokens=100, output_tokens=50))])
        result = runner.run_case(case(
            required_tools=["echo"],
            forbidden_tools=["bash"],
            check_answer=lambda a: False,
            max_tokens=10,
        ))
        assert not result.passed
        joined = "; ".join(result.failures)
        # ALL of these visible at once is how you diagnose a case
        assert "answer check failed" in joined
        assert "required tool not used: echo" in joined
        assert "over token budget: 150 > 10" in joined
        assert "forbidden tool used" not in joined  # bash never ran

    def test_forbidden_tool_used_is_caught(self):
        runner, _ = make_runner([
            assistant_tool_call("t1", "echo", {"text": "x"}),
            assistant_text("ok"),
        ])
        result = runner.run_case(case(forbidden_tools=["echo"]))
        assert not result.passed
        assert "forbidden tool used: echo" in result.failures


class TestCompletion:
    def test_crash_becomes_failed_result_not_an_exception(self):
        class Exploding(ScriptedProvider):
            def stream(self, **kwargs):
                raise ProviderError("503 upstream", status=503)
                yield  # pragma: no cover

        provider = Exploding([])
        registry = ToolRegistry()
        registry.register(EchoTool())
        runner = EvalRunner(provider, "m", tools=registry, permissions=yolo)
        result = runner.run_case(case())
        assert not result.passed
        assert result.error.startswith("ProviderError")
        assert any(f.startswith("crashed:") for f in result.failures)

    def test_fresh_agent_per_case_no_history_bleed(self):
        runner, provider = make_runner([
            assistant_text("first"),
            assistant_text("second"),
        ])
        r1 = runner.run_case(case(id="a"))
        r2 = runner.run_case(case(id="b"))
        assert r1.passed and r2.passed
        # each request carried ONLY its own user message: one case's
        # history never leaked into the next
        for request in provider.requests:
            assert len(request["messages"]) == 1
        assert r2.tool_calls_seen == []  # seen-list reset between cases


class TestSetupSeam:
    def test_setup_gets_each_fresh_agent(self):
        runner, _ = make_runner([
            assistant_text("first"),
            assistant_text("second"),
        ])
        agents = []
        runner.run_case(case(id="a", setup=agents.append))
        runner.run_case(case(id="b", setup=agents.append))
        assert len(agents) == 2
        assert agents[0] is not agents[1]  # fresh wiring per case, not shared

    def test_setup_registered_tools_are_recorded_exactly_once(self):
        # idempotent re-wrapping: originals keep ONE recording layer even
        # though the registry is wrapped twice (before + after setup)
        runner, _ = make_runner([
            assistant_tool_call("t1", "echo", {"text": "x"}),
            assistant_tool_call("t2", "extra", {"text": "y"}),
            assistant_text("done"),
        ])

        class Extra(EchoTool):
            name: ClassVar[str] = "extra"

        def setup(agent) -> None:
            agent.registry.register(Extra())

        result = runner.run_case(case(
            required_tools=["echo", "extra"], setup=setup))

        assert result.passed, result.failures
        assert result.tool_calls_seen.count("echo") == 1
        assert result.tool_calls_seen.count("extra") == 1

    def test_spawn_case_delegation_is_assertable_and_conflated(self):
        script = [
            assistant_tool_call("p1", SPAWN_TOOL_NAME, {
                "objective": "have a child echo something",
                "output_format": "one line",
                "tools_allowed": ["echo"],
                "justification": "isolated window"}),
            assistant_tool_call("c1", "echo", {"text": "child worked"}),
            assistant_text("child done"),
            assistant_text("parent relays"),
        ]
        provider = ScriptedProvider(script)
        registry = ToolRegistry()
        registry.register(EchoTool())
        runner = EvalRunner(provider, "m", tools=registry, permissions=yolo)

        result = runner.run_case(case(
            id="spawn",
            required_tools=[SPAWN_TOOL_NAME, "echo"],
            check_answer=lambda a: "relays" in a,
            setup=spawn_setup(),
        ))

        assert result.passed, result.failures
        # delegation itself is in the record...
        assert result.tool_calls_seen.count(SPAWN_TOOL_NAME) == 1
        # ...and so is the child's execution of the shared tool instance
        # (documented conflation: delegation is the parent's doing)
        assert result.tool_calls_seen.count("echo") == 1


class TestJudge:
    def test_pass_fail_and_fail_closed(self):
        for verdict, expected in [("PASS because complete", True),
                                  ("FAIL missing detail", False),
                                  ("unsure, honestly", False)]:
            _, provider = make_runner([assistant_text(verdict)])
            ok, raw = judge(provider, "m", question="q?", answer="a")
            assert ok is expected
            assert raw == verdict

    def test_judge_prompt_carries_the_material(self):
        _, provider = make_runner([assistant_text("PASS x")])
        judge(provider, "m", question="What is 2+2?",
              answer="five", reference="four", criteria="arithmetic only")
        request = provider.requests[0]
        assert "impartial grader" in request["system"]
        blob = str(request["messages"])
        for fragment in ("What is 2+2?", "five", "four", "arithmetic only"):
            assert fragment in blob


class TestAsyncRunner:
    """The async twin: same cases, same grading rules, one loop driving
    several trajectories. These tests pin the CONTRACT of the twin --
    parity with sync scoring, submission-order results, per-case seen
    isolation, bounded concurrency -- against the same ScriptedProvider.
    """

    @staticmethod
    def make_async_runner(script: list, *, concurrency: int = 4
                          ) -> tuple[AsyncEvalRunner, ScriptedProvider]:
        provider = ScriptedProvider(script)
        registry = ToolRegistry()
        registry.register(EchoTool())
        return (AsyncEvalRunner(provider, "m", tools=registry,
                                permissions=yolo, concurrency=concurrency),
                provider)

    def test_grading_matches_the_sync_twin_field_for_field(self):
        script = [
            assistant_tool_call("t1", "echo", {"text": "hi"}),
            assistant_text("all done",
                           usage=Usage(input_tokens=10, output_tokens=5)),
        ]
        the_case = case(id="c1", check_answer=lambda a: "done" in a)

        sync_runner, _ = make_runner(script)
        sync_result = sync_runner.run_case(the_case)

        async def _async_one():
            runner, _ = self.make_async_runner(script)
            return await runner.run_case(the_case)

        async_result = asyncio.run(_async_one())
        for field in ("passed", "failures", "final_answer", "tokens_used",
                      "iterations_used", "tool_calls_seen"):
            assert getattr(sync_result, field) == getattr(async_result, field), \
                f"{field} diverged between twins"

    def test_results_align_with_submission_order_not_finish_order(self):
        # three one-request cases whose responses land in REVERSE order;
        # gather must still hand back results indexed like `cases`
        delays = iter([0.15, 0.10, 0.05])

        class Slow(ScriptedProvider):
            async def astream(self, **kwargs):
                await asyncio.sleep(next(delays))
                async for event in super().astream(**kwargs):
                    yield event

        provider = Slow([assistant_text(f"done {i}") for i in range(3)])
        registry = ToolRegistry()
        registry.register(EchoTool())
        runner = AsyncEvalRunner(provider, "m", tools=registry,
                                 permissions=yolo)
        cases = [case(id=cid) for cid in ("a", "b", "c")]

        results = asyncio.run(runner.run_all(cases))
        assert [r.case_id for r in results] == ["a", "b", "c"]
        assert all(r.passed for r in results)

    def test_concurrent_cases_do_not_share_their_seen_lists(self):
        # exactly ONE of the two trajectories executes echo; if they
        # shared a seen-list (the sync runner's cleared singleton),
        # BOTH results would record it
        script = [
            assistant_tool_call("t1", "echo", {"text": "x"}),
            assistant_text("a done"),
            assistant_text("b done"),
        ]

        async def _two():
            runner, _ = self.make_async_runner(script)
            return await asyncio.gather(runner.run_case(case(id="a")),
                                        runner.run_case(case(id="b")))

        ra, rb = asyncio.run(_two())
        seen_lists = sorted([ra.tool_calls_seen, rb.tool_calls_seen])
        assert seen_lists == [[], ["echo"]]

    def test_crash_becomes_failed_result_even_under_gather(self):
        class Exploding(ScriptedProvider):
            def stream(self, **kwargs):
                raise ProviderError("503 upstream", status=503)
                yield  # pragma: no cover

        provider = Exploding([])
        registry = ToolRegistry()
        registry.register(EchoTool())
        runner = AsyncEvalRunner(provider, "m", tools=registry,
                                 permissions=yolo)

        results = asyncio.run(runner.run_all([case(id="boom"), case(id="also-boom")]))
        assert len(results) == 2  # one broken case did not sink the batch
        for r in results:
            assert not r.passed
            assert r.error.startswith("ProviderError")
            assert any(f.startswith("crashed:") for f in r.failures)

    def test_concurrency_cap_bounds_inflight_cases(self):
        inflight = 0
        peak = 0

        class Tracked(ScriptedProvider):
            async def astream(self, **kwargs):
                nonlocal inflight, peak
                inflight += 1
                peak = max(peak, inflight)
                try:
                    await asyncio.sleep(0.02)  # hold the slot long enough to overlap
                    async for event in super().astream(**kwargs):
                        yield event
                finally:
                    inflight -= 1

        provider = Tracked([assistant_text(f"done {i}") for i in range(6)])
        registry = ToolRegistry()
        registry.register(EchoTool())
        runner = AsyncEvalRunner(provider, "m", tools=registry,
                                 permissions=yolo, concurrency=2)

        start = time.monotonic()
        results = asyncio.run(
            runner.run_all([case(id=f"c{i}") for i in range(6)]))
        elapsed = time.monotonic() - start

        assert len(results) == 6 and all(r.passed for r in results)
        assert peak <= 2, f"semaphore leaked: {peak} cases ran at once"
        assert peak > 1, "cases never overlapped -- nothing was concurrent"
        # 6 x 0.02s serial would be >= 0.12s; two-wide overlap finishes faster
        assert elapsed < 0.12

    def test_delegate_case_runs_through_the_async_twin_unchanged(self):
        # spawn_setup wires a SubagentSpawner onto an ASYNC agent here;
        # the child is a sync Agent run via to_thread inside the spawn
        # tool's default arun -- off the loop, so this exercises the
        # compatibility claim in AsyncEvalRunner's docstring
        script = [
            assistant_tool_call("p1", SPAWN_TOOL_NAME, {
                "objective": "have a child echo something",
                "output_format": "one line",
                "tools_allowed": ["echo"],
                "justification": "isolated window"}),
            assistant_tool_call("c1", "echo", {"text": "child worked"}),
            assistant_text("child done"),
            assistant_text("parent relays"),
        ]

        async def _spawn():
            runner, _ = self.make_async_runner(script)
            return await runner.run_case(case(
                id="spawn",
                required_tools=[SPAWN_TOOL_NAME, "echo"],
                check_answer=lambda a: "relays" in a,
                setup=spawn_setup(),
            ))

        result = asyncio.run(_spawn())
        assert result.passed, result.failures
        assert result.tool_calls_seen.count(SPAWN_TOOL_NAME) == 1
        # child's execution through the shared tool instance lands in
        # THIS case's own seen-list (conflation documented on spawn_setup)
        assert result.tool_calls_seen.count("echo") == 1


class TestFossilRuleAndReporting:
    def test_case_from_trace_budget_is_observed_times_15(self):
        c = case_from_trace("abcdef123456", "hallucinated a path",
                            "where is X defined?", tokens_used=10000,
                            required_tools=["grep"])
        assert c.id == "trace-abcdef12"
        assert c.description == "hallucinated a path"
        assert c.max_tokens == 15000
        assert c.required_tools == ["grep"]

    def test_summarize_lines_and_aggregate(self):
        results = [_result("c-ok", True), _result("c-bad", False,
                                                  ["answer check failed"])]
        report = summarize(results)
        assert report.count("\n") == 2
        assert "✓ c-ok" in report
        assert "✗ c-bad: 7tok · 1it" in report
        assert "-- answer check failed" in report
        assert report.endswith("1/2 passed")


def _result(case_id: str, passed: bool,
            failures: list[str] | None = None) -> EvalResult:
    return EvalResult(case_id=case_id, passed=passed, failures=failures or [],
                      final_answer="", tokens_used=7, iterations_used=1,
                      tool_calls_seen=[], duration_seconds=0.25)
