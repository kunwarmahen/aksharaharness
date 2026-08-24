"""The async agent loop: same rules as ``test_agent_loop.py``, awaited.

The point of every test here is NOT "asyncio works" -- it's that the
invariants survived the translation: errors-as-data, submission-order
results, the history invariant under every exit path (including task
cancellation mid-batch), and push-based streaming.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from akshara.agent import INTERRUPTED_MESSAGE, ToolExecuted, TurnEnd
from akshara.async_agent import AsyncAgent
from akshara.tools.base import Tool, ToolRegistry
from akshara.types import Message, ModelResponse, TextBlock, ToolCall, ToolResult, Usage

from conftest import (
    ScriptedProvider,
    assistant_text,
    assistant_tool_call,
)


# ---- fixtures --------------------------------------------------------------


class EchoTool(Tool):
    """Plain blocking tool: exercises the default arun() -> to_thread path."""

    name = "echo"
    description = "repeat text"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}
    read_only = True

    def summary(self, args, ctx):
        return f"echo {args.get('text', '')!r}"

    def run(self, args, ctx):
        return f"echoed: {args['text']}"


class SlowBlockingTool(EchoTool):
    """Sleeps IN A THREAD (default arun): models a real blocking syscall."""

    name = "slow_blocking"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def run(self, args, ctx):
        time.sleep(self.seconds)
        return f"slow done: {args['text']}"


class SlowAsyncTool(EchoTool):
    """Overrides arun with true async delay -- the override escape hatch."""

    name = "slow_async"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def arun(self, args, ctx):
        await asyncio.sleep(self.seconds)
        return f"async done: {args['text']}"


def make_agent(script, *, tools=None, permissions=None, context_window=200_000,
               max_iterations=25, **kwargs) -> AsyncAgent:
    registry = None
    if tools:
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
    return AsyncAgent(
        ScriptedProvider(script),
        model="m",
        tools=registry,
        permissions=permissions,
        context_window=context_window,
        max_iterations=max_iterations,
        **kwargs,
    )


def tool_results_of(history):
    out = []
    for m in history:
        if m.role == "user":
            out.extend(b for b in m.content if isinstance(b, ToolResult))
    return out


# ---- the happy paths --------------------------------------------------------


class TestHappyPaths:
    def test_end_turn_single_iteration(self):
        agent = make_agent([assistant_text("hi there")])

        async def _run():
            return [event async for event in agent.run_streaming("hello")]

        events = asyncio.run(_run())
        turns = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turns) == 1
        assert turns[0].response.message.text() == "hi there"
        assert turns[0].reason == "end_turn"

    def test_tool_round_trip_keeps_the_invariant(self):
        script = [
            assistant_tool_call("call_1", "echo", {"text": "ping"}),
            assistant_text("done"),
        ]
        agent = make_agent(script, tools=[EchoTool()])

        async def _run():
            return [event async for event in agent.run_streaming("use echo")]

        events = asyncio.run(_run())
        executed = [e for e in events if isinstance(e, ToolExecuted)]
        assert len(executed) == 1
        assert executed[0].result.content == "echoed: ping"
        assert not executed[0].result.is_error
        # invariant: the call got its answer before the next request
        requests = agent.provider.requests
        assert len(requests) == 2
        second_messages = requests[1]["messages"]
        trailing = [b for b in second_messages[-1].content]
        assert any(getattr(b, "tool_call_id", None) == "call_1" for b in trailing)

    def test_unknown_tool_is_data_not_a_crash(self):
        script = [
            assistant_tool_call("call_1", "no_such_tool", {}),
            assistant_text("recovered"),
        ]
        agent = make_agent(script)
        events = asyncio.run(_collect(agent.run_streaming("go")))
        executed = [e for e in events if isinstance(e, ToolExecuted)]
        assert executed[0].result.is_error
        assert "no such tool" in executed[0].result.content

    def test_batch_yields_in_submission_order_despite_completion_order(self):
        # c1 sleeps LONGER but was submitted FIRST: panels stay deterministic.
        script = [
            assistant_tool_call("c1", "slow_a", {"text": "a"}),
            assistant_tool_call("c2", "slow_b", {"text": "b"}),
            assistant_text("both done"),
        ]
        tools = [_named_delay("slow_a", 0.15), _named_delay("slow_b", 0.01)]
        agent = make_agent(script, tools=tools)

        events = asyncio.run(_collect(agent.run_streaming("two tools")))
        executed = [e for e in events if isinstance(e, ToolExecuted)]
        assert [e.result.content for e in executed] == [
            "async done: a", "async done: b"]  # submission order, not completion

    def test_default_arun_pushes_blocking_tools_to_threads(self):
        script = [
            assistant_tool_call("c1", "slow_blocking", {"text": "x"}),
            assistant_text("ok"),
        ]
        agent = make_agent(script, tools=[SlowBlockingTool(0.05)])
        events = asyncio.run(_collect(agent.run_streaming("blocking")))
        executed = [e for e in events if isinstance(e, ToolExecuted)]
        assert executed[0].result.content == "slow done: x"


async def _collect(agen):
    return [event async for event in agen]


def _named_delay(name: str, seconds: float) -> Tool:
    """A SlowAsyncTool subclass under a distinct registered name."""

    class _T(SlowAsyncTool):
        pass

    _T.name = name
    return _T(seconds)


# ---- the hard part: cancellation -------------------------------------------


class TestCancellation:
    def test_cancel_during_batch_records_real_results(self):
        """The async twin of the sync Ctrl-C-mid-batch story: to_thread
        workers can't be interrupted, so BOTH real results must land."""
        # ONE response carrying TWO calls -- that's what makes a batch
        two_call = ModelResponse(
            message=Message("assistant", [
                TextBlock("working"),
                ToolCall("c1", "slow_blocking", {"text": "one"}),
                ToolCall("c2", "slow_blocking", {"text": "two"}),
            ]),
            stop_reason="tool_use",
            usage=Usage(),
        )
        agent = make_agent([two_call], tools=[SlowBlockingTool(0.4)])

        async def _scenario():
            task = asyncio.ensure_future(_collect(agent.run_streaming("go")))
            await asyncio.sleep(0.15)   # both workers are mid-sleep now
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_scenario())

        results = tool_results_of(agent.history)
        assert len(results) == 2
        assert all(not r.is_error for r in results)          # work HAPPENED
        assert sorted(r.content for r in results) == [
            "slow done: one", "slow done: two"]
        # and the invariant holds: last message answers every call
        assert agent.history[-1].role == "user"

    def test_cancel_mid_stream_leaves_history_resumable(self):
        """Cancel while the FIRST response is still streaming: nothing was
        appended, so history must be untouched -- and the session must
        survive, provably, by continuing the conversation afterwards."""

        class SlowStream(ScriptedProvider):
            """Scripted responses whose EVENTS dribble out slowly."""

            async def astream(self, **kwargs):
                async for event in super().astream(**kwargs):
                    await asyncio.sleep(0.05)
                    yield event

        provider = SlowStream([
            assistant_tool_call("c1", "echo", {"text": "x"}),
            assistant_text("recovered"),
        ])
        registry = ToolRegistry()
        registry.register(EchoTool())
        agent = AsyncAgent(provider, model="m", tools=registry)

        async def _scenario():
            task = asyncio.ensure_future(_collect(agent.run_streaming("go")))
            await asyncio.sleep(0.02)    # parked inside the first response
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_scenario())
        # unwind left only the appended goal -- nothing half-recorded
        assert len(agent.history) == 1
        assert agent.history[-1].role == "user"
        # THE proof of resumability: next turn just works
        response = asyncio.run(agent.run("try again"))
        assert response.message.text() == "recovered"

    def test_closing_the_generator_early_still_answers_calls(self):
        """Async twin of sync generator.close(): abandon after first panel."""
        script = [
            assistant_tool_call("c1", "echo", {"text": "x"}),
            assistant_text("never"),
        ]
        agent = make_agent(script, tools=[EchoTool()])

        async def _scenario():
            agen = agent.run_streaming("go")
            async for event in agen:
                if isinstance(event, ToolExecuted):
                    break
            await agen.aclose()

        asyncio.run(_scenario())
        results = tool_results_of(agent.history)
        # the REAL result (already computed when the panel yielded) replays
        assert results[-1].content == "echoed: x"

    def test_iteration_cap_synthesizes_and_reports(self):
        script = [assistant_tool_call(f"c{n}", "echo", {"text": str(n)})
                  for n in range(5)]
        agent = make_agent(script, tools=[EchoTool()], max_iterations=2)
        events = asyncio.run(_collect(agent.run_streaming("loop forever")))
        turns = [e for e in events if isinstance(e, TurnEnd)]
        assert turns[-1].reason == "max_iterations"
        assert agent.history[-1].role == "user"  # invariant intact


# ---- context management ------------------------------------------------------


class TestCompaction:
    def test_auto_compact_fires_through_the_async_seam(self):
        """Red-zone utilization at the seam must trigger an ASYNC compact:
        the summarizer pops scripted response #1 via provider.acomplete."""
        summary = assistant_text("COMPACT SUMMARY")
        script = [summary, assistant_text("answer after compaction")]
        agent = make_agent(script, context_window=4000, max_tokens=1000)

        # seed a transcript big enough to pin utilization at 1.0 AND long
        # enough to leave a summarizable middle beyond KEEP_TURNS=6
        filler = "x" * 20_000  # ~5k tokens by the chars/4 estimate
        agent.history.append(Message("user", [TextBlock("goal")]))
        for n in range(4):
            agent.history.append(Message(
                "assistant", [ToolCall(f"c{n}", "echo", {"t": filler})]))
            agent.history.append(Message("user", [ToolResult(f"c{n}", filler)]))
        agent.history.append(Message("assistant", [TextBlock("progress")]))

        async def _run():
            return await agent.run("finish this")

        response = asyncio.run(_run())
        assert response.message.text() == "answer after compaction"
        assert agent.last_compaction is not None
        assert agent.last_compaction["summarized"]
        assert "COMPACT SUMMARY" in agent.history[1].content[0].text


# ---- the width cap ------------------------------------------------------------


class TestWidthCap:
    def test_batch_runs_at_most_max_parallel_tools_wide(self):
        """All four calls become tasks AT ONCE (gather semantics preserved)
        but at most ``max_parallel_tools`` execute concurrently -- and the
        harvest still lands in submission order."""

        state = {"current": 0, "peak": 0}

        def _probe(name: str) -> Tool:
            class _T(SlowAsyncTool):
                pass

            async def arun(self, args, ctx):
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
                await asyncio.sleep(0.03)
                state["current"] -= 1
                return f"async done: {args['text']}"

            _T.name = name
            _T.arun = arun
            return _T(0.03)

        script = [
            ModelResponse(
                message=Message("assistant", [
                    TextBlock("four calls"),
                    *[ToolCall(id=f"c{i}", name=f"probe_{i}", arguments={"text": str(i)})
                      for i in range(4)],
                ]),
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            assistant_text("all done"),
        ]
        agent = make_agent(script, tools=[_probe(f"probe_{i}") for i in range(4)],
                           max_parallel_tools=2)

        events = asyncio.run(_collect(agent.run_streaming("fan out")))
        executed = [e for e in events if isinstance(e, ToolExecuted)]

        assert state["peak"] == 2, "width cap breached"
        assert [e.result.content for e in executed] == [
            f"async done: {i}" for i in range(4)]  # submission order intact


# ---- approve-with-edits (async twin of the sync contract) -------------------


class TestApproveWithEdits:
    def test_gate_edited_arguments_are_what_execute(self):
        def editing_gate(request):
            request.arguments = {"text": "edited"}  # the human's amendment
            return True

        script = [
            assistant_tool_call("call_1", "echo", {"text": "original"}),
            assistant_text("done"),
        ]
        agent = make_agent(script, tools=[EchoTool()], permissions=editing_gate)

        async def _run():
            return [event async for event in agent.run_streaming("use echo")]

        events = asyncio.run(_run())
        executed = [e for e in events if isinstance(e, ToolExecuted)]
        # The EDITED args ran, not the model's originals.
        assert executed[0].result.content == "echoed: edited"

    def test_summarize_reaches_the_request(self):
        seen = []

        def capturing_gate(request):
            seen.append(request)
            return True

        script = [
            assistant_tool_call("call_1", "echo", {"text": "hi"}),
            assistant_text("done"),
        ]
        agent = make_agent(script, tools=[EchoTool()],
                           permissions=capturing_gate)

        async def _run():
            return [event async for event in agent.run_streaming("use echo")]

        asyncio.run(_run())
        assert seen[0].summarize is not None
        assert seen[0].summarize({"text": "yo"}) == "echo 'yo'"
