"""Tool hooks: observational before/after callbacks around execution.

The pinned contract (identical for both loop twins):

* hooks bracket every REAL execution -- before fires as the call
  starts, after fires with the wrapped result, error results included
* GATES DECIDE, HOOKS WATCH: denied calls and unknown tools never
  execute, so they are never observed; no hook can veto anything
* a raising hook crashes the turn LOUDLY -- hooks are developer
  infrastructure (logging/metrics/audit), not untrusted input, so
  their failures must NOT be laundered into model-visible data -- and
  the history invariant still holds afterward
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import ScriptedProvider, assistant_text, assistant_tool_call
from akshara.agent import Agent
from akshara.async_agent import AsyncAgent
from akshara.permissions import deny_all, yolo
from akshara.tools.base import Tool, ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "echo the text back"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
    read_only = True

    def summary(self, args, ctx):
        return f"echo({args.get('text')!r})"

    def run(self, args, ctx):
        return f"echo:{args.get('text', '')}"


class BombTool(Tool):
    name = "bomb"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    def summary(self, args, ctx):
        return "bomb()"

    def run(self, args, ctx):
        raise ValueError("boom")


class Recorder:
    """Collects hook invocations into parallel structures plus one
    ordered transcript of before/after kinds."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.befores: list[tuple[str, str]] = []       # (name, call id)
        self.afters: list[tuple[str, str, bool]] = []  # (call id, content, error)

    def before(self, call) -> None:
        self.order.append("before")
        self.befores.append((call.name, call.id))

    def after(self, call, result) -> None:
        self.order.append("after")
        self.afters.append((call.id, result.content, result.is_error))

    @property
    def empty(self) -> bool:
        return not self.order


def make_sync(script, *, tools=None, permissions=None, recorder=None):
    registry = ToolRegistry()
    for tool in tools or [EchoTool(), BombTool()]:
        registry.register(tool)
    agent = Agent(ScriptedProvider(script), model="m", tools=registry,
                  permissions=permissions or yolo)
    if recorder is not None:
        agent.on_before_tool = recorder.before
        agent.on_after_tool = recorder.after
    return agent


def make_async(script, *, tools=None, permissions=None, recorder=None):
    registry = ToolRegistry()
    for tool in tools or [EchoTool(), BombTool()]:
        registry.register(tool)
    agent = AsyncAgent(ScriptedProvider(script), model="m", tools=registry,
                       permissions=permissions or yolo)
    if recorder is not None:
        agent.on_before_tool = recorder.before
        agent.on_after_tool = recorder.after
    return agent


class TestSyncHooks:
    def test_hooks_bracket_a_real_execution_in_order(self):
        rec = Recorder()
        agent = make_sync([
            assistant_tool_call("t1", "echo", {"text": "hi"}),
            assistant_text("done"),
        ], recorder=rec)

        events = list(agent.run_streaming("go"))

        assert rec.order == ["before", "after"]
        assert rec.befores == [("echo", "t1")]
        assert rec.afters == [("t1", "echo:hi", False)]
        # the observation happened even though this consumer only pulled
        # events lazily -- push channel vs pull channel are independent
        assert any(getattr(e, "call", None) for e in events)

    def test_error_results_still_reach_the_after_hook(self):
        rec = Recorder()
        agent = make_sync([
            assistant_tool_call("t1", "bomb", {}),
            assistant_text("recovered"),
        ], recorder=rec)

        list(agent.run_streaming("go"))

        assert rec.order == ["before", "after"]  # a crash is observed too
        call_id, content, is_error = rec.afters[0]
        assert call_id == "t1"
        assert is_error and "ValueError" in content and "boom" in content

    def test_denied_calls_are_never_observed(self):
        rec = Recorder()
        agent = make_sync([
            assistant_tool_call("t1", "echo", {"text": "x"}),
            assistant_text("ok"),
        ], permissions=deny_all, recorder=rec)

        list(agent.run_streaming("go"))

        assert rec.empty  # gates decide; nothing executed, nothing seen

    def test_unknown_tools_are_never_observed(self):
        rec = Recorder()
        agent = make_sync([
            assistant_tool_call("t1", "no_such_tool", {}),
            assistant_text("ok"),
        ], recorder=rec)

        list(agent.run_streaming("go"))

        assert rec.empty

    def test_batch_observes_each_call_exactly_once(self):
        rec = Recorder()
        agent = make_sync([
            assistant_tool_call("t1", "echo", {"text": "a"}),
            assistant_tool_call("t2", "echo", {"text": "b"}),
            assistant_text("both done"),
        ], recorder=rec)

        list(agent.run_streaming("go"))

        assert sorted(i for _, i in rec.befores) == ["t1", "t2"]
        assert sorted(i for i, _, _ in rec.afters) == ["t1", "t2"]

    def test_raising_hook_crashes_loudly_and_history_stays_resumable(self):
        def exploding_before(call):
            raise RuntimeError("hook infrastructure failed")

        agent = make_sync([
            assistant_tool_call("t1", "echo", {"text": "x"}),
            assistant_text("never reached"),
        ])
        agent.on_before_tool = exploding_before

        with pytest.raises(RuntimeError, match="hook infrastructure"):
            list(agent.run_streaming("go"))

        from test_agent_loop import assert_history_resumable
        assert_history_resumable(agent)  # loud failure, valid history


class TestAsyncHooks:
    def test_parity_bracket_order_and_payload(self):
        rec = Recorder()

        async def _turn():
            agent = make_async([
                assistant_tool_call("t1", "echo", {"text": "hi"}),
                assistant_text("done"),
            ], recorder=rec)
            async for _ in agent.run_streaming("go"):
                pass

        asyncio.run(_turn())
        assert rec.order == ["before", "after"]
        assert rec.befores == [("echo", "t1")]
        assert rec.afters == [("t1", "echo:hi", False)]

    def test_async_denied_calls_are_never_observed(self):
        rec = Recorder()

        async def _turn():
            agent = make_async([
                assistant_tool_call("t1", "echo", {"text": "x"}),
                assistant_text("ok"),
            ], permissions=deny_all, recorder=rec)
            async for _ in agent.run_streaming("go"):
                pass

        asyncio.run(_turn())
        assert rec.empty

    def test_async_raising_hook_crashes_loudly_history_intact(self):
        from test_agent_loop import assert_history_resumable

        async def _crash_and_check():
            agent = make_async([
                assistant_tool_call("t1", "echo", {"text": "x"}),
                assistant_text("never reached"),
            ])

            def boom(call, result):
                raise RuntimeError("hook blew up")

            agent.on_after_tool = boom
            with pytest.raises(RuntimeError, match="hook blew up"):
                async for _ in agent.run_streaming("go"):
                    pass
            return agent

        agent = asyncio.run(_crash_and_check())
        assert_history_resumable(agent)
