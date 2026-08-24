"""Agent-loop tests, driven entirely by ScriptedProvider.

The loop is the heart of the harness, so these tests pin its CONTRACT:

* happy-path history shape (the shape both wire formats must encode)
* errors are data -- crashes/denials/unknown tools become is_error results
* the iteration cap stops cleanly with history still valid
* closing the generator mid-turn leaves history RESUMABLE (the invariant)

The invariant itself gets a dedicated checker used by several tests:

    every tool_call id that appears in an assistant message must have a
    matching ToolResult in a LATER user message -- otherwise the next
    request references an unanswered id and providers reject it.
"""

from __future__ import annotations

import time

import pytest

from conftest import (
    ScriptedProvider,
    assistant_text,
    assistant_tool_call,
)
from akshara.agent import Agent, ToolExecuted, TurnEnd
from akshara.permissions import yolo
from akshara.providers.base import collect
from akshara.tools.base import Tool, ToolRegistry
from akshara.types import (
    EndEvent,
    Message,
    ModelResponse,
    StartEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
    Usage,
)


# ---- fake tools -----------------------------------------------------------


class EchoTool(Tool):
    """Happy-path tool: returns its input."""

    name = "echo"
    description = "echo the text back"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
    read_only = True

    def summary(self, args, ctx):
        return f"echo({args.get('text')!r})"

    def run(self, args, ctx):
        return f"echo:{args.get('text', '')}"


class BombTool(Tool):
    """Crashes with a plain exception -- the loop must convert it to data."""

    name = "bomb"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    def summary(self, args, ctx):
        return "bomb()"

    def run(self, args, ctx):
        raise ValueError("boom")


class BadSummaryBomb(Tool):
    """Its SUMMARY crashes; execution would be fine.

    A broken summary must never take down the permission flow itself.
    """

    name = "bad_summary"
    description = "summary raises, run works"
    parameters = {"type": "object", "properties": {}}
    read_only = True

    def summary(self, args, ctx):
        raise RuntimeError("summary exploded")

    def run(self, args, ctx):
        return "ran fine"


def make_agent(script, *, tools=None, permissions=None, max_iterations=25) -> Agent:
    registry = ToolRegistry()
    for tool in tools or [EchoTool(), BombTool(), BadSummaryBomb()]:
        registry.register(tool)
    return Agent(
        ScriptedProvider(script),
        model="scripted-model",
        system="you are scripted",
        tools=registry,
        permissions=permissions,
        max_iterations=max_iterations,
    )


def last_user_results(agent: Agent) -> list[ToolResult]:
    """The most recent batch of tool results -- NOT history[-1], which is
    the final assistant answer once a turn completes normally."""
    for message in reversed(agent.history):
        if message.role == "user":
            results = [b for b in message.content if isinstance(b, ToolResult)]
            if results:
                return results
    raise AssertionError("no tool results anywhere in history")


def assert_history_resumable(agent: Agent) -> None:
    """THE INVARIANT: no tool_call id is left without a matching result."""
    open_ids: set[str] = set()
    for message in agent.history:
        match message.role:
            case "assistant":
                open_ids.update(call.id for call in message.tool_calls())
            case "user":
                for block in message.content:
                    if isinstance(block, ToolResult):
                        open_ids.discard(block.tool_call_id)
    assert not open_ids, f"unanswered tool calls: {sorted(open_ids)}"


def drain(agent: Agent, prompt="go"):
    """Consume a full turn; return (events, TurnEnd)."""
    events = list(agent.run_streaming(prompt))
    end = events[-1]
    assert isinstance(end, TurnEnd)
    return events, end


# ---- happy path ------------------------------------------------------------


def test_happy_path_history_shape():
    agent = make_agent([
        assistant_tool_call("call_1", "echo", {"text": "hi"}),
        assistant_text("done!"),
    ])
    events, end = drain(agent)

    # user -> assistant(tool_call) -> user(tool_result) -> assistant(final)
    assert [m.role for m in agent.history] == ["user", "assistant", "user", "assistant"]
    call = agent.history[1].tool_calls()[0]
    assert (call.id, call.name, call.arguments) == ("call_1", "echo", {"text": "hi"})
    (result,) = last_user_results(agent)
    assert result == ToolResult("call_1", "echo:hi")
    assert agent.history[3].text() == "done!"

    # event stream: exactly one ToolExecuted then TurnEnd(end_turn)
    executed = [e for e in events if isinstance(e, ToolExecuted)]
    assert len(executed) == 1
    assert end.reason == "end_turn" and end.iterations == 2
    assert end.response is not None
    assert_history_resumable(agent)


def test_stream_reconstruction_matches_scripted_response():
    """ScriptedProvider re-synthesizes stream events from each scripted
    response and the loop folds them with collect(). If EITHER half
    dropped or mangled data, the round-trip below fails."""
    scripted = assistant_tool_call(
        "c1", "echo", {"text": "hi", "n": 3}, text_before="checking..."
    )
    agent = make_agent([scripted, assistant_text("ok")])
    drain(agent)

    rebuilt = agent.history[1]
    assert rebuilt.role == "assistant"
    assert rebuilt.content == scripted.message.content


def test_request_kwargs_reach_provider():
    agent = make_agent([assistant_text("ok")])
    drain(agent, "hello")
    request = agent.provider.last_request()
    assert request["system"] == "you are scripted"
    assert request["model"] == "scripted-model"
    assert {"echo", "bomb", "bad_summary"} <= {s.name for s in request["tools"]}
    assert request["messages"][0].text() == "hello"


# ---- errors are data -------------------------------------------------------


@pytest.mark.parametrize("tool_name,args,expect_substr", [
    ("bomb", {}, "ValueError: boom"),          # tool crashed
    ("no_such_tool", {}, "no such tool"),      # registry miss
])
def test_failures_become_error_results_and_loop_continues(tool_name, args, expect_substr):
    agent = make_agent(
        [
            assistant_tool_call("c1", tool_name, args),
            assistant_text("recovered"),
        ],
        permissions=yolo,  # let the call REACH the tool; denial is tested elsewhere
    )
    _, end = drain(agent)

    (result,) = last_user_results(agent)
    assert result.tool_call_id == "c1"
    assert result.is_error is True
    assert expect_substr in result.content
    assert end.reason == "end_turn"
    assert agent.history[-1].text() == "recovered"
    assert_history_resumable(agent)


def test_denied_permission_is_data():
    agent = make_agent(
        [assistant_tool_call("c1", "echo", {}), assistant_text("understood")],
        permissions=lambda request: False,
    )
    _, end = drain(agent)

    (result,) = last_user_results(agent)
    assert result.is_error is True
    assert "denied" in result.content.lower()
    assert end.reason == "end_turn"


def test_failing_summary_does_not_block_approval_flow():
    """A crashing summary() falls back to a generic preview; the call runs."""
    agent = make_agent([
        assistant_tool_call("c1", "bad_summary", {}),
        assistant_text("fine"),
    ])
    drain(agent)

    (result,) = last_user_results(agent)
    assert result.is_error is False
    assert result.content == "ran fine"


def test_unparseable_arguments_become_visible_data():
    """collect() on malformed argument fragments must NOT crash: the call
    keeps its raw JSON under a sentinel key the executor will report."""
    broken = [
        StartEvent(model="m"),
        ToolCallStart(index=0, id="x", name="echo"),
        ToolCallDelta(index=0, partial_json='{"te'),
        ToolCallDelta(index=0, partial_json='xt": '),  # invalid JSON overall
        EndEvent(stop_reason="tool_use", usage=Usage()),
    ]
    response = collect(broken)
    assert response.stop_reason == "tool_use"
    (call,) = response.message.tool_calls()
    assert "_unparseable_json" in call.arguments


# ---- the invariant: resumable history ---------------------------------------


def test_iteration_cap_ends_with_valid_history():
    script = [
        assistant_tool_call(f"c{i}", "echo", {"text": str(i)}) for i in (1, 2)
    ]
    agent = make_agent([*script, assistant_text("never reached")], max_iterations=2)
    _, end = drain(agent)

    assert end.reason == "max_iterations"
    assert end.iterations == 2 and end.response is None
    # Both completed batches were answered normally...
    answered = [r.content for r in last_user_results(agent)]
    assert answered == ["echo:2"]  # trailing batch is the second one
    # ...and the loop stopped CALLING the model: the final scripted
    # response is still unconsumed.
    assert len(agent.provider.script) == 1
    assert_history_resumable(agent)


def _two_calls_response() -> ModelResponse:
    return ModelResponse(
        message=Message("assistant", [
            ToolCall(id="a", name="echo", arguments={"text": "a"}),
            ToolCall(id="b", name="bomb", arguments={}),
        ]),
        stop_reason="tool_use",
        usage=Usage(),
    )


def test_close_during_yield_replays_whole_batch_faithfully():
    """Close while suspended on a ToolExecuted yield. Since batches now
    execute FULLY before yielding (parallel execution, order-preserving),
    every call has a REAL result -- no synthesized interrupts needed."""
    agent = make_agent([_two_calls_response(), assistant_text("ok")])
    gen = agent.run_streaming("go")
    first = next(gen)
    assert isinstance(first, ToolExecuted) and first.call.id == "a"
    assert first.result.content == "echo:a"

    gen.close()

    results = {r.tool_call_id: r for r in last_user_results(agent)}
    assert results["a"] == ToolResult("a", "echo:a")
    # 'b' is bomb -- NOT read_only, so the default allow_read_only gate
    # denied it during the up-front gating pass. That denial IS its real
    # result; batches gate+execute fully before yielding anything.
    assert results["b"] == ToolResult("b", "Permission denied by user.",
                                      is_error=True)
    assert_history_resumable(agent)


# ---- parallel batches -------------------------------------------------------


class SlowEcho(Tool):
    """Sleeps then echoes -- completion order differs from submission."""

    read_only = True

    def __init__(self, name: str, delay: float, label: str) -> None:
        self.name = name
        self.description = f"sleep {delay}s then echo {label}"
        self.parameters = {"type": "object", "properties": {}}
        self.delay = delay
        self.label = label

    def summary(self, args, ctx):
        return f"{self.name}()"

    def run(self, args, ctx):
        time.sleep(self.delay)
        return self.label


def _multi_call_response(pairs: list[tuple[str, str]]) -> ModelResponse:
    """One assistant message requesting several calls at once."""
    return ModelResponse(
        message=Message(
            "assistant",
            [ToolCall(id=cid, name=name, arguments={}) for cid, name in pairs],
        ),
        stop_reason="tool_use",
        usage=Usage(),
    )


def test_parallel_results_align_with_submission_order():
    """Fast-then-slow submission must still pair results correctly even
    though the fast one finishes first."""
    slow = SlowEcho("slow_echo", 0.25, "SLOW")
    fast = SlowEcho("fast_echo", 0.01, "FAST")
    agent = make_agent([
        _multi_call_response([("c_slow", "slow_echo"), ("c_fast", "fast_echo")]),
        assistant_text("done"),
    ], tools=[slow, fast])

    _, end = drain(agent)

    results = {r.tool_call_id: r.content for r in last_user_results(agent)}
    assert results == {"c_slow": "SLOW", "c_fast": "FAST"}
    assert end.reason == "end_turn"
    assert_history_resumable(agent)


def test_approved_batch_runs_concurrently():
    """Three 0.15s tools should take ~0.15s wall-clock, not ~0.45s."""
    tools = [SlowEcho(f"t{i}", 0.15, f"out{i}") for i in range(3)]
    agent = make_agent([
        _multi_call_response([(f"c{i}", f"t{i}") for i in range(3)]),
        assistant_text("done"),
    ], tools=tools)

    start = time.monotonic()
    drain(agent)
    elapsed = time.monotonic() - start

    assert elapsed < 0.40, f"batch looks sequential: {elapsed:.2f}s"


def test_denied_calls_stay_inline_in_a_parallel_batch():
    """Gating happens sequentially BEFORE execution; denials slot into
    their positional result without disturbing approved siblings."""
    tools = [SlowEcho("allow_me", 0.01, "ran"), EchoTool()]
    agent = make_agent([
        _multi_call_response([("c1", "allow_me"), ("c2", "echo")]),
        assistant_text("done"),
    ], tools=tools,
        permissions=lambda req: req.tool_name == "allow_me")

    drain(agent)

    results = {r.tool_call_id: r for r in last_user_results(agent)}
    assert results["c1"].content == "ran" and not results["c1"].is_error
    assert results["c2"].is_error and "denied" in results["c2"].content.lower()
    assert_history_resumable(agent)


class KIBomb(Tool):
    """Its run raises KeyboardInterrupt -- like cancellation arriving
    inside a worker thread rather than the main one."""

    name = "ki_bomb"
    description = "raises KeyboardInterrupt"
    parameters = {"type": "object", "properties": {}}
    read_only = True

    def summary(self, args, ctx):
        return "ki_bomb()"

    def run(self, args, ctx):
        raise KeyboardInterrupt


def test_interrupt_mid_batch_records_real_sibling_results():
    """KI while waiting on the batch pool: workers finish during shutdown
    join, so siblings get their REAL results recorded (not synthetic
    interrupts) before the cancellation propagates."""
    agent = make_agent(
        [_multi_call_response([("a", "echo"), ("b", "ki_bomb")])],
        tools=[EchoTool(), KIBomb()],
    )

    with pytest.raises(KeyboardInterrupt):
        list(agent.run_streaming("go"))

    results = {r.tool_call_id: r for r in last_user_results(agent)}
    assert results["a"] == ToolResult("a", "echo:")    # real, faithful
    assert results["b"].is_error is True               # KIBomb's own slot
    assert_history_resumable(agent)


def test_close_mid_stream_leaves_history_resumable():
    """Ctrl-C arriving DURING the model pull: no calls executed yet, so
    every call in the trailing assistant message gets synthesized errors."""

    class InterruptingProvider(ScriptedProvider):
        def stream(self, **kwargs):
            yield from super().stream(**kwargs)
            raise KeyboardInterrupt  # like a real ctrl-c mid-stream

    agent = make_agent([_two_calls_response()])
    agent.provider.__class__ = InterruptingProvider

    with pytest.raises(KeyboardInterrupt):
        list(agent.run_streaming("go"))

    # collect() never returned, so the assistant message never landed:
    # history holds only the user turn -- trivially resumable, and the
    # next request references no tool ids at all.
    assert [m.role for m in agent.history] == ["user"]
    assert_history_resumable(agent)


# ---- accounting -------------------------------------------------------------


def test_usage_accumulates_across_iterations():
    first = assistant_tool_call("c1", "echo", {})
    first.usage = Usage(input_tokens=10, output_tokens=5)
    second = assistant_text("done", usage=Usage(input_tokens=7, output_tokens=3))
    agent = make_agent([first, second])
    drain(agent)

    assert agent.total_usage.input_tokens == 17
    assert agent.total_usage.output_tokens == 8


def test_run_returns_final_response_or_raises():
    agent = make_agent([assistant_text("plain answer")])
    response = agent.run("hi")
    assert response.stop_reason == "end_turn"
    assert response.message.text() == "plain answer"

    capped = make_agent(
        [assistant_tool_call("c1", "echo", {})] * 2, max_iterations=1
    )
    with pytest.raises(RuntimeError, match="max_iterations"):
        capped.run("hi")


def test_text_deltas_split_across_fragments_are_joined():
    """ScriptedProvider halves every TextBlock into two deltas; the loop's
    collect() must join them back into one block."""
    agent = make_agent([assistant_text("abcdef")])
    drain(agent)
    assert agent.history[1].content[0].text == "abcdef"


def test_stream_events_push_to_subscriber_during_collect():
    """Raw StreamEvents cannot be yielded from run_streaming (collect()
    owns the pull), so they are PUSHED to on_stream_event. A UI sets
    agent.on_stream_event = renderer; this test asserts the full pushed
    sequence across a two-iteration turn."""
    pushed: list[str] = []
    agent = make_agent([
        assistant_tool_call("c1", "echo", {"text": "hi"}, text_before="let me"),
        assistant_text("done"),
    ])
    agent.on_stream_event = lambda e: pushed.append(type(e).__name__)

    list(agent.run_streaming("go"))

    # iteration 1: start, text, tool start+delta, end -- then iteration 2
    assert pushed == [
        "StartEvent", "TextDelta", "TextDelta",
        "ToolCallStart", "ToolCallDelta", "EndEvent",
        "StartEvent", "TextDelta", "TextDelta", "EndEvent",
    ]
