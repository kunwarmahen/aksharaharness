"""ask_user tests: the human-in-the-loop tool, driven by ScriptedProvider.

Pinned contracts:

* the answer comes back as a tool-result string the model can act on
* args validation raises ToolError (data) the model can correct
* NO channel attached => UserUnavailable (BaseException!) escapes
  run_streaming AND history stays resumable -- the strict headless rule
* read_only: asking never gates; deny_all still turns it into data
* a mixed batch records the batch-mates' real results before the
  UserUnavailable surfaces (the _execute_batch cancel path)
* TerminalChannel: choices select, free text passes through, empty
  re-prompts, EOF becomes UserUnavailable
"""

from __future__ import annotations

import pytest

from conftest import ScriptedProvider, assistant_text, assistant_tool_call
from akshara.agent import Agent, ToolExecuted, TurnEnd
from akshara.errors import ToolError, UserUnavailable
from akshara.permissions import allow_read_only, deny_all
from akshara.tools.ask_user import AskUser, TerminalChannel
from akshara.tools.base import Tool, ToolRegistry
from akshara.types import ModelResponse, ToolCall, ToolResult


def make_agent(script: list[ModelResponse], channel, *,
               permissions=allow_read_only) -> Agent:
    registry = ToolRegistry()
    registry.register(AskUser(channel))
    return Agent(ScriptedProvider(script), model="m", tools=registry,
                 permissions=permissions)


def tool_results(agent: Agent) -> dict[str, str]:
    """tool_call_id -> result content, from history."""
    out = {}
    for message in agent.history:
        if message.role != "user":
            continue
        for block in message.content:
            if isinstance(block, ToolResult):
                out[block.tool_call_id] = block.content
    return out


def history_is_valid(agent: Agent) -> bool:
    """The invariant: every tool_call id has a matching later ToolResult."""
    pending: set[str] = set()
    for message in agent.history:
        if message.role == "assistant":
            pending.update(c.id for c in message.tool_calls())
        else:
            for block in message.content:
                if isinstance(block, ToolResult):
                    pending.discard(block.tool_call_id)
    return not pending


# ---- fake channels ---------------------------------------------------------


class FakeChannel:
    """Canned answers; records questions (and context) for assertions."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, list[str], str]] = []

    def ask(self, question: str, choices: list[str],
            context: str = "") -> str:
        self.asked.append((question, list(choices), context))
        if not self.answers:
            raise AssertionError("FakeChannel ran dry")
        return self.answers.pop(0)


class EchoTool(Tool):
    name = "echo"
    description = "echo text back"
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string"}}}
    read_only = True

    def summary(self, args, ctx):
        return f"echo({args.get('text')!r})"

    def run(self, args, ctx):
        return f"echo:{args.get('text', '')}"


# ---- happy paths ------------------------------------------------------------


def test_free_text_answer_reaches_history():
    channel = FakeChannel("SQLite, it's a single-user tool")
    agent = make_agent([
        assistant_tool_call("a1", "ask_user",
                            {"question": "Which storage backend?"}),
        assistant_text("done -- using SQLite"),
    ], channel)

    events = list(agent.run_streaming("build me a CLI"))
    assert isinstance(events[-1], TurnEnd)
    assert events[-1].reason == "end_turn"

    content = tool_results(agent)["a1"]
    assert "SQLite, it's a single-user tool" in content
    assert channel.asked == [("Which storage backend?", [], "")]
    assert history_is_valid(agent)


def test_choice_answer_carries_the_pick_marker():
    channel = FakeChannel("Postgres")
    agent = make_agent([
        assistant_tool_call("a1", "ask_user", {
            "question": "Which storage backend?",
            "choices": ["SQLite", "Postgres"],
        }),
        assistant_text("ok"),
    ], channel)

    list(agent.run_streaming("go"))
    assert tool_results(agent)["a1"] == \
        "user replied: (picked option 2/2) Postgres"


def test_context_field_is_accepted_and_shown_to_human():
    channel = FakeChannel("yes")
    agent = make_agent([
        assistant_tool_call("a1", "ask_user", {
            "question": "May I delete build/ ?",
            "context": "it holds stale artifacts; regenerable",
        }),
        assistant_text("done"),
    ], channel)

    list(agent.run_streaming("clean up"))
    # context rides in the schema only; the question is what the human sees
    assert channel.asked[0][0] == "May I delete build/ ?"


# ---- validation: errors are data -------------------------------------------


@pytest.mark.parametrize("args,fragment", [
    ({}, "question"),
    ({"question": "   "}, "question"),
    ({"question": "q", "choices": "SQLite"}, "list"),
    ({"question": "q", "choices": [1, 2]}, "non-empty string"),
    ({"question": "q", "choices": ["a"] * 7}, "max"),
])
def test_bad_args_become_error_data_not_exceptions(args, fragment):
    channel = FakeChannel("unused")
    agent = make_agent([
        assistant_tool_call("a1", "ask_user", args),
        assistant_text("recovered"),
    ], channel)

    events = [e for e in agent.run_streaming("go") if isinstance(e, ToolExecuted)]
    assert fragment in events[0].result.content
    assert events[0].result.is_error
    assert history_is_valid(agent)


# ---- headless: fail the turn, keep history resumable ------------------------


def test_no_channel_fails_the_turn_loudly():
    agent = make_agent([
        assistant_tool_call("a1", "ask_user", {"question": "which one?"}),
        # never reached -- the turn dies at the ask
    ], channel=None)

    with pytest.raises(UserUnavailable):
        list(agent.run_streaming("go"))

    # the ask got a synthesized error result; history stays valid so the
    # session survives and the NEXT turn is a well-formed request.
    assert history_is_valid(agent)
    assert "interrupted" in tool_results(agent)["a1"].lower()

    provider = agent.provider
    provider.script.append(assistant_text("fresh turn works"))
    events = list(agent.run_streaming("never mind"))
    assert events[-1].reason == "end_turn"


def test_mixed_batch_records_batchmates_before_failing():
    """ask_user + echo in ONE batch: echo's real result must land in history
    even though its batch-mate blew up -- work happened, history is faithful."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(AskUser(None))
    scripted_call = assistant_tool_call("e1", "echo", {"text": "hi"})
    scripted_call.message.content.append(
        ToolCall(id="u1", name="ask_user", arguments={"question": "proceed?"}))
    agent = Agent(ScriptedProvider([scripted_call]), model="m",
                  tools=registry, permissions=allow_read_only)

    with pytest.raises(UserUnavailable):
        list(agent.run_streaming("go"))

    results = tool_results(agent)
    assert results["e1"] == "echo:hi"          # real result preserved
    # The ask itself died mid-batch, so the cancel path records WHAT killed
    # it -- more faithful than the plain interrupted placeholder.
    assert "userunavailable" in results["u1"].lower()
    assert history_is_valid(agent)


# ---- permission interplay ---------------------------------------------------


def test_read_only_ask_runs_without_prompting_under_allow_read_only():
    channel = FakeChannel("blue")
    agent = make_agent([
        assistant_tool_call("a1", "ask_user", {"question": "color?"}),
        assistant_text("nice"),
    ], channel, permissions=allow_read_only)
    list(agent.run_streaming("go"))
    assert "blue" in tool_results(agent)["a1"]


def test_deny_all_turns_an_ask_into_data_like_any_tool():
    channel = FakeChannel("should never be called")
    agent = make_agent([
        assistant_tool_call("a1", "ask_user", {"question": "color?"}),
        assistant_text("understood, moving on"),
    ], channel, permissions=deny_all)
    events = [e for e in agent.run_streaming("go") if isinstance(e, ToolExecuted)]
    assert events[0].result.is_error
    assert channel.answers  # untouched -- the channel was never asked


# ---- TerminalChannel ---------------------------------------------------------


def test_terminal_channel_choice_by_number():
    lines = iter(["2"])
    channel = TerminalChannel(input_fn=lambda _p: next(lines))
    assert channel.ask("pick", ["red", "blue"]) == "blue"


def test_terminal_channel_free_text_passes_through():
    lines = iter(["something else entirely"])
    channel = TerminalChannel(input_fn=lambda _p: next(lines))
    assert channel.ask("pick", ["red", "blue"]) == "something else entirely"


def test_terminal_channel_empty_reprompts_then_answers():
    lines = iter(["", "  ", "final answer"])
    channel = TerminalChannel(input_fn=lambda _p: next(lines))
    assert channel.ask("pick", []) == "final answer"


def test_terminal_channel_eof_becomes_user_unavailable():
    def eof(_p):
        raise EOFError
    channel = TerminalChannel(input_fn=eof)
    with pytest.raises(UserUnavailable):
        channel.ask("anyone there?", [])


def test_terminal_channel_out_of_range_number_treated_as_text():
    # "9" isn't a valid option number, so it passes through literally --
    # the human may genuinely want to answer "9".
    lines = iter(["9"])
    channel = TerminalChannel(input_fn=lambda _p: next(lines))
    assert channel.ask("pick a file (1-3)", ["a", "b", "c"]) == "9"
