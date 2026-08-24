"""Sub-agents: agent-as-tool with the book's three constraints.

The scripted provider serves ONE queue shared by parent and child --
calls happen sequentially (parent request -> child's N requests ->
parent continues), so the script order is fully deterministic. The
tests pin the constraints in code, not prompts: one level deep,
bounded spawns, catalog-level scope enforcement, compact results,
fresh child context.
"""

from __future__ import annotations

import pytest

from conftest import ScriptedProvider, assistant_text, assistant_tool_call

from akshara.agent import Agent
from akshara.errors import ProviderError, ToolError
from akshara.permissions import yolo
from akshara.subagent import (
    EXECUTE_CLAUSE,
    SPAWN_TOOL_NAME,
    CHILD_SYSTEM_TEMPLATE,
    SpawnSubagent,
    SubagentSpawner,
)
from akshara.tools.fs import ReadFile
from akshara.types import ModelResponse, Usage

VALID_ARGS = {
    "objective": "Find which file defines the agent loop.",
    "output_format": "One line: 'path -> what it defines'.",
    "tools_allowed": ["read_file"],
    "justification": "Isolated search; parent context stays clean.",
}


def _spawn_response(**extra) -> ModelResponse:
    return assistant_tool_call("call_1", SPAWN_TOOL_NAME, {**VALID_ARGS, **extra})


def make_agent(script: list[ModelResponse], *,
               max_per_session: int = 5) -> tuple[Agent, SubagentSpawner]:
    provider = ScriptedProvider(script)
    # yolo, deliberately: spawn is read_only=False, so allow_read_only
    # would DENY it -- (that gate behavior has its own tests elsewhere).
    agent = Agent(provider, model="m", permissions=yolo)
    agent.registry.register(ReadFile())
    spawner = SubagentSpawner(agent, max_per_session=max_per_session)
    agent.registry.register(SpawnSubagent(spawner))
    return agent, spawner


class TestValidation:
    def test_justification_required(self):
        _, spawner = make_agent([])
        with pytest.raises(ToolError, match="justification"):
            spawner.spawn({**VALID_ARGS, "justification": ""})

    def test_output_format_required(self):
        _, spawner = make_agent([])
        with pytest.raises(ToolError, match="output_format"):
            spawner.spawn({**VALID_ARGS, "output_format": " "})

    def test_tools_allowed_must_be_non_empty_and_known(self):
        _, spawner = make_agent([])
        with pytest.raises(ToolError, match="NON-EMPTY"):
            spawner.spawn({**VALID_ARGS, "tools_allowed": []})
        with pytest.raises(ToolError, match="unknown tool"):
            spawner.spawn({**VALID_ARGS, "tools_allowed": ["time_travel"]})

    def test_one_level_deep_enforced_in_code_not_prompts(self):
        _, spawner = make_agent([])
        with pytest.raises(ToolError, match="cannot spawn sub-agents"):
            spawner.spawn({**VALID_ARGS, "tools_allowed": [SPAWN_TOOL_NAME]})


class TestBudget:
    def test_exhaustion_is_a_tool_error_the_model_can_read(self):
        _, spawner = make_agent([assistant_text("a"), assistant_text("b")],
                                max_per_session=2)
        spawner.spawn(VALID_ARGS)
        spawner.spawn(VALID_ARGS)
        with pytest.raises(ToolError, match="budget exhausted"):
            spawner.spawn(VALID_ARGS)

    def test_failed_spawn_still_consumes_budget(self):
        # counted when attempted: otherwise models could probe for free
        _, spawner = make_agent([assistant_text("a")], max_per_session=1)
        spawner.spawn(VALID_ARGS)
        with pytest.raises(ToolError, match="budget"):
            spawner.spawn(VALID_ARGS)


class TestChildRun:
    def test_fresh_context_filtered_catalog_compact_return(self):
        script = [
            _spawn_response(),                              # parent spawns
            assistant_text("src/akshara/agent.py -> the loop",
                           usage=Usage(input_tokens=10, output_tokens=5)),
            assistant_text("the loop lives in src/akshara/agent.py"),
        ]
        agent, spawner = make_agent(script)
        provider = agent.provider

        response = agent.run("Where is the loop defined?")

        parent_initial, child_only, parent_final = provider.requests

        # FRESH CONTEXT: the child saw exactly one user message -- the
        # objective -- and none of the parent's conversation leaked in
        assert len(child_only["messages"]) == 1
        assert VALID_ARGS["objective"] in str(child_only["messages"])
        assert "Where is the loop defined?" not in str(child_only["messages"])

        # CATALOG-LEVEL SCOPING: the child physically has only read_file...
        assert [s.name for s in child_only["tools"]] == ["read_file"]
        # ...while the parent keeps its full set including the spawn tool
        assert SPAWN_TOOL_NAME in [s.name for s in parent_final["tools"]]

        # child system prompt carries format + mandatory-execute clause
        assert "MUST call at least one" in child_only["system"]
        assert VALID_ARGS["output_format"] in child_only["system"]

        # COMPACT RESULT: parent's history holds the report + cost meta,
        # never the child's transcript
        batch = next(m for m in agent.history if m.role == "user"
                     and any(getattr(b, "tool_call_id", None) == "call_1"
                             for b in m.content))
        content = batch.content[0].content
        assert "src/akshara/agent.py -> the loop" in content  # the report...
        content = batch.content[0].content
        assert "src/akshara/agent.py -> the loop" in content  # the report...
        assert "[sub-agent ·" in content                     # ...plus cost meta
        assert "10in/5out tokens]" in content
        assert spawner.spawned == 1

    def test_narration_guard_in_system_prompt(self):
        objective, fmt, allowed, max_it = None, None, None, None
        _, spawner = make_agent([])
        objective, fmt, allowed, max_it = spawner._validate(VALID_ARGS)
        system = CHILD_SYSTEM_TEMPLATE.format(
            objective=objective, output_format=fmt,
            tool_names=", ".join(allowed),
            execute_clause=EXECUTE_CLAUSE, max_iterations=max_it)
        assert "MUST call at least one" in system
        assert "SAY SO explicitly" in system


class TestChildFailureModes:
    def test_provider_error_inside_child_becomes_data(self):
        """A terminal provider failure in the CHILD must not kill the
        PARENT's turn -- errors are data all the way up."""

        class ExplodingForChildren(ScriptedProvider):
            def stream(self, **kwargs):
                system = kwargs.get("system") or ""
                if "sub-agent" in system:
                    raise ProviderError("503: upstream dead", status=503)
                yield from super().stream(**kwargs)

        provider = ExplodingForChildren([
            _spawn_response(),
            assistant_text("continuing without the sub-agent"),
        ])
        agent = Agent(provider, model="m", permissions=yolo)
        agent.registry.register(ReadFile())
        agent.registry.register(SpawnSubagent(SubagentSpawner(agent)))

        response = agent.run("q")  # must NOT raise

        assert "continuing without" in response.message.text()
        # the parent saw an is_error result explaining what happened
        batch = next(m for m in agent.history if m.role == "user"
                     and any(getattr(b, "is_error", False) for b in m.content))
        assert "provider error inside sub-agent" in batch.content[0].content

    def test_iteration_cap_salvages_partial_answer(self):
        script = [
            # child capped at TWO iterations; it burns both on tool calls
            _spawn_response(max_iterations=2),
            assistant_tool_call("c1", "read_file", {"path": "a.txt"},
                                text_before="checking a.txt first..."),
            assistant_tool_call("c2", "read_file", {"path": "b.txt"}),
            assistant_text("wrapping up without the child's full answer"),
        ]
        agent, spawner = make_agent(script)

        agent.run("q")

        (result,) = spawner.results
        assert result.error and "iteration cap" in result.error
        # best-effort salvage: the child's last prose survives
        assert result.summary == "checking a.txt first..."
        batch = next(m for m in agent.history if m.role == "user"
                     and any(getattr(b, "tool_call_id", None) == "call_1"
                             for b in m.content))
        assert "[INCOMPLETE" in batch.content[0].content


class TestStreamTee:
    """on_child_event: a UI seam -- child StreamEvents pushed out live,
    tagged with the spawn number. The parent's context never sees any of
    it (these tests double as the proof that teeing changes nothing)."""

    def test_child_events_pushed_live_and_numbered(self):
        script = [
            _spawn_response(),
            assistant_text("the loop lives in agent.py",
                           usage=Usage(input_tokens=10, output_tokens=5)),
            assistant_text("done"),
        ]
        agent, spawner = make_agent(script)
        seen: list[tuple[int, str]] = []
        spawner.on_child_event = lambda n, ev: seen.append((n, type(ev).__name__))

        response = agent.run("q")

        kinds = [kind for _, kind in seen]
        assert kinds[0] == "StartEvent"
        assert "TextDelta" in kinds          # streamed as fragments, not at end
        assert kinds[-1] == "EndEvent"
        assert {n for n, _ in seen} == {1}   # first spawn is number 1
        assert "done" in response.message.text()  # parent turn unaffected

    def test_each_spawn_gets_its_own_number(self):
        script = [
            _spawn_response(),
            assistant_text("first child",
                           usage=Usage(input_tokens=1, output_tokens=1)),
            assistant_tool_call("call_2", SPAWN_TOOL_NAME, VALID_ARGS),
            assistant_text("second child",
                           usage=Usage(input_tokens=1, output_tokens=1)),
            assistant_text("done"),
        ]
        agent, spawner = make_agent(script)
        seen: set[int] = set()
        spawner.on_child_event = lambda n, ev: seen.add(n)

        agent.run("q")

        assert seen == {1, 2}

    def test_unset_tee_changes_nothing(self):
        # default stays silent: sub-agent internals stream nowhere
        script = [
            _spawn_response(),
            assistant_text("child report",
                           usage=Usage(input_tokens=10, output_tokens=5)),
            assistant_text("done"),
        ]
        agent, spawner = make_agent(script)
        assert spawner.on_child_event is None

        response = agent.run("q")

        # the compact result still reached the parent's history
        batch = next(m for m in agent.history if m.role == "user"
                     and any(getattr(b, "tool_call_id", None) == "call_1"
                             for b in m.content))
        assert "child report" in batch.content[0].content
        assert "done" in response.message.text()
        assert spawner.spawned == 1

    def test_cli_helper_registers_tool_and_tee(self):
        from rich.console import Console

        from akshara.cli.main import enable_subagents

        # a plain agent -- enable_subagents does the whole wiring itself
        agent = Agent(ScriptedProvider([]), model="m", permissions=yolo)
        spawner = enable_subagents(agent, Console())

        assert SPAWN_TOOL_NAME in agent.registry.names()
        assert isinstance(spawner, SubagentSpawner)
        assert spawner.on_child_event is not None
