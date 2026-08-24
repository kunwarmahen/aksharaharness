"""Tool-selection tests: BM25 ranking, pins, the discovery hatch, and
the loop-level contract (selected specs are what get SENT and the ONLY
things that can EXECUTE).
"""

from __future__ import annotations

import json

import pytest

from akshara.agent import Agent
from akshara.permissions import yolo
from akshara.tools.base import Tool, ToolRegistry
from akshara.tools.selector import (
    AUTO_SELECTION_THRESHOLD,
    ListAvailableTools,
    ToolCatalog,
    enable_selection,
    query_from_transcript,
)
from akshara.types import (Message, ModelResponse, TextBlock, TextDelta,
                           ToolCall, ToolResult, Usage)

from conftest import ScriptedProvider


def _mk(name: str, description: str) -> Tool:
    attrs = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {},
                       "additionalProperties": False},
        "read_only": True,
        "summary": lambda self, args, ctx: name,
        "run": lambda self, args, ctx: f"{name} ran",
    }
    return type(name.title().replace("_", ""), (Tool,), attrs)()


def _catalog(n_fillers: int = 36) -> ToolCatalog:
    tools = [
        _mk("mcp__slack__post_message", "send a message to a slack channel"),
        _mk("mcp__slack__list_channels", "list slack channels in a workspace"),
        _mk("mcp__db__run_query", "execute sql against a postgres database"),
        _mk("mcp__db__list_tables", "show tables in the database schema"),
    ]
    tools += [_mk(f"filler_{i}", f"misc utility number {i}") for i in range(n_fillers)]
    return ToolCatalog(tools)


# ---- ranking -----------------------------------------------------------------


class TestSelect:
    def test_relevant_tools_rank_into_top_k(self):
        catalog = _catalog()
        picked = {t.name for t in catalog.select("post a message to slack", k=7)}
        assert "mcp__slack__post_message" in picked
        assert "mcp__slack__list_channels" in picked

    def test_database_query_surfaces_db_tools(self):
        catalog = _catalog()
        picked = {t.name for t in catalog.select("run an sql query", k=7)}
        assert "mcp__db__run_query" in picked

    def test_score_floor_excludes_non_matches(self):
        """No vocabulary overlap -> excluded, NOT ranked last."""
        catalog = _catalog()
        picked = catalog.select("zzz qqq xyzzyplugh", k=7, must_include=())
        assert picked == []

    def test_pins_always_return_even_on_zero_score(self):
        catalog = _catalog()
        picked = catalog.select("zzz qqq xyzzyplugh", k=7,
                                must_include=("filler_3", "list_available_tools"))
        names = [t.name for t in picked]
        assert names == ["filler_3"]  # pin returned; nothing else scored

    def test_k_bounds_result_size(self):
        catalog = _catalog(60)
        assert len(catalog.select("misc utility number", k=5)) <= 5

    def test_default_pins_come_from_catalog(self):
        catalog = _catalog()
        catalog.must_include = ("filler_0",)
        picked = catalog.select("nothing matches this at all zzz", k=4)
        assert [t.name for t in picked] == ["filler_0"]

    def test_add_rejects_duplicates(self):
        catalog = ToolCatalog([_mk("a", "tool a")])
        with pytest.raises(ValueError):
            catalog.add(_mk("a", "another a"))


class TestQueryFromTranscript:
    def test_first_user_message_is_the_anchor(self):
        history = [
            Message("user", [TextBlock("deploy the slack notifier")]),
            Message("assistant", [TextBlock("working on it")]),
        ]
        query = query_from_transcript(history)
        assert "slack" in query and "notifier" in query

    def test_tool_call_names_feed_vocabulary(self):
        """Having called mcp__db__* puts db vocabulary into the query --
        that is how mid-task pivots surface their new tools."""
        call = type("X", (), {})  # minimal block with .name
        call.name = "mcp__db__run_query"
        history = [
            Message("user", [TextBlock("explore the database")]),
            Message("assistant", [TextBlock("checking"), call]),
        ]
        assert "mcp__db__run_query" in query_from_transcript(history)

    def test_discovery_result_feeds_the_next_selection(self):
        """THE discovery contract, mechanically: a list_available_tools
        answer is a user-role ToolResult full of tool names. If results
        didn't join the query, 'discovered here -> usable next step'
        would silently fail (live-verified failure mode)."""
        listing = "8 tool(s) available:\nmcp__s1__add — Add two integers."
        history = [
            Message("user", [TextBlock("use an mcp tool to add 2 and 3")]),
            Message("assistant", []),
            Message("user", [ToolResult("c1", listing)]),
        ]
        query = query_from_transcript(history)
        assert "mcp__s1__add" in query

        # ...and through the ranking, replaying the LIVE failure: with
        # only the opener as query, an MCP-flavored echo description
        # outranked the terse add tool; the discovery result in the
        # query flips it.
        class T(Tool):
            def __init__(self, name, description):
                self.name, self.description = name, description
                self.parameters = {"type": "object", "properties": {}}
                self.read_only = True

            def summary(self, args, ctx):
                return self.name

            def run(self, args, ctx):
                return ""

        catalog = ToolCatalog([
            T("mcp__s1__echo",
              "Repeat the message back, prefixed with 'echo:'. Useful "
              "for proving an MCP round trip works."),
            T("mcp__s1__add", "Add two integers."),
        ])
        opener_only = "use an mcp tool to add 2 and 3"
        assert catalog.select(opener_only, k=2,
                              must_include=())[0].name == "mcp__s1__echo"
        assert catalog.select(query, k=2,
                              must_include=())[0].name == "mcp__s1__add"


class TestDiscoveryTool:
    def test_lists_everything(self):
        catalog = _catalog(10)  # 4 named + 10 fillers; discovery NOT included
        out = ListAvailableTools(catalog).run({}, None)
        lines = out.splitlines()
        assert lines[0].startswith("14 tool(s)")
        for name in ("mcp__slack__post_message", "filler_9"):
            assert any(line.startswith(name) for line in lines)

    def test_filter_narrows(self):
        catalog = _catalog(10)
        out = ListAvailableTools(catalog).run({"filter_term": "sql"}, None)
        assert "mcp__db__run_query" in out
        assert "filler_1" not in out

    def test_no_match_reports_cleanly(self):
        out = ListAvailableTools(_catalog()).run({"filter_term": "qqqzzz"}, None)
        assert "no tool matches" in out


class TestEnableSelection:
    def test_registers_discovery_and_builds_index(self):
        registry = ToolRegistry()
        for i in range(AUTO_SELECTION_THRESHOLD + 5):  # past the cliff
            registry.register(_mk(f"svc_{i}", f"service tool {i}"))
        catalog, discovery = enable_selection(registry)
        assert "list_available_tools" in registry
        assert catalog.get("list_available_tools") is discovery
        assert len(catalog.tools) == len(registry)

    def test_idempotent_registration(self):
        registry = ToolRegistry()
        for i in range(3):
            registry.register(_mk(f"t{i}", f"tool {i}"))
        enable_selection(registry)
        enable_selection(registry)  # must not raise on the second wire-up


# ---- loop integration ----------------------------------------------------------


def _agent_with_selection(provider, k=5):
    registry = ToolRegistry()
    for i in range(AUTO_SELECTION_THRESHOLD + 10):
        registry.register(_mk(f"filler_{i}", f"misc utility number {i}"))
    registry.register(_mk("mcp__db__run_query",
                          "execute sql against a postgres database"))
    catalog, _ = enable_selection(registry)
    agent = Agent(provider, model="scripted", tools=registry,
                  permissions=yolo, tool_catalog=catalog, tools_per_turn=k)
    return agent


class TestLoopIntegration:
    def test_only_selected_specs_are_sent(self):
        provider = ScriptedProvider([
            ModelResponse(Message("assistant",
                                  [TextBlock("done")]), "end_turn", Usage()),
        ])
        agent = _agent_with_selection(provider, k=6)
        # First user message anchors the query toward... itself. The
        # catalog is mostly fillers, so whatever ranks top-6, the request
        # must carry EXACTLY that many specs (+ none of the rest).
        agent.run("misc utility work please")
        sent = provider.requests[0]["tools"]
        assert len(sent) == 6
        assert all(s.name != "mcp__db__run_query" or s.name in
                   {t.name for t in agent._turn_tools} for s in sent)

    def test_unselected_tool_becomes_actionable_error_data(self):
        """The book's try-fail-retry mechanism: calling a hidden-but-real
        tool yields an error result that TEACHES the discovery path."""
        provider = ScriptedProvider([
            ModelResponse(Message("assistant", [
                TextBlock("querying"),
                ToolCall("c1", "mcp__db__run_query", {"sql": "select 1"}),
            ]), "tool_use", Usage(input_tokens=10, output_tokens=5)),
            ModelResponse(Message("assistant",
                                  [TextBlock("understood")]), "end_turn", Usage()),
        ])
        agent = _agent_with_selection(provider, k=5)
        events = list(agent.run_streaming("totally unrelated filler talk"))
        errors = [e.result for e in events
                  if hasattr(e, "result") and e.result.is_error]
        assert errors, "unselected call should fail as data"
        assert "not loaded this turn" in errors[0].content
        assert "list_available_tools" in errors[0].content
        # ...and the failed NAME is now in history, so the next selection
        # query carries its vocabulary (the convergence engine).
        rendered = json.dumps([b.__dict__ if hasattr(b, "__dict__")
                               else str(b) for b in []])  # placeholder no-op
        last = provider.requests[-1]["messages"]
        flat = "".join(getattr(b, "text", "") or getattr(b, "name", "")
                       or "" for m in last for b in m.content)
        assert "mcp__db__run_query" in flat

    def test_no_catalog_keeps_full_registry(self):
        """Without a catalog everything is byte-identical to before."""
        from akshara.tools.base import ToolRegistry as R
        registry = R()
        for i in range(30):
            registry.register(_mk(f"t{i}", f"tool {i}"))
        provider = ScriptedProvider([
            ModelResponse(Message("assistant", [TextBlock("ok")]),
                          "end_turn", Usage()),
        ])
        agent = Agent(provider, model="scripted", tools=registry,
                      permissions=yolo)
        agent.run("anything")
        assert len(provider.requests[0]["tools"]) == 30
