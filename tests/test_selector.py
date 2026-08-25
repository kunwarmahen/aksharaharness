"""Tool-selection tests: BM25 ranking, pins, the discovery hatch, and
the loop-level contract (selected specs are what get SENT and the ONLY
things that can EXECUTE).
"""

from __future__ import annotations

import pytest

from akshara.agent import Agent
from akshara.permissions import yolo
from akshara.tools.base import Tool, ToolRegistry
from akshara.tools.selector import (
    AUTO_SELECTION_THRESHOLD,
    CORE_PINS,
    DEFAULT_TOOLS_PER_TURN,
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

    def test_core_pins_load_even_at_zero_score(self):
        """Pins beat the query: the autonomy loop's floor is loadable on
        every turn, whatever the conversation is about."""
        registry = ToolRegistry()
        for name in ("read_file", "write_file", "bash"):
            registry.register(_mk(name, f"{name}: terse description"))
        for i in range(30):
            registry.register(_mk(f"svc_{i}", f"service tool {i}"))
        catalog, _ = enable_selection(registry)
        picked = {t.name for t in
                  catalog.select("zzz qqq nothing matches", k=12)}
        assert {"read_file", "write_file", "bash",
                "list_available_tools"} <= picked

    def test_pins_absent_from_registry_are_skipped(self):
        """A pin list may be written against the FULL default toolset;
        names that are not registered (or were disabled) just don't pin."""
        registry = ToolRegistry()
        for i in range(25):
            registry.register(_mk(f"svc_{i}", f"service tool {i}"))
        catalog, _ = enable_selection(registry)
        assert catalog.must_include == ("list_available_tools",)

    def test_custom_pins_replace_the_default_set(self):
        registry = ToolRegistry()
        for i in range(25):
            registry.register(_mk(f"svc_{i}", f"service tool {i}"))
        registry.register(_mk("special", "the one that matters"))
        catalog, _ = enable_selection(registry, pins=("special",))
        assert "special" in catalog.must_include
        assert "read_file" not in catalog.must_include


class TestRegistryUnregister:
    """unregister exists for the operator kill-switch: AKSHARA_DISABLED_TOOLS
    pulls tools BEFORE catalog building, so a disabled tool is never sent,
    suggested, or pinned."""

    def test_round_trip(self):
        registry = ToolRegistry()
        registry.register(_mk("t0", "tool zero"))
        assert registry.unregister("t0") is True
        assert "t0" not in registry

    def test_missing_name_reports_false(self):
        registry = ToolRegistry()
        assert registry.unregister("ghost") is False


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

    def test_unselected_tool_soft_admits_and_executes(self):
        """Selection caps what gets SENT, not what can execute: a model
        that names an existing tool has already discovered it, so the
        call is admitted and runs THIS turn -- no punishment lap."""
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
        results = [e.result for e in events if hasattr(e, "result")]
        assert results, "the call should have executed"
        assert not any(r.is_error for r in results)
        assert "mcp__db__run_query ran" in results[0].content
        # Admission sticks: the name is now selected, no BM25 luck needed.
        assert "mcp__db__run_query" in {t.name for t in agent._turn_tools}
        # ...and the name landed in history, feeding future queries.
        flat = "".join(getattr(b, "text", "") or getattr(b, "name", "")
                       or "" for m in provider.requests[-1]["messages"]
                       for b in m.content)
        assert "mcp__db__run_query" in flat

    def test_unknown_name_still_errors_as_data(self):
        """Soft admission leaves exactly one failure mode: a genuinely
        hallucinated name, whose error still teaches the discovery path."""
        provider = ScriptedProvider([
            ModelResponse(Message("assistant", [
                ToolCall("c1", "no_such_tool", {}),
            ]), "tool_use", Usage(input_tokens=10, output_tokens=5)),
            ModelResponse(Message("assistant",
                                  [TextBlock("ok")]), "end_turn", Usage()),
        ])
        agent = _agent_with_selection(provider, k=5)
        events = list(agent.run_streaming("anything at all"))
        errors = [e.result for e in events
                  if hasattr(e, "result") and e.result.is_error]
        assert errors, "hallucinated call should fail as data"
        assert "no such tool" in errors[0].content

    def test_core_pins_reach_every_request(self):
        """THE regression this fixes: 'refactor the parser module' shares
        zero vocabulary with write_file -- retrieval alone would drop it,
        pins keep the autonomy loop loadable every single turn."""
        provider = ScriptedProvider([
            ModelResponse(Message("assistant",
                                  [TextBlock("done")]), "end_turn", Usage()),
        ])
        registry = ToolRegistry()
        for name in CORE_PINS:
            registry.register(_mk(name, f"{name}: terse generic description"))
        for i in range(30):
            registry.register(_mk(f"filler_{i}", f"misc utility number {i}"))
        catalog, _ = enable_selection(registry)
        agent = Agent(provider, model="scripted", tools=registry,
                      permissions=yolo, tool_catalog=catalog,
                      tools_per_turn=DEFAULT_TOOLS_PER_TURN)
        agent.run("refactor the parser module")
        sent = provider.requests[0]["tools"]
        assert set(CORE_PINS) <= {s.name for s in sent}
        # The query matches nothing else, so the floor leaves exactly the
        # pins -- never filler ranked last.
        assert len(sent) == len(CORE_PINS) + 1  # + discovery

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


# ---- runtime-disabled tools -----------------------------------------------------


class TestDisabledTools:
    """The operator's mid-session pulls must hold at every layer: no
    selection slot, no pin slot, no discovery suggestion, no execution --
    all consulted LIVE, so /tools off takes effect on the next call."""

    def test_select_skips_hidden_even_as_pins(self):
        """CORE_PINS is a static list; a pinned tool that the operator
        pulled this session must not load anyway."""
        registry = ToolRegistry()
        for name in CORE_PINS:
            registry.register(_mk(name, f"{name}: terse generic description"))
        registry.disable("bash")
        catalog, _ = enable_selection(registry)
        picked = {t.name for t in catalog.select("refactor anything", k=12)}
        assert "bash" not in picked

    def test_discovery_never_suggests_hidden(self):
        registry = ToolRegistry()
        registry.register(_mk("mcp__db__run_query",
                              "execute sql against a database"))
        registry.register(_mk("mcp__db__drop_all", "drop everything"))
        registry.disable("mcp__db__drop_all")
        catalog, discovery = enable_selection(registry)
        out = discovery.run({}, None)
        assert "mcp__db__run_query" in out
        assert "mcp__db__drop_all" not in out

    def test_catalog_get_returns_none_for_hidden(self):
        """Soft admission goes through catalog.get -- returning None here
        (not the tool) is what stops a model-named disabled call from
        being admitted into the visible set."""
        registry = ToolRegistry()
        registry.register(_mk("special", "a special utility"))
        registry.disable("special")
        catalog, _ = enable_selection(registry)
        assert catalog.get("special") is None

    def test_disable_after_wiring_applies_live(self):
        """hidden is consulted per call, not baked in at enable_selection
        time -- flipping it afterwards changes selection immediately."""
        registry = ToolRegistry()
        for name in CORE_PINS:
            registry.register(_mk(name, f"{name}: terse generic description"))
        catalog, _ = enable_selection(registry)
        assert "bash" in {t.name for t in catalog.select("x y z", k=12)}
        registry.disable("bash")
        assert "bash" not in {t.name for t in catalog.select("x y z", k=12)}


class TestLoopDisabledTool:
    def test_disabled_call_errors_as_data_mid_selection(self):
        """With selection active AND the tool unselected-but-real, a call
        to an operator-pulled tool fails as data -- the agent-level choke
        point (_get_visible_tool) refuses before soft admission."""
        provider = ScriptedProvider([
            ModelResponse(Message("assistant", [
                ToolCall("c1", "mcp__db__run_query", {"sql": "select 1"}),
            ]), "tool_use", Usage(input_tokens=10, output_tokens=5)),
            ModelResponse(Message("assistant",
                                  [TextBlock("moving on")]), "end_turn", Usage()),
        ])
        agent = _agent_with_selection(provider, k=5)
        agent.registry.disable("mcp__db__run_query")

        events = list(agent.run_streaming("database work please"))
        errors = [e.result for e in events
                  if hasattr(e, "result") and e.result.is_error]
        assert errors, "disabled call must fail as data"
        assert "disabled by the operator" in errors[0].content
