"""REPL input mechanics: backslash-continuation multi-line entry.

Kept to the input seam on purpose -- the rest of the REPL is thin glue
over Agent + Renderer, which have their own test modules.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from conftest import ScriptedProvider, assistant_text

from akshara.agent import Agent
from akshara.cli.repl import Repl, confirm_gate
from akshara.permissions import PermissionRequest, SwitchableGate, allow_read_only
from akshara.types import ImageBlock, StartEvent, TextBlock, TextDelta


def make_repl(lines: list[str]) -> tuple[Repl, list[str]]:
    """A Repl whose prompts come from ``lines``; records the prompts shown."""
    agent = Agent(ScriptedProvider([]), model="m", permissions=allow_read_only)
    feeder = iter(lines)
    seen_prompts: list[str] = []
    repl = Repl(
        agent,
        Console(file=io.StringIO(), width=100),
        input_fn=lambda prompt: (seen_prompts.append(prompt), next(feeder))[1],
    )
    return repl, seen_prompts


class TestMultilineInput:
    def test_trailing_backslash_joins_lines(self):
        repl, _ = make_repl(["review this:\\", "  def main(): ..."])
        assert repl._read_line() == "review this:\n  def main(): ..."

    def test_plain_line_needs_no_continuation(self):
        repl, _ = make_repl(["just this"])
        assert repl._read_line() == "just this"

    def test_continuation_prompt_shown_for_followups(self):
        repl, seen = make_repl(["a\\", "b\\", "c"])
        assert repl._read_line() == "a\nb\nc"
        assert seen == ["> ", "... ", "... "]

    def test_backslash_consumed_even_mid_whitespace(self):
        repl, _ = make_repl(["ends with backslash \\ ", "second"])
        # trailing spaces stripped by .strip(); exactly ONE backslash eaten
        assert repl._read_line() == "ends with backslash \nsecond"


class TestSpinnerLifecycle:
    """The dead-air spinner must die at the FIRST STREAMED event.

    Deltas reach the renderer through the PUSH channel
    (agent.on_stream_event), never through run_turn's pull loop -- which,
    for a tool-free reply, sees nothing until TurnEnd. Stopping there left
    the whole reply streaming inside an active rich Live region, whose
    repaints erase the partial lines they wrap; on a real terminal that
    rendered as an EMPTY reply. These tests pin the stop to the push
    channel with a recording stub, since a non-TTY Console disables Live
    and would hide the bug entirely.
    """

    def make_spinner_repl(self, script: list) -> tuple[Repl, list[str]]:
        log: list[str] = []

        class StubStatus:
            def start(self):
                log.append("start")

            def stop(self):
                log.append("stop")

        class StubConsole(Console):
            def status(self, *args, **kwargs):
                return StubStatus()

        agent = Agent(ScriptedProvider(script), model="m",
                      permissions=allow_read_only)
        repl = Repl(agent, StubConsole(file=io.StringIO(), width=100),
                    input_fn=lambda prompt: "")

        inner = repl.renderer

        def spy(event):
            if isinstance(event, (StartEvent, TextDelta)):
                log.append(f"render:{type(event).__name__}")
            inner(event)

        repl.renderer = spy  # observe ordering without changing what paints
        return repl, log

    def test_stops_at_first_stream_event(self):
        repl, log = self.make_spinner_repl([assistant_text("hi")])
        repl.run_turn("hello")
        assert log[0] == "start"
        assert log.count("stop") == 1
        # the stop precedes even StartEvent -- nothing paints under Live
        assert log[:3] == ["start", "stop", "render:StartEvent"]

    def test_stops_even_when_nothing_streams(self):
        # provider dies before its first event -- finally() is the last line
        # of defense, or the spinner outlives the turn that started it
        repl, log = self.make_spinner_repl([])
        with pytest.raises(AssertionError):
            repl.run_turn("hello")
        assert log == ["start", "stop"]


class TestImageCommand:
    """/image stages attachments onto the NEXT message.

    Same seam as the CLI flag -- validated at attach time (a typo errors
    immediately), consumed by the next run_turn, invisible to slash
    commands in between.
    """

    PNG = b"\x89PNG\r\n\x1a\n"

    @pytest.fixture
    def shot(self, tmp_path):
        path = tmp_path / "shot.png"
        path.write_bytes(self.PNG)
        return str(path)

    def make_image_repl(self, lines: list[str], script: list) -> tuple[Repl, io.StringIO]:
        agent = Agent(ScriptedProvider(script), model="m",
                      permissions=allow_read_only)
        feeder = iter(lines)
        out = io.StringIO()
        repl = Repl(agent, Console(file=out, width=100),
                    input_fn=lambda prompt: next(feeder))
        return repl, out

    def test_staged_images_ride_the_next_turn(self, shot):
        repl, out = self.make_image_repl(
            [f"/image {shot}", "what is it", "/quit"],
            [assistant_text("a png")])

        repl.run()

        assert "attached shot.png" in out.getvalue()
        first_user = repl.agent.history[0]
        assert isinstance(first_user.content[1], ImageBlock)
        assert first_user.content[0] == TextBlock("what is it")
        assert repl._pending_images == []  # consumed, not hoarded

    def test_bad_path_errors_at_attach_time(self, tmp_path):
        repl, out = self.make_image_repl(
            [f"/image {tmp_path / 'ghost.png'}", "/quit"], [])

        repl.run()

        # console may soft-wrap mid-phrase at width 100 -- compare unwrapped
        assert "not attached" in " ".join(out.getvalue().split())
        assert repl._pending_images == []

    def test_staging_survives_a_slash_command_in_between(self, shot):
        repl, out = self.make_image_repl(
            [f"/image {shot}", "/help", "go", "/quit"],
            [assistant_text("seen")])

        repl.run()

        assert isinstance(repl.agent.history[0].content[1], ImageBlock)

    def test_bare_command_reports_the_stage(self, shot):
        repl, out = self.make_image_repl([f"/image {shot}", "/image", "/quit"], [])

        repl.run()

        assert "1 image(s) staged" in out.getvalue()

    def test_clear_unstages_everything(self, shot):
        repl, out = self.make_image_repl([f"/image {shot}", "/image clear", "/quit"], [])

        repl.run()

        assert "cleared" in out.getvalue()
        assert repl._pending_images == []

    def test_quoted_paths_with_spaces_survive(self, tmp_path):
        path = tmp_path / "my shot.png"
        path.write_bytes(self.PNG)
        repl, out = self.make_image_repl([f'/image "{path}"', "/quit"], [])

        repl.run()

        assert "attached my shot.png" in out.getvalue()


def test_confirm_gate_never_prompts_read_only_tools():
    """A tool that declared itself side-effect-free has nothing to
    confirm -- and the discovery hatch MUST be friction-free or the
    selection loop dies behind a permission wall."""
    console = Console()
    gate = confirm_gate(console)
    request = PermissionRequest(tool_name="list_available_tools",
                                arguments={}, summary="listing",
                                read_only=True)
    assert gate(request) is True


class TestConfirmGateEdits:
    """The y/n/e prompt: 'e' amends the pending call, re-previews, re-asks.

    The editor is injected -- the same seam input_fn uses above -- so no
    test ever opens a real $EDITOR.
    """

    @staticmethod
    def _request(**overrides) -> PermissionRequest:
        fields = dict(tool_name="bash", arguments={"command": "echo hi"},
                      summary="$ echo hi", read_only=False,
                      summarize=lambda args: f"$ {args.get('command', '')}")
        fields.update(overrides)
        return PermissionRequest(**fields)

    @staticmethod
    def _console() -> Console:
        return Console(file=io.StringIO(), width=200)

    def test_plain_yes_approves_unchanged(self):
        import unittest.mock as mock
        from rich.prompt import Prompt

        def exploding_editor(args):
            raise AssertionError("editor must not open on plain y/n")

        gate = confirm_gate(self._console(), editor=exploding_editor)
        request = self._request()
        with mock.patch.object(Prompt, "ask", staticmethod(lambda *a, **k: "y")):
            assert gate(request) is True
        assert request.arguments == {"command": "echo hi"}  # untouched

    def test_edit_then_approve_swaps_arguments_and_repreviews(self):
        def editor(args):
            assert args == {"command": "echo hi"}  # edits start from current
            return {"command": "echo goodbye"}

        console = self._console()
        gate = confirm_gate(console, editor=editor)
        # feed: first round 'e' (edit), second round 'y'
        from rich.prompt import Prompt
        answers = iter(["e", "y"])
        import akshara.cli.repl as repl_mod
        real_ask = Prompt.ask
        monkeyed = lambda *a, **k: next(answers)  # noqa: E731

        import unittest.mock as mock
        with mock.patch.object(Prompt, "ask", staticmethod(monkeyed)):
            assert gate(self._request()) is True
        assert "edited" in console.file.getvalue()
        assert "$ echo goodbye" in console.file.getvalue()  # re-preview ran

    def test_bad_json_edit_asks_again_instead_of_denying(self):
        def bad_editor(args):
            raise ValueError("expecting ',' delimiter")

        console = self._console()
        gate = confirm_gate(console, editor=bad_editor)
        answers = iter(["e", "n"])  # edit fails once, then deny
        from rich.prompt import Prompt
        import unittest.mock as mock
        with mock.patch.object(Prompt, "ask",
                               staticmethod(lambda *a, **k: next(answers))):
            assert gate(self._request()) is False  # denial is EXPLICIT
        assert "bad edit" in console.file.getvalue()

    def test_cancelled_edit_returns_to_the_prompt(self):
        gate = confirm_gate(self._console(), editor=lambda args: None)
        answers = iter(["e", "n"])
        from rich.prompt import Prompt
        import unittest.mock as mock
        with mock.patch.object(Prompt, "ask",
                               staticmethod(lambda *a, **k: next(answers))):
            assert gate(self._request()) is False

    def test_fallback_summary_when_summarize_missing(self):
        """No summarize pre-bound -> edited preview falls back to raw JSON."""
        request = self._request(summarize=None)
        gate = confirm_gate(self._console(),
                            editor=lambda args: {"command": "rm -rf /tmp/x"})
        answers = iter(["e", "y"])
        from rich.prompt import Prompt
        import unittest.mock as mock
        with mock.patch.object(Prompt, "ask",
                               staticmethod(lambda *a, **k: next(answers))):
            assert gate(request) is True
        assert request.arguments == {"command": "rm -rf /tmp/x"}


class TestYoloCommand:
    """/yolo flips the session between bypass-everything and ask-first,
    mid-session -- the terminal half of runtime permission switching."""

    @staticmethod
    def make_gate_repl(lines: list[str], permissions=None):
        """A Repl over a switchable (or injected) gate; exposes the console
        so the announcements can be asserted, and the prompts actually fed."""
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=permissions or SwitchableGate(allow_read_only))
        feeder = iter(lines)
        seen_prompts: list[str] = []
        console = Console(file=io.StringIO(), width=100)
        repl = Repl(
            agent, console,
            input_fn=lambda prompt: (seen_prompts.append(prompt), next(feeder))[1],
        )
        return repl, console, agent.permissions, seen_prompts

    def test_bare_command_toggles_on_then_off(self):
        repl, console, gate, _ = self.make_gate_repl([])
        assert repl._command("/yolo") is False
        assert gate.mode == "yolo"
        assert "WITHOUT asking" in console.file.getvalue()
        assert repl._command("/yolo") is False
        assert gate.mode == "ask"
        assert "prompts back on" in console.file.getvalue()

    def test_explicit_on_and_off(self):
        repl, _, gate, _ = self.make_gate_repl([])
        repl._command("/yolo on")
        assert gate.mode == "yolo"
        repl._command("/yolo off")
        assert gate.mode == "ask"

    def test_invalid_argument_reports_usage_and_keeps_mode(self):
        repl, console, gate, _ = self.make_gate_repl([])
        repl._command("/yolo maybe")
        assert gate.mode == "ask"
        assert "usage:" in console.file.getvalue()

    def test_prompt_prefix_shows_yolo_while_bypassed(self):
        repl, _, gate, seen = self.make_gate_repl(["first line", "second line"])
        assert repl._read_line() == "first line"
        assert seen == ["> "]
        gate.set_mode("yolo")
        assert repl._read_line() == "second line"
        assert seen[-1] == "yolo> "

    def test_banner_warns_only_while_bypassed(self):
        repl, console, gate, _ = self.make_gate_repl([])
        repl._banner()
        assert "yolo" not in console.file.getvalue()
        gate.set_mode("yolo")
        console.file.truncate(0)
        console.file.seek(0)
        repl._banner()
        assert "no permission prompts" in console.file.getvalue()

    def test_fixed_gate_reports_instead_of_crashing(self):
        repl, console, gate, _ = self.make_gate_repl([],
                                                     permissions=allow_read_only)
        repl._command("/yolo on")
        assert not isinstance(gate, SwitchableGate)  # untouched static fn
        assert "fixed" in console.file.getvalue()


class TestToolsCommand:
    """`/tools off|on NAME|GLOB`: the terminal twin of the web panel's
    switches. Pulls are live immediately and reversible in-session."""

    def _repl(self):
        from akshara.tools.base import ToolRegistry
        from akshara.tools.fs import ReadFile, WriteFile
        from akshara.tools.glob import Glob

        registry = ToolRegistry()
        registry.register(ReadFile())
        registry.register(WriteFile())
        agent = Agent(ScriptedProvider([]), model="m", tools=registry,
                      permissions=SwitchableGate(allow_read_only))
        console = Console(file=io.StringIO(), width=100)
        repl = Repl(agent, console, input_fn=lambda _prompt: "")
        return repl, console

    def test_off_then_on_round_trip(self):
        repl, console = self._repl()
        assert repl._command("/tools off write_file") is False
        assert repl.agent.registry.is_disabled("write_file")
        assert "disabled 1 tool(s)" in console.file.getvalue()
        assert repl._command("/tools on write_file") is False
        assert not repl.agent.registry.is_disabled("write_file")

    def test_glob_patterns_match_like_the_env_kill_switch(self):
        from akshara.tools.glob import Glob

        repl, console = self._repl()
        repl.agent.registry.register(Glob())  # name: "glob"
        repl._command("/tools off *_file")
        assert repl.agent.registry.disabled_names() == ["read_file",
                                                        "write_file"]
        repl._command("/tools on read_file")
        assert repl.agent.registry.disabled_names() == ["write_file"]

    def test_listing_marks_disabled_and_keeps_them_visible(self):
        repl, console = self._repl()
        repl._command("/tools off write_file")
        console.file.truncate(0)
        console.file.seek(0)
        repl._command("/tools")
        out = console.file.getvalue()
        assert "[off]" in out                    # marked ...
        assert "write_file" in out               # ... but still listed
        assert "disabled this session (1)" in out

    def test_unknown_pattern_warns_without_touching_registry(self):
        repl, console = self._repl()
        repl._command("/tools off ghost_tool")
        assert "no tool matches" in console.file.getvalue()
        assert repl.agent.registry.disabled_names() == []

    def test_bare_off_needs_a_name(self):
        repl, console = self._repl()
        repl._command("/tools off")
        assert "usage:" in console.file.getvalue()


class TestMcpCommand:
    """/mcp: the terminal twin of the panel's servers section -- list,
    soft-toggle, remove, and add MID-SESSION over the same MCPManager."""

    @staticmethod
    def _fake_connector(closed_log=None):
        from akshara.mcp import MCPServerConfig
        from akshara.tools.base import Tool

        class FakeSess:
            def __init__(self, config):
                self.config = config
                self.closed = False
            def healthy(self):
                return not self.closed
            def close(self):
                self.closed = True
                if closed_log is not None:
                    closed_log.append(self.config.name)

        class FakeTool(Tool):
            name = ""
            description = "fake mcp tool"
            parameters = {"type": "object", "properties": {}}
            read_only = True
            def __init__(self, session, raw_name, server):
                self.name = f"mcp__{server}__{raw_name}"
            def summary(self, args, ctx):
                return self.name
            def run(self, args, ctx):
                return "ran"

        def connector(config, timeout=30.0):
            s = FakeSess(config)
            return s, [FakeTool(s, "echo", config.name),
                       FakeTool(s, "ping", config.name)]

        return connector

    @staticmethod
    def _mcpcfg(name):
        from akshara.mcp import MCPServerConfig
        return MCPServerConfig(name=name, command="python",
                               args=["srv.py"])

    def _repl(self, tmp_path=None, connector=None):
        from akshara.mcp import MCPManager
        from akshara.tools.base import ToolRegistry
        from akshara.tools.fs import ReadFile

        registry = ToolRegistry()
        registry.register(ReadFile())
        manager = MCPManager(
            registry,
            memory_path=(tmp_path / ".akshara" / "mcp.json")
            if tmp_path else None,
            connector=connector or self._fake_connector())
        agent = Agent(ScriptedProvider([]), model="m", tools=registry,
                      permissions=SwitchableGate(allow_read_only))
        console = Console(file=io.StringIO(), width=120)
        repl = Repl(agent, console, input_fn=lambda _prompt: "", mcp=manager)
        return repl, console, manager

    def test_no_manager_reports_instead_of_crashing(self):
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=allow_read_only)
        console = Console(file=io.StringIO(), width=100)
        repl = Repl(agent, console, input_fn=lambda _p: "")
        assert repl._command("/mcp") is False
        assert "no mcp manager" in console.file.getvalue()

    def test_empty_listing_points_at_add(self):
        repl, console, _ = self._repl()
        repl._command("/mcp")
        assert "no mcp servers connected" in console.file.getvalue()

    def test_listing_shows_transport_target_and_counts(self):
        repl, console, manager = self._repl()
        manager.connect(self._mcpcfg("tiny"))
        console.file.truncate(0)
        console.file.seek(0)
        repl._command("/mcp")
        out = console.file.getvalue()
        assert "tiny" in out and "(stdio)" in out
        assert "python srv.py" in out
        assert "2 tool(s)" in out

    def test_off_then_on_soft_toggles_whole_server(self):
        repl, console, manager = self._repl()
        manager.connect(self._mcpcfg("tiny"))
        name = "mcp__tiny__echo"
        assert repl._command("/mcp off tiny") is False
        assert repl.agent.registry.is_disabled(name)
        assert "disabled 2 tool(s)" in console.file.getvalue()
        assert repl._command("/mcp on tiny") is False
        assert not repl.agent.registry.is_disabled(name)

    def test_toggle_usage_and_unknown_server_are_loud(self):
        repl, console, manager = self._repl()
        manager.connect(self._mcpcfg("tiny"))
        repl._command("/mcp off")
        assert "usage:" in console.file.getvalue()
        repl._command("/mcp off ghost")
        assert "no mcp server named 'ghost'" in console.file.getvalue()

    def test_remove_unregisters_tools_and_closes_transport(self):
        closed = []
        repl, console, manager = self._repl(
            connector=self._fake_connector(closed))
        manager.connect(self._mcpcfg("tiny"))
        assert repl._command("/mcp remove tiny") is False
        assert "mcp__tiny__echo" not in repl.agent.registry
        assert closed == ["tiny"]
        out = console.file.getvalue()
        assert "disconnected 'tiny'" in out
        assert "saved entry forgotten" in out

    def test_add_url_connects_over_http_shape(self):
        import unittest.mock as mock
        from rich.prompt import Prompt

        repl, console, manager = self._repl(tmp_path=None)
        with mock.patch.object(Prompt, "ask",
                               staticmethod(lambda *a, **k: "N")):
            assert repl._command("/mcp add remote http://127.0.0.1:9/mcp") \
                is False
        info = manager.servers()[0]
        assert info["transport"] == "http"
        assert "connected remote: 2 tool(s)" in console.file.getvalue()

    def test_add_command_with_args_and_remember_saves_entry(self, tmp_path=None):
        import tempfile
        import unittest.mock as mock
        from pathlib import Path
        from rich.prompt import Prompt

        tmp = Path(tempfile.mkdtemp())
        repl, console, manager = self._repl(tmp_path=tmp)
        with mock.patch.object(Prompt, "ask",
                               staticmethod(lambda *a, **k: "y")):
            assert repl._command(
                "/mcp add tiny python srv.py --port 9") is False
        cfgs = manager.servers()
        assert cfgs[0]["target"] == "python srv.py --port 9"
        assert cfgs[0]["remembered"] is True
        saved = tmp / ".akshara" / "mcp.json"
        assert saved.exists() and "tiny" in saved.read_text()
        assert "saved to" in console.file.getvalue()

    def test_add_failure_is_reported_not_raised(self):
        from akshara.mcp import MCPError

        def failing(config, timeout=30.0):
            raise MCPError("cannot spawn mcp server 'dead'")

        repl, console, _ = self._repl(connector=failing)
        repl._command("/mcp add dead python x.py")
        assert "could not connect 'dead'" in console.file.getvalue()
