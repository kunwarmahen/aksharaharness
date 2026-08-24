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
from akshara.permissions import PermissionRequest, allow_read_only
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
