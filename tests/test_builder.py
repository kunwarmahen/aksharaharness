"""Builder tests: spec in -> project out, VERIFIED independently.

The whole point of akshara.builder is that it trusts nothing the model
claims, so the tests attack exactly that: a build that goes green, a
model whose acceptance checks fail anyway, and a repair job that tries
to weaken its checksummed tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from akshara.agent import Agent
from akshara.builder import (
    BUILD_SYSTEM,
    BuildSpec,
    default_checks,
    run_build,
)
from akshara.permissions import yolo
from akshara.tools import default_registry
from akshara.types import Message, ModelResponse, TextBlock, ToolCall, Usage

from conftest import ScriptedProvider


def _resp(blocks, stop="end_turn", usage=None):
    return ModelResponse(Message("assistant", blocks), stop,
                         usage or Usage(input_tokens=10, output_tokens=5))


def _write(call_id: str, path: str, content: str) -> ToolCall:
    return ToolCall(call_id, "write_file", {"path": path, "content": content})


def _factory(provider) -> "callable":
    def factory(workspace: Path) -> Agent:
        return Agent(provider, model="scripted", system=BUILD_SYSTEM,
                     tools=default_registry(), permissions=yolo,
                     cwd=workspace)
    return factory


PASSING_MODULE = "def answer():\n    return 42\n"
PASSING_TEST = (
    "import unittest\nimport mod\n"
    "class T(unittest.TestCase):\n"
    "    def test_answer(self):\n"
    "        self.assertEqual(mod.answer(), 42)\n"
)


class TestRunBuild:
    def test_greenfield_build_goes_green(self, tmp_path):
        provider = ScriptedProvider([
            _resp([TextBlock("writing"), _write("t1", "mod.py", PASSING_MODULE),
                   _write("t2", "test_mod.py", PASSING_TEST)], stop="tool_use"),
            _resp([TextBlock("done; unittest passes")]),
        ])
        result = run_build(_factory(provider), BuildSpec(task="build mod"),
                           tmp_path / "ws")
        assert result.ok is True
        assert result.iterations == 2
        assert sorted(result.files) == ["mod.py", "test_mod.py"]
        assert all(c.passed for c in result.checks)
        assert result.usage_in == 20 and result.usage_out == 10
        assert "unittest" in result.response_text or result.response_text

    def test_lying_model_fails_verification(self, tmp_path):
        """Model SAYS done; the module is wrong; independent re-run must
        catch it -- the model's word is never evidence. No repair budget
        here: the script has nothing after 'trust me', and the point is
        the verdict, not the recovery."""
        provider = ScriptedProvider([
            _resp([TextBlock("writing broken code"),
                   _write("t1", "mod.py", "def answer():\n    return 0\n"),
                   _write("t2", "test_mod.py", PASSING_TEST)], stop="tool_use"),
            _resp([TextBlock("all tests pass, trust me")]),
        ])
        result = run_build(_factory(provider),
                           BuildSpec(task="build mod", max_repair_rounds=0),
                           tmp_path / "ws")
        assert result.ok is False
        assert any(not c.passed for c in result.checks)

    def test_red_verification_is_fed_back_and_repairs(self, tmp_path):
        """The Claude Code loop: a red verify goes back into the SAME
        conversation as actionable feedback, and a competent fix turns
        the build green."""
        provider = ScriptedProvider([
            _resp([TextBlock("writing broken code"),
                   _write("t1", "mod.py", "def answer():\n    return 0\n"),
                   _write("t2", "test_mod.py", PASSING_TEST)], stop="tool_use"),
            _resp([TextBlock("all tests pass, trust me")]),
            # ...verification came back RED; the harness feeds it back...
            _resp([TextBlock("fixing"), _write("t2", "mod.py",
                                               PASSING_MODULE)],
                  stop="tool_use"),
            _resp([TextBlock("now actually green")]),
        ])
        result = run_build(_factory(provider), BuildSpec(task="build mod"),
                           tmp_path / "ws")
        assert result.ok is True
        # the failure report reached the model as conversation data
        last_messages = provider.requests[-1]["messages"]
        flat = "".join(getattr(b, "text", "") or "" for m in last_messages
                       for b in m.content)
        assert "independent verification" in flat

    def test_repair_budget_exhaustion_stays_red(self, tmp_path):
        """A model that cannot fix it stays red -- bounded effort, honest
        verdict."""
        provider = ScriptedProvider([
            _resp([TextBlock("writing broken code"),
                   _write("t1", "mod.py", "def answer():\n    return 0\n"),
                   _write("t2", "test_mod.py", PASSING_TEST)], stop="tool_use"),
            _resp([TextBlock("trust me")]),
            _resp([TextBlock("nope, still broken")]),
        ])
        result = run_build(_factory(provider),
                           BuildSpec(task="build mod", max_repair_rounds=1),
                           tmp_path / "ws")
        assert result.ok is False

    def test_checksummed_test_tampering_fails_the_build(self, tmp_path):
        """Repair contract: modifying a checksummed test fails even if
        the suite then passes."""
        ws = tmp_path / "ws"
        ws.mkdir(parents=True)
        (ws / "test_contract.py").write_text("value = 1\n")
        provider = ScriptedProvider([
            _resp([ToolCall("a", "edit_file", {
                "path": "test_contract.py",
                "old_string": "value = 1", "new_string": "value = 2",
            })], stop="tool_use"),
            _resp([TextBlock("green now!")]),
        ])
        result = run_build(_factory(provider), BuildSpec(task="repair"), ws)
        assert result.tampered_tests == ["test_contract.py"]
        assert result.ok is False

    def test_seed_dir_copies_broken_project(self, tmp_path):
        seed = tmp_path / "seed"
        seed.mkdir()
        (seed / "impl.py").write_text("x = 'broken'\n")
        ws = tmp_path / "ws"
        provider = ScriptedProvider([_resp([TextBlock("done")])])
        spec = BuildSpec(task="fix it", seed_dir=seed,
                         checks=[(["true"], 0, None)])  # trivially passing gate
        result = run_build(_factory(provider), spec, ws)
        assert (ws / "impl.py").read_text() == "x = 'broken'\n"
        assert result.ok is True

    def test_custom_checks_enforced(self, tmp_path):
        """A preset's own acceptance commands are honored verbatim --
        including an expected NON-ZERO exit."""
        provider = ScriptedProvider([
            _resp([_write("t1", "flag.txt", "ready")], stop="tool_use"),
            _resp([TextBlock("done")]),
        ])
        spec = BuildSpec(task="write flag", checks=[
            ([sys.executable, "-c", "raise SystemExit(7)"], 7, None),
            ([sys.executable, "-c", "print('marker-ok')"], 0, "marker-ok"),
        ])
        result = run_build(_factory(provider), spec, tmp_path / "ws")
        assert result.ok is True

    def test_substring_mismatch_fails_check(self, tmp_path):
        provider = ScriptedProvider([
            _resp([_write("t1", "out.txt", "wrong output")], stop="tool_use"),
            _resp([TextBlock("done")]),
        ])
        spec = BuildSpec(task="write marker", max_repair_rounds=0, checks=[
            ([sys.executable, "-c", "print(open('out.txt').read())"],
             0, "expected-marker"),
        ])
        result = run_build(_factory(provider), spec, tmp_path / "ws")
        assert result.ok is False
        failed = [c for c in result.checks if not c.passed]
        assert failed and failed[0].missing_substring == "expected-marker"

    def test_on_event_receives_stream(self, tmp_path):
        provider = ScriptedProvider([_resp([TextBlock("hello build")])])
        seen: list[str] = []
        run_build(_factory(provider), BuildSpec(task="x"), tmp_path / "ws",
                  on_event=lambda e: seen.append(type(e).__name__))
        assert "TurnEnd" in seen

    def test_default_checks_is_a_unittest_gate(self):
        checks = default_checks()
        argv, expect_exit, substring = checks[0]
        assert "-m" in argv and "unittest" in argv and expect_exit == 0
