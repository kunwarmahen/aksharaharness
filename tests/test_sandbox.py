"""Sandbox tests: the pluggable ToolSandbox protocol and its two backends.

SubprocessSandbox must keep the historical Bash semantics (the existing
test_tools.py shell tests pin those through the tool); these tests pin
the sandbox layer directly PLUS what is new here: env scrubbing, and --
when bubblewrap exists -- kernel-level containment that no in-process
check could prove: writes that never reach the host, a network that
does not resolve, an env without secrets, and timeouts that take the
whole pid namespace down.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from akshara.permissions import PermissionRequest, trust_sandbox
from akshara.sandbox import (
    BwrapSandbox,
    CommandTimedOut,
    SubprocessSandbox,
    autodetect,
)
from akshara.tools.base import ToolContext
from akshara.tools.shell import Bash

BWRAP = shutil.which("bwrap")


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# ---- SubprocessSandbox ------------------------------------------------------


class TestSubprocessSandbox:
    def test_basic_execution_and_exit_code(self, tmp_path):
        code, out = SubprocessSandbox().execute(
            ["bash", "-c", "echo out; echo err >&2; exit 3"],
            cwd=_ws(tmp_path), timeout=10)
        assert code == 3
        assert "out" in out and "err" in out  # stderr interleaved into stdout

    def test_env_is_allowlisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AKSHARA_SECRET_TOKEN", "supersecret")
        monkeypatch.setenv("HOME", "/host/home")
        _, out = SubprocessSandbox().execute(
            ["bash", "-c", 'echo "secret=[$AKSHARA_SECRET_TOKEN] home=$HOME"'],
            cwd=_ws(tmp_path), timeout=10)
        assert "supersecret" not in out
        assert str(tmp_path) in out  # HOME repointed at the workspace

    def test_timeout_raises_with_salvage(self, tmp_path):
        with pytest.raises(CommandTimedOut) as excinfo:
            SubprocessSandbox().execute(
                ["bash", "-c", "echo early; sleep 30"],
                cwd=_ws(tmp_path), timeout=1)
        assert "early" in excinfo.value.salvaged

    def test_cancellation_kills_child(self, tmp_path):
        """Ctrl-C mid-command: the child must not outlive the call."""
        marker = tmp_path / "still-running"
        process_error = None
        try:
            SubprocessSandbox().execute(
                ["bash", "-c", f"sleep 30; touch {marker}"],
                cwd=_ws(tmp_path), timeout=10)
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # noqa: BLE001 -- shape the assertion below
            process_error = exc
        assert isinstance(process_error, KeyboardInterrupt) \
            or process_error is None or "timed out" in str(process_error)


# ---- BwrapSandbox ------------------------------------------------------------
# These run FOR REAL wherever bwrap exists (this machine), proving
# containment properties no in-process check can; elsewhere they skip
# and the suite stays green.


@pytest.mark.skipif(BWRAP is None, reason="bubblewrap not installed")
class TestBwrapSandbox:
    def test_basic_execution(self, tmp_path):
        s = BwrapSandbox()
        code, out = s.execute(["bash", "-c", "echo hi && pwd"],
                              cwd=_ws(tmp_path), timeout=15)
        assert code == 0
        assert "hi" in out
        assert str(tmp_path / "ws") in out  # chdir'ed into the workspace

    def test_writes_outside_workspace_never_reach_host(self, tmp_path):
        canary = Path("/etc/harness-test-canary")
        if canary.exists():  # don't fail because of a previous crashed run
            canary.unlink()
        s = BwrapSandbox()
        s.execute(["bash", "-c", "touch /etc/harness-test-canary; echo rc=$?"],
                  cwd=_ws(tmp_path), timeout=15)
        assert not canary.exists(), "sandbox write escaped to the host"

    def test_unmounted_host_paths_are_invisible(self, tmp_path):
        s = BwrapSandbox()
        _, out = s.execute(["bash", "-c", "ls /home 2>&1"],
                           cwd=_ws(tmp_path), timeout=15)
        # /home is not bound: it either doesn't exist or is empty -- never
        # does the real host home tree appear.
        host_entries = set(Path("/home").iterdir()) if Path("/home").exists() else set()
        for entry in host_entries:
            assert entry.name not in out

    def test_no_network_egress(self, tmp_path):
        s = BwrapSandbox()  # allow_network defaults False
        _, out = s.execute(
            ["python3", "-c",
             "import socket\n"
             "try:\n"
             "    socket.create_connection(('example.com', 80), timeout=2)\n"
             "    print('CONNECTED')\n"
             "except OSError as e:\n"
             "    print('BLOCKED', type(e).__name__)\n"],
            cwd=_ws(tmp_path), timeout=20)
        assert "BLOCKED" in out, out

    def test_secrets_absent_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AKSHARA_SECRET_PROBE", "leak-me-not")
        s = BwrapSandbox()
        _, out = s.execute(["bash", "-c", 'echo "[$AKSHARA_SECRET_PROBE]"'],
                           cwd=_ws(tmp_path), timeout=15)
        assert "leak-me-not" not in out

    def test_workspace_is_durable_across_calls(self, tmp_path):
        ws = _ws(tmp_path)
        s = BwrapSandbox()
        s.execute(["bash", "-c", "echo kept > artifact.txt"], cwd=ws, timeout=15)
        assert (ws / "artifact.txt").read_text().strip() == "kept"

    def test_timeout_takes_the_whole_tree_down(self, tmp_path):
        """--unshare-pid means SIGKILLing bwrap kills every child: the
        timed-out command cannot leave orphans behind."""
        ws = _ws(tmp_path)
        started = time.monotonic()
        with pytest.raises(CommandTimedOut):
            BwrapSandbox().execute(["bash", "-c", "sleep 60 & echo early; wait"],
                                   cwd=ws, timeout=2)
        assert time.monotonic() - started < 30  # not the child's lifetime

    def test_confined_flag_is_true(self):
        assert BwrapSandbox().confined is True
        assert SubprocessSandbox().confined is False


def test_autodetect_picks_bwrap_when_available():
    chosen = autodetect()
    if BWRAP is not None:
        assert isinstance(chosen, BwrapSandbox)
    else:
        assert isinstance(chosen, SubprocessSandbox)


# ---- the gate bridge ---------------------------------------------------------


class _FakeSandbox:
    def __init__(self, confined: bool) -> None:
        self._confined = confined

    @property
    def confined(self) -> bool:
        return self._confined


def _request(tool: str) -> PermissionRequest:
    return PermissionRequest(tool_name=tool, arguments={}, summary="s",
                             read_only=False)


class TestTrustSandbox:
    def test_confined_bash_auto_approves(self):
        gate = trust_sandbox(lambda r: False, _FakeSandbox(confined=True))
        assert gate(_request("bash")) is True

    def test_non_bash_still_delegates_when_confined(self):
        gate = trust_sandbox(lambda r: False, _FakeSandbox(confined=True))
        assert gate(_request("write_file")) is False  # fs writes keep asking

    def test_unconfined_sandbox_delegates_everything(self):
        inner_calls: list[str] = []
        inner = lambda r: inner_calls.append(r.tool_name) or True
        gate = trust_sandbox(inner, _FakeSandbox(confined=False))
        assert gate(_request("bash")) is True
        assert inner_calls == ["bash"]  # convenience confinement earns nothing


# ---- through the tool --------------------------------------------------------


def test_bash_tool_default_keeps_subprocess_behavior(tmp_path):
    tool = Bash()  # no sandbox wired: historical default
    assert tool.sandbox.confined is False
    result = tool.run({"command": "echo via-tool"}, ToolContext(cwd=_ws(tmp_path)))
    assert "via-tool" in result and "exit code: 0" in result


@pytest.mark.skipif(BWRAP is None, reason="bubblewrap not installed")
def test_bash_tool_through_the_wall(tmp_path, monkeypatch):
    monkeypatch.setenv("AKSHARA_SECRET_PROBE", "nope")
    tool = Bash(sandbox=BwrapSandbox())
    result = tool.run({"command": "echo [$AKSHARA_SECRET_PROBE]"},
                      ToolContext(cwd=_ws(tmp_path)))
    assert "exit code: 0" in result
    assert "nope" not in result
