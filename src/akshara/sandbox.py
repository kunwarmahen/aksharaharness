"""Where bash actually runs -- the pluggable sandbox protocol.

Book-inspired (agent-harness-book ch14), with one deliberate step
beyond it: the book ships only a dev-grade subprocess fallback behind a
``ToolSandbox`` protocol and waves at containers for production. This
machine HAS bubblewrap and unprivileged user namespaces, so the
production-shaped backend is real, not aspirational.

Two implementations behind one protocol:

* ``SubprocessSandbox`` -- today's behavior exactly (Popen in the
  workspace, own process group). Confinement is CONVENIENCE, not
  security: bash can cd anywhere and reach the network. This is why
  unsandboxed bash is permission-gated.
* ``BwrapSandbox`` -- bubblewrap: network OFF (--unshare-net), its own
  pid namespace (--unshare-pid), scrubbed environment (--clearenv),
  host filesystem mounted READ-ONLY except the workspace. A command
  that "escapes" finds /usr read-only, no route to the internet, and
  an env without your API keys.

GATES DECIDE, SANDBOXES CONTAIN (the two layers stay orthogonal): the
sandbox bounds the blast radius of an APPROVED call; it never decides
approval itself. The shipped bridge is ``trust_sandbox`` -- a gate
decorator that auto-approves bash ONLY while it is genuinely confined,
so autonomous builds can run without --yolo when bwrap is available.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Protocol

#: Environment variables handed to every sandboxed process. An ALLOWLIST,
#: not a blocklist: anything not named here (API keys, proxy creds, AWS
#: profiles...) simply does not exist inside the sandbox.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TERM", "SHELL")

# Module-level so tests can intercept without real sleeping/waiting.
_popen = subprocess.Popen
_killpg = os.killpg


class CommandTimedOut(Exception):
    """The sandbox killed the command at ``timeout`` seconds.

    Carries whatever output was already produced -- errors live at the
    END of output, which is exactly what a timeout amputates, so the
    partial text is usually the traceback the model needs.
    """

    def __init__(self, timeout: int, salvaged: str) -> None:
        self.timeout = timeout
        self.salvaged = salvaged
        super().__init__(f"command timed out after {timeout}s")


def _child_env(workspace: Path) -> dict[str, str]:
    """Allowlist env + pagers pinned non-interactive + HOME at the workspace."""
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
    env |= {"PAGER": "cat", "GIT_PAGER": "cat",  # no interactive pagers
            "HOME": str(workspace)}
    return env


class ToolSandbox(Protocol):
    """The one surface the bash tool knows about."""

    def execute(self, command: list[str], *, cwd: Path,
                timeout: int) -> tuple[int, str]:
        """Run argv in ``cwd``; return (exit_code, combined stdout+stderr).
        Raises CommandTimedOut (with salvage) or re-raises cancellation."""
        ...
        raise NotImplementedError

    @property
    def describe(self) -> str:
        """One line for banners/summaries: what this sandbox contains."""
        ...

    @property
    def confined(self) -> bool:
        """True => kernel-enforced containment (justifies auto-approval)."""
        ...


class SubprocessSandbox:
    """Direct subprocess in the workspace -- convenience confinement.

    The honest baseline: same trust level as running commands yourself
    in that directory. Env IS scrubbed (allowlist), which closes the
    accidental-leak hole but nothing else.
    """

    @property
    def describe(self) -> str:
        return "subprocess (env-scrubbed; NOT isolated: fs + network reachable)"

    @property
    def confined(self) -> bool:
        return False

    def execute(self, command: list[str], *, cwd: Path,
                timeout: int) -> tuple[int, str]:
        # start_new_session: the child gets its OWN process group, so a
        # timeout/Ctrl-C here can kill the whole tree without touching us.
        process = _popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # interleave; order matters when debugging
            env=_child_env(cwd),
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _killpg(process.pid, signal.SIGKILL)
            # communicate() AGAIN after the kill: per the subprocess docs,
            # retrying loses no output -- the first call may have already
            # buffered stdout in its reader threads (reading the pipe
            # directly would race them and often get nothing).
            output, _ = process.communicate()
            raise CommandTimedOut(
                timeout, output.decode("utf-8", errors="replace").strip())
        except BaseException:
            # Ctrl-C (or generator close): the child never saw the
            # interrupt -- kill the group or it outlives the turn.
            # Re-raise: cancellation is not a tool failure.
            _killpg(process.pid, signal.SIGKILL)
            process.communicate()  # reap; output irrelevant
            raise
        return process.returncode, output.decode("utf-8", errors="replace")


class BwrapSandbox:
    """bubblewrap confinement: no network, no host processes, read-only
    host filesystem, workspace the single writable path.

    How the wall is built, flag by flag:

    * ``--ro-bind /usr /usr`` (+ bind-tries for /bin, /lib*, ld.so.cache)
      -- the toolchain (bash, python, git...) runs from READ-ONLY mounts;
      writes to anywhere outside the workspace fail at the syscall layer.
    * NOT binding /etc wholesale on purpose: inside a user namespace our
      fake root maps onto files owned by real root, so a careless whole-/etc
      mount could make shadow-style files readable. Bind the few bits
      dynamic linking and timezones need instead.
    * ``--unshare-pid`` -- the command sees only its own process tree AND,
      when we SIGKILL bwrap, the kernel kills the namespace's init, taking
      every child with it. The orphan-escape bug class from notes/04 dies
      here structurally, not by careful signal plumbing.
    * ``--unshare-net`` -- loopback only. Exfiltration needs egress.
    * ``--clearenv`` + explicit --setenv -- the allowlist env from
      :func:`_child_env`, rebuilt inside the namespace.

    Write semantics OUTSIDE the workspace (verified live): unmounted
    host paths don't exist ("No such file or directory"); paths under
    the mounts we DO create (/etc's cache anchor, /tmp) accept writes
    into an EPHEMERAL private tmpfs that vanishes when the call ends --
    they never reach the host. Either way the model must treat the
    workspace as the one durable place for artifacts.
    """

    def __init__(self, *, allow_network: bool = False,
                 bwrap_path: str | None = None) -> None:
        self.bwrap = bwrap_path or shutil.which("bwrap")
        if self.bwrap is None:
            raise RuntimeError("bwrap not found on PATH")
        self.allow_network = allow_network

    @property
    def describe(self) -> str:
        net = "network ON" if self.allow_network else "no network"
        return f"bubblewrap ({net}, pid ns, host read-only, workspace writable)"

    @property
    def confined(self) -> bool:
        return True

    def _argv(self, command: list[str], cwd: Path) -> list[str]:
        argv = [
            self.bwrap,
            "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            *([] if self.allow_network else ["--unshare-net"]),
            "--dev", "/dev",          # fresh minimal device tree, not host /dev
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/bin", "/bin",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--clearenv",
        ]
        env = _child_env(cwd)
        for key, value in env.items():
            argv += [f"--setenv", key, value]
        argv += [
            "--bind", str(cwd), str(cwd),   # THE writable place
            "--chdir", str(cwd),
            *command,
        ]
        return argv

    def execute(self, command: list[str], *, cwd: Path,
                timeout: int) -> tuple[int, str]:
        process = _popen(
            self._argv(command, cwd),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            raise CommandTimedOut(
                timeout, output.decode("utf-8", errors="replace").strip())
        except BaseException:
            _killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return process.returncode, output.decode("utf-8", errors="replace")


def autodetect(*, prefer_network_off: bool = True) -> ToolSandbox:
    """Best available sandbox, never raising: bwrap when a probe run
    works (user namespaces can be disabled by sysctl), plain subprocess
    otherwise. The probe uses the REAL flag set minus the workspace --
    cheap, and it exercises exactly the kernel features we depend on."""
    bwrap = shutil.which("bwrap")
    if bwrap is not None:
        probe = [
            bwrap, "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            *([] if prefer_network_off else ["--unshare-net"]),
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--ro-bind", "/", "/", "--clearenv",
            "--setenv", "PATH", "/usr/bin:/bin",
            "/bin/true",
        ]
        try:
            done = subprocess.run(probe, capture_output=True, timeout=10)
            if done.returncode == 0:
                return BwrapSandbox(allow_network=not prefer_network_off,
                                    bwrap_path=bwrap)
        except (OSError, subprocess.SubprocessError):
            pass
    return SubprocessSandbox()
