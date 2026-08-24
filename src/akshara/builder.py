"""The build job as a library primitive -- spec in, verified project out.

Claude Code's core loop applied to whole projects: read a spec -> write
files -> RUN the acceptance commands -> read failures -> fix -> repeat.
Promoted here from examples/builder_demo.py because "can it actually
build things" is the question every agent harness must answer for real,
and the answer belongs to the harness core, not a demo script.

THE CONTRACT IS VERIFICATION, NOT TRUST. ``run_build`` never reads the
model's claims about what works:

* acceptance commands are RE-RUN by this module, independently, after
  the turn ends (exit codes + required substrings);
* every ``test_*.py`` present when the build starts is checksummed --
  a repair-style job cannot satisfy "make it green" by weakening the
  tests, because modified tests fail the build outright;
* the exit code doubles as a CI gate: ``BuildResult.ok`` -> 0, else 1.

Workspace layout is the caller's choice (a tempdir for demos,
``.akshara/builds/<ts>/`` for the CLI); fs tools confine themselves to
it via their existing sandbox root, so file writes stay inside no
matter how the model words its paths.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from akshara.agent import Agent, AgentEvent, TurnEnd

#: The system prompt for building agents. Acting over describing; short
#: narration; the model's own bugs are its problem to read and fix.
BUILD_SYSTEM = (
    "You are a software engineer working inside an empty project directory. "
    "You are confined to it: never create files outside it. Prefer acting "
    "over describing -- create the files, RUN every acceptance command, read "
    "failures carefully, fix the code, and re-run until everything passes. "
    "Keep narration to one short line per step."
)


def default_checks() -> list[tuple[list[str], int, str | None]]:
    """Generic acceptance gate: a unittest suite that must pass."""
    return [([sys.executable, "-m", "unittest", "discover", "-s", ".", "-v"],
             0, None)]


@dataclass(frozen=True)
class BuildSpec:
    """One build job.

    ``task``          -- the spec text shown to the model.
    ``checks``        -- (argv, expect_exit, expect_substring) triples,
                         re-run independently after the turn.
    ``seed_dir``      -- optional directory copied into the workspace
                         first (repair-style jobs start BROKEN).
    ``checksum_tests``-- hash every test_*.py present at start; any
                         later modification fails the build.
    ``max_repair_rounds`` -- when independent verification comes back RED,
                         the failures are fed back into the SAME
                         conversation this many times before giving up.
                         Tampering is never repaired -- it ends the build.
    """

    task: str
    checks: list[tuple[list[str], int, str | None]] = field(
        default_factory=default_checks)
    seed_dir: Path | None = None
    checksum_tests: bool = True
    max_repair_rounds: int = 2


@dataclass(frozen=True)
class CheckOutcome:
    """What one acceptance command actually did."""

    argv: list[str]
    passed: bool
    actual_exit: int
    expect_exit: int
    missing_substring: str | None = None  # set when the substring check failed
    tail: str = ""                        # output tail on failure


@dataclass(frozen=True)
class BuildResult:
    """Everything a caller (or CI) needs."""

    ok: bool
    elapsed_seconds: float
    files: list[str]
    response_text: str
    checks: list[CheckOutcome]
    tampered_tests: list[str]
    usage_in: int
    usage_out: int
    iterations: int | None  # None when the turn ended abnormally


def _seed(workspace: Path, seed_dir: Path) -> int:
    """Copy a broken project into place; return how many files landed."""
    count = 0
    for src in sorted(seed_dir.iterdir()):
        if src.is_file():
            shutil.copy(src, workspace / src.name)
            count += 1
    return count


def _checksum_tests(workspace: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(workspace.glob("test_*.py"))}


def _run_checks(workspace: Path,
                checks: list[tuple[list[str], int, str | None]],
                ) -> list[CheckOutcome]:
    """Independently re-run every acceptance command IN the workspace."""
    outcomes: list[CheckOutcome] = []
    for argv, expect_exit, expect_substring in checks:
        try:
            proc = subprocess.run(argv, cwd=workspace, capture_output=True,
                                  text=True, timeout=120)
        except subprocess.TimeoutExpired:
            outcomes.append(CheckOutcome(
                argv=argv, passed=False, actual_exit=-999,
                expect_exit=expect_exit, missing_substring=None,
                tail="(verification command timed out)"))
            continue
        combined = proc.stdout + proc.stderr
        exit_ok = proc.returncode == expect_exit
        # Substrings are matched against stdout on a clean exit (acceptance
        # output goes there by convention), against everything otherwise --
        # a failing run's evidence usually lands on stderr.
        missing = None
        haystack = proc.stdout if exit_ok else combined
        if exit_ok and expect_substring is not None \
                and expect_substring not in haystack:
            missing = expect_substring
        passed = exit_ok and missing is None
        outcomes.append(CheckOutcome(
            argv=argv, passed=passed, actual_exit=proc.returncode,
            expect_exit=expect_exit, missing_substring=missing,
            tail="" if passed else combined.strip()[-300:],
        ))
    return outcomes


def _failure_report(outcomes: list[CheckOutcome], tampered: list[str]) -> str:
    """The verification verdict, as feedback the model can act on."""
    lines = ["Your build FAILED the harness's independent verification",
             "(the acceptance commands were re-run by the harness itself):", ""]
    for outcome in outcomes:
        if outcome.passed:
            continue
        argv = " ".join(outcome.argv)
        lines.append(f"FAIL: {argv}")
        lines.append(f"  exit code {outcome.actual_exit} "
                     f"(expected {outcome.expect_exit})")
        if outcome.missing_substring is not None:
            lines.append(f"  output did not contain: {outcome.missing_substring!r}")
        if outcome.tail:
            lines.append("  output tail:")
            lines.extend(f"    {line}" for line in outcome.tail.splitlines()[-10:])
        lines.append("")
    lines.append("Fix the project until every check passes. Do NOT modify "
                 "existing test files -- they are checksummed and any edit "
                 "fails the build outright.")
    return "\n".join(lines)


def run_build(agent_factory: Callable[[Path], Agent],
              spec: BuildSpec, workspace: Path, *,
              on_event: Callable[[AgentEvent], None] | None = None,
              max_iterations: int = 30) -> BuildResult:
    """Run one build job to completion and verify it independently.

    ``agent_factory(workspace)`` builds the Agent (provider, gate policy
    and tool registry are CALLER decisions -- this module stays out of
    the trust debate). Events stream through ``on_event`` for live UIs.

    A red verification is FED BACK into the same conversation (up to
    ``spec.max_repair_rounds`` times): the model sees exactly what the
    harness saw. Tampering is never repaired -- a modified checksummed
    test ends the build regardless of what else gets fixed.

    KeyboardInterrupt propagates AFTER cleanup-free state is left
    resumable by the loop itself; callers decide their own exit code.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    seeded = _seed(workspace, spec.seed_dir) if spec.seed_dir else 0
    test_hashes = _checksum_tests(workspace) if spec.checksum_tests else {}

    def verify() -> tuple[list[CheckOutcome], list[str]]:
        outcomes = _run_checks(workspace, spec.checks)
        tampered = sorted(
            name for name, digest in test_hashes.items()
            if not (workspace / name).exists()
            or hashlib.sha256(
                (workspace / name).read_bytes()).hexdigest() != digest)
        return outcomes, tampered

    agent = agent_factory(workspace)
    started = time.perf_counter()

    iterations: int | None = None
    response_text = ""

    def drive(prompt: str) -> None:
        nonlocal iterations, response_text
        for event in agent.run_streaming(prompt):
            if on_event is not None:
                on_event(event)
            if isinstance(event, TurnEnd) and event.response is not None:
                iterations = event.iterations
                response_text = event.response.message.text().strip()

    try:
        drive(spec.task)
        outcomes, tampered = verify()
        rounds_left = spec.max_repair_rounds
        while rounds_left > 0 and not tampered \
                and not all(o.passed for o in outcomes):
            rounds_left -= 1
            try:
                drive(_failure_report(outcomes, tampered))
            except Exception:
                # Provider died mid-repair (budget exhausted, script ran
                # dry): the last verification stands. KeyboardInterrupt is
                # a BaseException and still propagates.
                break
            outcomes, tampered = verify()
    finally:
        elapsed = time.perf_counter() - started

    usage = agent.total_usage
    ok = all(o.passed for o in outcomes) and not tampered
    files = sorted(p.name for p in workspace.iterdir() if p.is_file())
    return BuildResult(ok=ok, elapsed_seconds=elapsed, files=files,
                       response_text=response_text, checks=outcomes,
                       tampered_tests=tampered,
                       usage_in=usage.input_tokens,
                       usage_out=usage.output_tokens,
                       iterations=iterations)
