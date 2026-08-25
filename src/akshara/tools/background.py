"""Background bash -- start long commands now, read their output later.

Synchronous bash caps every command at ten minutes and blocks the whole
batch while it runs, which fits "run pytest" and rules out everything
that is a PHASE rather than a step: the dev server the model wants to
curl afterwards, the watch-mode build, the data pipeline it can work
around meanwhile. Three small verbs fill the gap:

* ``bash_start(command)`` -> job id, output teeing to
  ``.akshara/jobs/<id>.log`` from birth (nothing buffered in memory --
  a three-hour job never grows a three-hour string).
* ``bash_poll([job_id])   -> one job's status + recent output, or an
  index of every job when called without one. Doubles as the reaper:
  poll() is what notices a process exited.
* ``bash_kill(job_id)     -> SIGKILL the whole process group.

Deliberate honesty about trust: background jobs do NOT go through the
ToolSandbox protocol. The sandbox's execute() is synchronous by shape,
and wrapping bubblewrap around a job that must OUTLIVE this call would
tie its lifetime to ours anyway. So jobs are plain env-scrubbed
subprocesses and every one of them gates like unsandboxed bash does --
trust_sandbox's auto-approval covers the synchronous tool only.

Lifecycle is equally honest: jobs belong to the HARNESS PROCESS. A
started server keeps running after the session ends (usually what you
want); after a restart, old ids are gone though their logs stay on
disk where read_file can still reach them.
"""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.sandbox import _child_env, _killpg, _popen
from akshara.tools.base import Tool, ToolContext, require_int, require_str

MAX_JOBS = 8
POLL_TAIL_CHARS = 4_000


def _clip_tail(text: str, cap: int = POLL_TAIL_CHARS) -> str:
    """Keep the END of long logs -- progress bars and errors live there."""
    if len(text) <= cap:
        return text
    return f"[... {len(text) - cap} earlier chars omitted ...]\n{text[-cap:]}"


class Job:
    """One background command: its process, its log, when it began."""

    def __init__(self, job_id: str, command: str, process: subprocess.Popen,
                 log_path: Path, started: float) -> None:
        self.id = job_id
        self.command = command
        self.process = process
        self.log_path = log_path
        self.started = started

    @property
    def running(self) -> bool:
        return self.process.poll() is None  # side effect: reaps zombies


class JobManager:
    """id -> Job, safe against parallel batch workers via one lock."""

    def __init__(self, root: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._next = 0
        self._lock = threading.Lock()
        self.root = root  # set per-start from ctx.cwd; kept for tests

    def start(self, command: str, cwd: Path) -> Job:
        with self._lock:
            running = [j for j in self._jobs.values() if j.running]
            if len(running) >= MAX_JOBS:
                raise ToolError(
                    f"{len(running)} jobs already running (max {MAX_JOBS}) "
                    "-- bash_poll/bash_kill them first")
            self._next += 1
            job_id = f"job-{self._next}"
            log_dir = cwd / ".akshara" / "jobs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job_id}.log"
            # append mode: a restarted harness writing the same path can't
            # clobber evidence of what the first run did
            process = _popen(
                ["bash", "-c", command],
                cwd=cwd,
                stdout=open(log_path, "ab"),
                stderr=subprocess.STDOUT,
                env=_child_env(cwd),
                start_new_session=True,  # own group: kill takes the tree
            )
            job = Job(job_id, command, process, log_path, time.monotonic())
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            known = ", ".join(sorted(self._jobs)) or "(none)"
            raise ToolError(f"no such job {job_id!r}; known jobs: {known}")
        return job

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def kill(self, job: Job) -> bool:
        """SIGKILL the process group. Returns False if already gone."""
        if not job.running:
            return False
        try:
            _killpg(job.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return False  # exited between the poll and the kill
        job.process.wait(timeout=10)
        return True


def _status_line(job: Job) -> str:
    if job.running:
        return f"{job.id} RUNNING ({time.monotonic() - job.started:.0f}s): {job.command}"
    return (f"{job.id} EXITED code={job.process.returncode}: "
            f"{job.command}")


class BashStart(Tool):
    name = "bash_start"
    description = (
        "Start a shell command in the BACKGROUND and return immediately "
        "with a job id. Output streams to .akshara/jobs/<id>.log; use "
        "bash_poll to check on it. For servers, watchers, long builds -- "
        "anything that runs past one tool call. The job keeps running "
        "after the session ends."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, jobs: JobManager) -> None:
        self.jobs = jobs

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"$ {require_str(args, 'command')}  (background)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        command = require_str(args, "command")
        job = self.jobs.start(command, ctx.cwd)
        rel_log = job.log_path.relative_to(ctx.cwd)
        return (f"started {job.id}\nlog: {rel_log} -- bash_poll('{job.id}') "
                "to check output")


class BashPoll(Tool):
    name = "bash_poll"
    description = (
        "Check a background job: whether it is still running or exited "
        "(with which code), plus its recent output. Omit job_id to list "
        "every job. Polling also cleans up finished jobs."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string",
                       "description": "e.g. 'job-3'. Omit to list all jobs."},
            "chars": {"type": "integer",
                      "description": f"How much recent output to show "
                                     f"(default {POLL_TAIL_CHARS})."},
        },
        "additionalProperties": False,
    }
    read_only = True  # reads a log file under .akshara/

    def __init__(self, jobs: JobManager) -> None:
        self.jobs = jobs

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"poll {args.get('job_id', 'all jobs')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        chars = require_int(args, "chars", default=POLL_TAIL_CHARS)
        if chars < 1 or chars > POLL_TAIL_CHARS * 8:
            raise ToolError(f"chars must be between 1 and {POLL_TAIL_CHARS * 8}")

        if args.get("job_id") is None:
            jobs = self.jobs.all()
            if not jobs:
                return ("no background jobs -- bash_start starts one")
            return "\n".join(_status_line(j) for j in sorted(jobs, key=lambda j: j.id))

        job = self.jobs.get(require_str(args, "job_id"))
        try:
            log = job.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            log = "[no output yet]"
        body = _clip_tail(log.strip(), chars)
        return f"{_status_line(job)}\n{body}"


class BashKill(Tool):
    name = "bash_kill"
    description = (
        "Kill a background job and its whole process tree (SIGKILL to "
        "the process group). No effect on jobs that already exited."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "e.g. 'job-3'."},
        },
        "required": ["job_id"],
        "additionalProperties": False,
    }

    def __init__(self, jobs: JobManager) -> None:
        self.jobs = jobs

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        job_id = require_str(args, "job_id")
        return f"kill {job_id}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        job = self.jobs.get(require_str(args, "job_id"))
        if self.jobs.kill(job):
            return f"killed {job.id}"
        return (f"{job.id} had already exited (code={job.process.returncode}); "
                "nothing to kill")
