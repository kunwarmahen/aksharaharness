"""Background bash: start/poll/kill against real short-lived processes.

Real subprocesses (echo, sleep) keep the pin honest -- this is process
lifecycle, not string shaping. Every wait is deadline-bounded so a
regression fails the test instead of hanging it.
"""

from __future__ import annotations

import time

import pytest

from akshara.errors import ToolError
from akshara.tools import BashKill, BashPoll, BashStart, JobManager
from akshara.tools.background import MAX_JOBS, Job
from akshara.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


@pytest.fixture
def jobs() -> JobManager:
    return JobManager()


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TestStartPoll:
    def test_start_returns_job_id_and_log_hint(self, jobs, ctx):
        out = BashStart(jobs).run({"command": "echo hi"}, ctx)
        assert "started job-1" in out
        assert ".akshara/jobs/job-1.log" in out

    def test_finished_job_polls_exit_code_and_output(self, jobs, ctx):
        BashStart(jobs).run({"command": "echo all done"}, ctx)
        assert wait_until(lambda: not jobs.get("job-1").running)
        out = BashPoll(jobs).run({"job_id": "job-1"}, ctx)
        assert "EXITED code=0" in out
        assert "all done" in out

    def test_running_job_reports_running(self, jobs, ctx):
        BashStart(jobs).run({"command": "sleep 5"}, ctx)
        out = BashPoll(jobs).run({"job_id": "job-1"}, ctx)
        assert "RUNNING" in out
        assert BashKill(jobs).run({"job_id": "job-1"}, ctx)  # cleanup

    def test_stderr_lands_in_the_log(self, jobs, ctx):
        BashStart(jobs).run({"command": "echo oops >&2"}, ctx)
        assert wait_until(lambda: jobs.get("job-1").log_path.exists()
                          and jobs.get("job-1").log_path.read_text().strip())
        assert "oops" in jobs.get("job-1").log_path.read_text()

    def test_nonzero_exit_reported_not_raised(self, jobs, ctx):
        BashStart(jobs).run({"command": "exit 7"}, ctx)
        assert wait_until(lambda: not jobs.get("job-1").running)
        assert "code=7" in BashPoll(jobs).run({"job_id": "job-1"}, ctx)


class TestListAndKill:
    def test_poll_without_id_lists_every_job(self, jobs, ctx):
        BashStart(jobs).run({"command": "sleep 5"}, ctx)
        out = BashPoll(jobs).run({}, ctx)
        assert "job-1" in out and "sleep 5" in out
        assert BashKill(jobs).run({"job_id": "job-1"}, ctx)

    def test_empty_list_hints_at_start(self, jobs, ctx):
        assert "bash_start" in BashPoll(jobs).run({}, ctx)

    def test_kill_stops_a_running_job(self, jobs, ctx):
        BashStart(jobs).run({"command": "sleep 30"}, ctx)
        assert wait_until(lambda: jobs.get("job-1").running)
        out = BashKill(jobs).run({"job_id": "job-1"}, ctx)
        assert "killed job-1" in out
        assert wait_until(lambda: not jobs.get("job-1").running)

    def test_kill_tree_takes_children(self, jobs, ctx):
        # a backgrounded child inherits the group; killing the job must
        # reap it too, or `sleep 30` orphans linger past the harness
        BashStart(jobs).run({"command": "sleep 30 & sleep 30"}, ctx)
        assert wait_until(lambda: jobs.get("job-1").running)
        BashKill(jobs).run({"job_id": "job-1"}, ctx)
        assert wait_until(lambda: not jobs.get("job-1").running)

    def test_already_exited_job_reports_cleanly(self, jobs, ctx):
        BashStart(jobs).run({"command": "true"}, ctx)
        assert wait_until(lambda: not jobs.get("job-1").running)
        out = BashKill(jobs).run({"job_id": "job-1"}, ctx)
        assert "already exited" in out and "nothing to kill" in out


class TestValidation:
    def test_unknown_job_id_is_model_readable(self, jobs, ctx):
        with pytest.raises(ToolError, match="no such job 'job-9'"):
            BashPoll(jobs).run({"job_id": "job-9"}, ctx)

    def test_chars_cap_enforced(self, jobs, ctx):
        with pytest.raises(ToolError, match="chars must be"):
            BashPoll(jobs).run({"job_id": "x", "chars": 0}, ctx)

    def test_concurrent_run_cap(self, jobs, ctx):
        class FakeProc:
            def poll(self):
                return None  # eternally running

        for i in range(MAX_JOBS):
            jobs._jobs[f"job-{i}"] = Job(f"job-{i}", "sleep ∞",
                                         FakeProc(), ctx.cwd / "x.log",
                                         time.monotonic())
        with pytest.raises(ToolError, match=f"max {MAX_JOBS}"):
            BashStart(jobs).run({"command": "one too many"}, ctx)


def test_gating_flags():
    assert BashStart.read_only is False   # starts processes: gates
    assert BashKill.read_only is False    # kills processes: gates
    assert BashPoll.read_only is True     # reads a log file: auto-approved
