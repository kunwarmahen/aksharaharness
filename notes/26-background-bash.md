# 26 — background bash: commands that outlive their tool call

*Synchronous bash caps every command at ten minutes and blocks the
whole batch while it runs. That fits "run pytest" and rules out
everything that is a PHASE rather than a step: the dev server you want
to curl afterwards, the watch build, the long pipeline. This note is
about giving the model processes of its own — and about the trust
letter that choice leaves open.*

## Three verbs, not one mega-tool

- `bash_start(command)` → returns a job id immediately. Output tees to
  `.akshara/jobs/<id>.log` FROM BIRTH (a `Popen` writing straight to
  the file) so a three-hour job never grows a three-hour string in RAM.
- `bash_poll([job_id])` → one job's status plus recent output, or an
  index of every job with no argument. Doubles as the reaper: poll()
  is what notices exit.
- `bash_kill(job_id)` → SIGKILL to the process GROUP, so a job's
  children die with it (`sleep 30 &` orphans are how daemons leak).

Small single-purpose tools keep schemas honest; one do-everything tool
with an `action` enum teaches the model to guess.

## The honesty section

Two deliberate deviations from "bash but backgrounded," both written
into the module docstring because they're the actual lesson:

**Background jobs bypass the sandbox.** `ToolSandbox.execute()` is
synchronous by shape — it IS the wait. And wrapping bubblewrap around
a job that must outlive its call ties the job's lifetime to the
harness process anyway. So jobs run as plain env-scrubbed subprocesses
(the same allowlist `_child_env` builds for every sandbox), and — the
load-bearing part — **they gate individually.** `trust_sandbox`
auto-approves only the synchronous bash tool; bash_start always asks,
even when confined bash doesn't. Containment earning approval must not
silently extend to a code path without containment.

**Jobs belong to the harness process, on purpose.** A started server
outliving the session is usually the POINT ("start postgres, keep it
running"). After a restart the ids are gone but the logs stay on disk
where read_file still reaches them. Append-mode log opens mean a
restarted harness writing the same path can't clobber evidence.

## Small mechanics worth recording

- Job ids are a per-process counter (`job-1`, `job-2`): unique within
  anything that can address them, no uuid noise in transcripts.
- A running-jobs cap (8) turns "model leaks servers" into a readable
  error instead of resource exhaustion.
- poll output clips TAIL-only (4k chars) — progress bars and errors
  live at the END of logs; head-tail would waste half the window on a
  banner from hour one.
- `process.poll()` is the whole zombie story: calling it reaps.

## What the tests pin

Real short-lived subprocesses throughout (echo/sleep), every wait
deadline-bounded:

- start returns id + log hint; finished job polls exit code AND output
- stderr lands in the log; nonzero exit reported not raised
- kill stops a running job; killing takes the CHILDREN too (the tree)
- already-exited job kills cleanly ("nothing to kill")
- unknown id → ToolError listing known jobs; chars cap enforced;
  concurrent-run cap refused before Popen
- gating: start/kill prompt, poll auto-approves (it reads a file)

## Receipts

Offline suite green against real processes.

Live receipt (Ollama `qwen3.8`, one-shot `--yolo`): one turn,
4 iterations — `bash_start("echo job-says-hello")` → `job-1`, poll
reported `EXITED code=0` with output `job-says-hello`; the same turn
also fetched a page via web_fetch and recorded a 3-item todo list
(all `[x]`). Turn footer: `3047 in / 95 out · 4 iteration(s)`.
