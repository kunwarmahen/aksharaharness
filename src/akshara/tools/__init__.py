"""The default tool set the CLI ships with.

Sixteen built-ins cover the whole autonomy loop: SEE (read_file,
list_dir, glob, grep, read_image), CHANGE (write_file, edit_file,
bash), RUN LONG (bash_start/poll/kill), LOOK OUT (web_fetch),
REMEMBER (write_note/recall_notes for durable facts,
todo_write/todo_read for live plan state). Anything beyond these is
MCP's job -- that's what the mcp__server__tool namespace is for.

One family self-selects: OPERATE (browser_open/click/fill/close on a
headless Chromium) registers ONLY when playwright is importable -- the
optional ``[browse]`` extra is its own opt-in. Users who never asked
for a browser keep sixteen tools; installing it brings twenty.
"""

from __future__ import annotations

from akshara.sandbox import ToolSandbox
from akshara.tools.ask_user import AskUser  # re-exported: CLI wires the channel
from akshara.tools.background import (
    BashKill,
    BashPoll,
    BashStart,
    JobManager,
)
from akshara.tools.base import Tool, ToolContext, ToolOutput, ToolRegistry
from akshara.tools.browser import (
    BrowserClick,
    BrowserClose,
    BrowserFill,
    BrowserOpen,
    BrowserSession,
)
from akshara.tools.fs import EditFile, ListDir, ReadFile, WriteFile
from akshara.tools.glob import Glob
from akshara.tools.memory import RecallNotes, WriteNote
from akshara.tools.read_image import ReadImage
from akshara.tools.search import Grep
from akshara.tools.shell import Bash
from akshara.tools.todo import TodoRead, TodoWrite
from akshara.tools.web_fetch import WebFetch

__all__ = [
    "AskUser",
    "Bash",
    "BashKill",
    "BashPoll",
    "BashStart",
    "BrowserClick",
    "BrowserClose",
    "BrowserFill",
    "BrowserOpen",
    "BrowserSession",
    "EditFile",
    "Glob",
    "Grep",
    "JobManager",
    "ListDir",
    "ReadFile",
    "ReadImage",
    "RecallNotes",
    "TodoRead",
    "TodoWrite",
    "Tool",
    "ToolContext",
    "ToolOutput",
    "ToolRegistry",
    "WebFetch",
    "WriteFile",
    "WriteNote",
    "default_registry",
]


def default_registry(sandbox: ToolSandbox | None = None) -> ToolRegistry:
    """A registry preloaded with the built-ins.

    ``sandbox`` (optional) replaces bash's executor -- pass a
    BwrapSandbox for kernel-level confinement or autodetect() to pick
    the best available; None keeps the plain subprocess behavior.
    Background jobs (bash_start/poll/kill) share one JobManager and
    always run as plain env-scrubbed subprocesses -- they outlive their
    sandbox by design, so they gate individually instead of riding
    trust_sandbox's auto-approval.

    The browser_* family (headless Chromium) joins only when playwright
    is importable -- the [browse] extra is its own opt-in. All four are
    network egress and gate like web_fetch.

    read_only flags drive permission gating automatically: read_file,
    list_dir, glob, grep, read_image, recall_notes, todo_read,
    bash_poll auto-approve; write_file, edit_file, bash, bash_start,
    bash_kill, web_fetch (network egress), browser_open/click/fill/
    close (when registered -- same egress rule), write_note, todo_write
    prompt.
    """
    registry = ToolRegistry()
    jobs = JobManager()
    for tool in (ReadFile(), ListDir(), Glob(), Grep(),
                 WriteFile(), EditFile(),
                 ReadImage(),
                 Bash(sandbox),
                 BashStart(jobs), BashPoll(jobs), BashKill(jobs),
                 WebFetch(),
                 WriteNote(), RecallNotes(),
                 TodoWrite(), TodoRead()):
        registry.register(tool)
    from akshara.tools.browser import browser_tools
    for tool in browser_tools():  # () unless the [browse] extra is installed
        registry.register(tool)
    return registry
