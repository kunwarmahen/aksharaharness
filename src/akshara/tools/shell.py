"""The bash tool -- the most powerful and least contained tool.

Execution lives in a pluggable ``ToolSandbox`` (see akshara.sandbox):
``SubprocessSandbox`` is today's behavior (direct Popen, convenience
confinement only); ``BwrapSandbox`` adds kernel-level walls (no network,
read-only host, workspace-only writes). The tool itself owns the parts
the MODEL sees -- argument validation, the permission summary, and
turning a timeout into an error result that includes the salvaged
partial output.

This tool is never read_only, so it always goes through the permission
gate unless --yolo (or trust_sandbox with a confined backend) is set.
"""

from __future__ import annotations

from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.sandbox import CommandTimedOut, SubprocessSandbox, ToolSandbox
from akshara.tools.base import Tool, ToolContext, require_int, require_str

MAX_OUTPUT_CHARS = 10_000


def _head_tail(text: str, cap: int = MAX_OUTPUT_CHARS) -> str:
    """Keep both ends of long output: errors live at the end, banners
    and tracebacks at the start."""
    if len(text) <= cap:
        return text
    keep = cap // 2
    omitted = len(text) - cap
    return f"{text[:keep]}\n[... {omitted} chars omitted ...]\n{text[-keep:]}"


class Bash(Tool):
    name = "bash"
    description = (
        "Run a shell command in the sandbox directory and return "
        "stdout+stderr with the exit code. Use for git, builds, tests, "
        "and anything the file tools can't do."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 30)."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox: ToolSandbox | None = None) -> None:
        # Default keeps the historical behavior byte-for-byte; wire a
        # BwrapSandbox (or autodetect()) to actually contain commands.
        self.sandbox = sandbox or SubprocessSandbox()

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"$ {require_str(args, 'command')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        command = require_str(args, "command")
        timeout = require_int(args, "timeout", default=30)
        if timeout < 1 or timeout > 600:
            raise ToolError("timeout must be between 1 and 600 seconds")

        try:
            code, output = self.sandbox.execute(
                ["bash", "-c", command], cwd=ctx.cwd, timeout=timeout)
        except CommandTimedOut as exc:
            detail = f" last output:\n{exc.salvaged}" if exc.salvaged else ""
            raise ToolError(f"command timed out after {timeout}s.{detail}") from None

        result = _head_tail(output.strip())
        return f"{result}\nexit code: {code}"
