"""The default tool set the CLI ships with."""

from __future__ import annotations

from akshara.sandbox import ToolSandbox
from akshara.tools.base import Tool, ToolContext, ToolRegistry
from akshara.tools.fs import EditFile, ListDir, ReadFile, WriteFile
from akshara.tools.memory import RecallNotes, WriteNote
from akshara.tools.search import Grep
from akshara.tools.shell import Bash

__all__ = [
    "Bash",
    "EditFile",
    "Grep",
    "ListDir",
    "ReadFile",
    "RecallNotes",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "WriteFile",
    "WriteNote",
    "default_registry",
]


def default_registry(sandbox: ToolSandbox | None = None) -> ToolRegistry:
    """A registry preloaded with the built-ins.

    ``sandbox`` (optional) replaces bash's executor -- pass a
    BwrapSandbox for kernel-level confinement or autodetect() to pick
    the best available; None keeps the plain subprocess behavior.

    read_only flags drive permission gating automatically: read_file,
    list_dir, grep, recall_notes auto-approve; write_file, edit_file,
    bash, write_note prompt.
    """
    registry = ToolRegistry()
    for tool in (ReadFile(), ListDir(), Grep(), WriteFile(), EditFile(),
                 Bash(sandbox), WriteNote(), RecallNotes()):
        registry.register(tool)
    return registry
