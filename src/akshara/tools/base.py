"""Tool abstractions: what a tool IS, and where tools live.

A tool is three things glued together:

* ``parameters``   -- hand-written JSON Schema. Writing schemas by hand
  (with real ``description`` strings, ``required`` lists, and
  ``additionalProperties: false``) is part of the lesson: vague schemas
  produce vague tool calls.
* ``summary(args)`` -- what the PERMISSION PROMPT shows a human before a
  dangerous call runs ("bash: rm -rf ..."). Built by the tool because the
  tool knows its own shape (diffs for writes, the literal command for bash).
* ``run(args, ctx)`` -- do the thing, return a string. Raise ToolError
  for anything the MODEL could plausibly fix; the agent loop converts
  that into an error result and keeps going.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.leases import LeaseManager
from akshara.types import ImageBlock, ToolSpec


@dataclass(slots=True)
class ToolContext:
    """What tools know about their environment.

    ``cwd`` doubles as the sandbox root for file tools AND the working
    directory for bash. It is convenience confinement, NOT security --
    bash can cd anywhere; that is exactly why bash is permission-gated.

    ``leases`` is shared by every call running against this context --
    which is how parallel batch workers serialize conflicting writes.
    """

    cwd: Path
    leases: LeaseManager = field(default_factory=LeaseManager)


@dataclass(slots=True)
class ToolOutput:
    """A tool result that text alone can't carry: text PLUS images.

    Every tool returns a str and always may again -- but read_image has
    pixels to deliver, so its run() returns one of these instead. The
    loop splits it: ``text`` becomes the ToolResult content (truncated,
    displayed, persisted exactly like any other result) and ``images``
    are hoisted onto history right after it. Returning plain str from a
    tool stays the norm; this exists so ONE capability doesn't drag
    every tool through a richer contract.
    """

    text: str
    images: list[ImageBlock] = field(default_factory=list)


class Tool(ABC):
    """Base class for every tool."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    read_only: ClassVar[bool] = False  # drives permission auto-approval

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description,
                        parameters=self.parameters)

    @abstractmethod
    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Human-readable preview of what this call will do -- shown in
        the permission prompt BEFORE it runs. Same context as run(), so
        previews resolve paths exactly like execution will."""

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> "str | ToolOutput":
        """Execute; return output as a string -- or a ToolOutput when the
        result carries images alongside the text."""

    async def arun(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Async skin over run(): the ONE method the async loop calls.

        Default: push the blocking ``run()`` onto a worker thread. That
        is deliberate, not a stopgap -- our tools do blocking syscall IO
        (open/subprocess/os.replace), and declaring blocking syscalls
        ``async`` without a true async backend would block the event
        loop while lying about it. One implementation per tool, honest
        concurrency at both layers. A tool with a genuinely non-blocking
        backend may override this.
        """
        return await asyncio.to_thread(self.run, args, ctx)


class ToolRegistry:
    """Name -> Tool mapping with duplicate protection."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by exact name; False when it was not registered."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"no such tool: {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def require_str(args: dict[str, Any], key: str, *, optional: bool = False,
                default: str | None = None) -> str:
    """Pull a required string argument out of model-supplied args.

    Tools receive whatever JSON the model produced -- validate defensively
    and raise ToolError the model can read and correct.
    """
    value = args.get(key)
    if value is None:
        if optional:
            return default or ""
        raise ToolError(f"missing required argument {key!r}")
    if not isinstance(value, str):
        raise ToolError(f"argument {key!r} must be a string, got {type(value).__name__}")
    return value


def require_int(args: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = args.get(key)
    if value is None:
        if default is None:
            raise ToolError(f"missing required argument {key!r}")
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"argument {key!r} must be an integer, got {type(value).__name__}")
    return value
