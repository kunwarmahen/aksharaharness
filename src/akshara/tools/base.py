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
import json
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
    def run(self, args: dict[str, Any], ctx: ToolContext) -> str | ToolOutput:
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
    """Name -> Tool mapping with duplicate protection.

    Tools can also be DISABLED at runtime (``disable``/``enable``): they
    stay registered -- history and checkpoints keep referencing them --
    but disappear from every request's specs, from discovery listings,
    and their calls fail as readable data instead of executing. That is
    the soft twin of ``unregister`` (the AKSHARA_DISABLED_TOOLS startup
    kill-switch, which removes tools outright): a disable is reversible
    mid-session, by the operator, on purpose.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._disabled: set[str] = set()

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by exact name; False when it was not registered."""
        self._disabled.discard(name)
        return self._tools.pop(name, None) is not None

    # ---- runtime enable/disable ---------------------------------------------

    def disable(self, name: str) -> bool:
        """Hide a registered tool from the model until re-enabled. Unknown
        names report False rather than pre-registering a phantom."""
        if name not in self._tools:
            return False
        self._disabled.add(name)
        return True

    def enable(self, name: str) -> bool:
        """Undo a disable (idempotent). False only for unknown names."""
        if name not in self._tools:
            return False
        self._disabled.discard(name)
        return True

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    def disabled_names(self) -> list[str]:
        return sorted(self._disabled)

    def get(self, name: str) -> Tool:
        try:
            tool = self._tools[name]
        except KeyError:
            raise KeyError(f"no such tool: {name!r}") from None
        if name in self._disabled:
            # A distinct message from "no such tool": the model naming it
            # did nothing wrong -- the operator pulled it this session.
            raise KeyError(f"tool {name!r} is disabled by the operator")
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()
                if t.name not in self._disabled]

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


# ---------------------------------------------------------------------------
# Argument coercion: repairing the stringified-JSON habit
# ---------------------------------------------------------------------------
#
# Models -- local ones especially, but not only -- routinely emit a
# non-scalar argument as a STRING of JSON:
#
#     {"symbols": "[\"AAPL\"]"}      instead of      {"symbols": ["AAPL"]}
#
# Nothing in the harness is mis-parsing when that happens. The wire
# carried a string, ``json.loads`` faithfully produced a string, and the
# tool (or an MCP server's JSON-Schema validator) correctly rejects it:
#
#     validating /properties/symbols: type: ["AAPL"] has type "string",
#     want one of "null, array"
#
# The schema is right there, though, and it says what was meant. So we
# repair it rather than make every array-taking tool unusable on models
# with this habit -- which is most of them at small sizes, and this
# project treats local models as a first-class road.
#
# The one rule that keeps this honest: NEVER coerce a parameter whose
# schema also accepts a string. There, a string is a legitimate value and
# "looks like JSON" is not permission to reinterpret it -- a grep pattern
# of "[0-9]" must stay the text the caller typed.


def _declared_types(spec: dict[str, Any]) -> set[str]:
    """JSON-Schema ``type`` as a set. Empty = unstated, so hands off.

    anyOf/oneOf deliberately read as unstated: a union we have not
    reasoned about is not a licence to rewrite the value.
    """
    declared = spec.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return {t for t in declared if isinstance(t, str)}
    return set()


def _matches(value: Any, types: set[str]) -> bool:
    """Does a parsed value satisfy any of these declared types?"""
    if isinstance(value, bool):
        # bool before int: True is an int in Python, but a schema asking
        # for a number does not mean it wants True.
        return bool({"boolean"} & types)
    if value is None:
        return "null" in types
    if isinstance(value, list):
        return "array" in types
    if isinstance(value, dict):
        return "object" in types
    if isinstance(value, int):
        return bool({"integer", "number"} & types)
    if isinstance(value, float):
        return "number" in types
    return False


def _coerce_value(value: Any, spec: dict[str, Any]) -> Any:
    types = _declared_types(spec)
    if not types or "string" in types:
        return value  # see the rule above
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            parsed = json.loads(text)
        except ValueError:
            # Not JSON at all: leave it, and let the tool's own validation
            # produce the honest complaint. Guessing would be worse.
            return value
        if not _matches(parsed, types):
            return value
        value = parsed
    if isinstance(value, dict) and "object" in types:
        return coerce_arguments(value, spec)
    if isinstance(value, list) and "array" in types:
        items = spec.get("items")
        if isinstance(items, dict):
            repaired = [_coerce_value(item, items) for item in value]
            if any(a is not b for a, b in zip(repaired, value, strict=True)):
                return repaired
    return value


def coerce_arguments(arguments: Any, schema: dict[str, Any] | None) -> Any:
    """Return ``arguments`` with stringified non-scalars parsed back.

    Schema-driven and conservative: only properties the schema names,
    only where the declared type cannot be a string, only when the string
    actually parses to the declared shape. Unknown keys, unstated types
    and unparseable strings all pass through untouched, so a tool's own
    validation still sees exactly what the model sent.

    Returns the ORIGINAL object when nothing changed, which lets callers
    use an identity check to tell whether anything was repaired.
    """
    properties = (schema or {}).get("properties")
    if not isinstance(properties, dict) or not isinstance(arguments, dict):
        return arguments
    repaired = dict(arguments)
    changed = False
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue
        fixed = _coerce_value(value, spec)
        if fixed is not value:
            repaired[key] = fixed
            changed = True
    return repaired if changed else arguments
