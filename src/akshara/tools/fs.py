"""File tools: read_file, write_file, edit_file, list_dir.

All four resolve paths through the same sandbox helper. The sandbox is
cwd-relative CONVENIENCE (it stops the model from wandering off by
accident), not a security boundary -- bash isn't bound by it, which is
precisely why bash requires human approval.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_int, require_str

MAX_READ_BYTES = 256 * 1024
MAX_READ_LINES = 2000
MAX_LIST_ENTRIES = 500


def resolve_in_sandbox(ctx: ToolContext, raw: str) -> Path:
    """Resolve ``raw`` against the sandbox root, refusing escapes."""
    if not raw:
        raise ToolError("empty path")
    path = (ctx.cwd / raw).resolve()  # resolves .., ~, symlinks
    if not path.is_relative_to(ctx.cwd.resolve()):
        raise ToolError(f"path escapes sandbox: {raw!r}")
    return path


def unified_diff(old: str, new: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def file_lease_key(path: Path) -> str:
    """Resource key under which write_file/edit_file hold their lease.

    One format, shared by the tools and the tests that pre-hold it.
    """
    return f"file:{path}"


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file from the sandbox. Returns numbered lines "
        "(cat -n style). Use offset/limit to page through big files."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the sandbox root."},
            "offset": {"type": "integer", "description": "1-based first line to read (default 1)."},
            "limit": {"type": "integer", "description": f"Max lines to read (default {MAX_READ_LINES})."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"read {require_str(args, 'path')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = resolve_in_sandbox(ctx, require_str(args, "path"))
        offset = require_int(args, "offset", default=1)
        limit = require_int(args, "limit", default=MAX_READ_LINES)
        if offset < 1:
            raise ToolError("offset is 1-based; must be >= 1")

        if not path.exists():
            raise ToolError(f"file not found: {path}")
        if path.is_dir():
            raise ToolError(f"{path} is a directory (use list_dir)")

        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            raise ToolError(f"{path} looks binary; refusing to read")
        text = data.decode("utf-8", errors="replace")

        lines = text.splitlines()
        window = lines[offset - 1 : offset - 1 + limit]
        if not window:
            return f"[no lines in range {offset}..{offset + limit - 1}; file has {len(lines)} lines]"

        numbered = [f"{offset + i:>6}\t{line}" for i, line in enumerate(window)]
        result = "\n".join(numbered)
        if len(data) > MAX_READ_BYTES:
            result += f"\n[truncated: file is {len(data)} bytes, read capped at {MAX_READ_BYTES}]"
        if offset - 1 + limit < len(lines):
            result += f"\n[{len(lines) - (offset - 1 + limit)} more lines; increase limit or use offset]"
        return result


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List a directory's entries (one per line, '/' suffix on "
        "directories, sorted)."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path, relative to sandbox root (default '.')."},
        },
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"list {args.get('path', '.')}/"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = resolve_in_sandbox(ctx, require_str(args, "path", optional=True, default="."))
        if not path.is_dir():
            raise ToolError(f"not a directory: {path}")

        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        suffix = ""
        if len(entries) > MAX_LIST_ENTRIES:
            entries = entries[:MAX_LIST_ENTRIES]
            suffix = f"\n[listing capped at {MAX_LIST_ENTRIES} entries]"
        lines = [e.name + ("/" if e.is_dir() else "") for e in entries]
        return ("\n".join(lines) if lines else "[empty directory]") + suffix


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Create or overwrite a file with the given content. Creates "
        "parent directories. Prefer edit_file for modifying existing files."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to sandbox root."},
            "content": {"type": "string", "description": "Full file contents to write."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = require_str(args, "path")
        content = require_str(args, "content")
        try:
            target = resolve_in_sandbox(ctx, path)
        except ToolError:
            return f"write_file {path} (WARNING: outside sandbox)"
        if target.exists():
            return unified_diff(target.read_text(errors="replace"), content, path) or \
                f"write_file {path} (identical content)"
        return f"NEW FILE {path} ({len(content.splitlines())} lines)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = resolve_in_sandbox(ctx, require_str(args, "path"))
        content = require_str(args, "content")
        # lease: a parallel batch sibling writing the SAME path waits
        # here instead of interleaving half-writes (reads need no leases)
        with ctx.leases.hold(file_lease_key(path)):
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        verb = "overwrote" if existed else "created"
        return f"{verb} {path} ({len(content)} bytes)"


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Exact-match string replacement in a file. Fails unless "
        "old_string matches exactly once (use replace_all to change "
        "every occurrence). Include enough surrounding text to be unique."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to sandbox root."},
            "old_string": {"type": "string", "description": "Exact text to find."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = require_str(args, "path")
        old = require_str(args, "old_string")
        new = require_str(args, "new_string")
        try:
            target = resolve_in_sandbox(ctx, path)
            current = target.read_text(errors="replace") if target.exists() else ""
        except ToolError:
            return f"edit_file {path} (WARNING: outside sandbox)"
        preview = current.replace(old, new) if old in current else current + "\n" + new
        return unified_diff(current, preview, path) or f"edit_file {path} (no visible change)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        path = resolve_in_sandbox(ctx, require_str(args, "path"))
        old = require_str(args, "old_string")
        new = require_str(args, "new_string")
        replace_all = bool(args.get("replace_all", False))

        # whole read-validate-replace-write inside the lease: two batch
        # siblings editing one file serialize instead of eating edits
        with ctx.leases.hold(file_lease_key(path)):
            if not path.exists():
                raise ToolError(f"file not found: {path}")
            text = path.read_text(encoding="utf-8", errors="replace")
            count = text.count(old)
            if count == 0:
                raise ToolError(f"old_string not found in {path}")
            if count > 1 and not replace_all:
                raise ToolError(
                    f"old_string matches {count} times in {path}; include more "
                    "surrounding text to make it unique, or pass replace_all=true"
                )
            updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            path.write_text(updated, encoding="utf-8")
        return f"edited {path} ({count if replace_all else 1} replacement(s))"
