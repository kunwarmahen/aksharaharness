"""The glob tool -- find files by NAME, not by contents.

grep answers "which files mention X?"; this answers "which files exist
matching this shape?" -- the question behind "where do the tests live?",
"find every config file", "what changed most recently?". Before it
existed, a filename hunt meant either grep'ing for text the file might
not contain or a bash ``find`` call -- and bash is permission-gated,
so the model paid a human prompt for what is really a read.

One tool, stdlib only: ``pathlib.Path.glob`` patterns (``**`` recurses)
over the sandbox root. Results sort newest-first, because on the
questions this tool actually gets asked ("what did the build just
produce?", "which file did I edit last?"), recency IS the ranking.
Ties break by path so output stays deterministic.

The same skip rules as grep apply (see search.py): .git, node_modules
and friends never appear, no matter what the pattern says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_int, require_str
from akshara.tools.fs import resolve_in_sandbox
from akshara.tools.search import SKIP_DIRS

MAX_RESULTS = 200


def _not_skipped(rel: Path) -> bool:
    """True when no segment of the sandbox-relative path is a skip dir."""
    return not (set(rel.parts) & SKIP_DIRS)


class Glob(Tool):
    name = "glob"
    description = (
        "Find files whose PATH matches a glob pattern ('*' wildcards, "
        "'**' recurses into directories). Returns paths sorted newest-"
        "first. Use to locate files by name; use grep to search their "
        "contents."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string",
                        "description": "Glob such as '*.py', 'src/**/*.ts', "
                                       "or '.akshara/*.json'. Relative to "
                                       "the sandbox root."},
            "path": {"type": "string",
                     "description": "Directory to search, relative to the "
                                    "sandbox root (default '.')."},
            "limit": {"type": "integer",
                      "description": f"Max matches returned (default "
                                     f"{MAX_RESULTS})."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        where = args.get("path", ".")
        return f"glob {require_str(args, 'pattern')!r} in {where}/"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        pattern = require_str(args, "pattern")
        limit = require_int(args, "limit", default=MAX_RESULTS)
        if limit < 1 or limit > MAX_RESULTS:
            raise ToolError(f"limit must be between 1 and {MAX_RESULTS}")

        root = resolve_in_sandbox(ctx, require_str(args, "path",
                                                   optional=True, default="."))
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")

        try:
            hits = [p for p in root.glob(pattern)
                    if p.is_file() and _not_skipped(p.relative_to(root))]
        except ValueError as exc:  # bad ** placement etc.
            raise ToolError(f"invalid glob pattern {pattern!r}: {exc}") from exc

        # Newest first -- "what just appeared/changed" outranks everything;
        # ties break by path so the output never shuffles between calls.
        hits.sort(key=lambda p: (-(p.stat().st_mtime_ns), str(p)))
        total = len(hits)
        hits = hits[:limit]

        if not hits:
            return "no matches"
        lines = [str(h.relative_to(root)) for h in hits]
        suffix = ""
        if total > len(lines):
            suffix = f"\n[showing {len(lines)} of {total} matches; raise limit]"
        return "\n".join(lines) + suffix
