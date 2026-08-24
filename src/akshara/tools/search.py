"""The grep tool -- one tool contract, two swappable backends.

* ripgrep subprocess (when an ``rg`` binary is on PATH): production
  speed on big trees, streaming ``--json`` output parsed incrementally.
* pure-Python ``os.walk`` fallback: dependency-free and cross-platform;
  also used whenever the subprocess misbehaves (missing binary, regex
  rust-re rejects like lookaheads, nonzero exit).

Both produce the SAME output format (''path:line: text''), the same
MAX_MATCHES cap, and the same skip rules -- tests pin each behavior on
both sides, so callers cannot tell which backend answered.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str
from akshara.tools.fs import resolve_in_sandbox

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}
MAX_MATCHES = 200
BINARY_SNIFF_BYTES = 8192


def _find_rg() -> str | None:
    """Backend selector seam: tests monkeypatch this to pin a backend."""
    return shutil.which("rg")


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(BINARY_SNIFF_BYTES)
    except OSError:
        return True


def _search(root: Path, pattern: re.Pattern[str],
            include: re.Pattern[str] | None) -> list[str]:
    """Walk ``root``, return formatted matches (capped)."""
    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if include is not None and not include.match(filename):
                continue
            path = Path(dirpath) / filename
            if _looks_binary(path):
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if pattern.search(line):
                            rel = path.relative_to(root)
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(matches) >= MAX_MATCHES:
                                return matches
            except OSError:
                continue  # unreadable file: skip, don't die
    return matches


def _search_rg(rg: str, root: Path, raw_pattern: str,
               include: re.Pattern[str] | None, case_insensitive: bool,
               ) -> list[str] | None:
    """ripgrep backend. Returns None to mean "unusable this time -- fall
    back to the Python walker" (spawn failure, or rg rejecting something
    about the invocation). An empty list is a genuine "no matches".

    Flags mirror the walker's semantics exactly: ``--no-ignore`` +
    ``--hidden`` so results never depend on whatever .gitignore sits in
    the tree, and ``-g !dir`` exclusions reproducing SKIP_DIRS.
    """
    cmd = [rg, "--json", "--no-ignore", "--hidden", "--line-number",
           "--no-messages"]
    if case_insensitive:
        cmd.append("-i")
    for d in sorted(SKIP_DIRS):
        cmd += ["-g", f"!{d}", "-g", f"!{d}/**"]  # hide dir itself + contents
    cmd += [raw_pattern, str(root)]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True,
                                errors="replace")
    except OSError:
        return None

    matches: list[str] = []
    with proc:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "match":
                    continue  # begin/end/summary bookkeeping
                data = event.get("data") or {}
                path_text = (data.get("path") or {}).get("text")
                lineno = data.get("line_number")
                text = (data.get("lines") or {}).get("text") or ""
                if not path_text or lineno is None:
                    continue
                try:
                    rel = Path(path_text).relative_to(root)
                except ValueError:
                    continue  # outside the sandbox: never report it
                # .match() = anchored at start, exactly like the walker's filter
                if include is not None and not include.match(rel.name):
                    continue
                matches.append(f"{rel}:{lineno}: {text.rstrip()}")
                if len(matches) >= MAX_MATCHES:
                    proc.kill()  # stop reading the tree at the cap
                    break
        except OSError:
            return None  # pipe died mid-read: defer to the fallback
        code = proc.wait()

    # 0 = matches found, 1 = none found, negative = our own kill at the
    # cap. Anything else is an rg failure (bad regex dialect, bad flags):
    # a half-answer would be worse than the slower-but-correct fallback.
    killed_at_cap = len(matches) >= MAX_MATCHES
    if code not in (0, 1) and not killed_at_cap:
        return None
    return matches


class Grep(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Returns "
        "'path:line: text' matches. Use include to filter by filename "
        "pattern (e.g. include='.*\\.py')."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Directory or file to search, relative to sandbox root (default '.')."},
            "include": {"type": "string", "description": "Regex matched against filenames to filter which files are searched."},
            "case_insensitive": {"type": "boolean", "description": "Ignore case (default false)."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        where = args.get("path", ".")
        inc = f" --include {args['include']}" if args.get("include") else ""
        return f"grep /{require_str(args, 'pattern')}/ in {where}{inc}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        raw_pattern = require_str(args, "pattern")
        case_insensitive = bool(args.get("case_insensitive", False))
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            pattern = re.compile(raw_pattern, flags)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc

        include_str = require_str(args, "include", optional=True) or None
        include: re.Pattern[str] | None = None
        if include_str:
            try:
                include = re.compile(include_str)  # validate now, fail loudly
            except re.error as exc:
                raise ToolError(f"invalid include pattern: {exc}") from exc

        root = resolve_in_sandbox(ctx, require_str(args, "path", optional=True, default="."))
        if root.is_file():
            # single-file search: the walker adds nothing over a direct scan
            results = _search_file(root, pattern)
        else:
            results = None
            rg = _find_rg()
            if rg is not None:
                # same semantics, two dialects: case-insensitivity is baked
                # into the compiled pattern for the walker, passed as -i
                # for ripgrep
                results = _search_rg(rg, root, raw_pattern, include,
                                     case_insensitive)
            if results is None:
                results = _search(root, pattern, include)

        if not results:
            return "no matches"
        suffix = f"\n[stopped at {MAX_MATCHES} matches]" if len(results) >= MAX_MATCHES else ""
        return "\n".join(results) + suffix


def _search_file(path: Path, pattern: re.Pattern[str]) -> list[str]:
    if _looks_binary(path):
        return []
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            if pattern.search(line):
                out.append(f"{path.name}:{lineno}: {line.rstrip()}")
                if len(out) >= MAX_MATCHES:
                    break
    return out
