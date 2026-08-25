"""The glob tool: find files by name, newest-first, skips hidden dirs."""

from __future__ import annotations

import os
import time

import pytest

from akshara.errors import ToolError
from akshara.tools import Glob
from akshara.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


def touch(path, mtime: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestGlobBasics:
    def test_flat_match(self, ctx, tmp_path):
        touch(tmp_path / "a.py")
        touch(tmp_path / "b.txt")
        out = Glob().run({"pattern": "*.py"}, ctx)
        assert out == "a.py"

    def test_recursive_double_star(self, ctx, tmp_path):
        touch(tmp_path / "src/deep/mod.py")
        touch(tmp_path / "tests/test_mod.py")
        out = Glob().run({"pattern": "**/*.py"}, ctx)
        assert "src/deep/mod.py" in out.splitlines()
        assert "tests/test_mod.py" in out.splitlines()

    def test_no_matches(self, ctx):
        assert Glob().run({"pattern": "*.nope"}, ctx) == "no matches"

    def test_directories_never_listed(self, ctx, tmp_path):
        touch(tmp_path / "pkg/__init__.py")
        out = Glob().run({"pattern": "pkg/**/*"}, ctx)
        assert "__init__.py" in out and not out.endswith("/")


class TestSkipRules:
    def test_skip_dirs_excluded_even_when_matched(self, ctx, tmp_path):
        touch(tmp_path / ".git/hooks/pre-commit")
        touch(tmp_path / ".venv/lib/x.py")
        touch(tmp_path / "node_modules/pkg/index.js")
        touch(tmp_path / "real.js")
        out = Glob().run({"pattern": "**/*.js"}, ctx)
        assert out == "real.js"


class TestOrdering:
    def test_newest_first(self, ctx, tmp_path):
        base = time.time() - 100
        touch(tmp_path / "old.py", mtime=base)
        touch(tmp_path / "new.py", mtime=base + 50)
        out = Glob().run({"pattern": "*.py"}, ctx)
        assert out.splitlines() == ["new.py", "old.py"]

    def test_limit_keeps_the_newest(self, ctx, tmp_path):
        base = time.time() - 100
        for i in range(5):
            touch(tmp_path / f"f{i}.py", mtime=base + i)
        out = Glob().run({"pattern": "*.py", "limit": 2}, ctx)
        lines = [l for l in out.splitlines() if not l.startswith("[")]
        assert len(lines) == 2
        assert lines[0] == "f4.py" and lines[1] == "f3.py"
        assert "of 5 matches" in out


class TestValidation:
    def test_bad_pattern_is_tool_error(self, ctx):
        # ** can only trail a separator in pathlib globs
        with pytest.raises(ToolError, match="invalid glob"):
            Glob().run({"pattern": "src**/*.py"}, ctx)

    def test_not_a_directory(self, ctx, tmp_path):
        touch(tmp_path / "file.txt")
        with pytest.raises(ToolError, match="not a directory"):
            Glob().run({"pattern": "*", "path": "file.txt"}, ctx)

    def test_escape_refused(self, ctx):
        with pytest.raises(ToolError, match="escapes sandbox"):
            Glob().run({"pattern": "*.py", "path": "../elsewhere"}, ctx)

    def test_sandbox_escape_via_symlink_refused(self, ctx, tmp_path):
        outside = tmp_path.parent / "glob-outside"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "leak"
        try:
            link.symlink_to(outside, target_is_directory=True)
            with pytest.raises(ToolError, match="escapes sandbox"):
                Glob().run({"pattern": "*", "path": "leak"}, ctx)
        finally:
            link.unlink(missing_ok=True)


def test_read_only_and_summary(ctx):
    assert Glob.read_only is True
    assert "*.py" in Glob().summary({"pattern": "*.py"}, ctx)
