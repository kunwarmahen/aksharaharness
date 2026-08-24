"""The six built-in tools against tmp_path sandboxes.

These pin the safety behaviors (sandbox escapes refused, ambiguity
refused, timeouts salvage output) and the truncation caps.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from akshara.errors import ToolError
from akshara.tools import Bash, EditFile, Grep, ListDir, ReadFile, WriteFile
from akshara.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


class TestSandbox:
    def test_relative_paths_resolve_inside(self, ctx, tmp_path):
        (tmp_path / "a.txt").write_text("inside")
        out = ReadFile().run({"path": "a.txt"}, ctx)
        assert "inside" in out

    def test_parent_escape_refused(self, ctx, tmp_path):
        with pytest.raises(ToolError, match="escapes sandbox"):
            ReadFile().run({"path": "../outside.txt"}, ctx)

    def test_absolute_outside_refused(self, ctx):
        with pytest.raises(ToolError, match="escapes sandbox"):
            ReadFile().run({"path": "/etc/passwd"}, ctx)

    def test_sneaky_dotdot_refused(self, ctx):
        with pytest.raises(ToolError, match="escapes sandbox"):
            ListDir().run({"path": "sub/../../.."}, ctx)


class TestReadFile:
    def test_numbered_lines(self, ctx, tmp_path):
        (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
        out = ReadFile().run({"path": "a.txt"}, ctx)
        lines = out.splitlines()
        assert lines[0].strip().startswith("1") and "one" in lines[0]
        assert "three" in lines[2]

    def test_offset_and_limit(self, ctx, tmp_path):
        (tmp_path / "a.txt").write_text("l1\nl2\nl3\nl4\n")
        out = ReadFile().run({"path": "a.txt", "offset": 2, "limit": 2}, ctx)
        assert "l1" not in out and "l2" in out and "l3" in out and "l4" not in out

    def test_missing_file_is_tool_error(self, ctx):
        with pytest.raises(ToolError, match="not found"):
            ReadFile().run({"path": "nope.txt"}, ctx)

    def test_binary_file_refused(self, ctx, tmp_path):
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        with pytest.raises(ToolError, match="binary"):
            ReadFile().run({"path": "blob.bin"}, ctx)


class TestWriteEdit:
    def test_write_creates_parents(self, ctx, tmp_path):
        result = WriteFile().run({"path": "deep/dir/f.txt", "content": "hi"}, ctx)
        assert (tmp_path / "deep/dir/f.txt").read_text() == "hi"
        assert "created" in result

    def test_edit_exact_match(self, ctx, tmp_path):
        (tmp_path / "f.txt").write_text("hello world\n")
        EditFile().run({"path": "f.txt", "old_string": "world", "new_string": "there"}, ctx)
        assert (tmp_path / "f.txt").read_text() == "hello there\n"

    def test_edit_ambiguous_refused_with_count(self, ctx, tmp_path):
        (tmp_path / "f.txt").write_text("x x x")
        with pytest.raises(ToolError, match="matches 3 times"):
            EditFile().run({"path": "f.txt", "old_string": "x", "new_string": "y"}, ctx)

    def test_edit_replace_all(self, ctx, tmp_path):
        (tmp_path / "f.txt").write_text("x x x")
        EditFile().run({"path": "f.txt", "old_string": "x", "new_string": "y",
                        "replace_all": True}, ctx)
        assert (tmp_path / "f.txt").read_text() == "y y y"

    def test_edit_missing_string_reports_zero_matches(self, ctx, tmp_path):
        (tmp_path / "f.txt").write_text("abc")
        with pytest.raises(ToolError, match="not found"):
            EditFile().run({"path": "f.txt", "old_string": "zzz", "new_string": "y"}, ctx)

    def test_summaries_show_what_will_happen(self, ctx, tmp_path):
        write_summary = WriteFile().summary(
            {"path": "f.txt", "content": "brand new"}, ctx)
        assert "NEW FILE" in write_summary or "brand new" in write_summary

        (tmp_path / "g.txt").write_text("old line\n")
        edit_summary = EditFile().summary(
            {"path": "g.txt", "old_string": "old line", "new_string": "new line"}, ctx)
        assert "-old line" in edit_summary and "+new line" in edit_summary

    def test_bad_argument_types_are_model_readable_errors(self, ctx):
        with pytest.raises(ToolError, match="must be a string"):
            WriteFile().run({"path": 123, "content": "x"}, ctx)
        with pytest.raises(ToolError, match="missing required argument"):
            WriteFile().run({"content": "x"}, ctx)


class TestBash:
    def test_captures_output_and_exit_code(self, ctx):
        out = Bash().run({"command": "echo hello; echo err >&2"}, ctx)
        # stderr interleaves into stdout on this tool
        assert "hello" in out and "err" in out and "exit code: 0" in out

    def test_nonzero_exit_code_reported_not_raised(self, ctx):
        out = Bash().run({"command": "exit 3"}, ctx)
        assert "exit code: 3" in out  # the MODEL decides what failure means

    def test_timeout_salvages_partial_output(self, ctx):
        with pytest.raises(ToolError, match="timed out after") as excinfo:
            Bash().run({"command": "echo partial-marker; sleep 5",
                        "timeout": 1}, ctx)
        # subprocess.run would throw this away; we keep it
        assert "partial-marker" in str(excinfo.value)

    def test_runs_in_sandbox_dir(self, ctx, tmp_path):
        out = Bash().run({"command": "pwd"}, ctx)
        assert str(tmp_path) in out

    def test_invalid_timeout_refused(self, ctx):
        with pytest.raises(ToolError, match="timeout"):
            Bash().run({"command": "true", "timeout": 100000}, ctx)


class TestGrep:
    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main():\n    return 42\n")
        (tmp_path / "src" / "util.py").write_text("# TODO fix\nvalue = 'needle'\n")
        (tmp_path / "notes.md").write_text("the needle is here\n")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("needle in git dir\n")
        return tmp_path

    def test_finds_matches_with_locations(self, ctx, tree):
        out = Grep().run({"pattern": "needle"}, ctx)
        assert "notes.md:1:" in out
        assert "src/util.py:2:" in out
        assert ".git" not in out  # skipped directory

    def test_include_filters_filenames(self, ctx, tree):
        out = Grep().run({"pattern": "needle", "include": ".*\\.md"}, ctx)
        assert "notes.md" in out
        assert "util.py" not in out

    def test_case_insensitive_option(self, ctx, tree):
        assert Grep().run({"pattern": "NEEDLE"}, ctx) == "no matches"
        out = Grep().run({"pattern": "NEEDLE", "case_insensitive": True}, ctx)
        assert "notes.md" in out

    def test_match_cap(self, ctx, tree):
        (tree / "big.txt").write_text("hit\n" * 500)
        out = Grep().run({"pattern": "hit", "path": "big.txt"}, ctx)
        assert "stopped at 200 matches" in out

    def test_invalid_regex_is_tool_error(self, ctx):
        with pytest.raises(ToolError, match="invalid regular expression"):
            Grep().run({"pattern": "("}, ctx)


class TestGrepBackends:
    """The two backends behind one tool contract.

    The ripgrep side runs against a FAKE ``rg`` (a script emitting canned
    --json events), so these tests are hermetic whether or not real
    ripgrep is installed. The fallback is pinned by monkeypatching the
    selector seam to None.
    """

    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "util.py").write_text("value = 'needle'\n")
        (tmp_path / "notes.md").write_text("the needle is here\n")
        return tmp_path

    def test_fallback_backend_when_rg_missing(self, ctx, tree, monkeypatch):
        import akshara.tools.search as search
        monkeypatch.setattr(search, "_find_rg", lambda: None)

        out = Grep().run({"pattern": "needle"}, ctx)

        assert "notes.md:1:" in out
        assert "src/util.py:1:" in out

    def _fake_rg(self, tmp_path: Path, body: str, *, exit_code: int = 0,
                 log_args: Path | None = None) -> Path:
        """A stand-in ``rg`` that prints ``body`` then exits."""
        bin_dir = tmp_path / "fakebin"
        bin_dir.mkdir(exist_ok=True)
        script = bin_dir / "rg"
        lines = ["#!/usr/bin/env python3", "import os, sys"]
        if log_args is not None:
            # only emit logging when a path was actually requested --
            # embedding str(None) yields the truthy STRING 'None', which
            # made every unlogged test litter an argv dump named "None"
            # into the repo root on each run
            lines += [
                f"LOG = {str(log_args)!r}",
                "open(LOG, 'w').write(repr(sys.argv[1:]))",
            ]
        lines += [
            f"print({body!r}, end='')",
            f"sys.exit({exit_code})",
        ]
        script.write_text("\n".join(lines) + "\n")
        script.chmod(0o755)
        return script

    def test_ripgrep_json_stream_normalizes_to_tool_format(
            self, ctx, tree, tmp_path, monkeypatch):
        import akshara.tools.search as search
        # absolute path in, relative 'path:line: text' out -- same as walker
        abs_file = tree / "src" / "util.py"
        events = (
            json.dumps({"type": "begin", "data": {}}) + "\n"
            + json.dumps({"type": "match", "data": {
                "path": {"text": str(abs_file)},
                "line_number": 1,
                "lines": {"text": "value = 'needle'\n"}}}) + "\n"
            + json.dumps({"type": "match", "data": {
                "path": {"text": str(tmp_path.parent / "outside-root.txt")},
                "line_number": 9, "lines": {"text": "escaped?\n"}}}) + "\n"
            + json.dumps({"type": "end", "data": {}}) + "\n"
        )
        monkeypatch.setattr(
            search, "_find_rg",
            lambda: self._fake_rg(tmp_path, events))

        out = Grep().run({"pattern": "needle"}, ctx)

        assert "src/util.py:1: value = 'needle'" in out
        assert "outside-root" not in out  # path outside root never reported

    def test_ripgrep_include_filter_and_case_flag(
            self, ctx, tree, tmp_path, monkeypatch):
        import akshara.tools.search as search
        args_log = tmp_path / "argv.txt"
        events = "".join(
            json.dumps({"type": "match", "data": {
                "path": {"text": str(tree / "src" / "util.py")},
                "line_number": n, "lines": {"text": f"needle {n}\n"}}}) + "\n"
            for n in (1, 2)
        )
        monkeypatch.setattr(
            search, "_find_rg",
            lambda: self._fake_rg(tmp_path, events, log_args=args_log))

        filtered = Grep().run({"pattern": "needle", "include": ".*\\.md"}, ctx)
        assert filtered == "no matches"  # .py matches dropped by include

        Grep().run({"pattern": "needle", "case_insensitive": True}, ctx)
        assert "'-i'" in args_log.read_text()  # flag forwarded to the backend

    def test_ripgrep_cap_stops_reading_the_tree(
            self, ctx, tree, tmp_path, monkeypatch):
        import akshara.tools.search as search
        events = "".join(
            json.dumps({"type": "match", "data": {
                "path": {"text": str(tree / "notes.md")},
                "line_number": n, "lines": {"text": "hit\n"}}}) + "\n"
            for n in range(1, 221)  # 220 offered, only 200 may survive
        )
        monkeypatch.setattr(
            search, "_find_rg",
            lambda: self._fake_rg(tmp_path, events))

        out = Grep().run({"pattern": "hit"}, ctx)

        assert "[stopped at 200 matches]" in out
        assert out.count("hit") == 200

    def test_ripgrep_failure_falls_back_to_walker(
            self, ctx, tree, tmp_path, monkeypatch):
        import akshara.tools.search as search
        # exit 3 with no output: e.g. rust-re rejected a python-only regex
        monkeypatch.setattr(
            search, "_find_rg",
            lambda: self._fake_rg(tmp_path, "", exit_code=3))

        out = Grep().run({"pattern": "needle"}, ctx)

        assert "notes.md:1:" in out  # fallback answered, same contract

    def test_rust_regex_gap_lookahead_still_searches(self, ctx, tree):
        """A pattern rust-re can't compile must still WORK -- via the
        fallback -- rather than erroring or returning garbage."""
        if shutil.which("rg") is None:
            pytest.skip("needs real rg to exercise the dialect gap")
        out = Grep().run({"pattern": "needle(?= is)"}, ctx)
        assert "notes.md:1:" in out


class TestListDir:
    def test_dirs_get_slash_suffix_sorted_first(self, ctx, tmp_path):
        (tmp_path / "zfile.txt").write_text("x")
        (tmp_path / "adir").mkdir()
        out = ListDir().run({"path": "."}, ctx)
        lines = out.splitlines()
        assert lines[0] == "adir/"
        assert "zfile.txt" in lines
