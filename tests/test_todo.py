"""The todo tracker: replace-whole-list semantics, three statuses, one store."""

from __future__ import annotations

import json

import pytest

from akshara.errors import ToolError
from akshara.tools import TodoRead, TodoWrite
from akshara.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


def items(*pairs):
    return [{"task": t, "status": s} for t, s in pairs]


class TestWriteSemantics:
    def test_replace_whole_list(self, ctx, tmp_path):
        TodoWrite().run({"items": items(("a", "pending"), ("b", "in_progress"))}, ctx)
        TodoWrite().run({"items": items(("c", "done"))}, ctx)
        stored = json.loads(
            (tmp_path / ".akshara/todos.json").read_text())["todos"]
        assert [i["task"] for i in stored] == ["c"]
        assert stored[0]["status"] == "done"

    def test_empty_list_is_legal(self, ctx):
        TodoWrite().run({"items": items(("a", "pending"))}, ctx)
        out = TodoWrite().run({"items": []}, ctx)
        assert "1 -> 0 items" in out

    def test_default_status_is_pending(self, ctx, tmp_path):
        TodoWrite().run({"items": [{"task": "just a task"}]}, ctx)
        stored = json.loads(
            (tmp_path / ".akshara/todos.json").read_text())["todos"]
        assert stored[0]["status"] == "pending"

    def test_result_renders_the_checklist(self, ctx):
        out = TodoWrite().run({"items": [
            {"task": "explore", "status": "done"},
            {"task": "fix", "status": "in_progress"},
            {"task": "verify", "status": "pending"},
        ]}, ctx)
        assert "[x] explore" in out
        assert "[~] fix" in out
        assert "[ ] verify" in out

    def test_persists_across_instances(self, ctx, tmp_path):
        TodoWrite().run({"items": items(("only", "pending"))}, ctx)
        out = TodoRead().run({}, ctx)
        assert "only" in out and "[ ]" in out


class TestValidation:
    @pytest.mark.parametrize("bad_status", ["Done", "", "archived", None, 3])
    def test_bad_status_refused(self, ctx, bad_status):
        with pytest.raises(ToolError, match="status must be"):
            TodoWrite().run({"items": [{"task": "t", "status": bad_status}]}, ctx)

    def test_empty_task_refused(self, ctx):
        with pytest.raises(ToolError, match="non-empty 'task'"):
            TodoWrite().run({"items": [{"task": "  "}]}, ctx)

    def test_items_not_a_list(self, ctx):
        with pytest.raises(ToolError, match="'items' must be a list"):
            TodoWrite().run({"items": "one whole plan"}, ctx)

    def test_over_cap_refused(self, ctx):
        with pytest.raises(ToolError, match="too many items"):
            TodoWrite().run({"items": [{"task": f"t{i}"} for i in range(51)]}, ctx)

    def test_task_over_length_cap(self, ctx):
        with pytest.raises(ToolError, match="too long"):
            TodoWrite().run({"items": [{"task": "x" * 501}]}, ctx)


class TestRead:
    def test_read_empty_store_hints_at_write(self, ctx):
        out = TodoRead().run({}, ctx)
        assert "no tasks" in out and "todo_write" in out

    def test_read_shows_counts(self, ctx):
        TodoWrite().run({"items": items(
            ("a", "done"), ("b", "in_progress"), ("c", "pending"),
            ("d", "pending"))}, ctx)
        out = TodoRead().run({}, ctx)
        assert "1 done / 1 active / 2 pending" in out

    def test_corrupted_store_is_model_readable(self, ctx, tmp_path):
        (tmp_path / ".akshara").mkdir()
        (tmp_path / ".akshara/todos.json").write_text("{not json")
        with pytest.raises(ToolError, match="corrupted"):
            TodoRead().run({}, ctx)

    def test_unknown_statuses_dropped_not_fatal(self, ctx, tmp_path):
        # a store from a NEWER harness version must not brick the old one
        (tmp_path / ".akshara").mkdir()
        (tmp_path / ".akshara/todos.json").write_text(json.dumps(
            {"todos": [{"task": "ok", "status": "pending"},
                       {"task": "weird", "status": "blocked"}]}))
        out = TodoRead().run({}, ctx)
        assert "ok" in out and "blocked" not in out


def test_gating_flags(ctx):
    assert TodoRead.read_only is True
    assert TodoWrite.read_only is False
