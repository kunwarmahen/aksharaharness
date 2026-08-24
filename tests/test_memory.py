"""Scratchpad/retrieval memory: notes outliving the context window.

The machinery is deterministic so it gets unit tests here: upsert
semantics, ranked retrieval, atomic persistence, model-readable errors.
The BEHAVIORAL claim (a model actually uses write_note to survive
compaction) belongs to live runs, like every tool.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from akshara.errors import ToolError
from akshara.tools.base import ToolContext
from akshara.tools.memory import NoteStore, RecallNotes, WriteNote


@pytest.fixture()
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


def run(tool, args: dict[str, Any], ctx: ToolContext) -> str:
    return tool.run(args, ctx)


class TestNoteStore:
    def test_upsert_get_round_trip(self, tmp_path):
        store = NoteStore(tmp_path)
        store.upsert("layout", "src/akshara has the package")
        assert store.get("layout") == "src/akshara has the package"

    def test_persists_across_instances(self, tmp_path):
        # the entire point of the file backing: a NEW process (or a new
        # store object) sees what was written before
        NoteStore(tmp_path).upsert("key", "value")
        assert NoteStore(tmp_path).get("key") == "value"

    def test_store_lives_under_sandbox_root(self, ctx):
        WriteNote().run({"topic": "t", "content": "c"}, ctx)
        assert (ctx.cwd / ".akshara" / "memory.json").exists()

    def test_rewrites_replace_never_append(self, tmp_path):
        store = NoteStore(tmp_path)
        store.upsert("t", "first")
        store.upsert("t", "second")
        assert store.get("t") == "second"
        assert len(store.all()) == 1

    def test_corrupt_store_is_a_model_readable_error(self, tmp_path):
        path = tmp_path / ".akshara" / "memory.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ToolError, match="corrupted"):
            NoteStore(tmp_path).topics()


class TestWriteNote:
    def test_write_then_report_mentions_update_vs_new(self, ctx):
        first = run(WriteNote(), {"topic": "t", "content": "aaa"}, ctx)
        again = run(WriteNote(), {"topic": "t", "content": "bbb"}, ctx)
        assert "saved" in first and "updated" in again

    def test_empty_arguments_are_model_readable_errors(self, ctx):
        with pytest.raises(ToolError, match="topic"):
            run(WriteNote(), {"content": "x"}, ctx)
        with pytest.raises(ToolError, match="content"):
            run(WriteNote(), {"topic": "t"}, ctx)

    def test_oversized_note_refused_with_advice(self, ctx):
        with pytest.raises(ToolError, match="distill"):
            run(WriteNote(), {"topic": "t", "content": "x" * 20_001}, ctx)

    def test_summary_shows_topic_and_size_for_the_gate(self, ctx):
        summary = WriteNote().summary(
            {"topic": "creds", "content": "12345"}, ctx)
        assert "creds" in summary and "5 chars" in summary


class TestRecallNotes:
    def _seed(self, ctx) -> None:
        run(WriteNote(), {"topic": "project-layout",
                          "content": "package lives in src/akshara"},
            ctx)
        run(WriteNote(), {"topic": "dead-ends",
                          "content": "ripgrep flags break on BSD grep"}, ctx)

    def test_empty_memory_is_guidance_not_an_error(self, ctx):
        out = run(RecallNotes(), {}, ctx)
        assert "no notes yet" in out

    def test_index_lists_topics_with_previews(self, ctx):
        self._seed(ctx)
        out = run(RecallNotes(), {}, ctx)
        assert "[dead-ends]" in out and "[project-layout]" in out
        assert "src/akshara" in out  # body preview visible

    def test_query_ranks_topic_hits_over_body_hits(self, ctx):
        self._seed(ctx)
        # 'layout' appears in one topic AND nowhere in bodies; 'grep'
        # only appears inside a body -- both findable either way
        by_topic = run(RecallNotes(), {"query": "layout"}, ctx)
        assert "#1 [project-layout]" in by_topic
        by_body = run(RecallNotes(), {"query": "BSD"}, ctx)
        assert "#1 [dead-ends]" in by_body

    def test_no_match_names_the_topics_that_do_exist(self, ctx):
        self._seed(ctx)
        out = run(RecallNotes(), {"query": "zzz-nothing"}, ctx)
        assert "no notes match" in out and "dead-ends" in out

    def test_exact_topic_fetch_verbatim(self, ctx):
        self._seed(ctx)
        out = run(RecallNotes(), {"topic": "project-layout"}, ctx)
        assert out.startswith("# project-layout")
        assert "src/akshara" in out

    def test_missing_exact_topic_suggests_alternatives(self, ctx):
        self._seed(ctx)
        with pytest.raises(ToolError, match="project-layout"):
            run(RecallNotes(), {"topic": "projekt-layout"}, ctx)

    def test_store_file_is_inspectable_json(self, ctx):
        self._seed(ctx)
        data = json.loads(
            (ctx.cwd / ".akshara" / "memory.json").read_text())
        assert set(data) == {"project-layout", "dead-ends"}
