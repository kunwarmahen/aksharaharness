"""Scratchpad + retrieval memory: notes that outlive the context window.

The context window is working memory; this is long-term memory. A model
that has learned something expensive (a file layout, a user preference,
a dead end) writes it to a NOTE instead of hoping it survives compaction
-- compaction elides old tool results by design, so anything not written
down is anything not known after an auto-compact fires.

Two verbs, one store:

* ``write_note(topic, content)``  -- upsert. Topics are keys, not files:
  one atomic JSON file under ``.akshara/`` rather than a directory of
  them, because the model should never have to remember WHERE its
  memory lives, only WHAT it called things.
* ``recall_notes([query])``       -- retrieval. With a query: ranked
  substring matches over topics AND bodies (cheap, deterministic,
  explainable -- vector search would be a dependency and a mystery).
  Without: a topic index with previews, so the model can browse what it
  knows before asking for detail.

Honest limits: single-process (parallel tool batches within one turn are
fine -- each write re-reads then atomically replaces the file -- but two
harness processes sharing a sandbox can lose updates), and search is
literal substring matching, which is exactly as smart as it sounds.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str

MAX_NOTE_CHARS = 20_000
MAX_TOPICS = 200
PREVIEW_CHARS = 120


class NoteStore:
    """Topic -> note text, persisted as one atomic JSON document."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / ".akshara" / "memory.json"

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"memory store corrupted ({exc}); fix or delete {self.path}")

    def _save(self, notes: dict[str, str]) -> None:
        # write temp + os.replace: a crash mid-write leaves the previous
        # good store intact, never a truncated half-file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(notes, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            finally:
                raise

    def upsert(self, topic: str, content: str) -> None:
        notes = self._load()
        if topic not in notes and len(notes) >= MAX_TOPICS:
            raise ToolError(f"memory full ({MAX_TOPICS} topics); recall first "
                            "and consolidate before writing more")
        notes[topic] = content
        self._save(notes)

    def get(self, topic: str) -> str | None:
        return self._load().get(topic)

    def topics(self) -> list[str]:
        return sorted(self._load())

    def all(self) -> dict[str, str]:
        return dict(self._load())


def score_match(haystack: str, query_lower: str) -> int:
    return haystack.lower().count(query_lower)


class WriteNote(Tool):
    name = "write_note"
    description = (
        "Persist a durable fact for later turns: file layouts, decisions, "
        "dead ends, preferences. Upsert by topic -- rewriting a topic "
        "replaces it. Survives /clear, /compact, and restarts."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "topic": {"type": "string",
                      "description": "Short stable key, e.g. "
                                     "'project-layout' or 'user-prefers'. "
                                     "Use the SAME topic to update."},
            "content": {"type": "string",
                        "description": "The fact itself, self-contained "
                                       "(future you has no other context)."},
        },
        "required": ["topic", "content"],
        "additionalProperties": False,
    }
    read_only = False  # it writes a file under .akshara/

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        topic = require_str(args, "topic")
        content = require_str(args, "content")
        existing = NoteStore(ctx.cwd).get(topic)
        verb = "update" if existing is not None else "write"
        return f"{verb} note {topic!r} ({len(content)} chars)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        topic = require_str(args, "topic")
        content = require_str(args, "content")
        if len(content) > MAX_NOTE_CHARS:
            raise ToolError(f"note too large: {len(content)} chars "
                            f"(max {MAX_NOTE_CHARS}) -- distill, don't dump")
        replaced = NoteStore(ctx.cwd).get(topic) is not None
        NoteStore(ctx.cwd).upsert(topic, content)
        return (f"note {'updated' if replaced else 'saved'}: {topic!r} "
                f"({len(content)} chars)")


class RecallNotes(Tool):
    name = "recall_notes"
    description = (
        "Read your own persisted notes. With 'query': ranked matches over "
        "topics and bodies. Without: an index of everything you have "
        "written, with previews. Check here BEFORE redoing expensive "
        "exploration from an earlier session."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Substring to match against topics and "
                                     "note bodies (case-insensitive). Omit "
                                     "to list all topics."},
            "topic": {"type": "string",
                      "description": "Exact topic to fetch verbatim."},
        },
        "required": [],
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if args.get("topic"):
            return f"recall note {args['topic']!r}"
        return f"recall notes matching {args.get('query', '(index)')!r}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        store = NoteStore(ctx.cwd)

        if args.get("topic") is not None:
            topic = require_str(args, "topic")
            note = store.get(topic)
            if note is None:
                known = ", ".join(store.topics()) or "(none)"
                raise ToolError(f"no note named {topic!r}; topics: {known}")
            return f"# {topic}\n{note}"

        notes = store.all()
        if not notes:
            return ("no notes yet -- write_note persists facts you will "
                    "need across turns or sessions")

        query = args.get("query")
        if query is not None:
            needle = require_str(args, "query").lower()
            scored = sorted(
                ((score_match(t, needle) * 3 + score_match(body, needle),
                  t, body)
                 for t, body in notes.items() if score_match(t, needle)
                 or score_match(body, needle)),
                reverse=True,
            )
            if not scored:
                return (f"no notes match {needle!r} "
                        f"(topics: {', '.join(sorted(notes))})")
            lines = [f"#{i+1} [{t}]" for i, (_, t, _) in enumerate(scored)]
            return "\n".join(lines)

        # no query: the index -- enough to know what exists, cheap to scan
        lines = []
        for topic in sorted(notes):
            body = notes[topic].replace("\n", " ")
            preview = body[:PREVIEW_CHARS]
            if len(body) > PREVIEW_CHARS:
                preview += "…"
            lines.append(f"[{topic}] {preview}")
        return "\n".join(lines)
