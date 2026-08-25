"""The todo list -- live plan state, as opposed to memory's durable facts.

write_note/recall_notes answer "what do I KNOW that outlives this
session?" A mission that spans many turns needs a different artifact:
"what am I DOING, what's left, what's done?" -- a plan the model can
look at after every tool batch instead of re-deriving from transcript
scroll. This matters most for local models, which drift further on
long multi-step missions; an explicit list to re-read is cheap
grounding.

One store, two verbs, deliberately shaped like memory.py:

* ``todo_write(items)`` -- REPLACE the whole list. Not upsert-by-index:
  plans mutate wholesale mid-mission (reorder, split, drop), and
  patch-style edits against indexes drift the moment two items swap.
  The model sends its current plan; the file holds exactly that.
* ``todo_read()``      -- the checklist as text.

Statuses are exactly three: pending / in_progress / done. One
in_progress at a time is a CONVENTION the description nudges, not a
rule the code enforces -- parallel work streams exist, and a harness
that hard-fails honest plans teaches the model to lie about them.

Persistence mirrors NoteStore: one JSON document under ``.akshara/``,
written temp-then-rename so a crash mid-write never truncates it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str

STATUSES = ("pending", "in_progress", "done")
MAX_ITEMS = 50
MAX_TASK_CHARS = 500


def _store_path(ctx: ToolContext) -> Path:
    return ctx.cwd / ".akshara" / "todos.json"


def _load(ctx: ToolContext) -> list[dict[str, str]]:
    try:
        raw = json.loads(_store_path(ctx).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        raise ToolError(f"todo store corrupted ({exc}); fix or delete "
                        f"{_store_path(ctx)}") from exc
    return [item for item in raw.get("todos", [])
            if item.get("task") and item.get("status") in STATUSES]


def _save(ctx: ToolContext, todos: list[dict[str, str]]) -> None:
    path = _store_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"todos": todos}, fh, indent=2)
        os.replace(tmp, path)  # atomic: readers see old or new, never half
    except BaseException:
        try:
            os.unlink(tmp)
        finally:
            raise


def render(todos: list[dict[str, str]]) -> str:
    """The checklist as plain text -- same shape everywhere it's shown."""
    marks = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
    return "\n".join(f"{marks[i['status']]} {i['task']}" for i in todos)


def _validated_items(args: dict[str, Any]) -> list[dict[str, str]]:
    """Pull the items array out of model-supplied JSON; replace-all means
    an empty list is legal (the plan was completed or abandoned)."""
    raw_items = args.get("items", [])
    if not isinstance(raw_items, list):
        raise ToolError("'items' must be a list of {task, status} objects")
    if len(raw_items) > MAX_ITEMS:
        raise ToolError(f"too many items ({len(raw_items)}); max {MAX_ITEMS} "
                        "-- split the mission, or keep the list at the "
                        "resolution of next steps, not sub-bullets")
    items: list[dict[str, str]] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            raise ToolError("every item must be an object with 'task' and 'status'")
        task = require_str(entry, "task").strip()
        status = entry.get("status", "pending")
        if not task:
            raise ToolError("every item needs a non-empty 'task'")
        if len(task) > MAX_TASK_CHARS:
            raise ToolError(f"task too long ({len(task)} chars; max "
                            f"{MAX_TASK_CHARS}) -- a step, not a spec")
        if status not in STATUSES:
            raise ToolError(f"status must be one of {', '.join(STATUSES)}; "
                            f"got {status!r}")
        items.append({"task": task, "status": status})
    return items


class TodoWrite(Tool):
    name = "todo_write"
    description = (
        "Replace your task list with the given items ({task, status}; "
        f"status is pending/in_progress/done, max {MAX_ITEMS}). Send the "
        "WHOLE current plan each time -- reorder, split, and drop freely. "
        "Keep roughly one item in_progress. Write here whenever the plan "
        "changes or a step completes, so the list never goes stale."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string",
                                 "description": "The step, imperative and "
                                                "concrete."},
                        "status": {"type": "string",
                                   "enum": list(STATUSES),
                                   "description": "Default: pending."},
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                },
                "description": "The full new list, in order.",
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        items = args.get("items") or []
        return f"replace todo list ({len(items)} items)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        items = _validated_items(args)
        had = len(_load(ctx))
        _save(ctx, items)
        return (f"todo list replaced ({had} -> {len(items)} items)\n"
                + render(items))


class TodoRead(Tool):
    name = "todo_read"
    description = (
        "Read your current task list ([ ] pending, [~] in_progress, "
        "[x] done). Check it when deciding what to do next -- it is the "
        "plan you wrote yourself."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "read todo list"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        items = _load(ctx)
        if not items:
            return ("no tasks tracked -- todo_write replaces the whole "
                    "list with your current plan")
        counts = {s: sum(1 for i in items if i["status"] == s)
                  for s in STATUSES}
        return (f"{counts['done']} done / {counts['in_progress']} active / "
                f"{counts['pending']} pending\n" + render(items))
