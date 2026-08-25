"""Durable sessions: the transcript survives process death.

SQLite (book ch21's choice) with VERSIONED checkpoints: every save
appends a row, nothing is updated in place. Cheap on disk; any prior
version stays loadable for debugging, and a crashed save can never
corrupt the previous good state -- that is what "durable" means here.

What we persist is deliberately small because of one design invariant:
history is ALWAYS resumable at prompt boundaries (the
invariant), so there is no such thing as saving mid-tool-call state.
No idempotency ledger is needed -- a saved session contains zero
outstanding tool_call ids by construction.

Payload shape (JSON inside checkpoints.payload):

    {"format": 1,
     "provider": "anthropic",        # rebuilt via get_provider on load
     "model": "...", "system": ..., "max_iterations": 25,
     "total_usage": {"input_tokens": ..., ...},
     "history": [{"role": "user",
                  "content": [{"kind": "text", "text": ...}, ...]}, ...]}

Blocks serialize with an explicit ``kind`` discriminator and restore
via match/case dispatch -- never guesswork. ThinkingBlock signatures
are opaque strings; they round-trip byte-exact or thinking-assisted
tool loops will 400 after a /load.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from akshara.types import (
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    version    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload    TEXT NOT NULL,
    UNIQUE (session_id, version)
);
"""


class SessionStore:
    """Append-only checkpoint store. One file holds many named sessions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: the store is constructed on the
        # main thread but the web UI's endpoints run on the server's event
        # loop thread -- sqlite connections are thread-affine by default,
        # which would 500 every /api/save. Writes stay serialized (one lock,
        # one connection), which is all a local single-process store needs.
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---- public API ----------------------------------------------------------

    def save(self, agent, *, provider_name: str, session_id: str = "default") -> int:
        """Snapshot ``agent`` as the next version; returns its version number."""
        latest = self.latest_version(session_id)
        payload = {
            "format": 1,
            "provider": provider_name,
            "model": agent.model,
            "system": agent.system,
            "max_iterations": agent.max_iterations,
            "total_usage": {
                "input_tokens": agent.total_usage.input_tokens,
                "output_tokens": agent.total_usage.output_tokens,
                "cache_read_tokens": agent.total_usage.cache_read_tokens,
                "cache_write_tokens": agent.total_usage.cache_write_tokens,
            },
            "history": [_dump_message(m) for m in agent.history],
        }
    def save(self, agent, *, provider_name: str, session_id: str = "default") -> int:
        """Snapshot ``agent`` as the next version; returns its version number."""
        with self._lock:
            payload = {
                "format": 1,
                "provider": provider_name,
                "model": agent.model,
                "system": agent.system,
                "max_iterations": agent.max_iterations,
                "total_usage": {
                    "input_tokens": agent.total_usage.input_tokens,
                    "output_tokens": agent.total_usage.output_tokens,
                    "cache_read_tokens": agent.total_usage.cache_read_tokens,
                    "cache_write_tokens": agent.total_usage.cache_write_tokens,
                },
                "history": [_dump_message(m) for m in agent.history],
            }
            latest = self._latest_version(session_id)
            self._db.execute(
                "INSERT INTO checkpoints (session_id, version, created_at, payload) "
                "VALUES (?, ?, ?, ?)",
                (session_id, latest + 1,
                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 json.dumps(payload)),
            )
            self._db.commit()
            return latest + 1

    def load_latest(self, session_id: str = "default") -> dict | None:
        """The newest payload for ``session_id``, or None if never saved."""
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM checkpoints WHERE session_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def latest_version(self, session_id: str = "default") -> int:
        with self._lock:
            return self._latest_version(session_id)

    def _latest_version(self, session_id: str) -> int:
        """Caller holds the lock."""
        row = self._db.execute(
            "SELECT MAX(version) FROM checkpoints WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else 0


# ---- block-level (de)serialization: kind-discriminated, match-dispatched ----


def _dump_message(message: Message) -> dict:
    content = []
    for block in message.content:
        match block:
            case TextBlock(text=text):
                content.append({"kind": "text", "text": text})
            case ToolCall(id=cid, name=name, arguments=args):
                content.append({"kind": "tool_call", "id": cid,
                                "name": name, "arguments": args})
            case ToolResult(tool_call_id=cid, content=out, is_error=err):
                content.append({"kind": "tool_result", "tool_call_id": cid,
                                "content": out, "is_error": err})
            case ThinkingBlock(thinking=text, signature=sig):
                content.append({"kind": "thinking", "thinking": text,
                                "signature": sig})
            case RedactedThinkingBlock(data=payload):
                # Same byte-exact contract as signatures: ciphertext the
                # provider validates on the next request.
                content.append({"kind": "redacted_thinking", "data": payload})
            case ImageBlock(media_type=mime, data=b64):
                # User attachments AND tool-produced images (read_image)
                # both round-trip: base64 is the payload, so this is
                # byte-exact by construction -- no re-encode, no loss.
                content.append({"kind": "image", "media_type": mime,
                                "data": b64})
    return {"role": message.role, "content": content}


def _load_message(raw: dict) -> Message:
    blocks = []
    for item in raw.get("content", []):
        match item.get("kind"):
            case "text":
                blocks.append(TextBlock(item["text"]))
            case "tool_call":
                blocks.append(ToolCall(item["id"], item["name"],
                                       item.get("arguments") or {}))
            case "tool_result":
                blocks.append(ToolResult(item["tool_call_id"],
                                         item.get("content", ""),
                                         is_error=item.get("is_error", False)))
            case "thinking":
                blocks.append(ThinkingBlock(item.get("thinking", ""),
                                            signature=item.get("signature", "")))
            case "redacted_thinking":
                blocks.append(RedactedThinkingBlock(item.get("data", "")))
            case "image":
                blocks.append(ImageBlock(item["media_type"], item["data"]))
            case unknown:
                raise ValueError(f"cannot restore block kind {unknown!r} "
                                 f"(payload format newer than this code?)")
    return Message(raw["role"], blocks)


def apply_payload(agent, payload: dict, *, settings_loader=None,
                  provider_factory=None) -> str:
    """Restore ``payload`` into ``agent`` IN PLACE. Returns a one-line
    summary for the UI.

    Provider/model rebuild goes through injected callables so this module
    stays decoupled from provider construction (and testable offline).
    """
    provider_name = payload["provider"]
    model = payload["model"]
    if provider_factory is not None and settings_loader is not None:
        agent.provider = provider_factory(provider_name, settings_loader(provider_name))
    agent.model = model
    agent.system = payload.get("system")
    agent.max_iterations = payload.get("max_iterations", 25)
    usage = payload.get("total_usage") or {}
    agent.total_usage = Usage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        cache_write_tokens=usage.get("cache_write_tokens", 0),
    )
    agent.history.clear()
    agent.history.extend(_load_message(m) for m in payload.get("history", []))
    n_msgs = len(agent.history)
    u = agent.total_usage
    return (f"restored {n_msgs} message(s), "
            f"{u.input_tokens}in/{u.output_tokens}out · "
            f"{provider_name}/{model}")
