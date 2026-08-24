"""Session persistence: SQLite round-trips of internal-typed history.

The critical assertion is byte-exact restoration of every block kind --
especially ThinkingBlock signatures, which MUST survive a save/load or
thinking-assisted tool loops 400 after /load. The invariant checker runs
on restored history too: a saved session contains zero outstanding
tool_call ids by construction, and these tests prove it stays that way.
"""

from __future__ import annotations

import json

import pytest

from conftest import ScriptedProvider

from akshara.agent import Agent
from akshara.permissions import allow_read_only
from akshara.session import SessionStore, apply_payload
from akshara.types import (
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
)


def make_agent(history=None, *, model="m1", system="sys", usage=None) -> Agent:
    agent = Agent(
        ScriptedProvider([]),
        model=model,
        system=system,
        permissions=allow_read_only,
    )
    if usage:
        agent.total_usage = usage
    if history:
        agent.history.extend(history)
    return agent


def full_history() -> list[Message]:
    """One message of EVERY block kind, including signed AND redacted
    thinking blocks."""
    return [
        Message("user", [TextBlock("summarize README.md")]),
        Message("assistant", [
            ThinkingBlock("need to read it first", signature="sig-OPAQUE-123"),
            RedactedThinkingBlock("Q0lQSEVSVEVYVA=="),
            ToolCall("call_7", "read_file", {"path": "README.md"}),
        ]),
        Message("user", [
            ToolResult("call_7", "# harness", is_error=False),
            ToolResult("call_9", "no such tool", is_error=True),
        ]),
        Message("assistant", [TextBlock("It's a from-scratch harness.")]),
    ]


def assert_histories_equal(left, right) -> None:
    assert len(left) == len(right)
    for lm, rm in zip(left, right):
        assert lm.role == rm.role
        assert len(lm.content) == len(rm.content)
        for lb, rb in zip(lm.content, rm.content):
            assert lb == rb, f"{type(lb).__name__} mismatch"


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "sessions" / "session.sqlite3")


class TestRoundTrip:
    def test_every_block_kind_restores_byte_exact(self, store):
        original = make_agent(full_history(),
                              usage=Usage(input_tokens=101, output_tokens=22,
                                          cache_read_tokens=5))

        store.save(original, provider_name="anthropic")

        restored = make_agent()
        summary = apply_payload(restored, store.load_latest())

        assert_histories_equal(original.history, restored.history)
        assert restored.total_usage == original.total_usage
        assert restored.model == "m1"
        assert restored.system == "sys"
        assert "4 message(s)" in summary and "101in" in summary

    def test_thinking_signature_survives_exactly(self, store):
        agent = make_agent([Message("assistant", [
            ThinkingBlock("reasoning", signature="sIg-NaTuRe-00"),
        ])])
        store.save(agent, provider_name="anthropic")
        restored = make_agent()
        apply_payload(restored, store.load_latest())
        (block,) = restored.history[0].content
        assert block.signature == "sIg-NaTuRe-00"

    def test_redacted_ciphertext_survives_exactly(self, store):
        """Same contract as signatures: the ciphertext round-trips
        byte-exact or the next tool-loop request 400s after /load."""
        agent = make_agent([Message("assistant", [
            RedactedThinkingBlock("c1pH-Er+TE/xT=="),
        ])])
        store.save(agent, provider_name="anthropic")
        restored = make_agent()
        apply_payload(restored, store.load_latest())
        (block,) = restored.history[0].content
        assert block.data == "c1pH-Er+TE/xT=="

    def test_restored_history_is_resumable(self, store):
        agent = make_agent(full_history())
        store.save(agent, provider_name="openai")
        restored = make_agent()
        apply_payload(restored, store.load_latest())
        open_ids: set[str] = set()
        for m in restored.history:
            match m.role:
                case "assistant":
                    open_ids.update(c.id for c in m.tool_calls())
                case "user":
                    for b in m.content:
                        if isinstance(b, ToolResult):
                            open_ids.discard(b.tool_call_id)
        assert not open_ids


class TestVersioning:
    def test_load_latest_wins_and_versions_append(self, store):
        agent = make_agent([Message("user", [TextBlock("v1 talk")])])
        v1 = store.save(agent, provider_name="anthropic")

        agent.history.append(Message("assistant", [TextBlock("more")]))
        v2 = store.save(agent, provider_name="anthropic")

        assert (v1, v2) == (1, 2)
        payload = store.load_latest()
        assert len(payload["history"]) == 2

    def test_sessions_are_isolated_by_name(self, store):
        store.save(make_agent(), provider_name="anthropic",
                   session_id="alpha")
        assert store.load_latest("beta") is None

    def test_empty_store_returns_none(self, store):
        assert store.load_latest() is None


class TestRestoreSafety:
    def test_unknown_block_kind_raises_cleanly(self, store):
        future_payload = json.dumps({
            "format": 99, "provider": "x", "model": "m",
            "history": [{"role": "user", "content": [{"kind": "holo"}]}],
        })
        store._db.execute(
            "INSERT INTO checkpoints (session_id, version, created_at, payload) "
            "VALUES (?, ?, ?, ?)",
            ("default", 1, "now", future_payload),  # params MUST be a tuple
        )
        store._db.commit()
        with pytest.raises(ValueError, match="holo"):
            apply_payload(make_agent(), store.load_latest())

    def test_provider_rebuild_uses_injected_factory(self, store):
        calls = []
        rebuilt = make_agent().provider

        def factory(name, settings):
            calls.append((name, settings))
            return rebuilt

        original = make_agent()
        store.save(original, provider_name="anthropic")

        agent = make_agent()
        apply_payload(agent, store.load_latest(),
                      settings_loader=lambda n: f"settings-for-{n}",
                      provider_factory=factory)

        assert calls == [("anthropic", "settings-for-anthropic")]
        assert agent.provider is rebuilt
