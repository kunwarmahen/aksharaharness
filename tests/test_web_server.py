"""Web-server tests: the browser UI's contract, fully offline.

fastapi comes from the optional [web] extra, so these skip when it isn't
installed -- the core suite must stay runnable with zero extras.

The scripts are deterministic (ScriptedProvider), so receives need no
timeouts: every envelope the worker emits is already decided by the
script. The pattern throughout: connect /ws (state arrives first),
POST /api/message, then drain envelopes until turn_done.
"""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient # noqa: E402

from conftest import ScriptedProvider, assistant_text, assistant_tool_call  # noqa: E402
from akshara.agent import Agent  # noqa: E402
from akshara.providers.base import ProviderSettings  # noqa: E402
from akshara.session import SessionStore  # noqa: E402
from akshara.tools.ask_user import AskUser  # noqa: E402
from akshara.tools.base import Tool, ToolRegistry  # noqa: E402
from akshara.types import Message, ModelResponse, TextBlock, Usage  # noqa: E402
from akshara.web.server import WebSession, make_app  # noqa: E402


class WriteThing(Tool):
    """A gated tool (not read_only): its calls produce permission requests."""

    name = "write_thing"
    description = "write something"
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string"}}}
    read_only = False

    def summary(self, args, ctx):
        return f"write_thing({args.get('text')!r})"

    def run(self, args, ctx):
        return f"wrote:{args.get('text', '')}"


class Echo(Tool):
    """A read-only tool: never gates, so a scripted call to it runs free
    and gives the turn a real second iteration."""

    name = "echo"
    description = "echo text back"
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string"}}}
    read_only = True

    def summary(self, args, ctx):
        return f"echo({args.get('text')!r})"

    def run(self, args, ctx):
        return f"echo:{args.get('text', '')}"


def make_session(script: list[ModelResponse], *, store=None,
                 extra_tools: tuple = (),
                 permissions=None) -> tuple[WebSession, Agent]:
    """Session + agent wired exactly like cli/main.py does for --web:
    the gate IS the session's browser gate (read-only tools still never
    prompt; writes round-trip over the websocket)."""
    session = WebSession()
    registry = ToolRegistry()
    registry.register(WriteThing())
    registry.register(Echo())
    registry.register(AskUser(session.channel))
    for tool in extra_tools:
        registry.register(tool)
    agent = Agent(ScriptedProvider(script), model="m", tools=registry,
                  permissions=permissions or session.permission_gate())
    session.attach(agent, store)
    return session, agent


def drain_until(ws, wanted: set[str], limit: int = 100) -> list[dict]:
    """Collect envelopes until one of ``wanted`` types arrives (inclusive)."""
    out = []
    while len(out) < limit:
        env = ws.receive_json()
        out.append(env)
        if env["type"] in wanted:
            return out
    raise AssertionError(f"never got any of {wanted}")


def send_and_finish(client, ws, text: str) -> list[dict]:
    """Post a message and collect everything up to and incl. turn_done."""
    assert client.post("/api/message", json={"text": text}).status_code == 200
    return drain_until(ws, {"turn_done"})


# ---- connection & replay -----------------------------------------------------


def test_connect_sends_state_then_history_replay():
    script = [assistant_text("hello there")]
    session, agent = make_session(script)
    list(agent.run_streaming("hi"))  # seed history before serving
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        state = ws.receive_json()
        assert state["type"] == "state"
        assert state["model"] == "m"
        replay = [ws.receive_json(), ws.receive_json()]
        assert replay[0]["type"] == "user_message"
        assert replay[0]["text"] == "hi"
        assert replay[1] == {"type": "assistant_text", "text": "hello there"}


# ---- plain streaming turn ------------------------------------------------------


def test_message_streams_deltas_tool_results_and_footer():
    script = [
        assistant_tool_call("e1", "echo", {"text": "one two three"}),
        assistant_text("done"),
    ]
    session, agent = make_session(script)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        envelopes = send_and_finish(client, ws, "count please")

        kinds = [e["type"] for e in envelopes]
        assert "turn_started" in kinds
        # ScriptedProvider splits each text in half -> 2 deltas for "done"
        assert kinds.count("delta") >= 2
        card = next(e for e in envelopes if e["type"] == "tool_result")
        assert card["name"] == "echo"
        assert card["output"] == "echo:one two three"
        end = next(e for e in envelopes if e["type"] == "turn_end")
        assert end["reason"] == "end_turn"
        assert end["iterations"] == 2  # tool round + final text
        footer = envelopes[-2]
        assert footer["type"] == "state"
        assert footer["turn_active"] is False


def seed_history(agent: Agent) -> None:
    """Write a user -> assistant(+tool_call) -> tool_result exchange straight
    into history. Deliberately NOT via run_streaming: a seeded gated call
    would block on the web gate, and no tab is connected yet."""
    from akshara.types import ToolCall as TC, ToolResult as TR

    agent.history.append(Message("user", [TextBlock("do it")]))
    agent.history.append(Message("assistant", [
        TextBlock("on it"),
        TC(id="t1", name="write_thing", arguments={"text": "x"}),
    ]))
    agent.history.append(Message("user", [
        TR(tool_call_id="t1", content="wrote:x", is_error=False),
    ]))


def test_history_replay_carries_tool_cards():
    session, agent = make_session([assistant_text("never called")])
    seed_history(agent)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        replay = drain_until(ws, {"tool_result"})
        kinds = [e["type"] for e in replay]
        assert "user_message" in kinds
        assert "assistant_text" in kinds
        card = next(e for e in replay if e["type"] == "tool_result")
        assert card["name"] == "write_thing"
        assert card["arguments"] == {"text": "x"}
        assert card["output"] == "wrote:x"
        assert card["is_error"] is False


# ---- permission gate over the wire ----------------------------------------------


def test_permission_deny_becomes_error_data():
    script = [
        assistant_tool_call("t1", "write_thing", {"text": "danger"}),
        assistant_text("okay, skipping that"),
    ]
    session, agent = make_session(script)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert client.post("/api/message", json={"text": "go"}).status_code == 200

        req = next(e for e in drain_until(ws, {"permission_request"})
                   if e["type"] == "permission_request")
        assert req["tool_name"] == "write_thing"
        assert req["summary"] == "write_thing('danger')"
        ws.send_json({"type": "answer", "id": req["id"], "decision": "deny"})

        envelopes = drain_until(ws, {"turn_done"})
        result = next(e for e in envelopes if e["type"] == "tool_result")
        assert result["is_error"]
        assert "denied" in result["output"].lower()


def test_permission_edit_round_trip_adopts_edited_args():
    script = [
        assistant_tool_call("t1", "write_thing", {"text": "wrong path"}),
        assistant_text("done"),
    ]
    session, agent = make_session(script)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert client.post("/api/message", json={"text": "go"}).status_code == 200

        first = next(e for e in drain_until(ws, {"permission_request"})
                     if e["type"] == "permission_request")
        ws.send_json({"type": "answer", "id": first["id"],
                      "decision": "edit",
                      "edited_args": {"text": "right path"}})

        second = next(e for e in drain_until(ws, {"permission_request"})
                      if e["type"] == "permission_request")
        assert second["edited"] is True
        assert second["arguments"] == {"text": "right path"}
        assert second["summary"] == "write_thing('right path')"  # re-rendered
        ws.send_json({"type": "answer", "id": second["id"],
                      "decision": "approve"})

        envelopes = drain_until(ws, {"turn_done"})
        result = next(e for e in envelopes if e["type"] == "tool_result")
        assert result["arguments"] == {"text": "right path"}
        assert result["output"] == "wrote:right path"
        # what ran is what history keeps (approve-with-edits invariant)
        assert any(b.arguments == {"text": "right path"}
                   for m in agent.history if m.role == "assistant"
                   for b in m.tool_calls())


# ---- ask_user over the wire ------------------------------------------------------


def test_ask_round_trip_continues_the_turn():
    script = [
        assistant_tool_call("a1", "ask_user",
                            {"question": "Which DB?",
                             "choices": ["SQLite", "Postgres"],
                             "context": "single user tool"}),
        assistant_text("going with SQLite then"),
    ]
    session, agent = make_session(script)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert client.post("/api/message", json={"text": "build"}).status_code == 200

        ask = next(e for e in drain_until(ws, {"ask"})
                   if e["type"] == "ask")
        assert ask["question"] == "Which DB?"
        assert ask["choices"] == ["SQLite", "Postgres"]
        assert ask["context"] == "single user tool"
        ws.send_json({"type": "answer", "id": ask["id"], "text": "SQLite"})

        envelopes = drain_until(ws, {"turn_done"})
        result = next(e for e in envelopes if e["type"] == "tool_result")
        assert "(picked option 1/2)" in result["output"]
        assert "SQLite" in result["output"]
        assert envelopes[-1]["type"] == "turn_done"


def test_cancel_while_ask_pending_cancels_turn_but_session_lives():
    script = [
        assistant_tool_call("a1", "ask_user", {"question": "proceed?"}),
        assistant_text("fresh turn"),  # consumed by the SECOND message
    ]
    session, agent = make_session(script)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert client.post("/api/message", json={"text": "go"}).status_code == 200
        ask = next(e for e in drain_until(ws, {"ask"}) if e["type"] == "ask")
        ws.send_json({"type": "cancel"})

        envelopes = drain_until(ws, {"turn_done"})
        kinds = [e["type"] for e in envelopes]
        assert "resolved" in kinds          # modal dismissed
        assert "turn_cancelled" in kinds    # honest report to the tab

        # resumable: the interrupted call was synthesized; a new turn works
        assert client.post("/api/message", json={"text": "again"}).status_code == 200
        envelopes = drain_until(ws, {"turn_done"})
        assert any(e.get("reason") == "end_turn"
                   for e in envelopes if e["type"] == "turn_end")


# ---- REST behavior -----------------------------------------------------------------


def test_concurrent_message_is_rejected_with_409():
    session, _ = make_session([assistant_text("hi")])
    session.turn_active = True  # force the race deterministically
    client = TestClient(make_app(session))
    assert client.post("/api/message", json={"text": "x"}).status_code == 409
    session.turn_active = False


def test_message_validation_errors():
    session, _ = make_session([])
    client = TestClient(make_app(session))
    assert client.post("/api/message", json={}).status_code == 400
    bad_image = {"text": "hi", "images": [{"filename": "a.txt",
                                           "data_base64": "aGk="}]}
    assert client.post("/api/message", json=bad_image).status_code == 400


def test_model_switch_reflected_in_state():
    session, _ = make_session([assistant_text("hi")])
    client = TestClient(make_app(session))
    data = client.post("/api/model", json={"model": "qwen3.8"}).json()
    assert data["model"] == "qwen3.8"
    assert client.post("/api/model", json={}).status_code == 400


def test_save_load_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "session.sqlite3")
    session, agent = make_session([assistant_text("hi")], store=store)
    seed_history(agent)
    # The checkpoint records provider 'scripted', which no real registry
    # can rebuild -- inject the factories apply_payload exposes so the
    # scripted provider survives the restore (production keeps its strict
    # default; see test below for the failure shape).
    client = TestClient(make_app(
        session,
        settings_loader=lambda name: ProviderSettings(api_key="test",
                                                      base_url="http://test"),
        provider_factory=lambda name, settings: agent.provider,
    ))

    saved = client.post("/api/save", json={"name": "keep"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["version"] == 1
    client.post("/api/model", json={"model": "changed-slug"})
    loaded = client.post("/api/load", json={"name": "keep"})
    assert loaded.status_code == 200, loaded.text
    assert len(agent.history) == 3  # restored from the checkpoint
    assert agent.model == "m"       # ...including the saved model slug
    assert client.post("/api/load", json={"name": "nope"}).status_code == 404


def test_load_with_unbuildable_provider_is_a_clean_400(tmp_path):
    """Production factories are strict: a checkpoint whose provider has no
    API key (or no registry entry) fails the restore without touching the
    live agent -- same contract as the REPL's /load."""
    session, agent = make_session([assistant_text("hi")],
                                  store=SessionStore(tmp_path / "s.sqlite3"))
    seed_history(agent)
    client = TestClient(make_app(session))

    client.post("/api/save", json={"name": "keep"})
    failed = client.post("/api/load", json={"name": "keep"})
    assert failed.status_code == 400
    assert "restore failed" in failed.json()["detail"]
    assert len(agent.history) == 3  # untouched: restore is all-or-nothing


def test_compact_and_clear_endpoints():
    session, agent = make_session([assistant_text("hi")])
    for i in range(6):  # enough messages that compaction has room to bite
        agent.history.append(Message("user", [TextBlock(f"msg {i}")]))
    client = TestClient(make_app(session))

    stats = client.post("/api/compact").json()["stats"]
    assert "messages_before" in stats
    n_before = len(agent.history)
    client.post("/api/clear")
    assert len(agent.history) < n_before


def test_state_endpoint():
    session, _ = make_session([assistant_text("hi")])
    client = TestClient(make_app(session))
    state = client.get("/api/state").json()
    assert state["provider"] == "scripted"
    assert "ask_user" in state["tools"]
