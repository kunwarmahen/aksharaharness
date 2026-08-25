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
from akshara.permissions import PermissionRequest, SwitchableGate, allow_read_only  # noqa: E402
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


# ---- permission-mode switching -------------------------------------------------


def test_permission_mode_flips_via_rest_and_broadcasts_to_tabs():
    """The mode chip's whole contract: POST flips the live gate, state
    reports it, and every connected tab hears a fresh state envelope."""
    session, agent = make_session([assistant_text("hi")])
    browser_gate = agent.permissions  # the session's real ask-gate
    agent.permissions = SwitchableGate(browser_gate)  # main.py's wiring
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"  # initial snapshot
        flipped = client.post("/api/permissions", json={"mode": "yolo"})
        assert flipped.status_code == 200, flipped.text
        assert flipped.json()["mode"] == "yolo"
        # the flip is announced to open tabs ...
        env = ws.receive_json()
        assert env["type"] == "state" and env["mode"] == "yolo"

    # ... and it is LIVE: while bypassed, a write-shaped request runs
    # without any human on the other end.
    request = PermissionRequest(tool_name="write_thing", arguments={},
                                summary="write_thing('x')", read_only=False)
    assert agent.permissions(request) is True

    back = client.post("/api/permissions", json={"mode": "ask"}).json()
    assert back["mode"] == "ask"
    # Asking again means handing control back to the SAME browser gate --
    # asserted structurally: a direct call here would block on a human
    # that no websocket is answering (that round-trip has its own tests).
    assert agent.permissions.ask is browser_gate


def test_permission_mode_validation_errors():
    session, agent = make_session([assistant_text("hi")])
    agent.permissions = SwitchableGate(allow_read_only)
    client = TestClient(make_app(session))
    assert client.post("/api/permissions", json={"mode": "bypass"}).status_code == 400
    assert client.post("/api/permissions", json={}).status_code == 400
    assert agent.permissions.mode == "ask"  # failures never half-flip


def test_permission_mode_can_flip_mid_turn():
    """/api/permissions deliberately skips require_idle: flipping during a
    running turn applies to its not-yet-approved calls (that's the point)."""
    session, agent = make_session([assistant_text("hi")])
    agent.permissions = SwitchableGate(agent.permissions)  # main.py's wiring
    session.turn_active = True  # force the race deterministically
    client = TestClient(make_app(session))
    try:
        response = client.post("/api/permissions", json={"mode": "yolo"})
        assert response.status_code == 200
    finally:
        session.turn_active = False


def test_permissions_endpoint_rejects_fixed_gate():
    """A session built with a bare gate (tests, embedders) answers with a
    clean 400 -- the UI chip shows 'fixed', not a silent no-op."""
    session, _ = make_session([assistant_text("hi")])  # bare browser gate
    client = TestClient(make_app(session))
    response = client.post("/api/permissions", json={"mode": "yolo"})
    assert response.status_code == 400
    assert "fixed" in response.json()["detail"]
    assert client.get("/api/state").json()["mode"] == "ask"


# ---- tools panel + runtime toggles ----------------------------------------------


def test_tools_listing_carries_detail():
    session, _ = make_session([assistant_text("hi")])
    client = TestClient(make_app(session))
    tools = {t["name"]: t for t in client.get("/api/tools").json()}
    assert set(tools) == {"write_thing", "echo", "ask_user"}
    assert tools["echo"]["read_only"] is True
    assert tools["write_thing"]["read_only"] is False
    assert all(t["enabled"] for t in tools.values())
    assert all(t["description"] for t in tools.values())


def test_toggle_disables_and_state_reports_it():
    session, agent = make_session([assistant_text("hi")])
    client = TestClient(make_app(session))
    out = client.post("/api/tools", json={"name": "echo", "enabled": False})
    assert out.status_code == 200, out.text
    assert out.json()["disabled_tools"] == ["echo"]
    assert agent.registry.is_disabled("echo")
    # and back
    out = client.post("/api/tools", json={"name": "echo", "enabled": True})
    assert out.json()["disabled_tools"] == []


def test_disabled_tool_call_fails_as_data_mid_turn():
    """The point of the mid-turn-safe toggle: pulling a tool applies to the
    RUNNING turn's next call -- the loop re-consults the registry per call.
    A pending ask_user parks the worker between calls, so the toggle lands
    deterministically before the second write_thing is ever gated."""
    script = [
        assistant_tool_call("c1", "write_thing", {"text": "one"}),
        assistant_tool_call("a1", "ask_user", {"question": "continue?"}),
        assistant_tool_call("c2", "write_thing", {"text": "two"}),
        assistant_text("done"),
    ]
    session, agent = make_session(script)
    client = TestClient(make_app(session))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert client.post("/api/message", json={"text": "go"}).status_code == 200

        req = next(e for e in drain_until(ws, {"permission_request"})
                   if e["type"] == "permission_request")
        ws.send_json({"type": "answer", "id": req["id"], "decision": "approve"})
        drain_until(ws, {"tool_result"})  # first call runs normally

        ask = next(e for e in drain_until(ws, {"ask"}) if e["type"] == "ask")
        # the turn is parked here -- flip the switch, then let it resume
        assert session.turn_active
        pulled = client.post("/api/tools",
                             json={"name": "write_thing", "enabled": False})
        assert pulled.status_code == 200  # no 409: mid-turn is the point
        ws.send_json({"type": "answer", "id": ask["id"], "text": "yes"})

        envelopes = drain_until(ws, {"turn_done"})
        cards = [e for e in envelopes if e["type"] == "tool_result"]
        assert len(cards) == 2
        assert not cards[0]["is_error"]
        assert cards[1]["is_error"]
        assert "disabled by the operator" in cards[1]["output"]

    # resumable, as on every abnormal-ish path -- three results: the
    # clean write, ask_user's answer receipt, then the pulled tool's data
    from akshara.types import ToolResult
    results = [b for m in agent.history if m.role == "user"
               for b in m.content if isinstance(b, ToolResult)]
    assert [r.is_error for r in results] == [False, False, True]


def test_toggle_validation_errors():
    session, _ = make_session([assistant_text("hi")])
    client = TestClient(make_app(session))
    assert client.post("/api/tools",
                       json={"name": "ghost", "enabled": False}).status_code == 404
    assert client.post("/api/tools", json={"name": "echo"}).status_code == 400
    assert client.post("/api/tools",
                       json={"enabled": False}).status_code == 400


def test_toggle_broadcasts_state_to_open_tabs():
    session, _ = make_session([assistant_text("hi")])
    client = TestClient(make_app(session))
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        client.post("/api/tools", json={"name": "echo", "enabled": False})
        env = ws.receive_json()
        assert env["type"] == "state"
        assert env["disabled_tools"] == ["echo"]


def test_attach_wires_interrupt_check_to_cancel_flag():
    """The Stop button's teeth are a wired flag, not a hope: attach() points
    the loop's poll at the session cancel event."""
    session, agent = make_session([assistant_text("hi")])
    assert callable(agent.interrupt_check)
    assert agent.interrupt_check() is False
    session.cancel_turn()
    assert agent.interrupt_check() is True


def test_state_carries_pressure_numbers():
    """The tooltip's honest numbers: the provider-reported footprint (not
    an estimate) once one exists, plus the window it fills -- and the
    estimated flag flipping accordingly."""
    session, agent = make_session([assistant_text("hi")])
    agent.history.append(Message("user", [TextBlock("seed")]))
    agent.last_context_tokens = 5000

    state = TestClient(make_app(session)).get("/api/state").json()
    assert state["context_estimated"] is False
    assert state["context_tokens"] == 5000
    assert state["context_window"] == 200_000
    assert 0 < state["utilization"] < 1

    agent.last_context_tokens = 0  # no response yet: estimate + flag
    state = TestClient(make_app(session)).get("/api/state").json()
    assert state["context_estimated"] is True


# ---- mcp servers: the panel's servers section ---------------------------------


class _FakeMCPSession:
    """Manager-grade stand-in for a live session (records close())."""

    def __init__(self, config):
        self.config = config
        self.closed = False

    def healthy(self):
        return not self.closed

    def close(self):
        self.closed = True


class _MCPTool(Tool):
    name = ""
    description = "fake mcp tool"
    parameters = {"type": "object", "properties": {}}
    read_only = True

    def __init__(self, session, raw_name, server):
        self.name = f"mcp__{server}__{raw_name}"

    def summary(self, args, ctx):
        return self.name

    def run(self, args, ctx):
        return "ran"


def fake_mcp_connector(fail_names=()):
    def connector(config, timeout=30.0):
        if config.name in fail_names:
            from akshara.mcp import MCPError
            raise MCPError(f"cannot spawn mcp server {config.name!r}")
        s = _FakeMCPSession(config)
        return s, [_MCPTool(s, "echo", config.name),
                   _MCPTool(s, "ping", config.name)]
    return connector


def make_mcp_session(script=None, *, fail_names=()):
    """Like make_session, plus an MCPManager over the same registry with
    the transport faked out (the transports have their own real-subprocess
    tests in test_mcp.py)."""
    from akshara.mcp import MCPManager

    session, agent = make_session(script or [])
    manager = MCPManager(agent.registry, connector=fake_mcp_connector(fail_names))
    session.attach(agent, None, mcp=manager)
    return session, agent, manager


def test_no_manager_answers_400_cleanly():
    client = TestClient(make_app(make_session([])[0]))
    assert client.get("/api/mcp").status_code == 400


def test_listing_and_add_then_remove_round_trip():
    session, agent, manager = make_mcp_session()
    client = TestClient(make_app(session))

    assert client.get("/api/mcp").json() == {"servers": []}

    r = client.post("/api/mcp/add", json={"name": "tiny",
                                          "command": "python",
                                          "args": ["srv.py"]})
    assert r.status_code == 200
    (result,) = r.json()["results"]
    assert result == {"name": "tiny", "ok": True, "tools": 2}
    assert "mcp__tiny__echo" in agent.registry

    listing = client.get("/api/mcp").json()["servers"]
    assert listing[0]["name"] == "tiny"
    assert listing[0]["transport"] == "stdio"
    assert listing[0]["healthy"] is True
    assert listing[0]["tools"] == 2
    assert listing[0]["disabled"] == 0

    # remove pulls exactly that server's tools out of the registry...
    r = client.post("/api/mcp/remove", json={"name": "tiny"})
    assert r.status_code == 200 and r.json()["removed"] == 2
    assert "mcp__tiny__echo" not in agent.registry
    tool_names = [t["name"] for t in client.get("/api/tools").json()]
    assert not any(n.startswith("mcp__tiny") for n in tool_names)
    assert client.post("/api/mcp/remove", json={"name": "tiny"}).status_code == 404


def test_toggle_soft_switches_whole_server_even_mid_turn():
    from akshara.mcp import MCPServerConfig

    session, agent, manager = make_mcp_session()
    manager.connect(MCPServerConfig(name="tiny", command="py"))
    client = TestClient(make_app(session))

    r = client.post("/api/mcp/toggle", json={"name": "tiny", "enabled": False})
    assert r.status_code == 200
    assert agent.registry.is_disabled("mcp__tiny__echo")
    assert agent.registry.is_disabled("mcp__tiny__ping")
    assert client.get("/api/mcp").json()["servers"][0]["disabled"] == 2

    client.post("/api/mcp/toggle", json={"name": "tiny", "enabled": True})
    assert agent.registry.disabled_names() == []

    # mid-turn is the POINT here, same as /api/tools: no require_idle
    session.turn_active = True
    assert client.post("/api/mcp/toggle",
                       json={"name": "tiny", "enabled": False}).status_code == 200

    assert client.post("/api/mcp/toggle",
                       json={"name": "ghost", "enabled": True}).status_code == 404
    assert client.post("/api/mcp/toggle",
                       json={"name": "tiny"}).status_code == 400


def test_add_reports_connection_failure_as_data_not_a_500():
    session, agent, manager = make_mcp_session(fail_names=("dead",))
    client = TestClient(make_app(session))
    r = client.post("/api/mcp/add", json={"name": "dead", "command": "nope"})
    assert r.status_code == 200
    (result,) = r.json()["results"]
    assert result["ok"] is False
    assert "dead" in result["error"]
    assert agent.registry.names().count("mcp__dead__echo") == 0


def test_add_json_mode_handles_multiple_servers_with_mixed_results():
    session, agent, manager = make_mcp_session(fail_names=("bad",))
    client = TestClient(make_app(session))
    config = ('{"servers": {"good": {"command": "py"}, '
              '"bad": {"url": "http://x/mcp"}}}')
    results = client.post("/api/mcp/add",
                          json={"config": config}).json()["results"]
    by_name = {r["name"]: r["ok"] for r in results}
    assert by_name == {"good": True, "bad": False}
    assert "mcp__good__ping" in agent.registry

    bad = client.post("/api/mcp/add", json={"config": "{not json"})
    assert bad.status_code == 400


def test_add_validates_the_field_shape():
    session, _, _ = make_mcp_session()
    client = TestClient(make_app(session))
    assert client.post("/api/mcp/add", json={"command": "py"}).status_code == 400
    assert client.post("/api/mcp/add",
                       json={"name": "x", "command": "a", "url": "http://y"}
                       ).status_code == 400
    assert client.post("/api/mcp/add",
                       json={"name": "x", "command": "py", "args": [1]}
                       ).status_code == 400
    assert client.post("/api/mcp/add",
                       json={"name": "x", "command": "py", "env": {"K": 1}}
                       ).status_code == 400


def test_add_and_remove_refuse_to_race_a_running_turn():
    session, _, _ = make_mcp_session()
    session.turn_active = True
    client = TestClient(make_app(session))
    assert client.post("/api/mcp/add",
                       json={"name": "x", "command": "py"}).status_code == 409
    assert client.post("/api/mcp/remove", json={"name": "x"}).status_code == 409


def test_remember_flag_persists_the_entry(tmp_path):
    from akshara.mcp import load_remembered

    session, _, manager = make_mcp_session()
    manager.memory_path = tmp_path / ".akshara" / "mcp.json"
    client = TestClient(make_app(session))
    client.post("/api/mcp/add", json={"name": "kept", "command": "py",
                                      "remember": True})
    client.post("/api/mcp/add", json={"name": "fleeting", "command": "py"})
    saved = {c.name for c in load_remembered(manager.memory_path)}
    assert saved == {"kept"}
    client.post("/api/mcp/remove", json={"name": "kept"})
    assert load_remembered(manager.memory_path) == []
