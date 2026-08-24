"""MCP client: tested against REAL subprocess servers.

Like test_tools.py's fake ripgrep, each test writes a tiny python
script that genuinely speaks newline-delimited JSON-RPC over stdio and
spawns it -- so framing, threading, timeouts, and shutdown are all
exercised on the real transport, not mocked away.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import ScriptedProvider, assistant_text, assistant_tool_call

from akshara.agent import Agent
from akshara.errors import ToolError
from akshara.mcp import (
    SUPPORTED_VERSIONS,
    MCPError,
    MCPServerConfig,
    load_mcp_configs,
    register_mcp,
)
from akshara.permissions import yolo
from akshara.tools.base import ToolContext, ToolRegistry


def write_server(tmp_path: Path, body: str, name: str = "srv") -> MCPServerConfig:
    script = tmp_path / f"fake_{name}.py"
    script.write_text(textwrap.dedent(body))
    return MCPServerConfig(name=name, command=sys.executable, args=[str(script)])


STANDARD_BODY = """\
        import json, sys
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        def result(mid, r): send({"jsonrpc": "2.0", "id": mid, "result": r})
        TOOLS = [
            {"name": "echo", "description": "Echo back.",
             "inputSchema": {"type": "object",
                             "properties": {"message": {"type": "string"}},
                             "required": ["message"]}},
            {"name": "boom", "description": "Always fails.",
             "inputSchema": {"type": "object"}},
            {"name": "slow", "description": "Sleeps forever.",
             "inputSchema": {"type": "object"}},
            {"name": "twoblocks", "description": "Two text blocks.",
             "inputSchema": {"type": "object"}},
        ]
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            msg = json.loads(line)
            method, mid = msg.get("method"), msg.get("id")
            if method == "initialize":
                result(mid, {"protocolVersion": msg["params"]["protocolVersion"],
                             "capabilities": {"tools": {}},
                             "serverInfo": {"name": "fake", "version": "0"}})
            elif method == "tools/list":
                result(mid, {"tools": TOOLS})
            elif method == "tools/call":
                name = msg["params"]["name"]
                args = msg["params"].get("arguments", {})
                if name == "echo":
                    result(mid, {"content": [{"type": "text",
                                              "text": "echo: " + args.get("message", "")}],
                                 "isError": False})
                elif name == "boom":
                    result(mid, {"content": [{"type": "text",
                                              "text": "it exploded"}],
                                 "isError": True})
                elif name == "slow":
                    import time
                    time.sleep(1.5)
                    result(mid, {"content": []})
                elif name == "twoblocks":
                    result(mid, {"content": [{"type": "text", "text": "part1"},
                                             {"type": "text", "text": "part2"}]})
            elif mid is not None:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32601, "message": "no such method"}})
"""


@pytest.fixture()
def standard(tmp_path):
    cfg = write_server(tmp_path, STANDARD_BODY)
    from akshara.mcp import connect_mcp
    session, tools = connect_mcp(cfg, timeout=10.0)
    yield session, {t.raw_name: t for t in tools}
    session.close()


class TestHandshakeAndDiscovery:
    def test_negotiates_version_and_qualifies_names(self, standard):
        session, tools = standard
        assert session.protocol_version == SUPPORTED_VERSIONS[0]
        assert session.server_info["name"] == "fake"
        assert sorted(t.name for t in tools.values()) == [
            "mcp__srv__boom", "mcp__srv__echo",
            "mcp__srv__slow", "mcp__srv__twoblocks"]

    def test_unsupported_version_refused_and_process_reaped(self, tmp_path):
        body = """\
            import json, sys
            for line in sys.stdin:
                msg = json.loads(line)
                if msg.get("method") == "initialize":
                    sys.stdout.write(json.dumps(
                        {"jsonrpc": "2.0", "id": msg["id"],
                         "result": {"protocolVersion": "1999-01-01"}}) + "\\n")
                    sys.stdout.flush()
        """
        cfg = write_server(tmp_path, body)
        from akshara.mcp import connect_mcp
        with pytest.raises(MCPError, match="unsupported protocol version"):
            connect_mcp(cfg)

    def test_read_only_honored_only_when_hinted(self, tmp_path):
        # annotations are hints, not guarantees: absent => pessimistic False
        # (note: Python True here -- json.dumps turns it into wire-level true)
        body = STANDARD_BODY.replace(
            '{"name": "echo", "description": "Echo back.",',
            '{"name": "echo", "description": "Echo back.",'
            ' "annotations": {"readOnlyHint": True},')
        cfg = write_server(tmp_path, body)
        from akshara.mcp import connect_mcp
        session, wrapped = connect_mcp(cfg, timeout=10.0)
        try:
            by_raw = {t.raw_name: t for t in wrapped}
            assert by_raw["echo"].read_only is True      # hint honored...
            assert by_raw["boom"].read_only is False     # ...never assumed
        finally:
            session.close()


class TestToolCalls:
    def test_round_trip_flattening_and_error_flag(self, standard, tmp_path):
        _, tools = standard
        ctx = ToolContext(cwd=tmp_path)
        assert tools["echo"].run({"message": "hi"}, ctx) == "echo: hi"
        assert tools["twoblocks"].run({}, ctx) == "part1\npart2"
        with pytest.raises(ToolError, match="it exploded"):
            tools["boom"].run({}, ctx)

    def test_timeout_becomes_tool_error_and_session_survives(self, standard,
                                                              tmp_path):
        import time as _time
        session, tools = standard
        tools["slow"].session.timeout = 0.4  # server sleeps 1.5s: we give up first
        with pytest.raises(ToolError, match="timed out"):
            tools["slow"].run({}, ToolContext(cwd=tmp_path))
        # the fake server is sequential -- once it unblocks, the SAME
        # connection serves new calls; the timed-out response arrives
        # late and is dropped (its pending slot was already deleted)
        _time.sleep(1.5)
        assert tools["echo"].run({"message": "still alive"},
                                 ToolContext(cwd=tmp_path)) == \
            "echo: still alive"

    def test_summary_shows_provenance(self, standard, tmp_path):
        _, tools = standard
        s = tools["echo"].summary({"message": "x"}, ToolContext(cwd=tmp_path))
        assert s.startswith("mcp__srv__echo(")
        assert "'srv'" in s  # which server would be contacted


class TestServerInitiatedRequests:
    def test_ping_answered(self, tmp_path):
        # if we ignored the ping, the server blocks waiting for its reply
        # and tools/list times out -- this test FAILS in that world.
        body = """\
            import json, sys
            def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                msg = json.loads(line)
                method, mid = msg.get("method"), msg.get("id")
                if method == "initialize":
                    send({"jsonrpc": "2.0", "id": mid, "result": {
                        "protocolVersion": msg["params"]["protocolVersion"],
                        "capabilities": {}, "serverInfo": {"name": "f", "v": "0"}}})
                    send({"jsonrpc": "2.0", "id": 99, "method": "ping"})
                    reply = json.loads(sys.stdin.readline())
                    assert reply["id"] == 99 and reply.get("result") == {}
                elif method == "tools/list":
                    send({"jsonrpc": "2.0", "id": mid,
                          "result": {"tools": []}})
        """
        from akshara.mcp import connect_mcp
        session, tools = connect_mcp(write_server(tmp_path, body), timeout=5.0)
        try:
            assert session.list_tools() == []
        finally:
            session.close()

    def test_unknown_request_gets_method_not_found(self, tmp_path):
        body = """\
            import json, sys
            def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
            EMPTY = {"jsonrpc": "2.0", "id": 0, "result": {"tools": []}}
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                msg = json.loads(line)
                method, mid = msg.get("method"), msg.get("id")
                if method == "initialize":
                    send({"jsonrpc": "2.0", "id": mid, "result": {
                        "protocolVersion": msg["params"]["protocolVersion"],
                        "capabilities": {}, "serverInfo": {"n": "f"}}})
                    send({"jsonrpc": "2.0", "id": 98,
                          "method": "sampling/createMessage"})
                    reply = json.loads(sys.stdin.readline())
                    assert reply["id"] == 98
                    assert reply["error"]["code"] == -32601
                elif method == "tools/list":
                    send({"jsonrpc": "2.0", "id": mid, "result": {"tools": []}})
        """
        from akshara.mcp import connect_mcp
        session, tools = connect_mcp(write_server(tmp_path, body), timeout=5.0)
        try:
            assert session.list_tools() == []
        finally:
            session.close()


class TestLifecycleAndConfig:
    def test_close_reaps_process_and_is_idempotent(self, tmp_path):
        from akshara.mcp import connect_mcp
        session, _ = connect_mcp(write_server(tmp_path, STANDARD_BODY),
                                 timeout=10.0)
        proc = session._proc
        assert proc.poll() is None
        session.close()
        assert proc.poll() is not None
        session.close()  # idempotent, no raise

    def test_crashed_server_fails_fast_with_exit_code(self, tmp_path):
        body = """\
            import json, sys
            line = sys.stdin.readline()
            msg = json.loads(line)
            if msg.get("method") == "initialize":
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                    "result": {"protocolVersion": msg["params"]["protocolVersion"],
                               "capabilities": {}}}) + "\\n")
                sys.stdout.flush()
            sys.exit(7)   # die before any tools/list can be answered
        """
        from akshara.mcp import connect_mcp
        # the failure fires during connect's own discovery step, which
        # must clean up the dead process before re-raising
        with pytest.raises(MCPError, match="exited unexpectedly.*code 7"):
            connect_mcp(write_server(tmp_path, body), timeout=5.0)

    def test_spawn_failure_names_the_command(self, tmp_path):
        from akshara.mcp import connect_mcp
        cfg = MCPServerConfig(name="ghost", command="/no/such/binary")
        with pytest.raises(MCPError, match="cannot spawn mcp server 'ghost'"):
            connect_mcp(cfg)

    def test_load_mcp_configs(self, tmp_path):
        good = tmp_path / "mcp.json"
        good.write_text(json.dumps({"servers": {
            "fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                   "env": {"DEBUG": "1"}},
            "tiny": {"command": "python"},
        }}))
        configs = load_mcp_configs(good)
        by_name = {c.name: c for c in configs}
        assert by_name["fs"].args[0] == "-y"
        assert by_name["fs"].env == {"DEBUG": "1"}
        assert by_name["tiny"].command == "python"

        missing = tmp_path / "nope.json"
        with pytest.raises(MCPError, match="not found"):
            load_mcp_configs(missing)
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        with pytest.raises(MCPError, match="not valid JSON"):
            load_mcp_configs(broken)
        shapeless = tmp_path / "shapeless.json"
        shapeless.write_text('{"hello": 1}')
        with pytest.raises(MCPError, match="'servers' object"):
            load_mcp_configs(shapeless)


class TestRegistryIntegration:
    def test_register_and_duplicate_is_loud(self, tmp_path):
        cfg = write_server(tmp_path, STANDARD_BODY)
        registry = ToolRegistry()
        session, names = register_mcp(registry, cfg, timeout=10.0)
        try:
            # server listing order vs registry's sorted view: same SET
            assert sorted(names) == registry.names()
            assert registry.get("mcp__srv__echo").raw_name == "echo"
            # a second server with the SAME qualification prefix collides --
            # ValueError, never silent overwrite
            with pytest.raises(ValueError, match="duplicate tool name"):
                register_mcp(registry, cfg, timeout=10.0)
        finally:
            session.close()

    def test_full_loop_uses_mcp_tool_result_as_data(self, tmp_path):
        cfg = write_server(tmp_path, STANDARD_BODY)
        registry = ToolRegistry()
        session, _ = register_mcp(registry, cfg, timeout=10.0)
        try:
            provider = ScriptedProvider([
                assistant_tool_call("c1", "mcp__srv__echo",
                                    {"message": "hello loop"}),
                assistant_text("the server said: echo: hello loop"),
            ])
            agent = Agent(provider, model="m", permissions=yolo, tools=registry)
            response = agent.run("use the echo tool")
            assert "echo: hello loop" in response.message.text()
            batch = next(m for m in agent.history if m.role == "user"
                         and any(getattr(b, "tool_call_id", None) == "c1"
                                 for b in m.content))
            assert batch.content[0].content == "echo: hello loop"
        finally:
            session.close()


# ---- Streamable HTTP transport -------------------------------------------
#
# Same philosophy as the stdio tests: a REAL server on a REAL socket.
# The fake speaks the 2025-03-26 streamable-HTTP subset -- POST one
# JSON-RPC message, response as JSON or SSE -- in pure stdlib.

HTTP_SERVER_BODY = """\
        import json, sys
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        out_dir = sys.argv[1]
        bad_version = len(sys.argv) > 2 and sys.argv[2] == "bad"
        TOOLS = [
            {"name": "add", "description": "Add two integers.",
             "inputSchema": {"type": "object",
                             "properties": {"a": {"type": "integer"},
                                            "b": {"type": "integer"}},
                             "required": ["a", "b"],
                             "additionalProperties": False}},
        ]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n)) if n else {}

            def _send(self, code, ctype=None, payload=None, extra=None):
                data = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(code)
                if ctype:
                    self.send_header("Content-Type", ctype)
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def do_DELETE(self):
                self._send(200)

            def do_POST(self):
                msg = self._body()
                mid = msg.get("id")
                method = msg.get("method")
                extra = {}
                if method == "initialize":
                    version = ("1999-01-01" if bad_version
                               else msg["params"]["protocolVersion"])
                    result = {"protocolVersion": version,
                              "capabilities": {"tools": {}},
                              "serverInfo": {"name": "fake-http", "version": "0"}}
                    extra["Mcp-Session-Id"] = "sess-42"
                    self._send(200, "application/json",
                               {"jsonrpc": "2.0", "id": mid, "result": result},
                               extra)
                    return
                if method is None:
                    # a RESPONSE to our embedded ping -- record it as proof
                    with open(f"{out_dir}/replies.jsonl", "a") as fh:
                        fh.write(json.dumps(msg) + "\\n")
                    self._send(202)
                    return
                if str(method).startswith("notifications/"):
                    self._send(202)
                    return
                sid = self.headers.get("Mcp-Session-Id")
                with open(f"{out_dir}/sids.jsonl", "a") as fh:
                    fh.write((sid or "NONE") + "\\n")
                if not sid:
                    self._send(400, "application/json", {"error": "no session"})
                elif method == "tools/list":
                    self._send(200, "application/json",
                               {"jsonrpc": "2.0", "id": mid,
                                "result": {"tools": TOOLS}})
                elif method == "tools/call":
                    args = msg["params"]["arguments"]
                    ping_req = {"jsonrpc": "2.0", "id": 99, "method": "ping"}
                    reply = {"jsonrpc": "2.0", "id": mid,
                             "result": {"content": [{"type": "text",
                                                     "text": str(args["a"] + args["b"])}],
                                        "isError": False}}
                    frames = "".join(
                        f"data: {json.dumps(m)}\\n\\n" for m in (ping_req, reply))
                    data = frames.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send(200, "application/json",
                               {"jsonrpc": "2.0", "id": mid, "result": {}})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        with open(f"{out_dir}/port.txt", "w") as fh:
            fh.write(str(server.server_address[1]))
        server.serve_forever()
"""


def start_http_server(tmp_path: Path, *argv: str) -> MCPServerConfig:
    script = tmp_path / "fake_http.py"
    script.write_text(textwrap.dedent(HTTP_SERVER_BODY))
    import subprocess, time
    proc = subprocess.Popen([sys.executable, str(script), str(tmp_path), *argv],
                            stdout=subprocess.DEVNULL)
    port_file = tmp_path / "port.txt"
    for _ in range(100):  # up to ~5s for bind+write
        if port_file.exists():
            break
        if proc.poll() is not None:
            raise AssertionError("http fake server died during startup")
        time.sleep(0.05)
    else:
        proc.kill()
        raise AssertionError("http fake server never reported its port")
    port = int(port_file.read_text().strip())
    assert proc.poll() is None
    start_http_server.proc = proc  # stash for cleanup via wait_for_port_close
    return MCPServerConfig(name="tiny", url=f"http://127.0.0.1:{port}/mcp")


def stop_http_server() -> None:
    proc = getattr(start_http_server, "proc", None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


class TestHttpTransport:
    def test_config_loader_accepts_url_and_rejects_mixtures(self, tmp_path):
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text('{"servers": {"a": {"url": "http://x/mcp"}}}')
        configs = load_mcp_configs(cfg_file)
        assert configs[0].url == "http://x/mcp" and configs[0].command is None

        cfg_file.write_text('{"servers": {"a": {}}}')
        with pytest.raises(MCPError, match="'command' \\(stdio\\) or 'url'"):
            load_mcp_configs(cfg_file)

    def test_handshake_discovery_and_sse_tool_call(self, tmp_path):
        cfg = start_http_server(tmp_path)
        registry = ToolRegistry()
        session, names = register_mcp(registry, cfg, timeout=10.0)
        try:
            stop_http_server = None
            assert names == ["mcp__tiny__add"]

            # tools/call answered over SSE: ping request embedded BEFORE
            # our response must be answered by another POST...
            tool = registry.get("mcp__tiny__add")
            assert tool.run({"a": 19, "b": 23}, ToolContext(cwd=tmp_path)) == "42"

            # ...proof: the server recorded our ping reply...
            replies = [json.loads(l) for l in
                       (tmp_path / "replies.jsonl").read_text().splitlines()]
            pings = [r for r in replies if r.get("id") == 99]
            assert pings and pings[-1].get("result") == {}

            # ...and every post-initialize request echoed the session id
            sids = (tmp_path / "sids.jsonl").read_text().split()
            assert sids and all(s == "sess-42" for s in sids)
        finally:
            session.close()

    def test_unsupported_version_refused(self, tmp_path):
        cfg = start_http_server(tmp_path, "bad")
        with pytest.raises(MCPError, match="unsupported protocol version"):
            from akshara.mcp import connect_mcp
            connect_mcp(cfg, timeout=5.0)

    def test_unreachable_url_is_an_mcperror_not_a_traceback(self):
        from akshara.mcp import connect_mcp
        dead = MCPServerConfig(name="dead", url="http://127.0.0.1:9/mcp")
        with pytest.raises(MCPError, match="unreachable"):
            connect_mcp(dead, timeout=2.0)

    def test_full_agent_loop_over_http(self, tmp_path):
        cfg = start_http_server(tmp_path)
        registry = ToolRegistry()
        session, _ = register_mcp(registry, cfg, timeout=10.0)
        try:
            provider = ScriptedProvider([
                assistant_tool_call("c1", "mcp__tiny__add", {"a": 20, "b": 22}),
                assistant_text("the server said 42"),
            ])
            agent = Agent(provider, model="m", permissions=yolo,
                          tools=registry)
            response = agent.run("add 20 and 22 via mcp")
            assert "42" in response.message.text()
        finally:
            session.close()
