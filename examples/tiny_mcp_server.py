"""A minimal MCP server in pure stdlib -- both sides of BOTH transports.

The book's suggested exercise ("write a minimal echo MCP server to
learn both sides of the protocol"), kept dependency-free so it doubles
as this repo's live test fixture. One protocol core, two transports --
mirroring the CLIENT side, where both sessions expose the identical
surface and everything downstream is transport-blind.

stdio (default) -- read newline-delimited JSON-RPC from stdin, write
responses to stdout, never let logs near stdout (stderr only):

    echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"me","version":"0"}}}' | python examples/tiny_mcp_server.py

    {"servers": {"tiny": {"command": "python",
                          "args": ["examples/tiny_mcp_server.py"]}}}
    akshara --mcp-config that-file.json

Streamable HTTP (--http) -- the SAME dispatcher behind ThreadingHTTPServer.
tools/call answers as text/event-stream with an EMBEDDED server->client
ping riding ahead of the result, so the client's SSE parser and its
answer-by-POST politeness get exercised against a real socket:

    python examples/tiny_mcp_server.py --http            # prints its URL
    {"servers": {"tiny": {"url": "http://127.0.0.1:PORT/mcp"}}}
    akshara --mcp-config http-file.json
"""

from __future__ import annotations

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "echo",
        "description": "Repeat the message back, prefixed with 'echo:'. "
                       "Useful for proving an MCP round trip works.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
]


class RPCError(Exception):
    """A JSON-RPC error payload waiting to be sent (code + message)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def protocol_core(method: str, params: dict) -> dict:
    """Dispatch one request -> RESULT payload. Raises RPCError on failure.

    The single source of protocol truth; stdio and HTTP both call this,
    which is what keeps the two transports provably equivalent.
    """
    if method == "initialize":
        requested = params.get("protocolVersion", "")
        # negotiate: echo the client's version if we speak it, otherwise
        # answer with our own newest and let them decide
        version = requested if requested == PROTOCOL_VERSION else PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tiny", "version": "0.2.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params["name"]
        args = params.get("arguments", {})
        if name == "echo":
            return {"content": [{"type": "text",
                                 "text": f"echo: {args.get('message', '')}"}],
                    "isError": False}
        if name == "add":
            try:
                total = int(args["a"]) + int(args["b"])
            except (KeyError, TypeError, ValueError):
                return {"content": [{"type": "text",
                                     "text": f"bad arguments: {args!r}"}],
                        "isError": True}
            return {"content": [{"type": "text", "text": str(total)}],
                    "isError": False}
        # a tool that doesn't exist is a TOOL error (isError), not an
        # RPC error -- the method reached us fine
        return {"content": [{"type": "text",
                             "text": f"no such tool: {name}"}],
                "isError": True}
    raise RPCError(-32601, f"no method {method}")


# ---- stdio transport -----------------------------------------------------

def send(message: dict) -> None:
    """One JSON-RPC message per line; flush immediately."""
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def run_stdio() -> None:
    print("tiny mcp server starting", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"unparseable line: {line[:100]}", file=sys.stderr)
            continue
        msg_id = msg.get("id")   # None => notification: no reply expected
        method = msg.get("method")
        if method is None or msg_id is None:
            continue             # our responses/notifications don't occur here
        try:
            payload = protocol_core(method, msg.get("params") or {})
            send({"jsonrpc": "2.0", "id": msg_id, "result": payload})
        except RPCError as exc:
            # unknown REQUEST must still be answered or the client hangs
            send({"jsonrpc": "2.0", "id": msg_id,
                  "error": {"code": exc.code, "message": exc.message}})


# ---- streamable HTTP transport --------------------------------------------

# ids for OUR server->client requests (embedded in response streams);
# offset high enough they can never collide with a client request id
_server_ids = count(start=10_000)


def sse_frame(message: dict) -> bytes:
    return (f"event: message\ndata: "
            f"{json.dumps(message, separators=(',', ':'))}\n\n").encode()


class MCPHandler(BaseHTTPRequestHandler):
    """One endpoint (/mcp), POST-only, sessions via Mcp-Session-Id."""

    session_id: str | None = None  # class attr: one session for the whole demo

    def log_message(self, fmt: str, *args) -> None:  # stderr, never stdout
        print(f"[tiny-http] {fmt % args}", file=sys.stderr)

    def _reply_json(self, code: int, message: dict | None = None,
                    extra_headers: dict | None = None) -> None:
        body = b"" if message is None else json.dumps(
            message, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _reply_sse(self, messages: list[dict]) -> None:
        body = b"".join(sse_frame(m) for m in messages)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        try:
            msg = json.loads(self.rfile.read(int(self.headers.get(
                "Content-Length", 0))))
        except json.JSONDecodeError:
            self._reply_json(400, {"error": "unparseable JSON"})
            return

        # a RESPONSE arriving by POST answers one of OUR embedded requests;
        # record it so a human can see politeness working end-to-end
        if "method" not in msg and msg.get("id") is not None:
            print(f"[tiny-http] got reply to our request "
                  f"id={msg['id']}: {msg.get('result', msg.get('error'))}",
                  file=sys.stderr)
            self._reply_json(202)
            return

        method, msg_id = msg.get("method"), msg.get("id")
        if msg_id is None:                      # notification: accepted, silent
            self._reply_json(202)
            return

        # session discipline: initialize ISSUES the header, everyone else
        # must ECHO it back (spec: stale/unknown session may 404)
        if method == "initialize":
            type(self).session_id = uuid.uuid4().hex
            try:
                payload = protocol_core(method, msg.get("params") or {})
            except RPCError as exc:
                self._reply_json(200, {"jsonrpc": "2.0", "id": msg_id,
                                       "error": {"code": exc.code,
                                                 "message": exc.message}})
                return
            self._reply_json(
                200, {"jsonrpc": "2.0", "id": msg_id, "result": payload},
                extra_headers={"Mcp-Session-Id": type(self).session_id})
            return
        if self.session_id is not None and \
                self.headers.get("Mcp-Session-Id") != self.session_id:
            self._reply_json(404, {"error": "unknown or expired session"})
            return

        try:
            payload = protocol_core(method, msg.get("params") or {})
        except RPCError as exc:
            self._reply_json(200, {"jsonrpc": "2.0", "id": msg_id,
                                   "error": {"code": exc.code,
                                             "message": exc.message}})
            return
        if method == "tools/call":
            # SSE response WITH an embedded server->client ping: the
            # polite-client rule exercised over a real socket
            ping_id = next(_server_ids)
            self._reply_sse([
                {"jsonrpc": "2.0", "id": ping_id, "method": "ping"},
                {"jsonrpc": "2.0", "id": msg_id, "result": payload},
            ])
        else:
            self._reply_json(200,
                             {"jsonrpc": "2.0", "id": msg_id, "result": payload})

    def do_GET(self) -> None:  # noqa: N802
        # honest non-feature: no standalone GET server->client stream
        self._reply_json(405, {"error": "GET streams not supported"})

    def do_DELETE(self) -> None:  # noqa: N802
        type(self).session_id = None
        self.send_response(204)
        self.end_headers()


def run_http(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), MCPHandler)
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    print(f"tiny mcp http server ready at {url}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        run_http(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        run_stdio()
