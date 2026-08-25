"""MCP client: Model Context Protocol, JSON-RPC 2.0 by hand, two transports.

The book (ch13) uses the official ``mcp`` SDK and calls a hand-written
client "300 lines of undifferentiated code". This project's whole point
is that no wire is undifferentiated -- the SSE parser is hand-rolled,
so both transports get the same treatment. The protocol reduces to
four exchanges:

1. client -> server: ``initialize`` request (version handshake)
2. server -> client: initialize RESPONSE (negotiated version+capabilities)
3. client -> server: ``notifications/initialized`` notification
4. ``tools/list`` / ``tools/call`` requests at will

Transports (``MCPServerConfig.command`` vs ``MCPServerConfig.url`` picks):

* **stdio** (`MCPSession`) -- one JSON message PER LINE on the subprocess's
  stdin/stdout; stderr belongs to the SERVER's logs and passes through.
* **Streamable HTTP** (`MCPHttpSession`, spec 2025-03-26+) -- POST each
  message to one endpoint; the response body is plain JSON or an SSE
  stream parsed with the SAME `providers/sse.py` the adapters use;
  session state rides in the ``Mcp-Session-Id`` header instead of a
  process.

Transport rules that make it small:

* one JSON-RPC message PER LINE on the subprocess's stdin/stdout --
  newline-delimited, never embedded newlines (unlike LSP's
  Content-Length framing);
* stderr belongs to the SERVER's logs and passes straight through to
  ours -- piping it would risk a full buffer deadlocking the protocol;
* requests carry an integer id; responses quote it back; notifications
  (no id, no reply expected) are dropped except for logging;
* servers may send REQUESTS of their own (``ping``, ``roots/list``).
  Ignoring them wedges polite servers, so the reader thread answers:
  ping -> empty result, anything unknown -> error -32601.

Threading model: one daemon reader thread owns stdout and dispatches
lines -- responses into per-id queues, server-initiated requests get
replies written under a lock (two threads share stdin). A caller waits
on its queue with a timeout; EOF pushes a sentinel into every pending
queue so a crashed server fails fast instead of burning everyone's
timeout.

Security stance (the chapter's loudest lesson): MCP is an INTEGRATION
standard, not a security boundary. Servers aggregate credentials,
tool OUTPUT is untrusted text that can carry prompt injection, and
malicious servers exist in the wild. Our mitigations are structural,
not hopeful: qualified names (``mcp__<server>__<tool>``) make
provenance visible to humans AND permission rules; tools are
permission-gated like any other (read_only only when the server's
``readOnlyHint`` annotation claims it -- hints are honored but never
assumed); and nothing here bypasses the permission gate a sub-agent
inherits.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, ClassVar

import httpx

from akshara.errors import ToolError
from akshara.providers.sse import iter_sse_lines, parse_events
from akshara.tools.base import Tool, ToolContext, ToolRegistry

# Versions this client can speak, newest first. We REQUEST the newest;
# the server answers with what IT supports -- if that answer isn't in
# our list we refuse rather than guess at incompatible semantics.
SUPPORTED_VERSIONS: tuple[str, ...] = (
    "2025-06-18", "2025-03-26", "2024-11-05",
)
CLIENT_INFO = {"name": "akshara", "version": "0.1.0"}


class MCPError(Exception):
    """Protocol/transport failure inside an MCP session.

    Deliberately NOT ToolError/ProviderError: session SETUP failures are
    fatal for that server's tools (the CLI warns and skips), while
    call-time failures are converted to ToolError at the tool wrapper so
    the model sees them as recoverable data.
    """


@dataclass(slots=True)
class MCPServerConfig:
    """How to reach one MCP server.

    Exactly one transport: ``command`` spawns a subprocess spoken to over
    newline-delimited stdio; ``url`` points at an HTTP endpoint spoken to
    with Streamable HTTP (POST JSON-RPC, response as JSON or SSE).
    """

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None


@dataclass(slots=True)
class MCPToolInfo:
    """One entry of a ``tools/list`` result."""

    name: str
    description: str
    input_schema: dict[str, Any]
    read_only_hint: bool = False


def _configs_from_servers_dict(servers: Any, source: str) -> list[MCPServerConfig]:
    """Shared validation for every way a servers-dict arrives: a
    --mcp-config file, pasted panel JSON, the remembered-sessions file.
    ``source`` names the origin in error messages."""
    if not isinstance(servers, dict):
        raise MCPError(
            f"{source} must have a 'servers' object mapping "
            "name -> {command,args,env} (stdio) or {url} (http)"
        )
    configs: list[MCPServerConfig] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            raise MCPError(f"{source}: server {name!r} must be an object")
        command = spec.get("command")
        url = spec.get("url")
        if command is None and url is None:
            raise MCPError(
                f"{source}: server {name!r} needs a string "
                "'command' (stdio) or 'url' (http)"
            )
        if command is not None and not isinstance(command, str):
            raise MCPError(f"{source}: server {name!r} 'command' must be "
                           "a string")
        if url is not None and not isinstance(url, str):
            raise MCPError(f"{source}: server {name!r} 'url' must be a string")
        args = spec.get("args", [])
        env = spec.get("env")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise MCPError(f"{source}: server {name!r} 'args' must be strings")
        configs.append(MCPServerConfig(
            name=name, command=command, args=args,
            env={str(k): str(v) for k, v in env.items()} if env else None,
            url=url,
        ))
    return configs


def load_mcp_configs(path: str | Path) -> list[MCPServerConfig]:
    """Parse an mcp config file: ``{"servers": {"name": {command,args,env}}}``."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise MCPError(f"mcp config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise MCPError(f"mcp config {path} is not valid JSON: {exc}") from None
    return _configs_from_servers_dict(raw.get("servers"),
                                      f"mcp config {path}")


def parse_mcp_text(text: str) -> list[MCPServerConfig]:
    """The web panel's paste-JSON mode: identical syntax to a config
    file, arriving as a string instead of a path."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MCPError(f"pasted config is not valid JSON: {exc}") from None
    return _configs_from_servers_dict(raw.get("servers"), "pasted config")


def _require_supported_version(negotiated: str, server_name: str) -> str:
    """Shared initialize-result validation -- both transports negotiate
    identically; only the plumbing differs."""
    if negotiated not in SUPPORTED_VERSIONS:
        raise MCPError(
            f"mcp server {server_name!r} negotiated unsupported "
            f"protocol version {negotiated!r} "
            f"(we speak: {', '.join(SUPPORTED_VERSIONS)})"
        )
    return negotiated


def _parse_tools(result: dict[str, Any]) -> list[MCPToolInfo]:
    tools: list[MCPToolInfo] = []
    for raw in result.get("tools", []):
        annotations = raw.get("annotations") or {}
        tools.append(MCPToolInfo(
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            input_schema=raw.get("inputSchema") or {"type": "object"},
            read_only_hint=bool(annotations.get("readOnlyHint")),
        ))
    return tools


def _flatten_tool_result(result: dict[str, Any]) -> tuple[str, bool]:
    parts: list[str] = []
    omitted = 0
    for block in result.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            omitted += 1
    text = "\n".join(p for p in parts if p != "" or len(parts) == 1)
    if omitted:
        text += f"\n[{omitted} non-text content block(s) omitted]"
    return text.strip(), bool(result.get("isError"))


def _reply_for_server_request(method: str) -> tuple[bool, dict[str, Any]]:
    """How we answer server->client REQUESTS (both transports must,
    or polite servers wedge): ping -> empty result, roots -> empty list,
    anything else -> method-not-found. Returns (is_result, payload)."""
    if method == "ping":
        return True, {}
    if method == "roots/list":
        return True, {"roots": []}
    return False, {"code": -32601, "message": f"method not found: {method}"}


class MCPSession:
    """One spawned server process, spoken to over newline-delimited JSON."""

    def __init__(self, config: MCPServerConfig, *, timeout: float = 30.0) -> None:
        self.config = config
        self.timeout = timeout
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self._proc: subprocess.Popen | None = None
        self._ids = count(1)
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._wlock = threading.Lock()  # reader thread replies + main requests share stdin
        self._reader: threading.Thread | None = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the subprocess and run the initialize handshake."""
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)
        try:
            self._proc = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                # stderr inherits: server logs flow to ours, and we can't
                # deadlock on a pipe nobody drains
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            raise MCPError(
                f"cannot spawn mcp server {self.config.name!r} "
                f"({self.config.command}): {exc}"
            ) from None

        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{self.config.name}", daemon=True)
        self._reader.start()

        try:
            result = self._request("initialize", {
                "protocolVersion": SUPPORTED_VERSIONS[0],
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            })
        except MCPError:
            self.close()
            raise
        self.protocol_version = result.get("protocolVersion", "")
        try:
            self.protocol_version = _require_supported_version(
                self.protocol_version, self.config.name)
        except MCPError:
            self.close()
            raise
        self.server_info = result.get("serverInfo", {}) or {}
        # lifecycle step 3: notification -- no id, no response ever comes
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def close(self) -> None:
        """Shut the server down; escalate SIGTERM -> SIGKILL. No zombies."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()  # EOF is the polite shutdown signal
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._reader is not None:
            self._reader.join(timeout=1)

    def healthy(self) -> bool:
        """Cheap liveness probe for status displays: the child process
        exists and has not exited. Never does protocol IO."""
        return self._proc is not None and self._proc.poll() is None

    # ---- public protocol operations ----------------------------------------

    def list_tools(self) -> list[MCPToolInfo]:
        return _parse_tools(self._request("tools/list", {}))

    def call_tool(self, name: str, arguments: dict[str, Any],
                  *, timeout: float | None = None) -> tuple[str, bool]:
        """Invoke a tool; returns ``(text, is_error)``.

        Content blocks flatten to text (images/resources are noted, not
        decoded -- our Tool contract is strings). ``isError`` means the
        TOOL ran and failed, which the wrapper turns into a ToolError so
        errors-as-data holds across the process boundary too.
        """
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments},
                               timeout=timeout)
        return _flatten_tool_result(result)

    # ---- wire plumbing ------------------------------------------------------

    def _request(self, method: str, params: dict[str, Any],
                 timeout: float | None = None) -> dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise MCPError(f"mcp server {self.config.name!r} is not running")
        msg_id = next(self._ids)
        box: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._pending[msg_id] = box  # registered BEFORE sending: no lost responses
        self._send({"jsonrpc": "2.0", "id": msg_id,
                    "method": method, "params": params})
        try:
            response = box.get(timeout=self.timeout if timeout is None else timeout)
        except queue.Empty:
            del self._pending[msg_id]
            raise MCPError(
                f"mcp server {self.config.name!r} timed out after "
                f"{timeout or self.timeout}s during {method!r}"
            ) from None
        del self._pending[msg_id]
        if response.get("__eof__"):
            raise MCPError(
                f"mcp server {self.config.name!r} exited unexpectedly "
                f"(code {self._proc.poll() if self._proc else '?'}) "
                f"during {method!r}"
            )
        if "error" in response:
            err = response["error"]
            raise MCPError(
                f"mcp server {self.config.name!r} rejected {method!r}: "
                f"{err.get('code')} {err.get('message')}"
            )
        return response.get("result", {})

    def _send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPError(f"mcp server {self.config.name!r} is not running")
        line = json.dumps(message, separators=(",", ":"))
        assert "\n" not in line  # transport invariant: one message per line
        with self._wlock:
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError):
                raise MCPError(
                    f"mcp server {self.config.name!r} stopped accepting input"
                ) from None

    # ---- the reader thread ---------------------------------------------------

    def _read_loop(self) -> None:
        """Own stdout forever: dispatch responses, answer server requests."""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[akshara:mcp:{self.config.name}] unparseable "
                          f"line: {line[:200]}", file=sys.stderr)
                    continue
                self._dispatch(message)
        finally:
            # EOF or crash: wake every waiter with the sentinel so they
            # fail fast instead of each burning the full timeout
            for box in list(self._pending.values()):
                box.put({"__eof__": True})

    def _dispatch(self, message: dict[str, Any]) -> None:
        msg_id = message.get("id")
        method = message.get("method")
        if method is None:  # a response to OUR request
            box = self._pending.get(msg_id)
            if box is not None:
                box.put(message)
            return
        if msg_id is None:  # notification: logged, never answered
            return
        # server -> client REQUEST: ignoring these wedges polite servers
        if method == "ping":
            self._reply(msg_id, {})
        elif method == "roots/list":
            self._reply(msg_id, {"roots": []})
        else:
            self._reply_error(msg_id, -32601, f"method not found: {method}")

    def _reply(self, msg_id: int, result: dict[str, Any]) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})
        except MCPError:
            pass  # server died mid-handshake; the EOF sentinel handles callers

    def _reply_error(self, msg_id: int, code: int, message: str) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": msg_id,
                        "error": {"code": code, "message": message}})
        except MCPError:
            pass


class MCPHttpSession:
    """MCP over Streamable HTTP (spec 2025-03-26+): ONE endpoint, POST
    JSON-RPC at it, response comes back as application/json or as a
    text/event-stream -- the hand-rolled SSE parser
    (``providers/sse.py``)
    reads that stream unchanged, which is the whole dividend of writing
    framing once, provider-agnostically.

    Differences from stdio worth noticing:

    * no reader thread: HTTP responses arrive synchronously with the
      request that caused them, so there is nothing to pump in the
      background;
    * the server hands us an ``Mcp-Session-Id`` header at initialize;
      every later request echoes it back -- session state lives in a
      HEADER now instead of a process lifetime;
    * a server-initiated request can ride INSIDE our response's SSE
      stream; we answer it with another POST (politeness rule intact).

    Not implemented, honestly: the standalone GET server->client stream
    (a client MAY skip it) and batching multiple RPCs per POST.
    """

    def __init__(self, config: MCPServerConfig, *, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        if config.url is None:
            raise MCPError(f"mcp server {config.name!r}: http transport "
                           "needs a 'url'")
        self.config = config
        self.timeout = timeout
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self._session_id: str | None = None
        self._closed = False
        self._ids = count(1)
        self._client = httpx.Client(timeout=timeout, transport=transport)

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Initialize handshake over POST; capture the session header."""
        try:
            result = self._rpc("initialize", {
                "protocolVersion": SUPPORTED_VERSIONS[0],
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            })
        except MCPError:
            self.close()
            raise
        self.protocol_version = _require_supported_version(
            result.get("protocolVersion", ""), self.config.name)
        self.server_info = result.get("serverInfo", {}) or {}
        self._notify({"jsonrpc": "2.0",
                      "method": "notifications/initialized"})

    def close(self) -> None:
        """Polite DELETE of the session; servers may ignore it."""
        if self._session_id is not None:
            try:
                self._client.delete(self.config.url, headers=self._headers())
            except httpx.TransportError:
                pass  # best effort by definition
        self._client.close()
        self._closed = True

    def healthy(self) -> bool:
        """Same contract as MCPSession.healthy: cheap, no protocol IO.
        An HTTP session has no process to poll, so 'closed by us' is the
        only state we can honestly report on."""
        return not self._closed

    # ---- protocol operations (same surface as MCPSession) ------------------

    def list_tools(self) -> list[MCPToolInfo]:
        return _parse_tools(self._rpc("tools/list", {}))

    def call_tool(self, name: str, arguments: dict[str, Any],
                  *, timeout: float | None = None) -> tuple[str, bool]:
        return _flatten_tool_result(
            self._rpc("tools/call", {"name": name, "arguments": arguments},
                      timeout=timeout))

    # ---- plumbing ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, text/event-stream",
                   "Content-Type": "application/json"}
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _post(self, message: dict[str, Any],
              timeout: float | None = None) -> httpx.Response:
        try:
            response = self._client.post(self.config.url, json=message,
                                         headers=self._headers(),
                                         timeout=self.timeout
                                         if timeout is None else timeout)
        except httpx.TransportError as exc:
            raise MCPError(f"mcp server {self.config.name!r} unreachable: "
                           f"{exc}") from None
        session_id = response.headers.get("mcp-session-id")
        if session_id and self._session_id is None:
            self._session_id = session_id  # issued at initialize
        return response

    def _rpc(self, method: str, params: dict[str, Any],
             timeout: float | None = None) -> dict[str, Any]:
        msg_id = next(self._ids)
        response = self._post({"jsonrpc": "2.0", "id": msg_id,
                               "method": method, "params": params},
                              timeout=timeout)
        if response.status_code != 200:
            raise MCPError(
                f"mcp server {self.config.name!r} rejected {method!r}: "
                f"HTTP {response.status_code} "
                f"{response.text[:200]!r}")
        reply = None
        for message in self._messages_in_response(response):
            if message.get("id") == msg_id and "method" not in message:
                reply = message  # OURS; keep scanning for embedded requests
            elif message.get("id") is not None:
                self._answer_server_request(message)
        if reply is None:
            raise MCPError(f"mcp server {self.config.name!r} returned no "
                           f"response to {method!r}")
        if "error" in reply:
            err = reply["error"]
            raise MCPError(
                f"mcp server {self.config.name!r} rejected {method!r}: "
                f"{err.get('code')} {err.get('message')}"
            )
        return reply.get("result", {})

    def _notify(self, message: dict[str, Any]) -> None:
        response = self._post(message)
        if response.status_code not in (200, 202, 204):
            raise MCPError(
                f"mcp server {self.config.name!r} refused notification: "
                f"HTTP {response.status_code}")

    def _messages_in_response(self, response: httpx.Response) -> Iterator[dict[str, Any]]:
        """Yield JSON-RPC messages from a JSON body or an SSE stream."""
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            yield json.loads(response.text)
            return
        for _, data in parse_events(iter_sse_lines([response.content])):
            line = data.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[akshara:mcp:{self.config.name}] unparseable SSE "
                      f"data: {line[:200]}", file=sys.stderr)

    def _answer_server_request(self, message: dict[str, Any]) -> None:
        ok, value = _reply_for_server_request(message.get("method", ""))
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": message.get("id")}
        if ok:
            payload["result"] = value
        else:
            payload["error"] = value
        try:
            self._post(payload)
        except MCPError:
            pass  # server died mid-exchange; our own error path reports it


def connect_mcp(config: MCPServerConfig, *,
                timeout: float = 30.0) -> tuple[MCPSession | MCPHttpSession,
                                                list[Tool]]:
    """Start a session, discover its tools, wrap them as registry-ready Tools.

    Transport choice falls out of the config: ``url`` set means Streamable
    HTTP, otherwise spawn ``command`` over stdio. Both sessions expose the
    identical surface, so everything downstream is transport-blind."""
    session: MCPSession | MCPHttpSession
    session = MCPHttpSession(config, timeout=timeout) if config.url \
        else MCPSession(config, timeout=timeout)
    session.start()
    try:
        infos = session.list_tools()
    except MCPError:
        session.close()
        raise
    return session, [MCPToolWrapper(session, info) for info in infos]


def register_mcp(registry: ToolRegistry, config: MCPServerConfig, *,
                 timeout: float = 30.0) -> tuple[MCPSession, list[str]]:
    """Connect one server and register its (qualified-name) tools.

    Returns the live session -- the CALLER owns closing it.
    """
    session, tools = connect_mcp(config, timeout=timeout)
    registered: list[str] = []
    for tool in tools:
        registry.register(tool)  # duplicate names raise ValueError: loud, not silent
        registered.append(tool.name)
    return session, registered


class MCPToolWrapper(Tool):
    """An MCP server's tool wearing our Tool interface.

    Instance attributes shadow the ClassVars -- the ABC reads
    ``self.description``, so dynamic metadata just works. The name is
    qualified ``mcp__<server>__<tool>`` (Claude Code's convention):
    collisions between servers become impossible, and provenance is
    visible in transcripts and permission rules alike.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict] = {}

    def __init__(self, session: MCPSession, info: MCPToolInfo) -> None:
        self.session = session
        self.raw_name = info.name
        self.name = f"mcp__{session.config.name}__{info.name}"
        self.description = info.description or "(no description provided)"
        self.parameters = info.input_schema
        # pessimistic default: honor readOnlyHint when present, assume
        # side effects otherwise -- annotations are hints, not guarantees
        self.read_only = info.read_only_hint

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return (f"{self.name}({json.dumps(args, default=str)}) "
                f"-- mcp server '{self.session.config.name}'")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        try:
            text, is_error = self.session.call_tool(self.raw_name, args)
        except MCPError as exc:
            raise ToolError(str(exc)) from None
        if is_error:
            # the tool RAN and reported failure -- same convention as our
            # built-ins: ToolError becomes an is_error result the model
            # can read and retry differently
            raise ToolError(text or "mcp tool reported failure")
        return text


# ---------------------------------------------------------------------------
# Remembered servers + the runtime manager (add/remove mid-session)
# ---------------------------------------------------------------------------


def remembered_path(cwd: Path | None = None) -> Path:
    """Where mid-session-added servers persist across launches."""
    return (cwd or Path.cwd()) / ".akshara" / "mcp.json"


def load_remembered(path: Path) -> list[MCPServerConfig]:
    """Servers saved by earlier sessions' 'remember' checkboxes. Missing
    file is the normal first-run case -> empty list; a CORRUPT file raises
    MCPError -- silent data loss would be worse than a loud startup."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MCPError(f"remembered mcp config {path} is not valid JSON: "
                       f"{exc}") from None
    return _configs_from_servers_dict(
        raw.get("servers") if isinstance(raw, dict) else None,
        f"remembered mcp config {path}")


def remember_server(config: MCPServerConfig, path: Path) -> None:
    """Upsert one server into the remembered file (atomic tmp+rename --
    the same write discipline as memory.json)."""
    data: dict[str, Any] = {"servers": {}}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and isinstance(loaded.get("servers"),
                                                       dict):
                data = loaded
        except json.JSONDecodeError:
            pass  # corrupt file gets replaced rather than blocking adds
    spec: dict[str, Any] = {}
    if config.command is not None:
        spec["command"] = config.command
        if config.args:
            spec["args"] = list(config.args)
    if config.url is not None:
        spec["url"] = config.url
    if config.env:
        spec["env"] = dict(config.env)
    data["servers"][config.name] = spec
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def forget_server(name: str, path: Path) -> bool:
    """Drop one server's entry. False means it was never remembered."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or name not in servers:
        return False
    del servers[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return True


class MCPManager:
    """Owns the LIVE MCP sessions of one registry -- what makes servers a
    runtime-manageable resource instead of startup-only wiring.

    Startup still works exactly as before (--mcp-config files); the
    manager just becomes that loop's bookkeeper so the web panel and the
    REPL can add, remove, and soft-toggle servers later:

    * ``connect`` registers ``mcp__<server>__<tool>`` wrappers and, when
      the agent runs per-turn selection, rebuilds the catalog so new
      tools are reachable without a restart;
    * ``disconnect`` closes the transport (stdio child reaped by close())
      and unregisters exactly that server's tools;
    * ``set_enabled`` is the SOFT switch: tools stay registered, calls
      fail as readable data until re-enabled -- identical semantics to
      the /api/tools toggles, one server at a time.
    """

    def __init__(self, registry: Any, *, agent: Any = None,
                 memory_path: Path | None = None,
                 connector: Any = connect_mcp) -> None:
        self.registry = registry
        self.agent = agent  # optional; only used to refresh the catalog
        self.memory_path = memory_path  # None = this session never persists
        self._connector = connector  # injectable: tests fake the transport
        self.sessions: dict[str, Any] = {}
        self.tool_names: dict[str, list[str]] = {}
        self.pinned: set[str] = set()  # names with an entry in memory_path

    # ---- lifecycle ---------------------------------------------------------

    def connect(self, config: MCPServerConfig, *, timeout: float = 30.0,
                remember: bool = False) -> list[str]:
        """Start one server and register its tools. Connection failures
        raise MCPError AFTER cleanup -- half-registered sessions must not
        linger."""
        if config.name in self.sessions:
            raise MCPError(f"mcp server {config.name!r} is already connected")
        session, tools = self._connector(config, timeout=timeout)
        names: list[str] = []
        try:
            for tool in tools:
                self.registry.register(tool)  # dup names raise ValueError
                names.append(tool.name)
        except Exception:
            # roll back EVERYTHING from this attempt -- a half-registered
            # server whose transport is closed must not leave zombie
            # entries no /mcp command can ever remove again
            for done in names:
                self.registry.unregister(done)
            session.close()
            raise
        self.sessions[config.name] = session
        self.tool_names[config.name] = names
        if remember and self.memory_path is not None:
            remember_server(config, self.memory_path)
        if self.memory_path is not None and \
                any(c.name == config.name for c in load_remembered(
                    self.memory_path)):
            self.pinned.add(config.name)
        self._refresh_catalog()
        return names

    def disconnect(self, name: str) -> int:
        """Close one server's connection and pull its tools. Also drops
        its remembered entry -- removing a server you had saved means
        gone-gone, not 'until next launch'. Returns the tool count."""
        session = self.sessions.pop(name, None)
        if session is None:
            raise MCPError(f"no mcp server named {name!r} is connected")
        session.close()
        removed = self.tool_names.pop(name, [])
        for tool_name in removed:
            self.registry.unregister(tool_name)
        self.pinned.discard(name)
        if self.memory_path is not None:
            forget_server(name, self.memory_path)
        self._refresh_catalog()
        return len(removed)

    def set_enabled(self, name: str, enabled: bool) -> int:
        """Soft toggle every tool of one server (process stays warm).
        Unknown servers are an MCPError; callers translate to their own
        error shapes (REPL prints, web 404s)."""
        names = self.tool_names.get(name)
        if names is None:
            raise MCPError(f"no mcp server named {name!r} is connected")
        for tool_name in names:
            (self.registry.enable if enabled else self.registry.disable)(
                tool_name)
        return len(names)

    def shutdown(self) -> None:
        """Close everything; main()'s finally calls this on EVERY exit."""
        for name in list(self.sessions):
            try:
                self.disconnect(name)
            except Exception:
                pass  # best effort at interpreter teardown

    # ---- status ------------------------------------------------------------

    def servers(self) -> list[dict[str, Any]]:
        """Status snapshot for listings and panels."""
        out: list[dict[str, Any]] = []
        for name in sorted(self.sessions):
            cfg = self.sessions[name].config
            names = self.tool_names.get(name, [])
            target = cfg.url if cfg.url else " ".join(
                [cfg.command or "", *cfg.args]).strip()
            out.append({
                "name": name,
                "transport": "http" if cfg.url else "stdio",
                "target": target,
                "healthy": self.sessions[name].healthy(),
                "tools": len(names),
                "disabled": sum(1 for n in names
                                if self.registry.is_disabled(n)),
                "remembered": name in self.pinned,
            })
        return out

    def _refresh_catalog(self) -> None:
        """Rebuild the selection catalog after membership changes, keeping
        whatever pin set it was built with. No-op without selection."""
        if self.agent is None or getattr(self.agent, "tool_catalog",
                                         None) is None:
            return
        from akshara.tools.selector import enable_selection

        pins = self.agent.tool_catalog.must_include
        catalog, _ = enable_selection(self.registry)
        catalog.must_include = pins
        self.agent.tool_catalog = catalog
