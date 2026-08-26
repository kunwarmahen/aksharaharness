"""The web server: one local browser page driving the SAME agent loop the
REPL drives, with the human interactions (permission prompts, ask_user)
round-tripping over a websocket instead of the terminal.

Architecture in three moves:

* THE LOOP STAYS SYNC. Each turn runs on a worker thread pulling the same
  ``run_streaming`` generator the REPL pulls -- identical semantics for
  gates, batches, cancellation, and resumable history. The server is a
  skin, not a second harness.

* A QUEUE BRIDGE crosses the sync/async boundary. The worker thread is
  blocking by design; the event loop must never. Every human interaction
  is a ``_Pending`` object with its own ``queue.Queue``: the worker blocks
  in ``get(timeout=...)`` polling for an answer AND for cancellation; the
  websocket handler (async side) delivers answers with ``put_nowait``.
  Outbound events fan out to every connected client the same way, via
  ``loop.call_soon_threadsafe`` captured per-connection.

* EVERYTHING IS ENVELOPES. One websocket (/ws) carries tagged JSON both
  ways -- stream deltas, tool cards, permission requests, asks, state --
  so the frontend is a dumb renderer and the session survives reconnects
  (a fresh tab receives ``state``, any still-pending interaction, then a
  transcript replay). REST endpoints cover plain request/response controls.

Cancellation mirrors Ctrl-C exactly: the cancel flag is honored while
streaming (between events), between tool executions, and while blocked on
a human answer -- each path closes the generator so the agent's
resumable-history synthesis runs. Because the loop polls the flag INSIDE
the model stream too (Agent.interrupt_check), a Stop click lands within
one SSE event of a long generation -- the one gap terminal Ctrl-C had
that a signal-less worker thread used to keep open.

Mutating REST endpoints refuse to run mid-turn (409): the REPL serves its
slash commands between turns too -- same single-operator assumption.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from akshara.agent import Agent, ToolExecuted, TurnEnd
from akshara.config import default_model, load_settings
from akshara.context import estimate_history
from akshara.errors import ProviderError, UserUnavailable
from akshara.images import image_block_from_bytes
from akshara.permissions import PermissionRequest
from akshara.pricing import session_cost
from akshara.providers import get_provider
from akshara.session import SessionStore, apply_payload
from akshara.types import (
    EndEvent,
    ImageBlock,
    RedactedThinking,
    RedactedThinkingBlock,
    StartEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCallStart,
    ToolResult,
)

CANCELLED = object()  # sentinel pushed into pending queues on cancel


def cost_line(agent: Agent) -> str:
    """The dollars footer, same honesty rules as the REPL's _cost_line:
    local models are free, unknown slugs show NO figure -- never $0."""
    buckets = agent.usage_by_model
    if not buckets:
        return ""
    if agent.provider.name == "ollama":
        return "$0.00 (local model)"
    total, complete = session_cost(buckets)
    if total == 0.0 and not complete:
        return ""
    suffix = "" if complete else " (priced models only)"
    return f"~${total:.4f}{suffix}"


# ---------------------------------------------------------------------------
# Session: owns the bridge between worker threads and websocket clients
# ---------------------------------------------------------------------------


@dataclass
class _Client:
    """One connected browser tab: its inbound event queue plus the loop that
    queue belongs to (call_soon_threadsafe needs the RIGHT loop)."""
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue


@dataclass
class _Pending:
    """One open question to the human (permission or ask), answered by id."""
    kind: str  # "permission" | "ask"
    envelope: dict[str, Any]  # what was broadcast; re-sent to late tabs
    answers: queue.Queue = field(default_factory=queue.Queue)  # dict|CANCELLED


class WebSession:
    """Shared state for one served session. Created BEFORE the Agent (the
    CLI needs this object's permission gate at construction), attached after.

    Also IS the ask_user ``UserChannel`` -- one bridge serves both kinds of
    human interaction.
    """

    def __init__(self) -> None:
        self.agent: Agent | None = None
        self.store: SessionStore | None = None
        self.mcp: Any | None = None  # MCPManager; optional -- tests may omit
        self._clients: list[_Client] = []
        self._clients_lock = threading.Lock()
        self._pending: _Pending | None = None
        self._cancel = threading.Event()
        self.turn_active = False

    # ---- wiring -------------------------------------------------------------

    def attach(self, agent: Agent, store: SessionStore | None,
               mcp: Any | None = None) -> None:
        self.agent = agent
        self.store = store
        if mcp is not None:
            self.mcp = mcp
        # Raw StreamEvents are PUSHED here while each response streams (they
        # cannot be yielded -- collect() owns that pull). Same wiring as the
        # REPL's renderer: without this, text/thinking deltas vanish.
        agent.on_stream_event = self._emit
        # The Stop button's teeth: the loop polls this flag between stream
        # events and between tool calls, so a click lands mid-model-call --
        # as close to terminal Ctrl-C as a signal-less worker can get.
        agent.interrupt_check = self._cancel.is_set

    @property
    def channel(self) -> "WebSession":
        """The ask_user channel: this object (it implements .ask below)."""
        return self

    # ---- outbound fan-out ---------------------------------------------------

    def connect(self, loop: asyncio.AbstractEventLoop,
                events: asyncio.Queue) -> _Client:
        client = _Client(loop=loop, events=events)
        with self._clients_lock:
            self._clients.append(client)
        return client

    def disconnect(self, client: _Client) -> None:
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)

    def broadcast(self, envelope: dict[str, Any]) -> None:
        """Thread-safe fan-out. Called from workers AND async handlers."""
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.loop.call_soon_threadsafe(
                    client.events.put_nowait, envelope)
            except RuntimeError:
                pass  # that tab's loop is gone; disconnect() will reap it

    # ---- inbound answers ------------------------------------------------------

    def deliver(self, message: dict[str, Any]) -> None:
        """Route a websocket message arriving from the browser."""
        mtype = message.get("type")
        if mtype == "cancel":
            self.cancel_turn()
            return
        if mtype == "answer" and self._pending is not None:
            self._pending.answers.put_nowait(message)

    def cancel_turn(self) -> None:
        """The Stop button / ctrl-c equivalent. Honored between stream
        events -- i.e. mid-model-call, within one SSE event -- between
        tools, and while a human interaction is pending (that queue gets
        the sentinel immediately)."""
        self._cancel.set()
        pending = self._pending
        if pending is not None:
            pending.answers.put_nowait(CANCELLED)

    def _wait_for_answer(self, pending: _Pending) -> dict[str, Any]:
        """Worker-side block: poll the answer queue AND the cancel flag.
        Polling (not a bare blocking get) is what lets cancel interrupt a
        wait without anyone knowing which thread notices first."""
        while True:
            try:
                item = pending.answers.get(timeout=0.2)
            except queue.Empty:
                if self._cancel.is_set():
                    raise KeyboardInterrupt  # turn-cancel, REPL semantics
                continue
            if item is CANCELLED:
                raise KeyboardInterrupt
            return item

    def _ask_human(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Broadcast a question, block for its answer, clean up after."""
        pending = _Pending(kind=envelope["type"], envelope=envelope)
        self._pending = pending
        try:
            self.broadcast(envelope)
            return self._wait_for_answer(pending)
        finally:
            self._pending = None
            self.broadcast({"type": "resolved", "id": envelope.get("id")})

    # ---- the two interactive surfaces ----------------------------------------

    def permission_gate(self):
        """PermissionFn for the browser: y/n/e with full edit round-trips,
        mirroring cli/repl.py's confirm_gate loop beat for beat."""

        def gate(request: PermissionRequest) -> bool:
            if request.read_only:
                # Same contract as the terminal gate: a tool that declared
                # itself side-effect-free has nothing to confirm.
                return True
            edited = False
            while True:
                answer = self._ask_human({
                    "type": "permission_request",
                    "id": uuid.uuid4().hex[:8],
                    "tool_name": request.tool_name,
                    "summary": request.summary,
                    "arguments": request.arguments,
                    "edited": edited,
                })
                decision = answer.get("decision")
                if decision == "approve":
                    return True
                if decision == "deny":
                    return False
                if decision == "edit":
                    amended = answer.get("edited_args")
                    if not isinstance(amended, dict):
                        continue  # unusable payload -- just ask again
                    # Same contract as the terminal editor: swap args,
                    # refresh the preview through the tool's own summary(),
                    # re-ask. What you approve is what runs.
                    request.arguments = amended
                    try:
                        request.summary = (request.summarize(amended)
                                           if request.summarize else
                                           f"{request.tool_name}"
                                           f"({json.dumps(amended)})")
                    except Exception:
                        request.summary = (
                            f"{request.tool_name}({json.dumps(amended)})")
                    edited = True

        return gate

    def ask(self, question: str, choices: list[str],
            context: str = "") -> str:
        """UserChannel implementation: ask_user blocks here until the tab
        answers. Empty/malformed answers count as a cancel, not an answer."""
        answer = self._ask_human({
            "type": "ask",
            "id": uuid.uuid4().hex[:8],
            "question": question,
            "context": context,
            "choices": choices,
        })
        text = answer.get("text")
        if not isinstance(text, str) or not text.strip():
            raise KeyboardInterrupt
        return text

    # ---- turns ---------------------------------------------------------------

    def start_turn(self, text: str, images: list[ImageBlock]) -> None:
        """Spawn the worker thread. Caller has already refused concurrent
        turns; this is single-operator, one-turn-at-a-time by design."""
        self._cancel.clear()
        self.turn_active = True
        self.broadcast({"type": "turn_started"})
        threading.Thread(target=self._run_turn, args=(text, images),
                         daemon=True, name="akshara-turn").start()

    def _run_turn(self, text: str, images: list[ImageBlock]) -> None:
        """The REPL's run_turn, transplanted: pull the generator, forward
        envelopes, honor cancel at every yield boundary."""
        agent = self.agent
        stream = agent.run_streaming(text, images=images or None)
        ended = False  # a natural TurnEnd went out -- don't double-report
        cancelled = False
        try:
            for event in stream:
                ended = ended or isinstance(event, TurnEnd)
                self._emit(event)
                if self._cancel.is_set():
                    cancelled = True
                    stream.close()  # runs the outstanding-call synthesis
                    break
        except UserUnavailable as exc:
            self.broadcast({"type": "turn_error", "message": str(exc)})
        except KeyboardInterrupt:
            cancelled = True  # cancel landed while blocked on the human
        except ProviderError as exc:
            wait = (f" retry after {exc.retry_after:.0f}s"
                    if exc.retry_after else "")
            self.broadcast({"type": "turn_error",
                            "message": f"{type(exc).__name__}: {exc}{wait}"})
        except Exception as exc:
            self.broadcast({"type": "turn_error",
                            "message": f"{type(exc).__name__}: {exc}"})
        if cancelled and not ended:
            self.broadcast({"type": "turn_cancelled"})
        self.turn_active = False
        self._cancel.clear()
        self.broadcast({"type": "state", **self.state()})
        self.broadcast({"type": "turn_done"})

    def _emit(self, event) -> None:
        """AgentEvent/StreamEvent -> envelope(s). Mirrors render.py's match."""
        match event:
            case StartEvent(model=model):
                self.broadcast({"type": "start", "model": model})
            case TextDelta(text=text):
                self.broadcast({"type": "delta", "text": text})
            case ThinkingDelta(index=_, text=text, signature=_):
                if text:  # signature fragments accumulate silently
                    self.broadcast({"type": "thinking_delta", "text": text})
            case RedactedThinking(index=_, data=data):
                self.broadcast({"type": "redacted_thinking", "chars": len(data)})
            case ToolCallStart(index=_, id=_, name=name):
                self.broadcast({"type": "tool_start", "name": name})
            case ToolExecuted(call=call, result=result):
                self.broadcast({
                    "type": "tool_result",
                    "name": call.name,
                    "arguments": call.arguments,
                    "output": result.content,
                    "is_error": result.is_error,
                    # Pressure moves during a turn too (each iteration
                    # refills the window); the header bar follows along
                    # instead of waiting for the end-of-turn state.
                    "utilization": self.agent.utilization(),
                })
            case EndEvent(stop_reason=_, usage=_):
                pass  # per-call usage; the TurnEnd footer carries totals
            case TurnEnd(reason="end_turn", response=response, iterations=n):
                usage = response.usage if response is not None else None
                self.broadcast({
                    "type": "turn_end",
                    "reason": "end_turn",
                    "stop_reason": response.stop_reason if response else "",
                    "text": response.message.text() if response is not None else "",
                    "input_tokens": usage.input_tokens if usage else 0,
                    "output_tokens": usage.output_tokens if usage else 0,
                    "cost_line": cost_line(self.agent),
                    "iterations": n,
                })
            case TurnEnd(reason=reason, response=None, iterations=n):
                self.broadcast({"type": "turn_end", "reason": reason,
                                "iterations": n, "cost_line": ""})

    # ---- snapshots -------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        agent = self.agent
        u = agent.total_usage
        return {
            "provider": agent.provider.name,
            "model": agent.model,
            "cwd": str(agent.ctx.cwd),
            # "ask" | "yolo" -- agents built with a bare gate (tests,
            # embedders) report the fixed default rather than crashing.
            "mode": getattr(agent.permissions, "mode", "ask"),
            # Session awareness snapshot ([notes/29]); None when the host
            # built the agent without an EnvContext.
            "env_context": (agent.env_context.describe()
                            if hasattr(agent, "env_context") else None),
            "tools": agent.registry.names(),
            # Runtime-disabled subset of ``tools`` (the /api/tools panel's
            # toggles); empty for a stock session.
            "disabled_tools": agent.registry.disabled_names(),
            "usage": {"input": u.input_tokens, "output": u.output_tokens,
                      "cache_read": u.cache_read_tokens,
                      "cache_write": u.cache_write_tokens},
            "utilization": agent.utilization(),
            # The honest numbers behind the pressure bar: what the last
            # request actually filled and how big the window is at all.
            # Before the first response arrives there is no provider
            # figure -- fall back to the chars/4 estimate and SAY so.
            "context_estimated": agent.last_context_tokens == 0,
            "context_tokens": (agent.last_context_tokens or
                               estimate_history(agent.history)),
            "context_window": agent.context_window,
            "cost_line": cost_line(agent),
            "turn_active": self.turn_active,
        }

    def history_envelopes(self) -> list[dict[str, Any]]:
        """Replayable transcript for a freshly (re)connected tab, rendered in
        almost the same shapes the live feed uses, minus streaming."""
        results_by_id: dict[str, dict] = {}
        for message in self.agent.history:
            if message.role != "user":
                continue
            for block in message.content:
                if isinstance(block, ToolResult):
                    results_by_id[block.tool_call_id] = {
                        "output": block.content, "is_error": block.is_error}

        out: list[dict[str, Any]] = []
        for message in self.agent.history:
            if message.role == "user":
                texts = [b.text for b in message.content
                         if isinstance(b, TextBlock)]
                images = sum(1 for b in message.content
                             if isinstance(b, ImageBlock))
                if texts or images:  # pure tool-result carriers are skipped
                    out.append({"type": "user_message",
                                "text": "\n".join(texts), "images": images})
                continue
            thinking = "".join(b.thinking for b in message.content
                               if isinstance(b, ThinkingBlock))
            redacted = sum(1 for b in message.content
                           if isinstance(b, RedactedThinkingBlock))
            if thinking:
                out.append({"type": "thinking_done", "text": thinking})
            if redacted:
                out.append({"type": "redacted_thinking",
                            "chars": sum(len(b.data) for b in
                                         message.content
                                         if isinstance(b, RedactedThinkingBlock))})
            if message.text().strip():
                out.append({"type": "assistant_text",
                            "text": message.text()})
            for call in message.tool_calls():
                result = results_by_id.get(call.id, {})
                out.append({"type": "tool_result", "name": call.name,
                            "arguments": call.arguments,
                            "output": result.get("output", ""),
                            "is_error": result.get("is_error", False)})
        return out


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def make_app(session: WebSession, static_dir: Path | None = None,
             settings_loader=None, provider_factory=None) -> FastAPI:
    """Build the app around a WebSession. The session is injected rather
    than constructed here so tests can drive ScriptedProvider agents
    offline through the exact production routes. The load-path provider
    factories are injectable too, on the same principle as
    ``session.apply_payload`` (a scripted provider has no real settings)."""
    static_dir = static_dir or Path(__file__).parent / "static"

    app = FastAPI(title="akshara", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        events: asyncio.Queue = asyncio.Queue()
        client = session.connect(asyncio.get_running_loop(), events)
        reader = asyncio.create_task(_ws_read(sock))
        writer = asyncio.create_task(_ws_write(sock, events))
        try:
            await sock.send_json({"type": "state", **session.state()})
            pending = session._pending
            if pending is not None:  # a question was already on screen
                await sock.send_json(pending.envelope)
            for envelope in session.history_envelopes():
                await sock.send_json(envelope)
            done, _ = await asyncio.wait({reader, writer},
                                         return_when=asyncio.FIRST_COMPLETED)
            for task in done:  # surface a read/write failure, if any
                if task.exception() is not None:
                    raise task.exception()
        except Exception:
            pass  # disconnect path -- finally does the cleanup either way
        finally:
            reader.cancel()
            writer.cancel()
            session.disconnect(client)

    async def _ws_read(sock: WebSocket) -> None:
        while True:
            session.deliver(await sock.receive_json())

    async def _ws_write(sock: WebSocket, events: asyncio.Queue) -> None:
        while True:
            await sock.send_json(await events.get())

    # ---- REST ----

    def require_ready() -> None:
        if session.agent is None:
            raise HTTPException(500, "no agent attached")

    def require_idle() -> None:
        require_ready()
        if session.turn_active:
            raise HTTPException(409, "a turn is running -- wait or cancel")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        require_ready()
        return session.state()

    @app.get("/api/history")
    def history() -> list[dict[str, Any]]:
        """Replayable transcript — the client rebuilds its view from this on
        connect/reload, so the server's history stays the single truth."""
        require_ready()
        return session.history_envelopes()

    @app.post("/api/message")
    async def message(req: Request) -> dict[str, Any]:
        require_idle()
        body = await req.json()
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "text is required")
        images: list[ImageBlock] = []
        for img in body.get("images") or []:
            try:
                raw = base64.b64decode(img["data_base64"])
                images.append(image_block_from_bytes(img["filename"], raw))
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, f"bad image: {exc}") from exc
        session.start_turn(text, images)
        return {"ok": True}

    @app.post("/api/model")
    async def set_model(req: Request) -> dict[str, Any]:
        require_idle()
        slug = (await req.json()).get("model")
        if not isinstance(slug, str) or not slug.strip():
            raise HTTPException(400, "model slug required")
        session.agent.model = slug.strip()
        return session.state()

    @app.post("/api/provider")
    async def set_provider(req: Request) -> dict[str, Any]:
        require_idle()
        name = (await req.json()).get("provider")
        try:
            settings = load_settings(name)
            session.agent.provider = get_provider(name, settings)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        # Model slugs are per-provider namespaces (same rule as the REPL's
        # /provider): carrying the old slug over would ask provider B for a
        # model it may not have.
        try:
            session.agent.model = default_model(name)
        except Exception:
            pass  # keep the old slug; the operator sets one explicitly
        return session.state()

    @app.post("/api/permissions")
    async def set_permissions(req: Request) -> dict[str, Any]:
        """Flip the permission mode ("ask" <-> "yolo"). Deliberately NO
        require_idle(): unlike model/provider swaps, flipping mid-turn is
        safe and is the point -- it applies to every not-yet-approved call
        of the running turn (rescues a turn stuck in approval modals).
        Broadcasts state so every open tab's mode chip follows along."""
        require_ready()
        gate = session.agent.permissions
        if not hasattr(gate, "set_mode"):
            raise HTTPException(400, "permission mode is fixed for this "
                                     "session (built without a switchable gate)")
        try:
            gate.set_mode((await req.json()).get("mode"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        session.broadcast({"type": "state", **session.state()})
        return session.state()

    @app.post("/api/env-context")
    async def set_env_context(req: Request) -> dict[str, Any]:
        """Flip session awareness (off/local/full). Deliberately NO
        require_idle(): the composed system is consulted at each iteration's
        request, so a flip lands on the running turn's next model call --
        same argument as the permission flip. The flip itself runs in a
        worker thread (to_thread): upgrading to 'full' may do the one-time
        geo lookup, which must never block the event loop. Broadcasts state
        so every open tab's chip follows along."""
        require_ready()
        ctx = getattr(session.agent, "env_context", None)
        if ctx is None:
            raise HTTPException(400, "this session has no env context "
                                     "(built without session awareness)")
        mode = (await req.json()).get("mode")
        try:
            await asyncio.to_thread(ctx.flip, mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        session.broadcast({"type": "state", **session.state()})
        return session.state()

    @app.get("/api/tools")
    def tools_list() -> list[dict[str, Any]]:
        """The tools panel's data: everything registered, with enough
        detail to decide what to pull -- description, whether it prompts,
        and its current enabled state."""
        require_ready()
        registry = session.agent.registry
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "read_only": tool.read_only,
                "enabled": not registry.is_disabled(tool.name),
            }
            for tool in sorted(registry, key=lambda t: t.name)
        ]

    @app.post("/api/tools")
    async def tools_toggle(req: Request) -> dict[str, Any]:
        """Enable/disable one tool by exact name. Deliberately NO
        require_idle(): like the permission flip, mid-turn is the POINT --
        pulling a tool applies to the running turn's next call (the loop
        re-consults the registry per call), so a runaway `bash` chain can
        be cut without waiting it out. Unknown names are a clean 400."""
        require_ready()
        body = await req.json()
        name, enabled = body.get("name"), body.get("enabled")
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise HTTPException(400, "name (str) and enabled (bool) required")
        registry = session.agent.registry
        if name not in registry:
            raise HTTPException(404, f"no such tool: {name!r}")
        if enabled:
            registry.enable(name)
        else:
            registry.disable(name)
        # Broadcast so every open tab's chip count follows along.
        session.broadcast({"type": "state", **session.state()})
        return session.state()

    # ---- MCP server management ----------------------------------------------

    def require_mcp() -> Any:
        require_ready()
        if session.mcp is None:
            raise HTTPException(400, "this session has no mcp manager")
        return session.mcp

    @app.get("/api/mcp")
    def mcp_list() -> dict[str, list[dict[str, Any]]]:
        """The servers section of the tools panel: transport, liveness,
        tool counts (incl. how many are soft-disabled), remembered flag."""
        return {"servers": require_mcp().servers()}

    @app.post("/api/mcp/toggle")
    async def mcp_toggle(req: Request) -> dict[str, Any]:
        """Soft-switch one server's whole toolset. Deliberately NO
        require_idle(), same argument as /api/tools: the disabled set is
        consulted per call, so cutting a server off lands on its very
        next call of a running turn. The process stays warm."""
        mcp = require_mcp()
        body = await req.json()
        name, enabled = body.get("name"), body.get("enabled")
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise HTTPException(400, "name (str) and enabled (bool) required")
        try:
            mcp.set_enabled(name, enabled)
        except Exception as exc:  # unknown server (MCPError) -> clean 404
            raise HTTPException(404, str(exc)) from exc
        session.broadcast({"type": "state", **session.state()})
        return session.state()

    @app.post("/api/mcp/add")
    async def mcp_add(req: Request) -> dict[str, Any]:
        """Connect a new server mid-session. Two shapes:

        * fields -- {name, command|url, args?, env?, remember?}: what the
          panel's simple form sends;
        * {config: "<json>"} -- the panel's paste-JSON mode, same
          ``{"servers": {...}}`` syntax as --mcp-config files; may add
          several at once.

        DOES take require_idle(): connecting spawns processes and
        mutates the registry, which must not interleave with a turn's
        batch iteration. Per-server results come back so one bad entry
        doesn't hide the others."""
        from akshara.mcp import MCPServerConfig, parse_mcp_text

        require_idle()
        mcp = require_mcp()
        body = await req.json()
        results: list[dict[str, Any]] = []

        def attempt(cfg) -> None:
            try:
                names = mcp.connect(cfg, remember=bool(body.get("remember")))
                results.append({"name": cfg.name, "ok": True,
                                "tools": len(names)})
            except Exception as exc:
                results.append({"name": cfg.name, "ok": False,
                                "error": str(exc)})

        if isinstance(body.get("config"), str):
            try:
                configs = parse_mcp_text(body["config"])
            except Exception as exc:  # MCPError from the shared parser
                raise HTTPException(400, str(exc)) from exc
            for cfg in configs:
                attempt(cfg)
        else:
            name = body.get("name")
            command, url = body.get("command"), body.get("url")
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(400, "name is required")
            if (command is None) == (url is None):
                raise HTTPException(400, "exactly one of 'command' (stdio)"
                                         " or 'url' (http) is required")
            args = body.get("args") or []
            env = body.get("env") or None
            if not isinstance(args, list) or \
                    not all(isinstance(a, str) for a in args):
                raise HTTPException(400, "args must be a list of strings")
            if env is not None and (not isinstance(env, dict) or not all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in env.items())):
                raise HTTPException(400, "env must map strings to strings")
            attempt(MCPServerConfig(
                name=name.strip(),
                command=command if command is None else str(command),
                args=args, url=url if url is None else str(url),
                env=env))
        session.broadcast({"type": "state", **session.state()})
        return {"results": results, **session.state()}

    @app.post("/api/mcp/remove")
    async def mcp_remove(req: Request) -> dict[str, Any]:
        """Disconnect one server and pull its tools (a saved entry is
        forgotten too). require_idle() like add: closing transports and
        unregistering must not race a running turn."""
        from akshara.mcp import MCPError

        require_idle()
        mcp = require_mcp()
        name = (await req.json()).get("name")
        if not isinstance(name, str):
            raise HTTPException(400, "name (str) required")
        try:
            removed = mcp.disconnect(name)
        except MCPError as exc:
            raise HTTPException(404, str(exc)) from exc
        session.broadcast({"type": "state", **session.state()})
        return {"removed": removed, **session.state()}

    @app.post("/api/save")
    async def save(req: Request) -> dict[str, Any]:
        require_idle()
        if session.store is None:
            raise HTTPException(500, "no session store configured")
        name = (await req.json()).get("name") or "default"
        version = session.store.save(session.agent,
                                     provider_name=session.agent.provider.name,
                                     session_id=name)
        return {"saved": name, "version": version}

    @app.post("/api/load")
    async def load(req: Request) -> dict[str, Any]:
        require_idle()
        if session.store is None:
            raise HTTPException(500, "no session store configured")
        name = (await req.json()).get("name") or "default"
        payload = session.store.load_latest(name)
        if payload is None:
            raise HTTPException(404, f"no checkpoint named '{name}'")
        try:
            apply_payload(session.agent, payload,
                          settings_loader=settings_loader or load_settings,
                          provider_factory=provider_factory or get_provider)
        except Exception as exc:
            raise HTTPException(400, f"restore failed: {exc}") from exc
        # Same rule as the REPL's load path: checkpoints store the COMPOSED
        # system, so recompose from the live EnvContext for fresh facts.
        ctx = getattr(session.agent, "env_context", None)
        if ctx is not None:
            ctx.reapply()
        return session.state()

    @app.post("/api/compact")
    def compact() -> dict[str, Any]:
        require_idle()
        stats = session.agent.compact()
        return {"stats": stats, **session.state()}

    @app.post("/api/clear")
    def clear() -> dict[str, Any]:
        require_idle()
        session.agent.history.clear()
        return session.state()

    return app


def launch(session: WebSession, agent: Agent, store: SessionStore | None,
           *, host: str = "127.0.0.1", port: int = 8321,
           mcp: Any | None = None) -> int:
    """Blocking entrypoint for `akshara --web`: attach, banner, serve."""
    import uvicorn

    session.attach(agent, store, mcp=mcp)
    app = make_app(session)
    servers = f" · mcp servers={len(mcp.sessions)}" if mcp else ""
    print(f"\n  akshara web UI -> http://{host}:{port}\n"
          f"  provider={agent.provider.name} · model={agent.model} · "
          f"tools={len(agent.registry)}{servers} · cwd={agent.ctx.cwd}\n"
          "  ctrl-c stops the server\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
