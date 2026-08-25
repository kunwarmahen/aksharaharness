"""The REPL: input loop, slash commands, and Ctrl-C semantics.

Ctrl-C contract (the part every interactive harness gets wrong once):
  * at the prompt      -> clear the line, keep the session
  * during a turn      -> cancel the TURN, never the session; the agent's
                          resumable-history cleanup runs via stream.close()
  * Ctrl-D / /quit     -> exit
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from akshara.agent import Agent
from akshara.builder import BUILD_SYSTEM, BuildSpec, default_checks, run_build
from akshara.cli.render import Renderer
from akshara.config import default_model, load_settings
from akshara.context import RED, estimate_history
from akshara.errors import ImageError, RateLimitError, UserUnavailable
from akshara.images import load_image_block
from akshara.pricing import session_cost
from akshara.types import ImageBlock
from akshara.permissions import MODES, PermissionRequest, SwitchableGate, yolo
from akshara.providers import get_provider
from akshara.sandbox import ToolSandbox
from akshara.session import SessionStore, apply_payload
from akshara.tools import default_registry

HELP = """[bold]commands[/bold]
  /help              this text
  /model [slug]      show or hot-swap the model
  /provider [name]   show or switch provider (history survives -- internal types!)
  /tools             list registered tools and their schemas
  /build SPEC        build a project from SPEC in a scratch workspace,
                     then independently verify it (BUILD GREEN/RED)
  /history           dump the conversation so far
  /usage             session token + dollar totals (+ context pressure)
  /save [name]       checkpoint this session (SQLite, append-only versions)
  /load [name]       restore the newest checkpoint of a session
  /compact           force context compaction now (auto-fires at 80%)
  /clear             reset history (keeps provider/model)
  /yolo [on|off]     flip permission prompts: bypass everything, or ask
                     again (bare /yolo toggles; applies mid-turn)
  /image PATH...     stage image(s) onto your NEXT message ("clear" unstages)
  /quit              exit

  //text             send a message starting with a literal slash
  trailing \\        continue the same message on the next line
  ctrl-c             cancel current turn (or clear the prompt line)
  permission prompts offer y/n/e -- 'e' edits the tool call before it runs
  (--resume loads the newest checkpoint at startup)"""


#: An edit round: amend the pending arguments, or cancel. Raises ValueError
#: with a human-readable message when the edit can't be parsed.
EditFn = Callable[[dict[str, Any]], "dict[str, Any] | None"]


def terminal_editor(args: dict[str, Any]) -> dict[str, Any] | None:
    """The default EditFn: $EDITOR on a temp file when interactive,
    a single input() line otherwise (piped sessions, tests).

    Returns the amended args dict, None to cancel. Bad JSON raises
    ValueError (json.JSONDecodeError already is one).
    """
    pretty = json.dumps(args, indent=2)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor and sys.stdin.isatty() and sys.stdout.isatty():
        fd, path = tempfile.mkstemp(suffix=".json", prefix="akshara-edit-")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(pretty + "\n")
            if subprocess.run([editor, path]).returncode != 0:
                return None  # editor died -- treat as cancel, not denial
            raw = Path(path).read_text()
        finally:
            os.unlink(path)
    else:
        raw = input(f"edited args JSON ({json.dumps(args)})> ").strip()
        if not raw:
            return None  # empty answer = cancel
    edited = json.loads(raw)
    if not isinstance(edited, dict):
        raise ValueError("edited arguments must be a JSON object")
    return edited


def confirm_gate(console: Console, editor: EditFn | None = None):
    """The CLI's PermissionFn: show exactly what will happen, default No --
    and 'e' amends the call before approving (approve-with-edits).

    The loop: preview -> y/n/e. Editing swaps ``request.arguments`` for
    the amended dict, re-renders the summary through the tool's own
    ``summary()`` (pre-bound by the agent loop as ``summarize``), tags
    the panel *(edited)*, and asks again -- so what you approve is what
    runs, now literally. A cancelled or unparseable edit never denies:
    it returns to the prompt.

    Read-only tools never prompt -- a tool that declared itself
    side-effect-free (the same flag allow_read_only trusts) has nothing
    to confirm, and a discovery hatch behind a permission wall would
    defeat the selection loop that pins it.
    """
    edit = editor or terminal_editor

    def gate(request: PermissionRequest) -> bool:
        if request.read_only:
            return True
        edited = False
        while True:
            console.print()
            console.print(Panel(
                request.summary,
                title=f"approve {request.tool_name}(){' (edited)' if edited else ''}?",
                border_style="yellow", title_align="left"))
            answer = Prompt.ask("run it?", choices=["y", "n", "e"],
                                default="n").lower()
            if answer == "y":
                return True  # edits ride along: the loop adopts them
            if answer == "n":
                return False
            try:
                amended = edit(request.arguments)
            except ValueError as exc:
                console.print(f"[red]bad edit: {exc} -- try again[/red]")
                continue
            if amended is None:  # cancelled -- back to the prompt
                continue
            request.arguments = amended
            try:
                request.summary = (request.summarize(amended) if
                                   request.summarize else
                                   f"{request.tool_name}({json.dumps(amended)})")
            except Exception:
                # A broken summary() must not trap the approval flow.
                request.summary = f"{request.tool_name}({json.dumps(amended)})"
            edited = True

    return gate


class Repl:
    def __init__(self, agent: Agent, console: Console,
                 store: SessionStore | None = None,
                 input_fn: Callable[[str], str] | None = None,
                 sandbox: ToolSandbox | None = None) -> None:
        self.agent = agent
        self.console = console
        self.store = store
        # Injectable so tests can feed lines without a real terminal.
        self._input = input_fn or input
        self.renderer = Renderer(console)
        # bash confinement for /build children (None = the tools' default).
        self.sandbox = sandbox
        # StreamEvents are PUSHED here while each model response streams;
        # ToolExecuted/TurnEnd still arrive as yielded events below. The
        # indirection also owns the per-turn spinner: deltas must drop it
        # as they arrive -- see _on_stream_event.
        agent.on_stream_event = self._on_stream_event
        # The dead-air spinner while one is installed (run_turn).
        self._spinner: Status | None = None
        # Images staged by /image, consumed by the NEXT turn (validated at
        # attach time so a bad path errors immediately, not mid-conversation).
        self._pending_images: list[ImageBlock] = []

    # ---- main loop ---------------------------------------------------------

    def run(self) -> None:
        self._banner()
        while True:
            try:
                line = self._read_line()
            except EOFError:
                self.console.print()
                return
            except KeyboardInterrupt:
                self.console.print()  # clear the line, keep the session
                continue

            if not line:
                continue
            if line.startswith("//"):  # escape hatch for literal slashes
                pass
            elif line.startswith("/"):
                if self._command(line):
                    return  # /quit
                continue

            try:
                self.run_turn(line, images=self._take_pending_images())
            except KeyboardInterrupt:
                # The generator below was interrupted mid-pull; closing it
                # runs the agent's outstanding-call synthesis.
                self.console.print("\n[yellow](cancelled)[/yellow]")
            except UserUnavailable as exc:
                # The model tried to consult its human and nobody was home
                # (piped session, or stdin hit EOF). Turn failed, history
                # resumable -- the session itself survives.
                self.console.print(f"\n[red]turn failed: {exc}[/red]")
            except RateLimitError as exc:
                wait = f" retry after {exc.retry_after:.0f}s" if exc.retry_after else ""
                self.console.print(f"\n[red]rate limited.{wait}[/red]")

    def _read_line(self) -> str:
        """One LOGICAL line of input. ``input()`` is line-oriented and we
        don't want a readline/curses dependency just for pasting code or
        multi-paragraph prompts -- so a trailing backslash simply continues
        on a ``... `` prompt. Exactly one backslash is consumed per line,
        and continuation lines keep their leading indentation (pasted
        code must survive)."""
        parts: list[str] = []
        # The prompt itself carries the permission mode -- bypassing is
        # precisely when you want a standing reminder of it. (Tests inject
        # input_fn and never see this string.)
        line = self._input(
            "yolo> " if self._gate_mode() == "yolo" else "> ").strip()
        while True:
            continues = line.endswith("\\")
            if continues:
                line = line[:-1]
            parts.append(line)
            if not continues:
                return "\n".join(parts)
            line = self._input("... ").rstrip()  # keep indentation

    def _on_stream_event(self, event) -> None:
        """Push-channel entry: drop the dead-air spinner at the FIRST
        streamed fragment, then paint.

        The spinner must stop HERE, not in run_turn's pull loop -- deltas
        never pass through that loop (they are pushed), so stopping there
        meant a tool-free reply streamed ENTIRELY inside an active rich
        Live region, whose repaints erase the partial lines they wrap.
        On a real terminal that rendered as an empty reply.
        """
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None
        self.renderer(event)

    def run_turn(self, user_input: str,
                 *, images: list[ImageBlock] | None = None) -> None:
        """One user turn, fully rendered. Public because one-shot mode and
        the REPL share this exact path. ``images`` (loaded via
        akshara.images) ride along in the same user message -- how
        ``akshara --image shot.png "what is this?"`` works."""
        stream = self.agent.run_streaming(user_input, images=images)
        spinner = self.console.status("[dim]… connecting[/dim]", spinner="dots")
        spinner.start()
        self._spinner = spinner  # _on_stream_event drops it at first delta
        try:
            for event in stream:  # ToolExecuted / TurnEnd only
                self.renderer(event)
        except BaseException:
            # Deterministic cleanup on ANY abnormal exit (including
            # KeyboardInterrupt raised mid-pull): closing the generator
            # runs the agent's outstanding-call synthesis.
            stream.close()
            raise
        finally:
            # Died before ANY event arrived (connection refused, instant
            # provider error) -- nothing else ever got the chance to stop.
            if self._spinner is not None:
                self._spinner.stop()
                self._spinner = None

    # ---- slash commands ----------------------------------------------------

    def _command(self, line: str) -> bool:  # returns True when exiting
        parts = line[1:].split(maxsplit=1)
        name, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")

        match name:
            case "help":
                self.console.print(HELP)
            case "model":
                if arg:
                    self.agent.model = arg
                    self.console.print(f"[green]model -> {arg}[/green]")
                else:
                    self.console.print(f"model: {self.agent.model}")
            case "provider":
                if arg:
                    self._switch_provider(arg)
                else:
                    self.console.print(
                        f"provider: {self.agent.provider.name}, model: {self.agent.model}"
                    )
            case "tools":
                if self.agent.tool_catalog is not None:
                    pinned = [n for n in self.agent.tool_catalog.must_include
                              if n != "list_available_tools"]
                    self.console.print(
                        f"[dim]selection active: top {self.agent.tools_per_turn} of "
                        f"{len(self.agent.tool_catalog.tools)} each turn; "
                        f"always loaded: {', '.join(pinned)} + "
                        f"list_available_tools[/dim]")
                for spec in self.agent.registry.specs():
                    self.console.print(
                        f"[bold]{spec.name}[/bold] — {spec.description}"
                    )
                    self.console.print_json(json.dumps(spec.parameters))
            case "build":
                if not arg:
                    self.console.print("[red]/build needs a spec: "
                                       "/build <what to build>[/red]")
                else:
                    self._build_command(arg)
            case "history":
                self._show_history()
            case "usage":
                u = self.agent.total_usage
                line = (f"session tokens: {u.input_tokens} in / {u.output_tokens} out "
                        f"(cache read {u.cache_read_tokens} / write "
                        f"{u.cache_write_tokens})")
                ratio = self.agent.utilization()
                if ratio is not None:
                    real = self.agent.last_context_tokens
                    size = (f"~{real}" if real else
                            f"~{estimate_history(self.agent.history)} estimated")
                    line += (f"\ncontext: {size} tokens "
                             f"({ratio:.0%} of usable window; "
                             f"red zone at {int(RED * 100)}%)")
                line += self._cost_line()
                self.console.print(line)
            case "compact":
                stats = self.agent.compact()
                parts = [f"[green]compacted[/green] "
                         f"{stats['messages_before']} -> "
                         f"{stats['messages_after']} message(s)"]
                if stats["masked"]:
                    parts.append(f"elided {stats['masked']} old tool result(s)")
                if stats["summarized"]:
                    parts.append(f"summarized a {stats['segment_size']}-message "
                                 "segment")
                self.console.print(" · ".join(parts))
            case "clear":
                self.agent.history.clear()
                self.console.print("[green]history cleared[/green]")
            case "yolo":
                self._yolo_command(arg)
            case "save":
                self._save_session(arg or "default")
            case "load":
                self._load_session(arg or "default")
            case "image":
                self._image_command(arg)
            case "quit" | "exit":
                return True
            case _:
                self.console.print(f"[red]unknown command {line!r} — /help[/red]")
        return False

    # ---- permission mode ------------------------------------------------------

    def _switchable_gate(self) -> SwitchableGate | None:
        """The runtime-switchable gate, if this session has one. Agents built
        with a bare PermissionFn (tests, embedders) keep the fixed-gate
        behavior: /yolo reports instead of crashing."""
        gate = self.agent.permissions
        return gate if isinstance(gate, SwitchableGate) else None

    def _gate_mode(self) -> str:
        """Current mode of ANY gate -- switchable, plain confirm, or yolo.
        What the prompt prefix and the banner display."""
        switch = self._switchable_gate()
        if switch is not None:
            return switch.mode
        return "yolo" if self.agent.permissions is yolo else "ask"

    def _yolo_command(self, arg: str) -> None:
        """/yolo [on|off]: flip between bypassing every approval and asking
        first. Effective immediately, INCLUDING the remaining tool calls of
        an in-flight turn -- the agent loop consults the gate per call."""
        switch = self._switchable_gate()
        if switch is None:
            self.console.print(
                "[red]this session's permission gate is fixed at startup -- "
                "/yolo can't switch it[/red]")
            return
        wanted = arg.strip().lower()
        if not wanted:
            new_mode = switch.toggle()
        elif wanted == "on":
            switch.set_mode("yolo")
            new_mode = "yolo"
        elif wanted == "off":
            switch.set_mode("ask")
            new_mode = "ask"
        else:
            self.console.print(f"[red]usage: /yolo [on|off][/red]")
            return
        if new_mode == "yolo":
            self.console.print("[red]yolo ON -- tools run WITHOUT asking; "
                               "you accept the risk[/red]")
        else:
            self.console.print("[green]permission prompts back on -- "
                               "risky calls ask first[/green]")

    # ---- cost ---------------------------------------------------------------

    def _cost_line(self) -> str:
        """The /usage dollars line: '' when nothing has been spent yet.

        Honesty rules (mirroring akshara.pricing): local models are
        genuinely free; unknown slugs are NEVER rendered as $0 -- a made-up
        zero looks identical to a real one and quietly trains the wrong
        instinct about what sessions cost.
        """
        buckets = self.agent.usage_by_model
        if not buckets:
            return ""
        if self.agent.provider.name == "ollama":
            return "\ncost: $0.00 (local model)"
        total, complete = session_cost(buckets)
        if total == 0.0 and not complete:
            return ("\n[dim]cost: no list price known for this "
                    "session's model(s)[/dim]")
        suffix = "" if complete else " [dim](priced models only)[/dim]"
        return f"\nsession cost: ~${total:.4f}{suffix}"

    # ---- build mode ---------------------------------------------------------

    def _build_command(self, spec_text: str) -> None:
        """/build SPEC: a fresh child agent builds in a scratch workspace
        (parent history untouched -- same isolation reasoning as sub-agents),
        then run_build re-verifies independently. The child inherits the
        parent's gate, with one upgrade: confined bash is auto-approved,
        so builds don't stall on y/n for every test run."""
        from akshara.permissions import trust_sandbox

        bash_tool = self.agent.registry.get("bash")
        sandbox = self.sandbox or bash_tool.sandbox
        workspace = (self.agent.ctx.cwd / ".akshara" / "builds"
                     / time.strftime("%Y%m%d-%H%M%S"))
        parent_gate = self.agent.permissions

        def factory(ws: Path) -> Agent:
            return Agent(
                self.agent.provider,
                model=self.agent.model,
                system=BUILD_SYSTEM,
                tools=default_registry(sandbox),
                permissions=trust_sandbox(parent_gate, sandbox),
                cwd=ws,
            )

        self.console.print(f"[dim]building in {workspace} "
                           f"(gate: inherited + sandboxed-bash auto-approve)"
                           f"[/dim]")
        try:
            result = run_build(factory, BuildSpec(task=spec_text), workspace,
                               on_event=self.renderer)
        except KeyboardInterrupt:
            self.console.print("\n[yellow](build cancelled)[/yellow]")
            return
        for outcome in result.checks:
            mark = "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]"
            shown = " ".join(Path(a).name if i == 1 else a
                             for i, a in enumerate(outcome.argv))
            self.console.print(f"  {mark}  $ {shown}")
        verdict = ("[bold green]BUILD GREEN[/bold green]" if result.ok
                   else "[bold red]BUILD RED[/bold red]")
        for name in result.tampered_tests:
            self.console.print(f"  [red]{name} was MODIFIED -- tests are the "
                               "contract[/red]")
        self.console.print(f"{verdict} · {result.elapsed_seconds:.1f}s · "
                           f"{len(result.files)} file(s) in {workspace}")

    # ---- image attachments --------------------------------------------------

    def _take_pending_images(self) -> list[ImageBlock]:
        """Hand staged images to the starting turn and unstage them."""
        pending, self._pending_images = self._pending_images, []
        return pending or []

    def _image_command(self, arg: str) -> None:
        """/image: stage attachments for the NEXT message. Validated now --
        a typo should error at attach time, not after three more turns."""
        import shlex  # stdlib; quoted paths survive intact

        if not arg:
            if self._pending_images:
                self.console.print(
                    f"[blue]{len(self._pending_images)} image(s) staged "
                    "for your next message[/blue]")
            else:
                self.console.print(
                    "usage: /image PATH [PATH...] (png/jpeg/gif/webp; "
                    "staged onto your next message)")
            return
        if arg.strip() == "clear":
            self._pending_images.clear()
            self.console.print("[green]staged images cleared[/green]")
            return
        for path in shlex.split(arg):
            try:
                block = load_image_block(Path(path))
            except ImageError as exc:
                self.console.print(f"[red]{exc} -- not attached[/red]")
                continue
            kb = len(block.data) * 3 // 4 // 1024
            self._pending_images.append(block)
            self.console.print(
                f"[blue]attached {Path(path).name} ({block.media_type}, "
                f"~{kb} KB) -- rides your next message[/blue]")

    # ---- durable sessions ---------------------------------------------------

    def _save_session(self, session_id: str) -> None:
        if self.store is None:
            self.console.print("[red]no session store configured[/red]")
            return
        version = self.store.save(
            self.agent, provider_name=self.agent.provider.name,
            session_id=session_id,
        )
        u = self.agent.total_usage
        self.console.print(
            f"[green]saved[/green] '{session_id}' v{version} · "
            f"{len(self.agent.history)} message(s) · "
            f"{u.input_tokens}in/{u.output_tokens}out "
            "[dim](append-only -- every save is a new version)[/dim]"
        )

    def _load_session(self, session_id: str) -> None:
        if self.store is None:
            self.console.print("[red]no session store configured[/red]")
            return
        payload = self.store.load_latest(session_id)
        if payload is None:
            self.console.print(f"[red]no checkpoint named '{session_id}'[/red]")
            return
        try:
            summary = apply_payload(
                self.agent, payload,
                settings_loader=load_settings,
                provider_factory=get_provider,
            )
        except Exception as exc:  # corrupt/newer payload must not kill the REPL
            self.console.print(f"[red]restore failed: {exc}[/red]")
            return
        self.console.print(f"[green]{summary}[/green]")

    def _switch_provider(self, name: str) -> None:
        try:
            settings = load_settings(name)
            self.agent.provider = get_provider(name, settings)
        except Exception as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        # Model slugs are per-provider namespaces: carrying the old slug
        # over would ask provider B for a model it may not have.
        try:
            self.agent.model = default_model(name)
            note = f"model reset to {self.agent.model}"
        except Exception:
            note = f"model left as {self.agent.model} -- set {name.upper()}_MODEL"
        self.console.print(
            f"[green]provider -> {name}; {note}. History intact: "
            "it is stored in internal types, not wire format.[/green]"
        )

    def _show_history(self) -> None:
        from akshara.types import (ImageBlock, RedactedThinkingBlock,
                                   TextBlock, ThinkingBlock, ToolCall,
                                   ToolResult)

        for i, message in enumerate(self.agent.history):
            self.console.rule(f"[{i}] {message.role}")
            for block in message.content:
                match block:
                    case ToolCall(id=cid, name=n, arguments=args):
                        self.console.print(f"[cyan]tool_call {n}({args}) id={cid}[/cyan]")
                    case ToolResult(tool_call_id=cid, content=c, is_error=e):
                        flag = " [error]" if e else ""
                        preview = c[:200] + ("..." if len(c) > 200 else "")
                        self.console.print(f"[magenta]tool_result[{cid}]{flag}: {preview}[/magenta]")
                    case ThinkingBlock(thinking=t, signature=s):
                        sig = f" (signed, {len(s)} chars)" if s else ""
                        preview = t[:200] + ("..." if len(t) > 200 else "")
                        self.console.print(f"[dim italic]thinking{sig}: "
                                           f"{preview}[/dim italic]")
                    case RedactedThinkingBlock(data=d):
                        self.console.print(f"[dim italic]thinking (redacted, "
                                           f"{len(d)} chars of ciphertext)[/dim italic]")
                    case ImageBlock(media_type=m, data=d):
                        kb = len(d) * 3 // 4 // 1024
                        self.console.print(f"[blue]image ({m}, ~{kb} KB decoded "
                                           f"payload not displayed)[/blue]")
                    case TextBlock(text=t):
                        self.console.print(t or "[dim](empty)[/dim]")
                    case _:
                        self.console.print(f"[red](unknown block: {type(block).__name__})[/red]")

    def _banner(self) -> None:
        # warn ONLY when the gate really is bypassed (startup flag or a
        # session that flipped since)
        gated = "  [red](yolo: no permission prompts)[/red]" \
            if self._gate_mode() == "yolo" else ""
        self.console.print(
            f"[bold]akshara[/bold] · provider={self.agent.provider.name} · "
            f"model={self.agent.model} · tools={len(self.agent.registry)} · "
            f"cwd={self.agent.ctx.cwd}{gated}"
        )
        self.console.print("[dim]/help for commands, ctrl-c cancels a turn[/dim]\n")
