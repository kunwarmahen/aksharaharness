"""CLI entrypoint: argparse -> configured Agent -> REPL or one-shot run."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

from rich.console import Console

from akshara.agent import Agent
from akshara.builder import BUILD_SYSTEM, BuildSpec, default_checks, run_build
from akshara.cli.render import Renderer, SubagentTee
from akshara.cli.repl import Repl, confirm_gate
from akshara.config import (
    _load_dotenv,
    browser_profile,
    default_context_window,
    default_model,
    default_tool_select,
    disabled_tool_patterns,
    load_settings,
)
from akshara.errors import ConfigError, ImageError, ToolError, UserUnavailable
from akshara.images import load_image_block
from akshara.mcp import (MCPError, MCPManager, load_mcp_configs,
                         load_remembered, remembered_path)
from akshara.permissions import SwitchableGate, trust_sandbox, yolo
from akshara.providers import get_provider
from akshara.sandbox import autodetect
from akshara.session import SessionStore, apply_payload
from akshara.subagent import SpawnSubagent, SubagentSpawner
from akshara.tools import default_registry
from akshara.tools.ask_user import AskUser, TerminalChannel
from akshara.tools.selector import (
    AUTO_SELECTION_THRESHOLD,
    DEFAULT_TOOLS_PER_TURN,
    enable_selection,
)


def _guess_provider() -> str:
    """Default to whichever provider has a key in the environment."""
    _load_dotenv()  # .env fills gaps; real env vars already set would win anyway
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("RESPONSES_API_KEY"):
        return "responses"
    raise ConfigError(
        "no API key found: set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in the "
        "environment or .env -- or run a local model with: akshara --provider ollama"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="akshara",
        description="AksharaHarness -- a from-scratch LLM agent harness (learning project).",
    )
    parser.add_argument("--provider",
                        choices=["anthropic", "openai", "responses", "ollama"],
                        help="defaults to whichever API key is set; 'responses' "
                             "speaks OpenAI's Responses API; 'ollama' runs "
                             "local models at localhost:11434 (no key)")
    parser.add_argument("--model", help="model slug (default from env / per-provider)")
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--max-iterations", type=int, default=25,
                        help="tool round-trips allowed per turn (default 25)")
    parser.add_argument("--context-window", type=int, default=None,
                        metavar="TOKENS",
                        help="model context window for auto-compaction "
                             "(default: per provider -- 200000 cloud, 8192 ollama)")
    parser.add_argument("--cwd", default=".", help="sandbox root for tools (default: here)")
    parser.add_argument("--cache", action="store_true",
                        help="prompt caching (anthropic dialect): mark the "
                             "request prefix with cache breakpoints so long "
                             "sessions bill cached tokens at ~0.1x")
    parser.add_argument("--yolo", action="store_true",
                        help="skip permission prompts -- tools run without asking")
    parser.add_argument("--sandbox", action="store_true",
                        help="confine bash with bubblewrap (no network, host "
                             "read-only, workspace-only writes); falls back "
                             "to the plain subprocess when bwrap is missing. "
                             "A confined bash is auto-approved by the gate.")
    parser.add_argument("--build", metavar="SPEC",
                        help="one-shot BUILD MODE: hand SPEC to a fresh agent "
                             "in a scratch workspace, then independently "
                             "re-verify the acceptance commands (unittest "
                             "gate). Exit 0 = build green, 1 = red. "
                             "Use --cwd to choose the workspace explicitly.")
    parser.add_argument("--browse-login", metavar="URL", dest="browse_login",
                        default=None,
                        help="one-time LOGIN SETUP for the browser_* tools: "
                             "opens a VISIBLE Chromium on the "
                             "$AKSHARA_BROWSER_PROFILE directory (set it in "
                             ".env first), you log in yourself -- 2FA and "
                             "captchas included -- then close the window. "
                             "Every later headless session on that profile "
                             "starts logged-in. No model, no API key needed")
    parser.add_argument("--prompt", help="one-shot mode: run this prompt and exit")
    parser.add_argument("--image", action="append", default=[], metavar="PATH",
                        help="attach an image (png/jpeg/gif/webp, <=5 MB) to "
                             "the one-shot prompt; repeat for several. "
                             "Requires --prompt / a positional PROMPT")
    parser.add_argument("--resume", action="store_true",
                        help="restore the newest checkpointed session "
                             "(.akshara/session.sqlite3 under --cwd) before "
                             "the first prompt")
    parser.add_argument("--mcp-config", action="append", default=[],
                        metavar="FILE",
                        help='MCP config file to connect (repeatable): '
                             '{"servers": {"name": {"command": ..., "args": [...]}}}. '
                             'Tools register as mcp__<server>__<tool>.')
    parser.add_argument("--subagents", action="store_true",
                        help="register spawn_subagent so the model can delegate "
                             "self-contained subtasks to fresh-context child "
                             "agents (budget: 5/session); child streams are "
                             "teed to the terminal live")
    parser.add_argument("--web", action="store_true",
                        help="serve a local browser UI instead of the "
                             "terminal REPL (chat, live streaming, approval "
                             "buttons, session controls). Needs the web "
                             "extra: uv sync --extra web")
    parser.add_argument("--host", default="127.0.0.1", metavar="ADDR",
                        help="bind address for --web (default: localhost only)")
    parser.add_argument("--port", type=int, default=8321, metavar="PORT",
                        help="port for --web (default: 8321)")
    parser.add_argument("--tool-select", type=int, default=None, metavar="K",
                        dest="tool_select",
                        help="dynamic tool loading: send only the K best-"
                             "matching tools each turn (BM25 over names/"
                             "descriptions; core tools + "
                             "list_available_tools always load). Auto-"
                             "enables at K=12 above 20 tools; 0 forces it "
                             "off. Same as $AKSHARA_TOOLS_PER_TURN")
    parser.add_argument("prompt_positional", nargs="?", metavar="PROMPT",
                        help="same as --prompt (akshara \"what is in README.md?\")")
    return parser


def enable_subagents(agent: Agent, console: Console) -> SubagentSpawner:
    """--subagents wiring: register the spawn tool and tee child streams to
    the terminal. Factored out of main() so tests (and embedders) can set
    sub-agents up without a full CLI parse."""
    spawner = SubagentSpawner(agent)
    spawner.on_child_event = SubagentTee(console)
    agent.registry.register(SpawnSubagent(spawner))
    return spawner


def _build_mode(args, provider_name: str, settings, model: str,
                console: Console) -> int:
    """--build SPEC: one-shot autonomous build + independent verification.

    Gate policy mirrors the trust story end-to-end: when bwrap confines
    bash, ONLY bash is auto-approved (fs writes still ask); without a
    real sandbox this is exactly the demo's honest yolo-with-warning.
    """
    sandbox = autodetect() if args.sandbox else None
    explicit_cwd = Path(args.cwd)
    workspace = (explicit_cwd if str(args.cwd) != "."
                 else Path(".akshara/builds") / time.strftime("%Y%m%d-%H%M%S"))

    if sandbox is not None and sandbox.confined:
        gate = trust_sandbox(confirm_gate(console), sandbox)
        gate_note = f"sandboxed bash auto-approved ({sandbox.describe}); other writes ask"
    elif args.yolo:
        gate = yolo
        gate_note = "yolo (autonomous build -- bash unsandboxed by choice)"
    else:
        gate = yolo
        gate_note = ("yolo (autonomous build needs it while bash is "
                     "unsandboxed -- rerun with --sandbox to confine bash)")

    spec = BuildSpec(task=args.build, checks=default_checks())

    def factory(ws: Path) -> Agent:
        return Agent(
            get_provider(provider_name, settings),
            model=model,
            system=BUILD_SYSTEM,
            tools=default_registry(sandbox),
            permissions=gate,
            cwd=ws,
            max_iterations=args.max_iterations,
        )

    console.print(f"[bold]{provider_name}[/bold] · {model} · "
                  f"workspace {workspace}\n[dim]gate: {gate_note}[/dim]\n")
    try:
        result = run_build(factory, spec, workspace, on_event=Renderer(console))
    except KeyboardInterrupt:
        console.print("\n[yellow](cancelled)[/yellow]")
        return 130

    console.print("\n[bold]independent verification[/bold]")
    for outcome in result.checks:
        mark = "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]"
        shown = " ".join(Path(a).name if i == 1 else a
                         for i, a in enumerate(outcome.argv))
        console.print(f"  {mark}  $ {shown}"
                      + ("" if outcome.passed else
                         f"  (exit {outcome.actual_exit}, "
                         f"expected {outcome.expect_exit})"))
        if not outcome.passed and outcome.tail:
            console.print(f"        {outcome.tail}", style="red",
                          markup=False, highlight=False)
    for name in result.tampered_tests:
        console.print(f"  [red]FAIL[/red]  {name} was MODIFIED -- the tests "
                      "are the contract; fixing them is cheating")

    usage = f"{result.usage_in} in / {result.usage_out} out"
    console.print(f"\n[bold]{'BUILD GREEN' if result.ok else 'BUILD RED'}[/bold]"
                  f" · {result.elapsed_seconds:.1f}s · {len(result.files)} files:"
                  f" {', '.join(result.files)}\ntokens: {usage}")
    return 0 if result.ok else 1


def _browse_login(url: str, console: Console) -> int:
    """--browse-login URL: headed one-time login on the persistent profile.

    Deliberately provider-free -- no model, no API key, so this runs
    BEFORE main()'s credential resolution. The human beats the login
    wall by hand once; the profile keeps the session for every later
    headless run ([notes/28](../notes/28-browser-tools.md)).
    """
    from akshara.tools.browser import run_login_session  # lazy: [browse] extra

    profile = browser_profile()
    if profile is None:
        print("error: --browse-login needs somewhere to KEEP the login: put\n"
              "  AKSHARA_BROWSER_PROFILE=~/.local/state/akshara/browser-profile"
              "\nin .env first (see .env.example)", file=sys.stderr)
        return 2
    console.print(f"[bold]login setup[/bold] · profile {profile}\n"
                  f"a visible Chromium is opening{f' at {url}' if url else ''} "
                  "-- log in yourself (2FA and\ncaptchas are yours to beat), "
                  "then CLOSE THE WINDOW. Everything you leave\nsigned-in "
                  "here, the agent finds signed-in later.")
    try:
        run_login_session(profile, url)
    except KeyboardInterrupt:
        console.print("\n[yellow](cancelled -- whatever you logged into "
                      "before now is already saved)[/yellow]")
        return 130
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    console.print("[green]profile saved[/green] -- future browser_* "
                  "sessions start from these logins")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    if args.image and args.prompt is None and not args.prompt_positional:
        print("error: --image needs a prompt to attach to (--prompt or a "
              "positional PROMPT); interactive image input is not "
              "supported yet", file=sys.stderr)
        return 2
    if args.browse_login is not None and (args.build or args.prompt
                                          or args.prompt_positional
                                          or args.web):
        print("error: --browse-login is its own mode: drop --build/--prompt/"
              "PROMPT/--web", file=sys.stderr)
        return 2
    if args.build and (args.prompt or args.prompt_positional):
        print("error: --build takes the spec itself; drop --prompt/PROMPT",
              file=sys.stderr)
        return 2
    if args.web and (args.build or args.prompt or args.prompt_positional):
        print("error: --web serves the interactive UI; drop --build/--prompt/"
              "PROMPT (send messages from the browser instead)",
              file=sys.stderr)
        return 2

    # Login setup never touches a model, so it must work without any API
    # key -- dispatched before provider resolution on purpose.
    if args.browse_login is not None:
        _load_dotenv()  # the knob usually lives in .env
        return _browse_login(args.browse_login, console)

    try:
        provider_name = args.provider or _guess_provider()
        settings = load_settings(provider_name)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model = args.model or default_model(provider_name)

    if args.build is not None:
        return _build_mode(args, provider_name, settings, model, console)

    sandbox = autodetect() if args.sandbox else None
    if sandbox is not None:
        console.print(f"[dim]bash sandbox: {sandbox.describe}[/dim]")

    # --web swaps the terminal for a browser: same agent construction below,
    # but the permission gate round-trips over the websocket instead of the
    # rich prompt, and ask_user's channel reaches the human through the page.
    web_session = None
    if args.web:
        try:
            from akshara.web.server import WebSession  # lazy: optional extra
        except ImportError:
            print("error: --web needs the web extra: uv sync --extra web",
                  file=sys.stderr)
            return 2
        web_session = WebSession()

    # The ask-gate is whatever surface the human sits on (terminal y/n/e or
    # browser modal); sandbox-trust composes into it, then a SwitchableGate
    # owns the runtime mode -- /yolo (REPL) and the mode chip (web) flip
    # between asking and bypassing without a restart.
    if web_session is not None:
        ask_gate = web_session.permission_gate()
    else:
        ask_gate = confirm_gate(console)
    if sandbox is not None and sandbox.confined:
        # containment earns autonomy: confined bash runs without asking,
        # every other tool keeps the ask-gate ([notes/16](../notes/16-sandboxing.md))
        ask_gate = trust_sandbox(ask_gate, sandbox)
        console.print("[dim]sandboxed bash auto-approved; other writes ask[/dim]")
    gate = SwitchableGate(ask_gate, mode="yolo" if args.yolo else "ask")

    agent = Agent(
        get_provider(provider_name, settings, cache_control=args.cache),
        model=model,
        system=args.system,
        tools=default_registry(sandbox),
        cwd=Path(args.cwd),
        max_iterations=args.max_iterations,
        context_window=(args.context_window
                        if args.context_window is not None
                        else default_context_window(provider_name)),
        permissions=gate,
    )
    if args.subagents:
        enable_subagents(agent, console)

    # ask_user: the model can consult its human mid-turn. The CHANNEL decides
    # what happens when nobody is home -- browser/websocket and TTY stdin
    # block until answered; piped stdin registers no channel, so an ask fails
    # the turn loudly (UserUnavailable) instead of guessing or hanging.
    # Build agents get nothing: builds are autonomous by definition.
    if web_session is not None:
        agent.registry.register(AskUser(web_session.channel))
    elif sys.stdin.isatty() and sys.stdout.isatty():
        agent.registry.register(AskUser(TerminalChannel()))
    else:
        agent.registry.register(AskUser(None))

    store = SessionStore(Path(args.cwd) / ".akshara" / "session.sqlite3")
    # The manager owns live MCP connections so servers can be added,
    # removed, and toggled MID-session (web panel + REPL /mcp), not just
    # wired at startup. It doubles as the startup bookkeeper below.
    mcp_manager = MCPManager(agent.registry, agent=agent,
                             memory_path=remembered_path(Path(args.cwd)))
    repl = Repl(agent, console, store=store, sandbox=sandbox,
                mcp=mcp_manager)

    # MCP servers: parse errors are fatal (exit 2); connection failures
    # only warn -- a dead optional integration shouldn't kill the
    # session. The manager's shutdown in the finally below closes every
    # opened session on EVERY exit path, including early returns here.
    try:
        for path in args.mcp_config:
            try:
                configs = load_mcp_configs(path)
            except MCPError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            for cfg in configs:
                try:
                    names = mcp_manager.connect(cfg)
                except MCPError as exc:
                    console.print(f"[yellow]mcp '{cfg.name}' unavailable: "
                                  f"{exc}[/yellow]")
                    continue
                console.print(f"[dim]mcp '{cfg.name}': {len(names)} tool(s) "
                              f"-- {', '.join(names)}[/dim]")

        # Servers saved by earlier sessions' "remember this server"
        # reconnect here, after explicit --mcp-config files (a name
        # already connected via flags wins; its saved entry just sits).
        try:
            saved = load_remembered(mcp_manager.memory_path)
        except MCPError as exc:
            console.print(f"[yellow]{exc}; ignoring remembered "
                          "servers[/yellow]")
            saved = []
        for cfg in saved:
            if cfg.name in mcp_manager.sessions:
                continue
            try:
                names = mcp_manager.connect(cfg)
            except MCPError as exc:
                console.print(f"[yellow]mcp '{cfg.name}' unavailable: "
                              f"{exc}[/yellow]")
                continue
            console.print(f"[dim]mcp '{cfg.name}' (remembered): "
                          f"{len(names)} tool(s)[/dim]")

        # Operator kill-switch: AKSHARA_DISABLED_TOOLS globs unregister
        # tools AFTER MCP registration (so whole mcp__ servers can go)
        # and BEFORE catalog building -- a disabled tool must neither be
        # sent, executed, suggested by discovery, nor pinned.
        patterns = disabled_tool_patterns()
        if patterns:
            present = agent.registry.names()
            doomed = sorted({n for pat in patterns
                             for n in present if fnmatch.fnmatch(n, pat)})
            for name in doomed:
                agent.registry.unregister(name)
            unmatched = [pat for pat in patterns
                         if not any(fnmatch.fnmatch(n, pat) for n in present)]
            if doomed:
                console.print(f"[dim]disabled {len(doomed)} tool(s): "
                              f"{', '.join(doomed)}[/dim]")
            if unmatched:
                console.print(f"[yellow]no tools match AKSHARA_DISABLED_"
                              f"TOOLS entry: {', '.join(unmatched)}[/yellow]")

        # Dynamic tool loading (book ch12): decided AFTER MCP registration
        # so external tools count toward the cliff threshold. Precedence
        # is flag > env > auto: --tool-select K (backed by
        # $AKSHARA_TOOLS_PER_TURN) forces it; 0 forces it off; both unset
        # auto-enables past the threshold.
        width = args.tool_select
        if width is None:
            width = default_tool_select()
        if width is None and len(agent.registry) > AUTO_SELECTION_THRESHOLD:
            width = DEFAULT_TOOLS_PER_TURN
        if width:
            catalog, _ = enable_selection(agent.registry)
            agent.tool_catalog = catalog
            agent.tools_per_turn = width
            console.print(f"[dim]tool selection: top {width} of "
                          f"{len(agent.registry) - 1}+discovery each turn "
                          f"(--tool-select 0 to disable)[/dim]")

        if args.resume:
            payload = store.load_latest("default")
            if payload is None:
                console.print("[yellow]--resume: no checkpoint found; "
                              "starting fresh[/yellow]")
            else:
                try:
                    console.print(apply_payload(
                        agent, payload,
                        settings_loader=load_settings,
                        provider_factory=get_provider,
                    ))
                except Exception as exc:
                    console.print(f"[red]--resume failed: {exc}; starting fresh[/red]")

        if web_session is not None:
            from akshara.web.server import launch
            return launch(web_session, agent, store,
                          host=args.host, port=args.port, mcp=mcp_manager)

        if args.prompt is not None or args.prompt_positional:
            try:
                images = [load_image_block(Path(p)) for p in args.image]
            except ImageError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            try:
                prompt = args.prompt or args.prompt_positional
                repl.run_turn(prompt, images=images)  # same rendering path as the REPL
            except KeyboardInterrupt:
                console.print("\n[yellow](cancelled)[/yellow]")
                return 130
            except UserUnavailable as exc:
                console.print(f"\n[red]turn failed: {exc}[/red]")
                return 1
            except Exception as exc:
                console.print(f"\n[red]{exc}[/red]")
                return 1
            return 0

        repl.run()
        return 0
    finally:
        mcp_manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
