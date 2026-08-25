# 06 · The CLI: REPL loop, cancellation UX, and rendering

> Files: `akshara/cli/{main,repl,render}.py`.

## Layering: the CLI is a consumer, never a participant

`main.py` (argparse) → `Repl` (input loop + slash commands) →
`Renderer` (rich painting). None of them contain agent logic; the CLI's
only job is translating human input into `agent.run_streaming()` pulls
and AgentEvents into pixels. Evidence the layering works:
[`examples/tool_round_trip.py`](../examples/tool_round_trip.py) gets full
tool-call rendering by reusing `Renderer` alone.

The permission gate is likewise injected: `confirm_gate(console)` is a
factory returning the `PermissionFn`, so the UI owns *how to ask* while
the agent owns only *whether to proceed* (denial → data, always). The
ask grew a third answer — y/n/**e** where `e` amends the pending call's
arguments before approval ([notes/20](20-approve-with-edits.md)).

## The Ctrl-C contract (the part every interactive harness gets wrong once)

Three different keystrokes-mean-three different things:

| Where | Behavior |
|---|---|
| at the `>` prompt | clear the line, keep the session |
| during a turn | cancel the TURN, never the session |
| Ctrl-D / `/quit` | exit |

"Cancel the turn" has teeth because of the agent-side cleanup: closing
the event generator (`stream.close()`) throws GeneratorExit into
`run_streaming`, which synthesizes answers for outstanding tool calls
before re-raising ([05-agent-loop.md](05-agent-loop.md)). The next prompt
is then safe: no dangling ids, nothing half-executed. The REPL's rule is
one line and absolute:

```python
except BaseException:
    stream.close()
    raise
```

Never `except Exception` here — KeyboardInterrupt and GeneratorExit are
BaseException subclasses, and swallowing either corrupts exactly the
cleanup path that makes cancellation safe.

## Rendering details that mattered

* **Stream model text with `markup=False, highlight=False`.** Rich would
  otherwise interpret `[TODO]` in model output as a markup tag and eat it.
  Tool panels wrap output in `rich.text.Text` for the same reason;
  `Syntax(json.dumps(args))` only for arguments we produced ourselves.
* **ToolCallDelta is deliberately unrendered** — argument fragments flash
  by too fast to read; the full parsed dict appears in the result panel
  one moment later. Render for comprehension, not for liveness theater.
* **Errors get red borders**, results preview at 400 chars with an
  explicit `[... N more chars ...]` marker instead of silent truncation.

## Slash commands as match/case over (name, arg)

`_command` splits once, matches on the verb, and `/provider name`
hot-swaps `agent.provider` while keeping history — possible ONLY because
history stores internal types. That single command is the payoff of the
whole normalization strategy; see [02-wire-formats.md](02-wire-formats.md).

Small escape hatch learned from real terminals: `//text` sends a literal
leading slash, because "/etc/hosts?" is a legitimate first character.

## Entrypoint shape

`main(argv)` returns an int exit code and is wired via
`akshara = "akshara.cli.main:main"` in pyproject — argparse stays
testable without spawning processes. Config errors exit 2 with the fix
in the message ("set ANTHROPIC_API_KEY (see .env.example)"); a cancelled
one-shot exits 130 (128+SIGINT convention).

## Polish round

* **Spinner for the dead air.** Between Enter and the first stream
  event there is pure network latency; a rich Status spinner covers it
  and stops at the first pushed event (`run_turn` owns start/stop with
  a `finally`, so Ctrl-C before any event can't strand it).
* **`/provider name` resets the model too.** Model slugs are
  per-provider namespaces -- carrying anthropic's slug into openai
  would request a model B doesn't have. Falls back to "set
  {NAME}_MODEL" if the new provider has no default configured.
* **`_turn` -> `run_turn`.** One-shot mode was calling the REPL's
  private method across modules. Both paths now share one public
  entry point, so CLI fixes land in both modes by construction.
* **Empty end_turn gets a visible "(no text in reply)"** instead of
  silence + footer (happens when the model burns the turn on thinking).
* `/usage` shows cache-write alongside cache-read; banner shows the
  sandbox cwd.

Observed live, worth remembering: some gateway responses report input
tokens as null even on success -- our `or 0` decode turns that into a
plain `0 in` footer rather than a crash. The counter may be lying, but
the session never notices.

## Durable sessions & context commands

New surface, all thin over `session.py` / `context.py`:

* `/save [name]` — checkpoint to SQLite (`.akshara/session.sqlite3`);
  every save appends a version row. `/load [name]` restores the newest.
* `--resume` — restore before the first prompt; verified live by saving
  a code word, exiting, and recalling it in a fresh process.
* `/compact` — force two-layer compaction now; auto-fires in the red
  zone behind `_before_model_call` (`--context-window` sizes the window).
* `/usage` gained a context line: provider-reported input tokens when
  available, else the chars/4 estimate labeled as such (gateways that
  null their counters make the fallback visible: "~N estimated").

## Local models & per-provider defaults

`--provider ollama` runs local models via the inherited OpenAI adapter
(see notes/02 for why that is all it takes). One CLI behavior changed
with it: **`--context-window` no longer hardcodes 200000** — when the
flag is omitted, each provider answers with its own default (cloud
200k, ollama 8192, overridable via `{PROVIDER}_CONTEXT_WINDOW`). The
window feeds auto-compaction's red-zone math, so a wrong guess silently
disables compaction on small local models.

## Multi-line input without a readline dependency

Pasting code or multi-paragraph prompts into `input()`-based REPLs
usually loses everything after the first newline. Full readline/curses
editing is a dependency-heavy rewrite -- instead: a **trailing
backslash continues on a `... ` prompt**. `input()` stays line-oriented;
`_read_line()` just loops over it and joins with real newlines.

Two details the tests caught:

* **Continuation lines keep leading indentation** (only trailing
  whitespace is dropped). The first version stripped every line --
  fine for prose, fatal for pasted Python.
* Consume exactly ONE marker backslash per line; check the flag BEFORE
  overwriting it (first loop draft re-tested an already-consumed line
  and swallowed the final segment).

`input` is injected (`input_fn=`), so tests drive prompts without a
terminal.

Bonus bug the multi-line smoke test flushed out: `_banner`'s yolo
warning had an inverted condition -- it announced "no permission
prompts" precisely when the gate WAS active. Nobody noticed while live
tests ran *with* `--yolo`; the warning only shows correctly in runs
that don't use it. Classic inverted-guard blind spot.

## A second front door: --web, and choosing where the human is

`main.py` grew one more mode with no new agent logic: `--web` builds a
`WebSession` *before* the Agent (the browser gate is injected at
construction, same as `confirm_gate`), then hands the finished session
to `akshara.web.server.launch()` ([22-web-ui.md](22-web-ui.md)). The
layering rule held: main.py picks the front door; the door owns how to
ask.

The same moment gave every run a second question to answer — not just
"which gate?" but "**where is the human?**", decided by TTY detection:

* interactive REPL / one-shot on a terminal → `TerminalChannel()`
  (numbered choices + free text at the prompt);
* piped stdin or captured output → `AskUser(None)` — headless, any call
  raises `UserUnavailable`, which fails a one-shot with exit 1 and a
  REPL turn without killing the session;
* `--web` → the session itself (it implements `.ask()` over its
  websocket bridge).

One registration site, three channels — the frontend decides how to
reach the person, the tool stays ignorant of all of them.

## /yolo — flipping the gate mid-session

`--yolo` used to be a startup-only decision: whatever gate went into
the Agent stayed for life. Now a `SwitchableGate` wraps the session's
real ask-gate (terminal y/n/e or browser modal), and both frontends can
flip it while running:

* **REPL**: `/yolo` (toggle), `/yolo on`, `/yolo off`. The prompt itself
  becomes `yolo> ` and the banner warns while bypassed — the mode is
  never silent.
* **Web**: a mode chip in the top bar; clicking POSTs `/api/permissions`
  ([22-web-ui.md](22-web-ui.md)), which broadcasts state so every open
  tab follows.

Two design points worth keeping:

* **Mid-turn flips are allowed on purpose.** The agent loop consults the
  gate once per call, so a flip lands at the next not-yet-approved call —
  exactly what you want when a turn is spamming approval modals. That is
  why `/api/permissions` skips the `require_idle` guard that
  `/api/model` and `/api/provider` keep.
* **Composition order matters.** `trust_sandbox` wraps *inside* the
  switchable (it decorates the ask-branch, not the whole gate), so
  `agent.permissions` IS the switch — no wrapper hides it from the
  frontends, and confined-bash auto-approval survives mode flips either
  way. Agents built with a bare function (tests, embedders) are told
  the gate is fixed instead of crashing.

## /tools off|on — pulling tools mid-session

The web UI got a panel of tool switches ([22-web-ui.md](22-web-ui.md));
the terminal twin is one command with glob matching, so whole families
go in one stroke:

```
/tools                     # list everything; pulled tools stay visible, marked [off]
/tools off browser_*       # disable by name or glob — live on the next request
/tools on browser_open     # ...and back
```

Design points that carried over from `/yolo`:

* **Reversible, session-scoped.** `registry.disable()` hides a tool but
  leaves it registered ([04-tools.md](04-tools.md)) — nothing to
  rebuild, `/tools on` restores exactly what was there. The permanent
  spelling stays `AKSHARA_DISABLED_TOOLS`, which unregisters at startup.
* **The listing never lies by omission.** Disabled tools still appear,
  marked `[off]` with a count in the header — an operator who forgot
  what they switched off should be able to see it, not diff against a
  memory.
* One rendering trap: rich parses `[off]` as a MARKUP tag and silently
  eats it — same failure mode as model output eating `[TODO]`, fixed
  the same way (`\[off]`). The CLI keeps re-learning that brackets are
  load-bearing.
