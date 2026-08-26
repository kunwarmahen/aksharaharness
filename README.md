# AksharaHarness — a from-scratch LLM agent harness

A learning project: build the machinery behind tools like Claude Code —
an agentic loop around a chat model, a JSON-Schema tool system, a
permission gate, hand-parsed streaming — **without SDKs, pydantic, or
frameworks**. Runtime deps: `httpx` and `rich` only; optional extras
add a browser UI (`fastapi` + `uvicorn`, `[web]`) and real-browser
tools (`playwright`, `[browse]`) — each installs only what it needs.

The code is the tutorial; [`notes/`](notes/) is the per-topic
write-up. New to agents entirely?
Two ways in: [TUTORIAL.md](TUTORIAL.md) — type a ~200-line working
agent into existence, step by step, no background assumed ([TUTORIAL.html](TUTORIAL.html)
is the same thing as a styled page you can open or send to anyone) — or
[notes/00-guided-tour.md](notes/00-guided-tour.md), a plain-English
tour of how this one works, with diagrams.

## Setup

```bash
uv sync                          # creates .venv from pyproject.toml
cp .env.example .env             # then fill in a key (never committed)
```

`.env` is loaded automatically (a ~15-line loader in `config.py` — no
python-dotenv dependency); real environment variables still win.

`.env` variables per provider (`PREFIX` = `ANTHROPIC`, `OPENAI`,
`RESPONSES`, or `OLLAMA`):

| Var | Meaning |
|---|---|
| `{PREFIX}_API_KEY` | secret (Anthropic also accepts `ANTHROPIC_AUTH_TOKEN`; Ollama needs none) |
| `{PREFIX}_BASE_URL` | Anthropic: excludes `/v1` · OpenAI-style (incl. ollama + responses): INCLUDES `/v1` |
| `{PREFIX}_MODEL` | default model slug |
| `{PREFIX}_CONTEXT_WINDOW` | window assumption for compaction (default 200000 cloud / 8192 ollama) |

Any OpenAI-compatible gateway works as `OPENAI_BASE_URL`; gateways that
also speak the Messages dialect work under `ANTHROPIC_BASE_URL`; the
Responses dialect (`RESPONSES_*`) targets OpenAI directly, OpenRouter's
stateless beta, or a local Ollama >= 0.13.3 — all on the same `/v1`
surface ([notes/19](notes/19-responses-api.md)).

## Usage

```bash
uv run akshara                                        # REPL (provider auto-guessed from keys)
uv run akshara --provider openai                      # pick a dialect explicitly
uv run akshara --provider ollama                      # LOCAL models (localhost:11434, no key)
uv run akshara --provider ollama --model qwen3.8      # any tag you have pulled
uv run akshara --yolo                                 # no permission prompts (careful)
                                                      #   ...and /yolo flips it back
                                                      #   mid-session (web UI: mode chip)
uv run akshara --cache                                # prompt-cache breakpoints on
uv run akshara --resume                               # restore the newest checkpoint
uv run akshara --env-context local                    # machine facts only (default: full)
uv run akshara "summarize README.md"                  # one-shot prompt, then exit
uv run akshara --image photo.png "what's in this picture?"   # vision one-shot
```

### It knows where (and when) it is

Sessions start aware instead of clueless: the system prompt carries your
time and timezone, host and working directory — and, by default, your
city from ONE public-IP lookup at startup. Ask *"what's the temperature
outside?"* and it answers for where you actually are, fetching the
weather with its own tools — it doesn't burn a turn asking which city
you're in. The same prompt tells it to try its tools before asking you
for any fact it could discover itself; questions stay reserved for what
only you know — preferences, permissions, irreversible calls.

Three levels (`AKSHARA_ENV_CONTEXT` / `--env-context` set the start,
`/env` or the web UI's env chip flip it live):

| level | what the agent gets |
|---|---|
| `full` *(default)* | machine facts **+ your city** — one keyless lookup to ipinfo.io per session |
| `local` | machine facts only; nothing leaves the machine beyond the chat itself |
| `off` | nothing injected — asks you everything, as before |

One honest tradeoff on `full`: your city rides inside every request sent
to your LLM provider. If you'd rather share nothing, `local` or `off`.

### One-command starts

`start.sh` wraps the common setups so you don't have to remember flags.
Run it bare for a numbered menu, or name what you want:

```bash
./start.sh                    # menu — pick by number
./start.sh local              # free & private: Ollama, no key needed
./start.sh cloud              # uses whichever API key your .env has
./start.sh web                # chat in your browser (foreground)
./start.sh local-web          # Ollama + browser UI

./start.sh web-start          # browser UI in the BACKGROUND (web-start
./start.sh web-stop           #   local' pins Ollama) — plus stop,
./start.sh web-status         #   status, restart and web-logs to
./start.sh web-logs           #   follow its output
```

Everything after the preset passes straight through (`./start.sh local
--yolo`, `./start.sh cloud --resume`, `./start.sh web-start --port 9000`).
Keys, models and URLs still come from `.env`; the script only adds the
flags that make each setup different. It needs `curl` for its health
checks — nothing else beyond bash and uv.

### Run in a container (podman/docker)

The UI can also run inside a container — handy to keep the agent's
hands off your real filesystem entirely, or to put it on an always-on
box. The image carries code only; keys arrive at run time.

```bash
podman build --format docker -t localhost/akshara-web .
# (--format docker so the HEALTHCHECK survives; OCI images ignore it)

# cloud road: mount your .env read-only (or pass -e ANTHROPIC_API_KEY=...).
# --userns=keep-id lets the in-container user read your 600-perm .env.
podman run -d --name akshara-web --userns=keep-id -p 8400:8321 \
    -v ./.env:/app/.env:ro localhost/akshara-web

# local road: reach Ollama on the HOST via its special name. The
# trailing flags compose with the image's entrypoint; --provider is
# needed because env vars alone don't tip the key-based guess.
podman run -d --name akshara-local -p 8401:8321 \
    -e OLLAMA_BASE_URL=http://host.containers.internal:11434/v1 \
    -e OLLAMA_MODEL=qwen3.8 \
    localhost/akshara-web --web --host 0.0.0.0 --provider ollama

podman logs -f akshara-web        # watch it boot
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8400/
```

Notes worth knowing:

* **Port 8321 is fixed inside** (the health check probes it); publish it
  wherever you like with `-p`.
* **The tools run in the container**, not on your machine — `read_file`
  sees `/app`, so give it a workspace if you want it to touch your
  files: mount one and add args after the image name (they compose):
  `-v ./workspace:/workspace localhost/akshara-web --web --cwd /workspace`.
* **Sessions live in `/app/.akshara/`** and vanish with the container;
  add `-v akshara-state:/app/.akshara` to keep checkpoints across runs.
* Docker users: swap `podman` → `docker`, and replace
  `host.containers.internal` with `host.docker.internal`.

Builder mode ([notes/18](notes/18-builder-mode.md)): spec → files →
acceptance checks re-run INDEPENDENTLY (never trusting the model's
word); a red verification is fed back into the same conversation for a
bounded repair round (default 2), then BUILD GREEN/RED. Exit code
doubles as a CI gate. Test files are checksummed — a repair job that
edits them fails the build even if the suite then passes, and tampering
is never repaired:

```bash
uv run akshara --build "CLI that converts between temperature units"   # BUILD GREEN → exit 0
uv run akshara --build --cwd somewhere/seeded "repair the broken CLI"
```

Bash sandboxing ([notes/16](notes/16-sandboxing.md)) — `--sandbox`
picks bubblewrap when usable (network off by default, host filesystem
invisible outside read-only system paths, env scrubbed of secrets,
timed-out process trees fully reaped) and falls back to the legacy
scrubbed-env subprocess otherwise:

```bash
uv run akshara --sandbox                    # autodetect (bwrap > subprocess)
uv run akshara --sandbox none               # explicit legacy behavior
```

Dynamic tool loading ([notes/17](notes/17-tool-selection.md)) — past
~20 tools the model's selection accuracy hits the cliff, so per turn
only the top-K best-matching tools are SENT (BM25 over name +
description). The autonomy loop's floor always loads regardless of
matching — `read_file`, `write_file`, `edit_file`, `bash`, `glob`,
`grep` plus the `list_available_tools` discovery hatch — and calling
any real tool by its exact name loads it on the spot, so a selection
miss costs nothing:

```bash
uv run akshara --mcp-config big.json --tool-select 12   # force width K
uv run akshara --tool-select 0                          # opt out of auto-enable
# (.env equivalent: AKSHARA_TOOLS_PER_TURN=12)
```

Trim tools you never want — not sent, not executed, not even suggested
by discovery ([.env.example](.env.example)):

```bash
AKSHARA_DISABLED_TOOLS=browser_*,mcp__slack__*   # comma-separated globs on tool names
```

That kill-switch is permanent (tools unregistered at startup). To pull
a tool for just this session and put it back later: `/tools off bash`,
`/tools on bash` in the REPL — globs work (`/tools off browser_*`) —
or the switches in the web UI's tools panel. Both take effect
immediately, even mid-turn ([notes/17](notes/17-tool-selection.md)).

MCP servers (hand-rolled JSON-RPC — no SDK; stdio and Streamable-HTTP
transports, picked by config shape). Config file, repeatable flag; tools
register as `mcp__<server>__<tool>`:

```json
{"servers": {"tiny":  {"command": "python",
                       "args": ["examples/tiny_mcp_server.py"]},
             "remote": {"url": "http://127.0.0.1:8000/mcp"}}}
```

```bash
uv run akshara --mcp-config mcp.json          # connect + discover at startup
python examples/tiny_mcp_server.py --http     # the same server over Streamable HTTP
```

Servers are runtime furniture, not just startup wiring: `/mcp` lists
them, `/mcp add NAME URL` (or `NAME COMMAND [ARGS...]`) connects one
mid-session — asking whether to remember it in `.akshara/mcp.json` for
future launches — and `/mcp off|on NAME` / `/mcp remove NAME` toggle or
tear down. The web panel's "servers & tools" section does the same:
health dots, add-by-form or paste-JSON, per-server switches, remove.
Disabling keeps the process warm; removing kills it and forgets any
saved entry ([notes/09](notes/09-mcp.md)).

Sub-agents (agent-as-tool: fresh-context children with a filtered tool
catalog, per-session spawn budget, compact results — child streams tee
to the terminal live):

```bash
uv run akshara --subagents "research X and report back"
```

Browser UI ([notes/22](notes/22-web-ui.md)) — `--web` serves the same
agent loop at `http://127.0.0.1:8321` (`--host`/`--port` to move it):
streamed replies and thinking rendered as markdown (headings, tables,
code blocks — by a ~150-line escape-first renderer, no library), tool
cards, permission prompts with approve/deny/**edit**, image attachments,
save/load/compact/model switches, a permission-mode chip (ask ⇄ yolo,
switchable mid-turn like the REPL's `/yolo`), a tools panel (every
registered tool, switchable off/on mid-run like the REPL's `/tools`)
with an MCP servers section above it — health dots, add by form or
paste-JSON, per-server switches, remove; same powers as `/mcp`
mid-session, and remembered servers auto-reconnect on future launches,
a live context-pressure meter (amber at 60%, red at 80% — where
auto-compaction starts caring), and a ■ Stop button that lands
mid-sentence, not just between tool calls (Esc works too). Install the
extra once: `uv sync --extra web`.

```bash
uv run akshara --provider ollama --web      # local model + browser UI
uv run akshara --web                        # provider auto-guessed, as usual
```

The `ask_user` tool rides along in every surface: when the model hits a
question only you can answer ("Postgres or SQLite?", "may I delete
it?"), it pauses mid-turn and asks — numbered choices plus free text at
the terminal prompt, a modal in the browser — then proceeds on your
answer. With no interactive terminal attached (piped stdin, cron,
evals), asking fails the turn loudly rather than guessing; history
stays resumable for a later interactive run ([notes/22](notes/22-web-ui.md)).

Evals (trajectory-level, real model, costs money — merge/nightly
cadence, not per-commit; exit code doubles as a CI gate):

```bash
uv run --env-file .env python examples/run_evals.py   # 7 golden trajectories
uv run --env-file .env python examples/run_evals.py --async   # same, concurrent
```

Images ([notes/15](notes/15-images.md)): `--image PATH` (repeatable)
attaches png/jpeg/gif/webp files (≤5 MB each) to a one-shot prompt;
in the REPL, `/image PATH...` stages them onto your *next* message
(`/image` alone shows the stage, `/image clear` unstages). The loader
base64-encodes before any turn starts — a bad path errors at attach
time, never mid-conversation; both adapters carry the resulting
`ImageBlock` in their own dialect, and compaction bills images by
decoded size.

Self-reliance tools — the difference between an agent that answers
and one that finishes ([notes/23](notes/23-glob.md)–
[27](notes/27-read-image.md)):

- **glob** finds files by NAME (`**` recursion, newest-first) without
  a permission-gated bash call; **grep** still searches contents.
- **todo_write / todo_read** keep live plan state (`.akshara/todos.json`,
  replace-whole-list semantics) — distinct from write_note's durable
  facts, and cheap grounding that keeps local models on script through
  long missions.
- **web_fetch URL** pulls one http(s) address as readable text (HTML
  stripped to prose, JSON pretty-printed, 2 MB download cap). Fetch,
  not search — and deliberately NOT read_only: it reaches the network
  from outside every sandbox wall, so it gates like bash and a human
  approves the address.
- **bash_start / bash_poll / bash_kill** run commands that outlive one
  tool call — dev servers, watchers, long builds — with output teeing
  to `.akshara/jobs/<id>.log`. Jobs always run as plain env-scrubbed
  subprocesses (they outlive any sandbox), so start/kill gate even when
  confined bash doesn't.
- **read_image PATH** lets the model LOOK at a png/jpeg/gif/webp in
  its sandbox — screenshots, diagrams, charts it just generated. The
  image rides history right after the tool result on all three wire
  dialects ([notes/27](notes/27-read-image.md)).
- **browser_open / browser_click / browser_fill / browser_close**
  drive a real headless Chromium (`[browse]` optional extra:
  `uv sync --extra browse && uv run playwright install chromium`).
  JavaScript runs, so JS-rendered apps work where web_fetch sees an
  empty shell; every action returns readable prose plus numbered
  element refs (`[e1]`, `[e2]`, …) harvested from the live DOM, and
  clicks/fills take a ref and return the refreshed page. Same egress
  rule as web_fetch — all four gate. Installing the extra IS the
  opt-in: the four register only when playwright is present
  ([notes/28](notes/28-browser-tools.md)). Logins persist too: set
  `AKSHARA_BROWSER_PROFILE=~/.local/state/akshara/browser-profile`
  and run `uv run akshara --browse-login <url>` once — a visible
  window opens, you sign in yourself (2FA included), close it — and
  every later session starts signed-in. Cookies never enter model
  context: the profile holds them on disk, outside the conversation.

REPL commands: `/help /model /provider /tools /history /usage /save /load
/compact /clear /image /build /quit` (`//text` sends a literal leading slash; a
trailing `\` continues the same message on the next line — paste-friendly
multi-line input that keeps indentation). `/tools` lists the toolset;
`/tools off|on NAME|GLOB` pulls tools out (and back) mid-session.
Ctrl-C cancels the current turn, not the session. `/build TASK` runs a child
builder agent in its own workspace and reports BUILD GREEN/RED without
touching this session's history. `/save`+`--resume` persist
sessions to `.akshara/session.sqlite3` (append-only versions);
`/compact` force-clears context pressure — auto-compaction also fires by
itself at 80% of the window (`--context-window` to set it; the web UI's
meter shows the same number live).

Cost accounting ([notes/21](notes/21-cost-accounting.md)): the turn
footer and `/usage` show approximate dollars from a built-in list-price
table (current Claude + GPT slugs, snapshot-dated), summed over
per-model buckets so mid-session model switches price correctly. An
unknown slug shows NO figure — never `$0`. Prices drift; point
`AKSHARA_PRICES` at a JSON file to override or extend:

```json
{"my-model": {"input": 3.0, "output": 15.0},
 "vendor/slug*": {"input": 1.0, "output": 2.0, "cached_read": 0.1}}
```

The billing convention underneath: usage counters are disjoint
(`input_tokens` counts full-rate tokens only — OpenAI-dialect adapters
subtract cached hits out of the wire's prompt total), so cached tokens
are never billed twice, and `Usage.window_tokens()` is what fills the
context window.

Library use:

```python
from akshara import Agent, allow_read_only, default_registry, get_provider, load_settings

agent = Agent(get_provider("anthropic", load_settings("anthropic")),
              model="claude-sonnet-4-5", tools=default_registry(),
              permissions=allow_read_only)          # read-only tools run free

# opt in to sub-agents: two objects, wired to each other
from akshara.subagent import SpawnSubagent, SubagentSpawner
spawner = SubagentSpawner(agent)                    # per-session budget lives here
agent.registry.register(SpawnSubagent(spawner))     # the model now sees the tool

print(agent.run("what's in README.md?").message.text())

# async: one event loop, many independent conversations
import asyncio
from akshara.async_agent import AsyncAgent

async def main():
    provider = get_provider("anthropic", load_settings("anthropic"))
    agents = [AsyncAgent(provider, model="claude-sonnet-4-5")
              for _ in range(4)]
    replies = await asyncio.gather(*(a.run(q) for a, q in zip(agents, QUESTIONS)))

asyncio.run(main())
```

Demos: [`examples/one_shot.py`](examples/one_shot.py) (request JSON → raw
response JSON → normalized response), [`examples/stream_demo.py`](examples/stream_demo.py)
(raw SSE events), [`examples/tool_round_trip.py`](examples/tool_round_trip.py)
(a full agent turn), [`examples/agent_loop_demo.py`](examples/agent_loop_demo.py)
(the loop itself, event by event — press Ctrl-C mid-turn to watch the
resumable-history invariant recover),
[`examples/tiny_mcp_server.py`](examples/tiny_mcp_server.py) (a minimal
MCP server in pure stdlib — both sides of both transports),
[`examples/async_demo.py`](examples/async_demo.py) (N conversations
sequential vs one-event-loop concurrent, with the speedup measured live),
[`examples/builder_demo.py`](examples/builder_demo.py) (the real-world
test: the agent builds a project from a spec — or repairs a seeded
broken one without touching its checksummed tests — and the demo
independently re-verifies; exit code doubles as a CI gate),
[`examples/cache_demo.py`](examples/cache_demo.py) (prompt-cache hit,
measured live), [`examples/hooks_demo.py`](examples/hooks_demo.py)
(watch every tool execution without touching the loop).

## Architecture

Normalization happens in exactly ONE layer: the provider adapters.
Internal types are the only representation the rest of the program sees.

```
src/akshara/
├── types.py        shared vocabulary: Message/Block/ToolCall/ToolResult, StreamEvent union
├── errors.py       ProviderError family (terminal for the turn) vs ToolError
│                   family (become data the model reads) vs UserUnavailable
│                   (control-flow BaseException: nobody home to ask)
├── config.py       env vars -> ProviderSettings (+ .env auto-load)
├── agent.py        THE LOOP: model -> tool calls -> results -> repeat; optional
│                   per-turn tool selection (top-K sent; exact-name calls admitted);
│                   interrupt_check hook — hosts cancel mid-stream, same unwind as Ctrl-C
├── async_agent.py  the loop's async twin: same rules, awaited -- one event
│                   loop drives K independent conversations ([notes/11](notes/11-async.md));
│                   batch width capped by max_parallel_tools (semaphore inside
│                   the workers -- gather starts all, runs N-wide)
├── builder.py      first-class build mode: BuildSpec -> seeded workspace ->
│                   agent turns -> INDEPENDENT re-verification + test-file
│                   checksums -> BuildResult.ok as a CI gate ([notes/18](notes/18-builder-mode.md))
├── sandbox.py      ToolSandbox protocol + two backends: SubprocessSandbox
│                   (scrubbed env, legacy semantics) and BwrapSandbox (bubblewrap:
│                   no net/fs/pid escape) + autodetect ([notes/16](notes/16-sandboxing.md))
├── subagent.py     agent-as-tool: fresh child Agent per spawn -- budget,
│                   one level deep, compact results, optional stream tee
│                   ([notes/08](notes/08-sub-agents.md))
├── permissions.py  PermissionRequest + gates: allow_read_only / yolo /
│                   deny_all / trust_sandbox (auto-approves bash ONLY while
│                   confined); SwitchableGate flips ask ⇄ yolo mid-session;
│                   approve-with-edits: a gate may rewrite arguments
│                   pre-approval ([notes/20](notes/20-approve-with-edits.md))
├── context.py      compaction: mask old tool results, then summarize (red
│                   zone) -- sync + async twins share all the arithmetic
├── leases.py       TTL leases for shared resources -- parallel batch writes
│                   to one path serialize instead of racing
├── session.py      SQLite checkpoints: append-only versions, /save /load --resume
├── mcp.py          MCP client, hand-rolled JSON-RPC over stdio AND
│                   Streamable HTTP (SSE responses via providers/sse.py):
│                   handshake, tools/list, tools/call; MCPManager adds/
│                   removes/toggles servers mid-session, .akshara/mcp.json
│                   remembers them ([notes/09](notes/09-mcp.md))
├── evals.py        trajectory evals: completion/correctness/process/cost,
│                   recording tool proxy, LLM judge; AsyncEvalRunner twin
│                   runs cases concurrently, shared scoring ([notes/10](notes/10-evals.md))
├── pricing.py      list-price table -> $ figures: slug matching (exact /
│                   date-suffix / vendor-prefix / family), per-model session
│                   buckets, AKSHARA_PRICES overrides; unknown = no figure,
│                   never a guess ([notes/21](notes/21-cost-accounting.md))
├── providers/
│   ├── base.py     Provider ABC + collect()/acollect(): stream events ->
│   │               ModelResponse (protocol cores shared by both skins);
│   │               cache_control opt-in lives here ([notes/13](notes/13-caching.md))
│   ├── retry.py    retry: backoff+jitter, budgets, Retry-After -- two notches:
│   │               the OPENING freely retried, a 200 stream that dies BEFORE
│   │               its first event re-opened under the same budgets; after
│   │               ANY event forwarded, never (sync + async twins)
│   ├── fallback.py opening-only failover across providers (retry fixes
│                   time problems, fallback fixes place problems)
│   ├── sse.py      hand-rolled SSE framing (incremental UTF-8, CRLF/CR/LF, comments)
│   ├── anthropic.py  Messages-dialect adapter (encode request / decode stream)
│   ├── openai.py     chat-completions-dialect adapter behind the same interface
│   ├── responses.py  Responses-API dialect (chat-completions' successor):
│   │                 typed input items, flat tools, NAMED SSE events ending
│   │                 [DONE] ([notes/19](notes/19-responses-api.md))
│   └── ollama.py     local models = OpenAI dialect profile (no auth, 8k window)
├── tools/
│   ├── base.py     Tool ABC (schema + summary + run -> str | ToolOutput),
│   │               ToolRegistry (runtime disable/enable — live-checked,
│   │               reversible; unregister stays the permanent kill-switch), arg validators
│   ├── fs.py       read_file / list_dir / write_file / edit_file (+ path sandbox)
│   ├── glob.py     glob — find files by NAME ('**' recursion, newest-first,
│   │               grep's skip rules); read-only so it never gates ([notes/23](notes/23-glob.md))
│   ├── shell.py    bash — delegates to any ToolSandbox (default: legacy
│   │               subprocess semantics; timeout -> killpg -> salvage)
│   ├── background.py bash_start / bash_poll / bash_kill: jobs that outlive
│   │               one tool call, log teeing to .akshara/jobs/, process-group
│   │               kill; plain subprocesses by design -> always gate ([notes/26](notes/26-background-bash.md))
│   ├── web_fetch.py fetch ONE url as readable text (stdlib HTML stripping,
│   │               2 MB cap, head-tail clip) — network egress, gated like
│   │               bash ([notes/25](notes/25-web-fetch.md))
│   ├── read_image.py the agent looking at a picture BY ITSELF: returns a
│   │               ToolOutput; the loop hoists images onto history after the
│   │               result ([notes/27](notes/27-read-image.md))
│   ├── browser.py  browser_open/click/fill/close — a real headless Chromium
│   │               behind the [browse] extra: JS-rendered pages come back as
│   │               text + numbered element refs; registers only when playwright
│   │               imports, so the tool count never moves uninvited; optional
│   │               $AKSHARA_BROWSER_PROFILE keeps logins between sessions
│   │               (--browse-login = headed one-time setup) — cookies stay on
│   │               disk, never in model context ([notes/28](notes/28-browser-tools.md))
│   ├── selector.py dynamic tool loading: BM25 ToolCatalog over name+
│   │               description, transcript-derived query, core pins +
│   │               list_available_tools discovery hatch ([notes/17](notes/17-tool-selection.md))
│   ├── search.py   grep — ripgrep subprocess when available, pure-python
│   │               walker fallback (identical output contract)
│   ├── memory.py   scratchpad: write_note / recall_notes — JSON store under
│   │               .akshara/, ranked substring retrieval, survives restarts
│   ├── todo.py     live plan state vs memory's durable facts: todo_write /
│   │               todo_read — replace-whole-list semantics ([notes/24](notes/24-todo-lists.md))
│   └── ask_user.py pause-and-ask-the-human tool: UserChannel protocol
│                   (terminal stdin, browser websocket, or None = headless
│                   fails the turn) — the model's escape hatch from guessing
│                   ([notes/22](notes/22-web-ui.md))
├── web/            FastAPI skin over the SAME sync loop: each turn runs on a
│                   worker thread pulling run_streaming(); a queue bridge
│                   carries human questions (permission + ask_user) to the
│                   browser over one websocket; envelope protocol,
│                   replay-on-reconnect, REST session controls; static/
│                   holds the no-build vanilla-JS page
│                   ([notes/22](notes/22-web-ui.md))
└── cli/            main.py (argparse) · repl.py (input loop) · render.py (rich)
```

Design rules worth stealing:

* **Errors are data.** A crashing/denied/nonexistent tool becomes an
  `is_error` tool result the model reads and recovers from; the loop
  cannot be crashed by a tool. Provider errors (auth/rate-limit/overflow)
  are the opposite: exceptions, terminal for the turn. A third family,
  `UserUnavailable`, is deliberately neither — a `BaseException` that
  fails the turn when `ask_user` runs with no human attached, because
  no error message could teach the model to conjure a user.
* **The history invariant:** every tool_call id gets a matching result
  before the next request — enforced on every exit path including
  iteration caps and mid-turn Ctrl-C ([notes/05](notes/05-agent-loop.md)).
* **Permission gate = plain callable** `Callable[[PermissionRequest], bool]`.
  Tools build their own human-readable `summary()` with the same context
  execution will get, so what you approve is what runs. Approve-with-edits
  falls out of one deliberate mutability: a gate may REPLACE
  `request.arguments` before answering True; the loop notices the swap by
  identity and adopts the edited form — so approval is a review step, not
  a rubber stamp ([notes/20](notes/20-approve-with-edits.md)).
* **Gates decide, hooks watch.** Observational `on_before_tool` /
  `on_after_tool` callbacks bracket every real execution (errors
  included) but can veto nothing — denial stays the gate's job. A
  raising hook crashes the turn loudly: hooks are developer
  infrastructure, not untrusted input ([notes/14](notes/14-hooks.md)).
* **Two channels out of a turn.** Raw StreamEvents (text/thinking
  deltas) are *pushed* to `agent.on_stream_event` while each response
  streams (collect() owns the pull, so they can't be yielded);
  ToolExecuted/TurnEnd are *yielded* from `run_streaming()`. Consumers
  pull, UIs subscribe ([notes/05](notes/05-agent-loop.md)).
* **Retry the opening, never the stream.** 429/5xx/connection errors
  back off with jitter up to hard budgets; once one event reached the
  caller there is no safe replay ([notes/07](notes/07-reliability-and-scale.md)).
* **Gates sequential, execution parallel.** y/n prompts own the
  terminal; approved batches run in a thread pool (or as asyncio tasks)
  but results yield in submission order. Ctrl-C mid-batch records REAL
  results — workers finish during the join, and the work happened
  (async needs `asyncio.shield` for this: a bare `await gather`
  forwards cancellation INTO its children and loses finished work,
  [notes/11](notes/11-async.md)).
* **Mask before summarize.** Compaction first elides old tool-result
  OUTPUT (reversible, calls stay verbatim); only if still in the red
  zone does an LLM summarize the middle — goal message intact, every
  tool call accounted for ([notes/07](notes/07-reliability-and-scale.md)).
* **Tool results are strings — until pixels.** A tool may return a
  `ToolOutput` (text + images); the loop splits it and HOISTS the
  images onto history after the results, because two of the three wire
  dialects cannot carry an image inside a tool payload at all. One
  transcript shape everywhere; adapters never see the field
  ([notes/27](notes/27-read-image.md)).
* **Sub-agents: constrained in code, not prompts.** One level deep
  (`spawn_subagent` rejected from child catalogs), bounded (per-session
  budget + mandatory justification), compact results (conclusions +
  cost metadata, never transcripts). Scope restriction lives at the
  ToolRegistry level; permissions are inherited so a sub-agent cannot
  escalate by being a sub-agent ([notes/08](notes/08-sub-agents.md)).
* **Sync generators everywhere** — blocking tools, blocking REPL, and
  cancellation for free (generator close unwinds into socket cleanup).
* **Async = cores + skins, not a rewrite.** Protocol logic lives in
  incremental sync classes (SSE framing, stream routers, response
  folding, compaction arithmetic); `acollect`/`astream`/`AsyncAgent`
  are thin awaited skins over the same rules — zero duplicated wire
  logic. Tools keep ONE blocking implementation; the async loop pushes
  them to threads via a default `arun()` (`to_thread`) instead of
  pretending syscalls are non-blocking. One event loop drives K
  independent conversations ([notes/11](notes/11-async.md)).

## The wire cheat-sheet

The whole reason three adapters exist. Same conversation, three encodings:

| Concern | Anthropic Messages | OpenAI chat-completions | OpenAI Responses |
|---|---|---|---|
| Auth | `x-api-key` + `anthropic-version: 2023-06-01` | `Authorization: Bearer` | `Authorization: Bearer` |
| Endpoint | `POST {base}/v1/messages` | `POST {base}/chat/completions` | `POST {base}/responses` |
| System prompt | top-level `"system"` field | `messages[0] role:"system"` | top-level `"instructions"` field |
| max_tokens | **required** | optional | `max_output_tokens` (optional) |
| History shape | block-shaped messages | role messages; user turns fan out into `role:"tool"` messages | ONE flat `input[]` of TYPED items (`message`, `function_call`, `function_call_output`) |
| Tool definition | `{name, description, input_schema}` | `{type:"function", function:{name, description, parameters}}` | FLAT: `{type:"function", name, description, parameters}` |
| Tool calls in reply | content blocks `type:"tool_use"` (input = object) | `message.tool_calls[]` (arguments = JSON **string**) | `output[]` items `type:"function_call"`, keyed by `call_id` (arguments = JSON **string**) |
| Sending results back | next `role:"user"` msg w/ `tool_result` blocks | one `role:"tool"` message per call | one `type:"function_call_output"` item per result (echoes `call_id`) |
| Stop signal | `stop_reason: end_turn\|tool_use\|max_tokens...` | `finish_reason: stop\|length\|tool_calls\|content_filter` | `status: completed\|incomplete` (+ any `function_call` item ⇒ tool_use; `incomplete_details.reason: max_output_tokens`) |
| Model reasoning | content blocks `type:"thinking"` + opaque `signature` — MUST round-trip verbatim in tool loops; `type:"redacted_thinking"` (ciphertext) has the same contract and arrives whole in streams | `reasoning` / `reasoning_content` fields — display-only | `reasoning` output items / summary deltas — display-only |
| Stream shape | named events (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`, `ping`) | anonymous chunks ending `data: [DONE]`; empty-choices chunks carry usage | named events (`response.created`, `response.output_text.delta`, `response.function_call_arguments.delta`, terminal `response.completed` carrying usage) ending `data: [DONE]` |

Streaming details both formats share: arguments arrive as **fragments**
that must accumulate keyed by stream index and parse exactly once at the
end ([notes/03](notes/03-sse-and-collect.md)).

## Run & test

```bash
uv run pytest -q                 # full offline suite: 681 tests, NO network, NO key

# everything below makes REAL model calls -- it needs a key in .env (auto-loaded):
uv run python examples/one_shot.py "Why is the sky blue?"
uv run python examples/one_shot.py --provider openai "Why is the sky blue?"
uv run python examples/stream_demo.py "Count to five"
uv run python examples/tool_round_trip.py "What's in README.md?"
uv run python examples/agent_loop_demo.py            # the loop, event by event
uv run python examples/agent_loop_demo.py --deny-all # denial-as-data demo
uv run python examples/async_demo.py                 # 4 conversations, seq vs concurrent
uv run python examples/builder_demo.py               # agent BUILDS a project, verified
uv run akshara                                       # REPL
./start.sh                                           # same thing, via menus/presets (see "One-command starts")
uv run akshara --provider ollama --web               # REPL in your browser (free, local)
uv run akshara --yolo "run: echo hi"                 # one-shot, no prompts
uv run akshara --cache                               # prompt caching on
uv run python examples/cache_demo.py                 # cache hit, measured live
uv run python examples/hooks_demo.py                 # watch tool executions live
```

The offline suite never touches the network: adapters run against
byte-exact SSE/JSON fixtures via `httpx.MockTransport`; the loop runs
against `ScriptedProvider`. The integration suite
(`tests/test_integration.py`) drives real adapters through a full
two-iteration tool turn per provider and asserts each wire's
result-encoding shape on the second request.


## Tested

`uv run pytest -q` — 699 offline tests against byte-exact SSE/JSON
fixtures (`httpx.MockTransport`) and a `ScriptedProvider` loop: no
network, no key. Retries are exercised offline too, against flaky
mock transports whose policy path is identical to the live one. The
browser UI runs fully offline as well: FastAPI's test client drives the
real routes against scripted turns — and web_fetch does the same trick
with a mocked transport behind its real request path. The browser_*
family goes further and stays green in BOTH worlds: without playwright
(fakes carry the session) and with the extra synced.

Everything above has also been exercised against real providers —
all three dialects via OpenRouter (cloud models) plus local Ollama
for end-to-end runs of sandboxing, build mode, MCP, sub-agents,
evals, caching, hooks, vision, `ask_user`, the web UI, and the
self-reliance set (background jobs + a localhost web_fetch + todo
tracking on `qwen3.8`; `read_image` vision on `gemma4:12b` — see the
receipts closing notes/24–27), and the browser_* family driving a
JavaScript-rendered local page end to end on `qwen3.8` — a page
web_fetch provably cannot read ([notes/28](notes/28-browser-tools.md)). The
`examples/` demos rerun most of it on demand. Live testing shook out
four real bugs along the way, all fixed and now regression-tested:
gateway `null` token counters poisoning `Usage.add()`, an orphaned
child process when Ctrl-C lands mid-bash, thinking blocks that must
round-trip verbatim through tool loops (including unsigned ones behind
a gateway that still validates the field), and an inverted
yolo-warning guard in the banner ([notes/02](notes/02-wire-formats.md),
[notes/04](notes/04-tools.md), [notes/06](notes/06-cli.md)). Building
the web layer shook out three more: SQLite connections used from the
server thread without `check_same_thread=False` (every save 500'd) and
a permission gate that popped approval modals for read-only tools —
both caught by its tests before they could ship — plus one that did
ship and was caught in use: the context meter pegged at 100% on small
local windows, where the reply budget exceeded an 8k ollama window and
clamped usable space to one token ([notes/22](notes/22-web-ui.md)).
