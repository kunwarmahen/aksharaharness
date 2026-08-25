# 22 — The web UI: a second skin over the same loop

*`--web` turns the terminal harness into a local browser app — without
forking the agent loop. Every hard problem (gates, cancellation,
resumable history) was already solved once; the web layer only builds a
new front door to the same rooms.*

## The one rule: the loop stays sync

The agent loop is a blocking generator ([notes/05](notes/05-agent-loop.md)).
The tempting rewrite is to port it to asyncio so it "fits" the server.
That would be a second harness to maintain, with subtle differences from
the terminal one. So instead: **each turn runs on an ordinary worker
thread** pulling the very same `run_streaming()` generator the REPL
pulls. Gates, parallel batches, Ctrl-C semantics, the history invariant —
all inherited free, because nothing about them changed.

What remains is crossing the border between that worker thread and the
server's event loop, and it is crossed with boring parts only:

* **Inbound** (human → agent): each open question owns a
  `queue.Queue`. The worker blocks in `get(timeout=0.2)` — polling
  rather than blocking forever, because the timeout tick is also where
  it notices the cancel flag. The async side drops answers in with
  `put_nowait`; it never waits on the worker.
* **Outbound** (agent → browsers): events fan out to every connected
  tab via `loop.call_soon_threadsafe`, captured per connection. A dead
  tab's failing dispatch is swallowed; `disconnect()` reaps it.

No locks around the agent (one turn at a time, enforced with a 409 on
concurrent requests), no shared mutable state across threads beyond the
client list (guarded) and the pending-question slot (single operator).

## One bridge, two questions

The permission prompt and `ask_user` turned out to be the *same
mechanism*: broadcast a question, block for an id-tagged answer, clean
up. A `_Pending` object carries whichever kind it is; the browser
answers `{type: "answer", id, ...}` and doesn't need to know which.
Cancellation pushes a sentinel into the same queue and the waiter
raises `KeyboardInterrupt` — literally the Ctrl-C exception, so the
turn unwinds through the exact code path the terminal uses.

Two details worth stealing from the terminal gate, because parity is
the product:

* **Read-only never prompts.** First line of the web gate:
  `if request.read_only: return True`. Without it, every `read_file`
  and every `ask_user` pops a needless approval modal — this bit me
  immediately once the tests wired the real gate in.
* **Edit round-trips.** Choosing *edit* sends new args; the server
  swaps them into `request.arguments`, re-renders the preview through
  the tool's own `summarize` (the seam from [notes/20](notes/20-approve-with-edits.md)),
  and re-asks tagged `edited: true`. What you approve in the browser is
  what runs, and history keeps the edited form.

`WebSession` also *is* the `UserChannel`: its `.channel` property
returns itself, since it already implements `ask()` over the same
bridge. One plumbing system serves both interactive surfaces — the same
trick as handing `SpawnSubagent` its spawner object.

## Everything is envelopes

One websocket (`/ws`) carries tagged JSON both ways; the page is a dumb
renderer over it.

| Server → browser | Meaning |
|---|---|
| `state` | provider/model/tools/usage/utilization snapshot |
| `start` · `delta` · `thinking_delta` | a response is streaming |
| `tool_start` · `tool_result` | card opened / filled (args, output, error?) |
| `permission_request` | approve / deny / edit, with the tool's own summary |
| `ask` | ask_user question + labeled choices + context |
| `resolved` | the question with this id left the screen (any outcome) |
| `turn_end` · `turn_cancelled` · `turn_error` | how the turn ended, honestly |
| `turn_started` · `turn_done` | bracket every turn |

Browser → server is tiny: `answer {id, decision \| text}` and `cancel`.

Because the transcript is rebuilt from `history_envelopes()` on every
connect, a refresh or a second tab rejoins mid-conversation: state,
any still-pending question, then a replay rendered in the same shapes
as the live feed. Plain request/response controls stay REST
(`/api/message`, `/api/model`, `/api/provider`, `/api/permissions`,
`/api/tools`, `/api/save`, `/api/load`, `/api/compact`, `/api/clear`),
and mutating ones refuse to run mid-turn (409) — the REPL serves slash
commands between turns too; same single-operator assumption. Two
deliberate exceptions skip that guard:

* `/api/permissions` — flipping ask ⇄ yolo mid-turn is safe (the loop
  consults the gate per call) and is exactly how you rescue a turn
  stuck in approval modals.
* `/api/tools` — `GET` lists every registered tool with its read-only
  flag and on/off state; `POST {"name": ..., "enabled": ...}` flips
  one. Safe for the same reason: the registry's disabled set is
  consulted LIVE at every layer ([17-tool-selection.md](17-tool-selection.md)),
  so pulling a tool mid-turn just makes its *next* call fail as data
  ("disabled by the operator"), which the model reads and routes
  around.
* `/api/mcp/toggle` — the same soft switch pointed at a whole MCP
  server's toolset (`MCPManager.set_enabled`, process stays warm);
  skips the idle guard for exactly the same live-consulted reason.

The rest of the MCP family DOES take the guard, deliberately:
`POST /api/mcp/add` spawns child processes and registers tools (and
`/api/mcp/remove` closes them) — mutating the registry mid-batch would
race iteration, so those two are 409 while a turn runs, like every
other mutator. `add` accepts either form fields (`name` +
`command`+`args`+`env`, or `url` for Streamable HTTP) or a pasted JSON
blob in the same shape as `--mcp-config`
(`{"servers": {...}}`) parsed server-side; each entry reports its own
result inside `results[]` — one bad server is an `ok:false` row, not a
500 hiding the three that worked. A checked-by-default "remember"
saves the entry to `.akshara/mcp.json` for auto-reconnect on future
launches; remove always forgets it ([09-mcp.md](09-mcp.md)).

All of these broadcast a fresh `state` so other open tabs follow
along. The top-bar mode chip shows the permission mode — red while
yolo — and clicking it flips; the ⚙ servers & tools chip opens a
panel holding both halves: the MCP rows first (health dot, transport
badge, `saved` tag, per-server switch, a two-step remove button whose
first click arms "sure?" instead of trusting a native dialog), then
the tool switches with a count of how many are off. Same objects back
the REPL's `/yolo`, `/tools off|on`, and `/mcp` commands
([06-cli.md](06-cli.md)). Sessions without an MCP manager (the
library path) simply hide the servers half.

The `state` snapshot also carries the context-pressure numbers
(`context_tokens`, `context_window`, an `estimated` flag for when only
the chars/4 fallback is available), and every `tool_result` envelope
carries fresh utilization — so the meter in the header updates during
a turn, not just between turns. The bar turns amber at 60% and red at
80%, mirroring exactly where auto-compaction starts caring
([07-reliability-and-scale.md](07-reliability-and-scale.md)); hovering
shows the raw numbers.

## Rendering the model's prose

Models answer in markdown — headers, tables, fenced code — and a
transcript that shows those literally is failing at its one job. The
fix is `static/md.js`, ~150 lines of hand-rolled renderer with no
library and no build step (the page can't fetch a CDN anyway; the CSP
forbids it, same rule that keeps the core deps at httpx + rich).

The security model is stated once in its header and enforced
everywhere: model text is HTML-escaped FIRST, then wrapped in tags we
generate ourselves. `<script>` arrives as visible text; link URLs must
look like `http(s)`/`mailto` or they degrade to plain text. The parser
covers what models actually emit — headings mapped two levels down (a
chat message isn't a document outline), nested lists by indentation,
pipe tables, fenced code, blockquotes — and deliberately not images.

Streaming shapes the integration more than parsing does: deltas
re-render the whole message but throttled to one pass per animation
frame, and a partial document (half a table, an open fence) simply
renders as far as it got; `turn_end` carries the complete text and its
final render settles any artifact. Ten offline tests execute md.js
under node (`test_md_renderer.py`, skipped when node is absent) — the
escaping rules are pinned hardest, because they're the part that must
never regress.

A war story from the REST side: `SessionStore` was built on the main
thread, but FastAPI handlers execute on the event-loop thread — and
SQLite connections are thread-affine by default, so every `/api/save`
500'd until the store learned `check_same_thread=False` plus one lock
around writes. The REPL never hit this because it is single-threaded;
the web skin's first gift to the core was finding that latent bug.

## When nobody is home

`ask_user` needs a human; headless runs have none. The channel is
registered as `None` whenever stdin/stdout aren't a TTY (piped runs,
cron, evals), and a call raises `UserUnavailable` — deliberately a
`BaseException`, so the loop's errors-are-data conversion cannot turn
"no user" into a tool result the model would explain away with a guess
([tools/ask_user.py](../src/akshara/tools/ask_user.py),
[errors.py](../src/akshara/errors.py)). The turn fails loudly, exit
code 1, history stays resumable. Receipt below.

## Cancel is Ctrl-C, at every checkpoint including mid-sentence

The Cancel button sets a flag honored everywhere the terminal honors
SIGINT — between yielded loop events (the generator is closed,
outstanding calls synthesized, history stays valid), while blocked on a
human question, and now **during the model call itself**: the session
hands the agent an `interrupt_check` callback (it reads the cancel
flag), and the loop polls it between stream events inside `_tee`.
Worker threads get no signals, so this is cooperation, not preemption —
but a healthy connection yields deltas constantly, so the stop lands
milliseconds after the click instead of at turn's end. The honest edge:
a provider gone silent sends nothing to poll between, so the worst case
is still "next checkpoint". Either way the unwind is the same code path
Ctrl-C takes, history resumable ([05-agent-loop.md](05-agent-loop.md)).
In the page, Esc does the same as the ■ button whenever a turn is
running and no modal owns the keyboard.

## What the tests pin

52 new offline tests, no network, no key:

* `test_ask_user.py` (17): free-text and choice answers reach history
  with the `(picked option k/N)` marker; five arg-validation ToolErrors;
  no-channel fails the turn AND the session survives to a clean next
  turn; a batch-mate running beside a failed ask keeps its real result;
  `read_only=True` never gates; TerminalChannel choice/free-text/
  empty-reprompt/EOF behaviors.
* `test_web_server.py` (33): connect handshake (state, pending question,
  replay); streaming deltas; deny becomes error data; the full
  edit→re-summary→approve round-trip adopting edited args into history;
  ask answered over the wire; cancel during a pending ask, then the
  session serves another turn; 409 on concurrency; 400 validations;
  save/load with injectable provider factories; the strict restore path
  whose provider has no key fails cleanly and leaves the live agent
  untouched — plus the newer pins: tools listing with detail, toggle
  validation errors, a disabled tool's call failing as data MID-turn
  (an `ask_user` parks the worker so the flip lands deterministically),
  state broadcasts reaching a second open tab, `interrupt_check` wired
  to the cancel flag, pressure numbers in `state`, and the MCP family:
  list/add/remove round trip over a fake connector, whole-server toggle
  allowed mid-turn while add/remove 409, connection failure arriving as
  an `ok:false` row rather than a 500, mixed good+bad JSON-mode results,
  field-shape validations, and the remember flag persisting only what
  was kept.
* `test_md_renderer.py` (10): the page's markdown renderer executed by
  node — escaping above all (model HTML stays text), emphasis/code/
  strike, heading mapping, fenced code, tables, nested and ordered
  lists, scheme-checked links, `<br>` paragraphs, blockquotes/rules.

## Live receipt

All against local Ollama `qwen3.8` (27B, Q4):

1. **Headless fail**: `uv run akshara --provider ollama --model qwen3.8
   "Write a poem about my favorite editor..."` — the model searched its
   notes first (`recall_notes` came up empty), then called `ask_user`,
   and the run ended: `turn failed: ask_user ran with no interactive
   user attached (headless run). Failing the turn rather than guessing.`,
   exit 1, no hang.
2. **REPL ask**: asked it to learn my favorite language before writing.
   It offered numbered choices (`[1-5, or your own answer]>`), I typed
   `2`, the tool result read `user replied: (picked option 2/5) Rust`,
   the gated `write_file` panel followed, `y` approved, and the haiku
   landed in `lang.txt` — end_turn after 3 iterations.
3. **Browser drive**: served `--web`, drove the real websocket. The
   `ask` envelope arrived with five editor choices; answering "Kakoune"
   produced `(picked option 1/5) Kakoune` in the tool card; the
   `write_file` approval showed a genuine diff preview (the file
   existed); approve wrote the sentence; the footer read
   `$0.00 (local model)`. Sending `cancel` while the modal was open gave
   `resolved` → `turn_cancelled`, and the next message ran clean.
4. **Honest miss → fixed**: the first time around, firing cancel right
   after the first text delta did NOT stop a 300-word essay — the
   in-flight model call ran on to completion, exactly as this note then
   documented as a limit. That miss is what bought the cooperative
   interrupt above: re-running the same experiment now stops within a
   delta or two of the click, with the partial reply left on screen and
   history clean for the next message.
