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
`/api/save`, `/api/load`, `/api/compact`, `/api/clear`), and mutating
ones refuse to run mid-turn (409) — the REPL serves slash commands
between turns too; same single-operator assumption. The one deliberate
exception is `/api/permissions`: flipping ask ⇄ yolo mid-turn is safe
(the loop consults the gate per call) and is exactly how you rescue a
turn stuck in approval modals. The top-bar mode chip shows the current
mode — red while yolo — and clicking it flips; the endpoint broadcasts
a fresh `state` so other open tabs follow along. Same switch backs the
REPL's `/yolo` ([06-cli.md](06-cli.md)).

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

## Cancel is Ctrl-C, at the same three places

The Cancel button sets a flag honored where the terminal honors SIGINT:
between yielded loop events (the generator is closed, outstanding calls
synthesized, history stays valid) and while blocked on a human
question. An in-flight MODEL call still can't be interrupted — workers
get no signals — so the cancel lands at the next checkpoint. Verified
live below, including the case where it *doesn't* feel instant.

## What the tests pin

41 new offline tests, no network, no key:

* `test_ask_user.py` (17): free-text and choice answers reach history
  with the `(picked option k/N)` marker; five arg-validation ToolErrors;
  no-channel fails the turn AND the session survives to a clean next
  turn; a batch-mate running beside a failed ask keeps its real result;
  `read_only=True` never gates; TerminalChannel choice/free-text/
  empty-reprompt/EOF behaviors.
* `test_web_server.py` (14): connect handshake (state, pending question,
  replay); streaming deltas; deny becomes error data; the full
  edit→re-summary→approve round-trip adopting edited args into history;
  ask answered over the wire; cancel during a pending ask, then the
  session serves another turn; 409 on concurrency; 400 validations;
  save/load with injectable provider factories — plus the strict path:
  restoring a checkpoint whose provider has no key fails cleanly and
  leaves the live agent untouched.
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
4. **Honest miss**: firing cancel right after the first text delta did
   NOT stop a 300-word essay — the in-flight model call ran on and the
   turn completed. Exactly the documented limit; cancel waits for the
   next checkpoint, same as the terminal.
