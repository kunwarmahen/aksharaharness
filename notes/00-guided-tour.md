# 00 · The guided tour — what this project is and how it works

> Written for someone who has never built an AI tool. No code-reading
> required — every idea here gets a picture and a plain-English story.
> The deeper write-ups live in [01](01-anatomy-of-a-request.md)–[13](13-caching.md).

---

## 1 · What are we building?

You have used a chatbot: you type, it types back. Useful, but it can
only *talk*. An **agent** is a chatbot that can also *act* — read your
files, run commands, fix a bug — while you supervise.

Claude Code, the thing you may be using right now, is an agent. So is
this project. We call the machinery around the model a **harness**:

```
      a chatbot                        an agent

  ┌───────────────┐              ┌────────────────────────┐
  │   you  ⇅  LLM │              │   you  ⇅  HARNESS  ⇅   │
  │               │              │        ┌───────┐  LLM  │
  │  talk only    │              │     files    shell     │
  │               │              │     search   editor    │
  └───────────────┘              │     + permission gate  │
                                 └────────────────────────┘
                                  acts on your computer,
                                  asks before doing anything risky
```

The harness is the difference between "a thing that describes what
*could* be done" and "a thing that does it — carefully."

## 2 · The one idea everything else rests on

Here is the whole secret, and it's smaller than you'd expect:

> **The model is a text-in, text-out function. Nothing more.**

It has no memory of yesterday. It cannot see your files. It cannot run
anything. It cannot even take its own advice. Every turn, we mail it a
transcript of the conversation so far, and it mails back text. That's it.

So how does Claude Code edit files? Because *we wrote the hands*.
The model can only **ask**: "please run this command." The harness
decides whether to allow it, runs it if allowed, pastes the result into
the transcript, and mails the whole thing back. From the outside it
looks like intelligence acting in the world; up close it is a very
persuasive pen pal with a very diligent postal service.

That postal service is what this repository builds.

## 3 · The cast of characters

One folder per job (`src/akshara/`):

```
 ┌─────────────────────────── YOU ───────────────────────────┐
 │                                                           │
 │   cli/            the dashboard: reads your typing,       │
 │    main·repl·render    draws pretty panels                │
 │        │                                                  │
 │        ▼                                                  │
 │   agent.py        THE LOOP (the engine — see §4)          │
 │    │      ▲                                               
 │    ▼      │                                               │
 │   providers/      translators (see §6)                    │
 │    anthropic · openai · sse                               
 │        │                                                  │
 │        ▼            🌍 internet 🌍                        
 │                 (the actual model lives far away)         
 │                                                           
 │   tools/          the hands: read_file, bash, grep …      
 │   ask_user.py     when the agent needs to ask YOU something
 │   permissions.py  the seatbelt: ask y/n/e before danger   
 │   web/            a browser window onto this same loop    
 │   types.py        the shared vocabulary everyone speaks   
 │   config.py       finds your API key in .env              
 └───────────────────────────────────────────────────────────┘
```

Two of these deserve a name-check now:

* **`types.py` — the vocabulary.** Internally, every message is stored
  in ONE neutral shape (text blocks, tool requests, tool results).
  Nobody downstream knows or cares which company's model was used.
* **`providers/` — the translators.** Each AI company expects messages
  in its own format, like countries driving on different sides of the
  road. Adapters translate our neutral shape out to a company's format,
  and their replies back in. Everything else in the program is
  blissfully unaware this translation happens.

## 4 · The heart: one loop, four steps

The entire agent is this circle. Everything else exists to serve it.

```
            ┌──────────────────────────────────────────┐
            │                                          │
            ▼                                          │
   1. ASK        send the whole transcript to the      │
                 model: "here's the chat so far…"      │
            │                                          │
            ▼                                          │
   2. READ       stream its reply word by word.        │
                 Usually just an answer → DONE ✋      │
                 Or… a request: "run `ls` for me"      │
            │                                          │
            ▼                                          │
   3. ACT        seatbelt check (y/n/e?) →             │
                 actually run the command              │
            │                                          │
            ▼                                          │
   4. RECORD     paste the result into the             │
                 transcript as if the user said it     │
            │                                          │
            └──────── back to step 1 ──────────────────┘
```

Why the loop instead of one pass? Because real work is multi-step:
*"find the file that defines Agent → open it → find the bug → fix it"*.
Each step needs the previous step's result, and only the model can
decide what the next step is. So we bounce: model proposes, we
execute, model reads the result and proposes again. A safety valve
caps this at **25 bounces** per request so it can never spin forever.

The model signals which way the reply goes with a simple flag:
`tool_use` means "I need an action performed"; `end_turn` means "here
is my final answer, no actions needed."

## 5 · A full round trip, narrated

You type: *"What's in README.md?"* Here is every message that flies,
with the transcript (the "filing cabinet") growing each time:

```
 YOU      "What's in README.md?"
   │
   ▼
 ┌─ TRANSCRIPT after step 1 ────────────────────────────┐
 │ user: "What's in README.md?"                         │
 └──────────────────────────────────────────────────────┘
   │  mailed to the model
   ▼
 MODEL    "I should look. Please run read_file('README.md')"
          (flag: tool_use)
   │
   ▼
 HARNESS  Is read_file dangerous? It's read-only → allow.
          Runs it. Gets 4,000 characters of file content back.
   │
   ▼
 ┌─ TRANSCRIPT after steps 3–4 ─────────────────────────┐
 │ user:      "What's in README.md?"                    │
 │ assistant: [request: read_file('README.md')]         │
 │ user:      [result: "# AksharaHarness — a from-scratch…"]   │
 └──────────────────────────────────────────────────────┘
   │  mailed to the model again
   ▼
 MODEL    "It's a from-scratch agent harness with streaming,
          tools and a permission gate…"  (flag: end_turn) ✋

 Loop exits. The answer streams onto your screen.
```

Three details worth savoring:

1. **Tool results are pasted in as if the *user* said them.** That's
   not laziness — it's the wire protocol. The model only ever receives
   two kinds of mail: user turns and its own past turns. Results ride
   inside user turns.
2. **The model never touches your disk.** It emits a *request*, as
   data. Our code decides whether to honor it. This is the wall that
   makes supervision possible.
3. **Every bounce costs a full re-mailing of the transcript.** The
   model has no memory between calls — the transcript IS the memory.

## 6 · Two languages, two translators

Anthropic and OpenAI describe identical ideas with different words.
A taste of the same sentence in both dialects:

| Idea | Anthropic-speak | OpenAI-speak |
|---|---|---|
| "who are you really?" | separate top-level `system:` field | first message says `role:"system"` |
| "please run X" | a block named `tool_use`, arguments as a real object | an entry named `tool_calls`, arguments as a JSON **string** |
| "here's what happened" | a `tool_result` block in a user message | its own message tagged `role:"tool"` |
| "I'm done" | `stop_reason: end_turn` | `finish_reason: stop` |

Because `types.py` holds one neutral vocabulary, supporting a second
company meant writing exactly one new translator (~300 lines) and zero
changes anywhere else. That is the payoff of normalizing at the border:
the loop, tools, and UI genuinely don't know or care who's talking.

## 7 · Answers arrive as a waterfall

When you ask a question, the answer isn't written and then sent — it's
typed live, like watching over someone's shoulder. The technique is
called **streaming**, and the delivery format is called **SSE**
(server-sent events): plain-text postcards, one per event, separated
by blank lines.

```
 what the network actually delivers, piece by piece:

 data: {"type":"content_block_delta","delta":{"text":"The "}}
 data: {"type":"content_block_delta","delta":{"text":"answer"}}
 data: {"type":"content_block_delta","delta":{"text":" is 4"}}
 ...
```

Our hand-written parser stitches those fragments into words on your
screen as they land — which is why answers appear to *type themselves*.
The same trick handles something sneakier: when the model wants a tool
run, its instructions also arrive as fragments (`{"pa` … `th":
"READ` … `ME.md"}`), sometimes split mid-word across network packets.
We accumulate the pieces and only try to understand the JSON once the
final piece lands. Parse early and you crash on half a sentence.

## 8 · The seatbelt: permissions

Some tools can't hurt anything (`read_file`, `grep`) — they auto-run.
Others can (`bash` runs any command; `write_file` changes your disk).
For those, the harness pauses and asks *you*, showing exactly what will
run before anything happens:

```
 model asks:  bash("rm -rf tmp/build")
                       │
                       ▼
            ┌─────────────────────┐
            │  dangerous?         │
            │  read-only → run it │
            │  otherwise:         │
            │  "run it? [y/n/e]"  │
            └────────┬────────────┘
       yes │   edit │     │ no
           ▼        ▼     ▼
        execute  fix the   a polite "denied"
        for real command,  note goes back
                 then ask  to the model
                 again
```

There's a third answer besides yes and no: **`e` edits the call**.
Noticed a typo in the proposed command, or want it to do something
slightly different? Fix the text, and what you approved is exactly
what runs — approval is a review step, not a rubber stamp
([notes/20](notes/20-approve-with-edits.md)).

And note what a denial is: **not a crash**. It becomes a short letter
back to the model — "the human said no" — so it can shrug and try a
different approach. (There's also a `--yolo` switch that removes the
seatbelt entirely, for when you trust the task.)

The seatbelt has a silent partner: *hooks*. Where the belt DECIDES
("may this run?"), hooks just WATCH ("it is running… it finished").
Want a log of every command the agent executed, or timing numbers?
Attach two small observer functions and never edit the loop at all.
One rule keeps them apart: a hook can never say no ([notes/14](notes/14-hooks.md)).

### The question travels the other way, too

Pausing for *yes-or-no* isn't the only conversation worth having.
Sometimes the model is missing a fact no tool can find — which database
you actually use, what tone you want, whether "deploy" means staging or
production. Guessing wastes real work. So there's an `ask_user` tool
that stops the turn and asks **you** a question: numbered choices if it
can offer options, free text always. Your answer goes back into the
transcript like any tool result, and work resumes on your word.

And where does that conversation happen? Wherever you are. At the
terminal it's an ordinary prompt line. With `--web`, the same loop
serves a small local web page: replies stream live, tools appear as
cards, approvals become buttons, images attach by drag-and-drop, and
questions from the agent pop up as modals. It's a window onto the very
same machine — not a second one ([notes/22](notes/22-web-ui.md)).

## 9 · When things go wrong: bad news becomes data

Tools misbehave: typos in commands, missing files, genuine bugs. A
naive harness crashes. Ours has one rule:

> **No tool failure ever throws an exception. It becomes a note.**

```
 tool explodes ──►  ToolResult(is_error=True,
                                "ValueError: boom")
                          │
                          ▼
        pasted into the transcript like any other result,
        and the model reads it and adjusts course.
```

The loop physically cannot be crashed by a broken tool — the worst case
is a wasted round-trip where the model learns "that didn't work."
Only three things are allowed to stop a turn: you pressing Ctrl-C
(cancellation), the provider being unreachable (auth/rate-limit — no
point continuing if there's nobody to talk to), or `ask_user` running
with nobody home to answer (a headless run can't conjure a human; see
§8).

## 10 · The promise book (why this doesn't 400)

Each tool request carries an ID, like a claim ticket: `call_7`. The
protocol's one hard rule: **every ticket must be redeemed**. If we
ever mail the transcript containing an unanswered ticket, the provider
rejects the whole request with a cryptic error. This single mistake is
probably the most common way people's home-made agents die.

Sounds easy — until you realize all the awkward moments happen when
things go sideways: Ctrl-C pressed mid-turn, the 25-bounce cap hit, a
command hanging forever. So the harness keeps a literal promise book:
on *every* exit path it scans for unredeemed tickets and writes
redemption slips — real results for finished work, "interrupted" notes
for the rest — before doing anything else.

```
 Ctrl-C mid-turn!
        │
        ▼
 scan: ticket call_7 answered ✔   call_8 unanswered ✘
        │
        ▼
 append: [result for call_8: "(interrupted by user)"]
        │
        ▼
 transcript is whole again → next prompt just works
```

That's why cancelling a turn in this CLI never breaks the session.

## 11 · Where the memory lives

One flat list, in neutral vocabulary, holding everything said *and*
everything done:

```
 agent.history (after our §5 example):

 [0] user      ["What's in README.md?"]
 [1] assistant [thinking…, request: read_file]
 [2] user      [result: "# AksharaHarness — a from-scratch…"]
 [3] assistant ["It's a from-scratch agent harness…"]
```

Storing it in the neutral shape (never in either company's dialect) is
what makes switching providers mid-session free: same list, different
translator, nothing lost.

## 12 · Hiring helpers: sub-agents

Big jobs drown one conversation. Ask this project's own coordinator to
"research how retry works" and it can hire an assistant: a brand-new
agent with a **fresh empty transcript**, a restricted toolkit (no
hiring assistants of its own — one floor per office tower), and a
budget of spawns per session.

```
 you ⇄ MANAGER AGENT ── hires ──► SUB-AGENT (fresh transcript)
        │        ▲                   does the legwork,
        │        └── conclusions ───  alone with its files
        │            only (never      and shell
        ▼             the mess)
     keeps its own transcript clean
```

Two rules make it safe. The child reports **conclusions, not
transcripts** — a tidy summary plus what it cost, never pages of
intermediate noise flooding the manager's filing cabinet. And the
"hire an assistant" tool is simply absent from child toolboxes, so
delegation depth is enforced by the toolbox, not by hoping the model
behaves. While the child works, you can watch its stream scroll past,
tagged as the child's — supervision without interruption.

## 13 · Remembering on purpose: the scratchpad

Everything in §11 lives in the transcript, and the transcript gets
trimmed when it grows (old tool outputs elided, old middle sections
summarized). So where does something go that must survive trimming —
and survive the program quitting?

A tiny deliberate memory: two tools backed by a JSON file on disk.

* `write_note("db_password", "in .env, key DB_PASS")` — file it away
* `recall_notes("password")` — search filed notes by keyword, best
  matches first

That's the whole trick, and that's the point: long-term memory is not
magic, it's *notes the agent chose to take*, kept somewhere the
compactor is forbidden to touch. Session one writes the note; a
session started next Tuesday recalls it.

## 14 · Universal plugs: MCP

Tools we wrote ourselves cover files, shell, search. But the world has
thousands of services, and wiring each one by hand is the M×N problem:
M AI apps × N services = M×N bespoke connectors. The Model Context
Protocol turns that into M+N: any service speaks MCP once, any agent
speaks MCP once.

Under the hood it is startlingly small — four exchanges. Hello (who
are you, what version?). Me too (let's talk). What can you do?
(a list of tools). Do X (a call, an answer). Our client speaks it over
a subprocess pipe OR over HTTP, with zero SDKs.

One etiquette rule worth knowing: servers may PING us mid-conversation,
and a ping left unanswered wedges a polite server. So the client
answers pings even while waiting on its own answers — the protocol
equivalent of not leaving people on read.

Configured servers' tools appear in the toolbox named like
`mcp__tiny__add` — the model asks for them exactly like built-ins,
permissions apply identically, nothing downstream changes.

## 15 · Many conversations, one engine

Everything above runs one conversation at a time. But batch jobs and
chat servers want MANY — and the secret of the loop is that it spends
most of its life *waiting*: for the network, for a command to finish.
Waiting is free to overlap.

So the whole stack grew a twin: the same wire rules fed through
Python's event loop (`async`/`await`), letting one engine drive many
independent conversations. Measured live: four question-answer
sessions sequentially took 12.4s; on one event loop together, 6.1s —
every session started instantly, replies interleaved.

```
 sequential:  ████ ████ ████ ████   12.4s   (each waits its turn)
 concurrent:  ████                 6.1s   (all wait TOGETHER)
              ████
              ████
              ████
```

The hard part was never speed, it was *cancel-safety*: pressing Ctrl-C
mid-batch in this world must still record the work that actually
finished. (It nearly didn't — the obvious code cancels the workers
along with everything else. The fix, `asyncio.shield`, is documented
in [notes/11](notes/11-async.md) as a war story.)

The same trick pays rent in testing. The project's graded exams
(evals — a set of real tasks the agent must complete correctly; see
the glossary) are independent of each other, so they now run all at
once too: the seven-exam suite took 117 seconds one-at-a-time and 42
together — identical grades, 2.8× faster ([notes/10](notes/10-evals.md)).

## 16 · The money section: prompt caching

Here is an uncomfortable bill: every bounce re-mails the ENTIRE
transcript (§5, detail 3), and the biggest parts of it — the tool
definitions, the system prompt — never change between bounces. Ten
bounces means paying full price for the same prefix ten times.

Anthropic-dialect caching fixes exactly this: mark the stable prefix
with special markers ("cache breakpoints"), and later mailings either
READ it back at roughly a tenth price or WRITE it once at 1.25×. We
mark three places — last tool, system prompt, newest message — because
that is the shape of a growing conversation: frozen head, growing tail.

Measured live with an ~8,600-token reference document in the system
prompt: the cold call billed the reference almost in full; the next
call with the identical prefix billed essentially nothing — the
document came back from cache for pennies. Off by default (it changes
billing behavior), one flag to enable: `--cache`. Full story with
caveats: [notes/13](notes/13-caching.md). And since the harness now knows what
tokens *cost*, `/usage` renders these counters as approximate dollars —
see [notes/21](notes/21-cost-accounting.md).

## 17 · The graduation test

What do you DO with such a machine? What Claude Code does: give it a
spec in an empty folder and let it build. Write files, run the tests,
read the failures, fix, repeat until green.

We made it prove itself twice. Greenfield: build a unit-conversion
library with passing tests and a working CLI (done in five bounces).
Repair: hand it a project with five planted bugs and say make CI green
— done in eight, diagnosing each bug correctly, WITHOUT touching the
tests (the demo fingerprints the test files and fails the build if
they change — "make tests pass" has a cheating shortcut, and the demo
removes it).

And the kicker: **nothing new had to be built.** The loop from §4,
the hands from the toolbox, the promise book from §10 — the features
above are refinements. An agent that builds software is not a bigger
machine; it is the same circle, pointed at a harder task.
([notes/12](notes/12-builder.md))

## 18 · Try it yourself

```bash
uv sync                                   # one-time setup
uv run pytest -q                          # full offline suite (~480 tests)

# the rest talks to a real model (needs a key, or a local Ollama):
uv run akshara                            # interactive session
uv run akshara --provider ollama --web    # the same session in your browser
uv run akshara --yolo "summarize README.md"
uv run python examples/agent_loop_demo.py # watch the loop, event by event
uv run python examples/builder_demo.py    # watch it BUILD a project
uv run python examples/cache_demo.py      # watch caching cut the bill
```

The best demonstration: start the REPL, give it a task that takes a
few steps, and press **Ctrl-C mid-turn**. Watch it cancel cleanly,
leave a redemption slip in the promise book, and accept your next
prompt without complaint. That recovery path is §10 made visible.

## Glossary

| Term | Plain meaning |
|---|---|
| LLM / model | the text-in-text-out brain, rented over the internet |
| harness | everything we built around it: hands, memory, seatbelt |
| agent loop | the ask→read→act→record circle (§4) |
| tool | a capability we let the model request (read files, run bash…) |
| tool call | the model's formal request, carrying an ID claim-ticket |
| tool result | our report back, pasted into the transcript |
| SSE / streaming | delivery format that lets answers appear live |
| wire format | a company's private language for chats (§6); we speak both |
| adapter / provider | the translator for one company's language |
| invariant | the never-break rule: redeem every ticket (§10) |
| history | the transcript — the model's only memory (§11) |
| sub-agent | a hired helper: fresh transcript, restricted tools, reports conclusions (§12) |
| scratchpad | the deliberate on-disk memory that survives compaction and restarts (§13) |
| MCP | the universal plug protocol: any service speaks it once, any agent listens once (§14) |
| event loop / async | the machinery for overlapping waits — many conversations, one engine (§15) |
| cache breakpoint | a marker saying "everything before here is stable — stop re-billing it" (§16) |
| MTok | million tokens, the unit prices are quoted in; `/usage` turns counters into dollars via a price table (notes/21) |
| eval | a graded exam for agents: did the trajectory do the right things, not just say them |
| hook | an observer clipped onto tool executions — it watches and records, it can never forbid |
| ask_user | a tool that pauses the turn to ask *you* a question, then continues on your answer (§8) |
| web UI | `--web`: a local browser page driving the same loop — streaming, approval buttons, agent questions as modals (notes/22) |
| image block | a picture the user attaches (base64); translated per-dialect on the way out only — see notes/15 |
