# 17 — Dynamic tool loading: BM25 against the tool cliff

*Companion to [notes/04](04-tools.md) (the registry) and
[notes/09](09-mcp.md) (where tool counts explode). Book ch12.*

## The problem in numbers

Selection accuracy is roughly flat to ~10 tools, visibly degraded by
20, and falls off a cliff between 30 and 50 — while every tool's schema
burns 100–500 tokens PER REQUEST. MCP makes 50 trivially easy: three
modest servers and you are past the cliff before the first user
message. The fix is to stop treating "which tools does the model see"
as static configuration and treat it as a RETRIEVAL problem, solved
per turn.

## Why BM25 and not embeddings

Queries come from the agent itself (its own transcript), so they share
vocabulary with tool descriptions — lexical overlap is the signal.
BM25 is ~60 lines with zero dependencies (`tools/selector.py`), fully
deterministic for tests, and swapping in embeddings later means
replacing exactly one method (`ToolCatalog.scores`). The parameters are
the textbook defaults: `k1=1.5` (term-frequency saturation), `b=0.75`
(length normalization). Tokenization splits on non-`[a-z0-9_]` so
snake_case survives intact — `post_message` stays one term, which
matters because tool names ARE the vocabulary.

## The score floor is the design

`select(query, k)` returns pins + positive-scoring tools only. A query
nothing matches returns JUST the pins — never filler ranked last,
because a filler ranked last still teaches the model a wrong tool
exists. Pins come from `catalog.must_include` (default: `CORE_PINS` +
`list_available_tools`, resolved against what is actually registered);
a caller can pass its own set via `enable_selection(registry,
pins=(...))` or none.

## Why the core is PINNED, not retrieved

Retrieval ranks descriptions against conversation vocabulary — but the
autonomy loop's floor doesn't describe the TASK, it enables every
task. Ask "fix this bug in parser.py" and `write_file`'s description
shares not one word with the query; BM25 correctly drops it, exactly
when the turn needs it next. So six tools beat the query and load
every turn regardless of score (`selector.CORE_PINS`): read_file,
write_file, edit_file, bash, glob, grep — plus the discovery hatch.
That arithmetic sets the auto-enable width: `DEFAULT_TOOLS_PER_TURN =
12` leaves ~5 slots for the retrieved long tail (MCP servers, browser,
background jobs), which is what selection was for in the first place.

## Three pieces

| Piece | Job |
|---|---|
| `ToolCatalog` | index over name+description; `select()` with floor |
| `query_from_transcript(history)` | first user message = session anchor; recent assistant text + **tool-call names** track the current sub-task. Names matter most at pivots: having called `mcp__db__run_query` puts db vocabulary into the query BEFORE any prose about databases exists |
| `ListAvailableTools` | the pinned discovery hatch: lists the FULL catalog on demand, optional substring filter |

The discovery tool closes the two holes retrieval alone cannot:
vague openers where everything scores zero, and mid-task pivots whose
new vocabulary hasn't reached the transcript yet — the model can ASK
what exists instead of guessing. (Execution-side, guessing is now
harmless — soft admission runs any real name — but harmless guesses
still waste a call; the hatch lets the model look before it leaps.)
Its description teaches the contract explicitly: *a tool discovered here
becomes usable from YOUR NEXT TURN* — when its name sits in history and
BM25 surfaces it naturally.

## The seam cuts once: selection caps what gets SENT

The original design cut twice — unselected tools were also
unexecutable, and a hidden-but-real call got an error teaching the
discovery path (*"call list_available_tools"*). Live use broke that.
The model names tools it legitimately knows: from an earlier listing,
from resumed history, or because `write_file` is simply what every
model has in its weights. Punishing exact knowledge with a failed
round-trip taught nothing, burned an iteration, and weaker local
models thrashed on the retry dance — one session lost several turns
to *"write_file exists but is not loaded this turn."*

So execution now SOFT-ADMITS (`_get_visible_tool`): a call naming a
real but unselected tool joins `_turn_tools` on the spot and runs THIS
turn. Permission gating is unchanged — admission only bypasses the
retrieval cap, never the seatbelt. And it converges by itself: the
executed name lands in history, which feeds the next selection query,
so BM25 keeps it ranked from here on without any special casing.

That leaves exactly one failure mode, and it's the honest one:
a genuinely hallucinated name still errors as data ("no such tool").
Selection stays what it should be — a context-economy decision about
what to SEND — rather than a second permission system.

The loop has one more input most designs miss: **recent tool RESULTS
join the query too** (truncated). A discovery answer is a user-role
ToolResult full of tool names; if results didn't feed back, the
promised "listed here → surfaced next step" would silently fail — the
live session proved it before the fix (model listed the tools, then
correctly reported it still couldn't call them). Results are capped at
~400 chars each and only from the recent window, so a noisy bash
output can't drown the query.

Without a catalog everything is byte-identical to before (pinned by
`test_no_catalog_keeps_full_registry`).

## Wiring

```
uv run akshara --tool-select 12    # force width K
                                   # auto-enables at 12 above 20 tools otherwise
uv run akshara --tool-select 0     # opt out of the auto-enable
AKSHARA_TOOLS_PER_TURN=12          # .env spelling of the same switch
```

## Trimming the toolset entirely

Selection decides what fits in a request; sometimes you want a tool to
not exist. `AKSHARA_DISABLED_TOOLS` takes comma-separated glob patterns
matched against tool names and UNREGISTERS matches after MCP servers
connect but before any catalog is built — so a disabled tool is never
sent, never executed, never suggested by `list_available_tools`, and
never pinned:

```bash
# hide one family ...
AKSHARA_DISABLED_TOOLS=browser_open,browser_click,browser_fill,browser_close
# ... or one tool, or a whole MCP server
AKSHARA_DISABLED_TOOLS=web_fetch,mcp__slack__*
```

A pattern matching nothing prints a warning — typos should be loud.
The registry's `unregister(name)` is the primitive; the env var is just
the operator-facing spelling.

`enable_selection(registry)` is idempotent: a second call re-points the
live discovery instance at the rebuilt index instead of colliding.
Below `AUTO_SELECTION_THRESHOLD = 20` don't bother (book's rule);
MCP tools need nothing special — they are plain `Tool`s by the time
they reach the registry ([notes/09](09-mcp.md)).

Sub-agents keep validating against the FULL registry: children get
small explicit tool lists already; selection is a wide-catalog cure.

## Honest scope

**Descriptions are retrieval documents — write them like it.** Live
receipt: against 24 tools, the query "Use an MCP tool to add 2 and 3"
surfaced six ECHO tools and no add tool, because echo's description
says "proving an MCP round trip works" (matching `an`, `mcp`, `the`)
while add's says only "Add two integers." That is BM25 being correct,
not broken: the user's vocabulary met the echo description more often.
Two consequences: (1) tool authors should put the task-domain words in
descriptions; (2) this exact failure is why the discovery hatch is PINNED —
the live model noticed the mismatch and called `list_available_tools`
on its own.

Which is also why discovery must be FRICTIONLESS: `confirm_gate` never
prompts for read-only tools, because a permission wall in front of the
hatch turns every discovery question into a dead end.

The honest cost ledger: per-turn selection still trades a little
context certainty for a lot of context economy. A tool outside the
pins can be missed in the SENT set on its first use — soft admission
makes that miss cheap (the call runs anyway) rather than free, and
description quality decides how often it happens at all.
