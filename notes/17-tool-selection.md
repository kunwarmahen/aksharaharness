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

`select(query, k=7)` returns pins + positive-scoring tools only. A
query nothing matches returns JUST the pins — never filler ranked last,
because a filler ranked last still teaches the model a wrong tool
exists. Pins come from `catalog.must_include` (default:
`list_available_tools`); a caller can pass its own set or none.

## Three pieces

| Piece | Job |
|---|---|
| `ToolCatalog` | index over name+description; `select()` with floor |
| `query_from_transcript(history)` | first user message = session anchor; recent assistant text + **tool-call names** track the current sub-task. Names matter most at pivots: having called `mcp__db__run_query` puts db vocabulary into the query BEFORE any prose about databases exists |
| `ListAvailableTools` | the pinned discovery hatch: lists the FULL catalog on demand, optional substring filter |

The discovery tool exists because try-fail-retry alone has two holes
it cannot close: vague openers where everything scores zero, and
mid-task pivots whose new vocabulary hasn't reached the transcript yet.
Its description teaches the contract explicitly: *a tool discovered here
becomes usable from YOUR NEXT TURN* — when its name sits in history and
BM25 surfaces it naturally.

## The seam must cut twice

Restricting only what gets SENT would leave deselected tools
executable — selection as theater. So when a catalog is active:

1. `_specs_for_request()` sends only this turn's selection;
2. `_get_visible_tool(name)` resolves calls through the SAME selection.

Hidden-but-real names get an error result that is data, not a crash —
and the message teaches the recovery path: *"tool 'x' exists but is not
loaded this turn … call list_available_tools."* A genuinely
nonexistent name keeps the plain old *"no such tool"*. Two distinct
messages, because they have distinct fixes. The failed call lands in
history like any other, which feeds its vocabulary into the next
selection query — that feedback loop is why try-fail-retry CONVERGES
instead of thrashing.

The loop has one more input most designs miss: **recent tool RESULTS
join the query too** (truncated). A discovery answer is a user-role
ToolResult full of tool names; if results didn't feed back, the
promised "discovered here → loaded for your next step" would silently
fail — the live session proved it before the fix (model listed the
tools, then correctly reported it still couldn't call them). Results
are capped at ~400 chars each and only from the recent window, so a
noisy bash output can't drown the query.

Without a catalog everything is byte-identical to before (pinned by
`test_no_catalog_keeps_full_registry`).

## Wiring

```
uv run akshara --tool-select 7     # force width K
                                   # auto-enables above 20 tools otherwise
uv run akshara --tool-select 0     # opt out of the auto-enable
```

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
hatch turns every selection miss into a dead end.

Beyond description quality: per-turn selection adds one failure mode
the full-registry world doesn't have (the hidden-but-real error above);
we judged teaching that once cheaper than paying the cliff forever.
