# 07 · Reliability and scale: retry, parallel tools, persistence, compaction

> Four features adapted from the agent-harness book (ch5, 7-8, 17, 21)
> to a sync-generator harness. Files: `providers/retry.py`,
> `agent.py` (batching), `session.py`, `context.py`.

## Retry: only the opening is retryable

Policy (`providers/retry.py`, shared by both adapters):

| | |
|---|---|
| RETRY | 429 / 500 / 502 / 503 / 504 + `httpx.TransportError` (connect/DNS/reset/timeout) |
| NEVER | 400 / 401 / 403 / 404 -- "a malformed request will not fix itself" |
| WAIT | `min(max_delay, base·2^(n-1) + U(0, base))`; provider's `Retry-After` wins verbatim on 429 |
| BUDGETS | 5 attempts total AND 120s wall-clock; last error raises |

**The streaming rule is the whole design:** retries cover only the
opening (request -> status line). Once one event has reached the caller
there is no safe replay -- you would emit duplicate tokens into someone's
terminal. Our adapters already check status before yielding anything, so
`connect_with_retries()` wraps exactly that phase and mid-stream failures
propagate untouched. Durable mid-stream recovery is checkpointing's job,
not a retry loop's.

Jitter is load-bearing: identical backoff schedules re-create the thundering
herd that caused the throttle. Budgets are hard ceilings because unbounded
retry is how agents run up silent four-figure bills.

Tests intercept `retry._sleep`, so backoff decisions are asserted in
milliseconds. Error-taxonomy tests construct providers with
`RetryPolicy(max_attempts=1)` so 429/5xx fixtures raise immediately.

## Fallback: failover across space

`providers/fallback.py` — `FallbackProvider(primary, *fallbacks)`.
Retry fixes TIME problems (same place, later); fallback fixes PLACE
problems (same moment, different backend). Same streaming rule, so they
compose cleanly: each adapter burns its own retry budget first; only an
exhausted opening gets failed over.

Worth failing over = provider-side trouble where another venue genuinely
differs: AuthError (the next key may be valid), RateLimitError, 5xx,
TransportError. NOT worth it = request-shaped failures: ContextOverflow
and plain 400s send the SAME bytes to every backend — changing venue
cannot heal them, so fail fast instead of paying to find out.

Two honesty details:

* **Commit point**: `next(gen)` is where failover dies. First event
  delivered → failures propagate untouched, exactly like mid-stream
  retry refusal.
* **All-fail aggregation**: the LAST error raises (as `__cause__`) and
  every earlier attempt rides in the message — "primary throttled, then
  fallback key invalid" is a diagnosis; either alone is a riddle.

Deliberately not a Provider subclass (no settings/base URL/client of its
own — it IS a list of those), and deliberately library-only for now:
session persistence rebuilds providers by NAME via the factory, and a
composite can't round-trip through a name yet.

Live-verified against OpenRouter by pointing the primary at a bogus base
URL: first turn fails over to the real backend transparently; answer,
streaming, and usage all come from the fallback with no caller-visible
error.

## Parallel tools: gates sequential, execution concurrent

Everything-parallel policy with two non-negotiables:

1. **Gates run first, sequentially** ([agent.py] `_execute_batch`). The
   y/n prompt owns the one terminal; interleaved Confirm calls would be
   unusable and unsafe.
2. **Results yield in submission order**, never completion order --
   `ThreadPoolExecutor` futures read positionally, mirroring the book's
   `asyncio.gather` guarantee. Panels stay deterministic.

Cancellation surprise worth remembering: **worker threads never receive
SIGINT**. Ctrl-C lands in the main thread waiting on `fut.result()`; the
`with ThreadPoolExecutor` block then JOINS the workers as they finish
(bounded by tool timeouts). So the work happened -- and we record the REAL
results before propagating, instead of synthesizing "interrupted" stubs for
side effects that actually ran. A worker that itself raises KeyboardInterrupt
gets a synthesized slot result during harvest; the invariant holds either way.

Because batches now execute fully before any ToolExecuted yields, closing
the generator after the first panel must not forget batch-mates: all ids
are recorded into `executed{}` BEFORE the first yield.

### Leases: the hazard list made real

Once batches went parallel, two `edit_file` calls aimed at one path could
interleave read-modify-write and silently eat an edit. `leases.py`
(called through a shared `ctx.leases`) answers it with four semantics:

| Choice | Reason |
|---|---|
| TTL on every lease | a crashed holder must not deadlock the file; expiry is checked lazily at acquire -- no reaper thread |
| blocking acquire with deadline | batch siblings finish in milliseconds; waiting converts most conflicts into free serialization. Past deadline → `LeaseBusy`, a ToolError, so errors-are-data applies |
| reentrant by owner, depth-counted | same-owner re-acquire refreshes TTL instead of self-deadlocking; last release frees |
| only the holder releases | releasing someone else's lease is a bug, surfaced as one |

Reads need no leases; keys are per-path (`fs.file_lease_key`), so
different files never block each other and no global lock sneaks in.
Bash bypasses both sandbox and leases -- consistent, since bash was
never confinable. Honest scope: guards ONE process's shared contexts;
sub-agents get their own ToolContext/manager (their runs are already
serialized against the parent's batch).

## Persistence: SQLite, append-only versions

`SessionStore` = one table, `checkpoints(session_id, version, created_at,
payload)`; every save INSERTs, nothing UPDATEs. Crash mid-write can't
corrupt the previous good state, and any old version stays inspectable.

Our scope is much smaller than the book's five-item durable state because
of one core decision: **history is always resumable at prompt boundaries
(the invariant), so there are no outstanding tool_call ids to ledger.**
No idempotency keys needed when you never save mid-call.

Payload JSON uses explicit `kind` discriminators restored via match/case;
unknown kinds raise a clean ValueError ("newer than this code") rather
than guessing. ThinkingBlock signatures round-trip byte-exact -- lose one
and every thinking-assisted tool loop 400s after `/load`.

sqlite3 gotcha that bit the tests: passing a bare string as `execute()`
params iterates it PER CHARACTER ("107 bindings supplied"). Params go in
tuples, always.

## Compaction: mask before summarize

The window is a resource (book ch7): effective budget is well under the
headline number, and tool results dominate transcripts (often 70-90%).
Two layers, strictly ordered:

1. **Mask** (`mask_old_results`): keep the newest 3 ToolResults verbatim,
   elide older ones to `"[tool result elided ... call_id=...; re-run if
   needed]"`. Reversible, idempotent, preserves is_error, keeps every call
   answered -- the invariant doesn't even notice.
2. **Summarize** (only if still red): replace a validated middle span with
   ONE user message containing an LLM summary. The first user message (the
   goal) survives verbatim. The summarizer prompt REQUIRES enumerating
   every tool call + outcome -- "the record of what was done must survive
   compaction", or the agent re-sends that email.

Cut-point safety: neither edge of the summarizable span may fall on a
results batch whose assistant partner sits on the other side -- otherwise
providers see orphaned tool_results or orphaned tool_use ids and reject
the request. `summarizable_span()` slides both edges until clean.

Measurement honesty: trigger decisions prefer the provider's own
last `input_tokens`; the chars/4 heuristic is a labeled PROXY used before
the first response arrives. Some gateways report input tokens as null even
on success, so `/usage` falls back to "~N estimated".

Wiring: `_before_model_call` -- an identity hook since the first loop --
finally became what it was named for: auto-compaction in the red zone (80% of
usable window), plus manual `/compact`. The model never needs to know
its history shrank between requests.

## What we deliberately did NOT build

Async everywhere and scratchpad/retrieval memory beyond what
`tools/memory.py` now covers. Each was built out in its own phase once
it earned its way off this list.

## The second notch: pre-first-event reconnect

The opening-only rule above had a hole exactly one event wide: a stream
that returns **status 200** and then dies BEFORE its first SSE block
was terminal, even though nothing had reached the caller —
indistinguishable from a failed opening, which we already know how to
retry.

The rule is now stated in two notches (`providers/retry.py` docstring):

1. **OPENING** — request → status line: freely retried (429/5xx/transport).
2. **PRE-FIRST-EVENT** — a 200 body that dies before its first event:
   re-opened under the SAME budgets via `budgeted_delay()` (shared
   arithmetic, so attempts + wall-clock ceilings still apply). Nothing
   was forwarded, so nothing can be duplicated.
3. **POST-FIRST-EVENT** — never. Unchanged from the original rule;
   replaying half-delivered tokens into someone's terminal is not
   recovery. Durable mid-stream recovery remains checkpointer territory.

Implementation lives in each adapter's `stream()`/`astream()` as an
attempt loop around the existing `connect_with_retries` opening (not a
wrapper class — the finally-close-per-attempt socket semantics stay in
one place). A per-attempt `forwarded` counter is the boundary; the
boundary itself is pinned by tests: a MockTransport that resets before
the first byte re-opens invisibly (2 requests, 1 backoff, one full
stream), one that resets AFTER a real event raises with EXACTLY the
forwarded events, no replay, no waits burned.
