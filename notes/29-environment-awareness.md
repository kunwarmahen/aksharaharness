# 29 — env context: sessions that know where (and when) they are

*The harness shipped its first hundred sessions with NO system prompt at
all. Every conversation therefore began in total amnesia — not today's
date, not the timezone, certainly not the city. Ask "what's the
temperature outside?" and the agent burned a turn asking YOU which city
you're in. Not laziness: honest ignorance, with ask_user as the only way
out of it. This note is why awareness ships as a graded default instead
of a clever trick ([notes/06](06-cli.md) has the CLI surface,
[notes/22](22-web-ui.md) the web chip).*

## Facts first, policy second

The fix has two halves, and only one of them is data.

**The fact block**: time + timezone, user@host + OS, working directory,
and — at `full` — your city/region/country from ONE keyless lookup to
ipinfo.io. Injected into the system prompt, it makes "which city?"
unaskable: the model can see the answer.

**The policy line**: two appended sentences telling the model to try its
tools BEFORE asking the human for any fact, and to keep questions for
what's genuinely yours — preferences, permission for the irreversible.
This half matters as much as the first: facts without policy change one
conversation; policy without facts is a slogan the model can't act on.
Together they turn "which city?" into "checking the weather in <your
city>…" — the model fetching `wttr.in/<city>` through web_fetch, which
is precisely the self-reliance the feature exists to teach. Weather
itself is deliberately NOT injected: it goes stale within the hour and
would cost either per-turn refreshes or a lie. The city is the fact;
the weather is an exercise.

## Why inject rather than expose a tool

An `env_facts` tool was considered and rejected. Tool descriptions are
visible to the model, but a model must already SUSPECT a fact exists to
reach for the tool that holds it — the blind spot is exactly what you
can't name. Injection makes ignorance structurally impossible, saves a
round-trip on every use, and gives the policy line a permanent home.
The cost is ~80 tokens of every request, paid even in sessions that
never needed the time; against a 200k cloud window that's rounding
error, and `local`'s four short lines are cheap enough for an 8k local
model too.

## The frozen block

`--cache` (anthropic dialect) caches the request PREFIX — including the
system prompt — so long sessions bill cached tokens at ~0.1x. A system
prompt containing a live clock would bust that cache EVERY TURN, turning
the cheapest part of the request into the most expensive. So facts are
collected ONCE per session, rendered once, and describe themselves as
possibly-stale ("auto-detected at session start"). Honesty about the
staleness is printed into the block itself; nobody has to remember it.

## Graded by default: full > local > off

- **full** (the default): machine facts + location. The honest cost:
  your city rides inside every request you send your LLM provider — and
  the lookup itself tells ipinfo.io your IP (which it knew anyway; that
  is how the lookup works). Convenience-first was chosen deliberately;
  the escape hatches are one flip away.
- **local**: machine facts only. Nothing leaves the machine beyond the
  conversation itself. Right for shared machines and privacy purists.
- **off**: nothing injected, no policy line — byte-identical to the
  pre-feature wire shape (`system=None` stays `None`). The old behavior
  remains one command away, not deleted.

Three surfaces agree on one tuple (`MODES_ENV`): `$AKSHARA_ENV_CONTEXT`
/ `--env-context` set the STARTING level; `/env` (REPL) and the env chip
(web) flip it live. Junk values fail loudly at startup (`ConfigError`),
same contract as every other knob in config.py.

## Composition is additive; capture happens once

The operator's `--system` prompt is captured verbatim as the BASE;
awareness appends BELOW it. Nothing edits the operator's words.

The once-ness is load-bearing because checkpoints store the COMPOSED
string (session.py saves `system` verbatim). If attach() re-captured the
base from an already-composed agent — say after `/load` restored an old
checkpoint — yesterday's stale block would be swallowed into today's
base and stack forever. Hence: base captured EXACTLY ONCE, and every
load path calls `reapply()` afterwards, so restored sessions get freshly
composed context instead of archaeology.

## Flips are live, upgrades pay once

The loop rebuilds each request from `agent.system` fresh, so a flip —
mid-session or MID-TURN — lands on the very next model call. Like the
permission and tools toggles there is deliberately no require_idle(): a
wrong level shouldn't have to wait out a running turn. Upgrading to
`full` performs the geo lookup lazily, once; the result survives later
downgrades, so flip-flopping never re-queries. Offline machines don't
stall: the lookup carries a 3s timeout, failure drops the Location line,
prints a dim notice, and the session proceeds with local facts.

## What the tests pin

- mode resolution: unset/blank → full; valid pass-through; junk →
  ConfigError; constructor rejects unknown levels (ValueError → clean
  400s at the web endpoint)
- collection: local lines render (+policy rides EVERY enabled level);
  full renders `- Location:` only when the payload yields one; failures
  are soft (line dropped, reason surfaced); adjacent duplicate
  region==city collapses
- composition: custom `--system` base survives above the block; off +
  no base composes to None; describe() snapshot shape
- flips: live on the agent; upgrade looks up EXACTLY once across
  downgrades; bad argument raises and keeps the old level; reapply()
  never adopts a restored composed string as the new base
- REPL: bare /env prints the panel; /env off|local|full flips; invalid
  usage reported; missing EnvContext reports instead of crashing
- web: state carries `env_context` (None for unaware agents); POST
  /api/env-context flips + broadcasts to open tabs; validation errors
  and missing-context both clean 400s

## Receipts

Offline suite green throughout (no test touches the wire — the geo seam
is monkeypatched, and an autouse fixture fails loudly if anything tries).

Live receipt, same one-shot prompt both times ("What is the temperature
outside right now? Check the actual current weather.", Ollama
`qwen3.8`, `--yolo --tool-select 0`):

- **full** *(default)*: knew "Raleigh, NC" from the block, fetched
  open-meteo for the right coordinates, answered *"Outside in Raleigh,
  NC right now… Temperature: 64°F (feels like 68°F)"* — end_turn, no
  questions asked. `3735 in / 138 out · 2 iteration(s)`.
- **off**: still resourceful enough to hit wttr.in, but without the
  context or the policy line it second-guessed the geolocation — *"I
  can't see your own location. If you tell me your city, I can check
  there"* — and called ask_user. Headless, that fails the turn loudly;
  interactively, it's precisely the "which city?" interruption this
  feature deletes.

The toggle moves real behavior, not just prompt text.
