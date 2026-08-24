# 21 · Cost accounting: turning token counters into dollars

> Tokens were always counted; dollars never were. `/usage` answered
> "how big?" but not "how much?" — and the second question is the one
> with consequences. Files: `pricing.py` (the whole price sheet),
> `types.py` (`Usage.window_tokens` + the disjoint-counter invariant),
> `providers/openai.py` + `providers/responses.py` (the subtraction),
> `agent.py`/`async_agent.py` (per-model buckets), `cli/repl.py`
> (`_cost_line`), `cli/render.py` (turn footer), `tests/test_pricing.py`.

## The naive version is wrong in an interesting way

Cost looks like a one-liner: multiply each counter by its rate. Try it
and you discover the internal vocabulary had a landmine in it —
**the two wire dialects disagree about what "input_tokens" means**:

* **Anthropic**: `input_tokens` EXCLUDES cache hits. Cached tokens
  arrive as their own counters (`cache_read_input_tokens`,
  `cache_creation_input_tokens`). Disjoint by nature.
* **OpenAI / Responses**: `prompt_tokens` / `input_tokens` INCLUDE the
  cached hits; they're only *separable* via a details sub-object
  (`prompt_tokens_details.cached_tokens`). Nested by nature.

Our adapters had faithfully mirrored each dialect into one shared
`Usage` type — so the same field meant two different things depending
on which adapter filled it. Multiply naively and every cached
OpenAI-dialect token is billed **twice**: once inside the headline
total at full rate, once again at the cached rate.

This is exactly the class of bug the project's founding rule exists
for: *normalize at the border*. So the fix went into the adapters, not
into pricing math — OpenAI-family adapters now store
`input = prompt − cached`, keeping cached tokens solely in
`cache_read_tokens`. One convention everywhere:

> **The four usage counters are disjoint. `input_tokens` counts only
> tokens billed at the full input rate; the window footprint is the SUM
> of all four (`Usage.window_tokens()`), because a request that bills
> almost nothing fresh still fills the window just as hard.**

That second clause matters: compaction pressure was previously computed
from fresh input alone, understating a warm-cache session's true
footprint. The fix fell out of the accounting work for free.

## What `$` shows up where

* **Turn footer**: `── end_turn · 942 in / 187 out · ~$0.0033 · 3 iteration(s)`
  whenever the model slug has a known list price.
* **`/usage`**: a session line — summed over **per-model buckets**
  (`agent.usage_by_model`), because a mid-session `/model` switch means
  one session spans two price sheets and a single blended multiplier
  would be a lie.
* **Library users**: `session_cost(agent.usage_by_model) -> (dollars,
  fully_priced)`.

Honesty rules, enforced by tests rather than convention:

* An **unknown slug shows no figure at all** — never `$0.0000`, which
  is indistinguishable from "this really was free" and quietly trains
  the wrong instinct. Partial coverage says so: `~$1.5000 (priced
  models only)`; zero coverage says `no list price known`.
* A **local model is genuinely free**: the `ollama` provider renders
  `$0.00 (local model)` without consulting any table.
* Prices are **list prices, snapshot-dated 2026-08**, and will drift.
  `AKSHARA_PRICES=/path/to.json` overrides or extends the built-ins:
  exact slugs plus `prefix*` family rules, USD per 1M tokens, with
  `cached_read`/`cached_write` optional (absent → billed at full input
  rate — the conservative fallback, matching vendors like the `-pro`
  tiers that publish no cache discount). A malformed file raises
  `ConfigError`; a wrong price silently shown is worse than a crash.

```json
{
  "openrouter/my-gateway-model": {"input": 2.0, "output": 8.0},
  "my-fine-tune-*": {"input": 0.5, "output": 1.5,
                     "cached_read": 0.05}
}
```

## Matching: four chances to recognize a slug

1. exact hit (`claude-opus-5`)
2. date suffix stripped (`claude-opus-5-20260201`)
3. vendor prefix stripped (`anthropic/claude-opus-5`, OpenRouter-style)
4. longest family prefix (`claude-opus-9-whenever` still lands on the
   Opus tier until overridden)

Unknown after all four → `None`. Long-context price tiers are
deliberately not distinguished (short-context rates used); the override
file is the escape hatch.

## What the tests pin

All four matching stages; override precedence (user prefix beats
built-in family); cache-split arithmetic against hand-computed sums;
the subtraction on both OpenAI skins AND the streaming path (same rule,
second implementation — the classic place to forget); clamp when a
gateway reports more cached than prompt tokens; per-model bucketing
across a simulated `/model` switch; `last_context_tokens == window
footprint`; every `/usage` honesty rule; footer `$` present/absent by
slug.

## What this changed elsewhere

`last_input_tokens` became `last_context_tokens` holding
`window_tokens()` (all consumers updated: utilization, `/usage`
context line). Sub-agent cost metadata, eval ceilings, and builder
summaries keep reading `.input_tokens` — now unambiguous, and for
OpenAI-dialect sessions slightly smaller and more truthful than before.
Published live-run numbers in older notes reflect the semantics at
capture time; this note is the boundary.
