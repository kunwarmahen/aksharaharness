"""Dollar figures for token counts.

The loop counts tokens (``Usage``); this module turns counts into
approximate dollars. It exists because ``/usage`` showing only tokens
answers half the question a learner actually has -- "what did this
session cost me?"

Honesty rules, up front:

* Prices here are LIST prices, hand-copied from each vendor's public
  price page and dated below. They WILL drift -- that is not a bug in
  this table, it is the nature of the market. The override file (see
  ``AKSHARA_PRICES``) is the intended mechanism for staying current;
  treat built-ins as sensible defaults, never gospel.
* An unknown model returns ``None`` -- callers OMIT the dollar figure
  rather than guess. A wrong price silently shown is worse than no
  price.
* Gateway resellers price differently from first-party list; if your
  receipts disagree with ``/usage``, your gateway wins.

Matching, in order (first hit wins):

1. exact slug ("claude-sonnet-4-5", "gpt-5-mini")
2. date-suffix stripped ("claude-opus-5-20260201" -> "claude-opus-5")
3. vendor prefix stripped ("anthropic/claude-sonnet-4-5" ->
   "claude-sonnet-4-5"; OpenRouter-style slugs)
4. longest family prefix ("claude-opus-" -> the Opus-tier row), so a
   brand-new point release still lands on its family's price

Long-context tiers are NOT distinguished: where a vendor splits short/
long pricing by request size, the SHORT-context row is used. Add an
override entry when that matters for you.

Billing math uses one convention everywhere (enforced in the adapters,
see notes/21): ``Usage.input_tokens`` counts ONLY tokens billed at the
full input rate; cached tokens appear exclusively in the cache counters
even though OpenAI-dialect wire formats fold them into the prompt
total. So cost is a plain weighted sum:

    input*in + cache_read*read + cache_write*write + output*out   / 1e6
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from akshara.errors import ConfigError
from akshara.types import Usage

# List-price snapshot: 2026-08 (Anthropic via current model docs; OpenAI
# via the published API pricing page). Per 1M tokens, USD.


@dataclass(frozen=True)
class ModelPrice:
    """One model's list price, per million tokens, USD.

    ``cache_read_per_mtok`` / ``cache_write_per_mtok`` default to None =
    "bill cached tokens at the full input rate". That is the conservative
    fallback (never UNDERstates cost) and matches vendors that publish no
    separate cache rate. Anthropic-family entries set both explicitly
    (write = 1.25x input, read = 0.1x input is their long-standing shape).
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


# Exact slugs first -- the common case, zero guessing.
_EXACT: dict[str, ModelPrice] = {
    # Anthropic (cache write 1.25x / read 0.1x of input across the family)
    "claude-fable-5": ModelPrice(10.00, 50.00, 1.00, 12.50),
    "claude-opus-5": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-7": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-6": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5": ModelPrice(3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00, 0.10, 1.25),
    # OpenAI ("-pro" rows publish no cached-input rate -> falls back to
    # full input price, which is exactly what those endpoints bill)
    "gpt-5-pro": ModelPrice(15.00, 120.00),
    "gpt-5.2-pro": ModelPrice(21.00, 168.00),
}

# Family prefixes second-longest-first -- new point releases inherit
# their family's price until someone overrides. Longest match wins, so
# "gpt-5.4-mini" must be listed before "gpt-5.4".
_PREFIXES: dict[str, ModelPrice] = {
    # Anthropic families
    "claude-opus-": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-": ModelPrice(3.00, 15.00, 0.30, 3.75),
    "claude-haiku-": ModelPrice(1.00, 5.00, 0.10, 1.25),
    # OpenAI families (short-context tier where the vendor splits)
    "gpt-5.6-sol": ModelPrice(4.00, 20.00, 0.40),
    "gpt-5.6-terra": ModelPrice(2.00, 12.00, 0.20),
    "gpt-5.6-luna": ModelPrice(0.20, 1.20, 0.02),
    "gpt-5.5": ModelPrice(5.00, 30.00, 0.50),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02),
    "gpt-5.4": ModelPrice(2.50, 15.00, 0.25),
    "gpt-5.2": ModelPrice(1.75, 14.00, 0.175),
    "gpt-5.1": ModelPrice(1.25, 10.00, 0.125),
    "gpt-5-mini": ModelPrice(0.25, 2.00, 0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.40, 0.005),
    "gpt-5": ModelPrice(1.25, 10.00, 0.125),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, 0.075),
    "gpt-4o": ModelPrice(2.50, 10.00, 1.25),
}

_DATE_SUFFIX = re.compile(r"-20\d{6}$")

# AKSHARA_PRICES: path to a JSON file of overrides/additions, applied on
# top of everything above. Shape:
#     {"my-model": {"input": 3.0, "output": 15.0},
#      "vendor/-prefixed-slug*": {"input": 1.0, "output": 2.0,
#                                 "cached_read": 0.1}}
# dollars per 1M tokens. Keys ending in "*" are family prefixes; user
# prefixes are consulted BEFORE built-in ones so overrides win.
_override_cache: tuple[dict[str, ModelPrice], dict[str, ModelPrice]] | None = None
_override_loaded_from: str | None = None


def _load_overrides() -> tuple[dict[str, ModelPrice], dict[str, ModelPrice]]:
    """Read $AKSHARA_PRICES once; (exact, prefix) dicts, empty if unset."""
    global _override_cache, _override_loaded_from
    path = os.environ.get("AKSHARA_PRICES", "")
    if _override_loaded_from == path and _override_cache is not None:
        return _override_cache
    _override_loaded_from = path
    exact: dict[str, ModelPrice] = {}
    prefixes: dict[str, ModelPrice] = {}
    if path:
        try:
            raw = json.loads(Path(path).expanduser().read_text())
        except OSError as exc:
            raise ConfigError(f"AKSHARA_PRICES unreadable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"AKSHARA_PRICES is not valid JSON: {exc}") from exc
        for key, fields in raw.items():
            try:
                price = ModelPrice(
                    input_per_mtok=float(fields["input"]),
                    output_per_mtok=float(fields["output"]),
                    cache_read_per_mtok=(
                        None if fields.get("cached_read") is None
                        else float(fields["cached_read"])),
                    cache_write_per_mtok=(
                        None if fields.get("cached_write") is None
                        else float(fields["cached_write"])),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(
                    f"AKSHARA_PRICES entry {key!r}: need at least "
                    f"'input' and 'output' (USD per 1M tokens)"
                ) from exc
            if key.endswith("*"):
                prefixes[key.rstrip("*")] = price
            else:
                exact[key] = price
    _override_cache = (exact, prefixes)
    return _override_cache


def _candidates(model: str) -> list[str]:
    """Slug spellings to try, most-specific first."""
    out = [model]

    def add(slug: str) -> None:
        if slug and slug not in out:
            out.append(slug)

    bare = _DATE_SUFFIX.sub("", model)
    add(bare)
    # Peel ONE leading "vendor/" at a time -- gateways nest them
    # ("openrouter/anthropic/claude-opus-5"), and any suffix may be the
    # recognizable slug.
    rest = model
    while "/" in rest:
        rest = rest.split("/", 1)[1]
        add(rest)
        add(_DATE_SUFFIX.sub("", rest))
    return out


def price_for(model: str) -> ModelPrice | None:
    """List price for a model slug, or None when genuinely unknown.

    Never guesses: callers must omit the dollar figure for None rather
    than invent one.
    """
    override_exact, override_prefixes = _load_overrides()
    candidates = _candidates(model)
    for slug in candidates:  # overrides win over built-ins at every stage
        if slug in override_exact:
            return override_exact[slug]
    for slug in candidates:
        if slug in _EXACT:
            return _EXACT[slug]
    for table in (override_prefixes, _PREFIXES):  # longest prefix wins
        for key in sorted(table, key=len, reverse=True):
            for slug in candidates:
                if slug.startswith(key):
                    return table[key]
    return None


def cost_of(usage: Usage, price: ModelPrice) -> float:
    """Dollars for one Usage under one price (see module docstring).

    Cached tokens billed at their own rates, falling back to the full
    input rate when a model publishes none -- conservative, never
    understated.
    """
    read_rate = (price.cache_read_per_mtok
                 if price.cache_read_per_mtok is not None
                 else price.input_per_mtok)
    write_rate = (price.cache_write_per_mtok
                  if price.cache_write_per_mtok is not None
                  else price.input_per_mtok)
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.cache_read_tokens * read_rate
        + usage.cache_write_tokens * write_rate
        + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000


def session_cost(usage_by_model: dict[str, Usage]) -> tuple[float, bool]:
    """(dollars, fully_priced) over per-model session buckets.

    Buckets whose model has no known price contribute tokens but no
    dollars -- reported honestly via ``fully_priced=False`` so the UI
    can say "priced models only" instead of implying completeness.
    """
    total = 0.0
    complete = True
    for model, usage in usage_by_model.items():
        price = price_for(model)
        if price is None:
            complete = False
            continue
        total += cost_of(usage, price)
    return total, complete
