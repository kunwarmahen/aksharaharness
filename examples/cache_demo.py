"""Prompt caching, measured live: does call N+1 re-read call N's prefix?

Run (real model calls -- needs a key in .env):

    uv run python examples/cache_demo.py
    uv run python examples/cache_demo.py --provider openai   # automatic upstream

The economics of agent loops: every bounce re-mails the whole transcript,
and [tools] + [system] never change between calls. Anthropic-dialect
prompt caching exploits that shape -- mark the stable prefix with
``cache_control`` breakpoints (our placement: last tool, system, last
message) and later requests READ it back at ~0.1x price instead of
re-ingesting it.

Two phases, because gateways differ in what they REPORT:

  phase 1  the agent loop, streamed -- proves cache_control=True rides
           through the normal Agent surface without breaking anything
           (this gateway sends no usage counters on streamed calls at
           all, so no numbers are expected here);
  phase 2  the measurement -- two NON-streaming calls sharing one long
           system prompt, printing per-call token accounting. This is
           the receipt: expect call 1 to ingest the whole prefix and
           call 2 to read most of it back as cached tokens.
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from akshara import Agent, default_registry
from akshara.config import default_model, load_settings
from akshara.providers import get_provider
from akshara.types import Message, TextBlock

# A long, boring reference document: long enough to clear the minimum
# cacheable-prefix threshold (~1k tokens), stable enough to be worth
# caching. Content doesn't matter; length and immutability do.
_REFERENCE = (
    "PROJECT STYLE REFERENCE (read carefully before answering):\n"
    "The team writes in complete sentences. Numbers under ten are spelled "
    "out except in measurements. Headings use sentence case.\n"
) + "".join(
    f"Rule {n}: always {verb} the {noun} before {verb2}ing it; never {verb2} "
    f"a {noun} that has not been {verb}ed.\n"
    for n, (verb, verb2, noun) in enumerate([
        ("verify", "log", "payload"), ("validate", "render", "schema"),
        ("annotate", "ship", "change"), ("review", "merge", "branch"),
        ("measure", "optimize", "query"), ("document", "publish", "result"),
        ("test", "refactor", "module"), ("stage", "commit", "revision"),
    ] * 40, start=1)  # ~2600 words -- comfortably past the cache floor
)


def _usage_line(label: str, u) -> str:
    return (f"  {label:<7} input={u.input_tokens:>6}  "
            f"cached-read={u.cache_read_tokens:>6}  "
            f"cache-write={u.cache_write_tokens:>6}  out={u.output_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    args = parser.parse_args()

    console = Console()
    settings = load_settings(args.provider)
    provider = get_provider(args.provider, settings, cache_control=True)
    # A unique suffix per phase busts any cache entry left by earlier runs
    # (entries live ~5 min and refresh on hit) so call 1 is genuinely COLD.
    model = default_model(args.provider)
    ref_phase1 = _REFERENCE + f"\n(session {time.time_ns()}-a)"
    ref_phase2 = _REFERENCE + f"\n(session {time.time_ns()}-b)"
    console.print(f"[bold]{args.provider}[/bold] · cache_control=True · "
                  f"system ≈ {len(ref_phase2) // 4} tokens\n")

    # ---- phase 1: the agent loop (streamed) ---------------------------------
    agent = Agent(provider, model=model,
                  system=ref_phase1, tools=default_registry())
    console.print("[bold]phase 1 · agent loop (streamed)[/bold]")
    for question in [
        "Per the style rules: what must you do to a payload before logging "
        "it? One short sentence.",
        "And what must happen before shipping a change? One short sentence.",
    ]:
        answer = agent.run(question).message.text()
        console.print(f"  {answer}")
    u = agent.total_usage
    console.print(f"  [dim]totals: in={u.input_tokens} "
                  f"read={u.cache_read_tokens} write={u.cache_write_tokens} "
                  "(many gateways report NO usage on streamed calls -- that "
                  "is a reporting gap, not an error)[/dim]\n")

    if args.provider == "openai":
        console.print("OpenAI dialect: caching is automatic upstream -- there "
                      "is nothing to send. Hits appear only when the gateway "
                      "reports prompt_tokens_details.cached_tokens.")
        return

    # ---- phase 2: the measurement (non-streaming) ---------------------------
    console.print("[bold]phase 2 · the measurement (non-streaming, "
                  "shared prefix)[/bold]")
    messages = [Message("user",
                        [TextBlock("What is rule 3 about? One short "
                                   "sentence.")])]
    usages = []
    for i in range(2):
        r = provider.complete(messages=messages,
                              system=ref_phase2, tools=[],
                              model=model, max_tokens=100)
        usages.append(r.usage)
        console.print(_usage_line(f"call {i + 1}", r.usage))

    first, second = usages
    console.print("\n[bold]verdict[/bold]")
    if first.cache_read_tokens == 0 and second.cache_read_tokens > 0 \
            and second.input_tokens < first.input_tokens:
        console.print(f"[green]CACHE HIT confirmed:[/green] cold call 1 "
                      f"ingested {first.input_tokens} fresh tokens; call 2 "
                      f"re-read {second.cache_read_tokens} of them and paid "
                      f"for only {second.input_tokens} fresh ones -- the "
                      "stable prefix stopped being re-billed, exactly as "
                      "designed.")
    elif second.cache_read_tokens > 0:
        console.print(f"[green]cache active:[/green] call 2 read "
                      f"{second.cache_read_tokens} prefix tokens back "
                      f"(call 1 was already warm: read="
                      f"{first.cache_read_tokens}).")
    else:
        console.print("[yellow]no cache activity reported:[/yellow] this "
                      "gateway/model did not honor or report prompt caching. "
                      "The wire encoding is still exercised; billing benefit "
                      "unproven here.")


if __name__ == "__main__":
    main()
