"""The async payoff, measured live: ONE event loop, MANY conversations.

Run (needs a real provider key in .env / environment):

    uv run python examples/async_demo.py
    uv run python examples/async_demo.py --conversations 6
    uv run python examples/async_demo.py --provider openai

Runs the SAME N independent Q&A conversations twice through AsyncAgent:

  1. SEQUENTIALLY -- awaited one after another (the sync world's shape)
  2. CONCURRENTLY  -- all gathered on one event loop, sharing one pooled
     httpx.AsyncClient

Each session reports when it STARTED, when its FIRST TOKEN arrived, and
when it FINISHED. In the concurrent phase every start lands at ~0.0s and
first tokens interleave -- that overlap is the entire point of the async
conversion. The speedup line at the bottom is the receipt.

Honest caveat printed with the results: this measures wall-clock, not
token throughput. The provider serves roughly the same total tokens
either way; concurrency hides per-request latency behind other
sessions' waits instead of making any single request faster.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from akshara.async_agent import AsyncAgent
from akshara.config import default_model, load_settings
from akshara.providers import get_provider
from akshara.types import TextDelta

QUESTIONS = [
    "In exactly one short sentence: why is the sky blue?",
    "In exactly one short sentence: why do cats purr?",
    "In exactly one short sentence: why does ice float?",
    "In exactly one short sentence: why do leaves change color in autumn?",
    "In exactly one short sentence: why is the sea salty?",
    "In exactly one short sentence: why do we dream?",
]

SYSTEM = "You are answering rapid-fire trivia. One sentence, no preamble."


class Session:
    """One independent conversation plus its wall-clock statistics."""

    def __init__(self, provider, model: str, idx: int, question: str) -> None:
        self.stat = {"idx": idx, "question": question,
                     "start": None, "first_token": None,
                     "end": None, "answer": "", "error": None}
        self.agent = AsyncAgent(
            provider, model=model, system=SYSTEM, max_tokens=300,
            on_stream_event=self._on_event,
        )

    def _on_event(self, event) -> None:
        # Push-based tap: first TEXT token marks real provider contact.
        if self.stat["first_token"] is None and isinstance(event, TextDelta):
            self.stat["first_token"] = time.perf_counter()

    async def run(self) -> dict:
        self.stat["start"] = time.perf_counter()
        try:
            response = await self.agent.run(self.stat["question"])
            self.stat["answer"] = response.message.text().strip()
        except Exception as exc:  # keep the demo running on one bad session
            self.stat["error"] = f"{type(exc).__name__}: {exc}"
        self.stat["end"] = time.perf_counter()
        return self.stat


def _report(stats: list[dict], label: str) -> float:
    total = max(s["end"] for s in stats) - min(s["start"] for s in stats)
    print(f"\n=== {label} ===")
    for s in sorted(stats, key=lambda s: s["end"]):
        if s["error"]:
            print(f"[{s['idx']}] FAILED: {s['error']}")
            continue
        t0 = s["start"]
        print(f"[{s['idx']}] start {s['start'] - t0:5.2f}s  "
              f"first token {s['first_token'] - t0:5.2f}s  "
              f"done {s['end'] - t0:5.2f}s")
        print(f"     Q: {s['question']}")
        print(f"     A: {s['answer']}")
    print(f"-- {label}: {len(stats)} conversations in {total:.2f}s "
          f"({total / len(stats):.2f}s/conversation avg)")
    return total


async def main_async(conversations: int, provider_name: str) -> None:
    settings = load_settings(provider_name)
    model = default_model(provider_name)
    questions = QUESTIONS[:conversations]
    while len(questions) < conversations:  # allow --conversations > len(QUESTIONS)
        questions.append(QUESTIONS[len(questions) % len(QUESTIONS)])

    # ---- phase 1: sequential -------------------------------------------
    # A fresh provider per phase keeps connection pools comparable; within
    # a phase every session shares one client (realistic server shape).
    provider = get_provider(provider_name, settings)
    seq_stats = []
    for idx, question in enumerate(questions):
        seq_stats.append(await Session(provider, model, idx, question).run())
    seq_total = _report(seq_stats, f"SEQUENTIAL: {len(questions)} conversations, "
                                   f"one after another")

    # ---- phase 2: concurrent --------------------------------------------
    provider = get_provider(provider_name, settings)
    sessions = [Session(provider, model, i, q) for i, q in enumerate(questions)]
    conc_stats = await asyncio.gather(*(s.run() for s in sessions))
    conc_total = _report(conc_stats, f"CONCURRENT: {len(questions)} conversations, "
                                     f"one event loop")

    # ---- the receipt ------------------------------------------------------
    speedup = seq_total / conc_total if conc_total else 0.0
    print(f"\n=== SPEEDUP: {speedup:.1f}x  "
          f"({seq_total:.2f}s sequential -> {conc_total:.2f}s concurrent) ===")
    print("Wall-clock only: total tokens served are ~the same either way;")
    print("concurrency hides each request's latency behind the others' waits.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--conversations", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main_async(args.conversations, args.provider))


if __name__ == "__main__":
    main()
