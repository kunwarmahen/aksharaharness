"""A full agent turn -- model asks for a tool, we run it,
the model answers -- rendered with the SAME renderer the CLI uses.

Run:
    uv run python examples/tool_round_trip.py "what's in README.md?"
    uv run python examples/tool_round_trip.py --provider openai "..."

What to watch for:
  * the `-> read_file()` line: the model EMITTED arguments as streamed JSON
    fragments; collect() reassembled them into a dict before execution
  * the result panel: what read_file actually returned inside our sandbox
  * the final answer quotes real file contents -- evidence of the round trip
    wire -> events -> loop -> filesystem -> history -> wire -> answer

Read-only tools auto-approve, so no prompts; add write/bash requests and
you'd see the y/n gate instead.
"""

from __future__ import annotations

import argparse

from rich.console import Console

from akshara.agent import Agent
from akshara.cli.render import Renderer
from akshara.config import default_model, load_settings
from akshara.permissions import allow_read_only
from akshara.providers import get_provider
from akshara.tools import default_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("prompt", nargs="*", default=["what's in README.md?"])
    args = parser.parse_args()

    console = Console()
    provider = get_provider(args.provider, load_settings(args.provider))

    agent = Agent(
        provider,
        model=default_model(args.provider),
        system="You are a concise assistant with filesystem tools.",
        tools=default_registry(),
        permissions=allow_read_only,
    )
    renderer = Renderer(console)  # same renderer the REPL uses
    agent.on_stream_event = renderer  # pushed: text/thinking deltas as they stream

    console.print(f"[bold]{args.provider}[/bold] · {agent.model}\n")
    for event in agent.run_streaming(" ".join(args.prompt)):
        renderer(event)  # yielded: tool panels and the turn footer


if __name__ == "__main__":
    main()
