"""THE AGENT LOOP, watched event by event.

Run:
    uv run python examples/agent_loop_demo.py
    uv run python examples/agent_loop_demo.py "what is in pyproject.toml?"
    uv run python examples/agent_loop_demo.py --deny-all      # denial-as-data

While it runs, PRESS CTRL-C DURING A TURN: the turn cancels, outstanding
tool calls get synthesized error results (the resumable-history
invariant), and the next prompt still works. That recovery path is the
single most important behavior in this file's sibling, akshara/agent.py.

The two channels of an agent turn:

* PUSHED  : raw StreamEvents (text/thinking deltas...) arrive via
  ``agent.on_stream_event`` WHILE each model response streams. They
  cannot be yielded -- collect() owns the pull -- so the UI subscribes.
* YIELDED : ToolExecuted and TurnEnd come out of run_streaming() itself,
  in execution order, AFTER all of a response's stream events.

The permission gate is just a callable: here we reuse the CLI's y/n
prompt factory, or pass deny_all to watch the model recover from
"Permission denied by user." arriving as DATA.
"""

from __future__ import annotations

import argparse

from rich.console import Console

from akshara.agent import Agent, ToolExecuted, TurnEnd
from akshara.cli.repl import confirm_gate
from akshara.config import default_model, load_settings
from akshara.permissions import allow_read_only, deny_all
from akshara.providers import get_provider
from akshara.tools import default_registry
from akshara.types import (
    StartEvent,
    RedactedThinking,
    TextDelta,
    ThinkingDelta,
    EndEvent,
    ToolCallDelta,
    ToolCallStart,
)


def make_painter(console: Console):
    """Annotated stream-event painter: one line per event KIND, so you can
    see the union's shape while it happens."""
    seen_thinking: set[int] = set()

    def paint(event) -> None:
        match event:
            case StartEvent(model=model):
                console.print(f"[dim]  StreamEvent StartEvent  model={model}[/dim]")
            case TextDelta(text=text):
                console.print(text, end="", markup=False, highlight=False,
                              soft_wrap=True)
            case ThinkingDelta(index=i, text=text, signature=_):
                if i not in seen_thinking:
                    seen_thinking.add(i)
                    console.print("[dim italic]  StreamEvent ThinkingDelta"
                                  " (reasoning follows)[/dim italic]")
                    console.print("[dim italic]· thinking[/dim italic]", end="")
                if text:
                    console.print(text, end="", style="dim italic",
                                  markup=False, highlight=False, soft_wrap=True)
            case RedactedThinking(index=_, data=payload):
                console.print(f"[dim italic]  StreamEvent RedactedThinking "
                              f"({len(payload)} chars of ciphertext -- arrives "
                              "whole, no deltas)[/dim italic]")
            case ToolCallStart(index=_, id=cid, name=name):
                console.print(f"\n[dim]  StreamEvent ToolCallStart "
                              f"{name}() id={cid}[/dim]")
            case ToolCallDelta(index=_, partial_json=piece):
                # argument fragments -- shown raw to MAKE the point that
                # JSON arrives in pieces and is parsed once, at the end
                console.print(f"[dim]    +{piece!r}[/dim]")
            case EndEvent(stop_reason=reason, usage=u):
                console.print(f"\n[dim]  StreamEvent EndEvent     "
                              f"{reason} ({u.input_tokens}in/{u.output_tokens}out)[/dim]")

    return paint


def show_history_shape(agent: Agent, console: Console) -> None:
    """The artifact the whole design is about: ONE internal history that
    any provider dialect can encode."""
    console.print("\n[bold]history shape[/bold]")
    for i, message in enumerate(agent.history):
        kinds = [type(b).__name__ for b in message.content]
        console.print(f"  [{i}] {message.role:9} {kinds}")
    console.print(
        "[dim]every tool_call id above has a matching tool_result --"
        " that invariant is what keeps the next request alive.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--deny-all", action="store_true",
                        help="deny every gated call; watch the model adapt")
    parser.add_argument("prompt", nargs="*",
                        default=["Use bash to run exactly: echo hello-from-the-loop"])
    args = parser.parse_args()

    console = Console()
    if args.deny_all:
        permissions = deny_all
    else:
        permissions = confirm_gate(console)  # read-only tools still auto-run

    agent = Agent(
        get_provider(args.provider, load_settings(args.provider)),
        model=default_model(args.provider),
        system="You are a concise assistant with shell and filesystem tools.",
        tools=default_registry(),
        permissions=permissions,
    )

    console.print(f"[bold]{args.provider}[/bold] · {agent.model}"
                  f" · gate={'deny_all' if args.deny_all else 'y/n prompt'}\n")
    agent.on_stream_event = make_painter(console)

    try:
        for event in agent.run_streaming(" ".join(args.prompt)):
            match event:
                case ToolExecuted(call=call, result=result):
                    flag = "ERROR (fed back as data)" if result.is_error else "ok"
                    preview = result.content.replace("\n", " ")[:80]
                    console.print(f"\n[bold]YIELDED ToolExecuted[/bold] "
                                  f"{call.name}() -> {flag}: {preview!r}")
                case TurnEnd(reason="end_turn", response=r, iterations=n):
                    console.print(f"\n\n[bold]YIELDED TurnEnd[/bold] end_turn "
                                  f"after {n} iteration(s)")
                    assert r is not None
                    console.print(f"[bold]final answer:[/bold] {r.message.text()}")
                case TurnEnd(reason=reason, iterations=n):
                    console.print(f"\n[bold yellow]YIELDED TurnEnd[/bold yellow] "
                                  f"{reason} after {n} iteration(s)")
    except KeyboardInterrupt:
        console.print("\n[yellow](cancelled)[/yellow]")

    show_history_shape(agent, console)


if __name__ == "__main__":
    main()
