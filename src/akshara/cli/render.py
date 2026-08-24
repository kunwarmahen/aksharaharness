"""Rich rendering of AgentEvents -- kept separate so the REPL stays logic-only."""

from __future__ import annotations

import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from akshara.agent import ToolExecuted, TurnEnd
from akshara.pricing import cost_of, price_for
from akshara.types import (
    EndEvent,
    RedactedThinking,
    StartEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStart,
)

RESULT_PREVIEW_CHARS = 400


class Renderer:
    """Consumes one AgentEvent stream and paints the terminal."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._turn_tokens = 0
        self._thinking_seen: set[int] = set()
        self._in_thinking = False

    def __call__(self, event) -> None:
        match event:
            case StartEvent(model=model):
                self._turn_tokens = 0
                self._thinking_seen.clear()
                self._in_thinking = False
                self.console.print(f"[dim]· {model}[/dim]")

            case TextDelta(text=fragment):
                if self._in_thinking:  # close the dim reasoning line first
                    self._in_thinking = False
                    self.console.print()
                # Raw streaming text: markup/highlight OFF so model output
                # like "[TODO]" renders literally instead of as rich tags.
                self.console.print(fragment, end="", markup=False,
                                   highlight=False, soft_wrap=True)

            case ThinkingDelta(index=i, text=fragment, signature=_):
                if i not in self._thinking_seen:  # lazy start-of-block header
                    self._thinking_seen.add(i)
                    self._in_thinking = True
                    self.console.print("\n[dim italic]· thinking[/dim italic]")
                if fragment:  # signature fragments accumulate silently
                    self.console.print(fragment, end="", style="dim italic",
                                       markup=False, highlight=False,
                                       soft_wrap=True)

            case RedactedThinking(index=_, data=payload):
                # Ciphertext -- nothing readable to show, but the user
                # should know reasoning happened and was withheld.
                if self._in_thinking:
                    self._in_thinking = False
                    self.console.print()
                self.console.print(
                    f"\n[dim italic]· redacted reasoning "
                    f"({len(payload)} chars, encrypted)[/dim italic]"
                )

            case ToolCallStart(index=_, id=_, name=name):
                self._thinking_seen.clear()  # a new block starts; header again OK
                self.console.print(f"\n[dim]→ {name}()[/dim]")

            case ToolCallDelta():
                pass  # argument fragments; shown in full in the result panel

            case ToolExecuted(call=call, result=result):
                self._render_tool(call.name, call.arguments, result.content,
                                  result.is_error)

            case EndEvent(stop_reason=_, usage=usage):
                self._turn_tokens = usage.input_tokens + usage.output_tokens

            case TurnEnd(reason="end_turn", response=response, iterations=n):
                self.console.print()  # close the streamed line
                if response is not None:
                    if not response.message.text().strip():
                        # end_turn with only thinking/tool noise -- say so
                        # rather than leaving the user staring at silence
                        self.console.print("[dim](no text in reply)[/dim]")
                    # $ only when the slug has a known list price; an
                    # unknown model shows no figure at all, never a guess.
                    price = price_for(response.model)
                    cost = (f" · ~${cost_of(response.usage, price):.4f}"
                            if price is not None else "")
                    self.console.print(
                        f"[dim]── {response.stop_reason} · "
                        f"{response.usage.input_tokens} in / "
                        f"{response.usage.output_tokens} out"
                        f"{cost} · {n} iteration(s)[/dim]"
                    )

            case TurnEnd(reason=reason, response=None, iterations=n):
                self.console.print(f"\n[yellow]── turn ended: {reason} "
                                   f"(after {n} iteration(s))[/yellow]")

    def _render_tool(self, name: str, args: dict, output: str, is_error: bool) -> None:
        style = "red" if is_error else "cyan"
        title = f"{name}()" + ("  [error]" if is_error else "")
        preview = output[:RESULT_PREVIEW_CHARS]
        if len(output) > RESULT_PREVIEW_CHARS:
            preview += f"\n[... {len(output) - RESULT_PREVIEW_CHARS} more chars ...]"
        # Text() renders raw (no markup interpretation of model output)
        content = Group(
            Syntax(json.dumps(args, indent=2), "json", background_color="default"),
            Text("\n" + preview),
        )
        self.console.print(Panel(content, title=title, border_style=style,
                                 title_align="left"))


class ChildStreamView:
    """One spawned child's StreamEvents, drawn nested under the parent turn.

    Owns its state per child and never touches the parent Renderer's -- two
    agents sharing mutable rendering state is exactly the bug that only
    shows up mid-turn. Framing derives from the events themselves:
    StartEvent opens a block, EndEvent closes it -- one block PER MODEL
    CALL, which is the truthful unit (a multi-iteration child shows
    several). The parent's own spawn_subagent result panel (report + cost
    metadata) closes the visual story.

    Only StreamEvents arrive here -- ToolExecuted/TurnEnd belong to the
    run_streaming generator, which child.run() consumes internally.
    """

    def __init__(self, console: Console, number: int) -> None:
        self.console = console
        self.number = number
        self._thinking_seen: set[int] = set()
        self._in_thinking = False

    def __call__(self, event) -> None:
        n = self.number
        match event:
            case StartEvent(model=model):
                self._thinking_seen.clear()
                self._in_thinking = False
                self.console.print(f"\n[dim]┌ child {n} · {model}[/dim]")

            case TextDelta(text=fragment):
                if self._in_thinking:  # close the dim reasoning line first
                    self._in_thinking = False
                    self.console.print()
                # grey = a secondary channel speaking; markup OFF for the
                # same literal-rendering reason as the parent's text
                self.console.print(fragment, end="", style="bright_black",
                                   markup=False, highlight=False,
                                   soft_wrap=True)

            case ThinkingDelta(index=i, text=fragment, signature=_):
                if i not in self._thinking_seen:
                    self._thinking_seen.add(i)
                    self._in_thinking = True
                    self.console.print(
                        f"\n[dim italic]· child {n} thinking[/dim italic]")
                if fragment:
                    self.console.print(fragment, end="", style="dim italic",
                                       markup=False, highlight=False,
                                       soft_wrap=True)

            case RedactedThinking(index=_, data=payload):
                if self._in_thinking:
                    self._in_thinking = False
                    self.console.print()
                self.console.print(
                    f"[dim italic]· child {n} redacted reasoning "
                    f"({len(payload)} chars, encrypted)[/dim italic]")

            case ToolCallStart(index=_, id=_, name=name):
                self._thinking_seen.clear()
                self.console.print(f"\n[dim]│ child {n} → {name}()[/dim]")

            case EndEvent(stop_reason=reason, usage=usage):
                self.console.print(
                    f"\n[dim]└ child {n} · {reason} · "
                    f"{usage.input_tokens} in / {usage.output_tokens} out[/dim]"
                )

            case _:
                pass  # ToolCallDelta fragments etc.


class SubagentTee:
    """The spawner's on_child_event target: routes (number, event) pairs to
    per-child views so every line stays attributable to its spawn."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._views: dict[int, ChildStreamView] = {}

    def __call__(self, number: int, event) -> None:
        view = self._views.get(number)
        if view is None:
            view = self._views[number] = ChildStreamView(self.console, number)
        view(event)
