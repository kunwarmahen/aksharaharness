"""ask_user: the tool that pauses a turn to talk to its human.

Every other tool answers a question the MODEL has about the world;
this one lets the model ask one of the USER. Ambiguity is normal in
real tasks ("which config file did you mean?", "Postgres or SQLite?",
"may I delete it?") and a harness that can't route that question to
the person who knows the answer forces the model to guess.

Three pieces:

* ``UserChannel``      -- WHERE the human is. A tiny protocol: ``ask()``
  blocks until an answer exists. The REPL supplies stdin; the web UI
  supplies a websocket round-trip; tests supply a canned lambda.
* ``AskUser(Tool)``    -- what the MODEL sees. Validates args, calls the
  channel, formats the answer as the tool-result string. ``read_only``
  because asking costs nothing: it never gates.
* ``UserUnavailable``  -- what happens when NOBODY is home (headless
  one-shot, piped stdin, evals, build agents). The tool raises instead
  of returning data -- deliberately ``BaseException``, so no
  ``except Exception`` in the loop converts "no user" into a result the
  model would try to explain away with a guess (see errors.py).

The blocking is honest, not a hack: tools run on worker threads, and
"waiting for a human" is IO like any other -- bounded by the human's
patience rather than a timeout, which is exactly right for a question
that was important enough to stop work over.

Sub-agents inherit the channel by reference when their catalog includes
this tool instance -- a child pausing to consult the same human through
the same UI is coherent, and needs no extra wiring.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Protocol

from akshara.errors import ToolError, UserUnavailable
from akshara.tools.base import Tool, ToolContext

MAX_CHOICES = 6


class UserChannel(Protocol):
    """How the harness reaches its human. One method; implement and go.

    Contract: block until the human answers, return their answer as a
    string. Raise KeyboardInterrupt if the user cancels the turn while
    the question is pending (same semantics as Ctrl-C mid-turn).
    Implementations may raise EOFError/UserUnavailable directly when
    there is no human at all.
    """

    def ask(self, question: str, choices: list[str],
            context: str = "") -> str: ...


class TerminalChannel:
    """stdin fallback: numbered choices plus free text, one input() line.

    Used by the REPL and one-shot mode. An empty line re-prompts (an
    answer was important enough to ask for); EOF means nobody is
    reading -- translated to UserUnavailable so headless runs fail the
    turn loudly instead of hanging on input() forever.
    """

    def __init__(self, input_fn: Callable[[str], str] = input) -> None:
        self._input = input_fn  # injectable for tests / alternate frontends

    def ask(self, question: str, choices: list[str],
            context: str = "") -> str:
        print(f"\nthe agent asks: {question}")
        if context.strip():
            print(f"  ({context.strip()})")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        prompt = f"[1-{len(choices)}, or your own answer]> " if choices else "> "
        while True:
            try:
                answer = self._input(prompt).strip()
            except EOFError:
                raise UserUnavailable(
                    "ask_user got EOF: no interactive user is attached "
                    "(piped stdin?)") from None
            if answer:
                if choices and answer.isdigit() and 1 <= int(answer) <= len(choices):
                    return choices[int(answer) - 1]
                return answer
            print("  (empty -- say something, or ctrl-c to cancel)")


def _validated_args(
        args: dict[str, Any]) -> tuple[str, list[str], str]:
    """Pull (question, choices, context) out of model-supplied JSON."""
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ToolError("missing required argument 'question' (a non-empty string)")
    raw_choices = args.get("choices", [])
    if raw_choices is None:
        raw_choices = []
    if not isinstance(raw_choices, list):
        raise ToolError("'choices' must be a list of short option strings")
    if len(raw_choices) > MAX_CHOICES:
        raise ToolError(f"too many choices ({len(raw_choices)}); max {MAX_CHOICES} "
                        "-- offer the best few, the user can always type more")
    choices: list[str] = []
    for choice in raw_choices:
        if not isinstance(choice, str) or not choice.strip():
            raise ToolError("every entry in 'choices' must be a non-empty string")
        choices.append(choice)
    context = args.get("context", "")
    if context is None:
        context = ""
    if not isinstance(context, str):
        raise ToolError("'context' must be a string")
    return question, choices, context


class AskUser(Tool):
    """The tool object the model sees; thin skin over a UserChannel."""

    name: ClassVar[str] = "ask_user"
    description: ClassVar[str] = (
        "Pause and ask the HUMAN OPERATOR a question, then continue with "
        "their answer. Use when proceeding on a guess would waste real "
        "work or touch something irreversible: which of two valid options "
        "they want, a missing fact only they know, permission for a step "
        "beyond the obvious. Do NOT use for facts you can look up with "
        "your other tools. Offer 'choices' when the answer is likely one "
        "of a few known options."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "One specific question. Self-contained "
                                        "-- the human may not be watching the "
                                        "transcript."},
            "choices": {"type": "array", "items": {"type": "string"},
                        "description": f"Optional short labeled options "
                                       f"(max {MAX_CHOICES}); the user can "
                                       f"still type a different answer."},
            "context": {"type": "string",
                        "description": "Optional: why you are asking / what "
                                       "hinges on the answer."},
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    read_only: ClassVar[bool] = True  # asking is free; never gates

    def __init__(self, channel: UserChannel | None) -> None:
        # None == headless: any call fails the turn loudly (UserUnavailable)
        # rather than pretending an answer arrived. Interactive frontends
        # pass their channel; evals/builds/piped runs pass nothing.
        self.channel = channel

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        question = str(args.get("question", ""))[:100]
        n = len(args.get("choices") or [])
        return f"ask the user{f' ({n} options)' if n else ''}: {question!r}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        question, choices, context = _validated_args(args)
        if self.channel is None:
            raise UserUnavailable(
                "ask_user ran with no interactive user attached (headless "
                "run). Failing the turn rather than guessing.")
        answer = self.channel.ask(question, choices, context)
        marker = ""
        for i, choice in enumerate(choices, 1):
            if answer == choice:
                marker = f" (picked option {i}/{len(choices)})"
                break
        return f"user replied:{marker} {answer}"
