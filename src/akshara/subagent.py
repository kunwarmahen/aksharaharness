"""Sub-agents: agent-as-tool (book ch15).

The parent Agent gets ONE new tool, ``spawn_subagent``. Calling it spins
up a FRESH child Agent -- empty history, its own system prompt, a
filtered tool catalog -- runs it to completion, and returns only the
child's final answer as the tool result string. The parent's context
inflates by what the child CONCLUDED, never by the transcript it took
to conclude it (a 40-iteration research run comes back as one result
block, not 40 turns of tool noise).

Three constraints from the book, all enforced in code rather than left
to prompts:

1. ONE LEVEL DEEP. Children can't spawn children -- "spawn_subagent"
   is rejected in tools_allowed. Nested delegation compounds failure
   rates (three 85%-reliable agents in series ≈ 61% end-to-end).
2. BOUNDED. A per-session spawn budget plus a mandatory justification
   string per call. Spawning feels like doing work; the friction makes
   over-delegation visible.
3. COMPACT RESULTS. The child returns summary + cost metadata --
   never its transcript.

Two rules worth internalizing:

* Fresh context is the FEATURE: independent windows are most of the
  multi-agent value (Anthropic's research finding). Nothing from the
  parent conversation leaks into the child except the objective string.
* Scope restriction is enforced at the TOOL-CATALOG level, not trusted
  to the child's system prompt: the child physically has no other
  tools registered.

Don't use sub-agents for sequential chains where B consumes A's output
-- errors compound; do it inline. Use them for parallelizable,
self-contained subtasks (research branches, per-file surveys).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from akshara.agent import Agent
from akshara.errors import ProviderError, ToolError
from akshara.tools.base import Tool, ToolContext, ToolRegistry

SPAWN_TOOL_NAME = "spawn_subagent"

EXECUTE_CLAUSE = (
    "Before answering you MUST call at least one of your allowed tools; "
    "describing a call in prose without invoking it is a failure."
)

CHILD_SYSTEM_TEMPLATE = """You are a focused sub-agent spawned for one \
objective. Complete it, report, stop.

OBJECTIVE
{objective}

OUTPUT FORMAT (your final message must follow this)
{output_format}

TOOLS
You may use ONLY these tools: {tool_names}.
{execute_clause}

RULES
- Stop as soon as the objective is answered; no gold-plating.
- If the objective cannot be completed, SAY SO explicitly instead of \
inventing results.
- Budget: at most {max_iterations} model iterations.
"""


@dataclass(slots=True)
class SubagentResult:
    """What a spawn returns to the parent -- conclusions, not transcripts."""

    summary: str
    iterations_used: int
    tool_calls_made: int
    input_tokens: int
    output_tokens: int
    error: str | None = None


class SubagentSpawner:
    """Builds and runs child Agents on behalf of the spawn tool.

    One spawner per session: the budget counter lives here.
    """

    def __init__(self, parent: Agent, *, max_per_session: int = 5,
                 default_max_iterations: int = 20,
                 on_child_event: Callable[[int, Any], None] | None = None) -> None:
        self.parent = parent
        self.max_per_session = max_per_session
        self.default_max_iterations = default_max_iterations
        self.spawned = 0
        self.results: list[SubagentResult] = []  # observability / evals
        # optional live view of child internals: called as
        # on_child_event(spawn_number, StreamEvent). Deliberately NOT part
        # of the StreamEvent union -- child events are a UI concern with a
        # different provenance, not new loop vocabulary.
        self.on_child_event: Callable[[int, Any], None] | None = on_child_event

    # ---- validation --------------------------------------------------------

    def _validate(self, args: dict[str, Any]) -> tuple[str, str, list[str], int]:
        objective = args.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ToolError("missing required argument 'objective'")

        output_format = args.get("output_format")
        if not isinstance(output_format, str) or not output_format.strip():
            raise ToolError(
                "missing required argument 'output_format' -- vague "
                "requests produce rambling; name the exact shape of the answer"
            )

        justification = args.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            raise ToolError(
                "a non-empty 'justification' is required -- why can't this "
                "be done inline? (anti-over-delegation friction)"
            )

        allowed = args.get("tools_allowed")
        if not isinstance(allowed, list) or not allowed \
                or not all(isinstance(t, str) for t in allowed):
            raise ToolError(
                "'tools_allowed' must be a NON-EMPTY list of tool names -- "
                "narrower is better; scope restriction is enforced here, "
                "not asked for nicely"
            )
        known = set(self.parent.registry.names())
        unknown = [t for t in allowed if t not in known]
        if unknown:
            raise ToolError(
                f"unknown tool(s) {unknown}; available: {sorted(known)}"
            )
        if SPAWN_TOOL_NAME in allowed:
            # one level deep -- nested delegation compounds failure rates
            raise ToolError("sub-agents cannot spawn sub-agents")

        max_iterations = args.get("max_iterations", self.default_max_iterations)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
                or not (1 <= max_iterations <= 50):
            raise ToolError("'max_iterations' must be an integer in [1, 50]")

        return objective, output_format, list(dict.fromkeys(allowed)), max_iterations

    # ---- execution ---------------------------------------------------------

    def spawn(self, args: dict[str, Any]) -> SubagentResult:
        """Validate, build the child, run it to completion."""
        if self.spawned >= self.max_per_session:
            raise ToolError(
                f"sub-agent budget exhausted ({self.spawned}/"
                f"{self.max_per_session} used this session); do the work "
                "inline or ask the user to raise the budget"
            )

        objective, output_format, allowed, max_iterations = self._validate(args)
        self.spawned += 1  # counted when attempted, successful or not

        registry = ToolRegistry()
        for name in allowed:
            registry.register(self.parent.registry.get(name))

        system = CHILD_SYSTEM_TEMPLATE.format(
            objective=objective,
            output_format=output_format,
            tool_names=", ".join(allowed),
            execute_clause=EXECUTE_CLAUSE if allowed else "",
            max_iterations=max_iterations,
        )
        child = Agent(
            self.parent.provider,
            model=self.parent.model,
            system=system,
            tools=registry,
            cwd=self.parent.ctx.cwd,
            max_tokens=self.parent.max_tokens,
            max_iterations=max_iterations,
            permissions=self.parent.permissions,  # cannot escalate by being a sub-agent
            context_window=self.parent.context_window,
        )
        if self.on_child_event is not None:
            # Stream tee: the child's raw StreamEvents are PUSHED to the
            # observer tagged with this spawn's 1-based number, so a UI can
            # watch child progress live without the parent's context ever
            # seeing any of it. Still deliberately NOT part of the
            # StreamEvent union -- this is a UI seam with different
            # provenance, not new loop vocabulary (hence the loose Any).
            spawn_no = self.spawned  # already incremented: this child's number
            child.on_stream_event = (
                lambda event, n=spawn_no: self.on_child_event(n, event)
            )

        try:
            child.run(objective)
        except ProviderError as exc:
            # terminal provider failure INSIDE the child: data to the
            # parent (which may decide to retry later), not a crashed turn
            return self._record(SubagentResult(
                "", 0, 0, child.total_usage.input_tokens,
                child.total_usage.output_tokens,
                error=f"provider error inside sub-agent: {exc}",
            ))
        except RuntimeError as exc:
            # child.run() raises this on max_iterations without end_turn;
            # salvage the best-effort last assistant text
            partial = ""
            for message in reversed(child.history):
                if message.role == "assistant" and message.text().strip():
                    partial = message.text().strip()
                    break
            return self._record(SubagentResult(
                partial,
                sum(1 for m in child.history if m.role == "assistant"),
                self._count_calls(child),
                child.total_usage.input_tokens,
                child.total_usage.output_tokens,
                error=f"sub-agent hit its iteration cap: {exc}",
            ))

        return self._record(SubagentResult(
            self._final_text(child),
            sum(1 for m in child.history if m.role == "assistant"),
            self._count_calls(child),
            child.total_usage.input_tokens,
            child.total_usage.output_tokens,
        ))

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _count_calls(child: Agent) -> int:
        return sum(len(m.tool_calls()) for m in child.history)

    @staticmethod
    def _final_text(child: Agent) -> str:
        for message in reversed(child.history):
            if message.role == "assistant":
                text = message.text().strip()
                if text:
                    return text
        return ""

    def _record(self, result: SubagentResult) -> SubagentResult:
        self.results.append(result)
        return result


class SpawnSubagent(Tool):
    """The tool object the parent model sees; thin skin over the spawner."""

    name: ClassVar[str] = SPAWN_TOOL_NAME
    description: ClassVar[str] = (
        "Spawn a focused sub-agent with a fresh context window to complete "
        "one self-contained objective. It sees NOTHING from this "
        "conversation except what you pass here, uses only 'tools_allowed', "
        "and returns ONLY its final answer. Use for self-contained "
        "subtasks (research branches, per-file surveys); do NOT use for "
        "simple lookups you can do inline or sequential dependent steps."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "objective": {"type": "string",
                          "description": "Operationally specific task. This is "
                                         "the ONLY context the sub-agent gets."},
            "output_format": {"type": "string",
                              "description": "Required structure of the final "
                                             "answer, precisely named (e.g. "
                                             "'JSON {found: bool, path: string}')."},
            "tools_allowed": {"type": "array", "items": {"type": "string"},
                              "description": "Exact tool names the sub-agent may "
                                             "use. Narrower is better."},
            "justification": {"type": "string",
                              "description": "Why this needs a sub-agent instead "
                                             "of being done inline."},
            "max_iterations": {"type": "integer",
                               "description": "Model-iteration cap for the "
                                              "child (default 20)."},
        },
        "required": ["objective", "output_format", "tools_allowed", "justification"],
        "additionalProperties": False,
    }
    # conservative marking: children can do whatever their tools can do
    read_only: ClassVar[bool] = False

    def __init__(self, spawner: SubagentSpawner) -> None:
        self.spawner = spawner

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        objective = str(args.get("objective", ""))[:100]
        remaining = self.spawner.max_per_session - self.spawner.spawned
        return (f"spawn sub-agent (budget: {remaining} left): "
                f"{objective!r} · tools={args.get('tools_allowed')}")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        result = self.spawner.spawn(args)
        meta = (f"\n[sub-agent · {result.iterations_used} iteration(s) · "
                f"{result.tool_calls_made} tool call(s) · "
                f"{result.input_tokens}in/{result.output_tokens}out tokens]")
        if result.error:
            if result.summary:
                return (f"{result.summary}\n[INCOMPLETE -- {result.error}]{meta}")
            raise ToolError(result.error)
        return result.summary + meta
