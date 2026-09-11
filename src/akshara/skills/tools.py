"""The two tools that put skills in the model's hands.

``load_skill`` is the whole feature from the model's side: one tool,
one argument, and the result is a set of instructions to follow. It is
read_only on purpose -- it reads a Markdown file the OPERATOR wrote and
put in their own repo. Gating that behind an approval prompt would
train people to mash 'y' on the one tool that is definitionally safe,
and the instructions it returns still reach the world only through
tools that gate normally (bash, write_file, web_fetch).

``run_skill`` is the delegated twin, registered only when some skill
declares ``mode: subagent``. It is where ``allowed-tools`` stops being a
note and becomes a fence: the body runs in a fresh child agent holding
exactly those tools and nothing else, and only the child's conclusion
comes back. It is NOT read_only -- the child can do whatever its tools
can do, and every one of them still gates individually inside the child.

``list_skills`` exists for the long-roster case only -- past
ROSTER_LIMIT the prompt carries names without descriptions, and this is
where the descriptions went. Below that it is never registered: the
roster already says everything it would.
"""

from __future__ import annotations

from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str

#: Prefix on every delivered body. The model has just pulled a document
#: mid-turn; say plainly what it is and what to do with it.
DELIVERY_HEADER = (
    "Skill {name!r} -- follow these instructions for the current task. "
    "They were written for this project and override your general habits "
    "where the two disagree."
)


class LoadSkill(Tool):
    """Tier 2: hand over one skill's instructions."""

    name: ClassVar[str] = "load_skill"
    description: ClassVar[str] = (
        "Load the full instructions for one of the skills listed in your "
        "system prompt. Call this BEFORE doing work the skill covers -- "
        "the roster line is only a summary; the real procedure, including "
        "any files and commands it expects, is in what this returns."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact skill name from the roster in your "
                               "system prompt (e.g. 'pr-review').",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    read_only: ClassVar[bool] = True  # reads a Markdown file the operator wrote

    def __init__(self, skills: Any) -> None:
        self.skills = skills

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"load skill: {args.get('name')!r}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        name = require_str(args, "name").strip()
        skill = self.skills.get(name)
        if skill is not None and skill.delegated:
            # Handing the body over inline would quietly undo the fence the
            # author asked for: the point of mode:subagent is that these
            # instructions run with THOSE tools, not with yours.
            raise ToolError(
                f"skill {name!r} runs as a scoped sub-agent -- call "
                f"run_skill with name={name!r} and a 'task' instead. Its "
                f"instructions are not loaded inline on purpose.")
        try:
            skill = self.skills.load(name)
        except PermissionError:
            raise ToolError(
                f"skill {name!r} is disabled by the operator this session"
            ) from None
        except KeyError:
            # Errors are data: name what exists instead of failing the turn.
            near = self.skills.suggestions(name)
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise ToolError(
                f"no such skill: {name!r}. Available: "
                f"{', '.join(self.skills.available()) or '(none)'}.{hint}"
            ) from None

        lines = [DELIVERY_HEADER.format(name=skill.name)]
        # Tier 3's anchor: bundled files are addressed from the skill's own
        # directory, and the model cannot guess an absolute path.
        lines.append(f"Files bundled with this skill live in: {skill.directory} "
                     f"(read them with read_file when the steps below say so).")
        if skill.allowed_tools:
            lines.append("Tools this skill expects: "
                         + ", ".join(skill.allowed_tools) + ".")
            # Narrowing, not widening: naming a tool here never grants it.
            # Saying which ones are absent beats failing four steps in.
            missing = self.skills.missing_tools(skill)
            if missing:
                lines.append(
                    "NOT available in this session: " + ", ".join(missing)
                    + " -- adapt the steps that need them, or say plainly "
                      "that the skill cannot be completed here.")
        lines.append("")
        lines.append(skill.body)
        return "\n".join(lines)


class RunSkill(Tool):
    """Tier 2, fenced: run one skill in a scoped sub-agent."""

    name: ClassVar[str] = "run_skill"
    description: ClassVar[str] = (
        "Run one of the skills marked [delegated] in your system prompt. "
        "It executes in a SEPARATE agent that sees nothing of this "
        "conversation except the task you pass, may use only the tools the "
        "skill declares, and returns only its final answer. Use it for the "
        "self-contained jobs those skills describe; everything else is a "
        "plain load_skill."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact name of a [delegated] skill from the "
                               "roster in your system prompt.",
            },
            "task": {
                "type": "string",
                "description": "The specifics this run is about. It is the "
                               "ONLY context the child gets besides the "
                               "skill's own instructions, so name files, "
                               "branches and targets explicitly.",
            },
        },
        "required": ["name", "task"],
        "additionalProperties": False,
    }
    #: The child can do whatever its tools can do. Its calls still gate
    #: individually (it inherits the parent's permission function), but the
    #: TOOL ITSELF is not read-only and must not be auto-approved.
    read_only: ClassVar[bool] = False

    def __init__(self, skills: Any) -> None:
        self.skills = skills

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        name = str(args.get("name", ""))
        skill = self.skills.get(name)
        tools = ", ".join(skill.allowed_tools) if skill else "?"
        return (f"run skill {name!r} in a sub-agent (tools: {tools}) -- "
                f"{str(args.get('task', ''))[:80]!r}")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        name = require_str(args, "name").strip()
        task = require_str(args, "task").strip()
        skill = self.skills.get(name)
        if skill is None:
            near = self.skills.suggestions(name)
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise ToolError(
                f"no such skill: {name!r}. Available: "
                f"{', '.join(self.skills.available()) or '(none)'}.{hint}")
        if self.skills.is_disabled(name):
            raise ToolError(
                f"skill {name!r} is disabled by the operator this session")
        if not skill.delegated:
            raise ToolError(
                f"skill {name!r} runs inline -- call load_skill with "
                f"name={name!r} and follow the instructions yourself.")
        if not task:
            raise ToolError(
                "'task' must not be empty -- it is the only context the "
                "child gets besides the skill's own instructions")

        result = self.skills.delegate(skill, task)
        meta = (f"\n[skill {skill.name!r} · sub-agent · "
                f"{result.iterations_used} iteration(s) · "
                f"{result.tool_calls_made} tool call(s) · "
                f"{result.input_tokens}in/{result.output_tokens}out tokens]")
        if result.error:
            if result.summary:
                return f"{result.summary}\n[INCOMPLETE -- {result.error}]{meta}"
            raise ToolError(result.error)
        return result.summary + meta


class ListSkills(Tool):
    """The long-roster hatch: descriptions that no longer fit the prompt."""

    name: ClassVar[str] = "list_skills"
    description: ClassVar[str] = (
        "List every available skill with what it covers. Use this when the "
        "roster in your system prompt shows names only and you need to know "
        "which one fits the task, then call load_skill with the name."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    read_only: ClassVar[bool] = True

    def __init__(self, skills: Any) -> None:
        self.skills = skills

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"list {len(self.skills)} skill(s)"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        active = self.skills.active()
        if not active:
            return "No skills are available in this session."
        return "\n".join(skill.roster_line() for skill in active)
