"""Skills: a procedure you wrote once, loaded only when it applies.

Builds two skills in a scratch directory, points an agent at them, and
asks a question that matches ONE of them. Watch three things:

* the ROSTER in the system prompt -- two lines, ~25 tokens each. That is
  the whole standing cost of having skills available.
* the model calling ``load_skill`` on its own. Nothing in the question
  says "skill"; the description is what earns the call.
* the second skill never loading. You do not pay for what does not apply.

Add ``--delegated`` to run a third skill that declares ``mode:
subagent``: its ``allowed-tools`` stop being advice and become a fence,
enforced by building the child's registry from exactly that list.

    uv run --env-file .env python examples/skills_demo.py
    uv run --env-file .env python examples/skills_demo.py --ollama
    uv run --env-file .env python examples/skills_demo.py --ollama --delegated
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from akshara.agent import Agent, ToolExecuted  # noqa: E402
from akshara.config import (  # noqa: E402
    default_context_window,
    default_model,
    load_settings,
)
from akshara.permissions import yolo  # noqa: E402
from akshara.providers import get_provider  # noqa: E402
from akshara.skills import enable_skills  # noqa: E402
from akshara.tools import default_registry  # noqa: E402

CHANGELOG = """\
---
name: changelog-entry
description: Add an entry to this project's CHANGELOG in its house format.
  Use when asked to record a change, write a changelog line, or note what
  shipped.
---

# Writing a changelog entry

House format, and it is picky on purpose:

1. Newest entries go at the TOP, under `## Unreleased`.
2. One line per change: `- <area>: <what changed, present tense>`.
3. The area is the package path without `src/` (`tools/glob`, not
   `src/akshara/tools/glob.py`).
4. No issue numbers, no author names. `git log` already knows both.

Example: `- tools/glob: match hidden files when the pattern asks for them`
"""

SUPPORT = """\
---
name: triage-bug
description: Triage an incoming bug report against this project -- reproduce
  it, classify it, decide who owns it. Use when handed a bug report, a
  stack trace, or a user complaint to sort out.
---

# Triaging a bug report

1. Reproduce before reading code. A bug you cannot reproduce is a
   question, not a bug.
2. Classify: wire format, loop invariant, tool behaviour, or UI.
3. Name the invariant it violates, if any. "It feels wrong" is not one.
"""

SURVEY = """\
---
name: file-survey
description: Survey a directory and report what lives there -- files, sizes,
  what each one is for. Use when asked to explore or map out a folder.
mode: subagent
allowed-tools: list_dir, read_file
output-format: One line per file -- name, then what it is for.
max-iterations: 8
---

# Surveying a directory

You are reading, not changing: you have no write tools and no shell.

1. `list_dir` first. Do not read a file before you know its neighbours.
2. Read only what you need to say what each file is FOR.
3. Report in the format you were given. Stop when the question is answered.
"""


def write_skills(root: Path, *, delegated: bool) -> None:
    """Lay out skill folders exactly as a project would keep them."""
    skills = {"changelog-entry": CHANGELOG, "triage-bug": SUPPORT}
    if delegated:
        skills["file-survey"] = SURVEY
    for name, text in skills.items():
        folder = root / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(text)


def main() -> None:
    delegated = "--delegated" in sys.argv
    provider_name = "ollama" if "--ollama" in sys.argv else "anthropic"
    provider = get_provider(provider_name, load_settings(provider_name))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_skills(root, delegated=delegated)
        (root / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n")

        agent = Agent(
            provider,
            model=default_model(provider_name),
            tools=default_registry(),
            cwd=root,
            # Everything here happens inside a TemporaryDirectory this
            # script just made, so the demo runs unattended. Do not copy
            # this line into anything pointed at a real workspace.
            permissions=yolo,
            context_window=default_context_window(provider_name),
        )
        skills = enable_skills(agent, root)

        print("=" * 70)
        print("THE ROSTER (this is the whole per-turn cost of having skills)")
        print("=" * 70)
        print(agent.prompt.get("skills"))

        question = (
            "Survey the skills directory and tell me what is in it."
            if delegated else
            "I just taught glob to match hidden files when the pattern "
            "asks for them. Record that, then show me the file."
        )
        print(f"\n{'=' * 70}\nASKING: {question}\n{'=' * 70}")

        for event in agent.run_streaming(question):
            if isinstance(event, ToolExecuted):
                first = event.result.content.splitlines()[0][:88]
                print(f"  → {event.call.name}({event.call.arguments}) "
                      f"\n      {first}")

        print(f"\nanswer: {agent.history[-1].text()[:600]}")
        if not delegated:
            print(f"\nCHANGELOG.md now reads:\n"
                  f"{(root / 'CHANGELOG.md').read_text()}")
        print(f"\nskills loaded this session: {skills.loaded or 'none'}")
        print(f"skills never touched: "
              f"{[n for n in skills.names() if n not in skills.loaded]}")
        if delegated:
            spawner = agent.subagents
            print(f"sub-agents spent: {spawner.spawned}/"
                  f"{spawner.max_per_session}")


if __name__ == "__main__":
    main()
