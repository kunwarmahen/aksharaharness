"""The live skill set: what is on disk, what the prompt says about it,
and what the model may pull.

Three tiers of cost, from the format's design (notes/30):

    tier 1  name + description, in the system prompt, EVERY turn
    tier 2  the body, as a tool result, only when the model asks
    tier 3  bundled files, via read_file, only if the body sends it

This module owns tier 1 and hands out tier 2. Two constraints shape it:

* THE ROSTER IS FROZEN AT SESSION START -- the same rule env_context
  lives under. ``--cache`` caches the request prefix INCLUDING the
  system prompt, so a roster that re-ranked itself per turn would bust
  the cache on every single request. That rules out the obvious trick
  (BM25 the roster against the transcript) and leaves a static list,
  which is the honest shape anyway: the operator wrote these, there are
  rarely more than a dozen, and a name plus one sentence is ~25 tokens.
* ONE TOOL, NOT N. Registering each skill as its own tool would walk
  straight into the tool cliff (selector.py: accuracy falls off between
  30 and 50 tools) and spend a schema per skill on every request.
  ``load_skill`` is one tool whose argument happens to be a name.

Past ``ROSTER_LIMIT`` skills the roster degrades to names only and
``list_skills`` registers alongside -- still static, still cache-safe,
with the descriptions moved from every request to one tool result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from akshara.prompt import attach_prompt
from akshara.skills.loader import Skill, SkillSet, discover, load_skill

#: Above this many skills the roster drops descriptions (see module docstring).
ROSTER_LIMIT = 25

ROSTER_HEADER = (
    "Skills available to you -- procedural knowledge written for THIS "
    "project. When a task matches one, call load_skill with its name and "
    "follow the instructions it returns BEFORE doing the work; they are "
    "more specific than your defaults and were written by the operator."
)

NAMES_ONLY_FOOTER = (
    "Call list_skills for what each one covers, then load_skill to use it."
)


class SkillRegistry:
    """One session's skills: the set, the roster layer, the load record.

    Lives on the agent as ``agent.skills`` (the same wiring EnvContext
    uses) so the REPL, the web panel, and tests all reach it uniformly.
    An agent with no skills still gets one -- it just renders no layer
    and registers no tools, so ``/skills`` can say where to put them
    instead of pretending the feature does not exist.
    """

    def __init__(self, found: SkillSet | None = None,
                 *, cwd: Path | None = None, home: Path | None = None,
                 roster_limit: int = ROSTER_LIMIT) -> None:
        self.cwd = (cwd or Path.cwd()).resolve()
        #: Injectable for the same reason discover() takes it: a test (or a
        #: container) must be able to scan roots that are not $HOME.
        self.home = home
        self.found = (found if found is not None
                      else discover(self.cwd, home=self.home))
        self.roster_limit = roster_limit
        #: Names the model actually pulled this session, in order. Cheap
        #: observability: it answers "did the skill even fire?", which is
        #: the first question every skill author asks.
        self.loaded: list[str] = []
        self._agent: Any = None

    # ---- the set ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.found)

    def __iter__(self):
        return iter(self.found)

    def get(self, name: str) -> Skill | None:
        return self.found.get(name)

    def names(self) -> list[str]:
        return self.found.names()

    @property
    def names_only(self) -> bool:
        """True when the roster is too long to carry descriptions."""
        return len(self.found) > self.roster_limit

    # ---- tier 1: the roster ------------------------------------------------

    def render_roster(self) -> str | None:
        """The ``skills`` prompt layer; None when there is nothing to say.

        PURE -- no IO, no ranking, no per-turn state. That is what keeps
        the cached prefix stable for a whole session.
        """
        if not self.found:
            return None
        lines = [ROSTER_HEADER]
        if self.names_only:
            lines.append(", ".join(self.names()))
            lines.append(NAMES_ONLY_FOOTER)
        else:
            lines.extend(skill.roster_line() for skill in self.found)
        return "\n".join(lines)

    # ---- tier 2: handing one over -------------------------------------------

    def load(self, name: str) -> Skill:
        """Fetch one skill for delivery, RE-READ from disk when possible.

        Re-reading is what makes authoring iterative: edit SKILL.md, ask
        again, no restart. A file that has since vanished or broken falls
        back to the copy discovered at startup -- a half-saved edit
        should not take a working skill away mid-task. KeyError (not
        SkillError) for an unknown name: the caller turns that into
        model-readable data with the near-miss list attached.
        """
        skill = self.found.get(name)
        if skill is None:
            raise KeyError(name)
        try:
            skill = load_skill(skill.path, source=skill.source)
        except Exception:
            pass  # keep the startup copy; the model still gets instructions
        if name not in self.loaded:
            self.loaded.append(name)
        return skill

    def missing_tools(self, skill: Skill) -> list[str]:
        """Which of a skill's ``allowed-tools`` this session does not have.

        WHY THIS IS A NOTE AND NOT A GATE. ``allowed-tools`` cannot be
        enforced the way it reads: the harness has no way to know whether
        the model is still "inside" a skill three tool calls later, so a
        rule keyed to that would be theatre. What it CAN do is narrow --
        never widen -- expectations, and the useful half of that is
        honesty at delivery time: a skill that plans around browser_open
        on a machine without the [browse] extra should say so in the same
        breath as its instructions, not fail four steps in.

        The real boundary stays where it always was: every dangerous tool
        a skill names is still permission-gated when it runs, and a skill
        that needs true isolation belongs in a sub-agent, whose tool scope
        IS enforced -- at the catalog level, in subagent.py.
        """
        registry = getattr(self._agent, "registry", None)
        if registry is None or not skill.allowed_tools:
            return []
        return [name for name in skill.allowed_tools
                if name not in registry or registry.is_disabled(name)]

    def suggestions(self, name: str, limit: int = 3) -> list[str]:
        """Near-misses for an unknown name -- substring first, then prefix."""
        needle = name.strip().lower()
        hits = [n for n in self.names() if needle and needle in n]
        hits += [n for n in self.names()
                 if n not in hits and needle[:3] and n.startswith(needle[:3])]
        return hits[:limit]

    # ---- wiring --------------------------------------------------------------

    def attach(self, agent: Any) -> "SkillRegistry":
        """Compose the roster onto the agent and register the skill tools.

        Idempotent in the way that matters: re-attaching (after a
        ``/skills reload``) rewrites the same layer and re-registers
        nothing it already registered.
        """
        self._agent = agent
        agent.skills = self
        self._register_tools(agent)
        self.reapply()
        return self

    def reapply(self) -> None:
        """Rewrite the ``skills`` layer and recompose. The other layers --
        the operator's --system, env awareness -- are untouched."""
        if self._agent is None:
            return
        prompt = attach_prompt(self._agent)
        prompt.set("skills", self.render_roster())
        prompt.apply()

    def _register_tools(self, agent: Any) -> None:
        """load_skill when there are skills at all; list_skills only when
        the roster had to drop its descriptions. A project with three
        skills therefore pays exactly ONE extra tool schema."""
        from akshara.skills.tools import ListSkills, LoadSkill

        registry = getattr(agent, "registry", None)
        if registry is None or not self.found:
            return
        if LoadSkill.name not in registry:
            registry.register(LoadSkill(self))
        if self.names_only and ListSkills.name not in registry:
            registry.register(ListSkills(self))

    def reload(self) -> SkillSet:
        """Re-scan every root -- ``/skills reload`` after writing one.

        Tools already registered stay registered (the registry refuses
        duplicates and history may reference them); only the SET and the
        roster change.
        """
        self.found = discover(self.cwd, home=self.home)
        self.loaded = [n for n in self.loaded if self.found.get(n)]
        if self._agent is not None:
            self._register_tools(self._agent)
        self.reapply()
        return self.found

    # ---- display --------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Snapshot for the web state envelope and tests."""
        return {
            "skills": [
                {"name": s.name, "description": s.description,
                 "source": s.source, "path": str(s.path),
                 "allowed_tools": list(s.allowed_tools),
                 "missing_tools": self.missing_tools(s),
                 "loaded": s.name in self.loaded}
                for s in self.found
            ],
            "broken": [{"path": str(b.path), "reason": b.reason,
                        "source": b.source} for b in self.found.broken],
            "loaded": list(self.loaded),
            "names_only": self.names_only,
        }


def enable_skills(agent: Any, cwd: Path | None = None,
                  *, found: SkillSet | None = None,
                  home: Path | None = None) -> SkillRegistry:
    """Discover, compose, register -- the one call a host makes.

    Factored like enable_subagents so embedders and tests can turn
    skills on without a CLI parse. Always returns a registry, even an
    empty one: ``/skills`` on a project with none should explain where
    they go, not report that the feature is missing.
    """
    return SkillRegistry(found, cwd=cwd, home=home).attach(agent)
