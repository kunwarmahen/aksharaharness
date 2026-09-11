"""Skills: procedural knowledge the agent loads only when it applies.

Discovery and the on-disk format live in ``loader``; see
[notes/30](../../notes/30-skills.md) for the why.
"""

from __future__ import annotations

from akshara.skills.loader import (
    MIN_DESCRIPTION,
    SKILL_FILE,
    BrokenSkill,
    Skill,
    SkillError,
    SkillSet,
    discover,
    load_skill,
    parse_frontmatter,
    skill_roots,
)

__all__ = [
    "BrokenSkill",
    "discover",
    "load_skill",
    "MIN_DESCRIPTION",
    "parse_frontmatter",
    "Skill",
    "SKILL_FILE",
    "SkillError",
    "SkillSet",
    "skill_roots",
]
