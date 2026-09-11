"""Skills on disk: a folder, a SKILL.md, and the rules for reading one.

A SKILL is a chunk of procedural knowledge the operator writes down
once -- how THIS team reviews a PR, how THIS repo cuts a release -- and
the agent loads only when the work calls for it. It is not code and not
a tool: it is instructions, plus whatever files those instructions
reference.

    skills/pr-review/
    |-- SKILL.md          frontmatter + the instructions themselves
    |-- checklist.md      referenced BY the body; read_file fetches it
    `-- scripts/diff.sh   the model runs it with bash

The format, in full:

    ---
    name: pr-review
    description: Review a git diff for correctness bugs and missing
      tests. Use when asked to review a PR, branch, or working tree.
    allowed-tools: bash, read_file, grep
    ---

    # PR review
    ...instructions...

WHY A FOLDER AND NOT A FILE. The description rides in the system prompt
for the whole session (~25 tokens); the body costs nothing until the
model asks for it; bundled files cost nothing until the body sends it
after one. Three tiers of cost, and a folder is what makes the third
tier possible ([notes/30](../notes/30-skills.md)).

WHY A HAND-ROLLED PARSER. The core ships httpx + rich and nothing else
-- the same reason config.py parses .env by hand. Frontmatter here is
deliberately FLAT: ``key: value`` lines, ``#`` comments, indented
continuation lines folded onto the previous key. No nesting, no YAML
types, and the error says so rather than letting a half-parsed block
through.

WHY VALIDATION IS STRICT. A skill with a vague two-word description is
worse than no skill: it burns prompt tokens every turn and still never
gets picked, because selection is retrieval over exactly that text. So
a thin description is an ERROR, reported by name at startup, not a
quiet degradation the operator discovers three sessions later.

TWO MODES. By default a skill is INLINE: its body comes back as a tool
result and the model follows it in the current conversation, with the
current tools. A skill may instead declare ``mode: subagent``, and then
its ``allowed-tools`` stop being a note and become a FENCE -- the body
runs in a fresh child agent that physically has no other tools
registered (see subagent.py). That is the only enforcement the harness
can honestly offer, and it costs a fresh context window to get.

Broken skills never raise into the session. discover() returns them in
a parallel ``broken`` list so ``/skills`` can show the file and the
reason -- one bad folder must not cost you the other nine.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: The one filename a skill folder must contain.
SKILL_FILE = "SKILL.md"

#: Frontmatter fence -- the whole block is optional in Markdown at large,
#: but mandatory here: no fence means no description, means unselectable.
FENCE = "---"

#: Descriptions shorter than this cannot carry a "use when..." clause,
#: which is the half that actually drives selection. The number is a
#: floor against `description: reviews code`, not a style guide.
MIN_DESCRIPTION = 20

#: How a skill runs. ``inline`` hands the body to the current
#: conversation; ``subagent`` runs it in a scoped child (see the module
#: docstring). Shared by the loader, the registry and the tools.
MODES = ("inline", "subagent")

#: Iteration cap a delegated skill may ask for -- the sub-agent spawner's
#: own ceiling, restated here so a bad number fails at AUTHORING time.
MAX_ITERATIONS_RANGE = (1, 50)

#: Folder/name grammar. Lowercase and hyphenated so a name is safe as a
#: slash command (/pr-review), a path segment, and a BM25 token at once.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: Where skills come from, nearest-wins. The order is the override rule:
#: an explicit path beats a private local skill, which beats the set the
#: repo commits, which beats whatever you keep in your home directory.
#:
#: ``.akshara/`` is gitignored (it holds session state -- sqlite, logs, a
#: chrome profile), so the COMMITTED set deliberately lives in a plain
#: top-level ``skills/`` instead: skills are hand-written source meant to
#: travel with the repo, not machine state.
ENV_PATH = "AKSHARA_SKILLS_PATH"
PROJECT_PRIVATE = (".akshara", "skills")
PROJECT_SHARED = ("skills",)
USER_ROOT = (".akshara", "skills")


class SkillError(ValueError):
    """A skill file that cannot be trusted to mean what it says."""


@dataclass(slots=True, frozen=True)
class Skill:
    """One loaded skill. ``body`` is the tier-2 payload, nothing else."""

    name: str
    description: str
    body: str
    path: Path                              # the SKILL.md itself
    source: str = "project"                 # which root it came from
    allowed_tools: tuple[str, ...] = ()
    mode: str = "inline"                    # inline | subagent
    output_format: str = ""                 # subagent mode: shape of the answer
    max_iterations: int = 0                 # subagent mode: 0 = spawner default

    @property
    def directory(self) -> Path:
        """The skill's folder -- the anchor for every bundled file."""
        return self.path.parent

    @property
    def delegated(self) -> bool:
        """True when this skill runs in a scoped sub-agent, not inline."""
        return self.mode == "subagent"

    def roster_line(self) -> str:
        """The tier-1 cost: one line in the system prompt, per session.

        A delegated skill is MARKED, because the model has to reach for a
        different tool to use it -- an unmarked roster would promise
        load_skill for something load_skill deliberately refuses.
        """
        mark = " [delegated]" if self.delegated else ""
        return f"- {self.name}{mark}: {self.description}"


@dataclass(slots=True, frozen=True)
class BrokenSkill:
    """A folder that tried to be a skill. Reported, never raised."""

    path: Path
    reason: str
    source: str = "project"


@dataclass(slots=True)
class SkillSet:
    """Everything one discovery pass found -- including what it rejected."""

    skills: list[Skill] = field(default_factory=list)
    broken: list[BrokenSkill] = field(default_factory=list)
    #: (name, losing path) for each folder a nearer root already claimed.
    shadowed: list[tuple[str, Path]] = field(default_factory=list)

    def names(self) -> list[str]:
        return [s.name for s in self.skills]

    def get(self, name: str) -> Skill | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def __len__(self) -> int:
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills)


# ---- parsing ---------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a fenced ``---`` header from the body it introduces.

    Flat by design (see the module docstring): ``key: value`` per line,
    full-line ``#`` comments, and a line starting with whitespace folded
    onto the previous key -- which is the one affordance descriptions
    actually need, since a good one runs past a comfortable line width.
    Keys are lowercased and ``-`` becomes ``_`` so ``allowed-tools``
    reads as ``allowed_tools`` everywhere downstream.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FENCE:
        raise SkillError(
            f"missing frontmatter: the file must begin with a {FENCE!r} line "
            f"followed by 'name:' and 'description:'")
    # `end` is read after the loop (lines[1:end]) -- not unused, just
    # not used INSIDE the body, which is what B007 measures.
    for end, line in enumerate(lines[1:], start=1):  # noqa: B007
        if line.strip() == FENCE:
            break
    else:
        raise SkillError(
            f"unterminated frontmatter: no closing {FENCE!r} line")

    fields: dict[str, str] = {}
    last_key: str | None = None
    for lineno, raw in enumerate(lines[1:end], start=2):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw[:1].isspace() and last_key is not None:
            fields[last_key] = f"{fields[last_key]} {raw.strip()}".strip()
            continue
        key, sep, value = raw.partition(":")
        if not sep:
            raise SkillError(
                f"line {lineno}: expected 'key: value', got {raw.strip()!r} "
                f"(frontmatter here is flat -- no lists, no nesting)")
        last_key = key.strip().lower().replace("-", "_")
        if not last_key:
            raise SkillError(f"line {lineno}: empty key before ':'")
        fields[last_key] = value.strip()

    return fields, "\n".join(lines[end + 1:]).strip()


def _split_list(value: str) -> tuple[str, ...]:
    """``bash, read_file  grep`` -> three entries. Commas OR whitespace,
    because both look right in a hand-written file."""
    return tuple(item for item in re.split(r"[,\s]+", value.strip()) if item)


def load_skill(path: Path, *, source: str = "project") -> Skill:
    """Read and validate ONE SKILL.md. Raises SkillError, never OSError-blind.

    The folder name is the authority on identity: a ``name:`` field that
    disagrees with it is an error rather than a silent winner, because
    the two would then shadow differently (roster by field, override by
    folder) and the operator would be debugging a ghost.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError(f"unreadable: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise SkillError(f"not UTF-8 text: {exc}") from exc
    return validate_text(text, path, source=source)


def validate_text(text: str, path: Path, *, source: str = "project") -> Skill:
    """Every rule a SKILL.md must satisfy, applied to text that may not be
    on disk yet -- which is what lets an editor check a draft BEFORE
    writing it. ``path`` is still required: identity comes from the folder
    name, so validation has to know where the text would live.
    """
    fields, body = parse_frontmatter(text)
    folder = path.parent.name

    name = fields.get("name", "").strip() or folder
    if not NAME_RE.match(name):
        raise SkillError(
            f"invalid name {name!r}: use lowercase letters, digits and "
            f"hyphens (it doubles as a slash command and a path segment)")
    if name != folder:
        raise SkillError(
            f"name {name!r} does not match its folder {folder!r} -- "
            f"rename one so identity has a single source")

    description = " ".join(fields.get("description", "").split())
    if not description:
        raise SkillError(
            "missing 'description': it is the ONLY text in the prompt every "
            "turn, and the only thing selection can match against")
    if len(description) < MIN_DESCRIPTION:
        raise SkillError(
            f"description is too thin ({len(description)} chars): say what "
            f"the skill does AND when to use it, or it will never be picked")
    if not body:
        raise SkillError("no instructions below the frontmatter")

    mode = (fields.get("mode", "") or "inline").strip().lower()
    if mode not in MODES:
        raise SkillError(
            f"unknown mode {mode!r}: expected {' or '.join(MODES)}")
    allowed_tools = _split_list(fields.get("allowed_tools", ""))
    low, high = MAX_ITERATIONS_RANGE
    max_iterations = 0
    if raw := fields.get("max_iterations", "").strip():
        try:
            max_iterations = int(raw)
        except ValueError:
            raise SkillError(
                f"max-iterations must be a whole number, got {raw!r}") from None
        if not low <= max_iterations <= high:
            raise SkillError(
                f"max-iterations must be between {low} and {high}, "
                f"got {max_iterations}")

    if mode == "subagent":
        # The whole point of this mode is the fence; without a tool list
        # there is nothing to fence, and a child with no tools cannot work.
        if not allowed_tools:
            raise SkillError(
                "mode: subagent requires allowed-tools -- the child agent "
                "gets EXACTLY those and nothing else, so an empty list "
                "fences off everything")
        if "spawn_subagent" in allowed_tools:
            # one level deep; the spawner refuses this at runtime too, but
            # an author should hear it now rather than mid-task
            raise SkillError(
                "a delegated skill cannot list 'spawn_subagent': "
                "sub-agents do not spawn sub-agents")
    elif fields.get("output_format") or max_iterations:
        raise SkillError(
            "output-format and max-iterations only mean something with "
            "mode: subagent -- an inline skill's answer is just the turn")

    return Skill(
        name=name,
        description=description,
        body=body,
        path=path,
        source=source,
        allowed_tools=allowed_tools,
        mode=mode,
        output_format=" ".join(fields.get("output_format", "").split()),
        max_iterations=max_iterations,
    )


# ---- writing one back -------------------------------------------------------


def _fold(key: str, value: str, width: int = 72) -> list[str]:
    """``key: value`` wrapped onto continuation lines the parser folds back.

    The round trip is the point: what this emits, parse_frontmatter reads
    as one value again. Continuation lines are indented two spaces, which
    is what every hand-written skill in this repo looks like.
    """
    words, lines, current = value.split(), [], key + ":"
    for word in words:
        candidate = f"{current} {word}"
        if len(candidate) > width and current not in (key + ":", "  "):
            lines.append(current)
            current = "  " + word
        else:
            current = candidate if current != "  " else "  " + word
    lines.append(current)
    return lines


def render_skill_md(name: str, description: str, body: str, *,
                    mode: str = "inline",
                    allowed_tools: Iterable[str] = (),
                    output_format: str = "",
                    max_iterations: int = 0) -> str:
    """Compose a SKILL.md from fields -- the inverse of the parser.

    Used by hosts that AUTHOR skills rather than only read them (the web
    panel's editor). Emitting through one function, then validating the
    result by parsing it back, is what keeps a generated file identical
    in shape to a hand-written one -- there is no second format that only
    the UI knows how to produce.
    """
    lines = [FENCE, f"name: {name}"]
    lines += _fold("description", " ".join(description.split()))
    tools = [t for t in allowed_tools if t]
    if tools:
        lines.append("allowed-tools: " + ", ".join(tools))
    if mode != "inline":
        lines.append(f"mode: {mode}")
    if output_format.strip():
        lines += _fold("output-format", " ".join(output_format.split()))
    if max_iterations:
        lines.append(f"max-iterations: {max_iterations}")
    lines.append(FENCE)
    return "\n".join(lines) + "\n\n" + body.strip() + "\n"


# ---- discovery --------------------------------------------------------------


def skill_roots(cwd: Path | None = None,
                *, home: Path | None = None) -> list[tuple[Path, str]]:
    """Every directory to scan, NEAREST FIRST (see ENV_PATH's note).

    Paths are returned whether or not they exist -- callers that report
    roots (``/skills``) want to say "no skills yet, put them in X".
    """
    cwd = (cwd or Path.cwd()).resolve()
    roots: list[tuple[Path, str]] = []
    for entry in os.environ.get(ENV_PATH, "").split(os.pathsep):
        entry = entry.strip()
        if entry:
            roots.append((Path(entry).expanduser(), "path"))
    roots.append((cwd.joinpath(*PROJECT_PRIVATE), "local"))
    roots.append((cwd.joinpath(*PROJECT_SHARED), "project"))
    roots.append(((home or Path.home()).joinpath(*USER_ROOT), "user"))
    return roots


def discover(cwd: Path | None = None,
             *, home: Path | None = None) -> SkillSet:
    """Scan every root and return what is loadable, broken, or shadowed.

    Never raises for bad CONTENT -- that is what SkillSet.broken is for.
    A root that cannot be listed at all (permissions, a dangling symlink
    in $AKSHARA_SKILLS_PATH) is simply skipped: an unreadable directory
    is not a skill problem.
    """
    found = SkillSet()
    claimed: dict[str, Path] = {}
    for root, source in skill_roots(cwd, home=home):
        try:
            entries = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for folder in entries:
            manifest = folder / SKILL_FILE
            if not manifest.is_file():
                # A stray directory is not an error -- but the one
                # mistake worth naming is casing, which only bites on
                # case-sensitive filesystems and looks fine in an editor.
                try:
                    near = [p for p in folder.iterdir()
                            if p.is_file()
                            and p.name.lower() == SKILL_FILE.lower()]
                except OSError:
                    continue
                if near:
                    found.broken.append(BrokenSkill(
                        near[0],
                        f"must be named exactly {SKILL_FILE} (case matters)",
                        source))
                continue
            if folder.name in claimed:
                found.shadowed.append((folder.name, manifest))
                continue
            # A BROKEN skill claims nothing, on purpose: a typo in your
            # local copy falls back to the committed one rather than
            # taking the whole skill down with it.
            try:
                skill = load_skill(manifest, source=source)
            except SkillError as exc:
                found.broken.append(BrokenSkill(manifest, str(exc), source))
                continue
            claimed[skill.name] = manifest
            found.skills.append(skill)
    found.skills.sort(key=lambda s: s.name)
    return found
