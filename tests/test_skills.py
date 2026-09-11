"""Skills on disk: the format, the parser, and the discovery rules.

Every assertion here is about what the operator WROTE, not about what
the model does with it. The theme is that a bad skill folder is
reported, by name and reason, and costs you nothing else: the other
skills load, the session starts, and /skills can explain the damage.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from akshara.errors import ToolError
from akshara.prompt import attach_prompt
from akshara.skills import (
    SKILL_FILE,
    SkillError,
    SkillRegistry,
    discover,
    enable_skills,
    load_skill,
    parse_frontmatter,
    skill_roots,
)
from akshara.skills.registry import ROSTER_HEADER
from akshara.tools.base import ToolRegistry


@pytest.fixture(autouse=True)
def _no_ambient_skills(monkeypatch):
    """An operator's own $AKSHARA_SKILLS_PATH must not reach the suite."""
    monkeypatch.delenv("AKSHARA_SKILLS_PATH", raising=False)

GOOD = """\
---
name: pr-review
description: Review a git diff for correctness bugs and missing tests.
  Use when asked to review a PR, a branch, or the working tree.
allowed-tools: bash, read_file, grep
---

# PR review

1. Get the diff with `git diff main...HEAD`.
"""


def write_skill(root: Path, name: str, text: str = GOOD) -> Path:
    """Drop a skill folder under ``root``; returns its SKILL.md."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SKILL_FILE
    path.write_text(text)
    return path


class TestFrontmatter:
    def test_fields_and_body_split_at_the_fence(self):
        fields, body = parse_frontmatter(GOOD)
        assert fields["name"] == "pr-review"
        assert body.startswith("# PR review")
        assert "---" not in body

    def test_indented_lines_fold_onto_the_previous_key(self):
        # the one affordance a good description actually needs
        fields, _ = parse_frontmatter(GOOD)
        assert fields["description"].endswith("working tree.")
        assert "\n" not in fields["description"]

    def test_hyphenated_keys_become_underscored(self):
        fields, _ = parse_frontmatter(GOOD)
        assert fields["allowed_tools"] == "bash, read_file, grep"

    def test_comments_and_blank_lines_are_skipped(self):
        fields, _ = parse_frontmatter(
            "---\n# a note to self\n\nname: x\n---\nbody\n")
        assert fields == {"name": "x"}

    def test_a_value_may_contain_colons(self):
        fields, _ = parse_frontmatter("---\ndescription: do x: then y\n---\nb\n")
        assert fields["description"] == "do x: then y"

    def test_missing_fence_is_an_error(self):
        with pytest.raises(SkillError, match="missing frontmatter"):
            parse_frontmatter("# Just a markdown file\n")

    def test_unterminated_fence_is_an_error(self):
        with pytest.raises(SkillError, match="unterminated"):
            parse_frontmatter("---\nname: x\nbody with no closing fence\n")

    def test_nested_yaml_is_rejected_with_a_readable_reason(self):
        # flat by design -- say so rather than half-parsing it
        with pytest.raises(SkillError, match="flat"):
            parse_frontmatter("---\nname: x\n- not a mapping\n---\nb\n")


class TestValidation:
    def test_a_good_skill_round_trips(self, tmp_path):
        skill = load_skill(write_skill(tmp_path, "pr-review"))
        assert skill.name == "pr-review"
        assert skill.allowed_tools == ("bash", "read_file", "grep")
        assert skill.directory == tmp_path / "pr-review"
        assert skill.roster_line().startswith("- pr-review: Review a git diff")

    def test_name_defaults_to_the_folder(self, tmp_path):
        path = write_skill(tmp_path, "release-cut",
                           GOOD.replace("name: pr-review\n", ""))
        assert load_skill(path).name == "release-cut"

    def test_name_disagreeing_with_the_folder_is_an_error(self, tmp_path):
        # otherwise the roster and the override rule would key differently
        path = write_skill(tmp_path, "other-folder")
        with pytest.raises(SkillError, match="does not match its folder"):
            load_skill(path)

    def test_shouty_names_are_rejected(self, tmp_path):
        path = write_skill(tmp_path, "PR_Review",
                           GOOD.replace("pr-review", "PR_Review"))
        with pytest.raises(SkillError, match="invalid name"):
            load_skill(path)

    def test_missing_description_is_an_error(self, tmp_path):
        path = write_skill(tmp_path, "thin", "---\nname: thin\n---\ndo it\n")
        with pytest.raises(SkillError, match="missing 'description'"):
            load_skill(path)

    def test_thin_description_is_an_error(self, tmp_path):
        # it would burn prompt tokens every turn and still never be picked
        path = write_skill(
            tmp_path, "thin", "---\nname: thin\ndescription: does stuff\n---\ndo it\n")
        with pytest.raises(SkillError, match="too thin"):
            load_skill(path)

    def test_empty_body_is_an_error(self, tmp_path):
        path = write_skill(tmp_path, "pr-review",
                           GOOD.split("# PR review")[0].rstrip() + "\n")
        with pytest.raises(SkillError, match="no instructions"):
            load_skill(path)


class TestDiscovery:
    def test_finds_skills_in_the_committed_project_root(self, tmp_path):
        write_skill(tmp_path / "skills", "pr-review")
        found = discover(tmp_path, home=tmp_path / "home")
        assert found.names() == ["pr-review"]
        assert found.skills[0].source == "project"
        assert found.get("pr-review") is not None

    def test_nearest_root_wins_and_the_loser_is_reported(self, tmp_path):
        # a private local copy overrides the committed one
        write_skill(tmp_path / "skills", "pr-review")
        write_skill(tmp_path / ".akshara" / "skills", "pr-review",
                    GOOD.replace("# PR review", "# PR review (local)"))
        found = discover(tmp_path, home=tmp_path / "home")
        assert len(found) == 1
        assert found.skills[0].source == "local"
        assert "(local)" in found.skills[0].body
        assert [name for name, _ in found.shadowed] == ["pr-review"]

    def test_env_path_beats_every_implicit_root(self, tmp_path, monkeypatch):
        write_skill(tmp_path / "skills", "pr-review")
        explicit = tmp_path / "elsewhere"
        write_skill(explicit, "pr-review",
                    GOOD.replace("# PR review", "# PR review (explicit)"))
        monkeypatch.setenv("AKSHARA_SKILLS_PATH", str(explicit))
        found = discover(tmp_path, home=tmp_path / "home")
        assert found.skills[0].source == "path"
        assert "(explicit)" in found.skills[0].body

    def test_user_root_contributes_what_the_project_lacks(self, tmp_path):
        write_skill(tmp_path / "skills", "pr-review")
        write_skill(tmp_path / "home" / ".akshara" / "skills", "release-cut",
                    GOOD.replace("pr-review", "release-cut"))
        found = discover(tmp_path, home=tmp_path / "home")
        assert found.names() == ["pr-review", "release-cut"]

    def test_one_broken_skill_does_not_cost_you_the_others(self, tmp_path):
        write_skill(tmp_path / "skills", "pr-review")
        write_skill(tmp_path / "skills", "broken", "no frontmatter here\n")
        found = discover(tmp_path, home=tmp_path / "home")
        assert found.names() == ["pr-review"]
        assert len(found.broken) == 1
        assert "missing frontmatter" in found.broken[0].reason

    def test_a_broken_local_copy_falls_back_to_the_committed_one(self, tmp_path):
        write_skill(tmp_path / "skills", "pr-review")
        write_skill(tmp_path / ".akshara" / "skills", "pr-review", "junk\n")
        found = discover(tmp_path, home=tmp_path / "home")
        assert found.names() == ["pr-review"]
        assert found.skills[0].source == "project"
        assert len(found.broken) == 1

    def test_miscased_manifest_is_named_as_the_problem(self, tmp_path):
        folder = tmp_path / "skills" / "pr-review"
        folder.mkdir(parents=True)
        (folder / "skill.md").write_text(GOOD)
        found = discover(tmp_path, home=tmp_path / "home")
        assert not found.skills
        assert "case matters" in found.broken[0].reason

    def test_a_stray_directory_is_not_an_error(self, tmp_path):
        (tmp_path / "skills" / "notes").mkdir(parents=True)
        found = discover(tmp_path, home=tmp_path / "home")
        assert not found.skills and not found.broken

    def test_missing_roots_are_simply_empty(self, tmp_path):
        found = discover(tmp_path / "nothing-here", home=tmp_path / "home")
        assert not found.skills and not found.broken

    def test_roots_are_reported_nearest_first(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AKSHARA_SKILLS_PATH", raising=False)
        sources = [source for _, source in
                   skill_roots(tmp_path, home=tmp_path / "home")]
        assert sources == ["local", "project", "user"]


# ---- the live registry: roster, delivery, wiring ---------------------------



def _agent(**kw):
    return SimpleNamespace(system=None, registry=ToolRegistry(), **kw)


def _wire(tmp_path, *names, system=None):
    """An agent with ``names`` as skills under the committed project root."""
    for name in names:
        write_skill(tmp_path / "skills", name, GOOD.replace("pr-review", name))
    agent = _agent()
    agent.system = system
    # home= keeps ~/.akshara/skills out of the suite, whatever the machine has
    return agent, enable_skills(agent, tmp_path, home=tmp_path / "home")


class TestRoster:
    def test_roster_lands_in_its_own_prompt_layer(self, tmp_path):
        agent, _ = _wire(tmp_path, "pr-review", system="You are a pirate.")
        assert agent.prompt.get("base") == "You are a pirate."
        assert ROSTER_HEADER in agent.prompt.get("skills")
        assert "- pr-review: Review a git diff" in agent.system

    def test_no_skills_means_no_layer_and_no_tool(self, tmp_path):
        agent = _agent()
        enable_skills(agent, tmp_path, home=tmp_path / "home")
        assert agent.system is None
        assert agent.registry.names() == []

    def test_a_small_project_pays_exactly_one_tool(self, tmp_path):
        agent, _ = _wire(tmp_path, "pr-review", "release-cut")
        assert agent.registry.names() == ["load_skill"]

    def test_a_long_roster_drops_descriptions_and_adds_list_skills(self, tmp_path):
        agent, skills = _wire(tmp_path, *(f"skill-{i:02d}" for i in range(4)))
        skills.roster_limit = 3
        skills.reapply()
        assert skills.names_only
        assert "skill-00, skill-01" in agent.system
        assert "Review a git diff" not in agent.system
        skills._register_tools(agent)
        assert "list_skills" in agent.registry

    def test_the_roster_is_frozen_across_turns(self, tmp_path):
        # --cache caches the request prefix INCLUDING the system prompt; a
        # roster that re-ranked itself per turn would bust it every time
        agent, skills = _wire(tmp_path, "pr-review", "release-cut")
        before = agent.system
        skills.load("pr-review")
        skills.reapply()
        assert agent.system == before


class TestDelivery:
    def test_load_skill_returns_the_body_and_anchors_bundled_files(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        out = agent.registry.get("load_skill").run({"name": "pr-review"}, None)
        assert "# PR review" in out
        assert str(tmp_path / "skills" / "pr-review") in out
        assert skills.loaded == ["pr-review"]

    def test_load_skill_reports_tools_this_session_lacks(self, tmp_path):
        # narrowing, not widening: naming a tool never grants it
        agent, _ = _wire(tmp_path, "pr-review")
        out = agent.registry.get("load_skill").run({"name": "pr-review"}, None)
        assert "NOT available in this session: bash, read_file, grep" in out

    def test_an_unknown_name_is_data_with_the_real_names_attached(self, tmp_path):
        agent, _ = _wire(tmp_path, "pr-review")
        with pytest.raises(ToolError) as exc:
            agent.registry.get("load_skill").run({"name": "pr-reviw"}, None)
        assert "pr-review" in str(exc.value)

    def test_load_skill_is_read_only(self, tmp_path):
        # it reads a markdown file the operator wrote; gating it would only
        # train people to mash 'y'
        agent, _ = _wire(tmp_path, "pr-review")
        assert agent.registry.get("load_skill").read_only is True

    def test_a_body_edited_mid_session_is_delivered_fresh(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        path = tmp_path / "skills" / "pr-review" / SKILL_FILE
        path.write_text(GOOD.replace("# PR review", "# PR review v2"))
        out = agent.registry.get("load_skill").run({"name": "pr-review"}, None)
        assert "v2" in out

    def test_a_broken_edit_falls_back_to_the_startup_copy(self, tmp_path):
        # a half-saved file must not take a working skill away mid-task
        agent, skills = _wire(tmp_path, "pr-review")
        (tmp_path / "skills" / "pr-review" / SKILL_FILE).write_text("junk\n")
        out = agent.registry.get("load_skill").run({"name": "pr-review"}, None)
        assert "# PR review" in out


class TestReload:
    def test_reload_picks_up_a_new_skill_and_rewrites_the_roster(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        write_skill(tmp_path / "skills", "release-cut",
                    GOOD.replace("pr-review", "release-cut"))
        skills.reload()
        assert skills.names() == ["pr-review", "release-cut"]
        assert "- release-cut:" in agent.system

    def test_reload_forgets_a_deleted_skill_it_had_loaded(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.load("pr-review")
        (tmp_path / "skills" / "pr-review" / SKILL_FILE).unlink()
        skills.reload()
        assert skills.loaded == []
        assert agent.system is None

    def test_reload_leaves_the_registered_tool_alone(self, tmp_path):
        # history may reference it, and the registry refuses duplicates
        agent, skills = _wire(tmp_path, "pr-review")
        skills.reload()
        assert agent.registry.names() == ["load_skill"]


class TestCoexistence:
    def test_the_roster_never_touches_the_operator_prompt(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review", system="You are a pirate.")
        skills.reload()
        assert agent.prompt.get("base") == "You are a pirate."
        assert agent.system.startswith("You are a pirate.")

    def test_an_env_layer_written_later_survives_a_reload(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        attach_prompt(agent).set("env", "ENV-BLOCK")
        attach_prompt(agent).apply()
        skills.reload()
        assert "ENV-BLOCK" in agent.system
        assert "- pr-review:" in agent.system


class TestDescribe:
    def test_snapshot_carries_what_a_panel_needs(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        write_skill(tmp_path / "skills", "broken", "junk\n")
        skills.reload()
        skills.load("pr-review")
        snap = skills.describe()
        assert snap["skills"][0]["name"] == "pr-review"
        assert snap["skills"][0]["loaded"] is True
        assert snap["skills"][0]["missing_tools"] == ["bash", "read_file", "grep"]
        assert snap["broken"][0]["reason"].startswith("missing frontmatter")
        assert snap["names_only"] is False

    def test_registry_without_an_agent_reports_no_missing_tools(self, tmp_path):
        write_skill(tmp_path / "skills", "pr-review")
        skills = SkillRegistry(cwd=tmp_path, home=tmp_path / "home")
        assert skills.missing_tools(skills.get("pr-review")) == []
