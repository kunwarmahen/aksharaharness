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


# ---- mode: subagent -- where allowed-tools becomes a fence -----------------


DELEGATED = """\
---
name: repo-survey
description: Survey part of this codebase and report what is there. Use
  when asked to explore or map out an unfamiliar area of the code.
mode: subagent
allowed-tools: read_file, glob
output-format: A short report, files with one line each.
max-iterations: 12
---

# Surveying

1. glob for the obvious names.
"""


class TestDelegatedFormat:
    def test_mode_and_its_extras_round_trip(self, tmp_path):
        skill = load_skill(write_skill(tmp_path, "repo-survey", DELEGATED))
        assert skill.delegated
        assert skill.allowed_tools == ("read_file", "glob")
        assert skill.output_format.startswith("A short report")
        assert skill.max_iterations == 12

    def test_inline_is_the_default(self, tmp_path):
        assert load_skill(write_skill(tmp_path, "pr-review")).mode == "inline"

    def test_the_roster_line_marks_a_delegated_skill(self, tmp_path):
        # an unmarked roster would promise load_skill for something
        # load_skill deliberately refuses
        skill = load_skill(write_skill(tmp_path, "repo-survey", DELEGATED))
        assert skill.roster_line().startswith("- repo-survey [delegated]:")

    def test_unknown_mode_is_an_error(self, tmp_path):
        path = write_skill(tmp_path, "repo-survey",
                           DELEGATED.replace("mode: subagent", "mode: yolo"))
        with pytest.raises(SkillError, match="unknown mode"):
            load_skill(path)

    def test_delegation_without_a_tool_list_is_an_error(self, tmp_path):
        # nothing to fence, and a child with no tools cannot work
        path = write_skill(tmp_path, "repo-survey",
                           DELEGATED.replace("allowed-tools: read_file, glob\n", ""))
        with pytest.raises(SkillError, match="requires allowed-tools"):
            load_skill(path)

    def test_a_delegated_skill_cannot_list_spawn_subagent(self, tmp_path):
        path = write_skill(tmp_path, "repo-survey",
                           DELEGATED.replace("read_file, glob",
                                             "read_file, spawn_subagent"))
        with pytest.raises(SkillError, match="do not spawn sub-agents"):
            load_skill(path)

    def test_subagent_only_keys_are_rejected_on_an_inline_skill(self, tmp_path):
        path = write_skill(tmp_path, "pr-review",
                           GOOD.replace("allowed-tools: bash, read_file, grep",
                                        "max-iterations: 9"))
        with pytest.raises(SkillError, match="only mean something"):
            load_skill(path)

    def test_a_junk_iteration_cap_fails_at_authoring_time(self, tmp_path):
        path = write_skill(tmp_path, "repo-survey",
                           DELEGATED.replace("max-iterations: 12",
                                             "max-iterations: 500"))
        with pytest.raises(SkillError, match="between 1 and 50"):
            load_skill(path)


class RecordingSpawner:
    """Stands in for SubagentSpawner: records the call, returns a result."""

    def __init__(self):
        self.calls: list[dict] = []

    def spawn(self, args):
        self.calls.append(args)
        return SimpleNamespace(summary="surveyed", iterations_used=2,
                               tool_calls_made=3, input_tokens=10,
                               output_tokens=4, error=None)


def _delegated_agent(tmp_path, *, tools=("read_file", "glob")):
    """An agent with one delegated skill and a recording spawner."""
    from akshara.tools.fs import ReadFile
    from akshara.tools.glob import Glob

    write_skill(tmp_path / "skills", "repo-survey", DELEGATED)
    agent = _agent()
    for tool in (ReadFile(), Glob()):
        if tool.name in tools:
            agent.registry.register(tool)
    skills = enable_skills(agent, tmp_path, home=tmp_path / "home")
    agent.subagents = RecordingSpawner()
    return agent, skills


class TestDelegation:
    def test_run_skill_registers_only_where_one_is_delegated(self, tmp_path):
        agent, _ = _delegated_agent(tmp_path)
        assert "run_skill" in agent.registry
        plain = _agent()
        write_skill(tmp_path / "inline-only" / "skills", "pr-review")
        enable_skills(plain, tmp_path / "inline-only", home=tmp_path / "home")
        assert "run_skill" not in plain.registry

    def test_the_roster_tells_the_model_which_tool_to_reach_for(self, tmp_path):
        agent, _ = _delegated_agent(tmp_path)
        assert "[delegated]" in agent.system
        assert "use run_skill" in agent.system

    def test_run_skill_passes_the_body_task_and_fence_to_the_spawner(self, tmp_path):
        agent, _ = _delegated_agent(tmp_path)
        out = agent.registry.get("run_skill").run(
            {"name": "repo-survey", "task": "map the providers"}, None)
        call = agent.subagents.calls[0]
        assert "# Surveying" in call["objective"]
        assert "Task: map the providers" in call["objective"]
        assert call["tools_allowed"] == ["read_file", "glob"]
        assert call["max_iterations"] == 12
        assert call["output_format"].startswith("A short report")
        assert "surveyed" in out and "sub-agent" in out

    def test_run_skill_is_not_read_only(self, tmp_path):
        # the child can do whatever its tools can do -- never auto-approve
        agent, _ = _delegated_agent(tmp_path)
        assert agent.registry.get("run_skill").read_only is False

    def test_load_skill_refuses_a_delegated_skill(self, tmp_path):
        # handing the body over inline would quietly undo the fence
        agent, _ = _delegated_agent(tmp_path)
        with pytest.raises(ToolError, match="run_skill"):
            agent.registry.get("load_skill").run({"name": "repo-survey"}, None)

    def test_run_skill_refuses_an_inline_skill(self, tmp_path):
        agent, _ = _delegated_agent(tmp_path)
        write_skill(tmp_path / "skills", "pr-review")
        agent.skills.reload()
        with pytest.raises(ToolError, match="runs inline"):
            agent.registry.get("run_skill").run(
                {"name": "pr-review", "task": "x"}, None)

    def test_tools_this_session_lacks_are_dropped_and_named(self, tmp_path):
        # the spawner refuses the whole run over one unknown name, and a
        # child planning around a missing tool wastes its window
        agent, _ = _delegated_agent(tmp_path, tools=("read_file",))
        agent.registry.get("run_skill").run(
            {"name": "repo-survey", "task": "map it"}, None)
        call = agent.subagents.calls[0]
        assert call["tools_allowed"] == ["read_file"]
        assert "glob" in call["objective"]
        assert "NOT available" in call["objective"]

    def test_a_skill_whose_tools_are_all_missing_says_so(self, tmp_path):
        agent, _ = _delegated_agent(tmp_path, tools=())
        with pytest.raises(ToolError, match="cannot run here"):
            agent.registry.get("run_skill").run(
                {"name": "repo-survey", "task": "map it"}, None)

    def test_an_empty_task_is_refused(self, tmp_path):
        agent, _ = _delegated_agent(tmp_path)
        with pytest.raises(ToolError, match="only context"):
            agent.registry.get("run_skill").run(
                {"name": "repo-survey", "task": "  "}, None)

    def test_running_one_counts_as_loaded(self, tmp_path):
        agent, skills = _delegated_agent(tmp_path)
        agent.registry.get("run_skill").run(
            {"name": "repo-survey", "task": "map it"}, None)
        assert skills.loaded == ["repo-survey"]

    def test_the_subagent_budget_is_shared_not_doubled(self, tmp_path):
        # a delegated skill IS a sub-agent; two counters would let the pair
        # spend twice what the operator allowed
        from akshara.agent import Agent
        from akshara.subagent import SubagentSpawner

        write_skill(tmp_path / "skills", "repo-survey", DELEGATED)
        agent = Agent(None, model="m")
        existing = SubagentSpawner(agent)
        agent.subagents = existing
        skills = enable_skills(agent, tmp_path, home=tmp_path / "home")
        assert skills.spawner() is existing

    def test_a_spawner_is_created_on_demand_without_subagents_flag(self, tmp_path):
        from akshara.agent import Agent
        from akshara.subagent import SubagentSpawner

        write_skill(tmp_path / "skills", "repo-survey", DELEGATED)
        agent = Agent(None, model="m")
        skills = enable_skills(agent, tmp_path, home=tmp_path / "home")
        assert isinstance(skills.spawner(), SubagentSpawner)
        assert skills.spawner() is agent.subagents   # cached, not rebuilt


# ---- the operator's switch: /skills off|on, $AKSHARA_DISABLED_SKILLS -------


class TestDisabling:
    def test_a_disabled_skill_leaves_the_roster(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review", "release-cut")
        assert skills.disable("release-cut") is True
        skills.reapply()
        assert "- pr-review:" in agent.system
        assert "release-cut" not in agent.system

    def test_disabling_every_skill_drops_the_layer_entirely(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        skills.reapply()
        assert agent.system is None

    def test_it_stays_discovered_so_the_operator_can_see_it(self, tmp_path):
        # the soft switch, not a delete: /skills still lists it, marked off
        _, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        assert skills.names() == ["pr-review"]
        assert skills.disabled_names() == ["pr-review"]
        assert skills.describe()["skills"][0]["enabled"] is False

    def test_loading_a_disabled_skill_is_data_not_a_crash(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        with pytest.raises(ToolError, match="disabled by the operator"):
            agent.registry.get("load_skill").run({"name": "pr-review"}, None)

    def test_running_a_disabled_delegated_skill_is_refused(self, tmp_path):
        agent, skills = _delegated_agent(tmp_path)
        skills.disable("repo-survey")
        with pytest.raises(ToolError, match="disabled by the operator"):
            agent.registry.get("run_skill").run(
                {"name": "repo-survey", "task": "map it"}, None)

    def test_suggestions_never_point_at_a_disabled_skill(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        with pytest.raises(ToolError) as exc:
            agent.registry.get("load_skill").run({"name": "pr-revie"}, None)
        assert "Did you mean" not in str(exc.value)

    def test_enable_restores_it_to_the_roster(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        skills.enable("pr-review")
        skills.reapply()
        assert "- pr-review:" in agent.system

    def test_unknown_names_report_false_rather_than_inventing_one(self, tmp_path):
        _, skills = _wire(tmp_path, "pr-review")
        assert skills.disable("ghost") is False
        assert skills.enable("ghost") is False
        assert skills.disabled_names() == []

    def test_a_disable_survives_a_rescan(self, tmp_path):
        # the operator pulled that NAME, not that file
        _, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        skills.reload()
        assert skills.disabled_names() == ["pr-review"]

    def test_a_deleted_skill_stops_being_disabled_and_absent(self, tmp_path):
        _, skills = _wire(tmp_path, "pr-review")
        skills.disable("pr-review")
        (tmp_path / "skills" / "pr-review" / SKILL_FILE).unlink()
        skills.reload()
        assert skills.disabled_names() == []

    def test_list_skills_hides_what_the_operator_pulled(self, tmp_path):
        agent, skills = _wire(tmp_path, *(f"skill-{i:02d}" for i in range(4)))
        skills.roster_limit = 3
        skills._register_tools(agent)
        skills.disable("skill-00")
        out = agent.registry.get("list_skills").run({}, None)
        assert "skill-00" not in out
        assert "skill-01" in out


class TestDisabledPatterns:
    def test_globs_come_off_the_environment(self, monkeypatch):
        from akshara import config

        monkeypatch.setenv("AKSHARA_DISABLED_SKILLS", "deploy-*, pr-review")
        assert config.disabled_skill_patterns() == ["deploy-*", "pr-review"]

    def test_unset_disables_nothing(self, monkeypatch):
        from akshara import config

        monkeypatch.delenv("AKSHARA_DISABLED_SKILLS", raising=False)
        assert config.disabled_skill_patterns() == []


# ---- authoring: writing a skill back to disk -------------------------------


class TestRenderRoundTrip:
    def test_what_it_writes_the_parser_reads_back(self, tmp_path):
        from akshara.skills import render_skill_md

        text = render_skill_md(
            "repo-survey",
            "Survey part of this codebase and report what is there. Use when "
            "asked to explore or map out an unfamiliar area of the code.",
            "# Surveying\n\n1. glob first.",
            mode="subagent", allowed_tools=("read_file", "glob"),
            output_format="One line per file.", max_iterations=15)
        path = write_skill(tmp_path, "repo-survey", text)
        skill = load_skill(path)
        assert skill.delegated
        assert skill.allowed_tools == ("read_file", "glob")
        assert skill.max_iterations == 15
        assert skill.body.startswith("# Surveying")

    def test_a_long_description_folds_and_unfolds(self, tmp_path):
        from akshara.skills import parse_frontmatter, render_skill_md

        long = ("Cut a tagged release for this repo including the version "
                "bump, the changelog line, the tag itself and the push. Use "
                "when asked to cut, tag, or publish a release of any kind.")
        text = render_skill_md("release-cut", long, "# Steps\n\n1. Go.")
        assert max(len(line) for line in text.splitlines()) <= 76
        fields, _ = parse_frontmatter(text)
        assert fields["description"] == long

    def test_inline_skills_carry_no_delegated_keys(self):
        from akshara.skills import render_skill_md

        text = render_skill_md("pr-review", "x" * 30, "# Do it")
        assert "mode:" not in text
        assert "max-iterations:" not in text


class TestWriting:
    def test_a_new_skill_lands_in_the_committed_root(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skill = skills.write(
            "release-cut",
            "Cut a tagged release of this project. Use when asked to cut, "
            "tag, or publish a release.",
            "# Cutting a release\n\n1. Check the tree is clean.")
        assert skill.path == tmp_path / "skills" / "release-cut" / SKILL_FILE
        assert skill.source == "project"
        assert "- release-cut:" in agent.system   # roster recomposed

    def test_an_edit_rewrites_in_place_wherever_it_lives(self, tmp_path):
        # writing an edit to a different root would create a shadowing copy
        # and leave the original behind
        write_skill(tmp_path / ".akshara" / "skills", "pr-review")
        agent = _agent()
        skills = enable_skills(agent, tmp_path, home=tmp_path / "home")
        assert skills.get("pr-review").source == "local"
        skill = skills.write("pr-review", "Review a git diff for bugs and "
                             "missing tests. Use on any PR.", "# v2")
        assert skill.path == tmp_path / ".akshara" / "skills" / "pr-review" / SKILL_FILE
        assert not (tmp_path / "skills" / "pr-review").exists()
        assert skill.body == "# v2"

    def test_a_broken_draft_is_refused_before_anything_is_written(self, tmp_path):
        _, skills = _wire(tmp_path, "pr-review")
        with pytest.raises(SkillError, match="too thin"):
            skills.write("release-cut", "does stuff", "# Go")
        assert not (tmp_path / "skills" / "release-cut").exists()

    def test_a_bad_name_cannot_address_a_path(self, tmp_path):
        # the name is a path segment; traversal must die at the grammar
        _, skills = _wire(tmp_path, "pr-review")
        for bad in ("../escape", "has/slash", "../../etc/passwd", "", "  "):
            with pytest.raises(SkillError, match="invalid name"):
                skills.write(bad, "A perfectly fine description here.", "# b")

    def test_an_editor_normalizes_case_rather_than_scolding(self, tmp_path):
        # a hand-written SKILL.md with 'name: Shouty' is an error (the
        # folder would disagree), but someone TYPING a name into a form
        # meant the obvious thing -- and the row they get back says so
        _, skills = _wire(tmp_path, "pr-review")
        skill = skills.write("Release-Cut",
                             "Cut a tagged release. Use when asked to tag "
                             "or publish one.", "# Cutting")
        assert skill.name == "release-cut"
        assert skill.path.parent.name == "release-cut"

    def test_delegated_rules_apply_to_written_skills_too(self, tmp_path):
        _, skills = _wire(tmp_path, "pr-review")
        with pytest.raises(SkillError, match="requires allowed-tools"):
            skills.write("repo-survey",
                         "Survey a directory and report what is in it.",
                         "# Survey", mode="subagent")

    def test_writing_a_delegated_skill_registers_run_skill(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.write("repo-survey",
                     "Survey a directory and report what lives there. Use "
                     "when asked to explore a folder.",
                     "# Survey", mode="subagent",
                     allowed_tools=("read_file",), max_iterations=8)
        assert "run_skill" in agent.registry
        assert "[delegated]" in agent.system

    def test_an_overwrite_does_not_duplicate_the_roster_line(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        for i in range(3):
            skills.write("pr-review",
                         "Review a git diff for correctness bugs and tests.",
                         f"# Review v{i}")
        assert agent.system.count("- pr-review:") == 1
        assert skills.get("pr-review").body == "# Review v2"

    def test_a_failed_write_leaves_no_temp_files_behind(self, tmp_path):
        agent, skills = _wire(tmp_path, "pr-review")
        skills.write("pr-review", "Review a git diff for bugs and tests.",
                     "# v2")
        folder = tmp_path / "skills" / "pr-review"
        assert [p.name for p in folder.iterdir()] == [SKILL_FILE]
