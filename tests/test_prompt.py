"""The layered system prompt: who owns which slice of ``agent.system``.

Pins the contracts every future layer depends on -- declared render
order, blank layers contributing no gap, the all-empty case staying
None (byte-identical to a session with no prompt at all), capture-once
attachment, and the reason the seam exists at all: two owners writing
two layers must not be able to erase each other.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from akshara.env_context import POLICY, EnvContext
from akshara.prompt import SystemPrompt, attach_prompt, recompose


def _agent(system=None):
    return SimpleNamespace(system=system)


class TestRendering:
    def test_layers_render_in_declared_order(self):
        prompt = SystemPrompt("BASE")
        prompt.set("skills", "SKILLS")
        prompt.set("env", "ENV")
        assert prompt.render() == "BASE\n\nENV\n\nSKILLS"

    def test_empty_layers_contribute_no_gap(self):
        prompt = SystemPrompt("BASE")
        prompt.set("skills", "SKILLS")
        assert prompt.render() == "BASE\n\nSKILLS"

    def test_all_empty_stays_none(self):
        # the pre-layers wire shape: an unconfigured session sends system=None
        assert SystemPrompt().render() is None

    def test_blank_text_clears_a_layer(self):
        prompt = SystemPrompt("BASE")
        prompt.set("env", "ENV")
        prompt.set("env", "")
        assert prompt.render() == "BASE"

    def test_unknown_layer_fails_loudly(self):
        # a typo'd name would otherwise vanish into a dict and render nothing
        prompt = SystemPrompt()
        with pytest.raises(ValueError):
            prompt.set("skils", "oops")
        with pytest.raises(ValueError):
            prompt.get("skils")

    def test_layers_snapshot_is_ordered_and_complete(self):
        prompt = SystemPrompt("BASE")
        assert list(prompt.layers()) == ["base", "env", "skills"]
        assert prompt.layers()["env"] is None


class TestAttachment:
    def test_attach_pushes_the_render_onto_the_agent(self):
        agent = _agent("BASE")
        attach_prompt(agent).set("env", "ENV")
        assert agent.system == "BASE"  # set() alone does not publish
        agent.prompt.apply()
        assert agent.system == "BASE\n\nENV"

    def test_first_caller_freezes_the_operator_prompt_as_base(self):
        agent = _agent("You are a pirate.")
        assert attach_prompt(agent).get("base") == "You are a pirate."

    def test_attach_is_idempotent_and_never_recaptures(self):
        # the exactly-once discipline that used to live in EnvContext.attach:
        # a second call must hand back the SAME composer, not swallow the
        # already-composed string into a new base
        agent = _agent(None)
        first = attach_prompt(agent)
        first.set("env", "ENV")
        first.apply()
        second = attach_prompt(agent)
        assert second is first
        assert second.get("base") is None
        assert agent.system == "ENV"


class TestRecompose:
    def test_recompose_discards_a_restored_composed_string(self):
        agent = _agent(None)
        prompt = attach_prompt(agent)
        prompt.set("env", "ENV")
        prompt.apply()
        agent.system = "STALE-RESTORED-CHECKPOINT"  # what /load does
        recompose(agent)
        assert agent.system == "ENV"

    def test_recompose_falls_back_to_a_bare_env_context(self):
        # an agent wired the pre-layers way (no SystemPrompt) still recomposes
        recomposed = []
        agent = SimpleNamespace(
            system="STALE",
            env_context=SimpleNamespace(reapply=lambda: recomposed.append(True)),
        )
        recompose(agent)
        assert recomposed == [True]

    def test_recompose_leaves_a_plain_agent_alone(self):
        # no layers, no awareness: the restored string IS the whole truth
        agent = _agent("RESTORED")
        recompose(agent)
        assert agent.system == "RESTORED"


class TestCoexistence:
    """The bug the seam exists to prevent: two owners, one string."""

    def test_an_env_flip_cannot_erase_the_skills_layer(self):
        agent = _agent("BASE")
        ctx = EnvContext("local", Path("/tmp/wksp"))
        ctx.attach(agent)
        agent.prompt.set("skills", "SKILLS-ROSTER")
        agent.prompt.apply()

        ctx.flip("off")
        assert agent.system == "BASE\n\nSKILLS-ROSTER"
        ctx.flip("local")
        assert agent.system.startswith("BASE")
        assert POLICY in agent.system
        assert agent.system.endswith("SKILLS-ROSTER")

    def test_a_skills_write_cannot_swallow_the_env_block(self):
        agent = _agent(None)
        ctx = EnvContext("local", Path("/tmp/wksp"))
        ctx.attach(agent)
        attach_prompt(agent).set("skills", "SKILLS-ROSTER")
        agent.prompt.apply()
        assert POLICY in agent.system
        assert agent.prompt.get("base") is None  # env's block stayed a LAYER
