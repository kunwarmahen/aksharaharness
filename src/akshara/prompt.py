"""The system prompt as LAYERS -- one string, several owners.

For most of this harness's life ``agent.system`` had exactly one author:
whatever the operator passed to ``--system`` (usually nothing). Then
env_context arrived and needed to APPEND a fact sheet, which it did the
only way a single string allows -- capture the operator's words as a
private base, recompose the whole string on every change, and hope
nobody else ever wants a turn. Re-read that sentence as a design and the
crack is obvious: a SECOND appender does the same capture, swallows the
first one's block into its own "base", and the next flip of either one
silently erases the other.

So the string gets a seam before the second appender exists. A
``SystemPrompt`` is an ordered map of NAMED layers:

    base    the operator's --system, captured verbatim, never edited
    env     env_context's fact sheet + policy (notes/29)
    skills  the skill roster (notes/30)

Each owner writes ONLY its own layer and re-applies; render() joins the
non-empty ones, in declared order, with blank lines. Two rules make it
safe to hand out:

* CAPTURE ONCE, CENTRALLY. ``attach_prompt`` is idempotent -- the first
  caller freezes ``agent.system`` as the base layer and every later
  caller gets that same object. The exactly-once discipline that used to
  live in env_context (and would have had to be re-implemented, bug for
  bug, in every future layer) now lives in one function.
* ``agent.system`` STAYS A PLAIN STRING. The loop reads it per iteration
  (agent.py passes it to every request) and checkpoints save it verbatim;
  making it a property would change both contracts. The layers are the
  source of truth, ``apply()`` pushes the render down onto the attribute,
  and an agent with no layers behaves exactly as it always did.

The checkpoint rule falls out of the same design: a /load restores the
stored COMPOSED string over the attribute, so every load path calls
``recompose(agent)`` afterwards to rebuild it from the live layers --
fresh facts, current roster, no stale copy stacked under itself.
"""

from __future__ import annotations

from typing import Any

#: Declared render order. Layers not in this tuple are rejected loudly:
#: a typo'd name would otherwise vanish into a dict and render nothing.
LAYER_ORDER = ("base", "env", "skills")

#: Attribute the composer lives under on an Agent (or any object with a
#: ``system``). Set by attach_prompt; read by recompose.
PROMPT_ATTR = "prompt"


class SystemPrompt:
    """The layered composer behind one agent's ``system`` string."""

    def __init__(self, base: str | None = None) -> None:
        self._layers: dict[str, str | None] = {name: None for name in LAYER_ORDER}
        self._layers["base"] = base
        self._agent: Any = None

    # ---- layers ------------------------------------------------------------

    def set(self, name: str, text: str | None) -> None:
        """Replace one layer. Blank text clears it (an empty layer and a
        missing one render identically -- neither contributes a gap)."""
        if name not in self._layers:
            raise ValueError(f"unknown prompt layer {name!r}; "
                             f"expected one of {'|'.join(LAYER_ORDER)}")
        self._layers[name] = text or None

    def get(self, name: str) -> str | None:
        if name not in self._layers:
            raise ValueError(f"unknown prompt layer {name!r}; "
                             f"expected one of {'|'.join(LAYER_ORDER)}")
        return self._layers[name]

    def layers(self) -> dict[str, str | None]:
        """Snapshot in render order -- for /prompt-style display and tests."""
        return {name: self._layers[name] for name in LAYER_ORDER}

    # ---- rendering ---------------------------------------------------------

    def render(self) -> str | None:
        """The composed prompt, or None when every layer is empty.

        None rather than "" on purpose: a session with no base, no
        awareness and no skills must send ``system=None``, byte-identical
        to the pre-layers wire shape.
        """
        parts = [self._layers[name] for name in LAYER_ORDER if self._layers[name]]
        return "\n\n".join(parts) or None

    def apply(self) -> str | None:
        """Push the render down onto ``agent.system``; returns it too.

        Every mutation path ends here, which is also what makes a load
        path's ``recompose`` a one-liner: re-applying is just rendering
        the layers nobody threw away.
        """
        composed = self.render()
        if self._agent is not None:
            self._agent.system = composed
        return composed

    # ---- wiring ------------------------------------------------------------

    def attach(self, agent: Any) -> SystemPrompt:
        """Bind to an agent and compose immediately. Prefer attach_prompt."""
        self._agent = agent
        setattr(agent, PROMPT_ATTR, self)
        self.apply()
        return self


def attach_prompt(agent: Any) -> SystemPrompt:
    """The agent's SystemPrompt -- creating it on FIRST call, reusing it after.

    Idempotent by construction, which is the whole point: whichever owner
    wires up first (env_context in the CLI, the skill registry in a
    library embedding, either order) freezes the operator's ``--system``
    as the base layer, and everyone else joins the same composition
    instead of starting a rival one.
    """
    existing = getattr(agent, PROMPT_ATTR, None)
    if isinstance(existing, SystemPrompt):
        return existing
    return SystemPrompt(getattr(agent, "system", None)).attach(agent)


def recompose(agent: Any) -> None:
    """Rebuild ``agent.system`` from its live layers -- the /load fixup.

    Checkpoints store the COMPOSED string (session.py saves ``system``
    verbatim), so ``apply_payload`` hands the agent a snapshot of last
    Tuesday's facts. Re-applying discards it in favour of the layers this
    session actually owns. Agents with no layered prompt fall back to the
    pre-layers path (an EnvContext recomposing itself), and agents with
    neither are left alone -- their restored string IS the whole truth.
    """
    prompt = getattr(agent, PROMPT_ATTR, None)
    if isinstance(prompt, SystemPrompt):
        prompt.apply()
        return
    ctx = getattr(agent, "env_context", None)
    if ctx is not None:
        ctx.reapply()
