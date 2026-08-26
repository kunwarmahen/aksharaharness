"""env_context: the agent's own senses -- what it knows without asking.

Out of the box an akshara session starts BLIND: no system prompt, so the
model doesn't know today's date, its timezone, or what city it is sitting
in. Ask it the temperature and it burns a turn asking YOU which city --
not laziness, just honest ignorance with ask_user as the only escape
hatch ([notes/29](../notes/29-environment-awareness.md)). This module
gives sessions graded ambient awareness instead:

* LOCAL facts (mode ``local``): wall-clock time + timezone, user@host,
  OS, working directory. All stdlib reads of the machine itself;
  they leave it only inside requests already being sent.
* LOCATION (mode ``full``, the default): ONE keyless HTTPS lookup to
  ipinfo.io maps the public IP to city/region/country. Cached for the
  whole session -- the network call happens once, at startup (or at the
  first upgrade to ``full``).
* POLICY: whenever awareness is on at any level, the composed prompt
  ends with two sentences telling the model to try its tools before
  asking the human. The facts alone don't change behavior; this line is
  what turns "which city?" into "checking the weather in <your city>".

Design constraints worth remembering:

* THE BLOCK IS FROZEN AT SESSION START. ``--cache`` (anthropic dialect)
  caches the request PREFIX including the system prompt; a per-request
  clock would bust it every turn. Facts are therefore collected once,
  rendered once, and describe themselves as possibly-stale.
* COMPOSITION IS ADDITIVE. The operator's own ``--system`` prompt is
  captured verbatim as the base; awareness appends below it. Nothing
  here edits the operator's words.
* FLIPS ARE LIVE. The loop rebuilds each request from ``agent.system``
  (agent.py passes it per iteration), so /env (REPL) and the env chip
  (web) recompose mid-session and the very next model call sees the new
  mode. No require_idle anywhere -- same argument as permission flips.

Checkpoints store the composed string (session.py saves ``system``
verbatim), which is why attach() captures the base EXACTLY ONCE and
every load path calls reapply() afterwards: a restored session gets a
freshly composed block, not last Tuesday's stale copy compounded under
itself.
"""

from __future__ import annotations

import getpass
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

#: Awareness levels, low to high. Shared by config parsing, the REPL
#: command, and the web endpoint -- one tuple, three surfaces agree.
MODES_ENV = ("off", "local", "full")

#: Keyless geo endpoint; ipinfo's free tier answers
#: {city, region, country, ...} with no token. One call per SESSION.
_GEO_URL = "https://ipinfo.io/json"

_GEO_TIMEOUT = 3.0  # seconds; offline machines shouldn't stall startup

#: The gentle nudge. Strong enough to kill "which city?", careful not to
#: talk the model out of questions only a human can answer (preferences,
#: permission for the irreversible).
POLICY = (
    "Before asking the human for any fact, try to discover it yourself "
    "with your tools (files, bash, web_fetch). Ask only when discovery "
    "fails, results are ambiguous, or the answer is genuinely theirs -- "
    "a preference, or permission for something irreversible."
)


def _http_get(url: str) -> dict[str, Any]:
    """The one network seam. Tests monkeypatch THIS -- nothing else in the
    module touches the wire, so the suite stays offline-safe."""
    resp = httpx.get(url, timeout=_GEO_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _utc_offset(offset) -> str:
    """A utcoffset timedelta -> '+05:30' style (strftime's %z gives +0530)."""
    total = abs(int(offset.total_seconds()))
    sign = "-" if offset.total_seconds() < 0 else "+"
    return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d}"


def _collect_local(cwd: Path) -> dict[str, str]:
    """Machine-local facts. Stdlib only; no IO beyond asking the OS
    about itself."""
    now = datetime.now().astimezone()
    tz = now.tzname() or "local time"
    return {
        "time": (f"{now.strftime('%Y-%m-%d %H:%M')} "
                 f"({tz}, UTC{_utc_offset(now.utcoffset())})"),
        "host": f"{getpass.getuser()}@{socket.gethostname()}",
        "os": f"{platform.system()} {platform.release()} "
              f"({platform.machine()})",
        "cwd": str(cwd),
    }


def _format_location(data: dict[str, Any]) -> str | None:
    """ipinfo payload -> 'Pune, Maharashtra, India'; None when the response
    carries nothing usable (an unclassifiable IP, a hardened exit node).
    Adjacent duplicates collapse -- some responses set region == city."""
    parts = [str(data[k]).strip() for k in ("city", "region", "country")
             if data.get(k)]
    deduped = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]
    return ", ".join(deduped) or None


class EnvContext:
    """One session's awareness: mode, cached facts, composed system prompt.

    Created and attached by the CLI right after the Agent exists (the
    constructor takes NO agent -- attach() does the wiring), then lives on
    the agent as ``agent.env_context`` so the REPL, the web session, and
    checkpoint-load paths reach it uniformly. Agents built WITHOUT one
    (builds, embedders, old checkpoints) behave exactly as before.
    """

    def __init__(self, mode: str = "full", cwd: Path | None = None) -> None:
        if mode not in MODES_ENV:
            raise ValueError(f"env-context mode must be one of "
                             f"{'|'.join(MODES_ENV)}, got {mode!r}")
        self.mode = mode
        self.cwd = (cwd or Path.cwd()).resolve()
        self._local: dict[str, str] | None = None  # machine facts, once
        self._location: str | None = None          # survives downgrades
        self._geo_done = False                     # lookup attempted (maybe failed)
        self.geo_error: str | None = None          # why the lookup failed
        self._base_system: str | None = None       # the operator's --system
        self._base_captured = False                # capture exactly once
        self._agent: Any = None                    # set by attach()

    # ---- collection ---------------------------------------------------------

    def ensure_facts(self) -> None:
        """Collect whatever the CURRENT mode needs, each piece once. Local
        facts are free; the location costs one HTTP call per SESSION --
        never per turn (the frozen-block rule in the module docstring)."""
        if self.mode == "off":
            return
        if self._local is None:
            self._local = _collect_local(self.cwd)
        if self.mode == "full" and not self._geo_done:
            self._geo_done = True
            try:
                self._location = _format_location(_http_get(_GEO_URL))
                if self._location is None:
                    self.geo_error = "lookup returned no usable fields"
            except Exception as exc:
                self._location = None
                self.geo_error = f"{type(exc).__name__}: {exc}"

    # ---- composition ----------------------------------------------------------

    def render_block(self) -> str | None:
        """The fact sheet exactly as the model sees it; None when off (or
        before collection). PURE -- no IO -- so callers may render freely."""
        if self.mode == "off" or self._local is None:
            return None
        lines = [
            "Session context (auto-detected at session start; may be stale):",
            f"- Time: {self._local['time']}",
            f"- Host: {self._local['host']} -- {self._local['os']}",
            f"- Working directory: {self._local['cwd']}",
        ]
        if self.mode == "full" and self._location:
            lines.append(f"- Location: {self._location} (from public IP)")
        lines.append(POLICY)
        return "\n".join(lines)

    def compose(self) -> str | None:
        """Base system prompt + awareness block; None when both are absent
        (so an unaware session keeps sending system=None, byte-identical
        to the pre-feature behavior)."""
        parts = [p for p in (self._base_system, self.render_block()) if p]
        return "\n\n".join(parts) or None

    # ---- wiring -----------------------------------------------------------------

    def attach(self, agent: Any) -> None:
        """First wiring: capture the operator's own system prompt (usually
        None), collect facts for the starting mode, compose onto the agent.

        The base is captured EXACTLY ONCE -- checkpoints save the composed
        string, so re-attaching to a loaded agent would swallow the old
        block into the base and stack stale copies (loads call reapply)."""
        if not self._base_captured:
            self._base_system = agent.system
            self._base_captured = True
        self._agent = agent
        agent.env_context = self
        self.reapply()

    def reapply(self) -> None:
        """Recompose from the stored base -- after /load restored a stale
        composed string over ours, or internally after flip()."""
        self.ensure_facts()
        if self._agent is not None:
            self._agent.system = self.compose()

    def flip(self, mode: str) -> str:
        """Switch level, collecting anything the new level still needs
        (upgrading to ``full`` does the one-time geo lookup here), then
        recompose live. Returns the new mode; ValueError on a bad one."""
        if mode not in MODES_ENV:
            raise ValueError(f"env-context mode must be one of "
                             f"{'|'.join(MODES_ENV)}, got {mode!r}")
        self.mode = mode
        self.ensure_facts()
        self.reapply()
        return mode

    # ---- display ------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Snapshot for the web state envelope and tests."""
        return {
            "mode": self.mode,
            "location": self._location,
            "error": self.geo_error,
            "block": self.render_block(),
        }

    def panel(self) -> str:
        """Bare-/env text for the terminal. PLAIN -- printed with markup
        disabled, because hostnames and cwds may contain rich syntax."""
        lines = [f"env context: {self.mode}"]
        block = self.render_block()
        if block:
            lines.extend(f"  {line}" for line in block.splitlines())
        elif self.geo_error:
            lines.append(f"  location lookup failed: {self.geo_error}")
        return "\n".join(lines)
