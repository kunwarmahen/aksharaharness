"""Tool selection as retrieval -- the book's answer to the TOOL CLIFF.

Selection accuracy is roughly flat to ~10 tools, degrades visibly by
20, and falls off a cliff between 30 and 50 -- while every tool's schema
burns 100-500 tokens PER REQUEST. MCP makes 50 easy: three modest
servers and you are past the cliff before the first user message.

The fix (book ch12): treat choosing tools as a RETRIEVAL problem.
Rank all tools against a query derived from the conversation, load only
the top K into the request, and pin the few that must always be there.
BM25 (not embeddings) because queries come from the agent itself and
share vocabulary with tool descriptions; it is ~60 lines, zero
dependencies, and swapping in embeddings later means replacing exactly
one method.

Selection is a context-economy decision about what gets SENT -- never
a second permission system. The loop soft-admits calls that name real
but unselected tools (see Agent._get_visible_tool); only hallucinated
names error.

Three pieces:

* ``ToolCatalog``      -- the index: BM25 over name+description tokens,
                         ``select(query, k, must_include)`` with a score
                         floor (no-match tools are EXCLUDED, not ranked last).
* ``query_from_transcript`` -- first user message as anchor + recent
                         assistant text and tool-call names. Tool names
                         carry vocabulary: "mcp__slack__post_message"
                         appearing in history is itself a query term.
* ``ListAvailableTools``    -- the pinned discovery hatch for the two
                         holes retrieval cannot close: vague openers
                         where everything scores zero, and mid-task pivots
                         whose new vocabulary hasn't reached the transcript.

Rule of thumb (book's numbers, kept as constants here): below
AUTO_THRESHOLD tools don't bother; above it this keeps the model sharp.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from akshara.tools.base import Tool, ToolContext

#: Book's rule of thumb: don't bother below this catalog size.
AUTO_SELECTION_THRESHOLD = 20

#: The autonomy loop's floor -- every task reads and writes files and
#: runs commands, but their descriptions only share vocabulary with a
#: query when the TASK is about files ("edit config.yaml"). "Fix this
#: bug in parser.py" matches none of them, so retrieval would drop
#: write_file exactly when the turn needs it next. Pins beat the query:
#: they load every turn regardless of score (book ch12: "pin the few
#: that must always be there"). The LONG TAIL -- MCP tools, browser,
#: background jobs -- stays retrievable; that is what selection is for.
CORE_PINS = ("read_file", "write_file", "edit_file",
             "bash", "glob", "grep")

#: Width used when selection auto-enables past AUTO_SELECTION_THRESHOLD.
#: Arithmetic: len(CORE_PINS) pins + the discovery hatch leaves ~5 slots
#: for retrieved long-tail tools -- k below the pin count would leave
#: BM25 nothing to do.
DEFAULT_TOOLS_PER_TURN = 12

_TOKEN_SPLIT = re.compile(r"[^a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-[a-z0-9_]. Underscores survive so
    ``post_message`` stays one term -- tool names are snake_case on purpose."""
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


@dataclass
class ToolCatalog:
    """All tools, ranked. The registry still OWNS the tools; this is a
    read-only index over them plus whatever got added after construction."""

    tools: list[Tool] = field(default_factory=list)
    k1: float = 1.5       # BM25 term-frequency saturation
    b: float = 0.75       # length-normalization strength
    must_include: tuple[str, ...] = ("list_available_tools",)

    def __post_init__(self) -> None:
        self._docs: list[list[str]] = []
        self._df: dict[str, int] = {}
        for tool in self.tools:
            self._index(tool)

    # ---- index maintenance -------------------------------------------------

    def add(self, tool: Tool) -> None:
        if self.get(tool.name) is not None:
            raise ValueError(f"duplicate tool name in catalog: {tool.name!r}")
        self.tools.append(tool)
        self._index(tool)

    def _index(self, tool: Tool) -> None:
        doc = tokenize(f"{tool.name} {tool.description}")
        for term in set(doc):
            self._df[term] = self._df.get(term, 0) + 1
        self._docs.append(doc)

    # ---- lookup ------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    # ---- the retrieval -----------------------------------------------------

    def scores(self, query: str) -> list[float]:
        """BM25 score of every tool against ``query`` (index order)."""
        n = len(self.tools)
        avgdl = sum(len(d) for d in self._docs) / n if n else 0.0
        out = []
        for doc in self._docs:
            score = 0.0
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            for term in tokenize(query):
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = counts.get(term, 0)
                denom = tf + self.k1 * (
                    1 - self.b + self.b * len(doc) / (avgdl or 1.0))
                score += idf * tf * (self.k1 + 1) / denom
            out.append(score)
        return out

    def select(self, query: str, k: int = 7,
               must_include: Iterable[str] | None = None) -> list[Tool]:
        """Top-K tools for this turn. Pins ALWAYS return (when they exist);
         the rest of the budget goes to positive-scoring matches, best first.
        A query nothing matches returns just the pins -- never a guess."""
        pins = self.must_include if must_include is None else must_include
        by_name = {t.name: t for t in self.tools}
        chosen: dict[str, Tool] = {}
        for name in pins:
            if name in by_name:
                chosen[name] = by_name[name]

        ranked = sorted(zip(self.scores(query), self.tools),
                        key=lambda pair: pair[0], reverse=True)
        for score, tool in ranked:
            if len(chosen) >= k:
                break
            if score <= 0:     # floor: no vocabulary overlap -> exclude
                break
            chosen.setdefault(tool.name, tool)
        return list(chosen.values())


def query_from_transcript(history: list, max_recent: int = 6) -> str:
    """What the model is UP TO, as one retrieval query.

    First user message anchors the session's goal; recent assistant text
    and tool-call NAMES track the current sub-task. Names matter most at
    the pivot moment: having called mcp__slack__list_channels puts slack
    vocabulary into the query BEFORE any prose about Slack exists.

    Recent TOOL RESULTS join too, truncated -- without them a
    ``list_available_tools`` answer (a user-role ToolResult full of tool
    names) never reaches the next selection, and the promised
    "discovered here, usable next step" silently fails.
    """
    parts: list[str] = []
    for message in history:
        if message.role == "user" and not parts:
            # The FIRST user message anchors the goal. Later role=="user"
            # messages are tool results: skipped here EXCEPT inside the
            # recent window below, where they carry fresh vocabulary.
            for block in message.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
    for message in [m for m in history[-max_recent:] if m.role == "assistant"]:
        for block in message.content:
            text = getattr(block, "text", None)
            name = getattr(block, "name", None)   # ToolCall blocks
            if text:
                parts.append(text)
            elif name:
                parts.append(name)
    for message in [m for m in history[-max_recent:] if m.role == "user"]:
        for block in message.content:
            text = str(getattr(block, "content", "") or "")
            if text:
                parts.append(text[:400])
    return " ".join(parts)


class ListAvailableTools(Tool):
    """The pinned discovery hatch (book ch12's critical piece).

    Selection hides tools; without this the model cannot know they
    exist, and hallucinated names are unrecoverable. This tool lists the
    FULL catalog on demand -- and its description teaches the contract:
    discovered tools become callable NEXT turn, when their names sit in
    the transcript and BM25 surfaces them naturally.
    """

    name = "list_available_tools"
    description = (
        "List every tool available in this environment, one per line "
        "(only a subset is loaded each turn). Call this whenever you "
        "suspect a tool for your task exists but isn't loaded right now. "
        "Tools named in this listing are loaded AUTOMATICALLY for your "
        "following steps -- after reading it, just use the tool you need."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "filter_term": {
                "type": "string",
                "description": "Optional substring filter on tool names/descriptions.",
            },
        },
        "additionalProperties": False,
    }
    read_only = True

    def __init__(self, catalog: ToolCatalog) -> None:
        self.catalog = catalog

    def summary(self, args: dict, ctx: ToolContext) -> str:
        return f"list_available_tools(filter={args.get('filter_term')!r})"

    def run(self, args: dict, ctx: ToolContext) -> str:
        needle = str(args.get("filter_term") or "").lower()
        lines = []
        for tool in self.catalog.tools:
            line = f"{tool.name} — {tool.description}"
            if not needle or needle in line.lower():
                lines.append(line)
        header = f"{len(lines)} tool(s) available (of {len(self.catalog.tools)}):"
        return "\n".join([header, *lines]) if lines else \
            f"no tool matches {needle!r}; try a shorter filter"


def enable_selection(registry, *, k: int = 7,
                     pins: Iterable[str] = CORE_PINS,
                     ) -> tuple[ToolCatalog, ListAvailableTools]:
    """Wire selection onto an existing registry: builds a fresh catalog
    over everything registered SO FAR and ensures exactly one discovery
    tool exists on both sides. ``pins`` name the always-loaded tools
    (default CORE_PINS plus the discovery hatch); names absent from the
    registry -- never registered, or disabled by the operator -- are
    skipped silently, so a pin list may be written against the FULL
    default toolset. IDEMPOTENT -- calling twice re-points the live
    discovery instance at the rebuilt index instead of colliding.
    Assign ``agent.tool_catalog`` and set ``agent.tools_per_turn = k``
    (explicit wiring over hidden magic)."""
    catalog = ToolCatalog([t for t in registry
                           if t.name != "list_available_tools"])
    if "list_available_tools" in registry:
        discovery = registry.get("list_available_tools")
        discovery.catalog = catalog  # re-point; keep one registered instance
        catalog.add(discovery)
    else:
        discovery = ListAvailableTools(catalog)
        catalog.add(discovery)
        registry.register(discovery)
    # Pins resolve against the FINAL toolset (discovery included), deduped,
    # so must_include always names what will actually pin.
    present = {t.name for t in catalog.tools}
    catalog.must_include = tuple(dict.fromkeys(
        name for name in (*pins, "list_available_tools")
        if name in present))
    return catalog, discovery
