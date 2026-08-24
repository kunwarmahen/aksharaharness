"""Shared vocabulary for the whole harness.

These types are the ONLY representation the rest of the program sees.
Provider adapters translate between these objects and each provider's
wire format -- nothing else in the codebase knows what a wire message
looks like. See notes/02-wire-formats.md for the two formats.

Two deliberate simplifications, both inherited from the Anthropic shape:

* There is no ``"system"`` role. The system prompt is per-request
  metadata (Anthropic puts it top-level; OpenAI puts it in
  ``messages[0]``), so it lives outside the conversation history.
* There is no ``"tool"`` role. A tool result is a *block inside a user
  message*, exactly like Anthropic's ``tool_result``. The OpenAI adapter
  fans those blocks out into ``role:"tool"`` wire messages mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TextBlock:
    """A span of plain text (user input or assistant output)."""

    text: str


@dataclass(slots=True)
class ToolCall:
    """The assistant asking *us* to run a tool.

    ``arguments`` is always a parsed dict internally. JSON-as-a-string
    exists only inside adapters (OpenAI transmits arguments as a string).
    """

    id: str  # echoed back verbatim in the matching ToolResult
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """Our answer to a ToolCall, appended inside a user message."""

    tool_call_id: str
    content: str  # stdout / file contents / error message -- always a str
    is_error: bool = False


@dataclass(slots=True)
class ThinkingBlock:
    """The model's visible reasoning (Anthropic ``type: "thinking"``).

    Preserved, not merely displayed: continuing an assistant turn that
    used thinking REQUIRES those blocks -- signature included -- to come
    back verbatim alongside the tool results, or the next request fails
    the signature check. So this block round-trips through history and
    back onto the wire like any other.

    The OpenAI-style dialect has no way to send reasoning back; its
    adapter drops ThinkingBlocks on encode (see providers/openai.py).
    """

    thinking: str
    signature: str = ""


@dataclass(slots=True)
class RedactedThinkingBlock:
    """Anthropic's ``type: "redacted_thinking"`` -- reasoning the safety
    system encrypted. Unlike every other unknown block, this one has a
    CONTRACT: it must round-trip verbatim (the ``data`` payload is opaque
    ciphertext the provider validates), so it is a first-class type
    rather than a placeholder. Display shows only that it exists.
    """

    data: str


@dataclass(slots=True)
class ImageBlock:
    """An image the USER supplies, base64-encoded (input-side only --
    models answer in text/thinking/tool calls, never with images).

    ``media_type`` is the MIME type ("image/png", "image/jpeg", ...);
    ``data`` is raw base64 WITHOUT any ``data:`` prefix -- each adapter
    adds its own wire dialect's framing (Anthropic nests a source
    object; OpenAI wants a data: URL).
    """

    media_type: str
    data: str


Block = (TextBlock | ToolCall | ToolResult | ThinkingBlock |
         RedactedThinkingBlock | ImageBlock)


@dataclass(slots=True)
class Message:
    """One turn of conversation: a role plus an ordered list of blocks."""

    role: Role
    content: list[Block]

    def text(self) -> str:
        """All TextBlocks concatenated (convenience for display/tests)."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_calls(self) -> list[ToolCall]:
        """ToolCalls made by the assistant in this message, in order."""
        return [b for b in self.content if isinstance(b, ToolCall)]


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Usage:
    """Token counts for one model response.

    One convention, enforced at the adapter boundary so downstream code
    (cost math especially) never needs to know which wire format a
    count came from: ``input_tokens`` counts ONLY tokens billed at the
    full input rate, and cached tokens appear exclusively in the cache
    counters. OpenAI-dialect wires fold cached hits INTO the prompt
    total; their adapters subtract them back out (see notes/21).

    Consequence: the four counters are DISJOINT. The window footprint of
    one request is the SUM of all four (``window_tokens``), not just the
    first -- a cached session bills almost nothing fresh yet still fills
    the context window.

    Zeros are normal: some providers/upstreams don't report usage on
    streamed calls. Never treat missing usage as an error.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0  # Anthropic: cache_read_input_tokens
    cache_write_tokens: int = 0  # Anthropic: cache_creation_input_tokens

    def add(self, other: Usage) -> None:
        """Accumulate another Usage into this one (for session totals)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def window_tokens(self) -> int:
        """How much of the model's context window this request filled:
        every token carried in the prompt, whatever rate it billed at."""
        return (self.input_tokens + self.cache_read_tokens +
                self.cache_write_tokens)


StopReason = Literal[
    "end_turn",  # model finished its answer          (OpenAI: stop)
    "tool_use",  # model wants tools executed         (OpenAI: tool_calls)
    "max_tokens",  # ran out of output budget         (OpenAI: length)
    "stop_sequence",  # hit a configured stop sequence (no OpenAI equivalent)
    "refusal",  # model refused                       (OpenAI: content_filter)
    "other",  # anything unrecognized (e.g. pause_turn)
]


@dataclass(slots=True)
class ModelResponse:
    """One complete model reply, normalized across providers."""

    message: Message  # append to history verbatim
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)  # zeros are normal (see Usage)
    model: str = ""
    raw: dict[str, Any] | None = None  # untouched provider JSON: for learning/debug


# ---------------------------------------------------------------------------
# Stream events -- what Provider.stream() yields
#
# A tagged union of tiny dataclasses instead of one god-event: consumers
# use structural pattern matching (match/case) and illegal states are
# unrepresentable. Deliberately minimal: five kinds cover everything the
# CLI renders and collect() folds.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StartEvent:
    """First event of a stream: names the model that answered."""

    model: str


@dataclass(slots=True)
class TextDelta:
    """A fragment of assistant text (render live, in order)."""

    text: str


@dataclass(slots=True)
class ToolCallStart:
    """A tool call began: id and name known, arguments not yet complete.

    ``index`` keys the call within the stream -- required because OpenAI
    fragments PARALLEL tool calls interleaved by index (id/name usually
    appear only on the first fragment for each index), and Anthropic keys
    blocks by their content-block index.
    """

    index: int
    id: str
    name: str


@dataclass(slots=True)
class ToolCallDelta:
    """A fragment of a tool call's arguments JSON.

    ``index`` identifies which tool call the fragment belongs to --
    required because OpenAI keys fragments by index within tool_calls,
    Anthropic by content-block index, and fragments arrive out-of-band
    from the id/name.
    """

    index: int
    partial_json: str


@dataclass(slots=True)
class ThinkingDelta:
    """A fragment of streamed reasoning.

    Carries EITHER a prose fragment (``text``) or a signature fragment
    (``signature``), mirroring Anthropic's ``thinking_delta`` /
    ``signature_delta``. Like ToolCallDelta, ``index`` keys the block
    within the stream; there is no start event -- collect() lazily
    creates the ThinkingBlock on first sight of an index, because unlike
    tool calls nothing here needs to be known up front.
    """

    index: int
    text: str = ""
    signature: str = ""


@dataclass(slots=True)
class RedactedThinking:
    """A complete ``type: "redacted_thinking"`` block, announced at
    content_block_start.

    The one streamed block that arrives WHOLE and delta-free: its
    ``data`` payload is opaque ciphertext, so there is nothing
    human-shaped to stream incrementally. It must still reach collect()
    -- dropping it here would 400 the NEXT request of the same tool
    loop, because redacted blocks round-trip verbatim like signed ones.
    """

    index: int
    data: str


@dataclass(slots=True)
class EndEvent:
    """Terminal event: why the model stopped + usage for the call."""

    stop_reason: StopReason
    usage: Usage


StreamEvent = (StartEvent | TextDelta | ThinkingDelta | RedactedThinking |
               ToolCallStart | ToolCallDelta | EndEvent)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Wire-neutral tool definition handed to providers.

    ``parameters`` is JSON Schema: {"type": "object", "properties": {...}}.
    """

    name: str
    description: str
    parameters: dict[str, Any]
