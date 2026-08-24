"""Context management: the window is a resource, spend it deliberately.

Two-layer strategy (book ch7-8, adapted), in strict order:

1. MASK older tool results -- reversible and precise. The tool CALL
   stays verbatim; only its bulky OUTPUT becomes a placeholder telling
   the model it can re-run if needed. Tool results dominate transcripts
   (often 70-90% by turn ten), so this alone usually clears red zone.
2. SUMMARIZE the old middle segment -- lossy, so only if still over.
   The first user message (the goal) survives verbatim; the summarizer
   must enumerate every tool call + outcome, because "the record of
   what was DONE must survive compaction" or the agent re-does it.

The invariant is untouchable by construction: masking keeps every
tool_call id answered, and summarize-cut points are validated to never
split an assistant(tool_use) <-> user(tool_result) pair.

Token numbers here are BUDGET PROXIES (chars/4 heuristic), never billing
figures. Real signal: the provider's own input_tokens for the last call.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from akshara.types import (
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

MASK_MARKER = "[tool result elided"
KEEP_RESULTS = 3          # newest tool results stay verbatim under masking
KEEP_TURNS = 6            # newest messages stay verbatim under summarization
YELLOW, RED = 0.60, 0.80  # utilization thresholds of the usable window

SUMMARY_PROMPT = """Summarize the conversation segment below for an AI agent \
that will continue the work with no other record of it. Requirements:
- List EVERY tool call in order with a one-line outcome (never omit one).
- Preserve exact file paths, commands, names, and numbers.
- Note open questions and what remains to be done.
- Summarize only; invent nothing. Max 400 words.

SEGMENT:
"""


# ---- measurement -----------------------------------------------------------


def estimate_tokens(message: Message) -> int:
    """Chars-over-4 proxy + small per-block overhead. A budget guess."""
    total = 4 * len(message.content) + 8
    for block in message.content:
        match block:
            case TextBlock(text=text):
                total += len(text)
            case ThinkingBlock(thinking=text, signature=_):
                total += len(text) + len("x") * 32
            case RedactedThinkingBlock(data=payload):
                total += len(payload)  # ciphertext still spends real window
            case ToolResult(content=out, is_error=_):
                total += len(out)
            case ToolCall(id=_, name=name, arguments=args):
                total += len(name) + len(json.dumps(args))
            case ImageBlock(data=b64):
                # Upstream bills images by decoded pixels, not base64
                # chars; decoded-bytes-over-4 is a deliberately generous
                # proxy -- overestimating triggers compaction early,
                # underestimating risks overflow 400s.
                total += len(b64) * 3 // 4
    return total // 4


def estimate_history(history: list[Message]) -> int:
    return sum(estimate_tokens(m) for m in history)


def utilization(history: list[Message], *, context_window: int,
                max_tokens: int) -> float | None:
    """Fraction of USABLE window consumed (usable = window - reply headroom)."""
    usable = max(context_window - max_tokens, 1)
    if not history:
        return None
    return min(estimate_history(history) / usable, 1.0)


# ---- layer 1: mask tool-result outputs -------------------------------------


def _mask_text(call_id: str, original_len: int) -> str:
    return (f"{MASK_MARKER} to save context; call_id={call_id}; "
            f"original {original_len} chars; re-run the tool if you need "
            "this output again]")


def mask_old_results(history: list[Message],
                     *, keep_recent: int = KEEP_RESULTS) -> tuple[list[Message], int]:
    """Return a NEW list where all but the newest ``keep_recent``
    ToolResults have elided content. Idempotent: already-masked blocks
    keep their placeholder (and count toward recency, harmlessly)."""
    positions: list[tuple[int, int]] = []
    for mi, message in enumerate(history):
        for bi, block in enumerate(message.content):
            if isinstance(block, ToolResult):
                positions.append((mi, bi))

    masked_count = 0
    out = [None] * len(history)  # type: ignore[list-item]
    for mi, bi in reversed(positions[:-keep_recent] if keep_recent else positions):
        block = history[mi].content[bi]
        assert isinstance(block, ToolResult)
        if block.content.startswith(MASK_MARKER):
            continue  # idempotent: masking twice must not stack placeholders
        replacement = ToolResult(block.tool_call_id,
                                 _mask_text(block.tool_call_id, len(block.content)),
                                 is_error=block.is_error)
        message = out[mi] if out[mi] is not None else history[mi]
        content = list(message.content)
        content[bi] = replacement
        out[mi] = Message(message.role, content)
        masked_count += 1

    return [out[i] if out[i] is not None else m
            for i, m in enumerate(history)], masked_count


# ---- layer 2: summarize the middle -----------------------------------------


def _is_results_batch(m: Message) -> bool:
    return m.role == "user" and any(isinstance(b, ToolResult) for b in m.content)


def _has_calls(m: Message) -> bool:
    return m.role == "assistant" and bool(m.tool_calls())


def summarizable_span(history: list[Message],
                      *, keep_tail: int = KEEP_TURNS) -> tuple[int, int] | None:
    """(start, end) of a safely removable middle segment, or None.

    Both cut edges must avoid splitting an assistant(tool_use) from its
    user(tool_result): neither edge may fall ON a results batch whose
    partner sits on the OTHER side of that edge.
    """
    end = len(history) - keep_tail
    while end > 1 and _is_results_batch(history[end]):
        end -= 1  # tail must not OPEN with orphaned results
    start = 1  # history[0] -- the goal -- stays verbatim (book rule)
    while start < end and _is_results_batch(history[start]):
        start += 1  # span must not START on results whose calls stay behind
    if end - start < 2:
        return None
    return start, end


def render_segment(segment: list[Message]) -> str:
    lines: list[str] = []
    for message in segment:
        role = "USER" if message.role == "user" else "ASSISTANT"
        for block in message.content:
            match block:
                case TextBlock(text=t):
                    lines.append(f"{role}: {t}")
                case ToolCall(id=cid, name=n, arguments=a):
                    lines.append(f"{role} tool_call: {n}({json.dumps(a)}) id={cid}")
                case ToolResult(tool_call_id=cid, content=c, is_error=e):
                    flag = "ERROR" if e else "ok"
                    preview = c if len(c) <= 300 else c[:300] + "..."
                    lines.append(f"{role} tool_result[{cid}] ({flag}): {preview}")
                case ThinkingBlock(thinking=t, signature=_):
                    lines.append(f"{role} (reasoning): {t[:200]}")
                case RedactedThinkingBlock(data=payload):
                    # Tell the summarizer it exists; the ciphertext itself
                    # would only burn the summary's own budget.
                    lines.append(f"{role} (redacted reasoning: "
                                 f"{len(payload)} chars, encrypted)")
    return "\n".join(lines)


def compact_history(
    history: list[Message],
    *,
    summarize: Callable[[str], str] | None = None,
    context_window: int,
    max_tokens: int,
) -> tuple[list[Message], dict]:
    """Run both layers as needed; returns (new_history, stats).

    ``summarize`` receives the rendered segment text and returns the
    summary (a real implementation makes an LLM call; tests pass stubs).
    """
    stats: dict = {"masked": 0, "summarized": False, "segment_size": 0}
    new_history, masked = mask_old_results(history)
    stats["masked"] = masked

    still_red = utilization(new_history, context_window=context_window,
                            max_tokens=max_tokens)
    if summarize is None or still_red is None or still_red < RED:
        return new_history, stats

    span = summarizable_span(new_history)
    if span is None:
        stats["span"] = None
        return new_history, stats
    start, end = span
    segment = new_history[start:end]
    summary_text = summarize(render_segment(segment))
    replacement = Message("user", [TextBlock(
        "[Earlier conversation summarized to save context. The full record "
        "of tool calls made so far follows.]\n" + summary_text)])
    new_history = [*new_history[:start], replacement, *new_history[end:]]
    stats.update(summarized=True, segment_size=len(segment))
    return new_history, stats


async def acompact_history(
    history: list[Message],
    *,
    asummarize: Callable[[str], Awaitable[str]] | None = None,
    context_window: int,
    max_tokens: int,
) -> tuple[list[Message], dict]:
    """Async twin of :func:`compact_history`.

    Masking, red-zone arithmetic, and cut-point validation are pure and
    SHARED verbatim; only the LLM summarization can suspend, so that is
    the single line that differs from the sync function.
    """
    stats: dict = {"masked": 0, "summarized": False, "segment_size": 0}
    new_history, masked = mask_old_results(history)
    stats["masked"] = masked

    still_red = utilization(new_history, context_window=context_window,
                            max_tokens=max_tokens)
    if asummarize is None or still_red is None or still_red < RED:
        return new_history, stats

    span = summarizable_span(new_history)
    if span is None:
        stats["span"] = None
        return new_history, stats
    start, end = span
    segment = new_history[start:end]
    summary_text = await asummarize(render_segment(segment))
    replacement = Message("user", [TextBlock(
        "[Earlier conversation summarized to save context. The full record "
        "of tool calls made so far follows.]\n" + summary_text)])
    new_history = [*new_history[:start], replacement, *new_history[end:]]
    stats.update(summarized=True, segment_size=len(segment))
    return new_history, stats
