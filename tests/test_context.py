"""Context management: masking, safe summarization cuts, auto-compact.

The invariant interacts with compaction everywhere, so every test that
touches history shape also runs assert_history_resumable: masking keeps
ids answered by construction; the summarize cut is validated to never
split an assistant(tool_use) <-> user(tool_result) pair.
"""

from __future__ import annotations

from conftest import ScriptedProvider, assistant_text

from akshara.agent import Agent
from akshara.context import (
    MASK_MARKER,
    RED,
    compact_history,
    mask_old_results,
    summarizable_span,
)
from akshara.permissions import allow_read_only, yolo
from akshara.types import Message, TextBlock, ToolCall, ToolResult

from test_agent_loop import assert_history_resumable


def big_results_history(n_results=5, chars=5_000) -> list[Message]:
    history = [Message("user", [TextBlock("investigate the repo")])]
    for i in range(n_results):
        history.append(Message("assistant", [
            ToolCall(f"c{i}", "bash", {"command": f"cmd {i}"})]))
        history.append(Message("user", [
            ToolResult(f"c{i}", "x" * chars, is_error=False)]))
    history.append(Message("assistant", [TextBlock("all done")]))
    return history


# ---- layer 1: masking ---------------------------------------------------------


class TestMasking:
    def test_all_but_newest_three_elided(self):
        history = big_results_history(5)
        masked, count = mask_old_results(history)

        assert count == 2                       # 5 results - keep 3
        contents = [m.content[0].content for m in masked if _result_msg(m)]
        # oldest two elided, newest three verbatim:
        assert contents[0].startswith(MASK_MARKER)
        assert contents[1].startswith(MASK_MARKER)
        for verbatim in contents[2:]:
            assert not verbatim.startswith(MASK_MARKER)

    def test_placeholder_carries_call_id_and_hint(self):
        history = big_results_history(4)
        masked, _ = mask_old_results(history)
        first_result = next(m for m in masked if _result_msg(m)).content[0]
        assert isinstance(first_result, ToolResult)
        assert first_result.tool_call_id in first_result.content
        assert "re-run" in first_result.content
        assert first_result.is_error is False          # error flag preserved

    def test_masking_is_idempotent(self):
        history = big_results_history(4)
        once, n1 = mask_old_results(history)
        twice, n2 = mask_old_results(once)
        assert n1 == 1 and n2 == 0                     # second pass no-ops

    def test_masking_preserves_the_invariant(self):
        history = big_results_history(5)
        masked, _ = mask_old_results(history)
        agent_like = _fake_agent(masked)
        assert_history_resumable(agent_like)


def _result_msg(m: Message) -> bool:
    return m.role == "user" and m.content and isinstance(m.content[0], ToolResult)


class _FakeHistoryAgent:
    """Just enough Agent for assert_history_resumable()."""
    history = []


def _fake_agent(history) -> _FakeHistoryAgent:
    holder = _FakeHistoryAgent()
    holder.history = history
    return holder


# ---- layer 2: summarization cuts ------------------------------------------------


class TestSummarizableSpan:
    def test_first_message_and_tail_stay_out_of_span(self):
        # 14 messages total; the naive tail edge lands ON a results batch,
        # so it slides back to the assistant call-message before it.
        history = big_results_history(6)
        assert len(history) == 14
        assert summarizable_span(history) == (1, 7)

    def test_tail_never_opens_on_a_results_batch(self):
        # ...assistant(calls), user(results), assistant(text) at the end:
        history = big_results_history(3)
        history.append(Message("assistant", [
            ToolCall("cx", "bash", {})]))
        history.append(Message("user", [ToolResult("cx", "out")]))
        span = summarizable_span(history, keep_tail=2)
        assert history[span[1]].role != "user" or not _is_results(history[span[1]])

    def test_too_small_history_gives_no_span(self):
        assert summarizable_span(big_results_history(2)) is None


def _is_results(m: Message) -> bool:
    return any(isinstance(b, ToolResult) for b in m.content)


class TestCompactHistory:
    def test_masking_alone_clears_yellow_no_summary(self):
        history = big_results_history(5)
        new, stats = compact_history(
            history,
            summarize=lambda text: "SHOULD NOT BE CALLED",
            context_window=10**9, max_tokens=1000,
        )
        assert stats["masked"] == 2 and stats["summarized"] is False
        assert_history_resumable(_fake_agent(new))

    def test_summarizer_runs_when_still_red_after_masking(self):
        # Huge results + tiny window: even after masking, estimate stays red.
        history = big_results_history(8, chars=40_000)

        calls = []

        def fake_summarize(rendered: str) -> str:
            calls.append(rendered)
            return "- user asked to investigate\n- ran cmds c0..c5"

        new, stats = compact_history(
            history, summarize=fake_summarize,
            context_window=20_000, max_tokens=2000,
        )

        assert stats["summarized"] is True
        assert len(calls) == 1
        assert "tool_call" in calls[0]                 # record of actions survives
        replacement = new[1]                           # [0] = original goal
        assert "summarized" in replacement.content[0].text.lower()
        assert len(new) < len(history)
        assert_history_resumable(_fake_agent(new))

    def test_goal_message_survives_verbatim(self):
        history = big_results_history(8, chars=40_000)
        goal = history[0]
        new, stats = compact_history(
            history, summarize=lambda _: "summary",
            context_window=20_000, max_tokens=2000,
        )
        assert stats["summarized"] is True
        assert new[0] == goal


# ---- integration: the loop compacts automatically -------------------------------


class TestAutoCompactInLoop:
    def test_red_zone_triggers_masking_before_the_request(self):
        """Pre-seed a heavy history and give the agent a tiny window: the
        very first request must already carry masked placeholders."""
        agent = Agent(
            ScriptedProvider([assistant_text("done")]),
            model="m", tools=_registry_with_echo(), permissions=yolo,
            context_window=8000, max_tokens=1000,
        )
        agent.history.extend(big_results_history(5, chars=6000))

        list(agent.run_streaming("one more question"))

        request = agent.provider.last_request()["messages"]
        results = [b.content for m in request for b in m.content
                   if isinstance(b, ToolResult)]
        assert any(MASK_MARKER in r for r in results)
        assert agent.last_compaction["masked"] == 2
        assert_history_resumable(agent)

    def test_green_zone_never_compacts(self):
        agent = Agent(
            ScriptedProvider([assistant_text("done")]),
            model="m", tools=_registry_with_echo(), permissions=yolo,
            context_window=10**9, max_tokens=1000,
        )
        list(agent.run_streaming("tiny"))
        assert agent.last_compaction is None

    def test_manual_compact_reports_stats(self):
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=allow_read_only,
                      context_window=50_000, max_tokens=2000)
        agent.history.extend(big_results_history(6, chars=30_000))
        stats = agent.compact()
        assert stats["masked"] >= 3
        assert stats["messages_after"] <= stats["messages_before"]
        assert_history_resumable(agent)


def _registry_with_echo():
    from akshara.tools.base import Tool, ToolRegistry

    class Echo(Tool):
        name = "echo"
        description = "echo"
        parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
        read_only = True

        def summary(self, args, ctx): return "echo()"
        def run(self, args, ctx): return args.get("text", "")

    reg = ToolRegistry()
    reg.register(Echo())
    return reg
