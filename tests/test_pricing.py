"""Cost accounting: slug matching, billing math, and where $ shows up.

Offline by construction -- prices are data, no network anywhere. The
numbers under test are the LIST-PRICE SNAPSHOT baked into pricing.py;
if that table is ever refreshed, update the arithmetic here with it.
"""

from __future__ import annotations

import httpx
import io
import json
import pytest
from rich.console import Console

from conftest import ScriptedProvider, assistant_text

from akshara.agent import Agent, TurnEnd
from akshara.cli.render import Renderer
from akshara.cli.repl import Repl
from akshara.errors import ConfigError
from akshara.pricing import (
    ModelPrice,
    cost_of,
    price_for,
    session_cost,
)
from akshara.permissions import allow_read_only
from akshara.types import Message, ModelResponse, TextBlock, Usage


# ---------------------------------------------------------------------------
# Slug matching: exact -> date-stripped -> vendor-stripped -> family prefix.
# ---------------------------------------------------------------------------


class TestMatching:
    def test_exact_slug(self):
        assert price_for("claude-sonnet-4-5").output_per_mtok == 15.0
        assert price_for("gpt-5-mini").input_per_mtok == 0.25

    @pytest.mark.parametrize("slug", [
        "claude-opus-5-20260201",           # dated snapshot suffix
        "anthropic/claude-opus-5",          # OpenRouter-style vendor prefix
        "anthropic/claude-opus-5-20260201",  # both at once
        "openrouter/anthropic/claude-opus-5",  # nested vendor prefixes
    ])
    def test_normalizations_land_on_same_row(self, slug):
        assert price_for(slug) == price_for("claude-opus-5")

    def test_family_prefix_catches_new_point_releases(self):
        # a model that did not exist when the table was written still
        # lands on its family's tier instead of "unknown"
        assert price_for("claude-opus-9-future") == price_for("claude-opus-5")
        assert price_for("gpt-5.7") == price_for("gpt-5")

    def test_longest_prefix_wins(self):
        # gpt-5.4-mini must not fall through to the plain gpt-5.4 row
        assert price_for("gpt-5.4-mini").input_per_mtok == 0.75
        assert price_for("gpt-5.4").input_per_mtok == 2.50

    def test_unknown_is_none_never_a_guess(self):
        assert price_for("qwen3.8") is None
        assert price_for("") is None

    def test_pro_rows_have_no_cached_rate_fallback_to_input(self):
        # "-pro" endpoints publish no cached-input price; conservative
        # fallback bills cached tokens at full input rate
        price = price_for("gpt-5-pro")
        assert price.cache_read_per_mtok is None
        usage = Usage(input_tokens=10, cache_read_tokens=90)
        assert cost_of(usage, price) == (100 * 15.00) / 1_000_000


class TestOverrides:
    def _write(self, tmp_path, payload) -> str:
        path = tmp_path / "prices.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_exact_override_and_new_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AKSHARA_PRICES", self._write(tmp_path, {
            "claude-haiku-4-5": {"input": 9.0, "output": 9.0},
            "my-gateway/model-x": {"input": 1.0, "output": 2.0},
        }))
        assert price_for("claude-haiku-4-5") == ModelPrice(9.0, 9.0)
        assert price_for("my-gateway/model-x") == ModelPrice(1.0, 2.0)

    def test_prefix_override_beats_builtins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AKSHARA_PRICES", self._write(tmp_path, {
            "gpt-5*": {"input": 42.0, "output": 43.0},
        }))
        assert price_for("gpt-5-mini").input_per_mtok == 42.0
        assert price_for("gpt-4o-mini").input_per_mtok == 0.15  # untouched

    def test_free_local_entry_zero_rates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AKSHARA_PRICES", self._write(tmp_path, {
            "qwen3.8": {"input": 0.0, "output": 0.0,
                        "cached_read": 0.0, "cached_write": 0.0},
        }))
        cost = cost_of(Usage(input_tokens=10**9), price_for("qwen3.8"))
        assert cost == 0.0

    def test_bad_json_is_loud(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.json"
        path.write_text("{nope")
        monkeypatch.setenv("AKSHARA_PRICES", str(path))
        with pytest.raises(ConfigError, match="not valid JSON"):
            price_for("gpt-4o")

    def test_missing_fields_are_loud(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AKSHARA_PRICES", self._write(
            tmp_path, {"m": {"output": 1.0}}))
        with pytest.raises(ConfigError, match="'input' and 'output'"):
            price_for("m")


# ---------------------------------------------------------------------------
# Billing math over the normalized (disjoint-counter) Usage convention.
# ---------------------------------------------------------------------------


SONNET = price_for("claude-sonnet-4-5")


class TestMath:
    def test_plain_in_out(self):
        assert cost_of(Usage(input_tokens=1_000_000, output_tokens=1_000_000),
                       ModelPrice(3.0, 15.0)) == pytest.approx(18.0)

    def test_cache_split_billed_at_own_rates(self):
        usage = Usage(input_tokens=1000, output_tokens=300,
                      cache_read_tokens=500, cache_write_tokens=200)
        expected = (1000 * 3.0 + 500 * 0.30 + 200 * 3.75 + 300 * 15.0) / 1e6
        assert cost_of(usage, SONNET) == pytest.approx(expected)

    def test_none_cache_rates_fall_back_to_input_price(self):
        usage = Usage(cache_read_tokens=1_000_000)
        assert cost_of(usage, ModelPrice(2.0, 10.0)) == pytest.approx(2.0)


class TestSessionCost:
    def test_empty_session_is_free_and_complete(self):
        assert session_cost({}) == (0.0, True)

    def test_unpriced_bucket_excluded_but_reported(self):
        total, complete = session_cost({
            "claude-sonnet-4-5": Usage(output_tokens=1_000_000),
            "qwen3.8": Usage(output_tokens=999),
        })
        assert total == pytest.approx(15.0)
        assert complete is False


def test_window_tokens_sums_all_prompt_side_counters():
    usage = Usage(input_tokens=10, output_tokens=5,
                  cache_read_tokens=100, cache_write_tokens=20)
    assert usage.window_tokens() == 130  # output excluded: it leaves, not fills


# ---------------------------------------------------------------------------
# Adapter boundary: cached hits ride INSIDE OpenAI-dialect headline totals
# and are subtracted back out there -- so pricing math never sees them twice.
# ---------------------------------------------------------------------------


def _openai_payload(model: str, prompt: int, cached: int, completion: int) -> dict:
    return {
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


@pytest.mark.parametrize("provider_name,settings_fx,payload", [
    ("openai", "openai_settings",
     _openai_payload("gpt-4o-mini", 1000, 400, 50)),
    ("responses", "responses_settings",
     {"model": "gpt-5", "status": "completed",
      "output": [{"type": "message", "role": "assistant",
                  "content": [{"type": "output_text", "text": "hi"}]}],
      "usage": {"input_tokens": 1000, "output_tokens": 50,
                "input_tokens_details": {"cached_tokens": 400}}}),
])
def test_openai_family_subtracts_cached_from_headline(
        provider_name, settings_fx, payload, request):
    from akshara.providers import get_provider
    settings = request.getfixturevalue(settings_fx)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    provider = get_provider(provider_name, settings,
                            transport=httpx.MockTransport(handler))
    response = provider.complete(messages=[], system=None, tools=[],
                                 model="x")
    # wire says 1000 prompt-side; 400 of those were cached hits, so the
    # internal vocabulary must store 600 full-rate + 400 read -- NOT 1000+400
    assert response.usage.input_tokens == 600
    assert response.usage.cache_read_tokens == 400
    assert response.usage.output_tokens == 50
    assert response.usage.window_tokens() == 1000


def test_openai_cached_over_prompt_clamps_to_zero():
    """A gateway reporting more cached than prompt tokens must not produce
    a negative counter -- clamp, don't poison the session totals."""
    payload = _openai_payload("gpt-4o-mini", 100, 400, 1)
    from akshara.providers.openai import OpenAIProvider
    from akshara.providers.base import ProviderSettings
    provider = OpenAIProvider(
        ProviderSettings(api_key="k", base_url="http://mock.local/v1"),
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
    usage = provider.complete(messages=[], system=None, tools=[],
                              model="x").usage
    assert usage.input_tokens == 0
    assert usage.cache_read_tokens == 400


def test_streaming_usage_chunk_gets_same_subtraction():
    """The streaming path assembles Usage incrementally; the convention
    must hold there too (same rule, second implementation)."""
    from akshara.providers.openai import OpenAIProvider
    from akshara.providers.base import ProviderSettings
    sse = (
        'data: {"model":"gpt-4o-mini","choices":[{"index":0,'
        '"delta":{"content":"hi"}}]}\n\n'
        'data: {"model":"gpt-4o-mini","choices":[],'
        '"usage":{"prompt_tokens":1000,"completion_tokens":7,'
        '"prompt_tokens_details":{"cached_tokens":400}}}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    provider = OpenAIProvider(
        ProviderSettings(api_key="k", base_url="http://mock.local/v1"),
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=sse)))
    events = list(provider.stream(messages=[], system=None, tools=[],
                                  model="x"))
    usage = events[-1].usage
    assert usage.input_tokens == 600
    assert usage.cache_read_tokens == 400


# ---------------------------------------------------------------------------
# The loop buckets usage per model, so /usage can price a mid-session
# /model switch correctly.
# ---------------------------------------------------------------------------


def _response(model: str, inp: int, out: int) -> ModelResponse:
    return ModelResponse(message=Message("assistant", [TextBlock("ok")]),
                         stop_reason="end_turn",
                         usage=Usage(input_tokens=inp, output_tokens=out),
                         model=model)


class TestAgentBuckets:
    def test_usage_buckets_per_model(self):
        agent = Agent(ScriptedProvider([
            _response("claude-sonnet-4-5", 100, 10),
            _response("gpt-4o-mini", 200, 20),
            _response("claude-sonnet-4-5", 30, 3),   # same bucket again
        ]), model="whatever", permissions=allow_read_only)
        for _ in range(3):
            agent.run("go")
        assert agent.usage_by_model["claude-sonnet-4-5"].input_tokens == 130
        assert agent.usage_by_model["gpt-4o-mini"].output_tokens == 20
        assert agent.total_usage.input_tokens == 330  # unchanged aggregate

    def test_last_context_tokens_is_window_footprint(self):
        resp = _response("m", 100, 5)
        resp.usage.cache_read_tokens = 40
        resp.usage.cache_write_tokens = 10
        agent = Agent(ScriptedProvider([resp]), model="m",
                      permissions=allow_read_only)
        agent.run("go")
        assert agent.last_context_tokens == 150


# ---------------------------------------------------------------------------
# Surfaces: /usage dollars line and the per-turn footer.
# ---------------------------------------------------------------------------


def _repl_with(agent: Agent) -> Repl:
    return Repl(agent, Console(file=io.StringIO(), width=120),
                input_fn=lambda prompt: "/quit")


def _console_text(repl: Repl) -> str:
    return repl.console.file.getvalue()  # type: ignore[attr-defined]


class TestReplCostLine:
    def test_priced_session_shows_dollars(self):
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=allow_read_only)
        agent.usage_by_model["claude-sonnet-4-5"] = \
            Usage(output_tokens=100_000)
        repl = _repl_with(agent)
        repl._command("/usage")
        assert "~$1.5000" in _console_text(repl)

    def test_unpriced_model_never_renders_as_zero(self):
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=allow_read_only)
        agent.usage_by_model["some-mystery-model"] = \
            Usage(output_tokens=50_000)
        repl = _repl_with(agent)
        repl._command("/usage")
        text = _console_text(repl)
        assert "$0.0000" not in text
        assert "no list price known" in text

    def test_mixed_session_flags_partial_coverage(self):
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=allow_read_only)
        agent.usage_by_model["claude-sonnet-4-5"] = \
            Usage(output_tokens=100_000)
        agent.usage_by_model["qwen3.8"] = Usage(output_tokens=1)
        repl = _repl_with(agent)
        repl._command("/usage")
        text = _console_text(repl)
        assert "~$1.5000" in text and "priced models only" in text

    def test_local_provider_is_genuinely_free(self):
        class FakeOllama(ScriptedProvider):
            name = "ollama"

        agent = Agent(FakeOllama([]), model="qwen3.8",
                      permissions=allow_read_only)
        agent.usage_by_model["qwen3.8"] = Usage(input_tokens=10**6)
        repl = _repl_with(agent)
        repl._command("/usage")
        assert "$0.00 (local model)" in _console_text(repl)

    def test_no_usage_no_cost_line(self):
        agent = Agent(ScriptedProvider([]), model="m",
                      permissions=allow_read_only)
        repl = _repl_with(agent)
        repl._command("/usage")
        assert "cost" not in _console_text(repl)


class TestFooterCost:
    def _turn_end(self, model: str) -> TurnEnd:
        return TurnEnd(response=_response(model, 1000, 100),
                       reason="end_turn", iterations=1)

    def _render(self, event) -> str:
        console = Console(file=io.StringIO(), width=200)
        Renderer(console)(event)
        return console.file.getvalue()  # type: ignore[attr-defined]

    def test_priced_model_shows_figure(self):
        text = self._render(self._turn_end("gpt-4o-mini"))
        # (1000*0.15 + 100*0.60)/1e6 = 0.00021 -> rounds to 0.0002 at 4dp;
        # assert on the format marker rather than floating dust
        assert "~$0.0002" in text

    def test_unknown_model_omits_dollars_entirely(self):
        text = self._render(self._turn_end("totally-unknown"))
        assert "$" not in text
