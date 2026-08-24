"""Ollama support: a local profile of the OpenAI dialect.

The wire half is inherited (that's the point -- chat-completions is
chat-completions), so these tests pin the parts that are actually NEW:
keyless config, the distinct provider name sessions must persist, small
local context-window defaults, and one smoke request proving the
inherited adapter really points at localhost's /v1/chat/completions.
"""

from __future__ import annotations

import httpx
import pytest

from akshara.config import (
    default_context_window,
    default_model,
    load_settings,
)
from akshara.providers import get_provider
from akshara.providers.ollama import OllamaProvider
from akshara.providers.openai import OpenAIProvider
from akshara.types import Message, TextBlock


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Pin the config surface regardless of the developer's shell."""
    for var in ("OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
                "OLLAMA_CONTEXT_WINDOW"):
        monkeypatch.delenv(var, raising=False)


class TestConfig:
    def test_works_with_no_key(self):
        settings = load_settings("ollama")
        # placeholder value: header exists, local server ignores it
        assert settings.api_key == "ollama"
        assert settings.base_url == "http://localhost:11434/v1"

    def test_env_overrides_everything(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
        monkeypatch.setenv("OLLAMA_API_KEY", "proxy-secret")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
        monkeypatch.setenv("OLLAMA_CONTEXT_WINDOW", "131072")

        settings = load_settings("ollama")
        assert settings.api_key == "proxy-secret"
        assert settings.base_url == "http://gpu-box:11434/v1"
        assert default_model("ollama") == "llama3.2"
        assert default_context_window("ollama") == 131_072

    def test_defaults_are_local_scale(self):
        assert default_model("ollama") == "qwen3.8:latest"
        # an 8k-window local model under the cloud 200k assumption would
        # never auto-compact before the provider rejected the request
        assert default_context_window("ollama") == 8192


class TestFactory:
    def test_is_openai_dialect_under_its_own_name(self):
        provider = get_provider("ollama", load_settings("ollama"))
        assert isinstance(provider, OpenAIProvider)  # inherits the wire
        assert provider.name == "ollama"  # /save stores THIS, --resume rebuilds it


class TestWireSmoke:
    def test_inherited_adapter_hits_local_chat_completions(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={
                "id": "cmpl-x", "object": "chat.completion", "created": 0,
                "model": "qwen3:4b",
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": "local hello"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            })

        provider = get_provider(
            "ollama", load_settings("ollama"),
            transport=httpx.MockTransport(handler),
        )
        response = provider.complete(
            messages=[Message("user", [TextBlock("hi")])],
            system=None, tools=[], model="qwen3:4b", max_tokens=100,
        )

        request = seen[0]
        assert request.url.host == "localhost"
        assert request.url.port == 11434
        assert request.url.path == "/v1/chat/completions"  # INCLUDES /v1
        assert request.headers["authorization"] == "Bearer ollama"
        assert response.message.text() == "local hello"


class TestSessionRoundTripName:
    def test_saved_payload_keeps_ollama_identity(self, tmp_path):
        """The whole reason for a distinct name: /save must record
        'ollama' so --resume rebuilds this provider, not plain openai."""
        from conftest import ScriptedProvider

        from akshara.agent import Agent
        from akshara.session import SessionStore

        agent = Agent(ScriptedProvider([]), model="qwen3:4b")
        store = SessionStore(tmp_path / "s.sqlite3")

        version = store.save(agent, provider_name="ollama")

        assert version == 1
        assert store.load_latest()["provider"] == "ollama"
