"""Shared test helpers.

The ONLY seam between adapters and the network is the ``transport``
argument on Provider.__init__: tests pass httpx.MockTransport(handler),
which serves canned responses from pure functions while exercising the
real request-building code path.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from akshara.providers.base import Provider, ProviderSettings
from akshara.types import (
    Message,
    ModelResponse,
    StartEvent,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    EndEvent,
    Usage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> bytes:
    """Raw fixture bytes -- kept raw so SSE tests can split chunks freely."""
    return (FIXTURES / name).read_bytes()


class Recorder:
    """MockTransport handler wrapper that records every request it sees.

    Usage:
        recorder = Recorder(lambda req: httpx.Response(200, json=payload))
        provider = SomeProvider(settings, transport=httpx.MockTransport(recorder))
        provider.complete(...)
        assert recorder.calls[0].url.path == "/v1/messages"
        assert recorder.last_body()["max_tokens"] == 1024
    """

    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self.calls: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self._responder(request)

    def last_body(self) -> dict:
        """The JSON body of the most recent request."""
        return json.loads(self.calls[-1].content)


@pytest.fixture
def anthropic_settings() -> ProviderSettings:
    return ProviderSettings(api_key="test-key", base_url="https://mock.anthropic.local")


@pytest.fixture
def openai_settings() -> ProviderSettings:
    return ProviderSettings(api_key="test-key", base_url="https://mock.openai.local/v1")


@pytest.fixture
def responses_settings() -> ProviderSettings:
    # Same /v1-inclusive base contract as openai; adapter appends /responses.
    return ProviderSettings(api_key="test-key", base_url="https://mock.responses.local/v1")


# ---------------------------------------------------------------------------
# ScriptedProvider: a fake Provider for agent-loop tests.
#
# Scripted with canned ModelResponses. stream() re-synthesizes the event
# vocabulary (StartEvent -> text/tool fragments -> EndEvent), so the loop
# under test exercises collect() too -- if collect() ever dropped or
# mangled a field, tests would see the mismatch against the scripted
# response, not silently pass.
# ---------------------------------------------------------------------------


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, script: list[ModelResponse]) -> None:
        super().__init__(ProviderSettings(api_key="script", base_url="http://script.local"))
        self.script = list(script)
        self.requests: list[dict] = []  # kwargs of every call, in order

    def _next(self, **kwargs) -> ModelResponse:
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError(
                "script exhausted -- the loop made an unscripted model call"
            )
        return self.script.pop(0)

    def complete(self, *, messages, system, tools, model,
                 max_tokens=16384, temperature=None) -> ModelResponse:
        return self._next(messages=list(messages), system=system, tools=tools,
                          model=model, max_tokens=max_tokens)

    def stream(self, *, messages, system, tools, model,
               max_tokens=16384, temperature=None):
        response = self._next(messages=list(messages), system=system, tools=tools,
                              model=model, max_tokens=max_tokens)
        yield from _events_for(response)

    async def acomplete(self, *, messages, system, tools, model,
                        max_tokens=16384, temperature=None) -> ModelResponse:
        return self.complete(messages=messages, system=system, tools=tools,
                             model=model, max_tokens=max_tokens)

    async def astream(self, *, messages, system, tools, model,
                      max_tokens=16384, temperature=None):
        for event in self.stream(messages=messages, system=system, tools=tools,
                                 model=model, max_tokens=max_tokens):
            yield event

    def last_request(self) -> dict:
        return self.requests[-1]


def _events_for(response: ModelResponse) -> Iterator[StreamEvent]:
    """Re-express one ModelResponse as the events a real provider emits."""
    yield StartEvent(model=response.model)
    index = 0  # tool-call stream index; Anthropic counts text blocks too,
    # but collect() keys calls by their OWN index either way.
    for block in response.message.content:
        match block:
            case TextBlock(text=text):
                mid = len(text) // 2  # split in half to prove fragment joining
                yield TextDelta(text[:mid])
                yield TextDelta(text[mid:])
            case ThinkingBlock(thinking=text, signature=sig):
                mid = len(text) // 2
                yield ThinkingDelta(index=index, text=text[:mid])
                yield ThinkingDelta(index=index, text=text[mid:])
                if sig:
                    yield ThinkingDelta(index=index, signature=sig)
            case ToolCall(id=cid, name=name, arguments=args):
                yield ToolCallStart(index=index, id=cid, name=name)
                raw = json.dumps(args)
                if raw != "{}":  # empty args => no deltas, like real streams
                    yield ToolCallDelta(index=index, partial_json=raw)
                index += 1
    yield EndEvent(stop_reason=response.stop_reason, usage=response.usage)


def assistant_text(text: str, *, usage: Usage | None = None) -> ModelResponse:
    """Script helper: a plain end_turn answer."""
    return ModelResponse(
        message=Message("assistant", [TextBlock(text)]),
        stop_reason="end_turn",
        usage=usage or Usage(),
    )


def assistant_tool_call(call_id: str, name: str, arguments: dict,
                        *, text_before: str = "") -> ModelResponse:
    """Script helper: a tool_use turn (optionally with leading prose)."""
    content: list = []
    if text_before:
        content.append(TextBlock(text_before))
    content.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return ModelResponse(
        message=Message("assistant", content),
        stop_reason="tool_use",
        usage=Usage(),
    )
