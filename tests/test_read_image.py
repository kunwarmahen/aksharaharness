"""read_image end to end: tool -> loop hoist -> three wire encodings ->
session round-trip.

The interesting seams are NOT the loader (akshara.images already pins
that) but what happens AFTER a tool returns pixels:

* the loop splits ToolOutput into a text result + images hoisted onto
  history AFTER the results (sync AND async twins),
* each dialect encodes that mixed user message VALIDLY: results must
  precede any image-carrying user message on the OpenAI-family wires,
  which cannot carry an image inside role:"tool" at all,
* checkpoints survive -- ImageBlocks used to be silently DROPPED by
  session persistence; now they round-trip byte-exact.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from akshara.agent import Agent
from akshara.async_agent import AsyncAgent
from akshara.errors import ToolError
from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.openai import OpenAIProvider
from akshara.providers.responses import ResponsesProvider
from akshara.session import _dump_message, _load_message
from akshara.tools import ReadImage, ToolRegistry
from akshara.tools.base import ToolContext, ToolOutput
from akshara.types import (
    ImageBlock,
    Message,
    TextBlock,
    ToolResult,
)

B64 = base64.b64encode(b"fake-png-bytes").decode()


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    (tmp_path / "pic.png").write_bytes(b"\x89PNG-fake")
    return ToolContext(cwd=tmp_path)


def _scripted(agent_cls, tmp_path):
    from conftest import (
        ScriptedProvider,
        assistant_text,
        assistant_tool_call,
    )
    provider = ScriptedProvider([
        assistant_tool_call("call_1", "read_image", {"path": "pic.png"}),
        assistant_text("I have seen the picture."),
    ])
    registry = ToolRegistry()
    registry.register(ReadImage())
    return agent_cls(provider, model="m", tools=registry, cwd=tmp_path)


# ---------------------------------------------------------------------------
# The tool itself
# ---------------------------------------------------------------------------


class TestReadImageTool:
    def test_returns_text_plus_image(self, ctx):
        out = ReadImage().run({"path": "pic.png"}, ctx)
        assert isinstance(out, ToolOutput)
        assert "pic.png" in out.text and "attached" in out.text
        assert len(out.images) == 1
        assert out.images[0].media_type == "image/png"

    def test_missing_file_is_tool_error(self, ctx):
        with pytest.raises(ToolError, match="image not found"):
            ReadImage().run({"path": "nope.png"}, ctx)

    def test_bad_extension_is_tool_error(self, ctx, tmp_path):
        (tmp_path / "notes.txt").write_bytes(b"just text")
        with pytest.raises(ToolError):
            ReadImage().run({"path": "notes.txt"}, ctx)

    def test_sandbox_escape_refused(self, ctx):
        with pytest.raises(ToolError, match="escapes sandbox"):
            ReadImage().run({"path": "../outside.png"}, ctx)

    def test_read_only_and_summary(self, ctx):
        assert ReadImage.read_only is True
        assert "pic.png" in ReadImage().summary({"path": "pic.png"}, ctx)


# ---------------------------------------------------------------------------
# The loop hoist (both twins)
# ---------------------------------------------------------------------------


class TestLoopHoist:
    @pytest.mark.parametrize("agent_cls", [Agent, AsyncAgent],
                             ids=["sync", "async"])
    def test_images_ride_history_after_the_result(self, agent_cls, tmp_path):
        (tmp_path / "pic.png").write_bytes(b"\x89PNG-fake")
        agent = _scripted(agent_cls, tmp_path)

        run = agent.run("look at pic.png") if agent_cls is Agent \
            else asyncio.run(agent.run("look at pic.png"))

        assert run.message.text() == "I have seen the picture."
        batch = agent.history[2]  # user -> assistant(tool_call) -> results
        assert batch.role == "user"
        assert isinstance(batch.content[0], ToolResult)
        assert batch.content[0].tool_call_id == "call_1"
        # THE assertion: the image follows the result IN HISTORY, so both
        # adapters see it through their ordinary block-encoding path.
        assert isinstance(batch.content[1], ImageBlock)
        assert base64.b64decode(batch.content[1].data) == b"\x89PNG-fake"


# ---------------------------------------------------------------------------
# Wire encodings of the mixed message (results first, image after)
# ---------------------------------------------------------------------------


def mixed_message() -> Message:
    return Message("user", [
        ToolResult("call_1", "image loaded: pic.png"),
        ImageBlock("image/png", B64),
        ImageBlock("image/jpeg", B64),
    ])


class TestWireEncodings:
    def test_anthropic_keeps_block_order(self, anthropic_settings):
        provider = AnthropicProvider(anthropic_settings)
        body = provider.build_request_body(
            messages=[mixed_message()], system=None, tools=[],
            model="m", max_tokens=10)
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "tool_result"
        assert content[1] == {"type": "image",
                              "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": B64}}
        assert content[2]["source"]["media_type"] == "image/jpeg"

    def test_openai_tools_first_then_user_parts(self, openai_settings):
        provider = OpenAIProvider(openai_settings)
        body = provider.build_request_body(
            messages=[mixed_message()], system=None, tools=[],
            model="m", max_tokens=10)
        msgs = body["messages"]
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_1"
        assert msgs[1]["role"] == "user"
        parts = msgs[1]["content"]
        assert [p["type"] for p in parts] == ["image_url", "image_url"]
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_responses_outputs_first_then_user_message(self, responses_settings):
        provider = ResponsesProvider(responses_settings)
        body = provider.build_request_body(
            messages=[mixed_message()], system=None, tools=[],
            model="m", max_tokens=10)
        items = body["input"]
        assert items[0]["type"] == "function_call_output"
        assert items[1]["type"] == "message"
        assert items[1]["content"][0]["type"] == "input_image"

    @pytest.mark.parametrize("provider_cls,settings_name", [
        (OpenAIProvider, "openai_settings"),
        (ResponsesProvider, "responses_settings"),
    ], ids=["openai", "responses"])
    def test_plain_shapes_byte_identical(self, provider_cls, settings_name,
                                         request):
        # no images in play => the historical shapes, untouched
        provider = provider_cls(request.getfixturevalue(settings_name))
        body = provider.build_request_body(
            messages=[Message("user", [TextBlock("hi")])],
            system=None, tools=[], model="m", max_tokens=10)
        first = body["input"] if settings_name == "responses_settings" \
            else body["messages"]
        if settings_name == "responses_settings":
            assert first[0]["content"][0] == {"type": "input_text",
                                              "text": "hi"}
        else:
            assert first == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Session round-trip (images used to be dropped here)
# ---------------------------------------------------------------------------


class TestSessionRoundTrip:
    def test_tool_result_with_image_survives_checkpoint(self):
        message = mixed_message()
        restored = _load_message(_dump_message(message))
        kinds = {type(b) for b in restored.content}
        assert kinds == {ToolResult, ImageBlock}
        assert restored.content[1].media_type == "image/png"
        assert restored.content[1].data == B64  # byte-exact, no re-encode

    def test_user_attachment_image_survives_too(self):
        message = Message("user", [TextBlock("what's this?"),
                                   ImageBlock("image/webp", B64)])
        restored = _load_message(_dump_message(message))
        assert isinstance(restored.content[1], ImageBlock)
        assert restored.content[1].media_type == "image/webp"
