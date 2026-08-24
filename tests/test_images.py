"""User-supplied images: loading, both wire encodings, loop attach.

The feature has three seams and each gets pinned here:

* ``akshara.images`` -- file -> ImageBlock (validation happens BEFORE
  any request exists; bad input is a usage error, not turn data),
* the adapters -- one internal ImageBlock, two wire dialects
  (Anthropic nests a source object; OpenAI wants a data: URL inside an
  ordered parts array -- while image-free requests keep their plain
  string shape, byte-identical to before vision existed),
* the loop -- text first, then images, all in ONE user message, in
  both the sync agent and its async twin.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from akshara.agent import Agent
from akshara.async_agent import AsyncAgent
from akshara.errors import ImageError
from akshara.images import MAX_IMAGE_BYTES, load_image_block
from akshara.providers.anthropic import AnthropicProvider
from akshara.providers.openai import OpenAIProvider
from akshara.tools.base import ToolRegistry
from akshara.types import ImageBlock, Message, TextBlock

FIXTURES = Path(__file__).parent / "fixtures"

# "ABC" base64-encoded -- content is irrelevant to every seam below;
# none of them decode pixels.
B64 = base64.b64encode(b"ABC").decode()


# ---------------------------------------------------------------------------
# Loading (akshara.images)
# ---------------------------------------------------------------------------


class TestLoadImageBlock:
    def test_png_roundtrip(self, tmp_path):
        raw = b"\x89PNG-not-really-but-bytes-are-bytes"
        path = tmp_path / "shot.png"
        path.write_bytes(raw)

        block = load_image_block(path)

        assert block.media_type == "image/png"
        assert base64.b64decode(block.data) == raw

    def test_jpg_maps_to_jpeg_mime(self, tmp_path):
        path = tmp_path / "photo.jpg"
        path.write_bytes(b"ff")

        assert load_image_block(path).media_type == "image/jpeg"

    def test_missing_file_raises_before_any_turn(self, tmp_path):
        with pytest.raises(ImageError, match="not found"):
            load_image_block(tmp_path / "ghost.png")

    def test_unsupported_extension_names_the_supported_set(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("plain")

        with pytest.raises(ImageError, match=r"\.png.*\.webp"):
            load_image_block(path)

    def test_oversize_is_rejected_on_raw_bytes(self, tmp_path, monkeypatch):
        import akshara.images as images_mod
        monkeypatch.setattr(images_mod, "MAX_IMAGE_BYTES", 4)
        path = tmp_path / "big.png"
        path.write_bytes(b"x" * 5)

        with pytest.raises(ImageError, match="cap"):
            load_image_block(path)

    def test_default_cap_is_five_megabytes(self):
        assert MAX_IMAGE_BYTES == 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Wire encodings (one block, two dialects)
# ---------------------------------------------------------------------------


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class TestAnthropicImageEncoding:
    def test_image_becomes_base64_source_block(self, anthropic_settings):
        sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json=_fixture("anthropic_text.json"))

        provider = AnthropicProvider(anthropic_settings,
                                     transport=httpx.MockTransport(handler))
        block = ImageBlock(media_type="image/png", data=B64)
        provider.complete(
            messages=[Message("user", [TextBlock("look"), block])],
            system=None, tools=[], model="m",
        )

        body = json.loads(sent[-1].content)
        assert body["messages"][0]["content"][-1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": B64},
        }
        # order preserved: text stayed first
        assert body["messages"][0]["content"][0] == {"type": "text",
                                                     "text": "look"}


class TestOpenAIImageEncoding:
    def _provider(self, settings, sent):
        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json=_fixture("openai_text.json"))
        return OpenAIProvider(settings, transport=httpx.MockTransport(handler))

    def test_multimodal_message_becomes_typed_parts_array(
            self, openai_settings):
        sent: list[httpx.Request] = []
        provider = self._provider(openai_settings, sent)
        block = ImageBlock(media_type="image/jpeg", data=B64)
        provider.complete(
            messages=[Message("user", [TextBlock("look"), block])],
            system=None, tools=[], model="m",
        )

        content = json.loads(sent[-1].content)["messages"][0]["content"]
        assert content == [
            {"type": "text", "text": "look"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{B64}"}},
        ]

    def test_plain_messages_keep_the_string_shape(self, openai_settings):
        # the parts-array shape must appear ONLY when an image is present
        sent: list[httpx.Request] = []
        provider = self._provider(openai_settings, sent)
        provider.complete(messages=[Message("user", [TextBlock("hi")])],
                          system=None, tools=[], model="m")

        body = json.loads(sent[-1].content)
        assert body["messages"][0]["content"] == "hi"


# ---------------------------------------------------------------------------
# The loop attaches images (sync + async twins)
# ---------------------------------------------------------------------------


def _agent(provider, *, async_=False):
    registry = ToolRegistry()
    cls = AsyncAgent if async_ else Agent
    return cls(provider, model="scripted-model", tools=registry)


class TestLoopAttach:
    def test_sync_run_rides_one_user_message(self):
        from conftest import ScriptedProvider, assistant_text

        block = ImageBlock(media_type="image/png", data=B64)
        provider = ScriptedProvider([assistant_text("seen")])
        agent = _agent(provider)

        response = agent.run("describe this", images=[block])

        assert response.message.text() == "seen"
        sent = provider.requests[0]["messages"]
        assert sent[0].content == [TextBlock("describe this"), block]
        # history keeps the same blocks for later turns / compaction
        assert agent.history[0].content[1] is block

    def test_async_twin_attaches_identically(self):
        from conftest import ScriptedProvider, assistant_text

        block = ImageBlock(media_type="image/png", data=B64)
        provider = ScriptedProvider([assistant_text("seen")])
        agent = _agent(provider, async_=True)

        async def _run():
            return await agent.run("describe this", images=[block])

        response = asyncio.run(_run())

        assert response.message.text() == "seen"
        sent = provider.requests[0]["messages"]
        assert sent[0].content == [TextBlock("describe this"), block]

    def test_no_images_leaves_the_classic_single_text_block(self):
        from conftest import ScriptedProvider, assistant_text

        provider = ScriptedProvider([assistant_text("ok")])
        agent = _agent(provider)
        agent.run("plain")

        assert provider.requests[0]["messages"][0].content == \
            [TextBlock("plain")]


# ---------------------------------------------------------------------------
# Context accounting (compaction must not be blind to images)
# ---------------------------------------------------------------------------


def test_estimate_bills_images_by_decoded_size():
    from akshara.context import estimate_tokens

    msg = Message("user", [TextBlock("hi"),
                           ImageBlock(media_type="image/png", data=B64)])
    # per-block overhead (4 * 2 + 8) + text chars + decoded-size proxy
    # (base64 len * 3/4), floor-divided once at the end
    assert estimate_tokens(msg) == (16 + len("hi") + len(B64) * 3 // 4) // 4


# ---------------------------------------------------------------------------
# CLI surface (--image -> one-shot turn)
# ---------------------------------------------------------------------------


class TestCliImageFlag:
    def test_flag_wires_images_into_the_one_shot_turn(
            self, tmp_path, monkeypatch):
        import akshara.cli.main as cli_main
        from conftest import ScriptedProvider, assistant_text

        img = tmp_path / "dot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        provider = ScriptedProvider([assistant_text("seen")])
        monkeypatch.setattr(cli_main, "load_settings", lambda name: object())
        monkeypatch.setattr(cli_main, "get_provider", lambda *a, **k: provider)
        captured: dict = {}

        def fake_run_turn(self, text, *, images=None):
            captured["text"], captured["images"] = text, images

        monkeypatch.setattr(cli_main.Repl, "run_turn", fake_run_turn)

        rc = cli_main.main([
            "--cwd", str(tmp_path), "--provider", "anthropic",
            "--model", "m", "--yolo",
            "--image", str(img), "describe this",
        ])

        assert rc == 0
        assert captured["text"] == "describe this"
        assert captured["images"] and \
            captured["images"][0].media_type == "image/png"

    def test_flag_without_a_prompt_is_a_usage_error(self, capsys):
        import akshara.cli.main as cli_main

        assert cli_main.main(["--image", "whatever.png"]) == 2
        assert "needs a prompt" in capsys.readouterr().err

    def test_bad_image_file_exits_2_before_any_turn(
            self, tmp_path, monkeypatch, capsys):
        import akshara.cli.main as cli_main

        called = False

        def boom(self, text, *, images=None):
            nonlocal called
            called = True

        monkeypatch.setattr(cli_main.Repl, "run_turn", boom)
        rc = cli_main.main([
            "--cwd", str(tmp_path), "--provider", "anthropic",
            "--model", "m", "--yolo",
            "--image", str(tmp_path / "ghost.png"), "go",
        ])

        assert rc == 2
        assert not called  # usage error: the turn never started
