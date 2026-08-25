"""read_image -- the agent looking at a picture BY ITSELF.

The /image flag lets the HUMAN attach a picture to a prompt; this tool
completes the other half of the vision loop: the model deciding it
needs eyes. "What's in screenshot.png?", "does the chart I just
generated look right?", "read the diagram in docs/architecture.png" --
before this existed those all bounced off read_file's binary refusal,
which is honest but useless.

Mechanically it is thin ON PURPOSE: load_image_block (akshara.images)
already owns validation -- extension allowlist, 5 MB cap -- and both
provider dialects already carry ImageBlocks in user messages. The only
new machinery is the hand-off, and THAT is where the design lives:

``run()`` returns a ToolOutput (text + images). The loop splits it --
text becomes the ordinary ToolResult string; images are hoisted onto
history right after the results, because the OpenAI-family wires have
no way to put an image inside a role:"tool" payload at all. Anthropic
COULD nest one inside tool_result content, but one path through history
keeps the two dialects' transcripts identical -- which is exactly what
the adapter-parity tests pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from akshara.errors import ImageError, ToolError
from akshara.images import load_image_block
from akshara.tools.base import Tool, ToolContext, ToolOutput, require_str
from akshara.tools.fs import resolve_in_sandbox


class ReadImage(Tool):
    name = "read_image"
    description = (
        "Look at an image file from the sandbox (png/jpeg/gif/webp, "
        "<=5 MB) -- screenshots, diagrams, charts, photos. The image is "
        "attached to the result so you can see it; use read_file for "
        "anything textual."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Image file path, relative to the "
                                    "sandbox root."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    read_only = True

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"read image {require_str(args, 'path')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        raw = require_str(args, "path")
        path: Path = resolve_in_sandbox(ctx, raw)
        try:
            block = load_image_block(path)
        except ImageError as exc:
            raise ToolError(str(exc)) from exc
        return ToolOutput(
            text=(f"image loaded: {raw} ({block.media_type}, "
                  f"{len(block.data) * 3 // 4:,} bytes) -- it is attached "
                  "to this result"),
            images=[block],
        )
