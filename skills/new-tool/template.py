"""Starting point for a new built-in tool. Copy, rename, delete this line.

Replace every ALL-CAPS placeholder. Keep the docstring: this repo's tools
explain what they are for, not just what they do.
"""

from __future__ import annotations

from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str


class MyTool(Tool):
    """ONE LINE ON WHAT THIS IS FOR, then a paragraph on the why."""

    name: ClassVar[str] = "my_tool"
    description: ClassVar[str] = (
        "WHAT IT DOES, and -- just as important -- WHEN THE MODEL SHOULD "
        "REACH FOR IT. This text is also the BM25 document that decides "
        "whether the tool gets sent at all under dynamic selection."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "A REAL description -- what shape, what units, "
                               "what happens at the edges.",
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    #: True only if this cannot change anything outside the process AND
    #: sends nothing over the network. It drives permission auto-approval.
    read_only: ClassVar[bool] = False

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """What a human sees in the approval prompt BEFORE this runs.
        Resolve paths the same way run() will -- a preview that lies is
        worse than no preview."""
        return f"my_tool: {args.get('target')!r}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        target = require_str(args, "target")
        if not target.strip():
            # Errors the MODEL can fix are data, not crashes.
            raise ToolError("target must not be empty")
        return f"did the thing to {target}"
