"""AksharaHarness -- a from-scratch LLM agent harness, written to learn.

Layers, outside in:

* ``akshara.cli``     -- terminal UI (REPL, rendering, permission prompts)
* ``akshara.agent``   -- THE LOOP: model -> tool calls -> results -> repeat
* ``akshara.providers`` -- wire-format adapters behind one streaming interface
* ``akshara.tools``   -- what the model may do (fs, shell, grep) + sandboxing
* ``types``/``errors`` -- shared vocabulary; the ONLY representation anywhere

See README.md for the map and notes/ for per-topic write-ups.
"""

from akshara.agent import Agent, AgentEvent, ToolExecuted, TurnEnd
from akshara.config import default_model, load_settings
from akshara.images import load_image_block
from akshara.mcp import MCPServerConfig, MCPSession, connect_mcp, register_mcp
from akshara.permissions import (
    PermissionFn,
    PermissionRequest,
    allow_read_only,
    deny_all,
    yolo,
)
from akshara.pricing import ModelPrice, cost_of, price_for, session_cost
from akshara.providers import get_provider
from akshara.providers.base import Provider, ProviderSettings, collect
from akshara.tools import default_registry
from akshara.types import (
    Block,
    EndEvent,
    ImageBlock,
    Message,
    ModelResponse,
    RedactedThinkingBlock,
    StartEvent,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "Agent",
    "AgentEvent",
    "allow_read_only",
    "Block",
    "collect",
    "default_model",
    "default_registry",
    "deny_all",
    "EndEvent",
    "get_provider",
    "ImageBlock",
    "load_image_block",
    "load_settings",
    "Message",
    "ModelResponse",
    "MCPServerConfig",
    "MCPSession",
    "ModelPrice",
    "connect_mcp",
    "cost_of",
    "register_mcp",
    "PermissionFn",
    "PermissionRequest",
    "Provider",
    "price_for",
    "session_cost",
    "ProviderSettings",
    "RedactedThinkingBlock",
    "StartEvent",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallStart",
    "ToolExecuted",
    "ToolResult",
    "ToolSpec",
    "TurnEnd",
    "Usage",
    "yolo",
]
