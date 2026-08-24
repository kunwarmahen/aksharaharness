"""Raw token-by-token streaming, no REPL machinery.

Run:
    uv run python examples/stream_demo.py "Tell me a haiku"

Watch tokens arrive as the provider sends them -- that's all "streaming"
is: the HTTP body arrives in chunks, and we print each text fragment the
moment the SSE parser extracts it.
"""

from __future__ import annotations

import sys

from akshara.config import default_model, load_settings
from akshara.providers.anthropic import AnthropicProvider
from akshara.types import EndEvent, Message, StartEvent, TextBlock, TextDelta


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Write a haiku about rivers."
    model = default_model("anthropic")
    provider = AnthropicProvider(load_settings("anthropic"))
    messages = [Message("user", [TextBlock(prompt)])]

    print(f"[model: {model}]\n")
    for event in provider.stream(
        messages=messages, system=None, tools=[], model=model, max_tokens=1024
    ):
        match event:
            case TextDelta(text=fragment):
                print(fragment, end="", flush=True)
            case StartEvent():
                pass  # could render a header here
            case EndEvent(stop_reason=reason, usage=usage):
                print(
                    f"\n\n[done: {reason} | {usage.input_tokens} in / "
                    f"{usage.output_tokens} out]"
                )


if __name__ == "__main__":
    main()
