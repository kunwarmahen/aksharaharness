"""One non-streaming call, all layers visible -- now for
either provider, so you can diff the two wire formats on the same prompt.

Run:
    uv run python examples/one_shot.py "Why is the sky blue?"
    uv run python examples/one_shot.py --provider openai "Why is the sky blue?"

Prints, in order:
  1. the exact JSON request body we are about to send
  2. the raw JSON the provider sent back
  3. the normalized ModelResponse our code actually works with

Side by side on purpose: "normalization" stops being abstract once you
have seen the same exchange in all three shapes. Run it once per provider
and compare sections 1-2; section 3 is byte-identical logic either way.
"""

from __future__ import annotations

import argparse
import json

from akshara.config import default_model, load_settings
from akshara.providers import get_provider
from akshara.types import Message, TextBlock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("prompt", nargs="*", default=["Why is the sky blue?"])
    args = parser.parse_args()
    prompt = " ".join(args.prompt)

    settings = load_settings(args.provider)
    provider = get_provider(args.provider, settings)
    model = default_model(args.provider)
    messages = [Message("user", [TextBlock(prompt)])]
    path = "/v1/messages" if args.provider == "anthropic" else "/chat/completions"

    body = provider.build_request_body(
        messages=messages, system=None, tools=[], model=model, max_tokens=1024
    )
    print("=" * 25, f"REQUEST  (POST {path})", "=" * 25)
    print(json.dumps(body, indent=2))

    response = provider.complete(
        messages=messages, system=None, tools=[], model=model, max_tokens=1024
    )

    print("\n" + "=" * 25, "RAW RESPONSE JSON", "=" * 25)
    print(json.dumps(response.raw, indent=2))

    print("\n" + "=" * 25, "NORMALIZED ModelResponse", "=" * 25)
    print(f"model:       {response.model}")
    print(f"stop_reason: {response.stop_reason}")
    print(f"usage:       {response.usage}")
    print(f"text:        {response.message.text()!r}")

    print("\n" + "=" * 25, "ASSISTANT", "=" * 25)
    print(response.message.text())


if __name__ == "__main__":
    main()
