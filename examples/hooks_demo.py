"""Tool hooks: watch every execution WITHOUT touching the loop.

Gates decide, hooks watch. The permission gate answers "may this
run?"; hooks observe "it IS running" / "it finished". Typical uses:
timing, audit logs, metrics. There is deliberately NO way to veto from
a hook -- that is the gate's job -- and a raising hook crashes loudly
(hooks are developer infrastructure, not untrusted input).

    uv run --env-file .env python examples/hooks_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from akshara.agent import Agent  # noqa: E402
from akshara.config import default_model, load_settings  # noqa: E402
from akshara.permissions import allow_read_only  # noqa: E402
from akshara.providers import get_provider  # noqa: E402
from akshara.tools import default_registry  # noqa: E402
from akshara.types import ToolCall, ToolResult  # noqa: E402


def main() -> None:
    provider_name = "openai" if "--openai" in sys.argv else "anthropic"
    provider = get_provider(provider_name, load_settings(provider_name))
    model = default_model(provider_name)

    def before(call: ToolCall) -> None:
        print(f"  ▶ {call.name} {json.dumps(call.arguments)}")

    def after(call: ToolCall, result: ToolResult) -> None:
        mark = "ERR" if result.is_error else "ok"
        print(f"  ◀ {call.name} -> {mark}, {len(result.content)} chars")

    agent = Agent(
        provider,
        model=model,
        system="Answer briefly.",
        tools=default_registry(),
        permissions=allow_read_only,   # the gate decides what may run
        on_before_tool=before,         # the hook watches what DID run
        on_after_tool=after,
    )

    answer = agent.run(
        "Read README.md and report how many top-level sections it has "
        "(lines starting with '# ')."
    )
    print(f"\nanswer: {answer.message.text()[:300]}")


if __name__ == "__main__":
    main()
