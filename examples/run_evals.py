"""Live eval suite: trajectory-level checks against a REAL model.

The offline suite proves the eval MACHINERY (tests/test_evals.py);
this script measures actual behavior -- completion, tool discipline,
budgets -- and is meant to run on a merge/nightly cadence, not per
commit (cost + flakiness; book ch19's tests-vs-evals split).

    uv run --env-file .env python examples/run_evals.py          # sequential
    uv run --env-file .env python examples/run_evals.py --async  # concurrent

Exit code doubles as a gate: 0 all green, 1 any failure -- wire into
CI as a pre-merge/pre-model-upgrade check. Runs with --yolo because
one case deliberately exercises bash; that is the point of the
forbidden/required-tool checks, not an endorsement of yolo generally.

--async drives the same cases through AsyncEvalRunner: identical
grading rules, several trajectories at once (bounded by a semaphore),
results still reported in submission order.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from akshara.config import default_model, load_settings  # noqa: E402
from akshara.evals import (  # noqa: E402
    AsyncEvalRunner,
    EvalCase,
    EvalRunner,
    spawn_setup,
    summarize,
)
from akshara.errors import ConfigError  # noqa: E402
from akshara.permissions import yolo  # noqa: E402
from akshara.providers import get_provider  # noqa: E402
from akshara.tools import default_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="readme-summary",
            description="summarize the project README",
            user_message="Read README.md and summarize what this project "
                         "is in two sentences.",
            required_tools=["read_file"],
            max_tokens=30_000,
            max_iterations=6,
        ),
        EvalCase(
            # first live run taught us why these are two cases: asked
            # neutrally, the model counted via `bash ls` -- correct
            # answer, "wrong" tool. A neutral question must not carry a
            # tool requirement; if you require a tool, SAY SO.
            id="notes-count",
            description="counting task, any sane route; measures cost",
            user_message="How many markdown files does the notes/ directory "
                         "contain? Report just the number.",
            check_answer=lambda ans: any(ch.isdigit() for ch in ans),
            max_tokens=20_000,
            max_iterations=5,
        ),
        EvalCase(
            id="list-dir-discipline",
            description="explicit tool preference must be followed",
            user_message="Using the dedicated directory listing tool (NOT "
                         "bash), show what's inside src/akshara/, then name "
                         "one file you saw.",
            required_tools=["list_dir"],
            forbidden_tools=["bash"],
            max_tokens=15_000,
            max_iterations=4,
        ),
        EvalCase(
            id="multiply-via-bash",
            description="must actually RUN the math, not answer from memory",
            user_message="Use bash to compute 137 * 214 exactly, then report "
                         "the result.",
            required_tools=["bash"],
            max_tokens=15_000,
            max_iterations=4,
        ),
        EvalCase(
            id="grep-discipline",
            description="code search must use grep, not shell tricks",
            user_message="Which file in src/ defines SubagentSpawner? "
                         "Answer with just the path.",
            required_tools=["grep"],
            forbidden_tools=["bash"],
            max_tokens=25_000,
            max_iterations=6,
        ),
        EvalCase(
            id="premature-finalization-trap",
            description="answer must contain every computed square",
            user_message="For each of 1..5, compute its square with bash "
                         "(one call per number), then list all five squares.",
            required_tools=["bash"],
            max_tokens=30_000,
            max_iterations=10,
        ),
        EvalCase(
            # the setup seam under test: this case's agent gets a
            # spawn_subagent tool; "required" asserts delegation actually
            # HAPPENED (narrating instead of delegating shows up as
            # "required tool not used"), and the child's list_dir flows
            # into the same record via shared tool instances.
            id="delegate-file-survey",
            description="delegation: coordinator spawns, child surveys",
            user_message="Use the spawn_subagent tool to delegate this: "
                         "spawn a child whose tools_allowed is exactly "
                         "[\"list_dir\"], objective 'list src/akshara/ and "
                         "name one .py file you saw'. Relay its answer.",
            required_tools=["spawn_subagent", "list_dir"],
            forbidden_tools=["bash"],
            max_tokens=30_000,
            max_iterations=5,
            setup=spawn_setup(),
        ),
    ]


def main() -> int:
    try:
        provider_name = ("ollama" if "--ollama" in sys.argv else None) or \
            _guess_provider()
        provider = get_provider(provider_name, load_settings(provider_name))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = os.environ.get(f"{provider_name.upper()}_MODEL") or \
        default_model(provider_name)

    runner: EvalRunner | AsyncEvalRunner
    if "--async" in sys.argv:
        runner = AsyncEvalRunner(provider, model, tools=default_registry(),
                                 permissions=yolo, cwd=REPO_ROOT)
        cases = build_cases()
        start = time.monotonic()
        results = asyncio.run(runner.run_all(cases))
        mode = f"async (concurrency={runner.concurrency})"
    else:
        runner = EvalRunner(provider, model, tools=default_registry(),
                            permissions=yolo, cwd=REPO_ROOT)
        cases = build_cases()
        start = time.monotonic()
        results = runner.run_all(cases)
        mode = "sequential"
    print(f"== {provider_name}/{model} -- {mode} -- "
          f"{time.monotonic() - start:.1f}s wall ==")
    print(summarize(results))
    return 0 if all(r.passed for r in results) else 1


def _guess_provider() -> str:
    from akshara.config import _load_dotenv
    _load_dotenv()
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise ConfigError("no API key found -- see .env.example")


if __name__ == "__main__":
    raise SystemExit(main())
