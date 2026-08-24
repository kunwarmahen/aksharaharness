"""The real-world test: the harness doing its OWN JOB -- building a project.

Claude Code's core loop is: read a spec -> write files -> run tests ->
read the failures -> fix -> re-run until green -> run the program.
This demo hands that exact job to OUR agent, in a fresh workspace, and
then VERIFIES THE RESULT INDEPENDENTLY (it does not trust the model's
claims).

Since the build loop was promoted into the library, this file is now a
thin shell around ``akshara.builder``: the presets below are DEMO
CONTENT (spec text + acceptance commands); everything else -- the
build-system prompt, the event rendering, the independent verification,
the checksum contract -- lives in src/akshara/builder.py, which is what
``akshara --build SPEC`` and the REPL's /build command drive too.

Two job shapes:

  * BUILD  (unitconv / todo): greenfield -- spec in, project out.
  * REPAIR (repair): a seeded BROKEN project -- 5 planted bugs across
    textstats.py, a correct failing test suite. Make CI green WITHOUT
    touching the tests (they are checksummed; modifying them fails).
    This is the edit_file + read-the-traceback loop, live.

Run (real model calls -- needs a key in .env):

    uv run python examples/builder_demo.py                    # unitconv preset
    uv run python examples/builder_demo.py --preset todo      # stateful CLI app
    uv run python examples/builder_demo.py --preset repair    # fix the bugs
    uv run python examples/builder_demo.py --task "your own spec"
    uv run python examples/builder_demo.py --keep             # keep workspace

What this exercises end-to-end, live:
  * multi-iteration tool loops with REAL file writes and REAL subprocesses
  * errors-as-data in both directions: failing tests come back as tool
    output the model must read and fix; its own bugs are its problem
  * the permission story of an autonomous build: fs writes are sandboxed
    to the workspace by the tools themselves; bash runs yolo here -- same
    trust decision as `akshara --yolo` (use `akshara --sandbox --build`
    for the bwrap-confined variant instead)
  * exit code doubles as a CI gate: green build -> 0, anything else -> 1

Honest scope: one conversation, one small project, ~10-25 tool calls
(a few cents at current prices). The point is the LOOP SHAPE, not scale.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from akshara.builder import (
    BUILD_SYSTEM,
    BuildSpec,
    default_checks,
    run_build,
)
from akshara.cli.render import Renderer
from akshara.config import default_model, load_settings
from akshara.permissions import yolo
from akshara.providers import get_provider
from akshara.tools import default_registry
from akshara.agent import Agent

PRESETS: dict[str, dict] = {
    "unitconv": {
        "task": """\
Build a small Python project in the current directory.

PROJECT: `unitconv` -- a unit-conversion CLI. Create:
1. conversions.py -- pure functions cm_to_inch(cm), kg_to_lb(kg),
   celsius_to_fahrenheit(c), km_to_miles(km); each returns a float
   rounded to 2 decimals.
2. test_conversions.py -- unittest tests covering ALL four functions,
   including 0 C -> 32 F and at least one negative temperature.
3. convert.py -- CLI: `python3 convert.py <value> <unit>` where unit is
   one of cm|kg|c|km (case-insensitive). Prints e.g. `100 C = 212.00 F`
   and exits 0. Unknown unit: print usage to stderr and exit 2.

CONSTRAINTS: Python standard library ONLY; no network.

DEFINITION OF DONE -- run these yourself and iterate until all pass:
  python3 -m unittest discover -s . -v     # exit 0
  python3 convert.py 100 c                 # prints: 100 C = 212.00 F
  python3 convert.py 5 nope                # exit code 2

When done, reply with the file list and the final unittest summary line.""",
        "verify": [
            # (argv, expect_exit, expect_substring)
            ([sys.executable, "-m", "unittest", "discover", "-s", ".", "-v"], 0, None),
            ([sys.executable, "convert.py", "100", "c"], 0, "212.00"),
            ([sys.executable, "convert.py", "5", "nope"], 2, None),
        ],
    },
    "todo": {
        "task": """\
Build a tiny TODO CLI in the current directory.

Create todo.py -- subcommands persisting to todo.json in THIS directory:
  python3 todo.py add "buy milk"    -> prints: added #1
  python3 todo.py list              -> numbered lines, `[ ]` open / `[x]` done
  python3 todo.py done 1            -> prints: done #1
Create test_todo.py -- unittest tests covering add -> list -> done -> list
(keep each test isolated: back up / remove todo.json around each test).

CONSTRAINTS: Python standard library ONLY; no network.

DEFINITION OF DONE -- run these yourself and iterate until all pass:
  python3 -m unittest discover -s . -v    # exit 0
  the add/list/done sequence above works exactly as specified

When done, reply with the file list and the final unittest summary line.""",
        "verify": [
            ([sys.executable, "-m", "unittest", "discover", "-s", ".", "-v"], 0, None),
            ([sys.executable, "todo.py", "add", "demo item"], 0, "added #1"),
            ([sys.executable, "todo.py", "list"], 0, "[ ]"),
            ([sys.executable, "todo.py", "done", "1"], 0, "done #1"),
            ([sys.executable, "todo.py", "list"], 0, "[x]"),
        ],
    },
    # REPAIR: a seeded broken project. The tests are correct and FAILING;
    # the implementation carries 5 planted bugs (whitespace splitting,
    # integer division, case normalization drift, sort direction + tie
    # break, boundary off-by-one). Offline-proven: 6 failures as shipped,
    # 14/14 after the intended fixes.
    "repair": {
        "seed": "textstats",
        "task": """\
The project in the current directory has FAILING tests. Your job is to
make `python3 -m unittest discover -s . -v` exit 0.

RULES:
- The tests are the contract. Do NOT modify test_textstats.py -- its
  contents are checksummed and any change fails the build.
- Fix the implementation in textstats.py.
- Each docstring in textstats.py states the intended behavior; where a
  docstring and the code disagree, the tests define truth.

Run the tests, read every failure carefully, fix the code, and re-run.
Repeat until all 14 tests pass. When done, reply with each bug found as
a one-line diagnosis (function / what was wrong / what you changed),
followed by the final unittest summary line.""",
        "verify": [
            ([sys.executable, "-m", "unittest", "discover", "-s", ".", "-v"], 0, None),
        ],
    },
}


def report(console: Console, result) -> None:
    """Independent verification + verdict, from the BuildResult."""
    console.print("\n[bold]independent verification[/bold] "
                  "(re-running acceptance commands ourselves)")
    for outcome in result.checks:
        shown = " ".join(Path(a).name if i == 1 else a
                         for i, a in enumerate(outcome.argv))
        mark = "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]"
        suffix = "" if outcome.passed else (
            f"  (exit {outcome.actual_exit}, expected {outcome.expect_exit})")
        console.print(f"  {mark}  $ {shown}{suffix}")
        if not outcome.passed and outcome.tail:
            console.print(f"        {outcome.tail}", style="red",
                          markup=False, highlight=False)
    for name in result.tampered_tests:
        console.print(f"  [red]FAIL[/red]  {name} was MODIFIED -- the "
                      "tests are the contract; fixing them is cheating")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--preset", default="unitconv", choices=sorted(PRESETS))
    parser.add_argument("--task", help="custom spec (overrides --preset)")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--keep", action="store_true",
                        help="keep the workspace instead of deleting it")
    args = parser.parse_args()

    console = Console()
    workspace = Path(tempfile.mkdtemp(prefix="akshara-build-"))

    # Repair jobs start from a seeded BROKEN project; the library then
    # checksums every test file so "make it green" cannot be satisfied by
    # weakening the tests.
    preset = PRESETS[args.preset]
    spec = BuildSpec(
        task=args.task or preset["task"],
        checks=preset["verify"] if not args.task else default_checks(),
        seed_dir=(Path(__file__).parent / "broken_projects" / preset["seed"]
                  if not args.task and preset.get("seed") else None),
    )

    def factory(ws: Path) -> Agent:
        return Agent(
            get_provider(args.provider, load_settings(args.provider)),
            model=default_model(args.provider),
            system=BUILD_SYSTEM,
            tools=default_registry(),
            permissions=yolo,          # autonomous build: same trust as --yolo;
            cwd=ws,                    # fs writes stay sandboxed by the tools
            max_iterations=args.max_iterations,
        )

    console.print(f"[bold]{args.provider}[/bold] · {factory(workspace).model} · "
                  f"workspace {workspace}\n[dim]gate: yolo (autonomous "
                  "build -- bash unsandboxed; try `akshara --sandbox "
                  "--build`)[/dim]\n")

    try:
        result = run_build(factory, spec, workspace, on_event=Renderer(console))
    except KeyboardInterrupt:
        console.print("\n[yellow](cancelled -- history left resumable)[/yellow]")
        sys.exit(130)

    report(console, result)
    usage = f"{result.usage_in} in / {result.usage_out} out"
    console.print(f"\n[bold]{'BUILD GREEN' if result.ok else 'BUILD RED'}"
                  f"[/bold] · {result.elapsed_seconds:.1f}s · "
                  f"{len(result.files)} files: {', '.join(result.files)}\n"
                  f"tokens: {usage}")

    if not args.keep:
        shutil.rmtree(workspace, ignore_errors=True)
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
