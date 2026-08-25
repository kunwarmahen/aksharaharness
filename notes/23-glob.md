# 23 — glob: finding files by name without paying for it

*Every tool that exists had a question that used to be awkward to ask.
glob's question is "where does this live?" — and the interesting part
of adding it was noticing who was paying for that question before.*

## The gap

grep searches CONTENTS; list_dir shows ONE directory. So "find every
config file", "where do the tests live?", "what did the build just
produce?" had only two answers, both bad:

1. **Grep for text the file might contain** — works until the file is
   new, binary-ish, or named by convention rather than content.
2. **Bash `find`** — works, but bash is permission-gated, so a pure
   read cost the human a y/n prompt. A harness that interrupts its
   human to ask "may I please look at the file names?" is using its
   gate backwards: gates exist for danger, not for curiosity.

The fix is one more read_only tool, and the whole implementation is
`pathlib.Path.glob` plus opinions.

## The opinions

**Newest first.** `ls -t`, not `ls`. The questions this tool actually
gets asked skew recency-hard ("which file did I just edit?", "what did
the build emit?"), so results sort by mtime descending with ties broken
by path for determinism. Claude Code's glob makes the same call.

**Same skip rules as grep.** `.git`, `node_modules`, `.venv` and
friends never appear regardless of pattern — imported straight from
`search.py`'s SKIP_DIRS so there is exactly ONE copy of the list.

**Escape rules come free.** Paths resolve through the same
`resolve_in_sandbox` helper as every file tool, so `../` escapes and
symlink escapes fail identically — including the symlink case, because
`.resolve()` canonicalizes BEFORE the containment check.

**A cap with an honest suffix**, like grep's match cap: 200 paths,
then `[showing N of M matches; raise limit]`.

## One small lesson in error surfaces

`pathlib` raises plain `ValueError` for structurally bad patterns
(`src**/*.py` — `**` glued mid-segment). Catching that and re-raising
as ToolError keeps the contract every other tool honors: *anything the
model could plausibly fix comes back as readable data*, not as an
exception the loop has to translate.

## What the tests pin

- flat and recursive (`**`) matching; directories never listed
- skip dirs excluded even when explicitly matched
- newest-first ordering, ties deterministic, limit keeps THE NEWEST
- `src**/*.py` → ToolError; non-directory path → ToolError
- `../` escape AND symlink-out-of-sandbox escape both refused
- read_only=True (the whole point)

## Receipts

Offline suite green. No live receipt needed beyond ordinary REPL use —
this tool exercises no new wire surface, sandbox seam, or loop path;
it is deliberately the boring one.
