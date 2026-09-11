# 30 · Skills — teaching the agent your procedures

Every project has knowledge that lives in someone's head. How *this*
team reviews a pull request. Which three files you always forget to
update when you add a feature. The five commands that cut a release, in
order, including the one that fails silently if you skip step two.

You can paste that into the chat every time. You can stuff it into the
system prompt and pay for it on every request, forever, including the
nine turns out of ten that have nothing to do with releases. Or you can
write it down once, in a file, and let the agent pick it up **only when
the work calls for it**.

That last one is a skill.

## What a skill is

A folder with a `SKILL.md` in it:

```
skills/new-tool/
├── SKILL.md        frontmatter + the instructions themselves
└── template.py     referenced BY the instructions; read on demand
```

```markdown
---
name: new-tool
description: Add a new built-in tool to this harness -- schema, summary,
  run, registration, permission gating, tests and docs. Use when asked to
  add, write, or wire up a tool that the model can call.
allowed-tools: read_file, write_file, edit_file, grep, glob, bash
---

# Adding a tool to AksharaHarness

1. Read the neighbours first. `src/akshara/tools/glob.py` is the
   smallest complete example...
```

That is the whole format. It is Markdown with a header, and the header
exists so the harness can know the skill's name and what it is for
without reading the rest.

## The one idea: three tiers of cost

The reason skills work is that they are **not** all in context at once.

| Tier | What the model gets | When | Cost |
|---|---|---|---|
| 1 | name + description | every turn, in the system prompt | ~25 tokens each |
| 2 | the full instructions | only when it calls `load_skill` | ~500–2000 tokens |
| 3 | bundled files | only if the instructions send it there | pay per use |

Ten skills cost you about 250 tokens a turn — a rounding error — and the
one that matters today arrives in full, on demand, exactly when the task
turns out to be about releases after all.

This is the same bet the tool-selection chapter makes
([notes/17](17-tool-selection.md)), one layer up: *the model does not
need to be holding a thing to know the thing exists.*

## Why the body arrives as a tool result

There were two ways to hand over tier 2. Swap the system prompt
mid-session and continue, or return the text as the result of a tool
call. We picked the tool call, for three reasons:

- **It costs no new machinery.** A tool result is already truncated,
  already displayed, already persisted in checkpoints, already
  compactable. The body is just text in the transcript.
- **It is visible.** `/history` shows exactly what the model was told
  and when. A prompt that mutates behind your back is a debugging
  nightmare in a harness whose whole point is that you can see the wire.
- **Local models handle it better.** "Call a tool, read the result" is
  the single most reliably trained behaviour in every model that
  supports tools at all. Rewriting the system prompt mid-conversation is
  not — and half this project's readers are running Ollama, not a
  frontier API.

Receipt, from `qwen3.8-64k` (a local 27B) on this repo. The prompt never
says the word *skill*:

```
$ akshara --provider ollama --model qwen3.8-64k:latest \
    --prompt "What would I have to change to add a count_lines tool
              to this project? Just give me the checklist."

skills: 2 loaded -- new-tool, notes-entry
→ load_skill()
  3. Register it — add to default_registry(), extend __all__, and bump
     the docstring count ("Sixteen built-ins…") in tools/__init__.py.
```

That "bump the docstring count" line appears nowhere in the model's
training data. It is in `skills/new-tool/SKILL.md`, because someone got
bitten by it once and wrote it down.

## Why the roster is frozen at session start

The obvious next idea is to rank the roster: BM25 the skills against the
conversation, show the model the best three, save the other tokens. We
do this for tools, and it works.

It is wrong here, and the reason is `--cache`. Prompt caching works on
the request **prefix**, which includes the system prompt
([notes/13](13-caching.md)). A roster that re-ranks itself each turn
rewrites that prefix each turn, which invalidates the cache on **every
single request** — to save 200 tokens of roster, you would re-pay for
thousands of tokens of cached prompt. Same rule env awareness lives
under ([notes/29](29-environment-awareness.md)): the block is composed
once and describes itself as possibly stale.

So the roster is static. Past 25 skills it degrades to names only and a
`list_skills` tool carries the descriptions — still static, still
cacheable, with the cost moved off the per-request path.

## Why one tool and not one tool per skill

Registering each skill as its own tool is tempting — it would let
selection rank them for free. It also walks straight off the tool cliff:
selection accuracy is flat to about ten tools, visibly worse by twenty,
and falls apart between thirty and fifty. Three modest MCP servers
already put you near the edge. Twelve skills would push you over it, and
every one of them would spend a JSON schema on every request.

`load_skill` is one tool whose argument happens to be a name. A project
with a dozen skills pays exactly one schema.

It is pinned in `CORE_PINS`, for a reason worth stating: the system
prompt *promises* the model it can load any skill on the roster. A
promise that BM25 can drop is a promise broken on precisely the vague
opening turns skills exist to handle.

## Layers: how the system prompt stopped being one string

Skills needed to append to the system prompt. So did env awareness, a
feature earlier. So, potentially, does whatever comes next.

The old shape had one author: capture the operator's `--system` as a
private base, recompose the whole string on every change. A second
author doing the same thing swallows the first one's block into *its*
base, and the next flip of either one silently erases the other. Not a
hypothetical — it is what would have happened the first time someone ran
`/env off` with skills loaded.

So `agent.system` got a seam before the second author existed
(`src/akshara/prompt.py`). It is now an ordered map of named layers:

```
base     the operator's --system, captured verbatim, never edited
env      the session fact sheet + policy      (notes/29)
skills   the roster                           (this note)
```

Each owner writes only its own layer. `attach_prompt()` is idempotent —
whoever wires up first freezes the base, everyone else joins the same
composition. `agent.system` stays a plain string, because the loop reads
it per request and checkpoints save it verbatim.

The checkpoint rule falls out of the same design: `/load` restores the
stored *composed* string, so every load path calls `recompose(agent)` to
rebuild it from the layers this session actually owns.

## What `allowed-tools` does, and what it does not

It is a note, not a gate.

The harness has no way to know whether the model is still "inside" a
skill three tool calls later. A restriction keyed to that would be
theatre — it would look like a security boundary in the docs and be a
suggestion in the code, which is the worst of both.

What it does instead is narrow expectations honestly. When you load a
skill that expects `browser_open` on a machine without the `[browse]`
extra, the result says so in the same breath as the instructions:

```
Tools this skill expects: bash, read_file, browser_open.
NOT available in this session: browser_open -- adapt the steps that need
them, or say plainly that the skill cannot be completed here.
```

The real boundary has not moved: every dangerous tool a skill names is
still permission-gated when it actually runs
([notes/06](06-cli.md#permissions)). A skill that genuinely needs
enforced isolation belongs in a sub-agent, whose tool scope *is* enforced
— at the catalog level, in `subagent.py` ([notes/08](08-sub-agents.md)).

`load_skill` itself is `read_only`. It reads a Markdown file you wrote
and put in your own repo; prompting for that would only train you to
mash `y` on the one tool that is definitionally safe.

## Where skills live

Nearest wins:

| Root | For |
|---|---|
| `$AKSHARA_SKILLS_PATH` / `--skills-dir` | explicit, wins everything |
| `.akshara/skills/` | private overrides — gitignored |
| `skills/` | the set your repo commits |
| `~/.akshara/skills/` | yours, everywhere |

`.akshara/` is machine state — a sqlite session store, job logs, a Chrome
profile — and it is gitignored. Skills are hand-written source meant to
travel with the repo, so the committed set lives in a plain top-level
`skills/`, and `.akshara/skills/` becomes the place to override one
locally without touching what your team sees.

## What a bad skill file costs you

Nothing but a line of output. Discovery never raises into a session: a
folder that fails to parse lands in a `broken` list with its path and the
reason, printed at startup and shown by `/skills`. The other nine skills
load fine.

Two validation rules are strict on purpose, and both are about the
description:

- **Missing is an error.** It is the only text the model ever sees every
  turn, and the only thing that decides whether the skill gets picked.
- **Thin is an error too.** `description: reviews code` is worse than no
  skill at all: it burns prompt tokens forever and still never fires,
  because there is nothing in it to match a real task against. Say what
  it does *and* when to use it.

A skill whose body you edit mid-session is re-read on the next
`load_skill`, so authoring is a loop: edit, ask again, watch what
changed. If your edit is half-saved and the file is broken, you get the
copy from startup rather than nothing — a syntax error should not take a
working skill away in the middle of a task.

## Using them

```bash
akshara                      # skills in ./skills are found automatically
/skills                      # what is on disk, what broke, what got loaded
/skills new-tool             # read one yourself — costs no model turn
/skills reload               # after you edit one
/new-tool add a count_lines tool     # run a skill directly
akshara --skills-dir ~/shared-skills # extra root, wins over the implicit ones
akshara --no-skills          # off entirely
```

In the web UI they get a section in the servers/skills/tools panel, with
a filled dot for each skill the model actually pulled this session —
which is the first question every skill author asks.
