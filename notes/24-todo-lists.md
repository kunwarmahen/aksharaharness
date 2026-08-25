# 24 — todo lists: live plan state, distinct from memory

*write_note already existed, so "just use memory" was the obvious
objection to building this. The interesting part is why a mission needs
a SECOND artifact with different physics.*

## Notes vs todos: two different half-lives

The scratchpad (notes/04's write_note/recall_notes) holds durable
FACTS: file layouts, decisions, dead ends. It is append-mostly,
topic-keyed, and meant to survive sessions.

A multi-step MISSION needs something else: live plan state. What's
done, what's in progress, what's next — rewritten wholesale as reality
reorders the work. Shoving that into notes would mean topics like
`plan-v3-final-actual`, and retrieval by substring search is exactly
the wrong shape for "what should I do next?" So: one store, one list,
one verb that replaces it whole.

Local models are the real beneficiaries. Frontier models hold a plan
in their head across dozens of tool calls; smaller local models drift
— they re-explore finished work or skip steps entirely. An explicit
checklist the model can re-read after every batch is cheap grounding:
the plan stops living in the context window's survival odds.

## Replace-whole-list, on purpose

todo_write takes the ENTIRE list every time. The alternative —
add/update/remove-by-index — drifts the moment two items swap or an
item splits; patch semantics against mutable positions are a classic
source of corrupted state. Sending the current plan each time is also
what makes the tool honest: the file always holds exactly what the
model last believed, contradictions included, instead of an accreted
history of edits nobody can reconcile.

Three statuses only: `pending / in_progress / done`. "Roughly one item
in_progress at a time" lives in the tool DESCRIPTION as a nudge, not
the code as a rule — a harness that hard-fails honest parallel plans
teaches the model to lie about status instead.

## Same skeleton as memory.py, deliberately

One JSON document under `.akshara/`, temp-file + atomic rename so a
crash mid-write never truncates the store, caps on items (50) and task
length (500 chars), corrupted-store errors that name the file. When a
pattern works, reuse it — consistency IS the tutorial here: two
stores, identical persistence discipline, different semantics.

## What the tests pin

- replace-whole-list (second write fully replaces the first)
- empty items legal — finishing or abandoning a mission is normal
- default status pending; unknown statuses refused with the valid set
- over-cap item counts and task lengths refused, model-readably
- result text renders `[x] [~] [ ]` checklist marks
- todo_read shows done/active/pending counts; empty store hints at
  todo_write; a store containing UNKNOWN statuses from a newer version
  drops them instead of bricking
- gating: read auto-approved, write prompts (it writes under .akshara)

## Receipts

Offline suite green.

Live receipt (Ollama `qwen3.8`, one-shot): given a three-step mission
the model called todo_write unprompted mid-turn and closed it out with
all three items `[x]` — the checklist rendered in its tool-result panel
exactly as todo_read would show it later.
