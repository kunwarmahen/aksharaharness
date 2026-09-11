---
name: new-tool
description: Add a new built-in tool to this harness -- schema, summary,
  run, registration, permission gating, tests and docs. Use when asked to
  add, write, or wire up a tool that the model can call.
allowed-tools: read_file, write_file, edit_file, grep, glob, bash
---

# Adding a tool to AksharaHarness

A tool is three things glued together (see `src/akshara/tools/base.py`):
a hand-written JSON Schema, a `summary()` the permission prompt shows a
human, and a `run()` that does the thing. Get all three right and the
loop handles the rest.

## Steps

1. **Read the neighbours first.** `src/akshara/tools/glob.py` is the
   smallest complete example; `src/akshara/tools/fs.py` shows a gated
   write. Match their comment density and voice -- this repo explains
   WHY in prose, not just what.

2. **Write the tool** in `src/akshara/tools/<name>.py`. Start from
   `template.py` in this skill's folder (read it with read_file). The
   rules that matter:
   - `parameters` is hand-written JSON Schema with a real `description`
     on every property, a `required` list, and
     `"additionalProperties": false`. Vague schemas produce vague calls.
   - `read_only = True` ONLY if the call cannot change anything outside
     the process and sends nothing over the network. It drives permission
     auto-approval, so getting it wrong removes a seatbelt silently.
   - Raise `ToolError` for anything the MODEL could plausibly fix (bad
     argument, missing file). The loop turns that into an error result
     the model reads and recovers from. Let real bugs crash.
   - Validate every argument with `require_str` / `require_int` -- args
     are whatever JSON the model produced, not what your schema asked for.

3. **Register it** in `default_registry()` in
   `src/akshara/tools/__init__.py`, add it to `__all__`, and update that
   module's docstring count ("Sixteen built-ins ...") -- it is prose that
   goes stale silently.

4. **Consider selection.** If the tool belongs to the autonomy floor
   (every task needs it), add it to `CORE_PINS` in
   `src/akshara/tools/selector.py`. Otherwise leave it retrievable.

5. **Test it** in `tests/test_tools.py` (or its own module if the surface
   is large). Cover: the happy path, a bad argument raising `ToolError`,
   the sandbox/permission story, and whatever the schema promises.

6. **Document it.** This repo ships docs WITH the feature, never after:
   - `README.md` -- the tool table and any usage section it touches
   - `TUTORIAL.md` (and regenerate `TUTORIAL.html`) if a beginner would
     meet it
   - `notes/NN-<topic>.md` if the tool introduces a concept worth a
     write-up

## Before you say you are done

Run `uv run pytest -q` and report the real result. If anything is still
failing or skipped, say so plainly rather than rounding up.
