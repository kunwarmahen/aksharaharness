---
name: notes-entry
description: Write a notes/NN-topic.md design write-up in this repo's
  house voice. Use when asked to document a feature's design, add a notes
  entry, or explain why something was built the way it was.
allowed-tools: read_file, write_file, glob, grep, bash
---

# Writing a notes/ entry

`notes/` holds one write-up per topic: the REASONING behind a piece of
the harness, not its API. The README is the map; notes are the argument.

## Steps

1. **Find the next number**: `ls notes/` -- entries are `NN-topic.md`,
   zero-padded, in the order features landed. Never renumber existing ones.

2. **Read two neighbours** before writing a word. `notes/17-tool-selection.md`
   and `notes/29-environment-awareness.md` are the current high-water
   marks for voice.

3. **Structure**: open with the PROBLEM as the reader would hit it, in
   plain English -- what goes wrong without this feature. Then the design,
   then the constraints that ruled other designs out, then what was
   deliberately NOT built and why.

4. **Voice rules this repo holds to**:
   - Layman-first. A reader who has never written an agent should follow
     the first two paragraphs. Jargon gets defined the first time it appears.
   - Show receipts. Real numbers, real transcript snippets, real failures
     you saw -- not claims about what "tends to" happen.
   - Name the tradeoff you accepted. Every design has one; a write-up that
     does not name it reads like marketing.
   - Both roads stay first-class: whenever a behaviour differs between a
     cloud model and a local one (Ollama), say so explicitly.

5. **Link it up**: add the entry to the map in `README.md` and cross-link
   from any note it continues.

## Before you say you are done

Re-read it once as someone who has not seen the code. If any paragraph
only makes sense with the source open beside it, rewrite that paragraph.
