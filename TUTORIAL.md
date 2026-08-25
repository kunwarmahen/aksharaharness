# Build Your Own AI Agent — a from-scratch tutorial for absolute beginners

You have talked to a chatbot. This is about the thing *around* the
chatbot — the machinery that lets it read your files, run your
commands, and fix real bugs while you watch. Claude Code, Cursor and
friends are all built on it. It has a name: an **agent harness**.

Here is the good news: that machinery is not magic, and it is not
much code. By the end of this tutorial you will have typed every line
of a small but *real* one yourself — maybe 250 lines of plain Python —
and you will have used it to do actual work on your computer.

**Who this is for:** someone who has used a chatbot, can copy-paste
into a terminal, and has never built any of this. No machine-learning
knowledge needed. Every new word gets explained the first time.

**Two roads, one destination.** We assume roughly half of you will
rent a cloud brain through an API key — and the other half would
rather run a model on your own machine with
[Ollama](https://ollama.com): no key, no bill, works offline. Both
roads are first-class here. Wherever they diverge you'll meet an
**Ollama variation** box with the exact code to type instead, and
[Appendix B](#appendix-b--the-complete-agentpy-ollama-edition) has the
whole local-road file assembled. The harness itself — hands, memory,
seatbelt — is byte-for-byte the same on both roads; only the wire
words change.

**How to read it:** slowly, with a terminal open. Type each snippet
into a file called `agent.py` as you reach it — the main snippets if
you picked the cloud road, the boxed variations if you picked the
local one. The snippets build on each other in order. If you just
want a finished file, both roads have one:
[Appendix A](#appendix-a--the-complete-agentpy) (Anthropic) and
[Appendix B](#appendix-b--the-complete-agentpy-ollama-edition)
(Ollama) — but typing it yourself is where the understanding comes
from.

This tutorial lives inside a bigger project ([AksharaHarness](README.md))
that builds the same ideas properly — three wire formats, retries,
memory, sandboxing, tests. At the end we'll map what you built onto
that code so you know where to go next.

---

## 0 · What exactly are we building?

A chatbot can only *talk*. You type, it types back. Ask it "what's in
my README?" and it can only guess, because it cannot see your files.

An **agent** is a chatbot that can also **act**:

```
      a chatbot                        an agent

  ┌───────────────┐              ┌────────────────────────┐
  │   you  ⇅  LLM │              │   you  ⇅  HARNESS  ⇅   │
  │               │              │        ┌───────┐  LLM  │
  │  talk only    │              │     files    shell     │
  │               │              │     search   editor    │
  └───────────────┘              │     + permission gate  │
                                 └────────────────────────┘
                                  acts on your computer,
                                  asks before doing anything risky
```

Two words in that picture:

* **LLM** ("large language model") — the text-predicting brain behind
  chatbots: ChatGPT's engine, Claude, Gemini. Usually it lives on a
  company's computers and we rent access over the internet through an
  **API** (a web address our programs can send messages to) — but it
  can just as well live on YOUR computer, reached at `localhost`
  instead of a company's address. Same brain-shaped hole in the
  middle, either way; this tutorial works with both.
* **Harness** — everything WE write around it: the hands (tools), the
  memory (a transcript), the seatbelt (permissions).

The harness is the difference between "a thing that describes what
*could* be done" and "a thing that does it — carefully."

## 1 · Setting up (five minutes)

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/) (a fast
Python tool installer — follow the one curl/winget command on its site).
Then — same for both roads:

```bash
mkdir my-agent && cd my-agent      # a fresh folder for our creation
uv init --bare                     # makes this folder a Python project
uv add httpx                       # the ONE library we need: talks to websites
echo .env >> .gitignore            # secrets must never be committed
```

Every agent needs two things before it can talk: a **web address** to
send mail to, and whatever proof-of-identity that address demands.
Yours will come from one of these roads:

**Road A — cloud brain (Anthropic).** Create a file named `.env` in
that folder containing your API key, from the Anthropic Console
(console.anthropic.com):

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

**Road B — local brain (Ollama).** No key, no bill, works offline —
the trade is that local models are smaller, so expect a less clever
conversation partner. Install Ollama and download a model (~5 GB,
once; pick a smaller tag like `llama3.2` if your machine has under
16 GB of RAM):

```bash
curl -fsSL https://ollama.com/install.sh | sh   # mac/linux; windows:
                                                # installer at ollama.com
ollama pull qwen3.8                # downloads the brain onto your disk
ollama list                        # any tag printed here works as MODEL
```

The server starts with the install and waits at `localhost:11434`.
Nothing to sign up for; nothing leaves your machine.

Both roads then share a tiny loader at the top of `agent.py` that
reads `.env` into the environment (real environment variables still
win — standard behavior). Road B readers: an empty or missing `.env`
is fine, the loader just does nothing:

```python
import json
import os

def load_env(path=".env"):
    """Fill os.environ from KEY=value lines; existing env vars win."""
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

load_env()
```

…then close the setup with ONE road-specific line:

```python
# Road A (Anthropic):
API_KEY = os.environ["ANTHROPIC_API_KEY"]       # crashes fast if .env is missing

# Road B (Ollama):
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8")   # or any tag from ollama list
```

Check it works:

```bash
uv run python -c "import agent"    # should print nothing = no errors
```

> **Using OpenAI instead?** Everything in this tutorial works there
> too — only the words change (different URL, different field names,
> same ideas). The parent project keeps a side-by-side translation
> table it calls [the wire cheat-sheet](README.md#the-wire-cheat-sheet).
> And if you picked Road B: from here on, watch for the **Ollama
> variation** boxes — same function names, same structure, different
> wire words.

## 2 · The one idea everything else rests on

Here is the whole secret, smaller than you'd expect:

> **The model is a text-in, text-out function. Nothing more.**

It has no memory of yesterday. It cannot see your files. It cannot
run anything. It cannot even take its own advice. Every time we call
it, we mail it the transcript of the conversation SO FAR, and it mails
back text.

So how does Claude Code edit files? Because *we wrote the hands*. The
model can only **ask**: "please run this command." Our program decides
whether to allow it, runs it if allowed, pastes the result into the
transcript, and mails everything back. From the outside it looks like
intelligence acting in the world; up close it is a very persuasive pen
pal with a very diligent postal service.

That postal service is what you're building. Keep the picture in your
head; every step below adds one organ to the same animal.

**A word before the wire, for Road B readers.** Everything from here
on is written on the Anthropic dialect — it's the easiest to read.
The *ideas* never change across roads; only their names do. This is
the entire dictionary you need to translate as you go (each row gets
a variation box at the step where it matters):

| The idea (identical everywhere) | Anthropic calls it | Ollama calls it |
|---|---|---|
| the address you mail | `https://api.anthropic.com/v1/messages` | `http://localhost:11434/v1/chat/completions` |
| proving who you are | `x-api-key` header + version header | nothing — it's your own machine |
| standing orders (`system`) | a top-level `system` field | a `role:"system"` message in the transcript |
| the model asking to act | a content block, `type: "tool_use"` | a `tool_calls` list on the reply message |
| reporting what happened | `tool_result` blocks inside the next user letter | its own message, `role: "tool"` |
| "I'm done" vs "I want to act" | `stop_reason: end_turn` / `tool_use` | `finish_reason: stop` / `tool_calls` |

One quirk to know early: Ollama delivers tool-call **arguments as a
JSON string** (`"{\"path\": \"README\"}"`) instead of a ready-made
object, so your code parses them with `json.loads()` once per call.
That's the whole surprise budget.

## 3 · Step 1: make it talk

Add to `agent.py`:

```python
import httpx

URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5"

history = []   # THE TRANSCRIPT: the model's only memory. A list of
               # {"role": ..., "content": ...} letters, in order.


def ask() -> dict:
    """Mail the whole transcript to the model; return its reply."""
    body = {
        "model": MODEL,
        "max_tokens": 4000,          # cap on reply length (required here)
        "messages": history,
        "system": "You are a careful assistant working on the user's "
                  "computer. Be concise.",
    }
    response = httpx.post(URL, json=body, timeout=120, headers={
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    })
    response.raise_for_status()      # turn HTTP errors into exceptions
    return response.json()


def chat(user_text: str) -> None:
    history.append({"role": "user", "content": user_text})
    reply = ask()
    print(reply["content"][0]["text"])
    # remember both sides of the exchange:
    history.append({"role": "assistant",
                    "content": reply["content"][0]["text"]})


if __name__ == "__main__":
    chat("Why is the sky blue? One sentence.")
```

> **🔧 Ollama variation — Step 1.** Same two functions, local wire
> words — type these instead of the listing above (`__main__` and all).
> The system prompt rides *inside* the transcript as a `role:"system"`
> message (there's no top-level `system` field here), `max_tokens` is
> optional so we drop it, there are no auth headers because it's your
> own machine, and the reply text lives at
> `choices[0].message.content`:
>
> ```python
> URL = "http://localhost:11434/v1/chat/completions"
>
> history = []   # THE TRANSCRIPT: same idea, plain-string letters.
>
>
> def ask() -> dict:
>     """Mail the whole transcript to the model; return its reply."""
>     body = {
>         "model": MODEL,
>         "messages": [{"role": "system",
>                       "content": "You are a careful assistant working "
>                                  "on the user's computer. Be concise."},
>                      *history],
>         # no max_tokens needed; no key, no auth headers -- localhost
>     }
>     response = httpx.post(URL, json=body, timeout=120)
>     response.raise_for_status()      # turn HTTP errors into exceptions
>     return response.json()
>
>
> def chat(user_text: str) -> None:
>     history.append({"role": "user", "content": user_text})
>     reply = ask()
>     text = reply["choices"][0]["message"]["content"]
>     print(text)
>     # remember both sides of the exchange:
>     history.append({"role": "assistant", "content": text})
>
>
> if __name__ == "__main__":
>     chat("Why is the sky blue? One sentence.")   # same smoke test
> ```
>
> First call may feel slower than a cloud API — the model is loading
> off your disk. It speeds up once warm.

Run it:

```bash
uv run python agent.py
```

Congratulations — you just called a real AI directly. Look at what
flew over the wire: our letter (`role: user`, some text) plus settings
(model, length cap, an optional `system` instruction = standing orders),
and back came JSON with `content` (the words) and `stop_reason`
(`end_turn` means "I'm done").

The `history` list deserves a slow nod. **The transcript IS the
model's memory.** Ask a follow-up without re-sending earlier turns and
the model has no idea they happened. That's why `chat()` appends both
sides before returning.

## 4 · Step 2: give it hands (tools)

Time for the superpower. We will let the model *request* actions.

First, describe each hand twice: once as a spec the MODEL reads (what
it's for, what arguments it takes — this little format is called JSON
Schema), once as a real Python function OUR code runs:

```python
import subprocess

TOOL_SPECS = [
    {
        "name": "read_file",
        "description": "Return the full contents of a file in the "
                       "current folder.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return combined "
                       "stdout+stderr. Working directory does not "
                       "persist between calls.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def read_file(path):
    with open(path) as f:
        return f.read()


def run_command(command):
    proc = subprocess.run(command, shell=True, capture_output=True,
                          text=True, timeout=30)
    out = (proc.stdout + proc.stderr).strip()
    return out or "(no output)"


TOOLS = {"read_file": read_file, "run_command": run_command}
```

> **🔧 Ollama variation — Step 2a (the specs).** Same names, same
> schemas, but each spec is wrapped in a `function` envelope and the
> schema key is `parameters` here (it was `input_schema`). Type this
> instead of the `TOOL_SPECS` above — the `read_file` / `run_command`
> functions and the `TOOLS` dict that follow it are identical on both
> roads, so take them straight from the listing:
>
> ```python
> TOOL_SPECS = [
>     {"type": "function",
>      "function": {
>          "name": "read_file",
>          "description": "Return the full contents of a file in the "
>                         "current folder.",
>          "parameters": {
>              "type": "object",
>              "properties": {"path": {"type": "string"}},
>              "required": ["path"],
>          }}},
>     {"type": "function",
>      "function": {
>          "name": "run_command",
>          "description": "Run a shell command and return combined "
>                         "stdout+stderr. Working directory does not "
>                         "persist between calls.",
>          "parameters": {
>              "type": "object",
>              "properties": {"command": {"type": "string"}},
>              "required": ["command"],
>          }}},
> ]
> ```

Second, tell the model these exist — one line added to the request body
in `ask()` (both roads, same line):

```python
        "tools": TOOL_SPECS,
```

Now change `chat()` into `turn()`, because replies may now contain
*requests* instead of answers. This next function is the heart of
every agent ever built, so go slow:

```python
def tool_result(call_id, output, is_error=False):
    """Our report back, shaped the way the API expects."""
    return {"type": "tool_result", "tool_use_id": call_id,
            "content": output[:20000],       # keep letters deliverable-sized
            "is_error": is_error}


def turn(user_text: str) -> None:
    history.append({"role": "user", "content": user_text})

    for _bounce in range(25):                # safety valve: never spin forever
        reply = ask()
        # mail the WHOLE reply back later, not just text — some replies
        # carry sealed attachments that must be returned untouched
        history.append({"role": "assistant", "content": reply["content"]})

        if reply["stop_reason"] != "tool_use":
            for block in reply["content"]:   # a plain answer: print it
                if block["type"] == "text":
                    print(block["text"])
            return

        results = []
        for block in reply["content"]:
            if block["type"] != "tool_use":
                continue                     # skip any text/thinking pieces
            output = TOOLS[block["name"]](**block["input"])
            print(f"  [{block['name']}({block['input']})]")
            results.append(tool_result(block["id"], output))

        # paste the results in AS IF THE USER SAID THEM — that is not a
        # shortcut, it is the protocol: results ride inside user letters
        history.append({"role": "user", "content": results})


if __name__ == "__main__":
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break                            # ctrl-d / ctrl-c at the prompt
        if text:
            turn(text)
```

> **🔧 Ollama variation — Step 2b (results & loop).** Two pieces change.
> Results go back as their own messages, so `tool_result()` builds a
> different envelope — there's no `is_error` field on this wire; errors
> ride as plain text with an `ERROR:` prefix, which works just as well
> because to the model it's all just words. And `turn()` reads requests
> off `message.tool_calls`, parsing each call's arguments out of their
> JSON-string form. Type these instead of the two functions above —
> the `__main__` REPL stays exactly as it is in the listing:
>
> ```python
> def tool_result(call_id, output, is_error=False):
>     """Our report back, shaped the way the API expects."""
>     prefix = "ERROR: " if is_error else ""
>     return {"role": "tool", "tool_call_id": call_id,
>             "content": prefix + str(output)[:20000]}
>
>
> def turn(user_text: str) -> None:
>     history.append({"role": "user", "content": user_text})
>
>     for _bounce in range(25):            # safety valve: never spin forever
>         reply = ask()
>         message = reply["choices"][0]["message"]
>         calls = message.get("tool_calls") or []
>         # mail the WHOLE reply back later, requests included:
>         entry = {"role": "assistant",
>                  "content": message.get("content") or ""}
>         if calls:
>             entry["tool_calls"] = calls  # sealed attachments ride along
>         history.append(entry)
>
>         if not calls:                    # a plain answer: print it
>             print(message.get("content") or "")
>             return
>
>         results = []
>         for call in calls:               # each request: id + function
>             name = call["function"]["name"]
>             args = json.loads(call["function"]["arguments"] or "{}")
>             # ^ remember the quirk: arguments arrive as a JSON STRING
>             output = TOOLS[name](**args)
>             print(f"  [{name}({args})]")
>             results.append(tool_result(call["id"], output))
>
>         history.extend(results)          # one role:"tool" message each
> ```

Run it and try:

```
> What does the pyproject.toml in this folder say?
```

Watch what happens: the model doesn't answer — it asks to run
`read_file`. Your code runs it, pastes the result into the transcript,
mails everything back, and the model answers with real knowledge of a
real file. **That round trip — ask → act → record → ask again — is the
agent loop.** Everything else in every agent product is refinement.

Three details worth savoring:

1. **The model never touches your disk.** It emits a request as data;
   your code decides whether to honor it. Right now it's unconditional
   (`TOOLS[block["name"]](...)`) — fixing that is Step 4.
2. **Tool results travel in dedicated result slots**, never in the
   model's own voice. Which slot differs by road: Anthropic tucks
   them into the next *user* letter as `tool_result` blocks; Ollama
   wants one separate `role:"tool"` message per result (your Step 2b
   variation). The rule either way: every request gets exactly one
   reply, in order.
3. **Every bounce re-mails the ENTIRE transcript.** Ten bounces =
   ten copies of everything. (Real products soften this with caching —
   see [notes/13](notes/13-caching.md).)

## 5 · Level up: watch it type (streaming)

Right now answers appear all at once after a pause. Real tools feel
alive because the answer *streams* — the server sends it as dozens of
tiny postcards (a format called SSE: server-sent events) instead of
one parcel:

```
data: {"type":"content_block_delta","delta":{"text":"The "}}
data: {"type":"content_block_delta","delta":{"text":"answer"}}
...
```

To stream, set `"stream": True` in the body and swap `httpx.post` for a
loop over lines. Text deltas print as they arrive. But here's the
sneaky part: when the model requests a tool, the ARGUMENTS also arrive
as fragments — `{"pa` … `th": "REA` … `DME.md"}` — sometimes split
mid-word across network packets. The trick is to accumulate fragments
keyed by their position number (`index`) and only try to understand
the JSON once the closing event arrives. Parse early and you crash on
half a sentence.

```python
def ask_streaming() -> dict:
    """Same letter to the model as ask(), streamed: print text live,
    assemble tool calls."""
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "messages": history,
        "system": SYSTEM,
        "tools": TOOL_SPECS,
        "stream": True,              # ← the only new field
    }
    parts = []                 # assembled content blocks, in order
    texts = {}                 # index -> list of text fragments
    calls = {}                 # index -> {"id","name","json-fragments"}
    with httpx.stream("POST", URL, json=body, timeout=300, headers={
        "x-api-key": API_KEY, "anthropic-version": "2023-06-01",
    }) as response:
        event = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
                if event == "content_block_start":
                    b = data["content_block"]; i = data["index"]
                    (calls if b["type"] == "tool_use" else texts)[i] = \
                        {"id": b.get("id"), "name": b.get("name"), "pieces": []}
                elif event == "content_block_delta":
                    i = data["index"]; d = data["delta"]
                    if d["type"] == "text_delta":
                        texts.setdefault(i, {"pieces": []})
                        print(d["text"], end="")          # ← the live part
                        texts[i]["pieces"].append(d["text"])
                    elif d["type"] == "input_json_delta":
                        calls[i]["pieces"].append(d["partial_json"])
                elif event == "message_stop":
                    break
    for i, t in sorted(texts.items()):
        parts.append({"type": "text", "text": "".join(t["pieces"])})
    for i, c in sorted(calls.items()):
        args = json.loads("".join(c["pieces"]) or "{}")   # parse ONCE, at end
        parts.append({"type": "tool_use", "id": c["id"],
                      "name": c["name"], "input": args})
    return {"content": parts,
            "stop_reason": "tool_use" if calls else "end_turn"}
```

Use it in place of `ask()` inside `turn()` and suddenly your agent
types itself. You have also just met the hardest bug farm in this
field — fragment accumulation — which the main project solves with
byte-exact recorded fixtures and tests ([notes/03](notes/03-sse-and-collect.md)).

> **🔧 Ollama variation — Step 3½ (streaming).** Same SSE plumbing,
> anonymous chunks instead of named events, terminated by a literal
> `data: [DONE]` line:
>
> ```
> data: {"choices":[{"delta":{"content":"The "}}]}
> data: {"choices":[{"delta":{"content":"answer"}}]}
> data: [DONE]
> ```
>
> Text arrives at `choices[0].delta.content`; when the model requests
> a tool, its arguments stream as string fragments inside
> `choices[0].delta.tool_calls[i].function.arguments` — different
> hiding place, identical discipline: accumulate fragments keyed by
> index, parse exactly once, only after `[DONE]`. The bug farm is the
> same farm. Streaming is left as an exercise on both roads; your
> non-streaming `turn()` works fine without it.

## 6 · Step 3: the seatbelt (permissions)

`run_command` can do anything. `rm -rf something-wrong` should not run
because a very confident text predictor suggested it. The fix is one
honest question asked by YOUR code, showing exactly what will run:

```python
READ_ONLY = {"read_file"}

def approved(name: str, args: dict) -> bool:
    if name in READ_ONLY:
        return True                      # looking cannot hurt
    answer = input(f"allow {name}({args})? [y/N] ")
    return answer.strip().lower() == "y"
```

And wire it into the loop — replacing the unconditional execution:

```python
            for block in reply["content"]:
                if block["type"] != "tool_use":
                    continue
                if not approved(block["name"], block["input"]):
                    # a denial is NOT a crash: it becomes a short note the
                    # model reads and adjusts course accordingly
                    results.append(tool_result(
                        block["id"], "the human said no", is_error=True))
                    continue
                output = TOOLS[block["name"]](**block["input"])
                print(f"  [{block['name']}({block['input']})]")
                results.append(tool_result(block["id"], output))
```

Try `> delete a file` and deny it: the model shrugs and tries a
different way. That is the whole security model of every serious agent
tool — **dangerous actions require a human yes; refusals become
information, not exceptions.**

(One honest caveat: `read_file` is only safe because OUR code opens
paths relative to THIS folder and nothing else. The main project
sandboxes its shell too — bubblewrap, no network, ephemeral filesystem —
because y/n alone is not containment. See [notes/16](notes/16-sandboxing.md)
when you're ready.)

## 7 · Step 4: bad news becomes data

Tools misbehave: typos, missing files, genuine bugs. A naive harness
crashes mid-turn. Ours needs one rule:

> **No tool failure ever throws an exception up into the loop. It
> becomes a note.**

Wrap the execution:

```python
                try:
                    output = TOOLS[block["name"]](**block["input"])
                except Exception as exc:
                    output = f"{type(exc).__name__}: {exc}"
                    results.append(tool_result(block["id"], output,
                                               is_error=True))
                    continue
                results.append(tool_result(block["id"], output))
```

The loop physically cannot be crashed by a broken tool now. Worst case
is one wasted bounce in which the model learns "that didn't work" and
tries again differently. Watch it happen: `> read the file nope.txt` —
the model receives `FileNotFoundError`, apologizes, moves on.

Only two things are allowed to stop a turn: you pressing Ctrl-C, or
the provider being unreachable (no internet, a bad key, or on the
local road simply Ollama not running — no point continuing when
there's nobody to talk to).

> **🔧 Ollama variation — Steps 4 & 5 (seatbelt, errors-as-data).**
> None. This is the punchline of the whole architecture: the seatbelt
> wraps *your* code that executes tools, so it cannot tell which wire
> is underneath. Type these two steps exactly as written, applying
> them around the tool loop from your Step 2 variation.

## 8 · Step 5: the promise book (why your agent won't randomly die)

Each tool request carries an ID like a claim ticket: `"id":
"toolu_7"...`. The protocol's one hard rule: **every ticket must be
redeemed.** Mail the transcript containing an unanswered ticket and the
provider rejects the entire request with a cryptic error. This single
mistake kills most home-made agents — and it bites exactly when things
go sideways: Ctrl-C pressed mid-turn, the 25-bounce cap hit.

So keep a literal promise book: on every exit path — including Ctrl-C
mid-turn — scan for unredeemed tickets and write redemption slips.
Real results for finished work, apology notes for the rest, all pasted
in as one user letter so the transcript stays mailable:

```python
def turn(user_text: str) -> None:
    history.append({"role": "user", "content": user_text})

    for _bounce in range(25):
        try:
            reply = ask()
        except KeyboardInterrupt:
            print("\n(cancelled)")       # between letters: nothing to redeem
            return

        history.append({"role": "assistant", "content": reply["content"]})

        if reply["stop_reason"] != "tool_use":
            for block in reply["content"]:
                if block["type"] == "text":
                    print(block["text"])
            return

        results = []
        try:
            for block in reply["content"]:
                if block["type"] != "tool_use":
                    continue
                name, args = block["name"], block["input"]
                if not approved(name, args):
                    results.append(tool_result(block["id"],
                                               "the human said no",
                                               is_error=True))
                    continue
                output = TOOLS[name](**args)
                print(f"  [{name}({args})]")
                results.append(tool_result(block["id"], output))
        except KeyboardInterrupt:
            # pressed MID-turn: finish the batch's bookkeeping by writing
            # apology slips for every ticket we never got to — the
            # transcript must leave this function whole
            answered = {r["tool_use_id"] for r in results}
            slips = [tool_result(b["id"], "(interrupted by the human)",
                                 is_error=True)
                     for b in reply["content"]
                     if b["type"] == "tool_use" and b["id"] not in answered]
            history.append({"role": "user", "content": results + slips})
            print("\n(cancelled — session intact)")
            return

        history.append({"role": "user", "content": results})
```

> **🔧 Ollama variation — Step 6 (the promise book).** Same promise,
> different slip paper: redemptions are `role:"tool"` messages keyed
> by `tool_call_id`, and the hard rule is identical — mail a transcript
> where an assistant message promised `tool_calls` and any id went
> unredeemed, and Ollama rejects the whole letter. So the apology
> slips exist here too:
>
> ```python
> def turn(user_text: str) -> None:
>     history.append({"role": "user", "content": user_text})
>
>     for _bounce in range(25):
>         try:
>             reply = ask()
>         except KeyboardInterrupt:
>             print("\n(cancelled)")   # between letters: nothing to redeem
>             return
>
>         message = reply["choices"][0]["message"]
>         calls = message.get("tool_calls") or []
>         entry = {"role": "assistant", "content": message.get("content") or ""}
>         if calls:
>             entry["tool_calls"] = calls
>         history.append(entry)
>
>         if not calls:
>             print(message.get("content") or "")
>             return
>
>         results = []
>         try:
>             for call in calls:
>                 name = call["function"]["name"]
>                 args = json.loads(call["function"]["arguments"] or "{}")
>                 if not approved(name, args):
>                     results.append(tool_result(call["id"],
>                                                "the human said no",
>                                                is_error=True))
>                     continue
>                 try:
>                     output = TOOLS[name](**args)      # errors-as-data
>                 except Exception as exc:
>                     output = f"{type(exc).__name__}: {exc}"
>                     results.append(tool_result(call["id"], output,
>                                                is_error=True))
>                 else:
>                     results.append(tool_result(call["id"], output))
>         except KeyboardInterrupt:
>             # pressed MID-turn: write apology slips for every ticket we
>             # never got to -- each a role:"tool" message redeeming its
>             # own tool_call_id, so the transcript leaves this function
>             # whole and mailable
>             answered = {r["tool_call_id"] for r in results}
>             slips = [tool_result(c["id"], "(interrupted by the human)",
>                                  is_error=True)
>                      for c in calls if c["id"] not in answered]
>             history.extend(results + slips)
>             print("\n(cancelled — session intact)")
>             return
>
>         history.extend(results)
> ```

After this change, hammering Ctrl-C mid-turn costs nothing: finished
tools keep their real results, unfinished ones get honest "(interrupted)"
notes, and your next prompt just works. Try it on a slow command and
watch the difference from before.

In the main project this cleanup runs on EVERY exit path including
iteration caps and cancellations, and it is regression-tested — the
authors call it [the resumable-history invariant](notes/05-agent-loop.md)
and it is the single most valuable idea in the whole codebase.

## 9 · Congratulations. Now use the thing.

Your `agent.py` is a working agent. Some things worth doing:

```
> summarize what this folder contains
> find every TODO comment in *.py files and list them
> make a file hello.py that prints hello world, then run it to prove it
> there is a typo in pyproject.toml somewhere — find it and propose a fix
```

The last two are the graduation test: multi-step work where each step
depends on the last — impossible for a chatbot, routine for anything
with hands, memory and a loop. Approve the writes, deny one on purpose,
watch it adapt.

What you built maps almost one-to-one onto the big tools:

| Your mini-harness | What the real ones add |
|---|---|
| `history` list | compaction: summarizing old context when it outgrows the window |
| `TOOLS` dict + specs | dozens of tools, loaded dynamically to fit the model's attention |
| `approved()` y/n | the grown-up gate asks y/n/**e** — `e` edits the command before you approve it ([notes/20](notes/20-approve-with-edits.md)) — plus sandboxed execution, audit hooks; and when a task has earned your trust, `/yolo` switches the questions off (and back on) without leaving the conversation |
| promise-book cleanup | sessions that survive restarts (SQLite checkpoints) |
| one wire dialect (yours: cloud or local) | several providers behind one interface, retries and failover |
| printed dollars? none yet | cost accounting: `$` figures per turn and per session |

## 10 · Where to go from here

The repository this tutorial lives in is the same animal, grown up.
Each topic has a short, plain-written note with the war stories:

| Topic | Start here |
|---|---|
| The guided tour (everything in pictures) | [notes/00-guided-tour.md](notes/00-guided-tour.md) |
| Anatomy of a request | [notes/01](notes/01-anatomy-of-a-request.md) |
| Two wire dialects side by side | [notes/02](notes/02-wire-formats.md) |
| Streaming & fragment assembly | [notes/03](notes/03-sse-and-collect.md) |
| Tools done properly (sandboxing, diffs) | [notes/04](notes/04-tools.md) · [notes/16](notes/16-sandboxing.md) |
| The loop & the promise book | [notes/05](notes/05-agent-loop.md) |
| The CLI experience | [notes/06](notes/06-cli.md) |
| Reliability: retries, failover, compaction | [notes/07](notes/07-reliability-and-scale.md) |
| Sub-agents (agents hiring agents) | [notes/08](notes/08-sub-agents.md) |
| MCP (universal plug for tools) | [notes/09](notes/09-mcp.md) |
| Grading agents automatically (evals) | [notes/10](notes/10-evals.md) |
| Many conversations at once (async) | [notes/11](notes/11-async.md) |
| Letting it BUILD software | [notes/12](notes/12-builder.md) |
| Cutting the bill (prompt caching) | [notes/13](notes/13-caching.md) |
| Watching without vetoing (hooks) | [notes/14](notes/14-hooks.md) |
| Images, cost accounting | [notes/15](notes/15-images.md) · [notes/21](notes/21-cost-accounting.md) |
| A third provider dialect (Responses API) | [notes/19](notes/19-responses-api.md) |
| Approve-with-edits (the y/n/e gate) | [notes/20](notes/20-approve-with-edits.md) |
| A browser window on your agent — and a tool for asking *you* things | [notes/22](notes/22-web-ui.md) |
| Tools that make it self-reliant: glob, todos, web fetch, background jobs, seeing pictures | [notes/23](notes/23-glob.md) · [notes/24](notes/24-todo-lists.md) · [notes/25](notes/25-web-fetch.md) · [notes/26](notes/26-background-bash.md) · [notes/27](notes/27-read-image.md) |
| Operating web apps (JS-rendered pages) on a headless browser: browser_open/click/fill/close | [notes/28](notes/28-browser-tools.md) |

And the best exercise known to man: point your finished `agent.py` at
its own source code and ask it how the loop works. It can read itself.

---

## Appendix A · the complete `agent.py`

Everything from steps 1–4 and 6–8 assembled — every piece exactly as
presented above, nothing held back (streaming left as the exercise it
was presented as: drop `ask_streaming()` in wherever `ask()` appears).
Road B readers: your assembled file is
[Appendix B](#appendix-b--the-complete-agentpy-ollama-edition).

```python
"""A minimal-but-real agent harness — ~200 lines, zero frameworks."""
import json
import os
import subprocess

import httpx


# ---- setup -----------------------------------------------------------------

def load_env(path=".env"):
    """Fill os.environ from KEY=value lines; existing env vars win."""
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()
API_KEY = os.environ["ANTHROPIC_API_KEY"]
URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5"

HEADERS = {"x-api-key": API_KEY, "anthropic-version": "2023-06-01"}
SYSTEM = ("You are a careful assistant working on the user's computer. "
          "Be concise.")
history = []          # the transcript: the model's ONLY memory


# ---- hands -----------------------------------------------------------------

TOOL_SPECS = [
    {
        "name": "read_file",
        "description": "Return the full contents of a file in the "
                       "current folder.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return combined "
                       "stdout+stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def read_file(path):
    with open(path) as f:
        return f.read()


def run_command(command):
    proc = subprocess.run(command, shell=True, capture_output=True,
                          text=True, timeout=30)
    return (proc.stdout + proc.stderr).strip() or "(no output)"


TOOLS = {"read_file": read_file, "run_command": run_command}


# ---- seatbelt ----------------------------------------------------------------

READ_ONLY = {"read_file"}


def approved(name, args):
    if name in READ_ONLY:
        return True
    return input(f"allow {name}({args})? [y/N] ").strip().lower() == "y"


# ---- the postal service -------------------------------------------------------

def ask() -> dict:
    body = {"model": MODEL, "max_tokens": 4000, "messages": history,
            "system": SYSTEM, "tools": TOOL_SPECS}
    response = httpx.post(URL, json=body, timeout=120, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def tool_result(call_id, output, is_error=False):
    return {"type": "tool_result", "tool_use_id": call_id,
            "content": str(output)[:20000], "is_error": is_error}


# ---- the loop (with its promise book) ------------------------------------------

def turn(user_text):
    history.append({"role": "user", "content": user_text})
    for _bounce in range(25):
        try:
            reply = ask()
        except KeyboardInterrupt:
            print("\n(cancelled)")     # between letters: nothing to redeem
            return
        except httpx.HTTPError as exc:
            print(f"(provider unreachable: {exc})")
            return
        history.append({"role": "assistant", "content": reply["content"]})

        if reply["stop_reason"] != "tool_use":
            for block in reply["content"]:
                if block["type"] == "text":
                    print(block["text"])
            return

        results = []
        try:
            for block in reply["content"]:
                if block["type"] != "tool_use":
                    continue
                name, args = block["name"], block["input"]
                if not approved(name, args):          # denial-as-data
                    results.append(tool_result(block["id"],
                                               "the human said no",
                                               is_error=True))
                    continue
                try:
                    output = TOOLS[name](**args)      # errors-as-data
                except Exception as exc:
                    output = f"{type(exc).__name__}: {exc}"
                    results.append(tool_result(block["id"], output,
                                               is_error=True))
                else:
                    results.append(tool_result(block["id"], output))
        except KeyboardInterrupt:
            # pressed MID-turn: write apology slips for every ticket we
            # never got to -- the transcript must leave this function whole
            answered = {r["tool_use_id"] for r in results}
            slips = [tool_result(b["id"], "(interrupted by the human)",
                                 is_error=True)
                     for b in reply["content"]
                     if b["type"] == "tool_use" and b["id"] not in answered]
            history.append({"role": "user", "content": results + slips})
            print("\n(cancelled — session intact)")
            return

        history.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("agent ready — ctrl-c cancels a turn, ctrl-d quits")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text:
            turn(text)
```

---

## Appendix B · the complete `agent.py` (Ollama edition)

The same harness on the local road: every piece from the variation
boxes assembled into one file, structure-identical to Appendix A so
you can diff them side by side and see that only the wire words
differ (streaming left as the same exercise):

```python
"""A minimal-but-real agent harness — local edition (Ollama).
~200 lines, zero frameworks. Same animal as the cloud version."""
import json
import os
import subprocess

import httpx


# ---- setup -----------------------------------------------------------------

def load_env(path=".env"):
    """Fill os.environ from KEY=value lines; existing env vars win."""
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()                       # an empty or missing .env is fine here
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8")  # any tag from ollama list
URL = "http://localhost:11434/v1/chat/completions"

SYSTEM = ("You are a careful assistant working on the user's computer. "
          "Be concise.")
history = []          # the transcript: the model's ONLY memory


# ---- hands -----------------------------------------------------------------

TOOL_SPECS = [
    {"type": "function",
     "function": {
         "name": "read_file",
         "description": "Return the full contents of a file in the "
                        "current folder.",
         "parameters": {
             "type": "object",
             "properties": {"path": {"type": "string"}},
             "required": ["path"],
         }}},
    {"type": "function",
     "function": {
         "name": "run_command",
         "description": "Run a shell command and return combined "
                        "stdout+stderr.",
         "parameters": {
             "type": "object",
             "properties": {"command": {"type": "string"}},
             "required": ["command"],
         }}},
]


def read_file(path):
    with open(path) as f:
        return f.read()


def run_command(command):
    proc = subprocess.run(command, shell=True, capture_output=True,
                          text=True, timeout=30)
    return (proc.stdout + proc.stderr).strip() or "(no output)"


TOOLS = {"read_file": read_file, "run_command": run_command}


# ---- seatbelt ----------------------------------------------------------------

READ_ONLY = {"read_file"}


def approved(name, args):
    if name in READ_ONLY:
        return True
    return input(f"allow {name}({args})? [y/N] ").strip().lower() == "y"


# ---- the postal service -------------------------------------------------------

def ask() -> dict:
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         *history],
            "tools": TOOL_SPECS}
    response = httpx.post(URL, json=body, timeout=120)   # no auth: localhost
    response.raise_for_status()
    return response.json()


def tool_result(call_id, output, is_error=False):
    prefix = "ERROR: " if is_error else ""   # no is_error field on this wire;
    return {"role": "tool",                  # errors are just words too
            "tool_call_id": call_id,
            "content": prefix + str(output)[:20000]}


# ---- the loop (with its promise book) ------------------------------------------

def turn(user_text):
    history.append({"role": "user", "content": user_text})
    for _bounce in range(25):
        try:
            reply = ask()
        except KeyboardInterrupt:
            print("\n(cancelled)")     # between letters: nothing to redeem
            return
        except httpx.HTTPError as exc:
            print(f"(provider unreachable: {exc})")   # e.g. ollama not running
            return

        message = reply["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        entry = {"role": "assistant", "content": message.get("content") or ""}
        if calls:
            entry["tool_calls"] = calls   # promises ride on the assistant mail
        history.append(entry)

        if not calls:
            print(message.get("content") or "")
            return

        results = []
        try:
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                if not approved(name, args):          # denial-as-data
                    results.append(tool_result(call["id"],
                                               "the human said no",
                                               is_error=True))
                    continue
                try:
                    output = TOOLS[name](**args)      # errors-as-data
                except Exception as exc:
                    output = f"{type(exc).__name__}: {exc}"
                    results.append(tool_result(call["id"], output,
                                               is_error=True))
                else:
                    results.append(tool_result(call["id"], output))
        except KeyboardInterrupt:
            # pressed MID-turn: apology slips for every ticket we never
            # got to -- each a role:"tool" message redeeming its own
            # tool_call_id -- the transcript must leave this function whole
            answered = {r["tool_call_id"] for r in results}
            slips = [tool_result(c["id"], "(interrupted by the human)",
                                 is_error=True)
                     for c in calls if c["id"] not in answered]
            history.extend(results + slips)
            print("\n(cancelled — session intact)")
            return

        history.extend(results)


if __name__ == "__main__":
    print(f"agent ready ({MODEL} on localhost) — ctrl-c cancels a turn, "
          "ctrl-d quits")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text:
            turn(text)
```

When you're done tinkering, the grown-up version of every piece is one
directory away — and it speaks your road too:
`uv run akshara --provider ollama`. Have fun.
