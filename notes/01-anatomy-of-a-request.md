# 01 · Anatomy of a request

> One raw, non-streaming call to the Anthropic Messages API — no SDK —
> and how what comes back gets normalized into our own types.

## What an agent harness actually is

At its core, a harness is a **loop**:

```
        ┌────────────────────────────────────────────┐
        │                                            │
        ▼                                            │
   user message                                      │
        │                                            │
        ▼                                            │
   ┌─────────┐   reply contains     ┌──────────┐      │
   │  model  │ ───tool calls?──---> │ run tools│ ─────┘
   │  call   │                      └──────────┘   results appended
   └─────────┘ <── results go back as messages       as messages
        │
        ▼
   no tool calls → final answer, loop exits
```

Everything else (tools, permissions, streaming, context management)
exists to serve that loop; [05](05-agent-loop.md) is where it gets
built for real.

## The Messages API request

Every request to `POST {base}/v1/messages` carries:

| Piece | Value | Why it matters |
|---|---|---|
| `x-api-key` header | your key | Anthropic authenticates via *header*, not Bearer |
| `anthropic-version` header | `2023-06-01` | pins wire behavior; APIs evolve without breaking you |
| `model` | e.g. `claude-sonnet-4-5` | |
| `max_tokens` | int, **required** | unlike OpenAI; omit it → 400 |
| `system` | top-level string | NOT part of `messages` — it's per-request metadata |
| `messages[]` | `{role, content}` pairs | roles alternate user/assistant |
| `content` | **list of typed blocks** | `[{"type":"text","text":...}]`, not a plain string |

The block-shaped content is the important design idea. A single
assistant turn can mix prose and tool calls because each is just another
block. Our internal types (`src/akshara/types.py`) mirror this shape on
purpose.

## Normalization — the one big architectural decision

The rest of our code will never see provider JSON. Each adapter owns two
translations:

```
internal types  --encode-->  wire JSON          (request building)
wire JSON/SSE   --decode-->  internal types     (response parsing)
```

Why it matters:

- The agent loop, tools, CLI — everything speaks ONE vocabulary.
- Switching providers mid-session costs nothing (history stays internal).
- The translation logic lives in exactly two files, so when a provider
  changes its format, you know where to look.

The cost: exotic features (thinking blocks, citations) don't map cleanly.
Our rule: unknown blocks become a **visible placeholder** + stderr
warning — never silently dropped, never a crash.

## How we test without a network

```python
provider = AnthropicProvider(
    settings,
    transport=httpx.MockTransport(handler),  # <-- the ONLY test seam
)
```

`MockTransport` serves canned responses from a pure function while the
REAL request-building code runs. Assertions then pin exact path, headers,
and body — see `tests/test_anthropic_adapter.py`. That file is the
executable version of the wire-format cheat-sheet.

## Files added this phase

- `pyproject.toml`, `.python-version` — uv-managed project, deps: httpx + rich only
- `src/akshara/types.py` — the shared vocabulary (read this first)
- `src/akshara/errors.py` — two exception families: provider failures propagate, tool failures become data
- `src/akshara/config.py` — env vars → settings; base-URL conventions documented at top
- `src/akshara/providers/base.py` — the `Provider` ABC
- `src/akshara/providers/anthropic.py` — encode/decode for the Messages dialect
- `examples/one_shot.py` — prints request JSON → raw response → normalized view

## Try it live

```bash
cp .env.example .env   # fill in your key
uv run pytest -q                                       # offline tests green
uv run python examples/one_shot.py "Why is the sky blue?"
```
