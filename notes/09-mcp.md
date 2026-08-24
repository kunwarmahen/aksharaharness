# 09 · MCP: the Model Context Protocol, by hand

> Book ch13 — which uses the official `mcp` SDK and calls a from-scratch
> client "300 lines of undifferentiated code". This project disagrees:
> no wire is undifferentiated (the SSE parser is hand-rolled too), so
> [mcp.py](../src/akshara/mcp.py) implements both transports — stdio and
> Streamable HTTP — plus JSON-RPC 2.0 directly. Files: `mcp.py`,
> `examples/tiny_mcp_server.py` (the book's suggested exercise — both
> sides of BOTH transports), `tests/test_mcp.py`.

## Why MCP exists, in one line

M AI apps × N services = M×N bespoke connectors; a common client/server
protocol turns that into M+N. The whole protocol reduces to FOUR
exchanges: `initialize` request → initialize response (version +
capabilities) → `notifications/initialized` → then `tools/list` /
`tools/call` at will.

## The stdio transport rules that make it small

* **One JSON-RPC message per line**, newline-delimited, never embedded
  newlines (this is NOT LSP's Content-Length framing). Our `_send`
  asserts it — a multi-line message would silently corrupt the stream.
* **stderr passes straight through to ours.** Piping it would risk a
  full buffer deadlocking the protocol; server logs are the server's
  business.
* Requests carry an integer id; responses quote it back; notifications
  (no id) are dropped. Live-verified framing:
  `{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"42"}],"isError":false}}`.

## Version negotiation

We request our newest (`2025-06-18`); the server answers with what IT
supports; if that's not in our known set we REFUSE and say so rather
than guess at incompatible semantics (spec-conformant: the client owns
compatibility).

## Threading model (the actual engineering)

One daemon reader thread owns stdout and dispatches every line:

* responses → per-id `queue.Queue` (registered BEFORE sending — a
  response can never be lost);
* server-initiated REQUESTS get answered (`ping` → `{}`,
  `roots/list` → empty, unknown → error -32601). **Ignoring these
  wedges polite servers** — a server waiting on its reply can't serve
  yours. The tests prove it: a fake server that blocks on its ping
  answer hangs `tools/list` into timeout if the client ignores it.
* two threads share the subprocess stdin → one write lock;
* on EOF the reader pushes a sentinel into every pending queue, so a
  crashed server fails fast with its exit code ("exited unexpectedly
  (code 7)") instead of every caller burning the full timeout.

Timeouts are per-request; a timed-out call does NOT poison the session
(the late response arrives, finds its slot deleted, and is dropped).

## Streamable HTTP: same protocol core, no reader thread

`MCPHttpSession` (spec 2025-03-26+), chosen by config shape — `url` set
means HTTP, `command` means stdio. Both sessions expose the IDENTICAL
surface (`start/list_tools/call_tool/close`), so the wrapper, the
registry, the permission gate, everything downstream is transport-blind.

The transport differences that matter:

* **No reader thread.** stdio needs one because responses can arrive at
  any time on a pipe nobody else watches. HTTP responses arrive
  synchronously with the request that caused them — there is nothing to
  pump in the background.
* **Session state is a header.** The server issues `Mcp-Session-Id` at
  initialize; every later request echoes it back. A process lifetime
  becomes an opaque string — which also makes sessions resumable across
  client restarts in principle (we don't persist ours; close() DELETEs).
* **The response body may be SSE.** `tools/call` answers often stream:
  `text/event-stream` frames parsed by the SAME `parse_events()` the
  provider adapters use. That's the whole dividend of writing framing once,
  provider-agnostically — the MCP integration got it for free.
* **Server→client requests ride INSIDE our response stream.** Our
  `_rpc` scans every message in the body: ours gets matched by id,
  embedded ones get answered with another POST (same
  ping/roots/-32601 table as stdio). Live proof below.

Honest non-implementations: the standalone GET server→client stream
(a client MAY skip it per spec) and batching several RPCs per POST.
A 404 on a stale session should trigger re-initialize per spec; we
surface the error instead of auto-recovering.

The example server grew the matching `--http` mode by refactoring its
protocol core into one function both transports call — which is what
makes "the two transports are equivalent" testable rather than claimed.

## The wrapper: external tools become ordinary Tools

`MCPToolWrapper(Tool)` — instance attributes shadow the ABC's ClassVars,
so server-supplied metadata just works:

* **Qualified names** `mcp__<server>__<tool>` (Claude Code's
  convention): collisions between servers are structurally impossible
  (duplicate registration raises ValueError — loud, never silent
  overwrite) and provenance is visible in transcripts and permission
  prompts.
* **Annotations are hints, never guarantees**: `readOnlyHint` is
  honored when present; absence means pessimistic `read_only=False`.
* **`isError` → ToolError**: the tool ran and failed — the same
  errors-as-data convention as our built-ins, preserved across the
  process boundary. Content blocks flatten to text; images/resources
  are counted, not decoded.
* Lifecycle: `close()` = stdin EOF (polite) → SIGTERM → SIGKILL, no
  zombies. This is our sync-world answer to the book's
  `AsyncExitStack`.

Live finding worth keeping: the CLI's `confirm_gate` prompts for
EVERYTHING — the `readOnlyHint` matters to the `allow_read_only` gate,
not to the interactive y/n gate. And a headless one-shot run with the
default gate produces the most honest demo of denial-as-data we have:
the prompt hits EOF on closed stdin, the denial becomes a tool result
("permission gate failed: EOF when reading a line"), and the model
reports the failure truthfully instead of hallucinating an echo.

## Security: an integration standard, not a security boundary

The chapter's loudest lesson, and the reason our structural choices
matter:

* **Token aggregation** — a server concentrates credentials (GitHub
  PATs, DB creds); its process tree is a single compromise away from
  leaking all of them. Pin versions; review before install; least
  privilege.
* **Indirect prompt injection** — tool OUTPUT is untrusted text that
  can carry instructions (Greshake et al. 2023; EchoLeak
  CVE-2025-32711 as the flagship). The first documented malicious npm
  MCP package (Sept 2025) exfiltrated filesystem state. Treat
  `mcp__*` results like any other model-visible text: no special trust
  for being "from a tool".
* Our mitigations are structural: qualified names, permission gates
  that sub-agents cannot escalate past, and pessimistic side-effect
  assumptions.

## Verified live

Both transports were exercised end-to-end against
`examples/tiny_mcp_server.py`, one process each side: discovery
banner at startup, a real `mcp__tiny__add(19,23)` → `42` round trip,
and — over Streamable HTTP — an SSE response stream whose embedded
ping was answered by POST on the same connection. A second stdio run
without `--yolo` exercised the gate path described above.

## Deliberately not built

Standalone GET stream and RPC batching (see above); resources, prompts,
and sampling capabilities (tools are the 90% case);
`structuredContent` decoding (we flatten to text); roots beyond an
empty reply; session resumption across client restarts. The module
docstring is the honest spec of what IS built.
