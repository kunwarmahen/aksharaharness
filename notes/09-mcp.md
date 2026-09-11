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

Over HTTP the agreed version then has to be repeated on every later
request. From spec `2025-06-18` each one carries an
`MCP-Protocol-Version: <version>` header — `initialize` is the single
exception, because at that point nothing has been agreed yet. Leave the
header off and a strict server is entitled to fall back to the older
`2025-03-26` rules, or to refuse the request outright: the client looks
broken and the error says nothing about why. We add the header the
moment the handshake answers, and a test reads it back off a real
socket for every post-handshake request.

Refusing a version is also a cleanup path, not just an error path. Both
transports hand the connection back before raising — stdio reaps its
child process, HTTP closes its connection pool — so a server we decline
to speak to leaves nothing behind.

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
  `close()` is idempotent, the same promise stdio's makes: shutdown
  paths overlap (a failed `start()` closes, then the manager closes
  again on exit), and httpx raises `RuntimeError` — not the
  `TransportError` you would think to catch — if you send on a client
  that is already closed.
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
  `AsyncExitStack`. Two details earn the "no zombies": a `wait()` after
  the SIGKILL (the signal cannot be caught, but the child still has to
  be collected — kill without wait is exactly how you make the thing
  you were trying to avoid), and closing `proc.stdout` at the end.
  Killing a child does NOT close the read end of a pipe WE opened, so
  without that last step every server costs one descriptor for the life
  of the harness. Both were missing until `ResourceWarning` was
  promoted to an error in the test config and said so.

Live finding worth keeping: the CLI's `confirm_gate` prompts for
EVERYTHING — the `readOnlyHint` matters to the `allow_read_only` gate,
not to the interactive y/n gate. And a headless one-shot run with the
default gate produces the most honest demo of denial-as-data we have:
the prompt hits EOF on closed stdin, the denial becomes a tool result
("permission gate failed: EOF when reading a line"), and the model
reports the failure truthfully instead of hallucinating an echo.

## Runtime management: servers you can unplug

Startup wiring (`--mcp-config`) answers "which servers exist"; it does
not answer "which servers exist RIGHT NOW". `MCPManager` makes the
answer mutable mid-session — one owner object for every live session,
shared by the web UI's servers panel and the REPL's `/mcp` commands so
both stay in step by construction:

```
/mcp                          # list: name (transport) [down] · saved — target — N tool(s)
/mcp off tiny   /mcp on tiny  # soft toggle: whole toolset, process stays warm
/mcp add tiny python srv.py   # connect NOW (asks whether to remember)
/mcp remove tiny              # kill child, unregister tools, forget saved entry
```

The decisions worth writing down:

* **Two verbs for two intents.** `/mcp off` is the tools switch
  ([17-tool-selection.md](17-tool-selection.md)) pointed at every
  qualified name of one server: registry-disabled per call, consulted
  LIVE, process untouched, reversible. `/mcp remove` is a teardown:
  close() escalates stdin-EOF → SIGTERM → SIGKILL, every
  `mcp__server__*` unregisters, and any saved entry is forgotten.
  Disabling is "stop asking this server"; removing is "this server was
  never here".
* **Failed connects roll back completely.** If registration dies
  halfway through a server's toolset (say, a name collision), the
  already-registered names unregister AND the session closes before the
  error re-raises — otherwise you get zombie entries no command can
  reach, because `tool_names` never learned them. The test pins it by
  asserting the half-added tool is gone and `created[0].closed`.
* **Membership changes rebuild the selection catalog**, but the
  rebuild must not silently revoke operator choices:
  `enable_selection()` re-runs over the new registry while the old
  `must_include` pins are copied onto the fresh catalog. A server added
  at runtime therefore shows up under exactly the same pinning policy
  as one wired at startup.
* **Remembering is opt-in per server, asked each time.** The web form
  has a checked-by-default checkbox; the REPL follows its success
  message with the y/N prompt. Yes → `.akshara/mcp.json` in the working
  directory (same building-block shape as `--mcp-config`, upserted with
  an atomic tmp+rename); future launches auto-reconnect those servers
  after flag configs, skipping any name already connected. Removal
  ALWAYS forgets — "gone-gone" beats a surprise resurrection.
* **A corrupt remembered file is loud.** Silent loss of an operator's
  hand-written config is worse than a refused launch; `load_remembered`
  raises rather than shrugging.

## Trying it in the portal, end to end

The panel is the easiest way to see MCP work without editing a config
file first. One terminal is enough -- for a stdio server the portal
spawns the child itself, so there is nothing to start beforehand.

```bash
cd <this repo>                      # the cwd matters; see the traps below
uv run akshara --web --provider ollama
```

Open `http://localhost:8321` and click the **⚙ chip** in the top bar
(it shows the live tool count). That opens **servers, skills & tools**,
with the **mcp servers** section first. Click **＋ add**:

* **name** `tiny` -- the tools will register as `mcp__tiny__…`
* **transport** leave *runs a command* selected (stdio is the default)
* **command** `python examples/tiny_mcp_server.py` -- one line,
  split into command + args the way a shell would
* untick **remember** for a throwaway (it is ticked by default)

**connect** adds a row: a green dot, a `stdio` badge, `2 tool(s)`, and
the ⚙ count climbs by two. Now ask for it in the chat box:

> Use the mcp__tiny__add tool to add 19 and 23. Report only the number.

A tool card appears in the transcript -- `mcp__tiny__add`,
`{"a": 19, "b": 23}`, output `42`. On `qwen3.8` the whole round trip
runs locally; no cloud key is involved at any point.

The row's **switch** is `/mcp off`: flip it and the row reads
`2 tool(s), 2 off`, the toast warns that its calls now fail as data,
and the child process stays warm. The **✕** is `/mcp remove`, and it
arms before it fires -- the button becomes *sure?* for 2.5 seconds
rather than opening a native dialog.

### What "paste json" is

The second tab of the add form takes the SAME `{"servers": {...}}`
object a `--mcp-config` file holds, as text instead of a path:

```json
{"servers": {"tiny":   {"command": "python",
                        "args": ["examples/tiny_mcp_server.py"]},
             "remote": {"url": "http://127.0.0.1:9731/mcp"}}}
```

`parse_mcp_text` hands that string to the same
`_configs_from_servers_dict` the file loader uses, so the two can never
drift: anything a config file accepts, the box accepts, and the error
messages are the ones the loader already writes. Three things it buys
over the fields form, which does one server at a time:

* **several servers in one paste** -- the example above connects both
  transports at once;
* **per-server results**, so one bad entry does not hide the good ones:
  the response is a list of `{name, ok, tools}` / `{name, ok, error}`;
* **a config someone handed you** goes straight in -- no file, no
  restart, no flag.

The **remember** tick applies to every server in the blob. Bad JSON is
refused before anything spawns, with the parser's own complaint
(`pasted config is not valid JSON: …`) shown under the box.

### Three traps, all of them real

* **Relative paths resolve against the portal's working directory**,
  not this repo. Started from the repo root, `examples/tiny_mcp_server.py`
  is right; started anywhere else it surfaces as
  `exited unexpectedly (code 2) during 'initialize'` -- which does not
  obviously mean "wrong path". Absolute paths always work.
* **remember is ticked by default.** Leave it on for a test server and
  it lands in `.akshara/mcp.json` and reconnects on every future
  launch; the row grows a **saved** badge when that has happened.
  Removing forgets it again.
* **The transport radio defaults to stdio.** For a Streamable-HTTP
  server (`python examples/tiny_mcp_server.py --http 9731`, second
  terminal) pick *talks to a url* and give it
  `http://127.0.0.1:9731/mcp`, or the form will try to run your URL as
  a command.

### Remote servers that want a token

Public example servers speak to anyone. Commercial ones generally do
not, and the first failure is immediate and total:

```
'vendor' rejected 'initialize': HTTP 401 'authentication required'
```

That is not a misconfigured URL. The server answered with the MCP
spec's authorization challenge:

```
www-authenticate: Bearer resource_metadata="https://…/.well-known/oauth-protected-resource/…"
```

which is an invitation, not a wall. Two ways in, and which one you
need depends on how the vendor issues credentials.

#### If the server hands out a static token

Put it in `headers`, the HTTP transport's answer to stdio's `env`:

```json
{"servers": {"vendor": {
   "url": "https://agent.example.com/mcp/trading",
   "headers": {"Authorization": "Bearer ${VENDOR_TOKEN}"}}}}
```

Refused outright on a server with no `url`: silently dropping a
credential resurfaces as an unexplained 401 an hour later.

**`${VAR}` is expanded from the environment at connect time**, and
that indirection is the point rather than a convenience. `remember`
writes `.akshara/mcp.json` into the working directory; storing the
literal token there would put a live credential one `git add -A` away
from a public repository. The placeholder is what lands on disk, and it
resolves afresh every launch. An UNSET variable is an error, not an
empty string:

```
mcp server 'vendor': header 'Authorization' references 'VENDOR_TOKEN',
which is not set in the environment (put it in .env or export it first)
```

Sending `Authorization: Bearer ` instead would earn a 401 that reads
like a rejected password rather than a variable nobody set.

Configured headers are merged FIRST and the protocol's own go in after,
so a stray `Content-Type` in somebody's config cannot break the
handshake in a way that looks like a server bug.

The live proof that the header reaches the wire, against the real
endpoint above: without it the server says `authentication required`;
with a deliberately wrong token it says `JWT verification failed`. The
second error is the server having read the credential and disliked it
-- which is exactly the state a correct token turns into a session.

#### If the server only issues tokens through a login

Most commercial ones do, and then there is no token to paste: the flow
IS the credential. `akshara --mcp-login NAME` walks it
([mcp_oauth.py](../src/akshara/mcp_oauth.py)), and the panel's 🔑 on any
http row does the same thing from the browser you already have open.

```bash
uv run akshara --mcp-login vendor --mcp-config vendor.json
```

Five RFCs' worth of moving parts, and the walk between them is short
enough to write out:

1. the 401's `WWW-Authenticate` names a **protected-resource metadata**
   document (RFC 9728) -> which authorization server guards this thing;
2. that server's own metadata (RFC 8414) -> where to register,
   authorize, and collect tokens. The `.well-known` segment goes BEFORE
   the issuer's path, which surprises everyone, so both spellings get
   tried;
3. **dynamic client registration** (RFC 7591) -> a `client_id`, with no
   developer-portal visit. These servers advertise
   `token_endpoint_auth_methods: ["none"]` -- a PUBLIC client, which is
   the only honest posture for a CLI that ships its own source;
4. the browser, carrying a **PKCE** challenge (RFC 7636). A public
   client has no secret, so the authorization code is the only thing
   between an attacker who can see the redirect and a live token; the
   verifier, held only in this process, is what proves the code is
   being redeemed by whoever asked for it;
5. the code plus the verifier buy an access token and a refresh token.

Decisions worth writing down:

* **The redirect port is bound FIRST and held for the whole flow.**
  Registration names the `redirect_uri`, so the port has to be ours
  before we can quote it -- a listener rebound after registering can
  land on a different port and fail at the last step with a redirect
  mismatch that explains nothing. This was written wrong once, exactly
  that way, before the test caught it.
* **`state` is checked and a mismatch refuses the redirect** -- but
  AFTER the server's own `error` parameter, and only when there is a
  code to accept. Checking it first cost a real debugging session: an
  error redirect that omits `state` (plenty do) got reported as a CSRF
  failure, burying the one fact the user needed.
* **Only a request carrying `code` or `error` counts as the redirect.**
  The browser sends more than you asked for -- it fetches
  `/favicon.ico` for the success page, and may prefetch on its own
  schedule. An earlier handler recorded every GET, so a query-less
  favicon overwrote the real redirect and the flow died claiming CSRF.
  The success page now also carries `<link rel="icon" href="data:,">`
  so the request is never made; either guard alone is enough, and
  having both is cheap.
* **Tokens live in `~/.local/state/akshara/mcp-tokens.json`, mode
  0600**, set on the temp file BEFORE the rename so the secret is never
  briefly world-readable. Not `.akshara/` -- that sits in a working
  directory, and a refresh token is a long-lived credential one
  `git add -A` in one careless repo away from being public.
* **A 401 raises `MCPAuthRequired`, a subclass of `MCPError`.** Every
  existing "warn and skip this server" path keeps working untouched,
  while callers that want to offer a login can catch the narrower type
  and read the challenge off it.
* **One refresh-and-retry on a 401 mid-session.** The server disagreeing
  with our expiry arithmetic means the server is right, so the token is
  refreshed even when it still looks live, and the request goes again
  exactly once.
* **A refresh response that omits `refresh_token` keeps the old one.**
  Dropping it would log the user out at the next expiry for no reason.
* **An explicitly configured `Authorization` header wins over a stored
  token.** The operator said what they wanted; second-guessing it would
  make the two features fight.

What is still missing: nothing in the flow, but it is only exercised
against a fake authorization server in the suite -- 37 tests, with the
redirect half running on a real socket because a bound port, a `state`
and a one-shot handler have to agree and a mock would only report that
they did. The discovery half IS verified against a live commercial
endpoint; the registration and browser halves need somebody's actual
account, so they are the operator's to exercise.

Every panel action is a plain endpoint underneath -- `/api/mcp`,
`/api/mcp/add`, `/api/mcp/toggle`, `/api/mcp/remove` -- so the same
walkthrough runs from `curl` when you would rather not click
([22-web-ui.md](22-web-ui.md)).

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

Re-verified against a header-logging server afterwards, which is how
three HTTP-side defects surfaced: `initialize` went out bare (correct)
but so did every request after it (`MCP-Protocol-Version` missing — a
strict server may refuse all of them); a refused version left the
connection pool open; and a second `close()` raised `RuntimeError` from
httpx instead of doing nothing. All three are now pinned by tests that
read the header back off a real socket and assert the pool is closed.

## Deliberately not built

Client-credentials and device-code grants (the browser flow covers the
interactive case, which is what MCP vendors ship); token revocation on
`/mcp remove` (we forget ours, the server keeps its record).

Standalone GET stream and RPC batching (see above); resources, prompts,
and sampling capabilities (tools are the 90% case);
`structuredContent` decoding (we flatten to text); roots beyond an
empty reply; session resumption across client restarts. The module
docstring is the honest spec of what IS built.
