# 25 — web_fetch: one road out of the sandbox

*The last hard wall around the agent wasn't security — it was
ignorance. A dependency's error message, an API changelog, the actual
docs for a flag: invisible unless a human pasted them in. Autonomous
missions kept dying on steps whose only requirement was "look it up."*

## Scope, decided up front

**Fetch, not search.** A search engine means API keys, terms of
service, scraping arms races — none of it honest to ship as "built-in."
Fetching a KNOWN url is plain HTTP, and models find urls constantly:
in package metadata, in tracebacks, in their own notes. The tool
description says this out loud ("there is no search engine here") so
the model stops trying before it starts.

**Text out.** Pages strip to readable prose via stdlib `HTMLParser` —
no scraping framework, consistent with the harness's two-dependency
diet (httpx + rich). `<script>/<style>/<svg>` drop whole; block tags
become newlines; list items get `- ` bullets; entities decode;
blank-line runs collapse. JSON pretty-prints so a model can navigate
it; `text/*` passes through raw; anything binary refuses with the
offending content-type named.

## The decision that matters: NOT read_only

Fetching mutates nothing local, and the lazy call was read_only=True,
auto-approved forever. But web_fetch breaks a property every other
tool respects: it talks to the network from OUTSIDE every sandbox
wall. Bwrap-confined bash has `--unshare-net` precisely so approved
commands can't phone home — an auto-approved fetch quietly reopens
that hole, and url query strings are a data-exfiltration channel even
for a "read."

So web_fetch gates like bash: a human sees the address before the
first fetch. Autonomous runs pass --yolo and own the tradeoff
explicitly, which is the honest version of autonomy — granted, not
smuggled.

## Engineering details worth keeping

- **Download cap before parsing**: 2 MB, enforced while streaming —
  fetching a url should never mean pulling gigabytes through the box.
- **Head-tail output clip** (8k chars): page errors live at the bottom,
  exactly like tracebacks — same both-ends rule as the bash tool.
- **Malformed HTML never kills the fetch**: whatever text extracted
  before the parse hiccup still ships.
- **Errors name the fix**: non-http schemes, HTTP >= 400 status,
  unsupported types, connection failures — all ToolError with the
  offending value quoted.

## Testing network code offline

The module exposes its httpx.Client behind a `_client_factory` seam;
tests swap in a `httpx.MockTransport` client (with production's real
options — follow_redirects bit me once: a mock client WITHOUT it
returned the 302 body itself, which the test caught). Status codes,
content-type branching, redirect following, connection failures — all
exercised through run()'s real path with zero sockets.

## What the tests pin

- title extraction + script/style stripping + entity decoding
- bullet-per-list-item; blank-line collapse; malformed HTML survives
- html → prose header (title+source), json → pretty-printed,
  text/plain passthrough, image/png refused by type
- HTTP 404 → ToolError; `file://` → ToolError without touching the fs;
  ConnectError → ToolError; redirects followed
- read_only is False ON PURPOSE; summary shows the url

## Receipts

Offline suite green via MockTransport.

Live receipt (local-only, no external traffic): `python -m http.server
8377` serving a hand-written page; Ollama `qwen3.8` one-shot `--yolo`
fetched `http://127.0.0.1:8377/hello.html` through the real tool and
reported back the title — *"Akshara Receipt Page"* — plus its body
text. The gate was skipped via --yolo by choice; without it the run
would have stopped at the approval prompt naming that exact url.
