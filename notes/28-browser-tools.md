# 28 — browser_*: operating web apps, not just reading documents

*web_fetch (notes/25) reads the web's DOCUMENTS. But a large and
growing share of the web is an empty HTML shell that JavaScript fills
in — to web_fetch those pages are blank — and bot-defended sites
refuse plain HTTP clients outright. The agent could read about the
world but not operate it. Four verbs on a real headless Chromium close
that gap, behind an optional extra.*

## Why an engine, not more parsing

No amount of HTML stripping executes JavaScript; there is no honest
shortcut. So this family rides Playwright rather than another parser —
the harness's third dependency tier, after the httpx+rich core and the
`--web` FastAPI exception, justified the same way: the capability
CALLS for a browser engine.

## The extra is the opt-in

```
uv add 'aksharaharness[browse]'
uv run playwright install chromium
```

Installing the dependency IS the signal: default_registry registers
browser_open/click/fill/close only when playwright is importable.
That conditionality is load-bearing, not cosmetic — the tool-selection
threshold is 20, sixteen built-ins + ask_user = 17 today, and
always-on browser verbs would push EVERY user past the cliff into
per-turn tool discovery they never asked for. No extra → no tools,
count stays 16/17. Extra installed → twenty (+ask_user = 21), which
DOES cross the threshold, deliberately: you opted into a heavier mode,
so the harness switches to its heavy-mode tool strategy (`--tool-select
0` puts the full list back).

## Refs are the model's hands

Every action returns readable prose plus numbered interactive elements
harvested from the LIVE DOM in one `evaluate()` pass: visible elements
matching link/button/input/select roles get `data-akshara-ref="eN"`
attributes AND appear as `[e3] textbox Search`. Clicking and filling
resolve refs through that same attribute — the model's view of the
page and its click targets can never disagree. Refs regenerate on
EVERY snapshot, so they always describe the page as of your last
action.

Text out, no screenshots: text is the wire format local models are
best at. Vision stays where it belongs, behind read_image's explicit
opt-in.

## Trust: the web_fetch rule, times four

Same argument as notes/25, stronger: this is network egress from
outside every sandbox wall, and a browser compounds it — form
submissions MUTATE remote state. All four verbs gate individually;
--yolo owns the tradeoff explicitly. And honesty about limits cuts
both ways: headless Chromium is NOT stealth. Login walls, captchas,
and bot checks still refuse us — when a site serves a robot check,
the snapshot shows the check instead of pretending the mission
succeeded.

## One thread for one browser

Playwright's sync API binds its objects to the thread that started
them — but tools execute on whatever worker thread the loop hands them
(`asyncio.to_thread` in the async agent). So every BrowserSession
operation funnels through a single-worker executor: exactly one thread
sees the traffic, whichever loop twin runs. The lifecycle state machine
is three words — nothing, open, closed — with close re-arming launch
and every action without a page saying so in model-readable words.

## What the tests pin

- snapshot rendering: title/source header, element lines, unlabeled
  elements clean, "(no interactive elements found)"
- head-tail clip on long pages; `[showing N of M]` element cap at 60
- click resolves `[data-akshara-ref]` and returns the REFRESHED page;
  fill types into textboxes, selects dropdown options by label, and
  refuses buttons with guidance; unknown ref lists known refs
- lifecycle: url-less open needs a page; second open reuses the page;
  close resets everything; reopen re-arms launch (proved by the
  missing-extra error firing again)
- launch paths: missing playwright names `aksharaharness[browse]`;
  missing chromium binary names `playwright install chromium`;
  `file://` refused BEFORE any launch cost
- ONE worker thread sees all traffic, never the caller's thread
- gating: all four NOT read_only; summaries name url/ref/text
- registration: without playwright 16 tools; with it 20 sharing ONE
  BrowserSession — the suite pins both worlds via the find_spec seam

## Receipts

Offline suite green — 592 passing WITHOUT playwright installed (the
fakes carry it), green again WITH the extra synced (and with `[web]`
alongside: uv syncs exactly the extras you name).

Live receipt (all traffic local, Ollama `qwen3.8` one-shot `--yolo
--tool-select 0`): a hand-written page served by `python -m http.server`
whose results div is EMPTY until its button's onclick builds the list —
and web_fetch on the same url provably ends at the word "Search"
('turbo' appears nowhere in its output; the script is stripped). The
model opened the page, typed `turbo` into the `[e1] textbox`, clicked
`[e2] button Search`, and read back the JS-built snapshot:
**"turbo encabulator -- $42.00"** — reported exactly that, end_turn.
`3419 in / 97 out · 4 iteration(s)`
