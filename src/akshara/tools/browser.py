"""Real browsing -- four verbs on a headless Chromium, behind an extra.

web_fetch reads the web's DOCUMENTS; these tools operate its APPS. The
difference is JavaScript: modern pages ship an empty shell and build
themselves in the browser, so web_fetch's stdlib stripper finds nothing
to strip -- and bot-defended sites refuse plain HTTP clients outright.
A real engine answers both, which is why this family rides Playwright
rather than more HTML parsing.

Scope decisions worth writing down:

* OPTIONAL EXTRA, CONDITIONALLY REGISTERED. ``uv add
  'aksharaharness[browse]'`` plus ``playwright install chromium`` is the
  whole opt-in -- installing the dependency IS the signal. The four
  browser_* tools appear in default_registry only when playwright is
  importable, so users who never asked for a ~300MB browser keep the
  old tool count (and the tool-selection threshold keyed to it) intact.
* TEXT OUT, NO SCREENSHOTS. Every action returns readable prose plus
  numbered interactive element refs ([e1], [e2], ...) harvested from
  the live DOM; clicking/filling takes a ref and returns the refreshed
  page. Screenshots would dogfood the image pipeline but text is the
  wire format local models are best at -- vision stays opt-in elsewhere.
* NOT read_only, ALL FOUR. Same rule as web_fetch: this is network
  egress from OUTSIDE every sandbox wall, and a browser compounds it
  (form submissions mutate remote state). Each verb gates individually;
  --yolo owns the tradeoff explicitly.
* HONEST ABOUT WALLS. Login walls, captchas, and bot detection still
  refuse us -- headless Chromium is not stealth. The tool descriptions
  say so, and when a site serves a robot check, the snapshot shows the
  check instead of pretending the mission succeeded.

One mechanical subtlety: Playwright's sync API binds its objects to
the thread that started them, but tools execute on whatever worker
thread the loop hands them (asyncio.to_thread in the async agent).
So every BrowserSession operation funnels through a single-worker
executor -- one thread sees all the traffic, whichever loop twin runs.
"""

from __future__ import annotations

from importlib.util import find_spec
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str
from akshara.tools.web_fetch import MAX_RESULT_CHARS, _clip

#: How long a navigation may take before we call the page dead.
GOTO_TIMEOUT_MS = 15_000
#: Per-action cap (click/fill) -- enough for slow frameworks, bounded.
ACTION_TIMEOUT_MS = 10_000
#: A short settle beat after navigation/actions: domcontentloaded fires
#: before SPA frameworks paint their content into the DOM.
SETTLE_MS = 400
#: Element refs listed per snapshot; beyond this the model gets a count.
MAX_ELEMENTS = 60

_BROWSER_EXTRA_HINT = (
    "playwright is not installed -- the browser_* tools are the optional "
    "[browse] extra:\n"
    "  uv add 'aksharaharness[browse]'\n"
    "  uv run playwright install chromium"
)

#: One evaluate() per snapshot: tag every visible interactive element
#: with a ref attribute AND collect the page text in the same pass, so
#: the model's view and the click targets can never disagree.
_SNAPSHOT_JS = """
() => {
  const sel = [
    'a[href]', 'button', 'input', 'textarea', 'select', 'summary',
    '[role="button"]', '[role="link"]', '[role="textbox"]',
    '[role="checkbox"]', '[role="combobox"]',
  ].join(', ');
  const labelOf = (el) => {
    const attr = el.getAttribute('aria-label') || el.getAttribute('placeholder')
      || el.value || el.getAttribute('name');
    return ((attr || el.innerText || '').replace(/\\s+/g, ' ').trim()
      ).slice(0, 80);
  };
  const kindOf = (el) => {
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role');
    const type = (el.type || '').toLowerCase();
    if (tag === 'a' || role === 'link') return 'link';
    if (tag === 'button' || tag === 'summary' || role === 'button'
        || type === 'submit' || type === 'button') return 'button';
    if (tag === 'textarea' || role === 'textbox'
        || (tag === 'input'
            && ['text', 'search', 'email', 'password', 'tel', 'url',
                'number'].includes(type))) return 'textbox';
    if (tag === 'select' || role === 'combobox') return 'select';
    if (type === 'checkbox' || role === 'checkbox') return 'checkbox';
    return 'other';
  };
  const elements = [];
  let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    const box = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if ((box.width === 0 && box.height === 0)
        || style.visibility === 'hidden' || style.display === 'none') continue;
    const ref = 'e' + (++n);
    el.setAttribute('data-akshara-ref', ref);
    elements.push({ref, kind: kindOf(el), label: labelOf(el)});
  }
  return {
    text: document.body ? document.body.innerText : '',
    elements,
  };
}
"""


def _require_http(url: str) -> str:
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        # file:// would turn the browser into an unconfined file reader --
        # refused for the same reason web_fetch refuses it.
        raise ToolError(f"only http(s) urls are supported, got {scheme!r}")
    return url


class BrowserSession:
    """The live browser behind all four tools: launch lazily, one page.

    State machine: nothing open -> page open -> closed. ``open`` is the
    only door in; ``close`` is idempotent; any action without a page
    says so in model-readable words. All playwright traffic runs on ONE
    dedicated worker thread (see module docstring for why).
    """

    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._page = None
        self._elements: dict[str, dict] = {}
        self._exec: ThreadPoolExecutor | None = None

    # -- plumbing ----------------------------------------------------------

    def _call(self, fn):
        """Run fn on the session's single worker thread, await the result."""
        if self._exec is None:
            self._exec = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="akshara-browser")
        return self._exec.submit(fn).result()

    def _launch(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ToolError(_BROWSER_EXTRA_HINT) from exc
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
            self._page = self._browser.new_page()
        except ToolError:
            raise
        except Exception as exc:
            self._teardown()
            raise ToolError(
                f"could not start Chromium: {type(exc).__name__}: {exc}\n"
                "if the browser binary itself is missing, run:\n"
                "  uv run playwright install chromium") from exc

    def _teardown(self) -> None:
        for obj, closer in ((self._page, "close"), (self._browser, "close"),
                            (self._pw, "stop")):
            if obj is not None:
                try:
                    getattr(obj, closer)()
                except Exception:
                    pass  # tearing down a corpse; never mask the real error
        self._pw = self._browser = self._page = None
        self._elements = {}

    def _require_page(self):
        if self._page is None:
            raise ToolError("no page open -- browser_open(url) first")
        return self._page

    # -- the four verbs ----------------------------------------------------

    def open(self, url: str) -> str:
        """Navigate (url given) or re-read the current page (empty url)."""
        return self._call(lambda: self._open(url))

    def click(self, ref: str) -> str:
        return self._call(lambda: self._click(ref))

    def fill(self, ref: str, text: str) -> str:
        return self._call(lambda: self._fill(ref, text))

    def close(self) -> bool:
        return self._call(self._shutdown)

    # -- bodies (all on the worker thread) ---------------------------------

    def _snapshot(self) -> str:
        data = self._page.evaluate(_SNAPSHOT_JS)
        title = (self._page.title() or "").strip()
        header = (f"title: {title}\nsource: {self._page.url}\n\n" if title
                  else f"source: {self._page.url}\n\n")
        text = (data.get("text") or "").strip() or "[no text content]"

        elements = data.get("elements") or []
        self._elements = {e["ref"]: e for e in elements}
        shown = elements[:MAX_ELEMENTS]
        lines = [f'[{e["ref"]}] {e["kind"]} {e["label"]}'.rstrip()
                 for e in shown]
        blocks = [header, _clip(text, MAX_RESULT_CHARS)]
        if lines:
            blocks.append(
                "interactive elements (pass a ref to browser_click/"
                "browser_fill):\n" + "\n".join(lines)
                + (f"\n[showing {MAX_ELEMENTS} of {len(elements)} elements]"
                   if len(elements) > MAX_ELEMENTS else ""))
        else:
            blocks.append("(no interactive elements found)")
        return "\n".join(blocks)

    def _settle(self) -> None:
        try:
            self._page.wait_for_timeout(SETTLE_MS)
            self._page.wait_for_load_state("domcontentloaded",
                                           timeout=ACTION_TIMEOUT_MS)
        except Exception:
            pass  # SPAs may never fire it again post-hydration; snap anyway

    def _resolve(self, ref: str) -> dict:
        element = self._elements.get(ref)
        if element is None:
            known = ", ".join(sorted(self._elements)[:8]) or "(none yet)"
            raise ToolError(
                f"unknown ref {ref!r} -- refs come from your LAST browser "
                f"output; known: {known}")
        return element

    @staticmethod
    def _locator(page, ref: str):
        return page.locator(f'[data-akshara-ref="{ref}"]')

    def _open(self, url: str) -> str:
        if url:
            _require_http(url)  # refuse before paying for a launch
        elif self._page is None:
            raise ToolError("no page open -- pass a url to open one")
        if self._pw is None:
            self._launch()
        if url:
            try:
                self._page.goto(url, wait_until="domcontentloaded",
                                timeout=GOTO_TIMEOUT_MS)
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(
                    f"could not load {url}: {type(exc).__name__}: {exc}"
                    ) from exc
            self._settle()
        return self._snapshot()

    def _act(self, ref: str, verb: str) -> str:
        page = self._require_page()
        self._resolve(ref)
        try:
            getattr(self._locator(page, ref), verb)(
                timeout=ACTION_TIMEOUT_MS)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                f"{verb} on {ref} failed: {type(exc).__name__}: {exc} -- "
                "the page may have changed since your last snapshot; "
                "browser_open(url) reloads it") from exc
        self._settle()
        return self._snapshot()

    def _click(self, ref: str) -> str:
        return self._act(ref, "click")

    def _fill(self, ref: str, text: str) -> str:
        page = self._require_page()
        element = self._resolve(ref)
        try:
            locator = self._locator(page, ref)
            if element["kind"] == "select":
                locator.select_option(label=text, timeout=ACTION_TIMEOUT_MS)
            elif element["kind"] in ("textbox",):
                locator.fill(text, timeout=ACTION_TIMEOUT_MS)
            else:
                raise ToolError(
                    f'{ref} is a {element["kind"]}; text goes into textboxes '
                    "(and dropdown options) -- browser_click presses buttons")
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                f"fill on {ref} failed: {type(exc).__name__}: {exc} -- "
                "the page may have changed; browser_open(url) reloads it"
                ) from exc
        self._settle()
        return self._snapshot()

    def _shutdown(self) -> bool:
        was_open = self._page is not None
        self._teardown()
        if self._exec is not None:
            self._exec.shutdown(wait=False)  # we ARE the worker; safe
            self._exec = None  # next verb builds a fresh one
        return was_open


class _BrowserTool(Tool):
    """Shared wiring: every verb speaks to the same BrowserSession."""

    read_only = False  # network egress family -- gates, like web_fetch

    def __init__(self, browser: BrowserSession) -> None:
        self.browser = browser


class BrowserOpen(_BrowserTool):
    name = "browser_open"
    description = (
        "Open a URL in a real headless Chromium browser -- JavaScript "
        "runs, so JS-rendered pages work where web_fetch sees an empty "
        "shell. Returns the page as readable text plus numbered "
        "interactive elements ([e1], [e2], ...) for browser_click/"
        "browser_fill. Call again with NO url to re-read the current "
        "page. Login walls and bot checks may refuse; one page at a time."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {"type": "string",
                    "description": "Absolute http(s) URL. Omit to re-read "
                                   "the page already open."},
        },
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return (f"browse {args['url']}" if args.get("url")
                else "re-read current page")

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return self.browser.open(require_str(args, "url", optional=True))


class BrowserClick(_BrowserTool):
    name = "browser_click"
    description = (
        "Click an element on the open browser page by its ref ([eN] from "
        "your latest browser output) and get the refreshed page back -- "
        "including wherever the click navigates."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "ref": {"type": "string",
                    "description": "Element ref from the latest browser "
                                   "output, e.g. 'e3'."},
        },
        "required": ["ref"],
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"click {require_str(args, 'ref')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return self.browser.click(require_str(args, "ref"))


class BrowserFill(_BrowserTool):
    name = "browser_fill"
    description = (
        "Type text into a textbox/textarea on the open browser page, or "
        "pick an option in a dropdown, by its [eN] ref; returns the "
        "refreshed page. Combine with browser_click on a submit button "
        "to run searches and forms."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "ref": {"type": "string",
                    "description": "Textbox/dropdown ref from the latest "
                                   "browser output, e.g. 'e2'."},
            "text": {"type": "string",
                     "description": "Text to type, or the dropdown option "
                                    "to select."},
        },
        "required": ["ref", "text"],
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        text = require_str(args, "text")
        return f'fill {require_str(args, "ref")} "{text[:40]}{"…" if len(text) > 40 else ""}"'

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return self.browser.fill(require_str(args, "ref"),
                                 require_str(args, "text"))


class BrowserClose(_BrowserTool):
    name = "browser_close"
    description = (
        "Shut the headless browser down and free it. Harmless if none is "
        "open; a later browser_open starts a fresh one."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return "close browser"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if self.browser.close():
            return "browser closed"
        return "no browser was open"


def browser_available() -> bool:
    """True when playwright is importable -- default_registry's gate."""
    return find_spec("playwright") is not None


def browser_tools() -> tuple[Tool, ...]:
    """The four browser verbs sharing one session, or () without the extra."""
    if not browser_available():
        return ()
    browser = BrowserSession()
    return (BrowserOpen(browser), BrowserClick(browser),
            BrowserFill(browser), BrowserClose(browser))
