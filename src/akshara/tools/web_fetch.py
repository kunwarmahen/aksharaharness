"""web_fetch -- read one URL, as text. The harness's only road out.

Before this tool, the model's world ended at the sandbox edge: an
error message from a dependency, an API reference, a changelog -- all
invisible unless the human pasted them in. Autonomous missions died on
exactly those steps ("check the docs" was advice it could not take).

Scope decisions worth writing down:

* FETCH ONLY, NO SEARCH. A search API needs keys and terms-of-service
  negotiations; fetching a KNOWN url is plain HTTP. The model finds
  urls in package metadata, error text, and its own notes.
* NOT read_only, ON PURPOSE. It mutates nothing local, but it is the
  one built-in that talks to the network from OUTSIDE every sandbox
  wall -- bwrap-confined bash has no route out precisely so approved
  calls can't phone home, and an auto-approved fetch would quietly
  reopen that hole (url query strings carry data). So it gates like
  bash: a human approves the address. Autonomous runs pass --yolo and
  own that tradeoff explicitly.
* TEXT OUT. Pages are stripped to readable text (stdlib HTMLParser --
  no scraping framework); non-HTML text formats pass through raw;
  anything binary is refused. One tool, one job.

Deps: httpx only (already the harness's HTTP layer).
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from akshara.errors import ToolError
from akshara.tools.base import Tool, ToolContext, require_str

#: Download cap. Big pages get clipped here BEFORE parsing -- reading a
#: URL should never mean pulling hundreds of megabytes through the box.
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
#: Output cap, applied head-tail style (errors live at the END of a page).
MAX_RESULT_CHARS = 8_000
#: Politeness + honesty in one header: what we are, nothing more.
USER_AGENT = "akshara-harness (+local learning agent)"

STRIP_TAGS = {"script", "style", "noscript", "template", "svg"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "nav",
    "aside", "br", "hr", "li", "ul", "ol", "tr", "table", "thead",
    "tbody", "blockquote", "pre", "form", "fieldset",
    "h1", "h2", "h3", "h4", "h5", "h6",
}

# Client seam: tests swap this for a MockTransport-backed client, so the
# suite stays offline-green while run() exercises real request/response
# handling unchanged.
_client_factory = lambda: httpx.Client(
    follow_redirects=True,
    timeout=httpx.Timeout(15.0),
    headers={"User-Agent": USER_AGENT},
)


class _TextExtractor(HTMLParser):
    """HTML -> readable text. Not a browser engine: structural whitespace
    around block tags, entities decoded by convert_charrefs, scripts and
    styles dropped whole, <title> captured separately."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in STRIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in STRIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
        elif data.strip():
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, body_text) for an HTML document."""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # Malformed markup must never kill the fetch -- whatever text was
        # extracted before the parse hiccup is still worth showing.
        pass

    lines = [line.strip() for line in "".join(extractor.parts).splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if line or (collapsed and collapsed[-1]):
            collapsed.append(line)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return extractor.title, "\n".join(collapsed)


def _clip(text: str, cap: int = MAX_RESULT_CHARS) -> str:
    """Head-tail clip -- the same both-ends rule bash uses, because page
    errors live at the bottom just like tracebacks do."""
    if len(text) <= cap:
        return text
    keep = cap // 2
    return (f"{text[:keep]}\n[... {len(text) - cap} chars omitted ...]\n"
            f"{text[-keep:]}")


def fetch(url: str) -> str:
    """GET ``url``, return readable text. Raises ToolError on everything
    the model could act on (bad scheme, HTTP status, unsupported type)."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ToolError(f"only http(s) urls are supported, got {scheme!r}")

    try:
        with _client_factory() as client:
            with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise ToolError(f"HTTP {response.status_code} from {url}")
                content_type = (response.headers.get("content-type")
                                or "").split(";")[0].strip().lower()
                body = b""
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_DOWNLOAD_BYTES:
                        raise ToolError(
                            f"response exceeds {MAX_DOWNLOAD_BYTES} bytes; "
                            "refusing to download more")
    except httpx.HTTPError as exc:
        raise ToolError(f"fetch failed: {type(exc).__name__}: {exc}") from exc

    if content_type in ("text/html", "application/xhtml+xml"):
        title, text = html_to_text(body.decode("utf-8", errors="replace"))
        header = f"title: {title}\nsource: {url}\n\n" if title \
            else f"source: {url}\n\n"
        return _clip(header + (text or "[no text content]"))

    if content_type.startswith("text/") or content_type in (
            "application/json", "application/javascript",
            "image/svg+xml"):
        text = body.decode("utf-8", errors="replace")
        if content_type == "application/json":
            try:  # pretty-print so models can navigate the structure
                text = json.dumps(json.loads(text), indent=2)
            except (json.JSONDecodeError, ValueError):
                pass  # server lied about the type; show it raw
        return _clip(f"source: {url} ({content_type})\n\n{text}")

    raise ToolError(f"unsupported content-type {content_type!r} from {url} "
                    "(text/html, text/*, and json only)")


class WebFetch(Tool):
    name = "web_fetch"
    description = (
        "Fetch a URL over HTTP(S) and return its content as readable "
        "text (HTML stripped to prose; JSON pretty-printed). For looking "
        "up documentation, error messages, changelogs, and any public "
        "page you have the address of. There is no search engine here "
        "-- you must know the URL."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    read_only = False  # network egress gates, exactly like bash (see above)

    def summary(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return f"fetch {require_str(args, 'url')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return fetch(require_str(args, "url"))
