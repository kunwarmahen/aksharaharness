"""web_fetch: HTML->text extraction, fetch behavior over a mock transport.

The suite stays offline-green: every network test swaps the module's
``_client_factory`` seam for a MockTransport-backed client, so run()
exercises its real request/status/content-type logic unchanged.
"""

from __future__ import annotations

import json

import httpx
import pytest

from akshara.errors import ToolError
from akshara.tools import WebFetch
from akshara.tools.base import ToolContext
from akshara.tools.web_fetch import _client_factory, fetch, html_to_text


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


def serve(handler) -> None:
    """Point web_fetch's client factory at a canned transport."""
    # same client options as production; only the transport is fake
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          follow_redirects=True)
    import akshara.tools.web_fetch as wf
    wf._client_factory = lambda: client


def restore() -> None:
    import akshara.tools.web_fetch as wf
    wf._client_factory = _client_factory


@pytest.fixture(autouse=True)
def _restore_factory():
    yield
    restore()


PAGE = b"""<html>
<head><title>Docs &amp; Notes</title><style>body{color:red}</style></head>
<body>
<script>track()</script>
<h1>Install</h1>
<p>Run <code>uv sync</code>.</p>
<ul><li>first</li><li>second</li></ul>
</body></html>"""


class TestHtmlToText:
    def test_title_and_text(self):
        title, text = html_to_text(PAGE.decode())
        assert title == "Docs & Notes"  # entity decoded, style excluded
        assert "Install" in text
        assert "uv sync" in text
        assert "color:red" not in text and "track()" not in text

    def test_list_items_get_bullets(self):
        _, text = html_to_text(PAGE.decode())
        assert "\n- first" in text and "\n- second" in text

    def test_malformed_html_never_raises(self):
        title, text = html_to_text("<p>unclosed <div><b>tags everywhere")
        assert "unclosed" in text

    def test_blank_lines_collapse(self):
        _, text = html_to_text("<p>a</p><p>b</p>")
        assert "\n\n\n" not in text


class TestFetch:
    def test_html_page_comes_back_as_prose(self):
        serve(lambda req: httpx.Response(200, content=PAGE,
                                         headers={"content-type": "text/html"}))
        out = fetch("https://docs.example.com/guide")
        assert "source: https://docs.example.com/guide" in out
        assert "title: Docs &amp; Notes".replace("&amp;", "&") in out or \
            "title: Docs & Notes" in out
        assert "Install" in out

    def test_json_is_pretty_printed(self):
        payload = {"ok": True, "items": [1, 2]}
        serve(lambda req: httpx.Response(
            200, headers={"content-type": "application/json"},
            content=json.dumps(payload)))
        out = fetch("https://api.example.com/v1/things")
        assert '"items"' in out and '\n' in out  # indented, not one line

    def test_plain_text_passthrough(self):
        serve(lambda req: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"hello"))
        assert "hello" in fetch("https://example.com/robots.txt")

    def test_http_error_status_is_tool_error(self):
        serve(lambda req: httpx.Response(404, content=b"gone"))
        with pytest.raises(ToolError, match="HTTP 404"):
            fetch("https://example.com/nope")

    def test_binary_content_type_refused(self):
        serve(lambda req: httpx.Response(
            200, headers={"content-type": "image/png"}, content=b"\x89PNG"))
        with pytest.raises(ToolError, match="unsupported content-type"):
            fetch("https://example.com/pic.png")

    def test_non_http_scheme_refused_without_networking(self):
        with pytest.raises(ToolError, match="only http"):
            fetch("file:///etc/passwd")

    def test_connection_failure_is_tool_error(self):
        def boom(request):
            raise httpx.ConnectError("no route")
        serve(boom)
        with pytest.raises(ToolError, match="fetch failed"):
            fetch("https://blackhole.example.com/")

    def test_redirects_followed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/old":
                return httpx.Response(302, headers={
                    "location": "https://example.com/new"})
            return httpx.Response(200, headers={"content-type": "text/plain"},
                                  content=b"arrived")
        serve(handler)
        assert "arrived" in fetch("https://example.com/old")


class TestGating:
    def test_not_read_only_on_purpose(self):
        # network egress is an explicit grant; it must gate like bash
        assert WebFetch.read_only is False

    def test_summary_shows_the_url(self, ctx):
        assert "example.com/x" in WebFetch().summary(
            {"url": "https://example.com/x"}, ctx)

    def test_missing_url_arg(self, ctx):
        with pytest.raises(ToolError, match="missing required argument"):
            WebFetch().run({}, ctx)
