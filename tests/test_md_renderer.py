"""Markdown renderer tests: the web page's md.js, executed by node.

The renderer is pure (string -> string) so the whole security-sensitive
surface can be pinned offline -- especially escaping: model output must
never become live HTML. Skips silently when node isn't installed; the
page itself never needs it (the suite's only JS-executing tests).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MD_JS = Path(__file__).resolve().parent.parent / "src/akshara/web/static/md.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not installed")

DRIVER = """
// `node -e script a b` gives argv = [node, a, b] -- no script entry
const { renderMarkdown } = require(process.argv[1]);
const out = {};
for (const [name, input] of Object.entries(JSON.parse(process.argv[2])))
  out[name] = renderMarkdown(input);
process.stdout.write(JSON.stringify(out));
"""


def render(**cases: str) -> list[str]:
    """Render each named case; returns outputs in the order passed."""
    proc = subprocess.run(
        ["node", "-e", DRIVER, str(MD_JS), json.dumps(cases)],
        capture_output=True, text=True, timeout=30, check=True)
    return list(json.loads(proc.stdout).values())


def test_html_is_escaped_never_executed():
    (out,) = render(x="<script>alert(1)</script>\n\n"
                      '<img src=x onerror="alert(1)">')
    assert "<script>" not in out
    assert "<img" not in out
    assert "&lt;script&gt;" in out


def test_emphasis_code_and_strike():
    (out,) = render(x="**bold** and *soft* and ~~gone~~ and `a*b*`")
    assert "<strong>bold</strong>" in out
    assert "<em>soft</em>" in out
    assert "<del>gone</del>" in out
    assert "<code>a*b*</code>" in out          # code span beats emphasis


def test_headings_map_two_levels_down():
    (out,) = render(x="# top\n\n## section\n\n### subsection")
    assert "<h3>top</h3>" in out
    assert "<h4>section</h4>" in out
    assert "<h5>subsection</h5>" in out


def test_fenced_code_block_escapes_and_keeps_newlines():
    (out,) = render(x="```python\nif x < 1 && y > 2:\n    run()\n```")
    assert 'class="lang-python"' in out
    assert "if x &lt; 1 &amp;&amp; y &gt; 2:" in out
    assert "<pre" in out and "</code></pre>" in out


def test_pipe_table_with_inline_formatting():
    (out,) = render(x="| k | v |\n|---|---|\n| **rss** | 10.4 GB |\n"
                      "| cpu | 274% |")
    assert "<table><thead><tr><th>k</th><th>v</th></tr></thead>" in out
    assert "<td><strong>rss</strong></td>" in out
    assert "<td>10.4 GB</td>" in out
    assert "<tr><td>cpu</td><td>274%</td></tr>" in out


def test_nested_lists_recurse_by_indentation():
    (out,) = render(x="- top\n- mid\n  - deep 1\n  - deep 2\n- tail")
    assert "<ul><li>top</li><li>mid<ul><li>deep 1</li>" \
           "<li>deep 2</li></ul></li><li>tail</li></ul>" in out


def test_ordered_list():
    (out,) = render(x="1. first\n2. second")
    assert "<ol><li>first</li><li>second</li></ol>" in out


def test_safe_links_only():
    (out,) = render(x="[ok](https://example.com/a?b=1&c=2) "
                      "[bad](javascript:alert(1))")
    assert '<a href="https://example.com/a?b=1&amp;c=2"' in out
    assert "javascript:" not in out.replace("href=", "", 0) \
        or '<a href="javascript' not in out
    assert "[bad](javascript:alert(1))" in out   # degraded to plain text


def test_paragraph_single_newlines_become_breaks():
    (out,) = render(x="line one\nline two")
    assert "<p>line one<br>line two</p>" in out


def test_blockquote_and_rule():
    (out,) = render(x="> quoted words\n\n---\n\nafter")
    assert "<blockquote><p>quoted words</p></blockquote>" in out
    assert "<hr>" in out
