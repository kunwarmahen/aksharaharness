"""The hand-rolled .env loader (no python-dotenv dependency).

.env.example annotates every variable with a trailing '#' comment, so a
new user's FIRST action -- copy it to .env and uncomment a line -- feeds
this parser the annotated form. A regression doesn't fail quietly: it
crashes at startup (int('8192  # default ...')). These tests pin that
copy-paste path, plus the documented contracts: real environment
variables always win, values may contain '=', and the load happens ONCE.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from akshara import config
from akshara.config import (
    _load_dotenv,
    default_context_window,
)


@pytest.fixture(autouse=True)
def _fresh_loader(monkeypatch, tmp_path):
    """Empty cwd + an un-latched loader, so each test parses its own .env.

    The loader writes straight into os.environ; deleting the test keys
    beforehand keeps one test's values from masquerading as "real"
    environment variables in the next.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_dotenv_loaded", False)
    for var in ("AKSHARA_TEST_KEY", "AKSHARA_TEST_WINDOW",
                "OLLAMA_CONTEXT_WINDOW"):
        monkeypatch.delenv(var, raising=False)


def _write_env(text: str) -> None:
    Path(".env").write_text(text)


class TestParsing:
    def test_trailing_comment_is_stripped(self):
        # the exact shape of every annotated line in .env.example
        _write_env(
            "AKSHARA_TEST_WINDOW=8192"
            "                            # default (auto-compaction math)\n"
        )
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_WINDOW"] == "8192"

    def test_hash_inside_quotes_is_value(self):
        _write_env('AKSHARA_TEST_KEY="sk-abc#def"\n')
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_KEY"] == "sk-abc#def"

    def test_glued_hash_in_unquoted_value_is_value(self):
        # only whitespace marks a comment start -- secrets may contain '#'
        _write_env("AKSHARA_TEST_KEY=abc#def\n")
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_KEY"] == "abc#def"

    def test_comment_after_closing_quote_ignored(self):
        _write_env('AKSHARA_TEST_KEY="hello"  # greeting\n')
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_KEY"] == "hello"

    def test_quotes_and_first_equals_split(self):
        _write_env('AKSHARA_TEST_KEY = "a=b=c"\n')
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_KEY"] == "a=b=c"


class TestContract:
    def test_blank_value_is_skipped(self):
        # ``KEY=   `` is template residue from copying .env.example; it
        # must not shadow code fallbacks with ""
        _write_env("AKSHARA_TEST_KEY=   \n")
        _load_dotenv()
        assert "AKSHARA_TEST_KEY" not in os.environ

    def test_real_environment_wins(self, monkeypatch):
        monkeypatch.setenv("AKSHARA_TEST_KEY", "from-shell")
        _write_env("AKSHARA_TEST_KEY=from-file\n")
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_KEY"] == "from-shell"

    def test_loads_once(self):
        _write_env("AKSHARA_TEST_KEY=first\n")
        _load_dotenv()
        _write_env("AKSHARA_TEST_KEY=second\n")
        _load_dotenv()
        assert os.environ["AKSHARA_TEST_KEY"] == "first"

    def test_no_file_no_error(self):
        _load_dotenv()  # must not raise


class TestEndToEnd:
    def test_example_file_line_reaches_context_window(self):
        """The reported crash: uncommented .env.example line -> int()."""
        _write_env(
            "OLLAMA_CONTEXT_WINDOW=8192"
            "                            # default (auto-compaction math)\n"
        )
        _load_dotenv()
        assert default_context_window("ollama") == 8192
