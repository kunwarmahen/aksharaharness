"""browser_* tools: refs, lifecycle, gating, conditional registration.

The suite stays offline-green WITHOUT playwright installed: tests
inject a FakePage implementing the sliver of the page API the session
touches, so run() exercises the real snapshot/ref/selector logic. The
launch paths are pinned by stubbing the playwright import itself.
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

import akshara.cli.main as cli_main
from akshara.errors import ToolError
from akshara.tools import BrowserClick, BrowserClose, BrowserFill, \
    BrowserOpen, default_registry
from akshara.tools.base import ToolContext
from akshara.tools.browser import MAX_ELEMENTS, BrowserSession, \
    run_login_session

URL = "https://fake.local/"


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# FakePage: the contract under test is that the session talks to a page
# through goto/url/title/evaluate/locator/wait_* ONLY, and that evaluate()
# receives THE tagging script (the thing real clicks resolve against).
# ---------------------------------------------------------------------------


class FakeLocator:
    def __init__(self, page: "FakePage", selector: str) -> None:
        self.page = page
        self.selector = selector

    def click(self, timeout=None) -> None:
        self.page.clicks.append(self.selector)
        self.page.threads.add(threading.get_ident())
        if self.page.on_click:
            self.page.on_click()

    def fill(self, text, timeout=None) -> None:
        self.page.fills.append((self.selector, text))

    def select_option(self, label=None, timeout=None) -> None:
        self.page.selects.append((self.selector, label))


class FakePage:
    def __init__(self, *, title="Fake Page", url=URL, text="hello world",
                 elements=None, on_click=None) -> None:
        self.title_value = title
        self.url = url
        self.text = text
        self.elements = elements if elements is not None else [
            {"ref": "e1", "kind": "link", "label": "Home"},
            {"ref": "e2", "kind": "textbox", "label": "Search"},
        ]
        self.on_click = on_click
        self.gotos: list[str] = []
        self.clicks: list[str] = []
        self.fills: list = []
        self.selects: list = []
        self.closed = False
        self.threads: set[int] = set()  # worker threads seen at the page

    def title(self):
        return self.title_value

    def goto(self, url, **kwargs):
        self.threads.add(threading.get_ident())
        self.gotos.append(url)

    def evaluate(self, script):
        assert "data-akshara-ref" in script  # snapshots must tag targets
        self.threads.add(threading.get_ident())
        return {"text": self.text, "elements": self.elements}

    def locator(self, selector):
        return FakeLocator(self, selector)

    def wait_for_timeout(self, ms):
        pass

    def wait_for_load_state(self, state, timeout=None):
        pass

    def close(self):
        self.closed = True


def make_session(page: FakePage) -> BrowserSession:
    """A session that believes it already launched, around a fake page."""
    session = BrowserSession()
    session._pw = object()  # truthy is all _open checks before touching it
    session._browser = object()
    session._page = page
    return session


ELEMENTS_LINE = "interactive elements (pass a ref to browser_click/browser_fill):"


class TestSnapshots:
    def test_open_returns_header_text_and_elements(self):
        out = make_session(FakePage()).open(URL)
        assert f"title: Fake Page\nsource: {URL}" in out
        assert "hello world" in out
        assert "[e1] link Home" in out
        assert "[e2] textbox Search" in out
        assert ELEMENTS_LINE in out

    def test_unlabeled_element_omits_the_label(self):
        page = FakePage(elements=[{"ref": "e3", "kind": "button",
                                   "label": ""}])
        out = make_session(page).open(URL)
        assert out.endswith("[e3] button")  # no stray space, no empty quotes

    def test_empty_page_says_so_without_elements_section(self):
        page = FakePage(text="", elements=[])
        out = make_session(page).open(URL)
        assert "[no text content]" in out
        assert "(no interactive elements found)" in out

    def test_long_text_clips_head_and_tail(self):
        out = make_session(FakePage(text="x" * 30_000)).open(URL)
        assert "[... 22000 chars omitted ...]" in out

    def test_element_list_caps_with_a_count(self):
        many = [{"ref": f"e{i}", "kind": "link", "label": f"L{i}"}
                for i in range(1, 71)]
        out = make_session(FakePage(elements=many)).open(URL)
        assert f"[showing {MAX_ELEMENTS} of 70 elements]" in out
        assert "[e60]" in out and "[e61]" not in out

    def test_open_without_url_needs_a_page(self):
        with pytest.raises(ToolError, match="no page open -- pass a url"):
            BrowserSession().open("")

    def test_second_open_navigates_the_same_page(self):
        page = FakePage()
        session = make_session(page)
        session.open(URL)
        session.open(f"{URL}page2")
        assert page.gotos == [URL, f"{URL}page2"]

    def test_non_http_scheme_refused_before_launching(self):
        with pytest.raises(ToolError, match="only http"):
            BrowserSession().open("file:///etc/passwd")


class TestRefs:
    def test_click_targets_the_ref_attribute_and_refreshes(self):
        def navigate():
            page.text = "results for query"
            page.elements = [{"ref": "e1", "kind": "link", "label": "Next"}]
        page = FakePage(on_click=navigate)
        session = make_session(page)
        session.open(URL)
        out = session.click("e1")
        assert page.clicks == ['[data-akshara-ref="e1"]']
        assert "results for query" in out  # refreshed snapshot came back
        assert '[e1] link Next' in out

    def test_fill_types_into_a_textbox_ref(self):
        page = FakePage()
        session = make_session(page)
        session.open(URL)
        out = session.fill("e2", "wire adapters")
        assert page.fills == [('[data-akshara-ref="e2"]', "wire adapters")]
        assert ELEMENTS_LINE in out  # refreshed page returned

    def test_fill_on_a_dropdown_selects_the_option_by_label(self):
        page = FakePage(elements=[{"ref": "e5", "kind": "select",
                                   "label": "Category"}])
        session = make_session(page)
        session.open(URL)
        session.fill("e5", "books")
        assert page.selects == [('[data-akshara-ref="e5"]', "books")]

    def test_fill_on_a_button_is_refused_with_guidance(self):
        page = FakePage(elements=[{"ref": "e4", "kind": "button",
                                   "label": "Go"}])
        session = make_session(page)
        session.open(URL)
        with pytest.raises(ToolError, match=r"e4 is a button"):
            session.fill("e4", "nope")

    def test_unknown_ref_lists_known_refs(self):
        session = make_session(FakePage())
        session.open(URL)
        with pytest.raises(ToolError, match="known: e1, e2"):
            session.click("e9")

    def test_action_without_a_page_names_the_door(self):
        with pytest.raises(ToolError, match="browser_open"):
            BrowserSession().click("e1")
        with pytest.raises(ToolError, match="browser_open"):
            BrowserSession().fill("e1", "x")


class TestLifecycle:
    def test_close_reports_and_resets(self):
        page = FakePage()
        session = make_session(page)
        session.open(URL)
        assert session.close() is True
        assert page.closed
        assert session._pw is None and session._page is None

    def test_close_when_never_opened_is_a_noop(self):
        assert BrowserSession().close() is False

    def test_reopen_after_close_re_arms_launch(self, monkeypatch):
        session = make_session(FakePage())
        session.open(URL)
        session.close()
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        with pytest.raises(ToolError, match=r"aksharaharness\[browse\]"):
            session.open(URL)  # launch door re-arms after close

    def test_all_traffic_on_one_worker_thread(self):
        # Playwright's sync API binds to its starting thread; whichever
        # loop twin calls us (asyncio.to_thread varies), the SESSION must
        # funnel every operation onto one dedicated worker.
        page = FakePage()
        session = make_session(page)
        session.open(URL)
        session.click("e1")
        session.fill("e2", "q")
        session.close()
        assert len(page.threads) == 1
        assert threading.get_ident() not in page.threads  # never the caller


class TestLaunchPaths:
    def test_missing_playwright_names_the_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        with pytest.raises(ToolError,
                           match=r"aksharaharness\[browse\]"):
            BrowserSession().open(URL)

    def test_chromium_binary_missing_names_playwright_install(self,
                                                              monkeypatch):
        class Boom:
            def start(self):
                raise RuntimeError("Executable doesn't exist at .../chromium")

        stub = types.ModuleType("playwright.sync_api")
        stub.sync_playwright = lambda: Boom()
        monkeypatch.setitem(sys.modules, "playwright",
                            types.ModuleType("playwright"))
        monkeypatch.setitem(sys.modules, "playwright.sync_api", stub)
        with pytest.raises(ToolError, match="playwright install chromium"):
            BrowserSession().open(URL)


class FakeContext:
    """The sliver of BrowserContext the session touches: pages list,
    new_page, wait_for_event('close'), close."""

    def __init__(self, pages=()) -> None:
        self.pages = list(pages)
        self.closed = False
        self.waited_for: str | None = None

    def new_page(self) -> FakePage:
        page = FakePage()
        self.pages.append(page)
        return page

    def wait_for_event(self, event: str) -> None:
        self.waited_for = event

    def close(self) -> None:
        self.closed = True


class FakeChromium:
    """Records WHICH launch door was used -- launch() vs
    launch_persistent_context() is the whole ephemeral/persistent
    contract."""

    def __init__(self, browser=None, context=None) -> None:
        self.launch_calls: list[dict] = []
        self.persistent_calls: list[dict] = []
        self._browser = browser
        self._context = context

    def launch(self, **kwargs):
        self.launch_calls.append(kwargs)
        return self._browser

    def launch_persistent_context(self, **kwargs):
        self.persistent_calls.append(kwargs)
        return self._context


def install_fake_playwright(monkeypatch, chromium: FakeChromium) -> list:
    """Swap the playwright import for a recorder; returns the started
    playwright instances (so tests can pin that stop() always runs)."""
    started: list = []

    class FakePW:
        def __init__(self) -> None:
            self.chromium = chromium
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True
            started.append(self)

    stub = types.ModuleType("playwright.sync_api")
    stub.sync_playwright = lambda: types.SimpleNamespace(start=FakePW)
    monkeypatch.setitem(sys.modules, "playwright",
                        types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", stub)
    return started


class TestPersistentProfile:
    """$AKSHARA_BROWSER_PROFILE set => launch_persistent_context on that
    dir; unset => today's plain launch. Same four verbs either way."""

    def test_ephemeral_session_launches_a_plain_browser(self, monkeypatch):
        ephemeral = types.SimpleNamespace(new_page=FakePage)
        chromium = FakeChromium(browser=ephemeral)
        install_fake_playwright(monkeypatch, chromium)
        session = BrowserSession(profile=None)
        session.open(URL)
        assert chromium.launch_calls == [{"headless": True}]
        assert chromium.persistent_calls == []

    def test_profile_launches_a_persistent_context_on_the_dir(
            self, monkeypatch, tmp_path):
        chromium = FakeChromium(context=FakeContext())
        install_fake_playwright(monkeypatch, chromium)
        out = BrowserSession(profile=tmp_path).open(URL)
        (kwargs,) = chromium.persistent_calls
        assert kwargs == {"user_data_dir": str(tmp_path), "headless": True}
        assert chromium.launch_calls == []
        assert "hello world" in out  # traffic flows through the context page

    def test_existing_context_page_is_reused_not_duplicated(
            self, monkeypatch):
        page = FakePage()
        context = FakeContext(pages=[page])  # persistent contexts ship one
        chromium = FakeChromium(context=context)
        install_fake_playwright(monkeypatch, chromium)
        session = BrowserSession(profile=Path("/tmp/never-made"))
        session.open(URL)
        session.open(f"{URL}page2")
        assert len(context.pages) == 1
        assert page.gotos == [URL, f"{URL}page2"]

    def test_close_shuts_down_the_context(self, monkeypatch):
        context = FakeContext()
        chromium = FakeChromium(context=context)
        install_fake_playwright(monkeypatch, chromium)
        session = BrowserSession(profile=Path("/tmp/never-made"))
        session.open(URL)
        assert session.close() is True
        assert context.closed

    def test_default_sentinel_reads_the_env_once(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AKSHARA_BROWSER_PROFILE", str(tmp_path))
        assert BrowserSession()._profile == tmp_path
        monkeypatch.delenv("AKSHARA_BROWSER_PROFILE")
        assert BrowserSession()._profile is None  # blank/unset => ephemeral

    def test_browser_tools_wire_the_env_knob_through_one_session(
            self, monkeypatch):
        monkeypatch.setattr("akshara.tools.browser.find_spec",
                            lambda name: True)
        monkeypatch.setenv("AKSHARA_BROWSER_PROFILE", "~/akshara-prof")
        registry = default_registry()
        names = ("browser_open", "browser_click", "browser_fill",
                 "browser_close")
        sessions = {registry.get(n).browser for n in names}
        assert len(sessions) == 1
        (session,) = sessions
        assert session._profile == Path.home() / "akshara-prof"


class TestLoginSession:
    """run_login_session: the headed half of persistence -- a human beats
    the login wall once; every later headless run inherits it."""

    def test_missing_extra_names_the_install(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        with pytest.raises(ToolError, match=r"aksharaharness\[browse\]"):
            run_login_session(tmp_path)

    def test_non_http_url_refused_before_any_launch(self, tmp_path):
        # no playwright stub installed -- refusal must not depend on it
        with pytest.raises(ToolError, match="only http"):
            run_login_session(tmp_path, "file:///etc/passwd")

    def test_opens_headed_on_the_profile_and_waits_for_the_window(
            self, monkeypatch, tmp_path):
        context = FakeContext()
        chromium = FakeChromium(context=context)
        started = install_fake_playwright(monkeypatch, chromium)
        run_login_session(tmp_path, URL)
        (kwargs,) = chromium.persistent_calls
        assert kwargs == {"user_data_dir": str(tmp_path), "headless": False}
        assert context.pages[0].gotos == [URL]
        assert context.waited_for == "close"  # the window IS the progress bar
        assert started and started[0].stopped

    def test_stop_runs_even_when_waiting_explodes(self, monkeypatch,
                                                  tmp_path):
        context = FakeContext()

        def boom(event):
            raise RuntimeError("boom")

        context.wait_for_event = boom
        chromium = FakeChromium(context=context)
        started = install_fake_playwright(monkeypatch, chromium)
        with pytest.raises(ToolError,
                           match="could not open the login window"):
            run_login_session(tmp_path)
        assert started and started[0].stopped


class TestBrowseLoginFlag:
    """CLI wiring: its own mode, dispatched before provider resolution --
    no model, no API key, on purpose."""

    def test_without_a_profile_names_the_env_var(self, monkeypatch, capsys):
        monkeypatch.delenv("AKSHARA_BROWSER_PROFILE", raising=False)
        assert cli_main.main(["--browse-login", URL]) == 2
        assert "AKSHARA_BROWSER_PROFILE" in capsys.readouterr().err

    def test_with_a_profile_opens_and_reports_saved(self, monkeypatch,
                                                    tmp_path, capsys):
        monkeypatch.setenv("AKSHARA_BROWSER_PROFILE", str(tmp_path))
        seen: dict = {}

        def fake_login(profile, url=None):
            seen["args"] = (profile, url)

        monkeypatch.setattr("akshara.tools.browser.run_login_session",
                            fake_login)
        assert cli_main.main(["--browse-login", URL]) == 0
        assert seen["args"] == (tmp_path, URL)
        assert "profile saved" in capsys.readouterr().out

    def test_combining_with_other_modes_is_refused(self):
        assert cli_main.main(["--browse-login", URL,
                              "--prompt", "hi"]) == 2


class TestGating:
    """Network egress family: every verb gates, like web_fetch."""

    @pytest.mark.parametrize("tool_cls",
                             [BrowserOpen, BrowserClick, BrowserFill,
                              BrowserClose])
    def test_not_read_only(self, tool_cls):
        assert tool_cls.read_only is False

    def test_summaries_show_what_will_happen(self, ctx):
        browser = BrowserSession()
        assert URL in BrowserOpen(browser).summary({"url": URL}, ctx)
        assert "e3" in BrowserClick(browser).summary({"ref": "e3"}, ctx)
        assert 'fill e2' in BrowserFill(browser).summary(
            {"ref": "e2", "text": "query"}, ctx)
        assert "close" in BrowserClose(browser).summary({}, ctx)

    def test_missing_required_args_are_model_readable(self, ctx):
        browser = BrowserSession()
        with pytest.raises(ToolError, match="missing required argument"):
            BrowserClick(browser).run({}, ctx)
        with pytest.raises(ToolError, match="missing required argument"):
            BrowserFill(browser).run({"ref": "e1"}, ctx)


class TestRegistration:
    """The [browse] extra is its own opt-in: no playwright, no tools --
    and the tool count the selection threshold keys on stays put."""

    def test_without_extra_registry_stays_at_sixteen(self, monkeypatch):
        monkeypatch.setattr("akshara.tools.browser.find_spec",
                            lambda name: None)
        registry = default_registry()
        assert len(registry) == 16
        assert "browser_open" not in registry

    def test_with_extra_four_tools_share_one_session(self, monkeypatch):
        monkeypatch.setattr("akshara.tools.browser.find_spec",
                            lambda name: True)
        registry = default_registry()
        names = ("browser_open", "browser_click", "browser_fill",
                 "browser_close")
        assert len(registry) == 20
        for name in names:
            assert name in registry
        sessions = {registry.get(n).browser for n in names}
        assert len(sessions) == 1  # one browser behind all four verbs
