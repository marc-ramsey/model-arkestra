"""E2E tests for the admin UI (non-admin chat client).

Verifies the served page: Jinja vars rendered, core elements present,
JS globals loaded, context persistence via IndexedDB, model dropdown.

Run: pytest tests/test_admin_ui -v --timeout=120
"""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).parent.parent.parent
TEST_CFG = ROOT / "tests" / "test-config.yaml"
PORT = 18500


@pytest.fixture(scope="module")
def server():
    """Start the ModelArkestra server on port 18500."""
    for pid in subprocess.run(["lsof", "-ti:%d" % PORT], capture_output=True, text=True).stdout.strip().split():
        try: os.kill(int(pid), 9)
        except (ValueError, ProcessLookupError, OSError):
            pass

    env = os.environ.copy()
    cmd = ["python", "-m", "model_arkestra.server", "--config", str(TEST_CFG),
           "--url", f"http://127.0.0.1:{PORT}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    deadline = time.time() + 30
    ok = False
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/admin/models", timeout=3)
            if r.status_code == 200 and "models" in r.text:
                ok = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ok:
        proc.kill()
        out, err = proc.communicate(timeout=3)
        raise RuntimeError(f"Server failed to start.\n{out.decode()}\n{err.decode()}")
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def browser(server):
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        page = b.new_page()
        yield page, b
        b.close()


# ═══════════════════════════════════════════════════════════
# Page + Jinja rendering
# ═══════════════════════════════════════════════════════════

class TestPage:
    def test_page_loads(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        expect(page).to_have_title("Arkestra")

    def test_jinja_vars_rendered(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        leftover = page.evaluate("document.body.innerHTML.match(/\\{\\{(API_KEY|BASE_URL|ADMIN_KEY)\\}\\)/)")
        assert leftover is None, f"Unrendered Jinja var: {leftover}"
        # The meta tag holds the API key (empty or set)
        meta = page.evaluate("document.querySelector('meta[name=\"arkestra-api-key\"]')?.content")
        assert meta is not None, "arkestra-api-key meta tag missing"

    def test_no_admin_key_in_html(self, browser):
        """Admin key is never rendered into the page (admin-only page later)."""
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        assert "{{ADMIN_KEY}}" not in page.content()


# ═══════════════════════════════════════════════════════════
# Core elements
# ═══════════════════════════════════════════════════════════

class TestElements:
    def test_core_elements_present(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        for sel in ("#tabs", "#new-tab", "#reopen-tab", "#model-select",
                    "#banner", "#msgs", "#input", "#send"):
            assert page.query_selector(sel) is not None, f"Missing {sel}"

    def test_js_globals_loaded(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        ok = page.evaluate("""() =>
            typeof window.api === 'object' &&
            typeof window.store === 'object' &&
            typeof window.contexts !== 'undefined' &&
            typeof window.db !== 'undefined'""")
        assert ok is True, "JS modules failed to load"

    def test_model_dropdown_populated(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        expect(page.locator("#model-select")).to_have_count(0)  # hidden until a context is open
        # Open a context via the + button
        page.click("#new-tab")
        count = page.evaluate("document.querySelectorAll('#model-select option').length")
        assert count >= 1, f"Model dropdown has {count} options"


# ═══════════════════════════════════════════════════════════
# Context lifecycle
# ═══════════════════════════════════════════════════════════

class TestContexts:
    def test_new_context_creates_tab(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        page.click("#new-tab")
        count = page.evaluate("document.querySelectorAll('.tab').length")
        assert count == 1, f"Expected 1 tab, got {count}"
        label = page.evaluate("document.querySelector('.tab .tab-name')?.textContent")
        assert label == "untitled", f"Expected 'untitled', got {label!r}"

    def test_close_tab_removes(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        page.click("#new-tab")
        page.click(".tab .tab-x")
        count = page.evaluate("document.querySelectorAll('.tab').length")
        assert count == 0, f"Expected 0 tabs after close, got {count}"

    def test_conversation_persisted_to_idb(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        page.click("#new-tab")
        # Seed a context directly through the db (simulates saved history)
        saved = page.evaluate("""async () => {
            const id = await window.contexts.create('test-model');
            const ctx = window.store.contexts.get(id);
            ctx.history.push({role: 'user', content: 'hello'});
            ctx.history.push({role: 'assistant', content: 'hi'});
            ctx.name = 'hello conversation';
            await window.db.save(ctx);
            return id;
        }""")
        loaded = page.evaluate("""async () => {
            const rec = await window.db.load(%s);
            return rec?.history?.length;
        }""" % saved)
        assert loaded == 3, f"Expected 3 messages (system+2), got {loaded}"

    def test_reopen_lists_saved(self, browser):
        page, _ = browser
        page.goto(f"http://127.0.0.1:{PORT}/")
        # Save one, then open the reopen list
        page.evaluate("""async () => {
            const id = await window.contexts.create('test-model');
            const ctx = window.store.contexts.get(id);
            ctx.name = 'saved-ctx';
            await window.db.save(ctx);
        }""")
        page.click("#reopen-tab")
        text = page.evaluate("document.querySelector('#reopen-menu')?.textContent")
        assert "saved-ctx" in (text or ""), "Saved context not listed in reopen menu"
