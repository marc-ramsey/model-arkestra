"""E2E tests for the context-tab chat UI.

Verifies the page loads, the three modules (api/store/context) are defined,
the tab strip renders, and opening a context renders the pane.

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

# Suppress async autouse fixture from conftest (shadow with sync no-op)
@pytest.fixture(autouse=True)
def _cleanup_after_test():
    yield

ROOT = Path(__file__).parent.parent.parent
TEST_CFG = ROOT / "tests" / "test-config.yaml"
PORT = 18501


def _start_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["MODEL_ARKESTRA_CONFIG"] = str(TEST_CFG)
    proc = subprocess.Popen(
        ["python", "-m", "model_arkestra.server",
         "--config", str(TEST_CFG),
         "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/admin/models", timeout=3)
            if r.status_code == 200 and "models" in r.text:
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    out, err = proc.communicate(timeout=3)
    raise RuntimeError(f"Server failed to start.\nstdout: {out.decode()}\nstderr: {err.decode()}")


@pytest.fixture(scope="module")
def server():
    for pid in subprocess.run(["lsof", "-ti:%d" % PORT], capture_output=True,
                              text=True).stdout.strip().split():
        try:
            os.kill(int(pid), 9)
        except (ValueError, ProcessLookupError, OSError):
            pass

    proc = _start_server(PORT)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def page(server):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        p = browser.new_page()
        yield p
        browser.close()


class TestPageLoads:
    def test_01_title(self, page):
        page.goto(f"http://127.0.0.1:{PORT}/")
        expect(page).to_have_title("Arkestra")

    def test_02_modules_defined(self, page):
        assert page.evaluate("() => typeof window.api?.request === 'function'")
        assert page.evaluate("() => typeof window.store?.contexts !== 'undefined'")
        assert page.evaluate("() => typeof window.context?.newContext === 'function'")

    def test_03_tabstrip_rendered(self, page):
        # Empty strip is zero-height (not "visible"), so assert attachment;
        # the chrome buttons do have size and are visible.
        expect(page.locator("#tab-strip")).to_be_attached()
        expect(page.locator("#btn-new")).to_be_visible()
        expect(page.locator("#btn-reopen")).to_be_visible()

    def test_04_jinja_vars_substituted(self, page):
        html = page.content()
        assert "{{BASE_URL}}" not in html
        assert "{{API_KEY}}" not in html


class TestContexts:
    def test_05_new_context_opens_pane(self, page):
        page.click("#btn-new")
        expect(page.locator("#chat-input")).to_be_visible()
        expect(page.locator("#tab-strip .tab")).to_have_count(1)

    def test_06_model_select_populated(self, page):
        # test-config.yaml defines cached models
        count = page.evaluate("() => document.getElementById('model-select').options.length")
        assert count >= 1

    def test_07_close_removes_tab(self, page):
        page.click("#tab-strip .tab .tab-close")
        expect(page.locator("#tab-strip .tab")).to_have_count(0)
        expect(page.locator("#pane-empty")).to_be_visible()
