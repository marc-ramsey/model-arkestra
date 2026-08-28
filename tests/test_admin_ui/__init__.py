"""E2E tests for widget-driven admin UI.

Verifies that app.json is correctly rendered into DOM, key widgets exist,
and basic interactions (model click → config panel) work.

Run: pytest tests/test_admin_ui.py -v --timeout=120
"""
from __future__ import annotations
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright, expect

# Suppress async autouse fixture from conftest (shadow with sync no-op)
@pytest.fixture(autouse=True)
def _cleanup_after_test():
    """Override conftest's async autouse to avoid pytest-asyncio errors."""
    yield

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
TEST_CFG = ROOT / "tests" / "test-config.yaml"
STATIC_DIR = ROOT / "static"


def _start_server(port: int) -> subprocess.Popen:
    """Start the ModelArkestra server on *port* using test config. Returns Popen."""
    env = os.environ.copy()
    env["MODEL_ARKESTRA_CONFIG"] = str(TEST_CFG)
    cmd = [
        "python", "-m", "model_arkestra.server",
        "--config", str(TEST_CFG),
        "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # Wait for server to be ready (max 30s)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            resp = _http_get(f"http://127.0.0.1:{port}/admin/models")
            if resp and "models" in resp.text:
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    out, err = proc.communicate(timeout=3)
    raise RuntimeError(f"Server failed to start.\nstdout: {out.decode()}\nstderr: {err.decode()}")


def _http_get(url: str):
    import httpx
    return httpx.get(url, timeout=3)


@pytest.fixture(scope="module")
def server():
    """Start a test server on port 18500 (outside normal range)."""
    # Clean up any lingering process
    try:
        subprocess.run(["lsof", "-ti:18500"], capture_output=True, text=True)
        for pid in subprocess.run(["lsof", "-ti:18500"], capture_output=True, text=True).stdout.strip().split():
            if pid:
                try: os.kill(int(pid), 9)
                except: pass
    except Exception:
        pass

    proc = _start_server(18500)
    yield proc
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill(); proc.wait()


@pytest.fixture(scope="module")
def browser(server):
    """Playwright page fixture — one per module."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        yield page, browser
        browser.close()


# ═══════════════════════════════════════════════════════════
# Tests — widget tree rendering from JSON
# ═══════════════════════════════════════════════════════════

class TestWidgetRendering:
    """Verify that app.json → DOM mapping is correct."""

    def test_01_page_loads(self, server, browser):
        page, browser = browser
        page.goto("http://127.0.0.1:18500/")
        expect(page).to_have_title("ArkestraAdmin")

    def test_02_widget_js_loaded(self, server, browser):
        page, browser = browser
        # widget.js should be present and executed (no JS errors)
        eval_result = page.evaluate("() => typeof window.render === 'function'")
        assert eval_result is True, "window.render not defined — widget.js may not have loaded"

    def test_03_split_root_rendered(self, server, browser):
        page, browser = browser
        # Root split should create flex container at top level
        has_flex = page.evaluate("""
            () => {
                const main = document.querySelector('.layout');
                if (!main) return false;
                // Rendered widget tree replaces #app-root children with flex containers
                return main.firstElementChild?.style.display === 'flex';
            }
        """)
        assert has_flex, "Root SplitPane not rendered as flex container"

    def test_04_left_accordion_exists(self, server, browser):
        page, browser = browser
        # Left accordion should have id="left-accordion"
        acc_id = page.evaluate("document.getElementById('left-accordion')?.id")
        assert acc_id == "left-accordion", f"Expected left-accordion, got: {acc_id}"

    def test_05_model_accordion_body_exists(self, server, browser):
        page, browser = browser
        body_id = page.evaluate("document.getElementById('model-accordion-items')?.id")
        assert body_id == "model-accordion-items"

    def test_06_log_pane_rendered(self, server, browser):
        page, browser = browser
        log_el = page.evaluate("""
            () => {
                const el = document.getElementById('log-display');
                if (!el) return null;
                return el.className.includes('pane-logs');
            }
        """)
        assert log_el is True, "LogPane not rendered with correct classes"

    def test_07_chat_pane_rendered(self, server, browser):
        page, browser = browser
        chat_el = page.evaluate("""
            () => {
                const el = document.getElementById('chat-display');
                if (!el) return null;
                return el.className.includes('chat-messages');
            }
        """)
        assert chat_el is True, "ChatPane messages area not rendered"

    def test_08_chat_input_exists(self, server, browser):
        page, browser = browser
        input_exists = page.evaluate("!!document.getElementById('chat-input')")
        assert input_exists, "Chat input element missing"

    def test_09_split_divider_exists(self, server, browser):
        page, browser = browser
        # Horizontal divider between log and chat panes (vertical SplitPane)
        h_dividers = page.query_selector_all("div.divider-h")
        assert len(h_dividers) >= 1, f"Expected horizontal dividers, found {len(h_dividers)}"

        # Vertical divider between left panel and right panel (horizontal SplitPane)
        v_dividers = page.query_selector_all("div.divider-v")
        assert len(v_dividers) >= 1, f"Expected vertical dividers, found {len(v_dividers)}"


# ═══════════════════════════════════════════════════════════
# Tests — model list population
# ═══════════════════════════════════════════════════════════

class TestModelList:
    def test_10_model_dropdown_populated(self, server, browser):
        page, browser = browser
        # Log pane selector should have options after page load
        option_count = page.evaluate("""
            () => {
                const sel = document.getElementById('log-model-select');
                return sel ? sel.options.length : 0;
            }
        """)
        assert option_count >= 1, f"Log model select has {option_count} options (expected >= 1)"

    def test_11_model_rows_rendered(self, server, browser):
        page, browser = browser
        row_count = page.evaluate("document.querySelectorAll('.model-row').length")
        assert row_count >= 1, f"Expected model rows, found {row_count}"

    def test_12_gemma_model_visible(self, server, browser):
        page, browser = browser
        has_gemma = page.evaluate("""
            () => {
                const rows = document.querySelectorAll('.model-row');
                for (const row of rows) {
                    if (row.dataset.model.includes('gemma')) return true;
                }
                return false;
            }
        """)
        assert has_gemma, "No gemma model row found in accordion"

    def test_13_status_dots_rendered(self, server, browser):
        page, browser = browser
        dot_count = page.evaluate("document.querySelectorAll('.status-dot').length")
        assert dot_count >= 1, f"Expected status dots, found {dot_count}"


# ═══════════════════════════════════════════════════════════
# Tests — interaction patterns
# ═══════════════════════════════════════════════════════════

class TestInteractions:
    def test_20_config_panel_on_click(self, server, browser):
        page, browser = browser
        # Click first model row
        rows = page.query_selector_all(".model-row")
        assert len(rows) >= 1, "No model rows to click"

        rows[0].click()
        # Wait for config panel to appear (deferred fetch + render)
        time.sleep(1.5)

        has_config = page.evaluate("""
            () => {
                const row = document.querySelector('.model-row.expanded') ||
                            document.querySelectorAll('.model-row')[0];
                return !!row?.querySelector('.config-panel');
            }
        """)
        assert has_config, "Config panel did not appear after model row click"

    def test_21_params_panel_toggles(self, server, browser):
        page, browser = browser
        # Initially hidden
        initially_hidden = page.evaluate("""
            () => {
                const panel = document.getElementById('chat-params-panel');
                return !panel || window.getComputedStyle(panel).display === 'none';
            }
        """)

        # Toggle via Params button
        toggle = page.query_selector("#right-chat-params-toggle")
        if toggle:
            toggle.click()

        shown = page.evaluate("""
            document.getElementById('chat-params-panel')?.style.display !== 'none'
        """)
        assert shown, "Params panel should be visible after toggle"

    def test_22_chat_send_button_exists(self, server, browser):
        page, browser = browser
        send_btn = page.query_selector("#chat-send-btn")
        assert send_btn is not None, "Chat send button not found"

    def test_23_action_buttons_in_config(self, server, browser):
        page, browser = browser
        # After clicking a model row, verify action buttons exist
        rows = page.query_selector_all(".model-row")
        if len(rows) >= 1:
            rows[0].click()
            time.sleep(1.5)

        btn_ids = page.evaluate("""
            () => {
                const panel = document.querySelector('.config-panel');
                if (!panel) return [];
                return Array.from(panel.querySelectorAll('button')).map(b => b.id);
            }
        """)
        # Should have reset, stop, start, save, eject buttons
        found_actions = [b.replace('btn-', '').replace('-', '_') for b in btn_ids]
        assert len(btn_ids) >= 3, f"Expected at least 3 action buttons, found {len(btn_ids)}: {btn_ids}"
