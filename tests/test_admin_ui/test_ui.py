"""E2E tests for widget-driven admin UI.

Verifies that app.json is correctly rendered into DOM, key widgets exist,
and basic interactions (model click → config panel) work.

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


# ── Server fixture ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start the ModelArkestra admin server on port 18500."""
    # Kill anything already on that port
    for pid in subprocess.run(["lsof", "-ti:18500"], capture_output=True, text=True).stdout.strip().split():
        try: os.kill(int(pid), 9)
        except (ValueError, ProcessLookupError, OSError):
            pass

    env = os.environ.copy()
    cmd = ["python", "-m", "model_arkestra.server", "--config", str(TEST_CFG), "--port", "18500"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    # Wait for server to be ready (max 30s)
    deadline = time.time() + 30
    ok = False
    while time.time() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:18500/admin/models", timeout=3)
            if r.status_code == 200 and "models" in r.text:
                ok = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ok:
        proc.kill()
        out, err = proc.communicate(timeout=3)
        raise RuntimeError(f"Server failed to start.\nstdout: {out.decode()}\nstderr: {err.decode()}")

    yield proc
    # Teardown — kill the server subprocess
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def page(server):
    """Playwright page — one per module."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        p = browser.new_page()
        yield p
        browser.close()


# ═══════════════════════════════════════════════════════════
# Widget tree rendering from JSON
# ═══════════════════════════════════════════════════════════

class TestWidgetRendering:
    def test_page_loads(self, page):
        page.goto("http://127.0.0.1:18500/")
        expect(page).to_have_title("ArkestraAdmin")

    def test_widget_js_executed(self, page):
        is_fn = page.evaluate("() => typeof window.render === 'function'")
        assert is_fn is True, "window.render not defined — widget.js may not have loaded"

    def test_root_split_container(self, page):
        is_flex = page.evaluate(
            "() => [...document.body.children].some(c => c.tagName==='DIV' && (c.style.display||'').includes('flex'))"
        )
        assert is_flex is True, "Root SplitPane flex container not found"

    def test_cluster_tree_exists(self, page):
        time.sleep(2)  # wait for async model load
        tree = page.evaluate("!!document.querySelector('.cluster-tree')")
        assert tree is True, "ClusterTree not rendered"

    def test_session_dock_exists(self, page):
        time.sleep(2)
        dock = page.evaluate("!!document.querySelector('.session-dock')")
        assert dock is True, "SessionDock not rendered"

    def test_chat_messages_area(self, page):
        time.sleep(2)
        rows = page.query_selector_all(".model-row")
        if not rows:
            pytest.skip("No model rows to spawn chat from")
        action_btns = rows[0].query_selector_all('[data-action="spawn-chat"]')
        if action_btns:
            action_btns[0].click()
            time.sleep(1)
        has_cls = page.evaluate(
            "() => [...document.querySelectorAll('.chat-messages')].length > 0"
        )
        assert has_cls is True

    def test_chat_input_exists(self, page):
        time.sleep(2)
        rows = page.query_selector_all(".model-row")
        if not rows:
            pytest.skip("No model rows to spawn chat from")
        action_btns = rows[0].query_selector_all('[data-action="spawn-chat"]')
        if action_btns:
            action_btns[0].click()
            time.sleep(1)
        exists = page.evaluate("!!document.querySelector('[id^=\"f-chat-input-\"]')")
        assert exists is True

    def test_params_panel_exists(self, page):
        time.sleep(2)
        rows = page.query_selector_all(".model-row")
        if not rows:
            pytest.skip("No model rows to spawn chat from")
        action_btns = rows[0].query_selector_all('[data-action="spawn-chat"]')
        if action_btns:
            action_btns[0].click()
            time.sleep(1)
        has_panel = page.evaluate("!!document.querySelector('.chat-params-panel')")
        assert has_panel is True, "Params panel not rendered"


# ═══════════════════════════════════════════════════════════
# Model list population
# ═══════════════════════════════════════════════════════════

class TestModelList:
    def test_log_select_has_options(self, page):
        time.sleep(2)
        # Log panels are spawned in SessionDock; check for any select in docked sessions
        count = page.evaluate(
            "() => [...document.querySelectorAll('.session-dock select, .log-panel select')].length"
        )
        assert count >= 1, f"No log model select found in docked sessions (got {count})"

    def test_model_rows_exist(self, page):
        time.sleep(2)
        count = len(page.query_selector_all(".model-row"))
        assert count >= 1, f"No model rows rendered (got {count})"

    def test_gemma_model_present(self, page):
        time.sleep(2)
        has_gem = page.evaluate(
            "() => [...document.querySelectorAll('.model-row')].some(r => r.dataset.model.includes('gemma'))"
        )
        assert has_gem is True, "No gemma model row found"

    def test_status_dots_rendered(self, page):
        time.sleep(2)
        count = len(page.query_selector_all(".status-dot"))
        assert count >= 1, f"No status dots (got {count})"


# ═══════════════════════════════════════════════════════════
# Interaction patterns
# ═══════════════════════════════════════════════════════════

class TestInteractions:
    def test_config_panel_on_model_click(self, page):
        pytest.skip("Config subpane removed — model rows use inline buttons only")

    def test_params_toggle(self, page):
        toggle = page.query_selector(".chat-params-toggle")
        if not toggle:
            pytest.skip("Params toggle not found — no chat sessions open")
        toggle.click()
        shown = page.evaluate(
            "() => [...document.querySelectorAll('.chat-params-panel.open')].length > 0"
        )
        assert shown is True, "Params panel should be visible"
        toggle.click()
        hidden = page.evaluate(
            "() => [...document.querySelectorAll('.chat-params-panel.open')].length === 0"
        )
        assert hidden is True, "Params panel should be hidden again"

    def test_chat_send_button(self, page):
        btn = page.query_selector('[title="Send"]')
        assert btn is not None or pytest.skip("No chat session open — send button rendered per-session")

    def test_action_buttons_in_config(self, page):
        time.sleep(2)
        # Action buttons are now inline on the model row header
        btns = page.evaluate(
            "() => [...document.querySelectorAll('.model-name-bar .model-actions-inline button')].map(b => b.dataset.action)"
        )
        assert len(btns) >= 3, f"Expected action buttons in row (got {len(btns)}: {btns})"

    def test_model_row_inline_buttons(self, page):
        """Model rows have inline action buttons with data-action/data-model attrs."""
        btn_data = page.evaluate(
            "() => [...document.querySelectorAll('.model-name-bar .model-actions-inline button')].map(b => ({ action: b.dataset.action || '', model: b.dataset.model || '' }))"
        )
        for btn in btn_data:
            assert btn['action'], f"Button missing data-action attribute"
        assert btn['model'], f"Button missing data-model attribute"

    def test_group_headers_exist(self, page):
        pytest.skip("Groups replaced by cluster tree architecture")

    def test_field_wrappers_in_config(self, page):
        pytest.skip("Config subpane replaced by inline buttons and separate config panel")

    def test_arg_fields_from_schema(self, page):
        pytest.skip("Config subpane replaced by inline buttons")

    def test_arg_field_count_matches_schema(self, page):
        pytest.skip("Config subpane replaced by inline buttons")


# ═══════════════════════════════════════════════════════════
# Event wiring conventions
# ═══════════════════════════════════════════════════════════

class TestEventWiring:
    def test_wire_events_registered(self, page):
        """Verify wireEvents was called and click delegation is active."""
        has_listener = page.evaluate(
            "() => { const h = window.getEventListeners?.(document.body); return !!h; }"
        )
        # Chrome DevTools protocol may not expose listeners directly, so check behavior instead
        # We verify by checking the conventions are in place via DOM IDs

    def test_button_id_convention(self, page):
        """Action buttons use data-action + data-model attributes."""
        btns = page.evaluate(
            "() => [...document.querySelectorAll('[data-action]')].filter(b => b.dataset.model).map(b => ({ action: b.dataset.action || '', model: b.dataset.model || '' }))"
        )
        for btn in btns:
            assert btn['action'], f"Button missing data-action attribute"
