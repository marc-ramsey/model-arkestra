"""Tests for ArkestraAdmin authentication — unit tests, no live models required."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_arkestra.admin import ArkestraAdmin


# ── Helpers ────────────────────────────────────────────────────────

class _FakeArkestra:
    """Minimal stub so ArkestraAdmin can be instantiated."""

    class _MockCM:
        data = {"models": {}, "backends": {}}
        config_path = "/dev/null"

        def export(self, path):
            pass

    def __init__(self, cm_data=None):
        self._arkestra = type("Stub", (), {"cm": self._MockCM()})()


def _make_client(admin_key=None, extra_config=None):
    """Build a FastAPI app with ArkestraAdmin installed, return TestClient."""
    app = FastAPI()
    arkestra = _FakeArkestra()
    if extra_config:
        arkestra._arkestra.cm.data.update(extra_config)
    admin = ArkestraAdmin(arkestra, admin_key=admin_key, app=app)
    admin.install()
    return TestClient(app), arkestra


# ── No key configured — everything pass-through ───────────────────

class TestNoAdminKey:
    """When admin_key is None or empty, all /admin/* endpoints are accessible."""

    @pytest.mark.parametrize("key", [None, ""])
    def test_no_auth_required(self, key):
        client, _ = _make_client(admin_key=key)
        # Auth middleware should pass through — status ≠ 401
        r = client.get("/admin/models")
        assert r.status_code != 401

    def test_root_accessible_no_key(self):
        client, _ = _make_client(admin_key=None)
        r = client.get("/")
        assert r.status_code == 200
        # Key substitution happens — empty string if no key set
        assert "{{ADMIN_KEY}}" not in r.text

    def test_index_html_accessible_no_key(self):
        client, _ = _make_client(admin_key="")
        r = client.get("/index.html")
        assert r.status_code == 200


# ── Key configured — auth enforced on /admin/* ────────────────────

class TestAdminKeyEnforced:
    """When admin_key is set, /admin/* requires X-Admin-Key header."""

    def _client_with_key(self):
        client, _ = _make_client(admin_key="secret123")
        return client

    # --- Missing/wrong header → 401 ---

    def test_no_header_returns_401(self):
        r = self._client_with_key().get("/admin/models")
        assert r.status_code == 401

    def test_config_list_no_header_returns_401(self):
        r = self._client_with_key().get("/admin/config")
        assert r.status_code == 401

    def test_wrong_header_returns_401(self):
        client = self._client_with_key()
        r = client.get("/admin/models", headers={"X-Admin-Key": "wrong"})
        assert r.status_code == 401

    # --- Correct header → not 401 (auth passes, endpoint logic may fail) ---

    def test_correct_header_passes_auth(self):
        """With the right header, auth is bypassed — status ≠ 401."""
        client = self._client_with_key()
        r = client.get("/admin/models", headers={"X-Admin-Key": "secret123"})
        assert r.status_code != 401

    def test_correct_header_config_passes_auth(self):
        client = self._client_with_key()
        r = client.get("/admin/config", headers={"X-Admin-Key": "secret123"})
        assert r.status_code != 401

    # --- POST endpoints also gated ---

    def test_post_config_requires_header(self):
        client = self._client_with_key()
        r = client.post("/admin/config", json={"checkpoint": "test:Q4"})
        assert r.status_code == 401

    def test_post_start_requires_header(self):
        client = self._client_with_key()
        r = client.post("/admin/start/fake-model")
        assert r.status_code == 401

    # --- Public paths never gated ---

    def test_root_accessible_without_auth_when_key_set(self):
        """GET / and GET /index.html are never gated — even when admin_key is set."""
        client = self._client_with_key()
        r = client.get("/")
        assert r.status_code == 200
        # Key should be embedded in the HTML template
        assert "secret123" in r.text

    def test_index_html_accessible_without_auth(self):
        client = self._client_with_key()
        r = client.get("/index.html")
        assert r.status_code == 200
