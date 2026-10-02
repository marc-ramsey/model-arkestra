"""Auth gate contract tests for ArkestraAdmin's embedded middleware.

Two keys, two scopes:

  api_key   gates the working namespaces - /v1/* (inference), /api/* and the
            dashboard at / + /index.html. admin_key is accepted there too, so a
            console operator needs only one credential.

  admin_key additionally locks /admin/*. A bare api_key must NOT reach /admin/*:
            that keeps a leaked inference token away from model lifecycle and
            config mutation.

  neither   everything open. A key is a switch, not decoration - with no keys in
            config there is nothing to present, so every namespace stays
            anonymous. That is the default posture for single-user LAN boxes.

Probes are never gated: /health, /v1/health and /api/v1/health answer
anonymously under every posture because Open WebUI polls them only to render
connection state (see static/api.js). They expose no model data.

Run: pytest tests/unit/test_auth_gate.py -v --timeout=90
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_arkestra.admin import (
    ASCII_KEY_ERR,
    AUTH_OPEN_PATHS,
    ArkestraAdmin,
    ascii_key,
    is_open_path,
    strip_base,
)


# ── Fixtures & helpers ─────────────────────────────────────────────────

API_KEY = "***"
ADMIN_KEY = "admin-console-Kp3xQm8vLz2wNr6ytA0iYe4uHd7gJf1bMq9sVc5oXaRtGhZuCjDlBkEeO"
BASE = "/ark"

INFERENCE_PATHS = ["/v1/chat/completions", "/v1/models"]
API_PATHS = ["/api/models", "/api/v1/health-live"]
ADMIN_PATHS = ["/admin/models", "/admin/config"]


class _StubCM:
    """Minimal ConfigManager stand-in."""

    config_path = "/dev/null"

    def __init__(self, data):
        self.data = data

    def export(self, path):
        pass


def _stub_arkestra():
    class _S:
        def __init__(self):
            self.cm = _StubCM({"models": {}, "backends": {}, "default": {}})
            self.models = {}
            self.clusters = {}

        def get_models(self):
            return []

        def get_model(self, name):
            return {}

        def get_v1_models(self):
            return {"data": []}

    _S.__name__ = "StubArkestra"
    return _S()


def make_client(api_key=None, admin_key=None, base_url="") -> TestClient:
    """App with ArkestraAdmin installed plus sentinel /v1 and probe routes.

    Route existence is deliberately separated from gating: a 401 means the gate
    blocked, anything else (200/404) means routing was reached.
    """
    app = FastAPI()
    server = type("Server", (), {"_arkestra": _stub_arkestra(), "base_url": base_url})()
    admin = ArkestraAdmin(server, admin_key=admin_key, app=app, api_key=api_key,
                          base_url=base_url)

    @app.get("/v1/chat/completions")
    async def chat_get():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_post():
        return {"ok": True}

    @app.get("/v1/models")
    async def v1_models():
        return {"data": []}

    for path in sorted(AUTH_OPEN_PATHS):
        def _probe(_p=path):
            async def handler():
                return {"status": "ok"}
            return handler
        app.add_api_route(path, _probe(), methods=["GET"])

    @app.get("/api/models")
    async def api_models():
        return {"data": []}

    if base_url:
        # Mirrors server.py: prefix rewriting installed after the gate.
        from starlette.responses import RedirectResponse  # noqa: F401

        @app.middleware("http")
        async def prefix(request, call_next):
            p = request.url.path
            if p == base_url or p.startswith(base_url + "/"):
                suffix = p[len(base_url):] or "/"
                request.scope["path"] = suffix
                request.scope["raw_path"] = (suffix or b"/").encode()
            return await call_next(request)

    admin.install()
    return TestClient(app)


def _hdr(token):
    """Authorization header; non-ASCII tokens sent as UTF-8 bytes.

    httpx encodes str header values as ASCII and raises otherwise, so a unicode
    password is handed over pre-encoded - exactly the bytes curl transmits.
    """
    if token is None:
        return {}
    try:
        f"Bearer {token}".encode("ascii")
        value = f"Bearer {token}"
    except UnicodeEncodeError:
        value = b"Bearer " + token.encode("utf-8", "surrogateescape")
    return {"Authorization": value}


def _get(client, path, token=None):
    return client.get(path, headers=_hdr(token))


# ── 1. No keys configured → everything open ───────────────────────────

class TestNoKeysAllOpen:
    """Without api-key/admin-key in config there is no token to present."""

    @pytest.mark.parametrize("path", ["/", "/index.html"] + INFERENCE_PATHS + ADMIN_PATHS)
    def test_open_anonymously(self, path):
        r = _get(make_client(), path)
        assert r.status_code != 401, f"{path} gated although no key configured"

    def test_post_inference_open(self):
        """POST /v1/chat/completions is the main consumer path."""
        client = make_client()
        r = client.post("/v1/chat/completions", json={"model": "x"})
        assert r.status_code != 401


# ── 2. api_key gates inference, config and dashboard ───────────────────

class TestApiKeyGatesWork:

    @pytest.mark.parametrize("path", INFERENCE_PATHS + ["/api/models"])
    def test_anonymous_rejected(self, path):
        assert _get(make_client(api_key=API_KEY), path).status_code == 401

    @pytest.mark.parametrize("path", INFERENCE_PATHS + ["/api/models"])
    def test_api_key_admits(self, path):
        r = _get(make_client(api_key=API_KEY), path, API_KEY)
        assert r.status_code != 401

    @pytest.mark.parametrize("path", INFERENCE_PATHS + ["/api/models"])
    def test_admin_key_also_admits(self, path):
        """One credential suffices for a console operator."""
        client = make_client(api_key=API_KEY, admin_key=ADMIN_KEY)
        assert _get(client, path, ADMIN_KEY).status_code != 401

    @pytest.mark.parametrize("path", ["/", "/index.html"])
    def test_dashboard_gated(self, path):
        assert _get(make_client(api_key=API_KEY), path).status_code == 401


# ── 3. admin_key locks /admin/* against api_key ───────────────────────

class TestAdminNamespaceIsPrivate:

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_anonymous_rejected(self, path):
        assert _get(make_client(admin_key=ADMIN_KEY), path).status_code == 401

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_admin_key_admits(self, path):
        client = make_client(api_key=API_KEY, admin_key=ADMIN_KEY)
        assert _get(client, path, ADMIN_KEY).status_code != 401

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_api_key_rejected(self, path):
        """A leaked inference token must not drive model lifecycle."""
        client = make_client(api_key=API_KEY, admin_key=ADMIN_KEY)
        assert _get(client, path, API_KEY).status_code == 401

    @pytest.mark.parametrize("path", ADMIN_PATHS)
    def test_open_when_no_admin_key(self, path):
        """No admin_key configured -> /admin/* is unauthenticated."""
        client = make_client(api_key=None, admin_key=None)
        assert _get(client, path).status_code != 401


# ── 4. Probes always open ─────────────────────────────────────────────

class TestProbesAlwaysOpen:
    """OWUI polls /api/v1/health for connection state; must never 401."""

    @pytest.mark.parametrize("path", sorted(AUTH_OPEN_PATHS))
    @pytest.mark.parametrize("kwargs", [
        {}, {"api_key": API_KEY}, {"admin_key": ADMIN_KEY},
        {"api_key": API_KEY, "admin_key": ADMIN_KEY},
    ])
    def test_probe_answer_anonymously(self, path, kwargs):
        assert _get(make_client(**kwargs), path).status_code != 401


# ── 5. Unicode keys: 401, never a crash ───────────────────────────────

class TestNonAsciiKeysRejected:
    """Header values are ASCII-only (RFC 7230), so a non-ASCII key is unusable.

    Rejecting at construction means secrets.compare_digest never sees a str it
    would raise on, and the operator learns of the typo at startup instead of
    seeing every request fail with a 500.
    """

    @pytest.mark.parametrize("kwargs", [
        {"api_key": "clé"},
        {"admin_key": "pässwörd"},
        {"api_key": API_KEY, "admin_key": "日本"},
    ])
    def test_startup_rejects(self, kwargs):
        with pytest.raises(ValueError):
            make_client(**kwargs)

    @pytest.mark.parametrize("secret", [API_KEY, ADMIN_KEY])
    def test_ascii_keys_accepted(self, secret):
        # Sanity: the validator must not reject legitimate keys.
        assert ascii_key(secret) == secret


# ── 6. Header handling edge cases ─────────────────────────────────────

class TestHeaderHandling:

    def test_missing_header_rejected(self):
        client = make_client(api_key=API_KEY)
        assert client.get("/api/models").status_code == 401

    @pytest.mark.parametrize("scheme", ["Basic Zm9v", "token", ""])
    def test_non_bearer_scheme_rejected(self, scheme):
        client = make_client(api_key=API_KEY)
        r = client.get("/api/models", headers={"Authorization": scheme})
        assert r.status_code == 401

    def test_empty_token_rejected(self):
        """A bare 'Bearer ' must not match an empty configured key."""
        client = make_client(api_key=API_KEY)
        assert _get(client, "/api/models", "").status_code == 401


# ── 7. base_url prefix must never open a bypass ───────────────────────

class TestStripBase:
    """The gate sees raw paths; classification happens after stripping."""

    @pytest.mark.parametrize("path,expected", [
        ("/ark/v1/chat/completions", "/v1/chat/completions"),
        ("/ark/api/models", "/api/models"),
        ("/ark/admin/models", "/admin/models"),
        ("/ark/", "/"),
        ("/ark", "/"),
    ])
    def test_prefix_removed(self, path, expected):
        assert strip_base(path, BASE) == expected

    @pytest.mark.parametrize("path", ["/other/v1/x", "/v1/chat"])
    def test_unrelated_path_untouched(self, path):
        assert strip_base(path, BASE) == path

    def test_empty_prefix_is_identity(self):
        assert strip_base("/v1/models", "") == "/v1/models"

    @pytest.mark.parametrize("path,expected", [
        ("/ark/health", True),
        ("/ark/api/v1/health", True),
        ("/ark/v1/chat/completions", False),
        ("/ark/admin/models", False),
    ])
    def test_open_path_under_prefix(self, path, expected):
        assert is_open_path(path, BASE) == expected

    @pytest.mark.parametrize("path", ["/ark/v1/chat/completions", "/ark/api/models"])
    def test_prefixed_namespaces_still_gated(self, path):
        """Unstripped paths would slip past the /v1 and /api gates entirely."""
        client = make_client(api_key=API_KEY, base_url=BASE)
        assert _get(client, path).status_code == 401

    def test_prefixed_admin_rejects_api_key(self):
        client = make_client(api_key=API_KEY, admin_key=ADMIN_KEY, base_url=BASE)
        assert _get(client, "/ark/admin/models", API_KEY).status_code == 401
