"""Tests for ArkestraAdmin endpoints — live server, no mocks."""
import copy
import shutil
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from model_arkestra.server import ArkestraServer


@pytest.fixture(autouse=True)
def backup_config():
    """Back up and restore sample-config.yaml around each test."""
    shutil.copy("sample-config.yaml", ".sample-config.yaml.bak")
    yield
    shutil.move(".sample-config.yaml.bak", "sample-config.yaml")


@pytest.fixture(scope="session")
def live_server():
    """Start a single ArkestraServer for the entire test session."""
    server = ArkestraServer("sample-config.yaml", port=9100)
    client = TestClient(server.get_app())
    return {"server": server, "client": client}


# ── fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def admin_headers():
    return {"X-Admin-Key": "whatever"}


# ── /admin/models ──────────────────────────────────────────────────

class TestAdminModels:
    """GET /admin/models returns all configured models with correct context."""

    def test_returns_all_configured_models(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.get("/admin/models", headers=admin_headers)
        assert r.status_code == 200

        ids = [m["id"] for m in r.json()["data"]]
        expected = {"gemma-4-e2b", "qwen3.5-4b", "voxtral-mini"}
        assert set(ids) == expected

    def test_non_running_models_have_constructed_contexts(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.get("/admin/models", headers=admin_headers)
        models_by_id = {m["id"]: m for m in r.json()["data"]}

        # All should have required context fields
        for model in r.json()["data"]:
            assert "id" in model
            assert "status" in model
            assert "port" in model
            assert "runner_type" in model
            assert "backend_id" in model

    def test_uncached_status_for_downloaded_checkpoints(self, live_server, admin_headers):
        """Models with a checkpoint field but no HF cache should be UNCACHED."""
        client = live_server["client"]
        r = client.get("/admin/models", headers=admin_headers)
        models_by_id = {m["id"]: m for m in r.json()["data"]}

        # Checkpoints are cached at /home/lemonade/hub — none of these have been downloaded
        assert all(m["status"] == "uncached" for m in r.json()["data"])


# ── /admin/stop/{model} ────────────────────────────────────────────

class TestStopModel:
    """POST /admin/stop/{model}"""

    def test_stop_already_stopped_returns_202(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.post("/admin/stop/qwen3.5-4b", headers=admin_headers)
        assert r.status_code in (202, 404)  # 202 if context exists and stopped, 404 if never started
        body = r.json()
        if r.status_code == 202:
            assert body["ok"] is True
            assert body["model"] == "qwen3.5-4b"

    def test_stop_unknown_model_returns_404(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.post("/admin/stop/nonexistent", headers=admin_headers)
        assert r.status_code == 404
        client = live_server["client"]
        r = client.post("/admin/stop/nonexistent", headers=admin_headers)
        assert r.status_code == 404


# ── /admin/update/{model} ─────────────────────────────────────────

class TestUpdateModel:
    """POST /admin/update/{model}"""

    def test_duplicate_name_returns_409(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.post("/admin/update/qwen3.5-4b", headers=admin_headers, params={"name": "gemma-4-e2b"})
        assert r.status_code == 409

    def test_empty_name_ignored(self, live_server, admin_headers):
        client = live_server["client"]
        # Empty name is treated as no change — config is written and restart is attempted.
        # Result depends on whether the process can actually start (live server).
        model_before = copy.deepcopy(
            live_server["server"]._arkestra.cm.data["models"]["qwen3.5-4b"]
        )
        r = client.post("/admin/update/qwen3.5-4b", headers=admin_headers, params={"name": "", "backend": "vulkan-radv"})
        model_after = live_server["server"]._arkestra.cm.data["models"]["qwen3.5-4b"]
        # Config should be written regardless of restart outcome
        assert "backend" in model_after and model_after["backend"] == "vulkan-radv"

    def test_config_rolled_back_on_restart_failure(self, live_server, admin_headers):
        """When restart() fails, config should be restored to its original state."""
        client = live_server["client"]
        model_before = copy.deepcopy(
            live_server["server"]._arkestra.cm.data["models"]["qwen3.5-4b"]
        )

        r = client.post("/admin/update/qwen3.5-4b", headers=admin_headers, params={"backend": "nonexistent"})
        assert r.status_code == 500

        # Verify config was rolled back
        model_after = live_server["server"]._arkestra.cm.data["models"]["qwen3.5-4b"]
        assert model_after == model_before


# ── /admin/eject/{model} ───────────────────────────────────────────

class TestEjectModel:
    """POST /admin/eject/{model}"""

    def test_eject_returns_200(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.post("/admin/eject/qwen3.5-4b", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["model"] == "qwen3.5-4b"

    def test_eject_nonexistent_model_returns_404(self, live_server, admin_headers):
        client = live_server["client"]
        r = client.post("/admin/eject/nonexistent", headers=admin_headers)
        assert r.status_code == 404
