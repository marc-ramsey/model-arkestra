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

@pytest.fixture(autouse=True)
def _set_admin_cookie(live_server):
    """Set admin cookie on the test client for all tests."""
    live_server["client"].cookies["admin_key"] = "whatever"


# ── /admin/models ──────────────────────────────────────────────────

class TestAdminModels:
    """GET /admin/models returns all configured models with correct context."""

    def test_returns_all_configured_models(self, live_server):
        client = live_server["client"]
        r = client.get("/admin/models")
        assert r.status_code == 200

        ids = [m["id"] for m in r.json()["models"]]
        expected = {"gemma-4-e2b", "qwen3.5-4b", "voxtral-mini"}
        assert set(ids) == expected

    def test_non_running_models_have_constructed_contexts(self, live_server):
        client = live_server["client"]
        r = client.get("/admin/models")
        models_by_id = {m["id"]: m for m in r.json()["models"]}

        # All should have required context fields
        for model in r.json()["models"]:
            assert "id" in model
            assert "status" in model
            assert "port" in model
            assert "runner_type" in model
            assert "backend_id" in model
            assert "args" in model
            assert "checkpoint" in model
            assert "capabilities" in model

    def test_uncached_status_for_downloaded_checkpoints(self, live_server):
        """Models with a checkpoint field but no HF cache should be UNCACHED.

        Cached-but-not-running models get status 'stopped'; truly uncached
        (no files in the cache dir) get 'uncached'.
        """
        client = live_server["client"]
        r = client.get("/admin/models")
        models_by_id = {m["id"]: m for m in r.json()["models"]}

        # Checkpoints cached at /home/lemonade/hub — mixed reality on a real machine
        gemma  = models_by_id["gemma-4-e2b"]
        qwen   = models_by_id["qwen3.5-4b"]
        vox    = models_by_id["voxtral-mini"]

        # Verify expected statuses based on actual cache presence:
        # gemma & qwen are cached, so stopped (not running);
        # vox is not cached, so uncached.
        assert gemma["status"] == "stopped"
        assert qwen["status"] == "stopped"
        assert vox["status"] == "uncached"


# ── /admin/stop/{model} ────────────────────────────────────────────

class TestStopModel:
    """POST /admin/stop/{model}"""

    def test_stop_already_stopped_returns_202(self, live_server):
        client = live_server["client"]
        r = client.post("/admin/stop/qwen3.5-4b")
        # 202 if context exists and stopped, 404 if never started
        assert r.status_code in (202, 404)


# ── /admin/config/ (collection) ────────────────────────────────────

class TestConfigCollection:
    """GET/POST /admin/config — list models, create new."""

    def test_get_list_returns_model_names(self, live_server):
        """GET /admin/config returns a list of model names."""
        client = live_server["client"]
        r = client.get("/admin/config")
        assert r.status_code == 200
        body = r.json()
        assert set(body["models"]) == {"gemma-4-e2b", "qwen3.5-4b", "voxtral-mini"}

    def test_create_basic_model(self, live_server):
        """POST /admin/config creates a new model with checkpoint."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={"checkpoint": "test/new-model:Q4", "args": "--temp 0.7"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["ok"] is True
        assert body["model"] == "new-model"

    def test_create_with_all_fields(self, live_server):
        """POST /admin/config with optional backend/tags."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={
                "checkpoint": "test/full-model:Q5",
                "backend": "rocm",
                "args": "--ctx 8192",
                "tags": ["chat", "reasoning"],
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["model"] == "full-model"

        # Verify the model was persisted in config
        cfg = live_server["server"]._arkestra.cm.data.get("models")
        assert "full-model" in cfg
        assert cfg["full-model"]["checkpoint"] == "test/full-model:Q5"
        assert cfg["full-model"]["backend"] == "rocm"
        assert cfg["full-model"]["args"] == "--ctx 8192"
        assert cfg["full-model"]["tags"] == ["chat", "reasoning"]

    def test_create_requires_checkpoint(self, live_server):
        """POST without checkpoint returns 400."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={"name": "no-checkpoint", "args": "--temp 1.0"},
        )
        assert r.status_code == 400

    def test_create_duplicate_name_returns_409(self, live_server):
        """POST with existing model name returns 409."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={"checkpoint": "existing/checkpoint:Q4", "name": "qwen3.5-4b"},
        )
        assert r.status_code == 409


# ── /admin/config/{model} (single) ───────────────────────────────────

class TestConfigModel:
    """GET/PUT /admin/config/{model} — read and update."""

    def test_get_returns_config(self, live_server):
        """GET /admin/config/{model} returns model config."""
        client = live_server["client"]
        r = client.get("/admin/config/qwen3.5-4b")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["model"] == "qwen3.5-4b"
        cfg = body["config"]
        assert "checkpoint" in cfg
        assert "args" in cfg
        assert cfg["checkpoint"] == "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"

    def test_get_nonexistent_returns_404(self, live_server):
        """GET for missing model returns 404."""
        client = live_server["client"]
        r = client.get("/admin/config/nonexistent")
        assert r.status_code == 404

    def test_get_can_be_modified_and_saved(self, live_server):
        """GET config, modify via PUT, verify persisted."""
        client = live_server["client"]
        # GET original
        r = client.get("/admin/config/qwen3.5-4b")
        assert r.status_code == 200
        original_args = r.json()["config"]["args"]

        # PUT modified args back
        r = client.put(
            "/admin/config/qwen3.5-4b",
            json={"args": "--temp 1.5 --ctx-size 32768"},
        )
        assert r.status_code == 200

        # GET again to verify
        r = client.get("/admin/config/qwen3.5-4b")
        assert r.json()["config"]["args"] == "--temp 1.5 --ctx-size 32768"

        # PUT original back so tests remain consistent
        client.put(
            "/admin/config/qwen3.5-4b",
            json={"args": original_args},
        )


# ── /admin/eject/{model} ───────────────────────────────────────────

class TestEjectModel:
    """POST /admin/eject/{model}"""

    def test_eject_returns_200(self, live_server):
        client = live_server["client"]
        r = client.post("/admin/eject/qwen3.5-4b")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["model"] == "qwen3.5-4b"

    def test_eject_nonexistent_model_returns_404(self, live_server):
        client = live_server["client"]
        r = client.post("/admin/eject/nonexistent")
        assert r.status_code == 404
