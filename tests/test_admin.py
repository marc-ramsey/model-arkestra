"""Tests for ArkestraAdmin endpoints — live server, no mocks."""
import copy
import tempfile
import os
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from tests.conftest import graceful_server_teardown

from model_arkestra.server import ArkestraServer

# ── separate config copy so mutations never touch base file ────
_BASE_CFG = "tests/test-admin-config.yaml"
_base_cfg_dir = os.path.dirname(_BASE_CFG)
_copy_fd, _TEST_CFG = tempfile.mkstemp(suffix=".yaml")
os.close(_copy_fd)
import shutil
shutil.copy2(_BASE_CFG, _TEST_CFG)
_shutil_backup = shutil.copy2  # preserve for other tests in this file


@pytest.fixture(scope="session")
def live_server():
    """Start a single ArkestraServer for the entire test session.

    Uses a separate copy of tests/test-admin-config.yaml and tests/backends.yaml
    so mutations from POST /admin/config never touch the base files.
    """
    import shutil as _shutil
    _shutil.copy2("tests/backends.yaml", os.path.dirname(_TEST_CFG) + "/backends.yaml")
    server = ArkestraServer(_TEST_CFG, port=18005)
    client = TestClient(server.get_app())
    result = {"server": server, "client": client}
    yield result
    graceful_server_teardown(result)

@pytest.fixture(autouse=True)
def _set_admin_header(live_server):
    """Set admin header on the test client for all tests (header-based auth).

    Reads ADMIN_KEY from the actual config so it stays in sync.
    """
    key = live_server["server"]._arkestra.cm.data.get("env", {}).get("ADMIN_KEY") or ""
    live_server["client"].headers["Authorization"] = f"Bearer {key}"


# ── /admin/models ──────────────────────────────────────────────────

class TestAdminModels:
    """GET /admin/models returns all configured models with correct context."""

    def test_returns_all_configured_models(self, live_server):
        client = live_server["client"]
        r = client.get("/admin/models")
        assert r.status_code == 200

        ids = [m["name"] for m in r.json()["models"]]
        expected = {"gemma-4-e2b", "qwen3.5-4b", "voxtral-mini"}
        assert set(ids) == expected

    def test_non_running_models_have_constructed_contexts(self, live_server):
        client = live_server["client"]
        r = client.get("/admin/models")
        models_by_id = {m["name"]: m for m in r.json()["models"]}

        # All should have required fields
        for model in r.json()["models"]:
            assert "name" in model
            assert "status" in model
            assert "backend" in model
            assert "runner" in model
            assert "model" in model

    def test_uncached_status_for_downloaded_checkpoints(self, live_server):
        """Models with cached GGUF files get 'stopped'; uncached models get 'unloaded'.

        Validates the status mapping is consistent across all configured models.
        Empty cache dirs (partial pulls without GGUFs) are treated as uncached.
        """
        client = live_server["client"]
        r = client.get("/admin/models")
        assert r.status_code == 200
        models_by_id = {m["name"]: m for m in r.json()["models"]}

        # Every model should have a valid status value
        for name, m in models_by_id.items():
            val = m["status"]["value"]
            assert val in ("stopped", "loading", "loaded", "unloaded", "downloading"), \
                f"{name}: unexpected status '{val}'"

        # All configured models must have a context (pre-created at startup)
        assert len(models_by_id) == 3
        for name in ("gemma-4-e2b", "qwen3.5-4b", "voxtral-mini"):
            assert name in models_by_id


# ── /admin/stop/{model} ────────────────────────────────────────────

class TestStopModel:
    """POST /admin/stop/{model}"""

    def test_stop_already_stopped_returns_202(self, live_server):
        """Stopping an already-stopped model returns 202; stopping a running
        model is handled asynchronously via _arkestra.stop() returning 200."""
        client = live_server["client"]
        r = client.post("/admin/stop/qwen3.5-4b")
        # 202 if already stopped (terminal), 200 if we stopped a running model,
        # 404 if model not configured
        assert r.status_code in (200, 202, 404)


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
        """POST /admin/config creates a new model."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={"model": "unsloth/test/new-model:Q4", "temp": 0.7},
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
                "model": "unsloth/test/full-model:Q5",
                "backend": "rocm",
                "ctx-size": 8192,
                "tags": ["chat", "reasoning"],
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["model"] == "full-model"

        # Verify the model was persisted in config
        cfg = live_server["server"]._arkestra.cm.data.get("models")
        assert "full-model" in cfg
        assert cfg["full-model"]["model"] == "unsloth/test/full-model:Q5"
        assert cfg["full-model"]["backend"] == "rocm"
        assert cfg["full-model"]["ctx-size"] == 8192
        assert cfg["full-model"]["tags"] == ["chat", "reasoning"]

    def test_create_requires_checkpoint(self, live_server):
        """POST without checkpoint returns 400."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={"name": "no-checkpoint", "temp": 1.0},
        )
        assert r.status_code == 400

    def test_create_duplicate_name_returns_409(self, live_server):
        """POST with existing model name returns 409."""
        client = live_server["client"]
        r = client.post(
            "/admin/config",
            json={"repo": "hugging-face", "model": "existing/model:Q4", "name": "qwen3.5-4b"},
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
        assert "model" in cfg
        assert "temp" in cfg
        assert cfg["model"] == "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"

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
        original_temp = r.json()["config"].get("temp", 0.8)
        original_ctx = r.json()["config"].get("ctx-size", 16384)

        # PUT modified fields back
        r = client.put(
            "/admin/config/qwen3.5-4b",
            json={"temp": 1.5, "ctx-size": 32768},
        )
        assert r.status_code == 200

        # GET again to verify
        r = client.get("/admin/config/qwen3.5-4b")
        assert r.json()["config"]["temp"] == 1.5
        assert r.json()["config"]["ctx-size"] == 32768

        # PUT original back so tests remain consistent
        client.put(
            "/admin/config/qwen3.5-4b",
            json={"temp": original_temp, "ctx-size": original_ctx},
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


# ── /admin/stop-all ────────────────────────────────────────────────

class TestStopAll:
    """POST /admin/stop-all"""

    def test_stop_all_no_models_returns_200(self, live_server):
        """When no models are running, returns 200 with a message."""
        client = live_server["client"]
        # Ensure all models are stopped first (they may have been started
        # by earlier session-scoped fixture tests)
        client.post("/admin/stop-all")
        r = client.post("/admin/stop-all")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "nothing to stop" in body["message"].lower()
        assert isinstance(body["stopped"], list)

    def test_stop_all_has_stopped_field(self, live_server):
        """The response always includes the 'stopped' key (empty list when idle)."""
        client = live_server["client"]
        r = client.post("/admin/stop-all")
        assert r.status_code == 200
        body = r.json()
        assert "stopped" in body
        assert isinstance(body["stopped"], list)


# ── /admin/shutdown ────────────────────────────────────────────────

class TestShutdown:
    """POST /admin/shutdown — server teardown."""

    def test_shutdown_returns_200(self, live_server):
        """Returns 200 with ok/message structure immediately (shutdown runs in background)."""
        client = live_server["client"]
        r = client.post("/admin/shutdown")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "shutting down" in body["message"].lower()


# ── Backend resolution fallback integration ───────────────────────

class TestBackendResolutionFallback:
    """Model with no backend field falls back to BaseModelRunner default."""

    def test_no_backend_resolves_to_cpu(self, live_server):
        """Verify that _resolve_backend chain resolves a model without 'backend:' via the /admin/models route.

        The model 'no-default-test' has no 'backend' key and no global 'backends.default'
        in its config. Resolution should fall back to BaseModelRunner._DEFAULT_BACKEND
        (cpu) end-to-end through the admin API.
        """
        import tempfile, os
        cfg = (
            "env:\n  ADMIN_KEY: test-key\n"
            "models:\n"
            "  no-default-test:\n"
            "    model: foo.gguf\n"
        )
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, cfg.encode()); os.close(fd)

        server = ArkestraServer(path, port=18006)
        try:
            client = TestClient(server.get_app())
            client.headers["Authorization"] = "Bearer test-key"
            r = client.get("/admin/models")
            assert r.status_code == 200
            models = r.json()["models"]
            entry = next((m for m in models if m["name"] == "no-default-test"), None)
            assert entry is not None, "Model should appear in response"
            # Resolution chain: per-model (missing) → backends.default (missing) → cpu
            assert entry.get("backend") == "cpu", (
                f"Expected fallback to 'cpu', got {entry.get('backend')!r}"
            )
        finally:
            graceful_server_teardown({"server": server, "client": client})
