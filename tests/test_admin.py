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
_copy_fd, _TEST_CFG = tempfile.mkstemp(suffix=".yaml")
os.close(_copy_fd)
import shutil
shutil.copy2(_BASE_CFG, _TEST_CFG)


@pytest.fixture(scope="session")
def live_server():
    """Start a single ArkestraServer for the entire test session.

    Uses a separate copy of tests/test-admin-config.yaml so mutations
    from POST /admin/config never touch the base file or sample-config.yaml.
    """
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
    live_server["client"].headers["X-Admin-Key"] = key


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
            assert "repo" in model
            assert "tags" in model

    def test_uncached_status_for_downloaded_checkpoints(self, live_server):
        """Models with a checkpoint field but no HF cache should be UNCACHED.

        Cached-but-not-running models get status 'stopped'; truly uncached
        (no files in the cache dir) get 'uncached'.

        Creates real temp dirs so it works on any machine — no hardcoded paths.
        """
        import os as _os
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            hf_cache = Path(tmpdir)

            # Create cache dirs for models that should be 'stopped'
            (hf_cache / "models--unsloth--Qwen3.5-4B-GGUF").mkdir()
            (hf_cache / "models--unsloth--gemma-4-E2B-it-GGUF").mkdir()

            # Set env var so admin endpoint finds it
            _os.environ["HF_HUB_CACHE"] = str(hf_cache)

            try:
                client = live_server["client"]
                r = client.get("/admin/models")
                models_by_id = {m["id"]: m for m in r.json()["models"]}

                gemma = models_by_id["gemma-4-e2b"]
                qwen = models_by_id["qwen3.5-4b"]
                vox = models_by_id["voxtral-mini"]

                assert gemma["status"]["value"] == "cached"
                assert qwen["status"]["value"] == "cached"
                assert vox["status"]["value"] == "uncached"
            finally:
                _os.environ.pop("HF_HUB_CACHE", None)


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
        assert cfg["full-model"]["repo"] == "hugging-face"
        assert cfg["full-model"]["model"] == "test/full-model:Q5"
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
        assert "repo" in cfg
        assert "model" in cfg
        assert "args" in cfg
        assert cfg["repo"] == "hugging-face"
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


# ── /admin/stop-all ────────────────────────────────────────────────

class TestStopAll:
    """POST /admin/stop-all"""

    def test_stop_all_no_models_returns_200(self, live_server):
        """When no models are running, returns 200 with a message."""
        client = live_server["client"]
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
            "    checkpoint: foo.gguf\n"
        )
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, cfg.encode()); os.close(fd)

        server = ArkestraServer(path, port=18006)
        try:
            client = TestClient(server.get_app())
            client.headers["X-Admin-Key"] = "test-key"
            r = client.get("/admin/models")
            assert r.status_code == 200
            models = r.json()["models"]
            entry = next((m for m in models if m["id"] == "no-default-test"), None)
            assert entry is not None, "Model should appear in response"
            # Resolution chain: per-model (missing) → backends.default (missing) → cpu
            assert entry.get("backend_id") == "cpu", (
                f"Expected fallback to 'cpu', got {entry.get('backend_id')!r}"
            )
        finally:
            graceful_server_teardown({"server": server, "client": client})
