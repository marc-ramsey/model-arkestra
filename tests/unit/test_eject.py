"""Tests for ModelArkestra.eject() — cache helpers, eject logic, admin wrapper."""

from __future__ import annotations
import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_arkestra.arkestra import ModelArkestra
from model_arkestra.server import ArkestraServer
from model_arkestra.types import RunnerState, _ModelContext


# ── Helpers ────────────────────────────────────────────────────────────────


class MockRunner:
    """Minimal runner mock that exposes _models like a real runner."""

    def __init__(self) -> None:
        self._models = {}

    @property
    def running_models(self):
        return {
            name for name, ctx in self._models.items()
            if ctx.state == RunnerState.RUNNING
        }


def make_ctx(name: str, port: int, state: RunnerState) -> _ModelContext:
    ctx = _ModelContext(name, port)
    ctx.state = state
    return ctx


# ── Step 1: Unit tests for cache helpers ────────────────────────────────

class TestCacheRoot:
    """Test ModelArkestra._cache_root() env resolution order."""

    def test_config_hf_hub_cache(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.yaml")
            with open(cfg_path, "w") as f:
                f.write("env:\n  HF_HUB_CACHE: /custom/hf/cache\nmodels: {}\n")
            ma = ModelArkestra(cfg_path)
            assert ma._cache_root() == Path("/custom/hf/cache")

    def test_os_env_hf_fallback(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.yaml")
            with open(cfg_path, "w") as f:
                f.write("models: {}\n")
            ma = ModelArkestra(cfg_path)
            monkeypatch.setenv("HF_HUB_CACHE", "/os-env/hf")
            assert ma._cache_root() == Path("/os-env/hf")

    def test_os_env_takes_priority_over_config(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.yaml")
            with open(cfg_path, "w") as f:
                f.write("env:\n  HF_HUB_CACHE: /config/hf\nmodels: {}\n")
            ma = ModelArkestra(cfg_path)
            monkeypatch.setenv("HF_HUB_CACHE", "/os-env/hf")
            assert ma._cache_root() == Path("/os-env/hf")

    def test_default_when_nothing_set(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.yaml")
            with open(cfg_path, "w") as f:
                f.write("models: {}\n")
            ma = ModelArkestra(cfg_path)
            monkeypatch.delenv("HF_HUB_CACHE", raising=False)
            assert ma._cache_root() == Path("~/.cache/huggingface/hub").expanduser()


class TestCacheDirForCheckpoint:
    """Test _cache_dir_for_checkpoint() path construction."""

    def test_path_with_revision(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.yaml")
            with open(cfg_path, "w") as f:
                f.write(f"env:\n  HF_HUB_CACHE: {td}\nmodels: {{}}\n")
            ma = ModelArkestra(cfg_path)
            result = ma._cache_dir_for_checkpoint("unsloth/Qwen3-4B-GGUF:Q4_K_M")
            expected = Path(td) / "models--unsloth--Qwen3-4B-GGUF:Q4_K_M"
            assert result == expected

    def test_path_without_revision(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "cfg.yaml")
            with open(cfg_path, "w") as f:
                f.write(f"env:\n  HF_HUB_CACHE: {td}\nmodels: {{}}\n")
            ma = ModelArkestra(cfg_path)
            result = ma._cache_dir_for_checkpoint("meta-llama/Llama-3.2-1B")
            expected = Path(td) / "models--meta-llama--Llama-3.2-1B"
            assert result == expected


# ── Step 2: Integration tests for eject() ───────────────────────────────

class TestEjectMethod:
    """Test ModelArkestra.eject() — real config, mocked runners."""

    def _make_arkestra(self):
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "test-config.yaml"
        )
        return ModelArkestra(str(Path(cfg_path).resolve()), ready_timeout=30)

    # ── Happy path: cache dir exists on disk ──────────────────────────

    def test_happy_path_cache_exists(self, monkeypatch):
        """Cache present → deleted, result reflects success."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        ma = self._make_arkestra()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override cache root to our temp dir
            ma._cm.data["env"] = {"HF_HUB_CACHE": tmpdir}

            # Build runner with qwen3.5-4b in RUNNING state
            runner = MockRunner()
            ctx = make_ctx("qwen3.5-4b", 18000, RunnerState.RUNNING)
            runner._models["qwen3.5-4b"] = ctx
            ma._runners["process"] = runner

            # Create the physical cache dir (cache paths strip quantizer tag)
            cache_dir = ma._cache_dir_for_checkpoint("unsloth/Qwen3.5-4B-GGUF")
            cache_dir.mkdir(parents=True, exist_ok=True)

            result = asyncio.run(ma.eject("qwen3.5-4b"))

            assert result["ok"] is True
            assert result["model"] == "qwen3.5-4b"
            assert result["cache_deleted"] is True
            assert str(result["cache_path"]) == str(cache_dir)
            assert result["contexts_cleared"] == 1
            assert not cache_dir.exists()

    # ── Happy path: cache dir doesn't exist on disk ───────────────────

    def test_happy_path_no_cache_on_disk(self, monkeypatch):
        """No physical cache → no deletion, but model still ejected."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        ma = self._make_arkestra()
        with tempfile.TemporaryDirectory() as tmpdir:
            ma._cm.data["env"] = {"HF_HUB_CACHE": tmpdir}

            runner = MockRunner()
            ctx = make_ctx("qwen3.5-4b", 18000, RunnerState.RUNNING)
            runner._models["qwen3.5-4b"] = ctx
            ma._runners["process"] = runner

            result = asyncio.run(ma.eject("qwen3.5-4b"))

            assert result["ok"] is True
            assert result["cache_deleted"] is False
            assert result["contexts_cleared"] == 1

    # ── No checkpoint in config ───────────────────────────────────────

    def test_no_checkpoint_in_config(self):
        """Model with no checkpoint → stop only, cache_deleted=false."""
        ma = self._make_arkestra()
        # Add a model that has no checkpoint defined
        ma._cm.data["models"]["bare-model"] = {}

        runner = MockRunner()
        ctx = make_ctx("bare-model", 18003, RunnerState.RUNNING)
        runner._models["bare-model"] = ctx
        ma._runners["runner-bare"] = runner

        result = asyncio.run(ma.eject("bare-model"))

        assert result["ok"] is True
        assert result["cache_deleted"] is False
        assert "cache_path" not in result
        assert result["contexts_cleared"] == 1

    # ── Model not in config ───────────────────────────────────────────

    def test_model_not_in_config(self):
        with pytest.raises(ValueError, match="not in config"):
            asyncio.run(self._make_arkestra().eject("nonexistent-model"))

    # ── Shared-cache conflict: two RUNNING models same checkpoint ─────

    def test_shared_cache_conflict(self, monkeypatch):
        """Two different models share the same cache → eject blocked."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        ma = self._make_arkestra()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set cache root first so all paths resolve correctly
            ma._cm.data["env"] = {"HF_HUB_CACHE": tmpdir}

            shared_checkpoint = "shared/same-model:Q4_K_M"
            # Cache path strips quantizer tag
            shared_cache_path = "shared/same-model"
            shared_cache = ma._cache_dir_for_checkpoint(shared_cache_path)
            shared_cache.mkdir(parents=True, exist_ok=True)

            ma._cm.data["models"]["model-a"] = {"repo": "hugging-face", "model": shared_checkpoint}
            ma._cm.data["models"]["model-b"] = {"repo": "hugging-face", "model": shared_checkpoint}

            runner_a = MockRunner()
            ctx_a = make_ctx("model-a", 18000, RunnerState.RUNNING)
            runner_a._models["model-a"] = ctx_a
            ma._runners["runner-a"] = runner_a

            runner_b = MockRunner()
            ctx_b = make_ctx("model-b", 18001, RunnerState.RUNNING)
            runner_b._models["model-b"] = ctx_b
            ma._runners["runner-b"] = runner_b

            with pytest.raises(ValueError, match="is in use by other running runners"):
                asyncio.run(ma.eject("model-a"))

            # Cache NOT deleted
            assert shared_cache.exists()

    # ── Shared checkpoint but STOPPED model → no conflict ─────────────

    def test_shared_checkpoint_no_conflict_stopped(self, monkeypatch):
        """Shared checkpoint, but other model is STOPPED → eject succeeds."""
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        ma = self._make_arkestra()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set cache root first so all paths resolve correctly
            ma._cm.data["env"] = {"HF_HUB_CACHE": tmpdir}

            shared_checkpoint = "shared/same-model:Q4_K_M"
            shared_cache_path = "shared/same-model"
            shared_cache = ma._cache_dir_for_checkpoint(shared_cache_path)
            shared_cache.mkdir(parents=True, exist_ok=True)
            ma._cm.data["models"]["model-a"] = {"repo": "hugging-face", "model": shared_checkpoint}
            ma._cm.data["models"]["model-b"] = {"repo": "hugging-face", "model": shared_checkpoint}

            runner_a = MockRunner()
            ctx_a = make_ctx("model-a", 18000, RunnerState.RUNNING)
            runner_a._models["model-a"] = ctx_a
            ma._runners["runner-a"] = runner_a

            runner_b = MockRunner()
            ctx_b = make_ctx("model-b", 18001, RunnerState.STOPPED)
            runner_b._models["model-b"] = ctx_b
            ma._runners["runner-b"] = runner_b

            result = asyncio.run(ma.eject("model-a"))

            assert result["ok"] is True
            assert result["cache_deleted"] is True
            assert not shared_cache.exists()


# ── Step 3: Admin endpoint wrapper tests ──────────────────────────────

@pytest.fixture(autouse=True)
def backup_config():
    """Back up and restore test-config.yaml around each test."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "test-config.yaml"
    )
    bak = str(cfg_path) + ".bak"
    shutil.copy(cfg_path, bak)
    yield
    shutil.move(bak, cfg_path)


@pytest.fixture(scope="session")
def live_server():
    server = ArkestraServer(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-config.yaml"),
        port=18006,
    )
    client = TestClient(server.get_app())
    return {"server": server, "client": client}


@pytest.fixture(autouse=True)
def _set_admin_cookie(live_server):
    live_server["client"].cookies["admin_key"] = ""


class TestAdminEjectEndpoint:
    """POST /admin/eject/{model} — endpoint wrapper tests."""

    def test_200_with_rich_body(self, live_server):
        r = live_server["client"].post("/admin/eject/qwen3.5-4b")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "model" in body
        assert "cache_deleted" in body
        assert "contexts_cleared" in body

    def test_404_nonexistent(self, live_server):
        r = live_server["client"].post("/admin/eject/nonexistent-model-xyz")
        assert r.status_code == 404
