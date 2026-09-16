"""Tests for GET /api/models — cached-model listing.

Regression: with checkpoint-style config (models reference ``checkpoint:`` and
the weight ref lives on the checkpoint), /api/models read the raw models:
section, found no ``model:`` field, and returned an empty list even when the
weights were cached. The endpoint must use the merged config.

Run: pytest tests/unit/test_api_models.py -v
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_arkestra.server import ArkestraServer


CHECKPOINT_CFG = """\
default:
  model-start-port: 18300
  model-ports: 4

backends:
  cpu:
    runner: process
    engine: llama-cpp

checkpoints:
  gemma:
    ref: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    backend: cpu

models:
  gemma-chat:
    checkpoint: gemma
    temp: 0.7
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Server with a checkpoint-style config and an isolated HF cache."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CHECKPOINT_CFG)

    server = ArkestraServer(str(cfg), port=18301, ready_timeout=5)
    # Point the cache root at an isolated temp dir (never touches real cache)
    server._arkestra._cm.data["default-env"] = {
        "hf-hub-cache": str(tmp_path / "hub")
    }
    server._arkestra._env = None  # force rebuild on next resolve_config
    return TestClient(server.get_app())


def _make_cached(hub: Path) -> None:
    """Create the minimal HF cache layout for the configured checkpoint."""
    d = hub / "models--unsloth--gemma-4-E2B-it-GGUF"
    (d / "snapshots" / "abc").mkdir(parents=True)
    (d / "snapshots" / "abc" / "g.gguf").touch()


class TestApiModels:
    def test_cached_model_listed_with_checkpoint_config(self, client, tmp_path):
        """Cached checkpoint-style model appears with name, ref, size."""
        _make_cached(tmp_path / "hub")

        r = client.get("/api/models")
        assert r.status_code == 200
        models = r.json()["models"]

        names = [m["name"] for m in models]
        assert "gemma-chat" in names
        entry = next(m for m in models if m["name"] == "gemma-chat")
        # Ref comes from the checkpoint — proves merged config was used
        assert entry["model"] == "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"

    def test_uncached_model_absent(self, client, tmp_path):
        """No cache dir → model not listed."""
        r = client.get("/api/models")
        assert r.status_code == 200
        assert r.json()["models"] == []
