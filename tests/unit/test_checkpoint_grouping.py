"""Tests: models sharing a checkpoint map to ONE shared process context."""

from __future__ import annotations
import os
import tempfile

from model_arkestra.arkestra import ModelArkestra


def _make(cfg_text: str) -> ModelArkestra:
    p = tempfile.mktemp(suffix=".yaml")
    with open(p, "w") as f:
        f.write(cfg_text)
    try:
        return ModelArkestra(p, ready_timeout=2, warmup_delay=0)
    finally:
        os.unlink(p)


SHARED_CFG = """
default:
  model-start-port: 18000
  model-ports: 8
  ctx-size: 8192

backends:
  cpu:
    runner: process
    engine: llama-cpp
    args:
      threads: 4

runners:
  process:
    class-name: ProcessRunner

checkpoints:
  qwen3.8-27b:
    ref: /tmp/fake-nonexistent.gguf
    backend: cpu
    parallel: 2
  gemma:
    ref: /tmp/fake2.gguf
    backend: cpu

models:
  qwen3.8-27b-instruct:
    checkpoint: qwen3.8-27b
    temp: 0.7
  qwen3.8-27b-think:
    checkpoint: qwen3.8-27b
    temp: 1.0
  gemma-4b:
    checkpoint: gemma
"""


class TestCheckpointGrouping:
    def test_shared_models_are_one_context(self):
        ma = _make(SHARED_CFG)
        models = ma.models
        # Both names resolve to the SAME context object.
        assert models["qwen3.8-27b-instruct"] is models["qwen3.8-27b-think"]
        # A different checkpoint is a distinct context.
        assert models["gemma-4b"] is not models["qwen3.8-27b-instruct"]

    def test_two_distinct_contexts_for_three_models(self):
        ma = _make(SHARED_CFG)
        uniq = {id(c) for c in ma.models.values()}
        assert len(uniq) == 2  # qwen (shared) + gemma

    def test_shared_ctx_named_by_checkpoint_id(self):
        ma = _make(SHARED_CFG)
        ctx = ma.model_obj("qwen3.8-27b-think")
        assert ctx.name == "qwen3.8-27b"

    def test_v1_models_status_for_alias(self):
        # Aliased model names (name != checkpoint id) must resolve to their
        # shared context in /v1/models — not report as uncached.
        ma = _make(SHARED_CFG)
        # Drive both shared contexts to LOADING (not the default UNCACHED)
        # so a broken lookup would be visible as the fallback "uncached".
        ma.model_obj("qwen3.8-27b-instruct").set_state("load")
        ma.model_obj("gemma-4b").set_state("load")
        data = {e["name"]: e for e in ma.get_v1_models()["data"]}
        for name in ("qwen3.8-27b-instruct", "qwen3.8-27b-think", "gemma-4b"):
            assert data[name]["status"]["value"] == "loading", name
        # Sanity: all alias names report the same (shared) status.
        assert data["qwen3.8-27b-instruct"]["status"] == data["qwen3.8-27b-think"]["status"]

    def test_merged_config_flows_parallel(self):
        # The shared checkpoint's parallel: 2 reaches the merged model config,
        # so the single process launches with --parallel 2 (KV-shared slots).
        ma = _make(SHARED_CFG)
        cfg = ma.get_model("qwen3.8-27b-instruct")
        assert cfg.get("parallel") == 2
