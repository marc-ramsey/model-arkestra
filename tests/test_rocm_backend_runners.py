"""Tests for backend → runner routing: rocm with process, podman, docker.

These tests verify the documented behavior that each backend can be paired
with any of the three runner types (process, podman, docker) via the
`backends.<id>.runner:` configuration key.

Documented behavior:
  backends.rocm.runner → "process"    → ProcessModelRunner
  backends.rocm.runner → "podman"     → PodmanModelRunner
  backends.rocm.runner → "docker"     → DockerModelRunner

See docs/config.md and docs/architecture.md for the full specification.

Server-level integration (uvicorn, health, chat completions) is covered by
test_admin_live.py which uses a real uvicorn process + httpx with no mocks.
"""

from __future__ import annotations
import asyncio
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from model_arkestra.arkestra import ModelArkestra


# ── Helpers ─────────────────────────────────────────────────────────────────


def _write_config(yaml_text: str) -> str:
    """Write *yaml_text* to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.write(fd, yaml_text.encode())
    os.close(fd)
    return path


# ── Config builders (module-level functions — not pytest fixtures) ──────────

def cfg_rocm_process() -> str:
    """rocm backend → runner: process"""
    return _write_config(textwrap.dedent("""\
        models-start-port: 18000
        model-ports: 32
        defaults:
          ctx-size: 16384
          n-gpu-layers: 999
        backends:
          rocm:
            runner: process
            args:
              flash-attn: "on"
        models:
          test-rocm-process:
            checkpoint: test/model:Q4_K_M
            backend: rocm
            args:
              temp: 0.7
    """))


def cfg_rocm_podman() -> str:
    """rocm backend → runner: podman"""
    return _write_config(textwrap.dedent("""\
        models-start-port: 18000
        model-ports: 32
        defaults:
          ctx-size: 16384
        backends:
          rocm:
            runner: podman
            image: test-rocm-image:v1
            args:
              flash-attn: "on"
        models:
          test-rocm-podman:
            checkpoint: test/model:Q4_K_M
            backend: rocm
    """))


def cfg_rocm_docker() -> str:
    """rocm backend → runner: docker"""
    return _write_config(textwrap.dedent("""\
        models-start-port: 18000
        model-ports: 32
        defaults:
          ctx-size: 16384
        backends:
          rocm:
            runner: docker
            image: test-rocm-docker:v1
            args:
              flash-attn: "on"
        models:
          test-rocm-docker:
            checkpoint: test/model:Q4_K_M
            backend: rocm
    """))


# ── 1. Runner type resolution from config ───────────────────────────────────

class TestRunnerTypeFromConfig:
    """backends.<id>.runner determines which runner class is used."""

    def test_rocm_process_resolves_to_process(self):
        mr = ModelArkestra(cfg_rocm_process())
        assert mr._resolve_runner_type("test-rocm-process", {}, None) == "process"

    def test_rocm_podman_resolves_to_podman(self):
        mr = ModelArkestra(cfg_rocm_podman())
        assert mr._resolve_runner_type("test-rocm-podman", {}, None) == "podman"

    def test_rocm_docker_resolves_to_docker(self):
        mr = ModelArkestra(cfg_rocm_docker())
        assert mr._resolve_runner_type("test-rocm-docker", {}, None) == "docker"


# ── 2. Runner class registry ───────────────────────────────────────────────

class TestRunnerClassRegistry:
    """All three runner types are registered by default."""

    def test_process_registered(self):
        mr = ModelArkestra(_write_config("models:\n  m1:\n    checkpoint: x\n"))
        assert "process" in mr._runner_classes

    def test_podman_registered(self):
        mr = ModelArkestra(_write_config("models:\n  m1:\n    checkpoint: x\n"))
        assert "podman" in mr._runner_classes

    def test_docker_registered(self):
        mr = ModelArkestra(_write_config("models:\n  m1:\n    checkpoint: x\n"))
        assert "docker" in mr._runner_classes


# ── 3. Unknown runner rejected on start ────────────────────────────────────

class TestUnknownRunnerRejected:
    """A backend referencing a non-existent runner type must raise on start."""

    def test_model_using_unknown_runner_raises(self):
        cfg = _write_config(textwrap.dedent("""\
            models-start-port: 18000
            backends:
              unknown-backend:
                runner: nonexistent-runner
            models:
              m1:
                checkpoint: test/x:Q4
                backend: unknown-backend
        """))
        mr = ModelArkestra(cfg)
        with pytest.raises(ValueError, match="unknown runner type"):
            asyncio_run(mr.start("m1"))



# ── Backend resolution ───────────────────────────────────────
# Note: model.backend priority and global default are tested in
# tests/unit/test_resolve_defaults.py (TestBackendPriority).
# This module keeps only the edge case requiring a full app instance.

class TestBackendResolution:
    """Edge cases in backend resolution that need a full ModelArkestra instance."""
    def test_no_backend_falls_back(self):
        cfg = _write_config("models:\n  m1:\n    checkpoint: x\n")
        mr = ModelArkestra(cfg)
        # Falls back to BaseModelRunner._DEFAULT_BACKEND when no backend set
        assert mr._resolve_backend_id("m1", {}, None) == "cpu"

    def test_flat_backend_default_key(self):
        """backend-default at top level is respected."""
        cfg = _write_config(
            "backend-default: my-custom-backend\n"
            "models:\n  m1:\n    checkpoint: x\n"
        )
        mr = ModelArkestra(cfg)
        assert mr._resolve_backend_id("m1", {}, None) == "my-custom-backend"


# ── 5. Backend args are resolved correctly ────────────────────────────────

class TestBackendArgsResolution:
    """backends.<id>.args dict is accessible and structurally correct."""

    def test_rocm_process_has_flash_attn(self):
        be = ModelArkestra(cfg_rocm_process()).cm.get_backend("rocm")
        assert be["args"]["flash-attn"] == "on"

    def test_rocm_podman_has_flash_attn(self):
        be = ModelArkestra(cfg_rocm_podman()).cm.get_backend("rocm")
        assert be["args"]["flash-attn"] == "on"

    def test_rocm_docker_has_flash_attn(self):
        be = ModelArkestra(cfg_rocm_docker()).cm.get_backend("rocm")
        assert be["args"]["flash-attn"] == "on"


# ── 6. Container config keys (image, devices) ───────────────────────────

class TestContainerConfigKeys:
    """podman/docker backends carry image + optional devices/env_container."""

    def test_rocm_podman_has_image(self):
        be = ModelArkestra(cfg_rocm_podman()).cm.get_backend("rocm")
        assert be["image"] == "test-rocm-image:v1"

    def test_rocm_docker_has_image(self):
        be = ModelArkestra(cfg_rocm_docker()).cm.get_backend("rocm")
        assert be["image"] == "test-rocm-docker:v1"


# ── 7. Multiple backends with different runners in one config ─────────

class TestMixedBackendsInConfig:
    """A single config can host backends mapped to process, podman, docker."""

    def test_mixed_process_and_podman(self):
        cfg = _write_config(textwrap.dedent("""\
            models-start-port: 18000
            backends:
              vulkan:
                runner: process
              rocm-podman:
                runner: podman
                image: rocm:v1
            models:
              fast-model:
                checkpoint: test/fast:Q4
                backend: vulkan
              gpu-model:
                checkpoint: test/gpu:Q4
                backend: rocm-podman
        """))
        mr = ModelArkestra(cfg)
        assert mr._resolve_runner_type("fast-model", {}, None) == "process"
        assert mr._resolve_runner_type("gpu-model", {}, None) == "podman"

    def test_all_three_runners_in_one_config(self):
        cfg = _write_config(textwrap.dedent("""\
            models-start-port: 18000
            backends:
              fast:
                runner: process
              gpu-podman:
                runner: podman
                image: rocm:v1
              gpu-docker:
                runner: docker
                image: rocm-v2:1
            models:
              m-proc:
                checkpoint: test/m1:Q4
                backend: fast
              m-pod:
                checkpoint: test/m2:Q4
                backend: gpu-podman
              m-docker:
                checkpoint: test/m3:Q4
                backend: gpu-docker
        """))
        mr = ModelArkestra(cfg)
        assert mr._resolve_runner_type("m-proc", {}, None) == "process"
        assert mr._resolve_runner_type("m-pod", {}, None) == "podman"
        assert mr._resolve_runner_type("m-docker", {}, None) == "docker"


# ── 8. Env vars from config are dicts (not _Environ) ─────────────────

class TestConfigEnvResolution:
    """The `env:` section in config.yaml resolves to a plain dict."""

    def test_env_is_dict(self):
        cfg = _write_config(textwrap.dedent("""\
            env:
              HF_HUB_CACHE: /tmp/hf-cache
              MY_VAR: hello
            models:
              m1:
                checkpoint: test/m1:Q4
        """))
        mr = ModelArkestra(cfg)
        env = mr.cm.data.get("env", {})
        assert isinstance(env, dict), f"env resolved to {type(env).__name__}, expected dict"
        assert env["HF_HUB_CACHE"] == "/tmp/hf-cache"


# ── Helpers ─────────────────────────────────────────────────────────────

def asyncio_run(coro):
    """Run a coroutine synchronously (for use inside sync test methods)."""
    return asyncio.get_event_loop().run_until_complete(coro)
