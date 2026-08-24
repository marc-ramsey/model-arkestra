"""GPU backend end-to-end tests.

Auto-detects available GPUs via ``gpu_detect`` and runs a full lifecycle test
(start → inference → logs → stop/start cycle) for each detected GPU × runner
combination:
  - process   : direct subprocess (requires GPU driver on host)
  - docker    : container with device passthrough
  - podman    : container with device passthrough

GPU-specific backends use small open models from bartowski that download quickly.

Run all GPU tests:
    pytest tests/test_gpu_e2e.py --gpu -v --timeout=600

Run only one GPU + runner combination:
    pytest tests/test_gpu_e2e.py::test_ainvoke[gfx1151-process] -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pytest
import yaml
import uvicorn

# ── GPU detection (used to build test matrix) ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_arkestra.gpu_detect import detect_all, has_rocm, has_vulkan


# ── Test model selection ─────────────────────────────────────────────────────
# Small open models that download quickly and work on both ROCm and Vulkan.
_GPU_TEST_MODEL = "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"


def _build_gpu_combos() -> List[Tuple[str, str]]:
    """Build (test_id, model_name) from detected hardware.

    Each GPU × runner combination gets its own entry so that process,
    docker, and podman are tested independently.
    """
    combos: List[Tuple[str, str]] = []
    detection = detect_all()
    gpus = detection["gpus"]

    for gpu in gpus:
        backend_hint = gpu["backend"]  # 'rocm', 'vulkan-radv', 'cuda'
        gfx_family = detection.get("gfx_family")  # e.g. 'gfx1151', 'gfx110X'

        runner_types: List[str] = ["process"]
        if shutil.which("docker"):
            runner_types.append("docker")
        if shutil.which("podman"):
            runner_types.append("podman")

        for runner in runner_types:
            # Skip roc-containers unless we have a ROCm backend configured
            if backend_hint not in ("rocm", "vulkan-radv", "cuda"):
                continue

            test_id = f"{gfx_family or backend_hint}-{runner}" if gfx_family else f"{backend_hint}-{runner}"
            model_name = f"gpu_test_{test_id.replace('-', '_')}"
            combos.append((test_id, model_name))

    return combos


GPU_COMBOS = _build_gpu_combos()
if GPU_COMBOS:
    _COMBO_IDS = [c[0] for c in GPU_COMBOS]
else:
    _COMBO_IDS = ["no-gpu"]
    GPU_COMBOS = [("no-gpu", "gpu_test_none")]


# ── E2E config builder ───────────────────────────────────────────────────────

def _build_e2e_config(combos: List[Tuple[str, str, str]]) -> str:
    """Build a YAML config string for the given GPU × runner combos."""
    backends_section = {
        "backends": {},
        "sources": {},
    }

    # Collect unique backend configs needed
    backend_cfgs: Dict[str, Dict[str, Any]] = {}

    test_models = []
    for test_id, model_name, runner in combos:
        parts = test_id.rsplit("-", 1)
        backend_hint = parts[0]  # gfx1151, gfx110X, cuda, etc.

        # Determine the actual backend name (deduplicate)
        if backend_hint not in backend_cfgs:
            if runner == "process":
                runner_class = "ProcessModelRunner"
                backend_name = f"gpu_{backend_hint}_proc"
            elif runner == "docker":
                runner_class = "DockerModelRunner"
                backend_name = f"gpu_{backend_hint}_docker"
            else:  # podman
                runner_class = "PodmanModelRunner"
                backend_name = f"gpu_{backend_hint}_podman"

            if backend_hint in ("rocm", "gfx1151", "gfx110X", "gfx1150",
                                "gfx120X", "gfx90a", "gfx103X"):
                # ROCm: need device passthrough for container runners
                source_ref = f"rocm-{backend_hint}" if backend_hint.startswith("gfx") else "rocm-gfx1151"
                devices = ["/dev/kfd", "/dev/dri"]
            elif backend_hint == "cuda":
                source_ref = "official-cuda"
                devices = []
            elif backend_hint == "vulkan-radv":
                source_ref = "official-vulkan-radv"
                devices = []
            else:
                source_ref = None
                devices = []

            entrypoint = "/usr/local/bin/llama-server" if runner != "process" else None

            backend_cfgs[backend_name] = {
                "runner": runner_class,
                "source_ref": source_ref,
            }
            if devices:
                backend_cfgs[backend_name]["devices"] = devices
            if entrypoint:
                backend_cfgs[backend_name]["entrypoint"] = entrypoint
            backend_cfgs[backend_name]["args"] = {
                "ngl": 999,
                "hf": "${CHECKPOINT}",
            }

        # Build the model config (reuses its own backend name)
        test_models.append({
            "model_name": model_name,
            "checkpoint": _GPU_TEST_MODEL,
            "backend": backend_name,
            "args": {
                "temp": 0.7,
                "top-p": 0.95,
                "ctx-size": 2048,
                "max_tokens": 32,
            },
        })

    # Build full YAML
    config_lines = ["backends:"]
    for be_name, cfg in backend_cfgs.items():
        config_lines.append(f"  {be_name}:")
        for k, v in cfg.items():
            if isinstance(v, list):
                config_lines.append(f"    {k}:")
                for item in v:
                    config_lines.append(f"      - {item}")
            else:
                config_lines.append(f"    {k}: {v}")

    # Sources section (only for non-ROCm containers which use OCI)
    sources_needed = set()
    for cfg in backend_cfgs.values():
        sr = cfg.get("source_ref")
        if sr and sr not in ("rocm-gfx1151",):  # ROCm already handled via backends.yaml
            sources_needed.add(sr)

    if sources_needed:
        config_lines.append("")
        config_lines.append("sources:")
        for src in sorted(sources_needed):
            config_lines.append(f"  {src}:")
            config_lines.append(f"    type: oci-image")
            config_lines.append(f"    repo: docker.io/kyuz0/amd-strix-halo-toolboxes")
            config_lines.append(f"    release_type: rocm-7.14")

    # Models section
    config_lines.append("")
    config_lines.append("models:")
    for m in test_models:
        config_lines.append(f"  {m['model_name']}:")
        config_lines.append(f"    checkpoint: {m['checkpoint']}")
        config_lines.append(f"    backend: {m['backend']}")
        config_lines.append(f"    args:")
        for k, v in m["args"].items():
            config_lines.append(f"      {k}: {v}")

    return "\n".join(config_lines)


# ── Fixture helpers (same pattern as CPU e2e tests) ─────────────────────────

def _start_server(port: int, config_yaml: str, admin_key: str = "test-gpu-key") -> tuple[Any, httpx.Client]:
    """Start an ArkestraServer with the given YAML config."""
    from model_arkestra.server import ArkestraServer
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    try:
        proxy = ArkestraServer(
            config_path=config_path,
            port=port,
            ready_timeout=60,
        )
        app = proxy.get_app()

        uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server_obj = uvicorn.Server(uvicorn_config)
        proxy._server = server_obj  # type: ignore

        def serve():
            asyncio.run(server_obj.serve())

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                httpx.get(url, timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise RuntimeError(f"Server on port {port} did not become ready")

        client = httpx.Client(timeout=None, headers={"X-Admin-Key": admin_key})
        return proxy, client
    except Exception:
        os.unlink(config_path)
        raise


def _stop_server(proxy: Any, client: httpx.Client, port: int) -> None:
    """Tear down server and clean up containers."""
    try:
        client.post(f"http://127.0.0.1:{port}/admin/shutdown", timeout=120)
    except Exception:
        pass

    # Clean up leftover containers
    for cmd in (["podman", "ps", "-a", "--filter", "name=llm-", "--format", "{{.ID}}"],
                ["docker", "ps", "-a", "--filter", "name=llm-", "--format", "{{.ID}}"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            for cid in result.stdout.strip().split():
                if cid:
                    subprocess.run(["podman", "rm", "-f", cid], capture_output=True, timeout=5)
                    subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5)
        except Exception:
            pass

    _wait_port_free(port, timeout=30)

    try:
        client.close()
    except Exception:
        pass


def _wait_port_free(port: int, timeout: float = 20.0) -> bool:
    """Block until no process is listening on *port*."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5,
        )
        if f":{port}" not in result.stdout:
            return True
        time.sleep(0.3)
    subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)
    return False


def _start_model(client: httpx.Client, base_url: str, model_name: str) -> bool:
    """Start a model and wait for 'running' state."""
    resp = client.post(f"{base_url}/admin/start/{model_name}", timeout=300)  # GPU downloads take longer
    if resp.status_code != 200:
        return False

    deadline = time.time() + 180  # Give model time to start inference
    while time.time() < deadline:
        r = client.get(f"{base_url}/admin/models", timeout=10)
        for m in r.json()["models"]:
            if m["id"] == model_name and m.get("status") == "running":
                return True
        time.sleep(0.5)
    return False


def _stop_model(client: httpx.Client, base_url: str, model_name: str) -> None:
    """Stop a model."""
    try:
        client.post(f"{base_url}/admin/stop/{model_name}", timeout=60)
    except Exception:
        pass
    try:
        client.post(f"{base_url}/admin/stop-all", timeout=10)
    except Exception:
        pass


# ── Fixtures ─────────────────────────────────────────────────────────────────

GPU_BASE_PORT = 19000  # Separate port range from CPU e2e tests


@pytest.fixture()
def gpu_server(request):
    """Per-test self-contained server + client for GPU testing."""
    if not GPU_COMBOS:
        pytest.skip("No GPUs detected on this machine")

    admin_key = "test-gpu-key"
    base_url = f"http://127.0.0.1:{GPU_BASE_PORT}"

    # Build config for this test's combo
    param = request.node.callspec.params  # type: ignore
    model_name = param.get("model_name")
    test_id = param.get("test_id")

    if not model_name:
        pytest.skip("No model parameters found")

    # Find the combos to build config from (all GPU combos share the same server)
    gpu_config = _build_e2e_config(GPU_COMBOS)

    proxy, client = _start_server(GPU_BASE_PORT, gpu_config, admin_key)

    request.instance._gpu_port = GPU_BASE_PORT
    request.instance._gpu_base_url = base_url
    request.instance._gpu_client = client

    yield {"server": proxy, "client": client, "port": GPU_BASE_PORT, "base_url": base_url}

    try:
        _stop_server(proxy, client, GPU_BASE_PORT)
    except Exception:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────

class TestGPULifecycle:
    """GPU lifecycle: start → inference → logs → stop/start cycle."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_ainvoke(self, gpu_server, test_id: str, model_name: str):
        """Single message → non-streaming chat response on GPU."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start on GPU"

            resp = client.post(f"{base_url}/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say one word: hello"}],
                "max_tokens": 8,
            }, timeout=60)

            assert resp.status_code == 200, f"Inference failed: {resp.text}"
            body = resp.json()
            assert body["object"] == "chat.completion"
            assert len(body["choices"]) > 0
            content = body["choices"][0]["message"]["content"]
            assert len(content) > 0, "Empty response from GPU model"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_astream(self, gpu_server, test_id: str, model_name: str):
        """Streaming response from GPU."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start on GPU"

            resp = client.post(f"{base_url}/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "List 1, 2, 3"}],
                "max_tokens": 16,
                "stream": True,
            }, timeout=60)

            assert resp.status_code == 200
            assert "[DONE]" in resp.text, "Missing [DONE] marker"

            data_lines = [
                line.removeprefix("data: ").strip()
                for line in resp.text.split("\n")
                if line.startswith("data: ") and "[DONE]" not in line
            ]
            assert len(data_lines) > 0, "No token chunks in GPU stream"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_logs_captured(self, gpu_server, test_id: str, model_name: str):
        """GPU model logs appear via /admin/log endpoint."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start on GPU"

            time.sleep(2)

            resp = client.get(f"{base_url}/admin/log/{model_name}", params={
                "since": 0, "lines": 100
            }, timeout=10)

            assert resp.status_code == 200
            data = resp.json()
            assert "lines" in data
            assert len(data["lines"]) > 0, f"No GPU log lines for {model_name}"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_stop_then_start_restores_stopped_state(self, gpu_server, test_id: str, model_name: str):
        """After stopping a GPU model, it reports 'stopped'. Restart works."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start on GPU"

            _stop_model(client, base_url, model_name)

            r = client.get(f"{base_url}/admin/models", timeout=10)
            for m in r.json()["models"]:
                if m["id"] == model_name:
                    assert m["status"] == "stopped", (
                        f"Expected 'stopped' after stop, got '{m['status']}'"
                    )

        finally:
            _stop_model(client, base_url, model_name)


class TestGPUErrorPaths:
    """GPU error handling."""

    @pytest.mark.gpu
    def test_unknown_model_404(self, gpu_server):
        """Starting a non-existent GPU model returns 404."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        resp = client.post(f"{base_url}/admin/start/nonexistent-gpu-model", timeout=10)
        assert resp.status_code == 404

    @pytest.mark.gpu
    def test_start_nonexistent_returns_503(self, gpu_server):
        """Chat endpoint for unstarted GPU model returns 503."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        resp = client.post(f"{base_url}/v1/chat/completions", json={
            "model": "nonexistent-gpu-model",
            "messages": [{"role": "user", "content": "hi"}],
        }, timeout=5)
        assert resp.status_code == 503
