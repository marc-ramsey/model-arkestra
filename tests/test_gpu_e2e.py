"""GPU backend end-to-end tests.

Auto-detects available GPUs via gpu_detect and runs a full lifecycle test
(start -> inference -> logs -> stop/start cycle) for each detected GPU with:
  - process   : direct subprocess (requires GPU driver on host)
  - docker    : container with device passthrough
  - podman    : container with device passthrough

GPU-specific backends use small open models from bartowski.

Run all GPU tests:
    pytest tests/test_gpu_e2e.py --gpu -v --timeout=600
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
import uvicorn

# GPU detection for test matrix
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_arkestra.gpu_detect import detect_all


_GPU_TEST_MODEL = "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"

# Pre-downloaded ROCm binary paths for process mode (gfx_family -> binary_dir)
_PROCESS_BIN_PATHS: Dict[str, str] = {}


def _pre_download_rocm_binary(gfx_family: str) -> str | None:
    """Download ROCm binary for a gfx family. Returns binary_dir or None."""
    try:
        from model_arkestra.binary_downloader import BinaryDownloader
    except ImportError:
        return None

    asset_pattern = f"llama-*-ubuntu-rocm-{gfx_family}-x64.zip"
    source_cfg = {
        "type": "github-release",
        "repo": "lemonade-sdk/llamacpp-rocm",
        "release_type": "latest",
        "asset_pattern": asset_pattern,
    }
    cache_dir = Path.home() / ".cache" / "arkestra-gpu-test"
    dl = BinaryDownloader(
        cache_dir=cache_dir, backend_id=f"rocm-{gfx_family}", source_cfg=source_cfg,
    )
    try:
        result_path = asyncio.run(dl.resolve(version="latest"))
    except Exception as e:
        print(f"  Warning: failed to download {gfx_family}: {e}")
        return None
    _PROCESS_BIN_PATHS[gfx_family] = str(Path(result_path).parent)
    return _PROCESS_BIN_PATHS[gfx_family]


def _build_gpu_combos() -> List[Tuple[str, str]]:
    """Build (test_id, model_name) from detected hardware + pre-downloaded binaries."""
    combos: List[Tuple[str, str]] = []
    detection = detect_all()

    gfx_family = detection.get("gfx_family")
    if gfx_family:
        _pre_download_rocm_binary(gfx_family)

    for gpu in detection["gpus"]:
        backend_hint = gpu["backend"]  # 'rocm', 'vulkan-radv', 'cuda'
        for runner in ["process", *(["docker"] if shutil.which("docker") else []), *(["podman"] if shutil.which("podman") else [])]:
            if backend_hint not in ("rocm", "vulkan-radv", "cuda"):
                continue
            test_id = f"{gfx_family or backend_hint}-{runner}"
            model_name = f"gpu_test_{test_id.replace('-', '_')}"
            combos.append((test_id, model_name))

    return combos


GPU_COMBOS = _build_gpu_combos()
if GPU_COMBOS:
    _COMBO_IDS = [c[0] for c in GPU_COMBOS]
else:
    _COMBO_IDS = ["no-gpu"]
    GPU_COMBOS = [("no-gpu", "gpu_test_none")]


def _build_e2e_config(combos: List[Tuple[str, str]]) -> str:
    """Build a YAML config string for the given GPU x runner combos."""
    backend_cfgs: Dict[str, Dict[str, Any]] = {}
    test_models: List[Dict[str, Any]] = []

    for test_id, model_name in combos:
        parts = test_id.rsplit("-", 1)
        backend_hint = parts[0]
        runner = parts[-1] if len(parts) > 1 else "process"

        # Per-runner backend name
        if runner == "process":
            backend_name = f"gpu_{backend_hint}_proc"
        elif runner == "docker":
            backend_name = f"gpu_{backend_hint}_docker"
        else:
            backend_name = f"gpu_{backend_hint}_podman"

        if backend_name in backend_cfgs:
            test_models.append({
                "model_name": model_name, "checkpoint": _GPU_TEST_MODEL,
                "backend": backend_name,
                "args": {"temp": 0.7, "top-p": 0.95, "ctx-size": 2048},
            })
            continue

        runner_class = {"process": "process", "docker": "docker", "podman": "podman"}[runner]

        # Source reference (for container/OCI resolution)
        if backend_hint.startswith("gfx") and runner != "process":
            source_ref = f"rocm-{backend_hint}"
        elif backend_hint == "cuda":
            source_ref = "official-cuda"
        elif backend_hint == "vulkan-radv":
            source_ref = "official-vulkan-radv"
        else:
            source_ref = None

        # Process mode uses pre-downloaded binary path
        if runner == "process" and backend_hint in _PROCESS_BIN_PATHS:
            bin_dir = _PROCESS_BIN_PATHS[backend_hint]
            binary_name = str(Path(bin_dir).name)
        else:
            bin_dir = ""
            binary_name = ""

        devices = ["/dev/kfd", "/dev/dri"] if runner != "process" else []
        entrypoint = "/usr/local/bin/llama-server" if runner != "process" else None
        port_arg = {"port": "${PORT}"} if runner == "process" else {}

        backend_cfgs[backend_name] = {"runner": runner_class, "source_ref": source_ref}
        if bin_dir:
            backend_cfgs[backend_name]["binary_dir"] = bin_dir
            backend_cfgs[backend_name]["binary"] = binary_name
        if devices:
            backend_cfgs[backend_name]["devices"] = devices
        if entrypoint:
            backend_cfgs[backend_name]["entrypoint"] = entrypoint
        if bin_dir and runner == "process":
            backend_cfgs[backend_name]["env_container"] = {"LD_LIBRARY_PATH": bin_dir}
        backend_cfgs[backend_name]["args"] = {"ngl": 999, "hf": "${CHECKPOINT}", **port_arg}

        test_models.append({
            "model_name": model_name, "checkpoint": _GPU_TEST_MODEL,
            "backend": backend_name,
            "args": {"temp": 0.7, "top-p": 0.95, "ctx-size": 2048},
        })

    # Build YAML helper: serialize a config dict to YAML lines
    def _yaml_serialize(cfg: dict, indent: int) -> list:
        """Serialize a config dict to YAML-indented lines."""
        out = []
        prefix = "  " * indent
        for k, v in cfg.items():
            if isinstance(v, dict):
                out.append(f"{prefix}{k}:")
                out.extend(_yaml_serialize(v, indent + 1))
            elif isinstance(v, list):
                out.append(f"{prefix}{k}:")
                for item in v:
                    if isinstance(item, dict):
                        first = True
                        for ik, iv in item.items():
                            if first:
                                out.append(f"{prefix}- {ik}: {_fmt_val(iv)}")
                                first = False
                            else:
                                out.append(f"{prefix}  {ik}: {_fmt_val(iv)}")
                    else:
                        out.append(f"{prefix}- {_fmt_val(item)}")
            else:
                out.append(f"{prefix}{k}: {_fmt_val(v)}")
        return out

    def _fmt_val(v: Any) -> str:
        """Format a scalar value for YAML."""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (dict, list)):
            return str(v)
        return str(v)

    # Backends
    lines = ["backends:"]
    for be_name, cfg in backend_cfgs.items():
        lines.append(f"  {be_name}:")
        lines.extend(_yaml_serialize(cfg, 2))

    # OCI sources for container ROCm (gfx families and general)
    oci_refs = set()
    for cfg in backend_cfgs.values():
        sr = cfg.get("source_ref")
        if sr and sr not in ("process", "docker", "podman"):
            oci_refs.add(sr)
    if oci_refs:
        lines.extend(["", "sources:"])
        for src in sorted(oci_refs):
            lines.extend([f"  {src}:", "    type: oci-image", "    repo: docker.io/kyuz0/amd-strix-halo-toolboxes", "    release_type: rocm-7.14"])

    lines.extend(["", "models:"])
    for m in test_models:
        lines.extend([f"  {m['model_name']}:", f"    checkpoint: {m['checkpoint']}", f"    backend: {m['backend']}", "    args:"])
        for k, v in m["args"].items():
            lines.append(f"      {k}: {v}")

    return "\n".join(lines)


# ── Fixtures & helpers ───────────────────────────────────────────────────────

GPU_BASE_PORT = 19000


def _start_server(port: int, config_yaml: str, admin_key: str = "test-gpu-key") -> Tuple[Any, httpx.Client]:
    from model_arkestra.server import ArkestraServer
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    try:
        proxy = ArkestraServer(config_path=config_path, port=port, ready_timeout=60)
        app = proxy.get_app()
        uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server_obj = uvicorn.Server(uvicorn_config)
        proxy._server = server_obj

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

        return proxy, httpx.Client(timeout=None, headers={"X-Admin-Key": admin_key})
    except Exception:
        os.unlink(config_path)
        raise


def _stop_server(proxy: Any, client: httpx.Client, port: int) -> None:
    try:
        client.post(f"http://127.0.0.1:{port}/admin/shutdown", timeout=120)
    except Exception:
        pass

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
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
        if f":{port}" not in result.stdout:
            return True
        time.sleep(0.3)
    subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)
    return False


def _start_model(client: httpx.Client, base_url: str, model_name: str) -> bool:
    resp = client.post(f"{base_url}/admin/start/{model_name}", timeout=300)
    if resp.status_code != 200:
        return False

    deadline = time.time() + 180
    while time.time() < deadline:
        r = client.get(f"{base_url}/admin/models", timeout=10)
        models = r.json().get("models", [])
        for m in models:
            if m["id"] == model_name:
                if m.get("status") == "running":
                    return True
                # Log status transitions for debugging
                pass  # keep polling
        time.sleep(0.5)
    
    # Check final state
    if models:
        m = next((m for m in models if m["id"] == model_name), None)
        if m:
            print(f"DEBUG: {model_name} final status={m.get('status')}, port={m.get('port')}")
    return False


def _stop_model(client: httpx.Client, base_url: str, model_name: str) -> None:
    try:
        client.post(f"{base_url}/admin/stop/{model_name}", timeout=60)
    except Exception:
        pass
    try:
        client.post(f"{base_url}/admin/stop-all", timeout=10)
    except Exception:
        pass


@pytest.fixture()
def gpu_server(request):
    """Per-test self-contained server + client for GPU testing."""
    if not GPU_COMBOS:
        pytest.skip("No GPUs detected on this machine")

    admin_key = "test-gpu-key"
    gpu_config = _build_e2e_config(GPU_COMBOS)
    proxy, client = _start_server(GPU_BASE_PORT, gpu_config, admin_key)

    yield {
        "server": proxy, "client": client,
        "port": GPU_BASE_PORT, "base_url": f"http://127.0.0.1:{GPU_BASE_PORT}",
    }

    try:
        _stop_server(proxy, client, GPU_BASE_PORT)
    except Exception:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────

class TestGPULifecycle:
    """GPU lifecycle tests."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_ainvoke(self, gpu_server, test_id: str, model_name: str):
        """Single message -> non-streaming chat response on GPU."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            if not ok:
                # Dump logs for diagnostics
                log_resp = client.get(f"{base_url}/admin/log/{model_name}", params={"since": 0, "lines": 100}, timeout=10)
                if log_resp.status_code == 200:
                    for line in log_resp.json().get("lines", []):
                        print(f"LOG: {line['text']}")
            assert ok, f"Model {model_name} ({test_id}) failed to start on GPU"

            resp = client.post(f"{base_url}/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say one word: hello"}],
                "max_new_tokens": 8,
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
                "max_new_tokens": 16, "stream": True,
            }, timeout=60)

            assert resp.status_code == 200
            assert "[DONE]" in resp.text, "Missing [DONE] marker"
            data_lines = [l.removeprefix("data: ").strip() for l in resp.text.split("\n") if l.startswith("data: ") and "[DONE]" not in l]
            assert len(data_lines) > 0, "No token chunks in GPU stream"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_logs_captured(self, gpu_server, test_id: str, model_name: str):
        """GPU model logs via /admin/log."""
        if not GPU_COMBOS:
            pytest.skip("No GPUs detected")

        client = gpu_server["client"]
        base_url = gpu_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({test_id}) failed to start on GPU"
            time.sleep(2)

            resp = client.get(f"{base_url}/admin/log/{model_name}", params={"since": 0, "lines": 100}, timeout=10)
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["lines"]) > 0, f"No GPU log lines for {model_name}"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.gpu
    @pytest.mark.parametrize("test_id,model_name", GPU_COMBOS, ids=[c[0] for c in GPU_COMBOS])
    def test_stop_then_start_restores_stopped_state(self, gpu_server, test_id: str, model_name: str):
        """After stopping a GPU model, it reports 'stopped'."""
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
                    assert m["status"] == "stopped", f"Expected 'stopped', got '{m['status']}'"

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
            "model": "nonexistent-gpu-model", "messages": [{"role": "user", "content": "hi"}],
        }, timeout=5)
        assert resp.status_code == 503
