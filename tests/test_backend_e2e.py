"""Unified backend runner end-to-end tests.

Auto-discovers all available backends at import time:
  - process (vulkan-radv): direct subprocess from pre-built binary
  - process (rocm-gfxXXX): direct subprocess, one per detected GPU family
  - docker: OCI container with GPU device passthrough
  - podman: OCI container with GPU device passthrough

Runs the same lifecycle + error-path tests across every backend combo.
No markers needed — runs by default and discovers what's available.

Run all e2e tests:
    pytest tests/test_backend_e2e.py -v --timeout=300

Run only one combination:
    pytest tests/test_backend_e2e.py::TestFullLifecycle::test_ainvoke[process-vulkan-process] -v
"""

from __future__ import annotations

import asyncio
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Models ───────────────────────────────────────────────────────────────────

# CPU models — small enough to run quickly on any machine
_CPU_MODELS = [
    ("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"),
    ("gemma-4-e2b", "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"),
]

# GPU models — use bartowski repos (open, no license gate)
_GPU_TEST_MODEL_REF = "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"

# ── Auto-discovery ───────────────────────────────────────────────────────────

def _detect_all_backends() -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Discover all available backends + pre-download binaries.

    Returns (combo_list, bin_paths) where:
      combo_list: list of (combo_id, model_name)
      bin_paths:  dict mapping process backend names -> binary directories
    """
    combos: List[Tuple[str, str]] = []
    bin_paths: Dict[str, str] = {}

    # --- Process: Vulkan CPU ---
    vulkan_dir = "/home/marc/local/llama.cpp/build-vulkan-radv/bin"
    if os.path.isdir(vulkan_dir) and os.path.isfile(os.path.join(vulkan_dir, "llama-server")):
        for model_key, model_ref in _CPU_MODELS:
            combo_id = f"process-vulkan-{model_key}"
            combos.append((combo_id, combo_id))

    # --- Process: ROCm per-gfx-family ---
    try:
        from model_arkestra.gpu_detect import detect_all as _detect_gpus
        detection = _detect_gpus()
        gfx_family = detection.get("gfx_family")

        for gpu in detection["gpus"]:
            backend_hint = gpu["backend"]
            if backend_hint not in ("rocm", "vulkan-radv", "cuda"):
                continue

            test_id_base = f"{gfx_family or backend_hint}-roc-m"
            combo_id = f"{test_id_base}-process"
            combos.append((combo_id, combo_id))

            # Pre-download ROCm binary for this gfx family
            if gfx_family:
                from model_arkestra.binary_downloader import BinaryDownloader
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
                    bin_dir = str(Path(result_path).parent)
                    # Register the process backend that will use this binary
                    for c in combos:
                        if c[0].startswith(test_id_base) and c[0].endswith("-process"):
                            bin_paths[c[0]] = bin_dir
                except Exception as e:
                    print(f"  Warning: failed to download {gfx_family}: {e}")

    except ImportError:
        pass

    # --- Process: CUDA (ai-dock/llama.cpp-cuda) ---
    try:
        from model_arkestra.gpu_detect import (
            has_nvidia, detect_cuda_compute_cap, get_cuda_gpu_names,
        )
        if has_nvidia():
            cuda_version = detect_cuda_compute_cap()
            gpu_names = get_cuda_gpu_names()
            gpu_tag = "-".join(gpu_names[:1]).replace(" ", "_") if gpu_names else "cuda"

            combo_id = f"process-cuda-{gpu_tag}"
            combos.append((combo_id, combo_id))

            # Pre-download CUDA binary from ai-dock releases
            if cuda_version:
                from model_arkestra.binary_downloader import BinaryDownloader
                asset_pattern = "llama.cpp-*-cuda-*-amd64.tar.gz"
                source_cfg = {
                    "type": "github-release",
                    "repo": "ai-dock/llama.cpp-cuda",
                    "release_type": "latest",
                    "asset_pattern": asset_pattern,
                }
                cache_dir = Path.home() / ".cache" / "arkestra-gpu-test"
                dl = BinaryDownloader(
                    cache_dir=cache_dir, backend_id="cuda-12.8", source_cfg=source_cfg,
                )
                try:
                    result_path = asyncio.run(dl.resolve(version="latest"))
                    bin_dir = str(Path(result_path).parent)
                    # Register the process backend that will use this binary
                    for c in combos:
                        if c[0].startswith("process-cuda-") and c[0].endswith("-process"):
                            bin_paths[c[0]] = bin_dir
                except Exception as e:
                    print(f"  Warning: failed to download CUDA binary: {e}")
    except ImportError:
        pass

    # --- Docker / Podman ---
    for runtime in ("docker", "podman"):
        if shutil.which(runtime):
            combo_id = f"{runtime}-gpu"
            combos.append((combo_id, combo_id))

    return combos, bin_paths


COMBOS = _detect_all_backends()
BACKEND_COMBOS: List[Tuple[str, str]] = COMBOS[0]
BIN_PATHS: Dict[str, str] = COMBOS[1]
COMBO_IDS = [c[0] for c in BACKEND_COMBOS] if BACKEND_COMBOS else []
if not BACKEND_COMBOS:
    BACKEND_COMBOS = [("no-backends", "test_none")]

# ── Config builder ───────────────────────────────────────────────────────────

def _build_e2e_config(combos: List[Tuple[str, str]], bin_paths: Dict[str, str]) -> str:
    """Build a YAML config string from the combo list."""
    backend_cfgs: Dict[str, Dict[str, Any]] = {}
    test_models: List[Dict[str, Any]] = []

    for combo_id, model_name in combos:
        # Determine backend config from combo_id
        model_key = None  # set by CPU model branches below
        if combo_id.startswith("process-vulkan-"):
            model_key = combo_id.split("-", 2)[-1]
            backend_name = "vulkan-process"
        elif "roc-m" in combo_id:
            gpu_prefix = combo_id.replace("-roc-m-process", "").rsplit("-", 1)[-1]
            backend_name = f"rocm-process-{gpu_prefix}"
        elif combo_id.startswith("process-cuda-"):
            gpu_tag = combo_id.replace("process-cuda-", "")
            backend_name = f"cuda-process-{gpu_tag}"
        elif combo_id.startswith("docker-") or combo_id.startswith("podman-"):
            backend_name = f"{combo_id.split('-')[0]}-backend"
        else:
            continue

        if model_key is not None:
            entry = next((m for m in _CPU_MODELS if m[0] == model_key), (_GPU_TEST_MODEL_REF,))
            model_ref = entry[1]
        else:
            model_ref = _GPU_TEST_MODEL_REF

        test_models.append({
            "model_name": combo_id,
            "model": model_ref,
            "backend": backend_name,
            "args": {"temp": 0.7, "top-p": 0.95, "ctx-size": 2048},
        })

        if backend_name in backend_cfgs:
            continue  # deduplicate — docker/podman each have one backend for all GPU models

        # Build backend config
        be: Dict[str, Any] = {}

        if combo_id.startswith("process-vulkan-"):
            vulkan_dir = "/home/marc/local/llama.cpp/build-vulkan-radv/bin"
            be.update({
                "runner": "process",
                "binary_dir": vulkan_dir,
                "binary": "llama-server",
                "args": {},
            })

        elif "roc-m" in combo_id:
            # Look up pre-downloaded binary by matching prefix
            gfx_prefix = combo_id.replace("-roc-m-process", "")
            for bkey, bdir in bin_paths.items():
                if gfx_prefix in bkey or bkey.startswith(gfx_prefix):
                    be.update({
                        "runner": "process",
                        "binary_dir": bdir,
                        "binary": str(Path(bdir).name),
                        "env_container": {"LD_LIBRARY_PATH": bdir},
                        "args": {"ngl": 999},
                    })
                    break
            if not be:
                be = {"runner": "process", "source_ref": f"rocm-{gfx_prefix}"}

        elif combo_id.startswith("process-cuda-"):
            # CUDA process — look up pre-downloaded binary
            gpu_tag = combo_id.replace("process-cuda-", "").replace("-process", "")
            for bkey, bdir in bin_paths.items():
                if "cuda" in bkey.lower() or "ai-dock" in bkey.lower():
                    be.update({
                        "runner": "process",
                        "binary_dir": bdir,
                        "binary": str(Path(bdir).name),
                        "env_container": {"LD_LIBRARY_PATH": bdir},
                        "args": {"ngl": 999},
                    })
                    break
            if not be:
                be = {"runner": "process", "source_ref": f"cuda-{gpu_tag}"}

        elif combo_id.startswith("docker-") or combo_id.startswith("podman-"):
            runtime = combo_id.split("-")[0]
            be.update({
                "runner": runtime,
                "source_ref": f"{runtime}-gpu",
                "entrypoint": "/usr/local/bin/llama-server",
                "devices": ["/dev/kfd", "/dev/dri"],
                "args": {"ngl": 999},
            })

        backend_cfgs[backend_name] = be

    # ── YAML serialization helpers ────────────────────────────────────────
    def _fmt(v: Any) -> str:
        if isinstance(v, bool): return "true" if v else "false"
        return str(v)

    def _serialize(cfg: dict, indent: int) -> List[str]:
        out = []
        pfx = "  " * indent
        for k, v in cfg.items():
            if isinstance(v, dict):
                out.append(f"{pfx}{k}:")
                out.extend(_serialize(v, indent + 1))
            elif isinstance(v, list):
                out.append(f"{pfx}{k}:")
                for item in v:
                    out.append(f"{pfx}- {_fmt(item)}")
            else:
                out.append(f"{pfx}{k}: {_fmt(v)}")
        return out

    # ── Build YAML ────────────────────────────────────────────────────────
    lines = ["backends:"]
    for be_name, cfg in backend_cfgs.items():
        lines.append(f"  {be_name}:")
        lines.extend(_serialize(cfg, 2))

    # OCI sources for container backends
    oci_refs = set(
        cfg.get("source_ref") for cfg in backend_cfgs.values()
        if isinstance(cfg.get("source_ref"), str) and cfg["source_ref"] not in ("process", "docker", "podman")
    )
    if oci_refs:
        lines.extend(["", "sources:"])
        for src in sorted(oci_refs):
            lines.extend([f"  {src}:", "    type: oci-image", "    repo: docker.io/kyuz0/amd-strix-halo-toolboxes", "    release_type: rocm-7.14"])

    lines.append("")
    # Runners section
    has_process = any("process" in cfg for cfg in backend_cfgs.values() if isinstance(cfg, dict))
    if has_process:
        lines.extend(["runners:", "  default: ProcessModelRunner", ""])

    # Models section
    lines.append("models:")
    for m in test_models:
        lines.extend([f"  {m['model_name']}:", f"    model: {m['model']}", f"    backend: {m['backend']}", "    args:"])
        for k, v in m["args"].items():
            lines.append(f"      {k}: {v}")

    return "\n".join(lines)


# ── Fixture helpers ──────────────────────────────────────────────────────────

E2E_PORT = 18005


def _start_server(port: int, admin_key: str = "test-e2e-key") -> Tuple[Any, httpx.Client]:
    from model_arkestra.server import ArkestraServer
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(_build_e2e_config(BACKEND_COMBOS, BIN_PATHS))
        config_path = f.name

    try:
        proxy = ArkestraServer(config_path=config_path, port=port, ready_timeout=60)
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
        for m in r.json()["models"]:
            if m["id"] == model_name and m.get("status", {}).get("value") == "loaded":
                return True
        time.sleep(0.5)
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


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def e2e_server(request):
    """Per-test self-contained server + client with guaranteed cleanup."""
    admin_key = "test-e2e-key"
    base_url = f"http://127.0.0.1:{E2E_PORT}"

    proxy, client = _start_server(E2E_PORT, admin_key)

    request.instance._e2e_port = E2E_PORT
    request.instance._e2e_base_url = base_url
    request.instance._e2e_client = client

    yield {"server": proxy, "client": client, "port": E2E_PORT, "base_url": base_url}

    try:
        _stop_server(proxy, client, E2E_PORT)
    except Exception:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────

class TestFullLifecycle:
    """Start → inference → logs → stop/start. Clean slate each time."""

    @pytest.mark.e2e
    @pytest.mark.parametrize("combo_id,model_name", BACKEND_COMBOS, ids=COMBO_IDS)
    def test_ainvoke(self, e2e_server, combo_id: str, model_name: str):
        """Single message → non-streaming response."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            if not ok:
                log_resp = client.get(f"{base_url}/admin/log/{model_name}", params={"since": 0, "lines": 100}, timeout=10)
                if log_resp.status_code == 200:
                    for line in log_resp.json().get("lines", []):
                        print(f"LOG: {line['text']}")
            assert ok, f"Model {model_name} ({combo_id}) failed to start"

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
            assert len(content) > 0

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.e2e
    @pytest.mark.parametrize("combo_id,model_name", BACKEND_COMBOS, ids=COMBO_IDS)
    def test_astream(self, e2e_server, combo_id: str, model_name: str):
        """Single message → streaming response with SSE chunks."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} ({combo_id}) failed to start"

            resp = client.post(f"{base_url}/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "List 1, 2, 3"}],
                "max_new_tokens": 16, "stream": True,
            }, timeout=60)

            assert resp.status_code == 200
            assert "[DONE]" in resp.text, "Missing [DONE] marker"
            data_lines = [
                l.removeprefix("data: ").strip()
                for l in resp.text.split("\n")
                if l.startswith("data: ") and "[DONE]" not in l
            ]
            assert len(data_lines) > 0, "No token chunks in stream"

        finally:
            _stop_model(client, base_url, model_name)

    @pytest.mark.e2e
    def test_start_nonexistent_returns_503_via_chat(self, e2e_server):
        """Chat endpoint for unstarted model returns 503."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        resp = client.post(f"{base_url}/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
        }, timeout=5)
        assert resp.status_code == 503
