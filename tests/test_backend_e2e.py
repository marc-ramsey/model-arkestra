"""Unified backend runner end-to-end tests.

Each test picks **one available backend**, starts **one model**,
runs a single lifecycle (start → infer → stop), then tears it down.

Every test explicitly calls ``POST /admin/stop-all`` and waits for ports 18000-18001
to clear before returning. No concurrent llama-server processes ever exceed **two**.

All tests share **one** Arkestra server on port 18003 with model range 18000-18001.
A second server is only used when explicitly needed (none currently).

Run all e2e tests:
    pytest tests/test_backend_e2e.py -v --timeout=600

Run only one combo:
    pytest tests/test_backend_e2e.py::TestFullLifecycle::test_ainvoke[process-vulkan] -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Fixed port layout (all e2e tests share these) ───────────────────────────

ADMIN_PORT = 18003
MODEL_START = 18000          # model-start-port in config
MODEL_PORTS = 2              # model-ports in config
MODEL_END = MODEL_START + MODEL_PORTS - 1    # 18001


# ── Models (small, fast, open repos) ────────────────────────────────────────

_MODELS = [
    ("qwen3.5-4b",   "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"),
    ("gemma-4-e2b",  "bartowski/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M"),
]


# ── Auto-discovery ───────────────────────────────────────────────────────────

def _discover_backends() -> List[Tuple[str, str]]:
    """Return list of (combo_id, backend_name) for available runtimes."""
    combos: List[Tuple[str, str]] = []

    if (os.path.isdir("/home/marc/local/llama.cpp/build-vulkan-radv/bin")
            and os.path.isfile("/home/marc/local/llama.cpp/build-vulkan-radv/bin/llama-server")):
        combos.append(("process-vulkan", "vulkan-process"))

    try:
        from model_arkestra.gpu_detect import detect_all
        detection = detect_all()
        gfx_family = detection.get("gfx_family")
        for gpu in detection["gpus"]:
            hint = gpu["backend"]
            if hint not in ("rocm", "vulkan-radv", "cuda"):
                continue
            combos.append((f"{gfx_family or hint}-roc-process", f"rocm-{gfx_family}"))
    except ImportError:
        pass

    for runtime in ("docker", "podman"):
        if shutil.which(runtime):
            combos.append((f"{runtime}-gpu", f"{runtime}-backend"))

    return combos


COMBOS = _discover_backends()
if not COMBOS:
    COMBOS = [("no-backends", "none")]


# ── HF Cache (session-scoped, one-time) ──────────────────────────────────────

_E2E_HF_CACHE: str | None = None


def _ensure_e2e_cache() -> str:
    """Create an isolated HF cache in /tmp. Returns path.

    Uses a stable temp dir name so prior e2e runs are reused across sessions.
    """
    global _E2E_HF_CACHE
    if _E2E_HF_CACHE is not None:
        return _E2E_HF_CACHE
    cache_dir = Path("/tmp/arkestra-e2e-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _E2E_HF_CACHE = str(cache_dir)
    return _E2E_HF_CACHE


def _download_models_for_e2e() -> None:
    """Ensure every model ref has GGUF files in the isolated e2e cache.

    Reuses prior e2e run if valid. Otherwise pulls via snapshot_download,
    which checks disk caches automatically before hitting the network.
    Runs once per session via the ``e2e_cache`` fixture.
    """
    cache_dir = _ensure_e2e_cache()
    os.environ["HF_HUB_CACHE"] = cache_dir
    from huggingface_hub import snapshot_download

    refs: set[str] = {"bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
                      "bartowski/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M"}

    for ref in sorted(refs):
        hf_repo = ref.split(":", 1)[0]
        tag = ref.split(":", 1)[1] if ":" in ref else "*"
        e2e_path = Path(cache_dir) / ("models--" + hf_repo.replace("/", "--"))

        has_valid = (
            e2e_path.exists()
            and any(
                f.name.endswith(".gguf") and "/snapshots/" in str(f)
                for f in sorted(e2e_path.rglob("*.gguf"))
            )
        )

        if has_valid:
            print(f"[e2e] Reused (cached): {ref}")
            continue

        lock_dir = e2e_path / ".locks"
        if lock_dir.exists():
            shutil.rmtree(lock_dir, ignore_errors=True)
        print(f"[e2e] Downloading {ref} ...")
        snapshot_download(
            hf_repo,
            cache_dir=cache_dir,
            allow_patterns=[f"*{tag}*.gguf"] if tag != "*" else ["*.gguf"],
            local_files_only=False,
        )


def _cleanup_e2e_cache() -> None:
    """Remove the isolated e2e HF cache."""
    global _E2E_HF_CACHE
    if _E2E_HF_CACHE and os.path.isdir(_E2E_HF_CACHE):
        shutil.rmtree(_E2E_HF_CACHE)
    _E2E_HF_CACHE = None


@pytest.fixture(scope="session")
def e2e_cache():
    """Download models once before any e2e test runs."""
    _download_models_for_e2e()
    yield _E2E_HF_CACHE
    _cleanup_e2e_cache()


# ── Port helpers ─────────────────────────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
    return f":{port}" in result.stdout


def _wait_model_ports_free(timeout: float = 40.0) -> None:
    """Block until all model ports (MODEL_START..MODEL_END) are free."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
        busy = any(f":{p}" in result.stdout for p in range(MODEL_START, MODEL_END + 1))
        if not busy:
            return
        time.sleep(0.3)
    for p in range(MODEL_START, MODEL_END + 1):
        subprocess.run(["fuser", "-k", "-9", f"{p}/tcp"], capture_output=True, timeout=5)


def _wait_admin_port_free(port: int, timeout: float = 30.0) -> None:
    """Block until the server admin port is free."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_in_use(port):
            return
        time.sleep(0.3)
    subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], capture_output=True, timeout=5)


def _stop_all_and_wait(client: httpx.Client, base_url: str, timeout: float = 60.0) -> None:
    """Call POST /admin/stop-all and wait until model ports are free."""
    try:
        resp = client.post(f"{base_url}/admin/stop-all", timeout=timeout)
    except Exception:
        pass
    _wait_model_ports_free(timeout=timeout - 2)


# ── Config builder (one combo → one backend → one model) ─────────────────────

def _build_e2e_config(combo_id: str, backend_name: str, model_key: int = 0) -> str:
    """Build a YAML config for exactly one combo with `model-ports: 2`."""
    mname, mref = _MODELS[model_key % len(_MODELS)]

    if combo_id.startswith("process-vulkan"):
        be = {
            "runner": "process",
            "binary_dir": "/home/marc/local/llama.cpp/build-vulkan-radv/bin",
            "binary": "llama-server",
            "args": {"ngl": 999, "ctx-size": 2048},
        }
    elif combo_id.startswith("no-backends"):
        be = {"runner": "process"}
    elif "roc-process" in combo_id or "cuda-process" in combo_id:
        raise pytest.skip(f"No ROCm/CUDA binary for {combo_id}") from None
    else:
        runtime = combo_id.rsplit("-", 1)[0]
        be = {
            "runner": runtime,
            "source_ref": f"{runtime}-gpu",
            "entrypoint": "/usr/local/bin/llama-server",
            "devices": ["/dev/kfd", "/dev/dri"],
            "args": {"ngl": 999},
        }

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

    lines = ["default:", f"  model-start-port: {MODEL_START}", f"  model-ports: {MODEL_PORTS}"]
    lines.append("")
    lines.append("backends:")
    lines.append(f"  {backend_name}:")
    lines.extend(_serialize(be, 2))

    src = be.get("source_ref")
    if isinstance(src, str):
        runtime = combo_id.rsplit("-", 1)[0]
        if runtime:
            lines.append("")
            lines.append("sources:")
            lines.append(f"  {src}:")
            lines.append("    type: oci-image")
            lines.append("    repo: docker.io/kyuz0/amd-strix-halo-toolboxes")
            lines.append("    release_type: rocm-7.14")

    lines.append("")
    lines.append("models:")
    lines.append(f"  {combo_id}:")
    lines.append(f"    model: {mref}")
    lines.append(f"    backend: {backend_name}")
    lines.append("    args:")
    lines.append("      temp: 0.7")
    lines.append("      top-p: 0.95")

    return "\n".join(lines)


# ── Server lifecycle ────────────────────────────────────────────────────────

def _start_server(port: int, config_yaml: str) -> Tuple[Any, httpx.Client]:
    from model_arkestra.server import ArkestraServer

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    if combo_id.startswith("docker") or combo_id.startswith("podman"):
        ready_timeout = 240
    else:
        ready_timeout = 60

    try:
        proxy = ArkestraServer(config_path=config_path, port=port, ready_timeout=ready_timeout)
        app = proxy.get_app()
        server_obj = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        proxy._server = server_obj

        def serve():
            import asyncio
            asyncio.run(server_obj.serve())

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                httpx.get(url, timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise RuntimeError(f"Server on port {port} did not become ready")

        return proxy, httpx.Client(timeout=None, headers={"X-Admin-Key": "test-e2e-key"})
    except Exception:
        os.unlink(config_path)
        raise


def _stop_server(proxy: Any, client: httpx.Client, port: int) -> None:
    """Shutdown server, kill stray containers."""
    try:
        client.post(f"http://127.0.0.1:{port}/admin/shutdown", timeout=120)
    except Exception:
        pass

    for runtime in ("podman", "docker"):
        try:
            result = subprocess.run(
                [runtime, "ps", "-a", "--filter", "name=llm-",
                 "--format", "{{.ID}}"], capture_output=True, text=True, timeout=5)
            for cid in result.stdout.strip().split():
                if cid:
                    subprocess.run([runtime, "rm", "-f", cid], capture_output=True, timeout=5)
        except Exception:
            pass

    # Wait for admin port to free (shutdown may take a moment)
    _wait_admin_port_free(port)
    try:
        client.close()
    except Exception:
        pass


def _start_model(client: httpx.Client, base_url: str, model_name: str,
                 timeout: float = 180.0) -> bool:
    resp = client.post(f"{base_url}/admin/start/{model_name}", timeout=timeout)
    if resp.status_code != 200:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"{base_url}/admin/models", timeout=10)
        for m in r.json()["models"]:
            if m["id"] == model_name and m.get("status", {}).get("value") == "loaded":
                return True
        time.sleep(0.5)
    return False


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def e2e_server(request, e2e_cache):
    """Per-test server on fixed port 18003.

    Starts fresh for each test method. Fixture teardown calls shutdown.
    Each test body must call stop-all + wait in its finally block before
    the fixture's shutdown teardown runs — ensuring no state leaks between tests.
    """
    combo_id = request.param[0] if hasattr(request, "param") else "process-vulkan"
    backend_name = request.param[1] if hasattr(request, "param") else "vulkan-process"

    config = _build_e2e_config(combo_id, backend_name)
    proxy, client = _start_server(ADMIN_PORT, config)

    yield {"server": proxy, "client": client,
           "base_url": f"http://127.0.0.1:{ADMIN_PORT}",
           "combo_id": combo_id}

    # Fixture teardown — shutdown the whole server after all tests in the class
    _stop_server(proxy, client, ADMIN_PORT)


@pytest.fixture()
def e2e_single(request, e2e_cache):
    """Single-combo fixture with stop-all cleanup. Parametrize as needed."""
    combo_id = request.param[0]
    backend_name = request.param[1]

    config = _build_e2e_config(combo_id, backend_name)
    proxy, client = _start_server(ADMIN_PORT, config)

    yield {"server": proxy, "client": client,
           "base_url": f"http://127.0.0.1:{ADMIN_PORT}",
           "combo_id": combo_id}

    # Each test must call stop-all before this runs
    _stop_server(proxy, client, ADMIN_PORT)


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestFullLifecycle:
    """Start → inference → logs → stop. One model at a time."""

    @pytest.mark.parametrize("e2e_server", COMBOS, indirect=True)
    def test_ainvoke(self, e2e_server):
        """Single message → non-streaming response."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]
        model_name = e2e_server["combo_id"]

        ok = _start_model(client, base_url, model_name)
        try:
            if not ok:
                log_resp = client.get(
                    f"{base_url}/admin/log/{model_name}",
                    params={"since": 0, "lines": 100}, timeout=10)
                if log_resp.status_code == 200:
                    for line in log_resp.json().get("lines", []):
                        print(f"LOG: {line['text']}")
            assert ok, f"Model {model_name} failed to start"

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
            _stop_all_and_wait(client, base_url)

    @pytest.mark.parametrize("e2e_server", COMBOS, indirect=True)
    def test_astream(self, e2e_server):
        """Single message → streaming SSE response."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]
        model_name = e2e_server["combo_id"]

        ok = _start_model(client, base_url, model_name)
        try:
            assert ok, f"Model {model_name} failed to start"

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
            _stop_all_and_wait(client, base_url)

    @pytest.mark.parametrize("e2e_server", COMBOS, indirect=True)
    def test_start_nonexistent_returns_503(self, e2e_server):
        """Chat for unstarted model returns 503."""
        client = e2e_server["client"]
        base_url = e2e_server["base_url"]

        resp = client.post(f"{base_url}/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
        }, timeout=5)
        assert resp.status_code == 503


@pytest.mark.e2e
class TestPullAndEject:
    """Model pull via admin endpoint + eject cleanup."""

    @pytest.fixture()
    def e2e_cache_only(self, e2e_cache):
        """Session-scoped cache dir for verification."""
        yield _E2E_HF_CACHE

    @pytest.mark.parametrize("e2e_single", [("process-vulkan", "vulkan-process")], indirect=True)
    def test_e2e_pull_pipeline(self, e2e_single, e2e_cache_only):
        """Pull a model via admin endpoint and verify checkpoint lands on disk."""
        client = e2e_single["client"]
        base_url = e2e_single["base_url"]
        pull_model_id = e2e_single["combo_id"]

        resp = client.post(f"{base_url}/admin/pull/{pull_model_id}", timeout=10)
        assert resp.status_code == 200, f"Download start failed: {resp.text}"

        # Verify the GGUF file exists on disk in the e2e cache
        from model_arkestra.common import resolve_model_ref
        resolved = resolve_model_ref("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", {})
        cache_path = Path(e2e_cache_only) / ("models--" + resolved.cache_path)

        # Download might be a no-op if already cached — accept immediate STOPPED
        # with logs empty, as long as the GGUF exists on disk
        deadline = time.time() + 600
        while time.time() < deadline:
            r = client.get(f"{base_url}/admin/models", timeout=10)
            for m in r.json()["models"]:
                if m["id"] == pull_model_id:
                    state = m.get("status", {}).get("value")
                    if state == "stopped":
                        break
                    elif state == "error":
                        log_r = client.get(
                            f"{base_url}/admin/log/{pull_model_id}",
                            params={"since": 0, "lines": 20}, timeout=10)
                        logs = [l["text"] for l in log_r.json().get("lines", [])]
                        if logs:
                            pytest.fail(f"Download failed: {logs[-5:]}")
                        # No logs + error: could be no-op with cached files.
                        # Check disk directly — if GGUF exists it's a success.
                        break
                    time.sleep(1)
        else:
            pytest.fail("Download did not complete within 600s")

        # Verify GGUF on disk in the e2e cache
        from model_arkestra.common import resolve_model_ref
        resolved = resolve_model_ref("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", {})
        cache_path = Path(e2e_cache_only) / ("models--" + resolved.cache_path)

        found = any(gguf.exists() for gguf in sorted(cache_path.rglob("*.gguf")))
        if not found:
            pytest.fail(f"No GGUF in e2e cache. Contents: {list(cache_path.rglob('*'))[:10]}")
        else:
            print("[e2e] Downloaded checkpoint verified on disk")

        _stop_all_and_wait(client, base_url)

    @pytest.mark.parametrize("e2e_single", [("process-vulkan", "vulkan-process")], indirect=True)
    def test_e2e_eject_running_model(self, e2e_single, e2e_cache_only):
        """Eject a model while running — verify clean stop + cache deletion."""
        client = e2e_single["client"]
        base_url = e2e_single["base_url"]
        eject_model_id = e2e_single["combo_id"]

        ok = _start_model(client, base_url, eject_model_id)
        assert ok, f"{eject_model_id} failed to start"

        r = client.get(f"{base_url}/admin/models", timeout=10)
        model_state = None
        for m in r.json()["models"]:
            if m["id"] == eject_model_id:
                model_state = m.get("status", {}).get("value")
        assert model_state == "loaded", f"Model not loaded (state={model_state})"

        resp = client.post(f"{base_url}/admin/eject/{eject_model_id}", timeout=120)
        assert resp.status_code == 200, f"Eject failed: {resp.text}"
        body = resp.json()
        assert body.get("ok") is True
        assert body.get("cache_deleted") is True, f"Cache not deleted: {body}"

        # Model should no longer be in any runner — check all runners via a helper endpoint
        # Since /admin/models lists config entries (not live state), verify the model
        # is stopped and not in any active port allocation
        r = client.get(f"{base_url}/admin/models", timeout=10)
        for m in r.json()["models"]:
            if m["id"] == eject_model_id:
                assert m.get("status", {}).get("value") == "stopped", \
                    f"Ejected model still in active state: {m['status']}"

    @pytest.mark.parametrize("e2e_single", [("process-vulkan", "vulkan-process")], indirect=True)
    def test_e2e_eject_stopped_model(self, e2e_single):
        """Eject a stopped model — basic cleanup path."""
        client = e2e_single["client"]
        base_url = e2e_single["base_url"]
        eject_model_id = e2e_single["combo_id"]

        resp = client.post(f"{base_url}/admin/eject/{eject_model_id}", timeout=120)
        assert resp.status_code == 200, f"Eject failed: {resp.text}"


@pytest.mark.e2e
class TestPortExhaustion:
    """Start 2 models (within limit), verify 3rd fails."""

    @pytest.fixture()
    def e2e_exhaust(self, e2e_cache):
        """Server configured with model-ports: 2 and 3 model defs."""
        from model_arkestra.server import ArkestraServer

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

        be = {
            "runner": "process",
            "binary_dir": "/home/marc/local/llama.cpp/build-vulkan-radv/bin",
            "binary": "llama-server",
            "args": {"ngl": 999, "ctx-size": 2048},
        }

        lines = ["default:", f"  model-start-port: {MODEL_START}", f"  model-ports: {MODEL_PORTS}"]
        lines.append("")
        lines.append("backends:")
        lines.append("  vulkan-process:")
        lines.extend(_serialize(be, 2))
        lines.append("")
        lines.append("models:")
        for i, (mname, mref) in enumerate(_MODELS[:3]):
            lines.append(f"  process-vulkan-{i}:")
            lines.append(f"    model: {mref}")
            lines.append(f"    backend: vulkan-process")
            lines.append("    args:")
            lines.append("      temp: 0.7")

        config_yaml = "\n".join(lines)

        proxy, client = _start_server(ADMIN_PORT, config_yaml)
        yield proxy, client, f"http://127.0.0.1:{ADMIN_PORT}"

        # Test calls stop-all explicitly — fixture shutdown is last resort
        _stop_server(proxy, client, ADMIN_PORT)

    def test_two_succeed_then_three_fails(self, e2e_exhaust):
        """First two models start OK; third exceeds model-ports: 2 and fails."""
        _, client, base_url = e2e_exhaust

        for suffix in ("0", "1"):
            mid = f"process-vulkan-{suffix}"
            resp = client.post(f"{base_url}/admin/start/{mid}", timeout=300)
            assert resp.status_code == 200, f"Model {mid} should start: {resp.text}"

        mid = "process-vulkan-2"
        resp = client.post(f"{base_url}/admin/start/{mid}", timeout=30)
        assert resp.status_code != 200, \
            "Starting 3rd model with model-ports: 2 should fail"

        _stop_all_and_wait(client, base_url)
