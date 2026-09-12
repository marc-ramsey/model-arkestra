# Model Arkestra

Model Arkestra manages your home-lab GPUs and cloud model access by running LLM inference engines on demand. Define models in a YAML config file — each one maps to a backend (ROCm, Vulkan RADV, CUDA, CPU) and a runner type (process or container). When you start a model, Arkestra allocates an available port and launches the engine via the configured runner — usually [llama.cpp](https://github.com/ggml-org/llama.cpp), optionally within a container. Remote clusters let you administer multiple servers from one console.

A companion ONNX runner handles embeddings, Whisper transcription, and TTS in memory — no ports, no subprocesses. The admin dashboard at `http://localhost:<port>/` shows model status, lets you edit configs, chat via SSE streaming, and manage the lifecycle of everything from a single page. OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/*`) make it drop-in compatible with any client.

For more developed applications with similar function, see [llama-swap](https://github.com/mostlygeek/llama-swap) and [Lemonade](https://lemonade-server.ai/), as well as [llama-server router mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#using-multiple-models).

## Get Started

| If you want to… | Go here |
|---|---|
| Install Model Arkestra on your machine | See [Installation](#installation) below |
| Start using Model Arkestra in Python code | [Usage Guide](./docs/usage.md) |
| Run the OpenAI-compatible API server | [Server Documentation](./docs/server.md) |
| Manage models via web UI / Admin panel | [Admin API & Dashboard](./docs/admin.md) |
| Use the `arkestra` CLI (models, chat, init) | See below |
| Use the `arkestra-admin` CLI | See below |
| Offload embeddings/TTS/STT to ONNX | [ONNX Server](./docs/onnx-server.md) |
| Federate inference across multiple machines | [Remote Federation](#quick-start---remote-federation) |
| Understand how routing, ports, and runners work | [Architecture](./docs/architecture.md) |
| Write or modify `config.yaml` | [Configuration Format](./docs/config.md) |

## Installation

```bash
cd arkestra
scripts/post_install.sh
```

This creates the venv, installs the package (editable mode with `[proxy]` extras), and adds `venv/bin` to your shell's PATH in both `.bashrc` and `.profile`. Source your profile or restart the terminal afterwards.

After setup, CLI commands work from any directory — no activation needed:

```bash
arkestra-server --config config.yaml --url http://127.0.0.1:8080
arkestra models
```

### Quick Start — Python API

```python
from model_arkestra.arkestra import ModelArkestra

async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")                          # start a model
    result = await runner.ainvoke("qwen3-4b", "Explain quantum entanglement")
    print(result)                                           # → full response string

    async for chunk in runner.astream("qwen3-4b", {"prompt": "Write a haiku"}):
        if "token" in chunk:
            print(chunk["token"], end="", flush=True)      # streaming tokens
```

### Quick Start — Server

```bash
python -m model_arkestra.server --config config.yaml --url http://127.0.0.1:8080
# or equivalently:
arkestra-server --config config.yaml --url http://127.0.0.1:8080
```

Then hit `POST /v1/chat/completions` with any OpenAI-compatible client, or visit the admin dashboard at `http://localhost:8080/`.

### Quick Start — User CLI

```bash
arkestra init                        # scaffold config.yaml + backends.yaml
arkestra models [-m qwen3-4b]        # list cached models (name, size)
arkestra chat -m qwen3-4b            # interactive chat (auto-starts if stopped)
arkestra chat -m qwen3-4b -T 0.5 --max-tokens 256   # with inference flags
```

Connection resolution (same for `arkestra`, `arkestra-admin`, and `arkestra-server`):

| Priority | Source | Example |
|---|---|---|
| 1 | CLI flag | `--url http://10.0.0.1:9000/base` |
| 2 | Environment | `ARKESTRA_URL=http://10.0.0.1:9000/base` |
| 3 | Config | `default: url: http://10.0.0.1:9000/base` in `config.yaml` |
| 4 | Default | `http://127.0.0.1:8080` |

The public URL carries the host, port, and any path prefix (`/base`). The server's bind address is set separately with `--bind` / `default.bind` (default `127.0.0.1`; use `0.0.0.0` to expose on the LAN).

Config location: `--config` / `ARKESTRA_CONFIG` / `ARKESTRA_DIR` / `~/.config/arkestra/config.yaml`.

In-loop commands during chat: `/help`, `/quit`, `/clear`, `/history`, `/system <txt>`, `/temperature <n>`, `/top-p <n>`, `/max-tokens <n>`, `/frequency-penalty <n>`, `/presence-penalty <n>`, `/stop <txt>`.

### Quick Start — Admin CLI

```bash
arkestra-admin models --url http://localhost:8080 --api-key SECRET
arkestra-admin start qwen3-4b temp=0.7 backend=vulkan-radv
arkestra-admin config get qwen3-4b
arkestra-admin logs qwen3-4b --lines 100
arkestra-admin images list
arkestra-admin eject qwen3-4b         # stop and delete checkpoint cache
arkestra-admin shutdown --url http://localhost:8080
```

See [Admin API & Dashboard](./docs/admin.md) for the full CLI reference.

### Quick Start — Container Runners

Set `container_type: podman` (or `docker`) in `config.yaml`, then use `runner: container` in any backend to defer to the global default:

```yaml
# config.yaml
container_type: podman   # or "docker" — change all containers with one line

# backends.yaml  
rocm-container:
  runner: container      # resolves to "podman" above
```

This lets you swap between Podman and Docker globally without editing individual backends.

### Quick Start — Auxiliary Workloads (ONNX)

Offload Whisper (STT), Kokoro TTS, or embedding models alongside LLM inference. ONNX models load directly into memory — no subprocess spawning, no port allocation.

**Via Python API:**
```python
from model_arkestra.arkestra import ModelArkestra

async with ModelArkestra("config.yaml") as arkestra:
    # Start an ONNX embedding model (auto-loads into memory)
    await arkestra.start("bge-small")
    
    # Direct inference — non-blocking, runs in thread pool
    emb = await arkestra.embed("bge-small", "hello world")
    print(emb["data"][0]["embedding"][:5])  # first 5 dims

    # Whisper transcription (raw WAV bytes in)
    text = await arkestra.transcribe("whisper-tiny", wav_bytes, language="en")
    
    # TTS synthesis (returns WAV bytes)
    audio = await arkestra.synthesize("kokoro", "Hello from Arkestra!")
```

**Via HTTP server (`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech`)** — same endpoints as OpenAI API, backed by ONNX models configured in your `config.yaml` with `type: embedding | whisper | tts`.

**Config example:**
```yaml
# config.yaml
models:
  bge-small:
    backend: onnx
    model_path: ~/.cache/huggingface/models--Xenova/bge-small-en-v1.5/snapshots/onnx/model.onnx
    type: embedding
  whisper-tiny:
    backend: onnx
    model_path: /path/to/whisper-onnx/model.onnx
    type: whisper
    tokenizer: ~/.cache/huggingface/models--Xenova/whisper-tiny-en
```

**Standalone ONNX server** (optional — for running ONNX inference as a separate process):
```bash
python -m model_arkestra.onnx_server \
    --model /path/to/model.onnx \
    --type embedding \
    --port 8090 \
    --tokenizer Xenova/bge-small-en-v1.5
```

See [ONNX Server](./docs/onnx-server.md) for full documentation.

### Quick Start — Remote Federation

Run inference on a GPU worker from a CPU-only master host. No model downloads, no port allocations, no binary dependencies on the master:

```yaml
# config.yaml (on your laptop / CPU server)
models:
  gpu-lab-1/gemma-4b:       # worker-name / model-id convention
    repo: hugging-face
    model: unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
    backend: gpu-lab-1
backends:
  default: gpu-lab-1
  gpu-lab-1:
    runner: remote           # proxy everything to this worker
    base_url: "http://192.168.1.42:18000"   # worker's admin port
    admin_key: "my-secret"                       # optional auth
```

Once configured, every request to `/v1/chat/completions` with `model: "gpu-lab-1/gemma-4b"` is forwarded transparently to the worker. Streaming responses flow back as SSE in real time.

```bash
# The master needs no GPU, no llama.cpp binary, nothing local
arkestra-server --config config.yaml --url http://127.0.0.1:8080
```

See [Configuration Format](./docs/config.md#remote-federation-runner-remote) for the full federation guide.

### Hugging Face Cache Location

Models are downloaded via HuggingFace Hub. Control where they land by setting `HF_HUB_CACHE`:

- **Via config.yaml** in the `default-env:` section (resolved through `_env` at startup):
  ```yaml
  default-env:
    hf-hub-cache: /data/hf-cache
  ```
- **Or as an environment variable** in the host shell (`HF_HUB_CACHE`).

The default is `~/.cache/huggingface/hub`. The [`config.md`](./docs/config.md#default-env---runtime-environment-variables) has full details on the `default-env:` section.

## Capabilities

| Feature | Details |
|---|---|
| **Process Runner** | Native llama.cpp subprocess — direct binary execution, no containers. Supports `Vulkan`, `ROCm`, `CUDA`, `CPU` backends with automatic port allocation from a configurable range. |
| **Container Runners** | Podman or Docker isolation — pick the runtime globally (`container_type:`) and every backend inherits it. |
| **Remote Federation** | The `runner: remote` type proxies inference and lifecycle commands to another arkestra worker on a different machine. Model names use the `<worker-name>/<model-id>` convention (e.g., `gpu-server/qwen3`). Master servers never download, spawn, or allocate ports for remote models — all HTTP calls are forwarded transparently. |
| **ONNX Inference** | Native in-memory sessions for embeddings (`bge-*`), Whisper STT, and Kokoro TTS. No subprocesses, no ports — loads directly into Python via `onnxruntime`. Exposed on OpenAI-compatible `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech`. |
| **Open WebUI Ready** | Admin dashboard at `http://localhost:8080/` with live model management, SSE chat streaming, and structured status reporting (`loaded`, `stopped`, `unloaded`) for auto-load integration. |
| **Auto-Start on Inference** | Chat to a stopped or sleeping model starts it automatically. Chat to an uncached model (no checkpoint) fails cleanly with 503. Configure wait timeout per-model: `model-start-timeout: 600`. |
| **Eject Preserves Visibility** | `unload` removes the cache but keeps the model visible in the admin panel as `unloaded`, ready for re-pull or config edit. |
| **XDG Config Defaults** | Config files default to `~/.config/arkestra/config.yaml` — no CLI flag needed. Backends resolved from a companion `backends.yaml`. |
| **Restart Resilience** | Crash detection with configurable restart limits and backoff delays. Stopped models reuse their original port on restart. |
| **CLI Tooling** | `arkestra` for user-facing commands: models, chat, init. `arkestra-admin` for server ops: start, stop, config, logs, images, shutdown. API-key secured. |

## Configuration Notes

- **Flat keys**: All model parameters (e.g., `temp`, `ctx-size`) are flat keys at the model level — no nested `args:` dict support.
- **Status mapping**: Internal states map to Open WebUI vocabulary: `STOPPED → stopped`, `UNCACHED → unloaded`, `RUNNING → loaded`.
- **Admin routes only**: Lifecycle control (`start`, `stop`, `restart`) is available via `/admin/*` endpoints — inference-only clients trigger auto-start automatically.

## Architecture Overview

Model Arkestra routes models through a config-driven runner registry — each model selects a backend, which maps to a runner type (process, podman, docker, container, or remote). The `runner: container` value resolves against the top-level `container_type:` in `config.yaml`, enabling global engine swapping. A global port allocator distributes ports from a configured range. Port assignments are sticky: stopping and restarting a model reuses the same port.

For **remote** (federated) models, no local port or process is allocated — all HTTP calls (start, stop, chat completions, embeddings) are forwarded to the target worker via its admin API. The master server acts purely as a proxy; individual workers are administered independently by visiting `http://<worker-ip>:18000/admin`.

For details see [Architecture](./docs/architecture.md) and [Lifecycle](./docs/lifecycle.md).

## Import Path

```python
from llm_config_manager.config_manager import ConfigManager    # data layer
from model_arkestra.arkestra import ModelArkestra              # orchestration (recommended)
from model_arkestra.base import BaseRunner                # abstract base class
from model_arkestra.process import ProcessRunner          # process runner
from model_arkestra.podman import PodmanRunner            # podman runner
from model_arkestra.docker import DockerRunner            # docker runner
from model_arkestra.container_runner import ContainerRunner  # container base class
from model_arkestra.unicode_ringbuffer import UnicodeRingBuffer  # log buffer internals
from model_arkestra.langchain_adapter import LangChainModelAdapter  # LangChain LCEL wrapper
from model_arkestra.server import ArkestraServer             # OpenAI v1-compatible API server
from model_arkestra.onnx_server import OnnxServer            # ONNX inference (auxiliary workloads)
from model_arkestra.onnx_runner import OnnxRunner              # in-memory ONNX runner
from model_arkestra.remote import RemoteRunner             # proxy to remote worker

# Convenience re-exports from __init__.py:
from model_arkestra import RunnerState, RunnerError, ServerReadyTimeout
from model_arkestra import ModelNotStarted, MaxRestartsExceeded, ModelShutdown
```

## Further Reading

- [API Reference — ModelArkestra](./docs/api/arkestra.md)
- [API Reference — Runners](./docs/api/runners.md)
- [LangChain Integration](./docs/langchain.md)
- [Error Hierarchy](./docs/errors.md)
- [HTTP Client](./docs/http-client.md)
- [ONNX Server](./docs/onnx-server.md) — auxiliary workloads (embeddings, TTS, STT)
- [Contributing & Tests](./docs/contributing.md)

## Running tests

Always use the wrapper script — it guarantees cleanup of ports, buildah dirs, and llama-server processes even if pytest is killed mid-run:

```bash
./tests/run-tests.sh -v              # unit + integration (excludes slow)
./tests/run-tests.sh --all           # includes slow tests  
./tests/run-tests.sh -m "not slow"   # same as default
```
