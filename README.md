# Model Runner Documentation

The `model_arkestra` package manages the full lifecycle of local LLM inference servers. It has three layers:

- **ModelArkestra** — centralized entry point that manages port allocation, resolves backends via configuration, and delegates to backend-specific runners.
- **ContainerModelRunner** (intermediate base) — shared container logic for Podman and Docker (port drain wait, health watching, restart handling, force-remove on teardown).
- **Runners** — concrete implementations: `ProcessModelRunner` (direct subprocess), `PodmanModelRunner`, and `DockerModelRunner`.

Models are addressed by **model name** alone. Callers interact purely by model name — the runner automatically picks the correct backend from config. Each model runs on exactly one backend at a time.

---

## Architecture

```
┌───────────────────────────────────────────────────┐
│                   ModelArkestra                     │
│  ── port allocation, backend→runner routing        │
├───────────────────────────────────────────────────┤
│    _build_runner_class_map() → config-driven       │
│    runners: section maps type → class              │
│    _get_runner_instance(type, model) → lazy factory│
├──────────────────────┬────────────────────────────┤
│     ProcessModel     │  ContainerModelRunner      │
│         Runner       │   (abstract base)          │
│                      ├──────────────┬─────────────┤
│    subprocesses      │  PodmanModel │ DockerModel │
│                      │    Runner    │   Runner    │
│                      │              │             │
│                      │  containers  │ containers  │
└──────────────────────┴──────────────┴─────────────┘
```

### Runner routing — explicit args override config

Runner type resolution follows a strict precedence: **explicit arguments take priority over all configuration values**. An explicit runner type that does not exist in the registry is rejected immediately.

#### Explicit override (highest priority)

When `runner=` is supplied, that value is used directly:

```
runner="podman" → PodmanModelRunner   (verified against registry)
runner="process" → ProcessModelRunner  (verified against registry)
```

#### Automatic resolution (config chain)

When no explicit argument is given, the runner type is resolved through config:

```
backends section exists?
  └─> model.backend → backends.<id>.runner → runners:<type> (or "process")
backends section absent?
  └─> runners:<default> (or "process")
```

### Port allocation — a global counter controlled by `ModelArkestra`. The starting port and range are driven entirely by config:

| Config key | Purpose |
|---|---|
| `models-start-port` | First port in the range (default 18000) |
| `model-ports` | Size of the pool — valid ports are `start_port` through `start_port + model-ports - 1` |

When `ModelArkestra.start()` is called without an explicit `port`, it allocates the next available number from this range sequentially. Once all ports in the pool are exhausted, `RuntimeError("Port range exceeded: …")` is raised immediately.

A **direct** runner instance (e.g., `ProcessModelRunner(cm)`) does **not** use a global counter — it picks `models-start-port` as the default for its first model, and subsequent calls reuse existing ports via `_dispatch()` / in-place restart.

Stopped models **retain their port assignments** in both the orchestrator and direct runner instances. Calling `start()` again on a stopped model restarts it on the same port (in-place). New models are assigned the next unused port from the pool when using `ModelArkestra`.

```
Config: start_port=18000, model-ports=32  →  valid range 18000–18031

Model started      Port assigned
─────────────────  ───────────
"alpha"            18000    ← auto-allocated from pool
"beta"             18001    ← next in pool
"gamma" (port=9000) 9000   ← explicit override, bypasses pool
"delta"            18002    ← next in pool
"alpha" stopped → start("alpha")  18000  ← restart-in-place, same port
"epsilon"          18003    ← next free port in pool
```

Explicit ports bypass the pool entirely — no range validation is performed. The caller bears that responsibility.

---

## Usage

### Basic Initialization

```python
# Orchestration layer (recommended)
from model_arkestra.arkestra import ModelArkestra

runner = ModelArkestra("config.yaml", ready_timeout=120.0)

async with runner:
    # ... use runner.start(), stop(), ainvoke(), etc ...
# → shutdown() called automatically on exit

# Direct runners for standalone use (they do support context managers)
from model_arkestra.process import ProcessModelRunner

cm = ConfigManager("config.yaml")  # optional: pass ConfigManager directly
runner = ProcessModelRunner(cm, shutdown_timeout=20.0, ready_timeout=120.0)
```

### Starting a Model

```python
# Auto-assigns port and picks runner from config chain
await runner.start("gpt-oss-20b")

# Model with backend: rocm → runner picks podman from config chain (no runner= needed)
await runner.start("qwen3-4b:rocm")

# Explicit backend override (also routes correctly via config chain)
await runner.start("qwen3-4b", backend="rocm")

# Explicit port assignment (caller responsibility to avoid conflicts)
await runner.start("qwen3.6-35B-think", port=18005)

# Explicit runner= override forces a specific runner type directly
await runner.start("qwen3-4b", runner="podman")

# Multiple models can run concurrently, each on its own port
print(runner.running_models)
# → {"gpt-oss-20b", "qwen3.6-35B-think", "gemma-4-26b-instruct"}
```

### Restarting a Model

Stops the running instance and starts a fresh one on the **same port**. Optional keyword args override backend or runner for the new instance.

Calling `start()` again on a stopped model restarts it in-place on the same port (no need to call `stop()` first). Calling `start()` on an already-running model is idempotent — it checks `/health` and returns immediately if the server responds with 200.

```python
# Identical restart — same config, same port
await runner.restart("gpt-oss-20b")

# Switch backend (runner resolves to podman from config)
await runner.restart("qwen3-4b", backend="rocm")

# Force a specific container runtime regardless of config
await runner.restart("qwen3-4b", runner="docker")

# Restart a stopped model via start() (same-port, same-backend)
await runner.stop("gpt-oss-20b")
await runner.start("gpt-oss-20b")  # reuses the original port

# Calling start() on an already-running model is idempotent
await runner.start("qwen3-4b")  # checks /health, returns if alive
```

### Sending a Prompt (Blocking)

```python
result = await runner.ainvoke("gpt-oss-20b", "Explain quantum entanglement")
print(result)
# → "Quantum entanglement is a phenomenon in which..."
```

### Streaming Response (Token-by-Token)

```python
async for event in runner.astream("gpt-oss-20b", {"prompt": "Write a haiku"}):
    if "token" in event:
        print(event["token"], end="", flush=True)  # Hell -> o -> Wo -> r -> l -> d
    elif "usage" in event:
        print()
        print(f"Done ({event['usage']['total_tokens']} tokens, {event['usage']['time_seconds']:.1f}s)")

# Output:
# Hello World
# Done (3 tokens, 0.2s)
```

### Graceful Shutdown

```python
# Stop a single model
await runner.stop("gemma-4-26b-instruct")

# Stop all running models — processes/containers are killed but model entries remain
# in STOPPED state so start() can restart them in-place on the same port.
await runner.stop_all()

# Full teardown — stops all models, clears internal state, resets port allocator
# For container runners, also force-removes all stopped containers.
await runner.shutdown()

# Or use async context manager (auto-calls shutdown() on exit)
async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")
    # ... do work ...
# → shutdown() called here, port counter resets to start value
```

---

## API Reference — `ModelArkestra`

The centralized entry point. Does not launch any processes at this point — models must be started explicitly via `start()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str` | *(required)* | Path to the YAML config file. |
| `start_port` | `int` | `18000` | Fallback starting port — only used when `models-start-port` is absent from config. Port allocation reads the actual values from config at init time. |
| `**runner_kwargs` | — | — | Passed through to each runner instance (e.g. `ready_timeout`, `warmup_delay`). |

### Backward-compat shims

The properties `.process_runner`, `.podman_runner`, and `.docker_runner` still exist for backward compatibility. Each delegates to the unified lazy factory (`_get_runner_instance`) so they behave identically to the config-driven path.

### Methods

#### `async start(model_name: str, port: int | None = None, backend: str | None = None, runner: str | None = None) -> None`

Starts the server process or container for *model_name*. Runner type is resolved automatically:
- If `runner=` is provided, that runner type is used directly.
- Otherwise: `backends.<id>.runner` (from config) → `runners:<type>` in config → falls back to `"process"`.

If the model already exists and is in a STOPPED state, it restarts in-place on the same port. If it is already RUNNING, `/health` is polled — 200 returns immediately, anything else continues startup. Polls `/health` until ready or timeout. Raises `ServerReadyTimeout` on health-check failure (not on intermediate 502/504; those raise `RunnerError`).

#### `async stop(model_name: str) -> None`

Stops the model matching *model_name*. Sends the appropriate shutdown signal (see [Lifecycle](#process-lifecycle)).

#### `async restart(model_name: str, *, backend: str | None = None, runner: str | None = None) -> None`

Stops the running instance and starts a new one on the **same port**.  Optional keyword args override backend or runner for the restarted instance.

#### `async stop_all() -> None`

Stops all model processes. Model entries remain in `_models` with state `STOPPED` — calling `start()` again on a stopped model restarts it in-place on the same port. For container runners, containers are **not** removed (only force-removed during `shutdown()`).

#### `async shutdown() -> None`

Full teardown: stops all models, clears all runner and model entries, and resets the port allocator to `models-start-port`. For container runners, also sends `rm -f` on every stopped container. Use this for final cleanup (e.g. context manager exit, test teardown).

#### `async ainvoke(model_name: str, prompt: str, backend: str | None = None) -> str`

Sends a blocking completion request and returns the full response string. Retries up to 12 times on connection errors or 503 status (with 2.5s backoff). Raises `RunnerError` for 502/504 (upstream failures), connection unreachable, or max retries exceeded.

#### `async astream(model_name: str, payload: Dict[str, Any], backend: str | None = None) -> AsyncIterator[Dict[str, Any]]`

Sends a streaming completion request and yields events for each SSE update. Returns an async generator — callers iterate with `async for`. The payload dict must contain `"prompt"` (string) or `"messages"` (list of message dicts).

Each yielded dict is one of two forms:

**Token event** (one per SSE message during generation):
```json
{ "token": "Hell" }
```

**Usage summary** (yielded once at the end, replacing `[DONE]`):
```json
{
  "usage": {
    "model": "...",
    "prompt_tokens": N,
    "completion_tokens": N,
    "total_tokens": N,
    "time_seconds": T,
    "tokens_per_second": R
  }
}
```

Token chunks may be partial — e.g., the first event might be `"token": "Hell"`, the next `"token": "o World"` rather than full words. Callers should concatenate tokens and/or check for a trailing space if word boundaries matter.

#### `async request(model_name: str, path: str, **kwargs) -> dict | bytes`

Low-level HTTP forwarder for custom server endpoints not covered by `ainvoke()` or `astream()`. Forwards *kwargs* as the request payload and returns the parsed JSON response (or raw bytes if the endpoint does not return JSON). Only sends POST requests with 15s timeout. Raises `RunnerError` on responses with status ≥ 400.

#### `running_models: set[str]`

Read-only property returning the names of all models currently in `"running"` state, aggregated across all runner instances.

---

## Model Introspection — runtime state

These three methods expose live model state tracked at runtime (port, backend, runner type, health). They operate on models that have been started — before any `start()` call the lists are empty.

#### `_get_model_contexts() -> list[_ModelContext]`

Internal aggregation: returns every tracked `_ModelContext` across all runners. Each context carries `name`, `port`, `state` (`RunnerState` enum), `backend_id`, `runner_type`, `restart_count`, and `last_error`. Callers who need detailed runtime info should use this.

#### `get_model_list() -> list[str]`

Convenience wrapper around `_get_model_contexts()` — returns just the model names as strings. Useful when you want a dynamic list of currently-tracked models (including stopped or errored ones) rather than static YAML config names.

```python
runner = ModelArkestra("config.yaml")
await runner.start("qwen3-4b")
print(runner.get_model_list())  # → ["qwen3-4b"]
```

#### `get_v1_models() -> dict`

OpenAI-compatible `/v1/models` response. Returns a dict with `"object": "list"` and a `"data"` array — one entry per tracked model containing:

| Field | Source |
|---|---|
| `id` | model name |
| `object` | `"model"` |
| `created` | Unix timestamp |
| `owned_by` | from config (`model_cfg.get("owned_by")`, default `"local"`) |
| `status` | lowercase state string (e.g. `"running"`, `"stopped"`, `"error"`) |
| `port` | allocated port number |
| `runner_type` | runner type string (`"process"`, `"podman"`, `"docker"`) |
| `backend_id` | resolved backend id (e.g. `"rocm"`) |

```python
models = runner.get_v1_models()
print(models)  # → {"object": "list", "data": [{"id": "qwen3-4b", "status": "running", ...}]}
```

---

## API Reference — `ProcessModelRunner`

For direct use when you only need subprocess-based execution. Bound to a `ConfigManager` instance; does not launch any processes until `start()` is called.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_manager` | `ConfigManager` | *(required)* | The configuration source for model definitions and global settings. |
| `shutdown_timeout` | `float` | `20.0` | Seconds to wait after SIGHUP before escalating to SIGKILL during shutdown. |
| `ready_timeout` | `float` | `120.0` | Maximum seconds to wait for the server's `/health` endpoint to respond after starting. |
| `ready_poll_ms` | `float` | `100.0` | Milliseconds between consecutive health-check polls during startup. |
| `warmup_delay` | `float` | `20.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`. |

> `restart_delay` and `restart_limit` control automatic restart behavior inherited from `BaseModelRunner`: when a managed process or container exits unexpectedly, the runner polls and attempts up to `restart_limit` restarts spaced by `restart_delay` seconds. `port_drain_timeout` is used by container runners to wait for port listeners to drain; it has no effect on direct subprocess execution.

Each runner manages exactly one model. Its `stop()` takes no arguments and shuts down that model; `stop_all()` delegates to `stop()`. The remaining methods (`start`, `ainvoke`, `astream`, `request`, `running_models`) follow the same contract as described in the `ModelArkestra` section above.

---

## Process Lifecycle

```
                ┌───────────┐
  start(model) ─│           │──► /health OK + warmup delay ─► state = "running"
                ▼           │
              "loading"     │
                              │ process/container exits unexpectedly
                              │ retry_count < restart_limit?
                              ├─ Yes → sleep(restart_delay) → restart
                              │         (reuses same port)
                              └─ No  → state = "error"
```

### Crash detection by backend

| Runner | Behavior |
|---|---|
| **Process** | Polls subprocess exit code via `process.wait()`. On unexpected exit (non-zero), automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. The watcher task runs in the background. |
| **Podman / Docker** | Polls container status every 2 seconds; on unexpected exit (`exited`, `dead`), automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. The watcher task runs in the background. |

### Shutdown sequencing (`stop` / `stop_all`)

1. State transitions to **STOPPING** immediately — prevents any watcher from attempting a restart.
2. Signal is sent:
   - **Process**: SIGHUP to process group → waits up to `shutdown_timeout` (default 20s) → escalates to SIGKILL if still running.
   - **Podman / Docker**: Container stop signal (`podman stop --time <port_drain_timeout>` or `docker stop --time <port_drain_timeout>`) with graceful shutdown timeout (`port_drain_timeout` default 20s).
3. After termination, watcher tasks are cancelled and state transitions to **STOPPED**. Port is released (with `port_drain_timeout` wait for container runners to let the port listener drain).

### Full teardown (`shutdown`)

Identical to stop sequencing with two additions:
1. All watcher tasks are cancelled and awaited.
2. `_models` and `_watchers` dictionaries are cleared — models cannot be restarted after shutdown.
3. **For container runners**: After stopping all containers, force-removes them via `podman rm -f` / `docker rm -f`.

---

## Error Hierarchy

All exceptions inherit from `RunnerError`, enabling callers to catch all runner-related failures with a single clause:

```python
try:
    result = await runner.ainvoke("my-model", prompt="...")
except RunnerError as e:
    logger.error(f"Runner failed: {e}")
```

| Exception | When raised |
|---|---|
| `ServerReadyTimeout` | The server did not become ready (health check returned no 200 within `ready_timeout`). Raised only on total health-check exhaustion. |
| `ModelNotStarted` | A request was made for a model name that doesn't exist in the config, or the model hasn't been started yet (`start()` not called). Raised by `_dispatch` when no tracked context exists, and during `BaseModelRunner.start()` when model config lookup fails. |
| `ModelShutdown` | A request was made to a model that has been explicitly stopped via `stop()` or `stop_all()` (state is STOPPED or STOPPING). Raised by `_dispatch`. |
| `MaxRestartsExceeded` | The runner crashed and exhausted all automatic restart attempts (`restart_limit`). State transitions to ERROR; raised by `_dispatch` on any subsequent request. |
| `RunnerError` | Generic base class for all runner failures, including HTTP errors (502, 503, 504) from the server, connection failures, and general request errors. |

---

## Configuration Format (YAML)

The runner reads from the same YAML configuration used by `ConfigManager`. Three sections are relevant:

### Top-level settings

| Key | Type | Default | Description |
|---|---|---|---|
| `models-start-port` | `int` | `18000` | First port in the auto-allocated range. |
| `model-ports` | `int` | `32` | Number of ports available — valid range is `models-start-port` through `models-start-port + model-ports - 1`. Allocation raises `RuntimeError("Port range exceeded: …")` when exhausted. |

### `env:` section — process environment variables

Environment variables defined here are merged into the subprocess/container environment at startup, in addition to the host's current environment. This is how paths like `LLAMA_CACHE` and `HF_HUB_CACHE` propagate to the server process.

```yaml
env:
  LLAMA_CACHE: /home/lemonade/hub
  HF_HUB_CACHE: /home/lemonade/hub
```

### `macros:` section — argument templates

Macros are expanded at boot time. At `start()` runtime the backend `args` template is resolved again with concrete `${PORT}` and `${CHECKPOINT}` values.

```yaml
macros:
  llama-args: --port ${PORT} --jinja -fa on -ngl 999
  ctx-default: 131072
```

### `backends:` section — executable registry

Each backend entry specifies an argument template and which runner type should handle it.

```yaml
backends:
  vulkan-radv:
    args: ${llama-args}
    runner: process     ← maps this backend to ProcessModelRunner
  rocm:
    args: ${llama-args}
    runner: podman      ← maps this backend to PodmanModelRunner
  default: vulkan-radv         ← global default backend (also has runner: process)
```

When a model has `backend: rocm`, the routing chain resolves: `rocm` → `runner: podman` → `runners.podman` → `PodmanModelRunner`.

**Backend configuration keys:**

| Key | Type | Description |
|---|---|---|
| `args` | str | Argument template for the llama-server. May reference macros via `${name}`. |
| `runner` | str | Runner type string (e.g. `"process"`, `"podman"`, `"docker"`). Resolved against `runners:` config or built-in registry. |
| `binary_dir` | str | Absolute path to host directory containing the llama-server binary. Direct-process runners use it directly; container runners resolve it indirectly via `resolve_binary_from_backend()` in `common.py` (which also checks `version` and image name heuristics). |
| `binary` | str | Binary name (default: `"llama-server"`). Used with `binary_dir` to form the full path. |
| `image` | str | Container image tag for Podman/Docker runners. |
| `devices` | list[str] | Device passthrough entries for container runs (e.g. `"/dev/dri/card1:rwm"`). |
| `env_container` | dict | Environment variables passed into the container. Merged on top of global `env:`. |
| `version` | str | ROCm/Vulkan version string — used to resolve binary dir from known build directory map (`_ROCM_BUILD_MAP`). |

### `runners:` section — runner class registry

This section maps runner type strings to concrete class names. Built-in types (`process`, `podman`, `docker`) are auto-registered; config entries can override them or add new ones (as long as the class is importable from this module).

```yaml
runners:
  podman: PodmanModelRunner
  docker: DockerModelRunner
  default: ProcessModelRunner
```

The `default` key specifies the fallback runner type when neither `backends.<id>.runner` nor a model's own `backend` field provides one.

### `models:` section — model definitions

Each model entry uses `checkpoint` for the model path, `args` for flags, and optionally `backend` to override which backend registry entry is used. **There is no `cmd` field** — commands are assembled at start time via the backend registry.

> HuggingFace models: use `-hf ${CHECKPOINT}` in the backend `args` template (see `sample-config.yaml`) so llama.cpp downloads the model instead of treating it as a local file path.

```yaml
models:
  "gpt-oss-20b":
    checkpoint: unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL
    args: |
      --temp 1.0 --top-k 0 --ctx-size ${ctx-default}

  "qwen3-4b":
    checkpoint: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: rocm          ← optional per-model override
    args: |
      --temp 0.7 --top-k 20 --ctx-size ${ctx-default}
```

---

## HTTP Client (`http_client.py`)

The package ships a lightweight `ModelHttpClient` class in `model_arkestra.http_client` that encapsulates aiohttp usage patterns:

| Method | Description |
|---|---|
| `get_json(url)` | GET and return parsed JSON body. |
| `post_json(url, json_body)` | POST with JSON body and return parsed response. |
| `post_raw(url, json_body)` | POST and return an async context manager for raw response streaming (SSE, large binaries). |
| `stream_sse(url, json_body)` | Iterate SSE `data:` lines from a POST endpoint — yields raw strings without the `data:` prefix. |

Sessions are scoped to each call; no manual session management needed.

```python
from model_arkestra.http_client import ModelHttpClient

async with ModelHttpClient(timeout=60) as client:
    data = await client.get_json("http://127.0.0.1:8080/health")
    async for line in client.stream_sse(url, {"prompt": "hi"}):
        print(line)
```

> **Note:** `ModelHttpClient` is a standalone utility — the runner classes use aiohttp directly internally and do not depend on this wrapper. It exists primarily for testing and external integrations.

---

## Import Path

The runner and its backends live in the `model_arkestra` package:

```python
from llm_config_manager.config_manager import ConfigManager    # data layer
from model_arkestra.arkestra import ModelArkestra              # orchestration (recommended)
from model_arkestra.base import BaseModelRunner                # abstract base class
from model_arkestra.process import ProcessModelRunner          # process runner
from model_arkestra.podman import PodmanModelRunner            # podman runner
from model_arkestra.docker import DockerModelRunner            # docker runner
from model_arkestra.container_runner import ContainerModelRunner  # container base class
from model_arkestra.http_client import ModelHttpClient         # lightweight HTTP client

# Convenience re-exports from __init__.py:
from model_arkestra import RunnerState, RunnerError, ServerReadyTimeout
from model_arkestra import ModelNotStarted, MaxRestartsExceeded, ModelShutdown
```
