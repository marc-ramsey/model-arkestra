# Model Runner Documentation

The `model_arkestra` package manages the full lifecycle of local LLM inference servers. It has two layers:

- **ModelArkestra** — centralized entry point that manages port allocation, resolves backends via configuration, and delegates to backend-specific runners.
- **Backend Runners** — concrete implementations that handle subprocess or container lifecycle per model.

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
│    _get_runner_instance(type) → lazy single factory│
├──────────────┬──────────────────┬─────────────────┤
│ ProcessModel │ PodmanModel      │ DockerModel     │
│ Runner       │ Runner           │ Runner          │
│              │                  │                 │
│ subprocesses │ containers (pod) │ containers(dkr) │
└──────────────┴──────────────────┴─────────────────┘
```

### Runner routing — explicit args override config

Runner type resolution follows a strict precedence: **explicit arguments take priority over all configuration values**. An explicit runner type that does not exist in the registry is rejected immediately.

#### Explicit override (highest priority)

When `container=` or `backend=` is supplied, that value is used directly:

```
container="podman" → PodmanModelRunner   (verified against registry)
backend="rocm"     → backends.rocm.runner → runner class
```

#### Automatic resolution (config chain)

When no explicit argument is given, the runner type is resolved through config:

```
model.backend (e.g. "rocm")
  → backends.rocm.runner (e.g. "podman")
  → runners.podman / built-in fallback ("podman")
  → runners.default / built-in fallback ("process")
```

### Port allocation — a global counter controlled by `ModelArkestra`. The starting port and range are driven entirely by config:

| Config key | Purpose |
|---|---|
| `models-start-port` | First port in the range (default 18000) |
| `model-ports` | Size of the pool — valid ports are `start_port` through `start_port + model-ports - 1` |

When `ModelArkestra.start()` is called without an explicit `port`, it allocates the next available number from this range. Once all ports in the pool are exhausted, `RuntimeError("Port range exceeded: …")` is raised immediately.

Stopped models **retain their port assignments**. Calling `start()` again on a stopped model restarts it on the same port (in-place). New models are assigned the next unused port from the pool.

```
Config: start_port=18000, model-ports=32  →  valid range 18000–18031

Model started      Port assigned
─────────────────  ───────────
"alpha"            18000
"beta"             18001
"gamma" (port=9000) 9000   ← explicit override, bypasses pool
"delta"            18002
"alpha" stopped → start("alpha")   18000    ← restart-in-place, same port
"epsilon"          18003    ← next free port in pool
"alpha" again (restart) 18000  ← explicit restart also uses same port
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
    ...

# Backend-specific runners are available for direct use
from model_arkestra.process import ProcessModelRunner

cm = ConfigManager("config.yaml")  # optional: pass ConfigManager directly
runner = ProcessModelRunner(cm, shutdown_timeout=20.0, ready_timeout=120.0)
```

### Starting a Model

```python
# Auto-assigns port and picks runner from config chain
await runner.start("gpt-oss-20b")

# Model with backend: rocm → automatic podman routing (no container= needed)
await runner.start("qwen3-4b:rocm")

# Explicit backend override (also routes correctly via config chain)
await runner.start("qwen3-4b", backend="rocm")

# Explicit port assignment (caller responsibility to avoid conflicts)
await runner.start("qwen3.6-35B-think", port=18005)

# Explicit container= override forces a specific runner type directly
await runner.start("qwen3-4b", container="podman")

# Multiple models can run concurrently, each on its own port
print(runner.running_models)
# → {"gpt-oss-20b", "qwen3.6-35B-think", "gemma-4-26b-instruct"}
```

### Restarting a Model

Stops the running instance and starts a fresh one on the **same port**. Optional keyword args override backend or container for the new instance.

A stopped model can also be restarted in-place by calling `start()` again — it reuses the same port automatically without needing an explicit `restart()` call.

```python
# Identical restart — same config, same port
await runner.restart("gpt-oss-20b")

# Switch backend (and thus runner) for the restarted instance
await runner.restart("qwen3-4b", backend="rocm")   # podman → process

# Force a specific container runtime regardless of config
await runner.restart("qwen3-4b", container="docker")

# Restart a stopped model via start() (same-port, same-backend)
await runner.stop("gpt-oss-20b")
await runner.start("gpt-oss-20b")  # reuses the original port

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

# Stop all running models — processes are killed but model entries remain
# in STOPPED state so start() can restart them in-place on the same port.
await runner.stop_all()

# Full teardown — stops all models, clears internal state, resets port allocator
await runner.shutdown()

# Or use the context manager (auto-calls shutdown() on exit)
async with ModelArkestra("config.yaml"):
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
| `**runner_kwargs` | — | — | Passed through to each backend runner (e.g. `ready_timeout`, `warmup_delay`). |

### Backward-compat shims

The properties `.process_runner`, `.podman_runner`, and `.docker_runner` still exist for backward compatibility. Each delegates to the unified lazy factory (`_get_runner_instance`) so they behave identically to the config-driven path.

### Methods

#### `async start(model_name: str, port: int | None = None, backend: str | None = None, container: str | None = None) -> None`

Starts the server process or container for *model_name*. Runner type is resolved automatically:
- If `container=` is provided, that runner type is used directly.
- Otherwise: `model.backend` → `backends.<id>.runner` → `runners.<type>` (falls back to `runners.default`, then `"process"`).

Polls `/health` until ready or timeout. Raises `ServerReadyTimeout` on failure.

#### `async stop(model_name: str) -> None`

Stops the model matching *model_name*. Sends the appropriate shutdown signal (see [Lifecycle](#process-lifecycle)).

#### `async restart(model_name: str, *, backend: str | None = None, container: str | None = None) -> None`

Stops the running instance and starts a new one on the **same port**.  Optional keyword args override backend or container for the restarted instance.

#### `async stop_all() -> None`

Stops all model processes. Model entries remain in `_models` with state ``STOPPED`` — calling `start()` again on a stopped model restarts it in-place on the same port.

#### `async shutdown() -> None`

Full teardown: stops all models, clears all runner and model entries, and resets the port allocator to ``models-start-port``. Use this for final cleanup (e.g. context manager exit, test teardown).

#### `async ainvoke(model_name: str, prompt: str) -> str`

Sends a blocking completion request and returns the full response string.

#### `async astream(model_name: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]`

Sends a streaming completion request and yields events for each SSE update. Returns an async generator — callers iterate with `async for`. Requires `{"prompt": "..."}` as the payload dict.

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

Low-level HTTP forwarder for custom server endpoints not covered by `ainvoke()` or `astream()`. Forwards *kwargs* as the request payload and returns the parsed JSON response (or raw bytes if the endpoint does not return JSON).

#### `running_models: set[str]`

Read-only property returning the names of all models currently in `"running"` state.

---

## API Reference — `ProcessModelRunner`

For direct use when you only need subprocess-based execution. Bound to a `ConfigManager` instance; does not launch any processes until `start()` is called.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_manager` | `ConfigManager` | *(required)* | The configuration source for model definitions and global settings. |
| `shutdown_timeout` | `float` | `20.0` | Seconds to wait after SIGHUP before escalating to SIGKILL during shutdown. |
| `ready_timeout` | `float` | `120.0` | Maximum seconds to wait for the server's `/health` endpoint to respond after starting. |
| `ready_poll_ms` | `float` | `100.0` | Milliseconds between consecutive health-check polls during startup. |
| `restart_delay` | `float` | `5.0` | Seconds to wait before attempting an automatic restart after a crash (podman/docker only). |
| `restart_limit` | `int` | `4` | Maximum number of automatic restart attempts (podman/docker only). |
| `warmup_delay` | `float` | `20.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`. |
| `port_drain_timeout` | `float` | `20.0` | Seconds to wait for a stopped container's port listener to release (podman/docker only). |

Each backend runner manages exactly one model. Its `stop()` takes no arguments and shuts down that model; `stop_all()` delegates to `stop()`. The remaining methods (`start`, `ainvoke`, `astream`, `request`, `running_models`) follow the same contract as described in the `ModelArkestra` section above.

---

## Process Lifecycle

```
                ┌───────────┐
  start(model) ─│           │──► /health OK + warmup delay ─► state = "running"
                ▼           │
              "loading"     │
                              │ process exits unexpectedly
                              ▼
                        watcher logs exit code, attempts restart (podman/docker)
```

### Crash detection by backend

| Runner | Behavior |
|---|---|
| **Process** | Logs exit code when the subprocess dies. No automatic restart — call `start()` again explicitly. |
| **Podman / Docker** | Polls container status; on unexpected exit, automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. |

### Shutdown sequencing (`stop` / `stop_all`)

1. State transitions to **STOPPING** immediately — prevents any watcher from attempting a restart.
2. Signal is sent:
   - **Process**: SIGHUP to process group → waits up to `shutdown_timeout` (default 20s) → escalates to SIGKILL if still running.
   - **Podman / Docker**: Container stop signal (`podman/dockers stop --time <port_drain_timeout>`) with graceful shutdown timeout.
3. After termination, watcher tasks are cancelled and state transitions to **STOPPED**. Port is released (with `port_drain_timeout` wait for container runners).

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
| `ServerReadyTimeout` | The server did not become ready (health check) within `ready_timeout`. |
| `ModelNotStarted` | A request was made for a model name that doesn't exist in the config, or the model hasn't been started yet (`start()` not called). |
| `ModelShutdown` | A request was made to a model that has been explicitly stopped via `stop()` or `stop_all()`. |
| `MaxRestartsExceeded` | Podman/Docker: process has crashed too many times (after `restart_limit` attempts). |
| `RunnerError` | Generic base class for all runner failures, including HTTP errors (502, 503, 504) from the server. |

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

Each backend entry specifies an argument template and which runner type should handle it. The actual executable path (`wrapper`) is resolved automatically from the `backend-registry` directory — no need to specify it in YAML.

```yaml
backend-registry: /path/to/backends   ← files here named 'radv', 'rocm' become wrapper paths
backends:
  radv:
    args: ${llama-args}
    runner: process     ← maps this backend to ProcessModelRunner
  rocm:
    args: ${llama-args}
    runner: podman      ← maps this backend to PodmanModelRunner
  default: radv         ← global default backend (also has runner: process)
```

When a model has `backend: rocm`, the routing chain resolves: `rocm` → `runner: podman` → `runners.podman` → `PodmanModelRunner`.

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

## Import Path

The runner and its backends live in the `model_arkestra` package:

```python
from llm_config_manager.config_manager import ConfigManager    # data layer
from model_arkestra.arkestra import ModelArkestra          # orchestration (recommended)
from model_arkestra.process import ProcessModelRunner        # process backend
from model_arkestra.podman import PodmanModelRunner          # podman backend
from model_arkestra.docker import DockerModelRunner          # docker backend
```

No changes are made to `config_manager.__init__.py` exports — each runner must be imported explicitly. This avoids circular imports and signals that process management is an optional, higher-level concern.
