# Usage Guide

This guide walks through common Model Arkestra operations: initializing the orchestrator or a direct runner, starting and stopping models, sending prompts, streaming responses, and graceful shutdown.

## Basic Initialization

### Orchestration layer (recommended)

```python
from model_arkestra.arkestra import ModelArkestra

runner = ModelArkestra("config.yaml", ready_timeout=120.0)  # passes to underlying runners

async with runner:
    # ... use runner.start(), stop(), ainvoke(), etc ...
# → shutdown() called automatically on exit
```

### Direct runners for standalone use

Direct runners bind to a `ConfigManager` and manage exactly one model. They also support context managers:

```python
from model_arkestra.process import ProcessModelRunner
from llm_config_manager.config_manager import ConfigManager

cm = ConfigManager("config.yaml")  # optional: pass ConfigManager directly
runner = ProcessModelRunner(cm, shutdown_timeout=20.0, ready_timeout=120.0)
```

See [API Reference — Runners](./api/runners.md) for full constructor signatures and parameters.

## Starting a Model

### Auto-assign port, pick runner from config chain

```python
await runner.start("gpt-oss-20b")
```

### Model with backend: rocm → runner picks podman from config chain (no runner= needed)

```python
await runner.start("qwen3-4b:rocm")
```

### Explicit backend override (also routes correctly via config chain)

```python
await runner.start("qwen3-4b", backend="rocm")
```

### Explicit port assignment (caller responsibility to avoid conflicts)

```python
await runner.start("qwen3.6-35B-think", port=18005)
```

### Explicit runner= override forces a specific runner type directly

```python
await runner.start("qwen3-4b", runner="podman")
```

### Backend with `runner: container` resolves to `container_type` from config

When a backend entry uses `runner: container`, the actual engine is taken from the top-level `container_type:` key in `config.yaml`:

```yaml
# config.yaml
container_type: podman   # or "docker"
```

A backend with `runner: container` will use whichever runner that resolves to — you can change it globally without touching individual backend definitions.

### Running multiple models concurrently

```python
await runner.start("gpt-oss-20b")
await runner.start("qwen3.6-35B-think")
await runner.start("gemma-4-26b-instruct")

print(runner.running_models)
# → {"gpt-oss-20b", "qwen3.6-35B-think", "gemma-4-26b-instruct"}
```

See [Architecture](./architecture.md) for port allocation and runner routing details.

## Restarting a Model

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

See [API Reference — ModelArkestra](./api/model-arkestra.md) for full method signatures.

## Sending a Prompt (Blocking)

### Single prompt

```python
result = await runner.ainvoke("gpt-oss-20b", "Explain quantum entanglement")
print(result)
# → "Quantum entanglement is a phenomenon in which..."
```

### Full conversation history (multi-turn)

```python
result = await runner.ainvoke(
    "gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a physics tutor."},
        {"role": "user", "content": "What is quantum entanglement?"},
        {"role": "assistant", "content": "It's when particles become correlated..."},
        {"role": "user", "content": "Can you explain it simply?"},
    ],
)
```

### With additional parameters

Additional keyword arguments (e.g. `temperature`, `stop`, `top_p`) are forwarded through to the underlying runner's invoke method:

```python
# Single prompt with stop tokens
result = await runner.ainvoke("qwen3-4b", "Write a poem.", stop=["\n\n"])

# With temperature
result = await runner.ainvoke("qwen3-4b", prompt="Hello", temperature=0.7)
```

## Streaming Response (Token-by-Token)

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

Token chunks may be partial — e.g., the first event might be `"token": "Hell"`, the next `"token": "o World"` rather than full words. Callers should concatenate tokens and/or check for a trailing space if word boundaries matter.

## Graceful Shutdown

### Stop a single model

```python
await runner.stop("gemma-4-26b-instruct")
```

### Stop all running models

Stops all model processes — processes/containers are killed but model entries remain in `STOPPED` state so `start()` can restart them in-place on the same port.

```python
await runner.stop_all()
```

### Full teardown

Stops all models, clears internal state, resets port allocator. For container runners, also force-removes all stopped containers. Use this for final cleanup (e.g. context manager exit, test teardown).

```python
await runner.shutdown()
```

### Async context manager (auto-calls shutdown on exit)

```python
async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")
    # ... do work ...
# → shutdown() called here, port counter resets to start value
```

See [Lifecycle](./lifecycle.md) for detailed crash detection and shutdown sequencing behavior.

## Model Introspection — Runtime State

These three methods expose live model state tracked at runtime (port, backend, runner type, health). They operate on models that have been started — before any `start()` call the lists are empty.

### List all tracked model contexts

```python
contexts = runner._get_model_contexts()
for ctx in contexts:
    print(f"{ctx.name}: port={ctx.port}, state={ctx.state}")
# → qwen3-4b: port=18000, state=RunnerState.RUNNING
```

Each `_ModelContext` carries `name`, `port`, `state` (`RunnerState` enum), `backend_id`, `runner_type`, `restart_count`, and `last_error`.

### OpenAI-compatible /v1/models listing

```python
models = runner.get_v1_models()
print(models)
# → {"object": "list", "data": [{"id": "qwen3-4b", "status": "running", ...}]}
```

Returns a dict with `"object": "list"` and a `"data"` array — one entry per configured model containing:

| Field | Source |
|---|---|
| `id` | Model name |
| `object` | `"model"` |
| `created` | Unix timestamp |
| `owned_by` | From config (`model_cfg.get("owned_by")`, default `"local"`) |
| `status` | Lowercase state string (e.g. `"running"`, `"stopped"`, `"uncached"`) |

## Log Retrieval

Fetch the last N lines from a model's in-memory log buffer:

```python
# Default: last 100 lines
lines = await runner.get_logs("qwen3-4b")
for line in lines:
    print(line)

# Custom line count
lines = await runner.get_logs("qwen3-4b", lines=50)
```

The buffer is a ring deque (default 500 lines) with per-line sequence numbers. Lines are tagged as they are appended by subprocess watchers or container log streaming. The admin endpoint returns deltas (`?since=N`) rather than full SSE streams — clients poll on a schedule to receive only new entries.

## Delta Log Protocol

Use `since` to receive only lines newer than what you've already seen:

```bash
curl 'http://localhost:8080/admin/log/qwen3-4b?since=847' \
     -H 'X-Admin-Key: your-secret-key'
# → {"since": 912, "missed_lines": 0, "lines": [{"seq": 848, "text": "..."}]}
```

Response header `X-Current-Max` gives the latest sequence number — use it as `since` on the next request. If `X-Missed-Lines > 0`, some lines were pruned from the buffer while disconnected.


## Error Handling

```python
from model_arkestra import RunnerError, ServerReadyTimeout, ModelNotStarted

try:
    result = await runner.ainvoke("my-model", prompt="...")
except ServerReadyTimeout:
    print("Model didn't start in time — check your config or GPU availability")
except ModelNotStarted:
    print("Model not in config or hasn't been started yet")
except RunnerError as e:
    # Catch-all for all runner-related failures:
    # HTTP errors (502, 503, 504), connection failures, max retries exceeded
    logger.error(f"Runner failed: {e}")
```

See [Error Hierarchy](./errors.md) for the full exception reference table.
