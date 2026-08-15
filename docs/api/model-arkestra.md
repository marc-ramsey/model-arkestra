# API Reference — ModelArkestra

The `ModelArkestra` class is the centralized entry point for orchestrating multiple models across different runners. It does not launch any processes at init time — models must be started explicitly via `start()`.

## Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str` | *(required)* | Path to the YAML config file. |
| `start_port` | `int` | `18000` | Fallback starting port — only used when `models-start-port` is absent from config. Port allocation reads the actual values from config at init time. |
| `**runner_kwargs` | — | — | Passed through to each runner instance (e.g. `ready_timeout`, `warmup_delay`). |

### Backward-compat shims

The properties `.process_runner`, `.podman_runner`, and `.docker_runner` still exist for backward compatibility. Each delegates to the unified lazy factory (`_get_runner_instance`) so they behave identically to the config-driven path.

## Methods

### `async start(model_name: str, **overrides) -> None`

Starts the server process or container for *model_name*. Accepts optional keyword overrides — all are transient (never persisted to disk). Keys separated into infrastructure and inference:

| Key | Type | Description |
|---|---|---|
| `port` | `int \| None` | Explicit port (auto-allocated from range if omitted) |
| `backend` | `str \| None` | Backend/target override — engine resolved from backend name |
| `runner` | `str \| None` | Explicit runner type (`process`, `podman`, `docker`) |
| *other* | any | Inference param (e.g., `temp=1.0`, `top-p=0.95`). Passed through to runner and converted to ``--flag value`` CLI flags at subprocess boundary. |

Infra keys (`port`, `backend`, `runner`) control routing and lifecycle. Everything else is an inference parameter — snake_case in Python code (e.g., `top_k`), kebab-case in YAML config (e.g., `top-p`). Both produce ``--top-k`` / ``--top-p`` CLI flags.

Runner type resolution follows precedence: explicit `runner=` arg → `backends.<id>.runner` → `runners:` config → default "process".

If the model already exists and is in a STOPPED state, it restarts in-place on the same port. If it is already RUNNING, `/health` is polled — 200 returns immediately, anything else continues startup. Polls `/health` until ready or timeout. Raises `ServerReadyTimeout` on health-check failure (not on intermediate 502/504; those raise `RunnerError`).

```python
# Simple — use config as-is
await arkestra.start("qwen3-4b")

# Override just temperature for this invocation
await arkestra.start("qwen3-4b", temp=1.0)

# Explicit port + backend override
await arkestra.start("qwen3-4b", port=18000, backend="vulkan-radv")
```

### `async stop(model_name: str) -> None`

Stops the model matching *model_name*. Sends the appropriate shutdown signal (see [Lifecycle](../lifecycle.md)).

### `async restart(model_name: str, **overrides) -> None`

Stops the running instance and starts a new one on the **same port**. Infra keys (`backend`, `runner`) override routing. Everything else is an inference param.

### `async stop_all() -> None`

Stops all model processes. Model entries remain in `_models` with state `STOPPED` — calling `start()` again on a stopped model restarts it in-place on the same port. For container runners, containers are **not** removed (only force-removed during `shutdown()`).

### `async shutdown() -> None`

Full teardown: stops all models, clears all runner and model entries, and resets the port allocator to `models-start-port`. For container runners, also sends `rm -f` on every stopped container. Use this for final cleanup (e.g. context manager exit, test teardown).

### `async ainvoke(model_name: str, prompt: str = "", backend: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs) -> str`

Sends a blocking completion request and returns the full response string. Retries up to 12 times on connection errors or 503 status (with 2.5s backoff). Raises `RunnerError` for 502/504 (upstream failures), connection unreachable, or max retries exceeded.

Accepts either a single `prompt` string **or** a full `messages` list (OpenAI-compatible format) for multi-turn conversations:

```python
# Single prompt (legacy)
result = await runner.ainvoke("qwen3-4b", prompt="Hello")

# Full conversation history — includes system message, user turns, assistant response
result = await runner.ainvoke(
    "qwen3-4b",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2 + 2 = 4"},
        {"role": "user", "content": "And 3 times that?"},
    ],
    temperature=0.7,          # forwarded to the runner
)

# Single prompt with stop tokens
result = await runner.ainvoke("qwen3-4b", "Write a poem.", stop=["\n\n"])
```

When both `prompt` and `messages` are provided, `messages` takes precedence.

Additional keyword arguments (e.g. `temperature`, `stop`, `top_p`) are forwarded through to the underlying runner's invoke method.

### `async astream(model_name: str, payload: Dict[str, Any], backend: str | None = None) -> AsyncIterator[Dict[str, Any]]`

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

### `async request(model_name: str, path: str, **kwargs) -> dict | bytes`

Low-level HTTP forwarder for custom server endpoints not covered by `ainvoke()` or `astream()`. Forwards *kwargs* as the request payload and returns the parsed JSON response (or raw bytes if the endpoint does not return JSON). Only sends POST requests with 15s timeout. Raises `RunnerError` on responses with status ≥ 400.

## Properties

### `running_models: set[str]`

Read-only property returning the names of all models currently in `"running"` state, aggregated across all runner instances.

## Model Introspection

### `_get_model_contexts() -> list[_ModelContext]`

Internal aggregation: returns every tracked `_ModelContext` across all runners. Each context carries `name`, `port`, `state` (`RunnerState` enum), `backend_id`, `runner_type`, `restart_count`, and `last_error`. Callers who need detailed runtime info should use this.

### `get_v1_models() -> dict`

OpenAI-compatible `/v1/models` response. Returns a dict with `"object": "list"` and a `"data"` array — one entry per configured model. See [Usage Guide](../usage.md#model-introspection) for details.

## Related Documentation

- [Architecture](../architecture.md) — runner routing, port allocation
- [Lifecycle](../lifecycle.md) — crash detection, shutdown sequencing
- [Configuration Format](../config.md) — how `backends:` and `runners:` sections work
- [Error Hierarchy](../errors.md) — exception reference
