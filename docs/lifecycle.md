# Process Lifecycle

This document covers the full lifecycle of models from startup through crash detection to shutdown — how state transitions work and what happens under the hood for each backend type.

## State Machine

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

  pull(model) ─► "downloading" ─► checkpoint present ─► state = "stopped"
                                                        └─► state = "error" (on failure)
```

## Crash Detection by Backend

| Runner | Behavior |
|---|---|
| **Process** | Polls subprocess exit code via `process.wait()`. On unexpected exit (non-zero), automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. The watcher task runs in the background. |
| **Podman / Docker** | Polls container status every 2 seconds; on unexpected exit (`exited`, `dead`), automatically attempts up to `restart_limit` restarts spaced by `restart_delay`. The watcher task runs in the background. |
| **Remote** | No local process, no port allocation, no crash detection. Lifecycle commands (start/stop) proxy via HTTP to the remote worker. Inference requests are forwarded directly. |

## Shutdown Sequencing (`stop` / `stop_all`)

1. State transitions to **STOPPING** immediately — prevents any watcher from attempting a restart.
2. Signal is sent:
   - **Process**: SIGHUP to process group → waits up to `shutdown_timeout` (default 20s) → escalates to SIGKILL if still running.
   - **Podman / Docker**: Container stop signal (`podman stop --time <port_drain_timeout>` or `docker stop --time <port_drain_timeout>`) with graceful shutdown timeout (`port_drain_timeout` default 20s).
3. After termination, watcher tasks are cancelled and state transitions to **STOPPED**. Port is released (with `port_drain_timeout` wait for container runners to let the port listener drain).

## Full Teardown (`shutdown`)

Identical to stop sequencing with two additions:

1. All watcher tasks are cancelled and awaited.
2. `_models` and `_watchers` dictionaries are cleared — models cannot be restarted after shutdown.
3. **For container runners**: After stopping all containers, force-removes them via `podman rm -f` / `docker rm -f`.

## Stopped Models Are Restartable

After `stop()` or `stop_all()`, models remain tracked with state `STOPPED`. Calling `start()` again on a stopped model restarts it in-place on the same port. This applies to both the orchestration layer and direct runners.

Full teardown (`shutdown`) is the only operation that clears model entries entirely.

## Model Pulls

Model checkpoints can be pulled independently from starting the model via
`POST /admin/pull/{model}`. This is useful for large models (10–100GB) where
the pull may take considerably longer than the server startup.

**Lifecycle:**

1. Pull request creates a context (if none exists) with state `DOWNLOADING`.
2. Background task calls `huggingface_hub.snapshot_download` via `asyncio.to_thread()`
   to avoid blocking the event loop.
3. Progress is streamed to the model's log buffer — visible via the log pane.
4. On success, state transitions to `STOPPED` (checkpoint present, ready to start).
5. On failure, state transitions to `ERROR` with error message in `last_error`.
6. `POST /admin/pull/stop/{model}` cancels the pull task.

**Cancellation:**

Cancelling a pull task stops the background coroutine. Partially downloaded
files may remain in the cache. On the next pull attempt, `snapshot_download`
resumes from the existing cache.

**Shutdown:**

All active pull tasks are cancelled during `shutdown()`. On server restart,
models revert to their config-defined state; models with a checkpoint will be STOPPED.

## Error States

When crash detection exhausts all restart attempts, the model transitions to ERROR state:

```python
await runner.start("unstable-model")  # crashes repeatedly
# → state = "error" after restart_limit exceeded

try:
    await runner.ainvoke("unstable-model", prompt="hi")
except MaxRestartsExceeded:
    print("Model exhausted restart attempts — check logs or config")
```

See [Error Hierarchy](./errors.md) for the full exception reference.

## Related Documentation

- [Architecture](./architecture.md) — port allocation, runner routing
- [API Reference — ModelArkestra](./api/model-arkestra.md) — `start()`, `stop()`, `shutdown()` signatures
- [Configuration Format](./config.md) — `restart_delay`, `restart_limit` config keys
