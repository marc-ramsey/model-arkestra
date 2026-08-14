# Error Hierarchy

All exceptions inherit from `RunnerError`, enabling callers to catch all runner-related failures with a single clause:

```python
from model_arkestra import RunnerError

try:
    result = await runner.ainvoke("my-model", prompt="...")
except RunnerError as e:
    logger.error(f"Runner failed: {e}")
```

## Exception Reference

| Exception | When raised |
|---|---|
| `ServerReadyTimeout` | The server did not become ready (health check returned no 200 within `ready_timeout`). Raised only on total health-check exhaustion. |
| `ModelNotStarted` | A request was made for a model name that doesn't exist in the config, or the model hasn't been started yet (`start()` not called). Raised by `_dispatch` when no tracked context exists, and during `BaseModelRunner.start()` when model config lookup fails. |
| `ModelShutdown` | A request was made to a model that has been explicitly stopped via `stop()` or `stop_all()` (state is STOPPED or STOPPING). Raised by `_dispatch`. |
| `MaxRestartsExceeded` | The runner crashed and exhausted all automatic restart attempts (`restart_limit`). State transitions to ERROR; raised by `_dispatch` on any subsequent request. |
| `RunnerError` | Generic base class for all runner failures, including HTTP errors (502, 503, 504) from the server, connection failures, and general request errors. |

## Usage Examples

```python
from model_arkestra import ServerReadyTimeout, ModelNotStarted, MaxRestartsExceeded, RunnerError

# Catch everything at once
try:
    result = await runner.ainvoke("my-model", prompt="...")
except RunnerError as e:
    logger.error(f"Runner failed: {e}")

# Handle specific cases
try:
    await runner.start("qwen3-4b")
except ServerReadyTimeout:
    print("Model didn't become ready — check GPU or binary path")
except ModelNotStarted:
    print(f"Model 'qwen3-4b' not found in config")

# Post-error requests raise MaxRestartsExceeded
try:
    await runner.ainvoke("unstable-model", prompt="hi")
except MaxRestartsExceeded:
    print("Model crashed too many times — check logs with admin endpoint")
```

## Related Documentation

- [Usage Guide](./usage.md) — error handling examples in context
- [Lifecycle](./lifecycle.md) — crash detection and state transitions
- [API Reference — ModelArkestra](./api/model-arkestra.md) — method signatures that raise these exceptions
