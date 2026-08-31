# Contributing & Tests

## Running Tests

The project uses `pytest` with a modular fixture system (`tests/conftest.py`) that provides shared model runners, port cleanup, and per-test isolation. All tests are collected from `tests/`, `tests/unit/`, and any module-scoped test files.

```bash
# Run all tests (fast + slow)
python -m pytest -v --tb=short

# Run only fast tests (skips models that download/run LLMs)
python -m pytest -v --tb=short -m "not slow"

# Run only slow integration tests (requires a working GPU/backend or container runtime)
python -m pytest -v --tb=short -m slow
```

### Fixture Summary

| Fixture | Scope | Purpose |
|---|---|---|
| `mr` | module | Shared `ModelArkestra` instance — port allocation and runner maps are shared across the module |
| `_cleanup_after_test` | function (autouse) | Stops all models after each test so every method sees a clean slate |
| `_cleanup_ports` | module (autouse) | Safety net that kills any lingering processes on configured ports before/after each module |
| `podman_cleanup` | function | Tracks podman containers/tasks/ports with guaranteed teardown — opt-in per-test fixture |

## Import Path

All public imports are listed in the [README](../README#import-path).

## Shared Utilities (`common.py`)

Functions used by the runner classes to build commands from configuration data:

| Function | Description |
|---|---|
| `build_model_args(cm, model_name, inference_kwargs=None, override_backend=None) → dict[str, Any] | None` | Merges model-level args with inference kwargs into a flat dict. Returns ``None`` if the model has no args section and no inference kwargs. Infrastructure flags (`--port`, `--model`) are injected by the engine layer, not this function. |
| `_resolve_backend(cm, model, model_name, override_backend) → str` | Determines which backend ID to use for a model. Resolution order: `override_backend` → `model["backend"]` → `backends.default`. Used internally by `build_model_args`. |

These replace the former `ConfigManager.assemble_command()` and `ConfigManager._resolve_backend_for_model()` methods, keeping command assembly logic inside arkestra where it belongs.

## Related Documentation

- [Configuration Format](./config.md) — YAML structure that configures tests
- [API Reference — ModelArkestra](./api/model-arkestra.md) — classes being tested
- [Lifecycle](./lifecycle.md) — behavior tested by integration tests
