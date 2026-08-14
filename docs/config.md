# Configuration Format (YAML)

The runner reads from the same YAML configuration used by `ConfigManager`. This document covers the three sections relevant to Model Arkestra: top-level settings, the `env:` section, `backends:`, `runners:`, and `models:`.

See [Usage Guide](./usage.md#basic-initialization) for how to load a config file.

## Top-Level Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `models-start-port` | `int` | `18000` | First port in the auto-allocated range. |
| `model-ports` | `int` | `32` | Number of ports available — valid range is `models-start-port` through `models-start-port + model-ports - 1`. Allocation raises `RuntimeError("Port range exceeded: …")` when exhausted. |
| `warmup-time` | `float` | `10.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`. Improves reliability of the first inference request by bridging the gap between HTTP readiness and weight loading completion. |

## `env:` Section — Process Environment Variables

Environment variables defined here are merged into the subprocess/container environment at startup, in addition to the host's current environment. This is how paths like `LLAMA_CACHE` and `HF_HUB_CACHE` propagate to the server process.

```yaml
env:
  LLAMA_CACHE: /home/lemonade/hub
  HF_HUB_CACHE: /home/lemonade/hub
```

Environment variable resolution follows priority: method argument > config.yaml `env:` section > OS environment.

## `macros:` Section — Argument Templates

Macros are expanded at boot time. At `start()` runtime the backend `args` template is resolved again with concrete `${PORT}` and `${CHECKPOINT}` values.

```yaml
macros:
  llama-args: --port ${PORT} --jinja -fa on -ngl 999
  ctx-default: 131072
```

> **Note on macro syntax:** Config files use the `${MACRO}` pattern — values are resolved by `llm_config_manager` from the `macros:` section. Shell-style `${VAR:-default}` is **not** supported.

## `backends:` Section — Executable Registry

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

### Backend Configuration Keys

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

## `runners:` Section — Runner Class Registry

This section maps runner type strings to concrete class names. Built-in types (`process`, `podman`, `docker`) are auto-registered; config entries can override them or add new ones (as long as the class is importable from this module).

```yaml
runners:
  podman: PodmanModelRunner
  docker: DockerModelRunner
  default: ProcessModelRunner
```

The `default` key specifies the fallback runner type when neither `backends.<id>.runner` nor a model's own `backend` field provides one.

See [Architecture](./architecture.md) for the runner routing precedence table.

## `models:` Section — Model Definitions

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

## Related Documentation

- [Usage Guide](./usage.md) — how to load and use a config file
- [Architecture](./architecture.md) — runner routing based on backend config
- [Admin API](./admin.md) — runtime config management via HTTP endpoints
- [Server Documentation](./server.md) — CLI options for server startup
