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

## `backends:` Section — Executable Registry

Each backend entry specifies its argument list and which runner type should handle it. Backend args use the same YAML list format as model args — a sequence of single-key entries that merge into the defaults cascade.

```yaml
backends:
  vulkan-radv:
    runner: process     ← maps this backend to ProcessModelRunner
    args:
      - ngpu_layers: -1
      - flash-attn: present
  rocm:
    runner: podman      ← maps this backend to PodmanModelRunner
    args:
      - ngpu_layers: -1
      - threads: 8
  default: vulkan-radv         ← global default backend (also has runner: process)
```

When a model has `backend: rocm`, the routing chain resolves: `rocm` → `runner: podman` → `runners.podman` → `PodmanModelRunner`.

### Backend Configuration Keys

| Key | Type | Description |
|---|---|---|
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

Each model entry uses `checkpoint` for the model path, structurally separated `args:` for CLI flags, and optionally `backend` to override which backend registry entry is used. **There is no `cmd` field** — commands are assembled at start time via the backend registry.

### `args:` Field Format

The `args:` key holds a YAML list of single-key entries representing typed CLI flag values instead of an opaque string. Each entry falls into one of three styles:

| Style | YAML Entry | Reconstructed Flag |
|---|---|---|
| Standard value | `- temp: 0.7` | `--temp 0.7` |
| Boolean with value | `- jinja: true` / `- jinja: false` | `--jinja true` (or `false`) |
| Presence-only flag | `- flash-attn: present` | `--flash-attn` (bare flag, no value) |

Keys use kebab-case in YAML, matching CLI flag names directly. Non-arg model metadata (checkpoint, capabilities, max_log_lines) lives at the model level alongside `args:`, not inside it — only `args:` keys reach subprocesses.

Backend args use the same list format. A backend entry looks like:

```yaml
backends:
  rocm:
    runner: process
    args:
      - ngpu-layers: -1
      - flash-attn: present
```

Model args and backend args merge through the same defaults cascade — no macro layer, no string substitution.

```yaml
models:
  qwen3-4b:
    checkpoint: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: rocm

    args:
      - temp: 0.7
      - top-p: 20
      - ctx-size: 16384
      - jinja: true
      - flash-attn: present

    max_log_lines: 500        # non-arg — stays in admin context
    capabilities: ["chat"]     # non-arg — never reaches subprocess
```

### Argument Resolution & Defaults Chain

Arguments follow a six-layer resolution cascade — each layer fills in values missing from the one above:

1. `**overrides` passed to `start()` — transient, single invocation only
2. Model-level `args:` dict — explicit per-model overrides
3. `defaults:` section at top level of config.yaml — global shared defaults
4. Backend class defaults — each backend defines fallback values
5. Engine + runner type defaults — inference engine and execution container baselines
6. Hardcoded fallbacks on the base runner class

Each backend (e.g., `rocm`, `vulkan-radv`) implies an inference engine (llama.cpp). The engine is resolved from the backend name — users do not specify it separately. This keeps config minimal while each engine maintains its own target registry and default chain.

### Admin UI Rendering

The admin dashboard maps arg styles to typed controls: standard values use number/text inputs, booleans use toggle switches, presence flags use checkboxes. Overrides sent via start/stop buttons are typed JSON matching these styles — the UI never sends opaque strings.

## Related Documentation

- [Usage Guide](./usage.md) — how to load and use a config file
- [Architecture](./architecture.md) — runner routing based on backend config
- [Admin API](./admin.md) — runtime config management via HTTP endpoints
- [Server Documentation](./server.md) — CLI options for server startup
