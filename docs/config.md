# Configuration Format (YAML)

The runner reads from the same YAML configuration used by `ConfigManager`. This document covers the top-level settings, the `env:` section, `images:`, `backends:`, `runners:`, and `models:`.

See [Usage Guide](./usage.md#basic-initialization) for how to load a config file.

## Top-Level Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `models-start-port` | `int` | `18000` | First port in the auto-allocated range. |
| `model-ports` | `int` | `32` | Number of ports available — valid range is `models-start-port` through `models-start-port + model-ports - 1`. Allocation raises `RuntimeError("Port range exceeded: …")` when exhausted. |
| `warmup-time` | `float` | `10.0` | Seconds to wait after `/health` returns OK before marking the model as `"running"`. Improves reliability of the first inference request by bridging the gap between HTTP readiness and weight loading completion. |
| `default-image` | `str` | `ark-llama:vulkan-radv` | Default container image tag for runners when no backend specifies an `image:`. Used as fallback in both PodmanModelRunner and DockerModelRunner. |
| `app-log-lines` | `int` | `2000` | Number of server-level log entries to retain in the global ring buffer (Admin API → `/admin/logs`). Each entry is ~200 bytes, so 2000 lines ≈ 400 KB. |
| `log-buffer-size` | `int` | `2000` | Maximum lines retained per-model in the process log ring buffer. Applied to both `_ModelContext` instances and runner-level buffers. |

## `env:` Section — Process Environment Variables

Environment variables defined here are merged into the subprocess/container environment at startup, in addition to the host's current environment. This is how paths like `HF_HUB_CACHE` propagate to the server process.

```yaml
env:
  HF_HUB_CACHE: /home/lemonade/hub
```

Environment variable resolution follows priority: method argument > config.yaml `env:` section > OS environment.

## `images:` Section — Container Image Registry

This section defines container images and their source Containerfiles, one entry per backend type. It is read by `default_image_for_backend()` (to resolve image tags) and `containerfile_for_backend()` (to locate the build artifact).

```yaml
images:
  rocm:
    image: ark-llama:rocm
    containerfile: Containerfile.rocm
    default: false
  vulkan-radv:
    image: ark-llama:vulkan-radv
    containerfile: Containerfile.vulkan-radv
    default: true

default-image: ark-llama:vulkan-radv
```

### Image Configuration Keys

| Key | Type | Description |
|---|---|---|
| `image` | str | Full image tag (e.g. `ark-llama:rocm`). Used as the default for this backend when no explicit `image:` is set in the backend config. |
| `containerfile` | str | Filename of the Containerfile used to build the image. Resolved relative to `tests/files/` at project root. |
| `default` | bool | Marks which image is the global default. Only one entry should be `true`. Controls the `default-image` top-level value. |

The `default-image` top-level key serves as the ultimate fallback — when no images section, no backend `image:`, and no runner class has a configured value, this tag is used.

## `engines:` Section — Inference Engine Registry

Each engine defines shared defaults (binary location, default args, capabilities) that are inherited by backends referencing it via `engine: <name>`. Backend-level values override engine-level ones.

```yaml
engines:
  llama_cpp:
    binary_dir: /home/marc/local/llama.cpp/build/bin
    binary: llama-server
    args:
      ctx-size: 16384
      jinja: true
    capabilities:
      - chat
      - embed
```

### Engine Configuration Keys

| Key | Type | Description |
|---|---|---|
| `binary_dir` | str | Default directory containing the inference engine binary. Backends inherit this when they do not specify their own. |
| `binary` | str | Default binary name (e.g. `"llama-server"`). Inherited by backends that omit it. |
| `args` | dict | Default argument values for models using this engine. Deep-merged into the args cascade with engine values as fallbacks. |
| `capabilities` | list[str] | Capability types supported by this engine (e.g. `["chat", "embed"]`). Used as a fallback when per-model and per-backend capabilities are not set. |

## `backends:` Section — Executable Registry

Each backend entry specifies its argument list and which runner type should handle it. Backend args use the same YAML dict format as model args — a flat mapping of flag names to values that merges into the defaults cascade.

```yaml
backends:
  vulkan-radv:
    engine: llama_cpp
    runner: process     ← maps this backend to ProcessModelRunner
    args:
      flash-attn: "on"
      hf: ${CHECKPOINT}
    image: ark-llama:vulkan-radv
  rocm:
    engine: llama_cpp
    runner: podman      ← maps this backend to PodmanModelRunner
    args:
      flash-attn: "on"
      hf: ${CHECKPOINT}
    image: ark-llama:rocm
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
| `capabilities` | list[str] | Capability types this backend supports (e.g. `["chat", "embed"]`). Used as the admin UI's available-capabilities fallback when per-model `capabilities` is not set. Hardcoded baseline is `["chat"]`. |

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

The `args:` key holds a **flat YAML dict** where each key is a CLI flag name (kebab-case) and each value becomes the flag's argument. Entries are resolved in config-dict order.

| Style | YAML Entry | Reconstructed Flag |
|---|---|---|
| Standard value | `temp: 0.7` | `--temp 0.7` |
| Boolean `true` | `jinja: true` | `--jinja` (presence-only, no value) |
| Boolean `false` | `no-mmap: false` | omitted from the list entirely |
| HuggingFace repo | `hf: user/repo:Q4_K_M` | `-hf user/repo:Q4_K_M` |

Keys use kebab-case in YAML, matching CLI flag names directly. **Special key `hf`** maps to the llama-server `-hf` flag for loading models from HuggingFace Hub (not a regular `--hf` flag). Non-arg model metadata (checkpoint, capabilities, max_log_lines) lives at the model level alongside `args:`, not inside it — only `args:` keys reach subprocesses.

Backend args use the same dict format.

```yaml
backends:
  rocm:
    runner: process
    args:
      flash-attn: "on"
```

> **Note:** YAML interprets bare `on` and `off` as boolean values. Always quote them (`"on"`, `"off"`) for llama-server flags that accept string values.

Model args and backend args merge through the same defaults cascade — dict-style entries from each layer are combined with later layers overriding earlier ones (last-wins for overlapping keys).

```yaml
models:
  qwen3-4b:
    checkpoint: unsloth/Qwen3-4B-GGUF:Q4_K_M
    backend: rocm

    args:
      temp: 0.7
      top-p: 20
      ctx-size: 16384
      jinja: true

    max_log_lines: 500        # per-model override of the default 500 (stored in admin context, not passed to subprocess)
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

Each backend (e.g., `rocm`, `vulkan-radv`) declares its inference engine in `engine:`. Engine defaults (binary location, arg defaults, capabilities) are inherited by the backend — backend-level values override engine-level ones.

### Admin UI Rendering

The admin dashboard maps arg styles to typed controls: standard values use number/text inputs, booleans use toggle switches, presence flags use checkboxes. Overrides sent via start/stop buttons are typed JSON matching these styles — the UI never sends opaque strings.

## Related Documentation

- [Usage Guide](./usage.md) — how to load and use a config file
- [Architecture](./architecture.md) — runner routing based on backend config
- [Admin API](./admin.md) — runtime config management via HTTP endpoints
- [Server Documentation](./server.md) — CLI options for server startup
