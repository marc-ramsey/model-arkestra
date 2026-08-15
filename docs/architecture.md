# Architecture

Model Arkestra handles port allocation and backend→runner routing.

## Component Diagram

```
┌─────────────────────────────────────────────────┐
│                  ModelArkestra                   │
│  ── port allocation, backend→runner routing      │
│                                                  │
│  __build__runner_class_map() → config-driven     │
│  runners: section maps type → class              │
│  _get_runner_instance(type, model) → lazy factory│
└──────┬──────────────────┬───────────────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌────────────────────────┐
│ProcessModel  │  │ ContainerModelRunner   │
│    Runner    │  │     (abstract base)     │
│              │  ├────────────┬───────────┤
│subprocesses  │  │PodmanModel │ DockerModel│
│              │  │   Runner   │  Runner   │
└──────────────┘  │            │           │
                  │OCI containers│OCI cont.│
                  └────────────┴───────────┘
```

## Runner Routing — Explicit Args Override Config

Runner type resolution follows a strict precedence: **explicit arguments take priority over all configuration values**. An explicit runner type that does not exist in the registry is rejected immediately.

### Explicit override (highest priority)

When `runner=` is supplied, that value is used directly:

```
runner="podman" → PodmanModelRunner   (verified against registry)
runner="process" → ProcessModelRunner  (verified against registry)
```

### Automatic resolution (config chain)

When no explicit argument is given, the runner type is resolved through config:

```
backends section exists?
  └─> model.backend → backends.<id>.runner → runners:<type> (or "process")
backends section absent?
  └─> runners:<default> (or "process")
```

See [Configuration Format](../config.md) for how `backends:` and `runners:` sections work.

## Port Allocation — A Global Counter Controlled by ModelArkestra

The starting port and range are driven entirely by config:

| Config key | Purpose |
|---|---|
| `models-start-port` | First port in the range (default 18000) |
| `model-ports` | Size of the pool — valid ports are `start_port` through `start_port + model-ports - 1` |

When `ModelArkestra.start()` is called without an explicit `port`, it allocates the next available number from this range sequentially. Once all ports in the pool are exhausted, `RuntimeError("Port range exceeded: …")` is raised immediately.

A **direct** runner instance (e.g. `ProcessModelRunner(cm)`) does **not** use a global counter — it picks `models-start-port` as the default for its first model, and subsequent calls reuse existing ports via `_dispatch()` / in-place restart.

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

See [API Reference — ModelArkestra](../api/model-arkestra.md) for full method signatures including `start()`.

## Process Lifecycle

For crash detection, shutdown sequencing, and teardown behavior, see [Lifecycle](../lifecycle.md).

## Argument Passing and Resolution

CLI arguments flow through a six-layer defaults chain before reaching subprocess construction. Values are typed YAML lists internally and reconstructed to CLI flags only at the invocation boundary.

### Defaults Cascade

Each layer fills in values missing from the one above:

1. `**overrides` — transient runtime values (single invocation)
2. Model-level `args:` list — explicit per-model overrides
3. `defaults:` top-level config section — global shared defaults
4. Backend class defaults — each backend defines fallback values
5. Engine + runner type defaults — inference engine and execution container baselines
6. Hardcoded fallbacks on the base runner class

### Engine → Target Resolution

Each backend name (e.g., `rocm`, `vulkan-radv`) implies an inference engine (llama.cpp). The engine is resolved from the backend name at runtime — users do not specify it separately. This allows each engine to maintain its own target registry and default chain without cluttering user-facing config.

### CLI Reconstruction (`_build_cmd_line`)

A private method on the runner base class converts the merged args list into a subprocess-compatible command:

| YAML Entry | Reconstructed Flag |
|---|---|
| `- temp: 0.7` | `--temp 0.7` (value appended) |
| `- jinja: true` / `- jinja: false` | `--jinja true` / `--jinja false` (boolean with value) |
| `- flash-attn: present` | `--flash-attn` (bare flag, no value) |

Keys use kebab-case in YAML, matching CLI flag names directly.

Infrastructure flags (`--port`, `--model`) are added by the method from runner context. The method is called right before `subprocess.Popen()` — this is the only place where arguments become strings. Internally everything stays structured as YAML lists.
