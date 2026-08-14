# API Reference — Runners

For direct use when you only need subprocess-based or container-based execution without the orchestration layer. Each runner binds to a `ConfigManager` instance and manages exactly one model. Its `stop()` takes no arguments and shuts down that model; `stop_all()` delegates to `stop()`. The remaining methods (`start`, `ainvoke`, `astream`, `request`, `running_models`) follow the same contract as described in [ModelArkestra](./model-arkestra.md).

## ProcessModelRunner

Bound to a `ConfigManager` instance; does not launch any processes until `start()` is called.

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_manager` | `ConfigManager` | *(required)* | The configuration source for model definitions and global settings. |
| `shutdown_timeout` | `float` | `20.0` | Seconds to wait after SIGHUP before escalating to SIGKILL during shutdown. |
| `ready_timeout` | `float` | `120.0` | Maximum seconds to wait for the server's `/health` endpoint to respond after starting. |
| `ready_poll_ms` | `float` | `100.0` | Milliseconds between consecutive health-check polls during startup. |
| `warmup_delay` | `float` | *(see below)* | Seconds to wait after `/health` returns OK before marking the model as `"running"`. Read from config key `warmup-time` (default 10s); pass directly as keyword arg to override. |

### Restart behavior

`restart_delay` and `restart_limit` control automatic restart behavior inherited from `BaseModelRunner`: when a managed process exits unexpectedly, the runner polls and attempts up to `restart_limit` restarts spaced by `restart_delay` seconds.

## ContainerModelRunner (Abstract Base)

Container runners add port-drain logic. The `port_drain_timeout` parameter is used to wait for port listeners to drain before reassignment. It has no effect on direct subprocess execution.

## Inheritance Hierarchy

```
BaseModelRunner          ← abstract base class, shared params + restart behavior
├── ProcessModelRunner   ← subprocess management
└── ContainerModelRunner  ← abstract, container-specific logic
    ├── PodmanModelRunner
    └── DockerModelRunner
```

## Constructor Parameters (Inherited from BaseModelRunner)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `restart_delay` | `float` | — | Seconds between automatic restart attempts after unexpected exit. |
| `restart_limit` | `int` | — | Maximum number of automatic restart attempts before transitioning to ERROR state. |
| `port_drain_timeout` | `float` | — | Container runners only: seconds to wait for port listeners to drain on stop. |

## Related Documentation

- [ModelArkestra API](./model-arkestra.md) — orchestration layer (recommended entry point)
- [Architecture](../architecture.md) — runner routing, port allocation
- [Lifecycle](../lifecycle.md) — crash detection by backend, shutdown sequencing
