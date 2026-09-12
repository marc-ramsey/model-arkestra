# Admin API (`admin.py`)

Model Arkestra ships an administrative panel that integrates into the same FastAPI app via `ArkestraServer`. It provides endpoints for monitoring and managing models, with optional API-key authentication on all admin paths.

## Public Endpoints (No Auth)

The `/api/*` namespace exposes read-only endpoints. When `api_key` is set in `config.default-env.api_key` (via the computed `_env` section), these endpoints require a `Bearer` token matching the key — enabling API-level auth.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/models` | List all cached models with name, model-ref, size (GB) |
| `GET` | `/api/clusters` | List federated clusters with name and url |

When `api_key` is configured, include `Authorization: Bearer <api_key>` to access these endpoints.

## Configuring Auth

Both namespaces gate independently. Set the keys in `config.yaml`'s `default-env:` section:

```yaml
default-env:
  admin_key: supersecret    # gates /admin/* — header: Bearer <key>
  api_key: apipublic        # gates /api/* — header: Bearer <key>
```

Neither key is required unless configured. With both keys set, `/admin/*` and `/api/*` require `Authorization: Bearer <key>`. Admin routes accept either key; API routes accept either key as well.

## Initialization

Admin routes are installed automatically when calling `server.get_app()` — pass the ``admin_key`` argument to enable:

```python
server = ArkestraServer(
    "config.yaml",
    port=8080,
    admin_key="your-secret-key",  # gates all /admin/* paths
)
app = server.get_app()
```

The key resolves with priority: **constructor argument** > `_env.admin_key` (computed from `config.default-env.admin_key` + process env) > disabled (no auth).

When `admin_key` is provided, every request to `/admin/*` must include the header:

```http
Authorization: Bearer your-secret-key
```

Missing or incorrect keys return `401 Unauthorized`. Public paths (`/`, `/index.html`) are unaffected.

## Endpoints

| Method | Path | Auth Required | Description |
|---|---|---|---|
| `GET` | `/` | No | Serves the admin dashboard HTML page |
| `GET` | `/index.html` | No | Same as `/` (explicit route) |
| `GET` | `/admin/models` | Yes | Full context listing for all configured models |
| `GET` | `/admin/config` | Yes | List model names in config |
| `POST` | `/admin/config` | Yes | Create a new model entry |
| `GET` | `/admin/config/{model}` | Yes | Retrieve a single model's configuration |
| `PUT` | `/admin/config/{model}` | Yes | Update an existing model's configuration (no restart) |
| `DELETE` | `/admin/config/{model}` | Yes | Remove a model from config (stops first if running) |
| `GET` | `/admin/clusters` | Yes | List clusters with health check |
| `POST` | `/admin/clusters/{name}` | Yes | Add a managed cluster |
| `DELETE` | `/admin/clusters/{name}` | Yes | Remove a managed cluster |
| `POST` | `/admin/start/{model}` | Yes | Start or restart a model (with optional transient overrides) |
| `POST` | `/admin/stop/{model}` | Yes | Stop a running model |
| `POST` | `/admin/eject/{model}` | Yes | Remove model from cache, clear contexts (no config change) |
| `GET` | `/admin/log/{model}?since=N&lines=M` | Yes | Delta or snapshot log query |
| `GET` | `/admin/logs?since=N&lines=M` | Yes | Server-level log entries (proxy traffic, lifecycle events) |
| `GET` | `/admin/images` | Yes | List configured container images with runner type and availability |
| `POST` | `/admin/images/build` | Yes | Build a single backend's image (body: `{"backend": "rocm"}`) |
| `DELETE` | `/admin/images/{image_tag}` | Yes | Remove an image from the local store |
| `POST` | `/admin/stop-all` | Yes | Stop all running models — models restart implicitly on next inference request |
| `POST` | `/admin/shutdown` | Yes | Full server teardown — stops uvicorn and all models |
| `POST` | `/admin/restart/{model}` | Yes | Stop and restart a running/loading model (accepts override params) |
| `POST` | `/admin/pull/{model}` | Yes | Start pulling a model checkpoint from HuggingFace |
| `POST` | `/admin/cancel-pull/{model}` | Yes | Cancel an in-progress model pull |

### GET /admin/models

Returns a list of all configured models with their full runtime context. Models that have been started get their real state from the runner context; unstarted models get a constructed entry:

```json
{
  "models": [
    {
      "id": "qwen3.5-4b",
      "status": "running",
      "port": 18000,
      "runner_type": "process",
      "backend_id": "rocm",
      "args": ["temp", 0.7, "top-k", 20, "ctx-size", 131072],
      "repo": "hugging-face",
      "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
      "capabilities": ["chat"]
    },
    {
      "id": "gemma-4-e2b",
      "status": "uncached",
      "port": null,
      "runner_type": null,
      "backend_id": "vulkan-radv",
      "args": ["temp", 0.7, "top-p", 0.95, "ctx-size", 131072],
      "repo": "hugging-face",
      "model": "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
      "capabilities": []
    }
  ],
  "backends": {"vulkan-radv": {...}, "rocm": {...}},
  "runner_types": ["process", "podman", "docker"]
}
```

| Field | Source |
|---|---|
| `id` | Model name from config |
| `status` | One of: `running`, `loading`, `error`, `stopping`, `stopped`, or `uncached` |
| `port` | Allocated port (null if not running) |
| `runner_type` | Runner type string (null if not running) |
| `backend_id` | Resolved backend (from context or config fallback) |
| `args` | Model args from config |
| `repo` | Model repo identifier |
| `model` | Model path within repo |
| `tags` | Capability tags for the model — resolves per-model `tags` → backend-declared `backends.<id>.capabilities` → hardcoded fallback `["chat"]` |

**Status values:**
- `running`, `loading`, `error`, `stopping` — real states from active runner contexts
- `stopped` — model was previously started but is now stopped; its weights **are** in the HF cache
- `uncached` — model exists in config but is not yet pulled
- `downloading` — model checkpoint is actively being pulled from HuggingFace

| Field | Source |
|---|---|
| `id` | Model name from config |
| `status` | One of the status values above |
| `port` | Allocated port (null if not running) |
| `runner_type` | Runner type string (null if not running) |
| `backend_id` | Resolved backend (from context or config fallback) |
| `args` | Model args from config |
| `repo` | Model repo identifier |
| `model` | Model path within repo |
| `tags` | Capability tags for the model |
| `downloading` | `true` if a checkpoint download is in progress, `false` otherwise |

Top-level metadata (`backends`, `runner_types`) is static for the lifetime of the server.

### POST /admin/config

Create a new model entry in config. Returns `201 Created` on success.

```bash
curl -X POST 'http://localhost:8080/admin/config' \
     -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer your-secret-key' \
     -d '{
       "name": "my-new-model",
       "repo": "hugging-face",
       "model": "unsloth/my-model-GGUF:Q4_K_M",
       "backend": "vulkan-radv",
       "args": {"temp": 0.7, "ctx-size": 131072}
     }'
```

| Field | Required | Description |
|---|---|---|
| `name` | No | Model name in config. Defaults to the last segment of `model` (before `:` and `/`). |
| `repo` | **Yes** | HuggingFace repo identifier. |
| `model` | **Yes** | HF model path within repo. |
| `args` | No | Command-line arguments for the model. |
| `backend` | No | Backend ID from config (e.g., `vulkan-radv`, `rocm`). |
| `capabilities` | No | Capability tags (default `["chat"]` if empty). |
| `tags` | No | Free-form tags. |

Returns `400 Bad Request` if `repo` or `model` is missing. Returns `409 Conflict` if the model name already exists.

### GET /admin/config/{model}

Retrieve a single model's configuration:

```bash
curl 'http://localhost:8080/admin/config/qwen3.5-4b' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns:
```json
{
  "ok": true,
  "model": "qwen3.5-4b",
  "status": "running",
  "config": {
    "repo": "hugging-face",
    "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
    "args": {"temp": 0.7, "top-k": 20, "ctx-size": 131072},
    "backend": "rocm"
  },
  "tags": ["chat"]
}
```

Returns `404` if the model is not found.

### PUT /admin/config/{model}

Update an existing model's configuration and write it to disk. **Does not restart** the model — use `POST /admin/start/{model}` separately.

```bash
curl -X PUT 'http://localhost:8080/admin/config/qwen3.5-4b' \
     -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer your-secret-key' \
     -d '{"args": {"temp": 1.0, "ctx-size": 32768}, "capabilities": ["chat"]}'
```

Valid fields: `args`, `repo`, `model`, `backend`, `capabilities`, `runner`, `tags`.

Returns `404` if the model does not exist. Returns `500` on write failure (config is rolled back).

### DELETE /admin/config/{model}

Remove a model from configuration entirely. Stops the model first if running, then removes its config entry permanently.

```bash
curl -X DELETE 'http://localhost:8080/admin/config/qwen3-4b' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns:
```json
{"ok": true, "model": "qwen3-4b"}
```

If the model is running during deletion, it is stopped first (silently). Returns `404` if the model does not exist.

### POST /admin/start/{model}

Start a model from a stopped or error state. Returns the port assigned.

**State gate:** Only accepts models in `STOPPED` or `ERROR` state. Models in `LOADING`, `RUNNING`, `STOPPING`, `UNCACHED`, or `DOWNLOADING` return `409 model not available` — use `POST /restart/{model}` for those states.

```bash
# Start a stopped model
curl -X POST 'http://localhost:8080/admin/start/qwen3.5-4b' \
     -H 'Authorization: Bearer your-secret-key'

# Start with transient overrides (no config change)
curl -X POST 'http://localhost:8080/admin/start/qwen3.5-4b' \
     -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer your-secret-key' \
     -d '{"backend": "docker", "repo": "hugging-face", "model": "unsloth/Qwen3.5-4B-GGUF:Q5_K_M"}'
```

Transient overrides are **not** persisted to disk. They apply only to this invocation:

Infra keys (resolved by ``ModelArkestra.start()``):
- ``args`` — transient args override
- ``backend`` — backend ID override (runner resolves from config chain)
- ``runner`` — explicit runner type (``process``, ``podman``, ``docker``, or ``remote``)
- ``model`` — model path within repo override (transient only)
- ``max_log_lines`` — per-invocation log buffer size

Any other keys are treated as inference parameters for llama.cpp. Only those present in the engine's ``LLAMA_INFER_ARGS`` whitelist (e.g. `temp`, `top-p`, `reasoning-budget`) reach CLI construction; unknown keys are silently dropped to prevent subprocess crashes.

Returns `503` if the model fails to start within `ready_timeout`.

### POST /admin/restart/{model}

Restart a model from a running or loading state. Stops the current instance, then starts fresh with optional override params.

**State gate:** Only accepts models in `STOPPED`, `ERROR`, `LOADING`, or `RUNNING` state. Models in `STOPPING`, `UNCACHED`, or `DOWNLOADING` return `409 model not available`.

```bash
curl -X POST 'http://localhost:8080/admin/restart/qwen3.5-4b' \
     -H 'Authorization: Bearer your-secret-key'

# Restart with new params
curl -X POST 'http://localhost:8080/admin/restart/qwen3.5-4b' \
     -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer your-secret-key' \
     -d '{"temp": 0.7, "top-p": 0.95}'
```

### POST /admin/stop/{model}

Stops the named model. Returns `202 Accepted` if the model is already stopped/stopping (no-op), or `200 OK` after a successful stop.

```json
// 200 — model was running, now stopped
{"ok": true, "model": "qwen3.5-4b", "previous_state": "running"}

// 202 — already stopped (no-op)
{"ok": true, "model": "qwen3.5-4b", "previous_state": "stopped"}
```

Returns `404` if the model is not found in any runner context.

### POST /admin/stop-all

Stops all running models at once. Models remain configured and will **restart implicitly** on their next inference request (same lazy-start behavior as a cold server).

```bash
curl -X POST 'http://localhost:8080/admin/stop-all' \
     -H 'Authorization: Bearer your-secret-key'
```

**When models are running:**
```json
{"ok": true, "message": "Stopped 2 model(s) — will restart implicitly on next request", "stopped": ["qwen3.5-4b", "gemma-4-e2b"]}
```

**When no models are running:**
```json
{"ok": true, "message": "No models running — nothing to stop", "stopped": []}
```

Always returns `200 OK`. The HTTP server stays alive; only model runners are stopped.

### POST /admin/shutdown

Full server teardown — stops the uvicorn HTTP listener and shuts down all model runners. This is an irreversible operation: after shutdown, model entries are cleared and cannot be restarted without restarting the entire server process.

```bash
curl -X POST 'http://localhost:8080/admin/shutdown' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns immediately with `200 OK`:
```json
{"ok": true, "message": "Server shutting down"}
```

Then shuts down in background:
1. All model runners stopped (same sequencing as `stop_all`) — graceful signals → 20s timeout → SIGKILL
2. Watcher tasks cancelled and awaited
3. `_models` and `_watchers` dictionaries cleared
4. Container runners force-removed (`podman rm -f` / `docker rm -f`)
5. Uvicorn HTTP server stops — process exits

This endpoint always returns `200 OK`. The response is sent before shutdown begins.

### GET /admin/clusters

List all managed clusters with connectivity health checks.

```bash
curl 'http://localhost:8080/admin/clusters' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns:
```json
{
  "clusters": [
    {"name": "gpu-lab-1", "url": "http://192.168.1.42:18000", "healthy": true},
    {"name": "gpu-lab-2", "url": "http://192.168.1.43:18000", "healthy": false}
  ]
}
```

Each cluster entry includes `name`, `url`, and `healthy` (bool, based on `/health` ping).

### POST /admin/clusters/{name}

Add a managed cluster. The cluster name becomes part of model naming convention (`<cluster>/<model-id>` for federation).

```bash
curl -X POST 'http://localhost:8080/admin/clusters/gpu-lab-3' \
     -H 'Content-Type: application/json' \
     -H 'Authorization: Bearer your-secret-key' \
     -d '{"url": "http://192.168.1.44:18000", "admin-key": "secret"}'
```

Returns:
```json
{"ok": true, "cluster": "gpu-lab-3"}
```

### DELETE /admin/clusters/{name}

Remove a managed cluster from configuration.

```bash
curl -X DELETE 'http://localhost:8080/admin/clusters/gpu-lab-2' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns:
```json
{"ok": true, "cluster": "gpu-lab-2"}
```

### POST /admin/eject/{model}

Removes a model from cache without modifying config. Stops the model first (if running), deletes its cached files from the HF cache directory, and clears all runner context entries.

**Safety check:** before deleting, if another *running* model shares the same underlying cache directory (same resolved model path under `HF_HUB_CACHE`), the eject is rejected:
```json
{"detail": "Model 'qwen3.5-4b' is in use by other running runners: gemma-v2, llama-rdma"}
```

Returns `200 OK` with a detail report on success:
```json
{
  "ok": true,
  "model": "qwen3.5-4b",
  "cache_deleted": true,
  "cache_path": "/home/user/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF",
  "contexts_cleared": 1
}
```
If the model has no model configured, or the cache directory doesn't exist, `cache_deleted` is `false`. Returns `404` if the model doesn't exist in config. Returns `409 Conflict` when a shared-cache conflict prevents eject.

### POST /admin/pull/{model}

Start pulling a model's checkpoint from HuggingFace. Returns immediately with `200 OK` — the pull runs as a background task. Progress is streamed to the model's log buffer (poll via `GET /admin/log/{model}`).

```bash
curl -X POST 'http://localhost:8080/admin/pull/qwen3.5-4b' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns on success:
```json
{"ok": true, "model": "qwen3.5-4b"}
```

Returns `200` with `{"already_pulling": true}` if a pull is already in progress for this model.

Returns `409 Conflict` if:
- Model is already running (`"Cannot pull: model is running"`)
- Model is stopping (`"Cannot pull: model is stopping"`)
- Model is already uncached (`"Model 'qwen3.5-4b' is already uncached"`)

Returns `404` if the model is not in config.

**Progress via log endpoint:**
```json
{"lines": [
  {"seq": 1, "text": "[pull] qwen3.5-4b: Fetching 3 files (14.2GB total)"},
  {"seq": 2, "text": "[pull] qwen3.5-4b: 25% (3.55/14.2GB, 180MB/s)"},
  {"seq": 3, "text": "[pull] qwen3.5-4b: 100% (14.2/14.2GB)"}
]}
```

### POST /admin/cancel-pull/{model}

Cancel an in-progress model pull. The pull task is cancelled and partially downloaded files may remain in cache (subsequent pulls resume from cache).

```bash
curl -X POST 'http://localhost:8080/admin/cancel-pull/qwen3.5-4b' \
     -H 'Authorization: Bearer your-secret-key'
```

Returns on success:
```json
{"ok": true, "model": "qwen3.5-4b"}
```

Returns `404` if no active pull exists for the model.

### GET /admin/log/{model}?since=N&lines=M

Return log lines for a running model. This is an **HTTP delta endpoint** — no streaming, no SSE. Clients poll on a schedule (typically 1–2s) to receive only new log lines since their last request.

**Snapshot mode** (no `since` parameter):
```bash
curl 'http://localhost:8080/admin/log/qwen3.5-4b' \
     -H 'Authorization: Bearer your-secret-key'
```
Returns the full current log buffer:
```json
{"lines": ["[INFO] Loading model...", "[INFO] Ready"]}
```

**Delta mode** (`since` parameter):
```bash
curl 'http://localhost:8080/admin/log/qwen3.5-4b?since=847&lines=50' \
     -H 'Authorization: Bearer your-secret-key'
```
Returns only log lines with sequence number greater than `847`:
```json
{
  "since": 912,
  "missed_lines": 0,
  "lines": [
    {"seq": 848, "text": "[INFO] KV cache init..."},
    {"seq": 849, "text": "[INFO] Loading model weights..."}
  ]
}
```

**Response headers** (present on every response):
| Header | Description |
|---|---|
| `X-Current-Max` | Latest log line sequence number on the server |
| `X-Missed-Lines` | Number of lines pruned from the buffer before or during the requested range. Zero if all requested lines are still in the buffer. |

**Missed-lines scenario:** when a client is disconnected too long, older lines may fall off the ring buffer (default capacity: 2000 lines). If `since=N` but line N has already been evicted, the response includes `X-Missed-Lines: K` indicating how many lines were skipped. The returned JSON still includes any remaining lines newer than the gap.

**Delta protocol usage pattern:**
1. Client starts with `since=0` to get all available lines
2. Each response header `X-Current-Max` becomes the next request's `since`
3. On reconnect, client sends its last known `since` value
4. If `X-Missed-Lines > 0`, the client knows some log lines were lost

**Implementation notes:** Log lines are tagged with a per-model monotonic sequence number as they are appended to the ring buffer by subprocess watchers (`ProcessRunner`) or container log streaming (`podman logs -f` / `docker logs -f`). The buffer uses a fixed-size ring (default 2000 lines, configurable via `max_log_lines` in config or startup override). Only lines within the current window are available — older entries are automatically evicted.

### GET /admin/logs?since=N&lines=M

Return **server-level** log entries for the entire ModelArkestra instance. This is a separate ring buffer from per-model logs (`/admin/log/{model}`) and captures proxy traffic, model lifecycle events, and server startup/shutdown.

**Request format:**
```bash
curl 'http://localhost:8080/admin/logs?since=0&lines=200' \
     -H 'Authorization: Bearer your-secret-key'
```

**Response (same shape as per-model log):**
```json
{
  "seq": 47,
  "missed_lines": 0,
  "lines": [
    {"seq": 1, "text": "[action=start server port=8080]"},
    {"seq": 2, "text": "[action=start model=qwen3 port=18001]"},
    {"seq": 3, "text": "[action=req model=qwen3 method=POST path=/v1/chat/completions status=200 latency_ms=420 tokens=240]"},
    {"seq": 4, "text": "[action=stream_start model=gemma messages=2]"},
    {"seq": 5, "text": "[action=stream_end model=gemma duration_ms=1850 tokens=342] status=ok"}
  ]
}
```

**Response headers:** Same as per-model log — `X-Current-Max` (latest seq), `X-Missed-Lines` (evicted entries before the `since` point).

**Log entry types and their metadata fields:**
| action | Fields logged |
|---|---|
| `start server` | port |
| `start model` | model, port |
| `stop model` | model |
| `shutdown` | — (server-level teardown) |
| `req` | model, method, path, status, latency_ms, tokens |
| `stream_start` | model, messages |
| `stream_end` | model, duration_ms, tokens, status (`ok`, `error`, or `no_tokens`) |

The buffer is configurable via `app-log-lines` in the YAML config (default: 2000 entries). In the admin dashboard, select **"Server logs"** from the log model dropdown to view these entries alongside per-model output.

### GET /admin/images

List all container images configured in `backends.yaml`, along with their runner type and availability status.

Returns a JSON array:
```json
[
  {
    "backend_id": "rocm",
    "runner": "podman",
    "runtime_detected": true,
    "image": "ark-llama:rocm",
    "containerfile": "Containerfile.rocm",
    "available": false
  }
]
```

| Field | Description |
|---|---|
| `backend_id` | Backend identifier from the `backends:` section of backends.yaml |
| `runner` | Resolved runner type (`podman`, `docker`, or `process`) |
| `runtime_detected` | Whether the container runtime for this runner is available on PATH |
| `image` | Full image tag configured for this backend |
| `containerfile` | Name of the Containerfile (resolved to `tests/files/<name>`) |
| `available` | Whether the image exists in the local container store (only checked when `runtime_detected` is true) |

### POST /admin/images/build

Build a single backend's container image. The `backend` key must be provided — no "build all" mode.

Request body:
```json
{"backend": "rocm"}
```

The endpoint resolves the configured runner type for the backend, detects which runtime (podman or docker) is available on PATH, then runs the appropriate build command. If the configured runtime isn't present, returns gracefully:
```json
{"skipped": true, "reason": "runner=podman but no 'podman' binary found on PATH", "image": "ark-llama:rocm"}
```

On attempt (whether successful or not):
```json
{
  "backend": "rocm",
  "image": "ark-llama:rocm",
  "success": false,
  "runtime": "podman",
  "output": "STEP 1/10: FROM ...\nError: ...",
  "error": "Failed to resolve the transaction:\nNo match for argument: hip-runtime-rocm"
}
```

Build runs synchronously with a 600s timeout. The full stdout/stderr from the container runtime is returned in `output`. Returns `400` if `backend` is missing from the body, or `404` if no Containerfile is found for the backend.

### DELETE /admin/images/{image_tag}

Remove an image tag from the local store. The tag must match one configured in a backend's `image:` entry in backends.yaml.

```json
DELETE /admin/images/ark-llama:rocm
```

Returns:
```json
{"removed": true, "image": "ark-llama:rocm", "error": null}
```

Resolves the backend this image belongs to, determines its runner type, and runs `podman rmi -f` or `docker rmi -f` accordingly. Returns `404` if the tag isn't configured in any backend. Returns `{"skipped": true}` with a reason when the runtime isn't available.

## Admin Dashboard (`static/index.html`)

Model Arkestra ships a single-file, zero-dependency admin dashboard — a vanilla JavaScript application served at `/` that provides a web UI for the Admin API. It requires no build step, no frameworks, and runs entirely in the browser.

### Deployment

Place `static/index.html` anywhere the server serves static files. The dashboard is served automatically when the admin routes are mounted:

```python
server = ArkestraServer(
    "config.yaml",
    port=8080,
    admin_key="my-secret",   # gates /admin/* endpoints
)
```

With `admin_key` set, visiting the server root (e.g. `http://localhost:8080/`) serves the dashboard HTML page.

### Configuration

At the very top of the script block is a single configurable constant:

```js
const ADMIN_KEY = 'whatever';   // must match admin_key in ArkestraServer
```

The dashboard automatically attaches it as the `Authorization: Bearer` header on every API call.

### Layout

The dashboard uses a **session-based vertical split layout**: cluster tree on the left (top), docked sessions on the right (bottom). API key users see only a single chat pane.

#### Admin Layout (Cluster Tree + Session Dock)

| Section | Content |
|---|---|
| **Cluster Tree** (top/left) | Hierarchical accordion: Local cluster → model entries, Remote clusters (via `/admin/clusters`) → model entries. Each model row shows status dot, name, size badge, and inline action buttons (+ Chat / + Log / ▶ / ■ / ⏏). |
| **Session Dock** (bottom/right) | Pinned chat/log sessions. Each session is an expandable card with a ChatPane or LogPane. Drag divider to resize (persisted in `localStorage`). Layout toggle button switches between vertical and horizontal split. |

#### API User Layout (single pane)

Non-admin users see only **one ChatPane** — no left panel, no log viewer. Logs are strictly admin-only.

- Click **+** on any model row to spawn a new chat session in the dock.

### Features

**Cluster Tree** — Hierarchical navigation of all known clusters and their models.

**Model rows** (cluster tree) — Each model entry shows:
- Status dot (running=green, stopped=black, loading=amber pulse, error=red, uncached=gray)
- Model name and size badge
- **Inline action buttons**: `+` Chat, `+` Log (admin only), `▶` Start, `■` Stop, `⏏` Eject, `✕` Cancel pull

**Sessions** (dock) — Pinned conversation/log workspaces:
- Click `+` Chat on a model row to create a new chat session in the dock.
- Click `+` Log on a model row (admin only) to create a log stream session.
- Each session is expandable/collapsible; click header to toggle.
- Close button (`×`) removes session from dock and aborts any active streams.

**Chat pane** — Conversational inference UI per-session:
- **Model selector dropdown**: choose which model to chat with in this session.
- **Ephemeral params panel**: Temperature, Top P, Max Tokens, Top K — click "Params ▸" to expand. Values persist live to IndexedDB (personal defaults) but are not saved to disk unless explicitly triggered.
- **Chat-triggered auto-start**: If a model is stopped when user sends a message, the UI shows "Starting…" status and starts the model using current chat params before streaming.
- Full conversation history maintained in-memory across turns (OpenAI-compatible message format).
- Token-by-token SSE streaming with animated cursor during generation.
- Markdown rendering via `marked.js` from CDN — code blocks, bold, lists, inline code all rendered.

**Log pane** (admin only) — Terminal-style log viewer per-session:
- **Model logs**: Select a specific model from the dropdown to view its process stdout/stderr via `GET /admin/log/{model}`
- **Server logs**: Select "Server logs" from the dropdown to view proxy traffic and lifecycle events via `GET /admin/logs`
- Both share the same delta-polling pattern (1–2s interval) with `?since=N` cursor
- Smart auto-scroll: scrolls to bottom during active streaming *unless* you've scrolled up to read older logs

### Technical Details

- **Zero dependencies**: No build step, no frameworks. Only `marked.js` loaded from CDN for markdown rendering.
- **Layout engine**: JSON-driven (app.json → SessionLayout → ClusterTree + SessionDock via SplitPane). Resize persisted in `localStorage`.
- **IndexedDB**: Personal chat defaults per-model stored in `arkestra` database, `settings` object store. Keyed by model ID.
- **Session lifecycle**: Sessions created via `+ Chat` / `+ Log` buttons on model rows. Each session maintains its own state: chat history (in-memory), log polling timer, abort controller for streaming.
- **Auth gating**: `HAS_ADMIN_KEY` derived from `<meta name="arkestra-admin-key">` content. Non-admin users get a single ChatPane only.

## Related Documentation

- [Server Documentation](./server.md) — how ArkestraServer works, CLI options
- [Configuration Format](./config.md) — YAML config that admin endpoints modify
- [Usage Guide](./usage.md) — Python API equivalent of admin operations
- [Lifecycle](./lifecycle.md) — state transitions reflected in `/admin/models`

## `arkestra-admin` CLI Tool

A command-line interface for all admin endpoints, installed alongside `arkestra` and `arkestra-server`. It targets the server via the shared connection surface (`--url` / `ARKESTRA_URL` / config `default.url`) and reads the admin key from config.yaml by default.

```bash
arkestra-admin --url http://localhost:8080 --api-key SECRET <command>
```

### Authentication Priority
1. `--api-key KEY` flag (highest)
2. `$ARKESTRA_API_KEY` environment variable
3. `admin_key` from `config.yaml`'s `default-env:` section

### Commands

| Command | Description |
|---|---|
| `arkestra-admin models` | List all configured models with status, port, backend |
| `arkestra-admin start <name>` | Start a model (supports `--port`, `--backend`, `--runner`, `key=value` params) |
| `arkestra-admin restart <name>` | Restart a running/loading model (supports override params) |
| `arkestra-admin stop <name>` | Stop a running model |
| `arkestra-admin stop-all` | Stop all running models |
| `arkestra-admin config list` | List model names in config |
| `arkestra-admin config get <name>` | Show one model's full config + runtime status |
| `arkestra-admin config set <name> key=value` | Update a model field (e.g., `backend=rocm`) |
| `arkestra-admin config create --model PATH` | Add a new model to config |
| `arkestra-admin config rm <name>` | Remove a model from config |
| `arkestra-admin clusters list` | List managed clusters with health status (API only) |
| `arkestra-admin clusters add <name> --url URL` | Add a managed cluster (API only) |
| `arkestra-admin clusters rm <name>` | Remove a managed cluster (API only) |
| `arkestra-admin logs <name\|all> [--lines 100]` | Tail model or global server logs |
| `arkestra-admin eject <name>` | Stop model and delete its cached files |
| `arkestra-admin images list` | Show OCI image availability per backend |
| `arkestra-admin images build <backend> [--tag TAG]` | Build an OCI container image |
| `arkestra-admin images rm <image_tag>` | Remove a container image |
| `arkestra-admin pull <name>` | Pull model checkpoint from HuggingFace |
| `arkestra-admin pull-stop <name>` | Cancel an in-progress pull |
| `arkestra-admin shutdown` | Gracefully stop the server |

### Examples

```bash
# List models (auto-reads admin key from config)
arkestra-admin --url http://127.0.0.1:8080 models

# Start with overrides
arkestra-admin start qwen3-4b --backend vulkan-radv temp=0.7 top-k=20

# Tail logs with custom count
arkestra-admin logs gemma-4-e2b --lines 50

# Build and check OCI images
arkestra-admin images build rocm-container --tag rocm-7.14
arkestra-admin images list

# Full server shutdown on remote host
arkestra-admin --url http://remote-host:8080 --api-key mysecret shutdown
```

### JSON Output

Add `--json` to any command for machine-readable output:

```bash
arkestra-admin --url http://localhost:8080 config get qwen3-4b --json
```
