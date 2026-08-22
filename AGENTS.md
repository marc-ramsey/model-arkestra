# AGENTS.md

## What this is
ModelArkestra runs LLM models (llama.cpp GGUF) via a lightweight HTTP server with an admin dashboard. Models are started/stopped on demand; the server auto-allocates ports from a configurable range.

## Directory layout
```
src/model_arkestra/   — Python package
  process.py          — ProcessModelRunner (subprocess llama-server)
  common.py           — arg building, image helpers, config resolution
  admin.py            — FastAPI routes (/admin/*, /, /index.html)
  arkestra.py         — Arkestra core: lifecycle, port allocation, model registry
  base.py             — BaseModelRunner (lifecycle state machine)
  docker.py / podman.py — Container runners (Docker/Podman)
  llama_cpp.py        — LlamaCppEngine (inference arg filtering for chat/embed)
  types.py            — RunnerState enum, _ModelContext dataclass
static/index.html     — Admin dashboard (self-contained, uses CDN marked.js)
sample-config.yaml    — Reference configuration file
tests/                — Unit + e2e tests (see pyproject.toml markers)
docs/                 — Full developer docs (read these for deep reference)
```

## Config model (3-tier inheritance)
```yaml
engines:                  # shared engine defaults (binary, args, capabilities)
  llama_cpp:
    binary_dir: /path/to/bin
    binary: llama-server
    capabilities: [chat, embed]

backends:                 # hardware/platform specifics
  rocm:
    engine: llama_cpp     # ← references an engine above
    runner: podman
    args:                 # ← deep-merged with engine defaults (backend wins)
      ngl: 999

models:                   # per-model overrides (highest priority)
  my-model:
    backend: rocm
    capabilities: [embed]  # ← explicit override
```

Resolution chain: `model` → `engine` → `backend` → hardcoded fallback. Same pattern for runner type and all other config fields. See `docs/config.md`.

## Key conventions
- **Backend** = hardware platform (`rocm`, `vulkan-radv`). Defines binary location, runner type, container settings.
- **Engine** = inference engine (`llama_cpp`). Defines shared defaults inherited by backends.
- **Runner** = launch method (`process`, `podman`, `docker`). Maps to class in `base.py` registry.
- Admin API routes: see `docs/admin.md`. All responses are JSON; HTML served at `/` and `/index.html`.
- E2E tests use `-m e2e` marker on `tests/test_backend_e2e.py`. 18 tests covering process, docker, podman runtimes.

## Doc map
| Topic | File |
|-------|------|
| Config format | `docs/config.md` |
| Admin API + UI | `docs/admin.md` |
| Lifecycle & state machine | `docs/lifecycle.md` |
| Architecture overview | `docs/architecture.md` |
| Usage guide | `docs/usage.md` |
| Error codes | `docs/errors.md` |

## ⚠️ Search Instruction (Critical)
**NEVER use the built-in `web_search` tool.**
The repository is configured to use a local **SearxNG** instance. To search, you must use `bash` to `curl` the local endpoint:
`curl "http://maceo.local:8888/search?q={query}&format=json"`
Then, parse the JSON to find URLs and use `read` (or `fetch`) to consume content.

## Running tests
```bash
python -m pytest tests/ --timeout=900          # unit + e2e markers excluded
python -m pytest tests/test_backend_e2e.py -v -m e2e  # 18 e2e (~5 min)

