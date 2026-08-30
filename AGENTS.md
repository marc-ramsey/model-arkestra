# AGENTS.md

## Recommendations and rules
- When writing something intended for human consumption, (comment, commit message, reply to prompt) use as few words as possible. Pick every word meticulously to reduce the volume to a strict minimum. Be down to the point. Less is more.

- Avoid superlatives and praise. Stop telling me I am absolutely right. Give me the cold hard truth.

- Avoid magic numbers and strings by extracting recurring or meaningful values into descriptive constants (const) or enums. Keep self-explanatory, one-off values inline to avoid clutter. If a value comes from a spec (e.g. HTTP 200 OK), use a constant regardless.

- Reduce code indentation. Avoid Arrow Anti-Pattern. Leverage early return and continue.

- Keep function names short. Less than 30 characters.

- Let the reader of the code breathe. Add empty lines between logical blocks of code.

- Add a small, to the point, comment to explain *what* the block does and *why*. Use examples when possible. Propose ASCII drawings to explain complete systems.

- Treat member visibility changes as a breaking design shift. Keep all fields and functions private unless external access is strictly required by the design. Prompt the user for explicit approval before changing any access modifier from private to internal or public.

- Program to levels of abstraction. Lower-level mechanics (e.g., raw hardware I/O, sector parsing, direct socket streams) must be encapsulated in a dedicated driver/abstraction layer. Expose clean, high-level APIs to the rest of the application so calling code works with domain concepts, not raw implementation details.

- Don't touch blocks of code unrelated to the feature you implement. e.g. Don't add comments to a block of code if you did not create it or modify it. As much as possible try to minimize the number of changed lines when implementing a feature.

- Strictly adhere to the layered boundary hierarchy: each layer may only communicate with its immediate neighbor directly below it. Never "punch holes" through layers (e.g., controllers or UI components must never directly call database queries, raw hardware drivers, or low-level network clients; always route through the intermediate service/abstraction layer).

## While CODING follow these laws:
Law 1: Never edit without approval — when I say "discuss", you say "here's the full plan with exact lines and files, confirm before any edit." No implementation until I explicitly approve.
Law 2: Show me the complete diff before any change — not a description of intent, the actual git diff output showing every line that will change. If it looks wrong, I stop there. 
Law 3: When I start going in circles on an issue, say "STOP" and wait for my correction — don't keep trying variations. The first correct approach I rejected means something fundamental is off.
Law 4: DO NOT USE sed to edit files! Use the edit tool correctly by reading the file first and matching sufficient context. No overlapping edits EVER, do them separately. DO NOT USE sed to edit files!                      

## When you write a commit message, follow these 7 rules:
Rule 1: Separate the subject line from the body with a single blank line.
Rule 2: Limit the subject line to 72 characters where possible.
Rule 3: Capitalize the first letter of the subject line.
Rule 4: Do not end the subject line with a period.
Rule 5: Use the imperative mood in the subject line (e.g., "Fix bug," "Add feature," 
        not "Fixed" or "Adds"). Test formula: It must complete the sentence: "If applied,
        this commit will [your subject line here]".
Rule 6: Wrap the body text manually at 72 characters to prevent Git formatting issues.
Rule 7: Use the body to explain what and why vs. how. Assume the code explains the how;
        the message must explain the context and reasoning. 

- If the prompt indicates that a bug is being fixed, don't write the fix right away. First write the test. Observe it failing. Then write the fix. And observe the test passing.        

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
```

