# Model Arkestra

Model Arkestra is a lightweight Python orchestrator for running local LLM inference engines — primarily [llama.cpp](https://github.com/ggerganov/llama.cpp) — across your choice of backends, from bare-metal subprocesses to isolated containers (Podman / Docker). It exists so you can deploy and manage models on your own hardware with **safety and stability**, without the overhead of a full-blown proxy or cluster manager.

This is **not** a replacement for [Lemonade](https://github.com/ollama/lemonade) or [llama-swap](https://github.com/sgl-project/llama-swap). No model registries, no auto-scaling, no Kubernetes babysitting. If you just want models up and running on your own GPU — with clean lifecycle management, graceful shutdowns, and restart resilience out of the box — Arkestra is a straight line between config file and inference.

> *Author's note: Consider this an experiment. My first attempt at a project coded entirely by an agent coding assistant. Using [unsloth/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_XL](https://huggingface.co/unsloth/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) on a [Corsair AI Workstation 300](https://www.corsair.com/us/en/cp/category/builds/corsair-ai-workstation-300/) with 128GB integrated memory. This is **not** vibe coding — I watched the reasoning traces carefully and intervened numerous times when things started going off track. Development started with a lemonade-inspired config.yaml, with a prompt to write python code to read and set up an equivalent python object. With several more short prompts that became [llm_config_manager](https://github.com/marc-ramsey/llm-config-manager), from which BaseModelRunner and its subclasses were derived. The only manual editing was on config.yaml as more details were added, then some cleanup edits on README.md. Most needed interventions were along the lines of "stop looping," "overthinking the problem," or "focus on the problem, nothing else," with occasional "NO, do it this way…," which can be reduced through suitable agent loops (next project). Total time from start to completion: roughly 80 hours.*

## Get Started

| If you want to… | Go here |
|---|---|
| Install Model Arkestra on your machine | See [Installation](#installation) below |
| Start using Model Arkestra in Python code | [Usage Guide](./docs/usage.md) |
| Run the OpenAI-compatible API server | [Server Documentation](./docs/server.md) |
| Manage models via web UI / Admin panel | [Admin API & Dashboard](./docs/admin.md) |
| Understand how routing, ports, and runners work | [Architecture](./docs/architecture.md) |
| Write or modify `config.yaml` | [Configuration Format](./docs/config.md) |

## Installation

```bash
cd model-arkestra
scripts/post_install.sh
```

This creates the venv, installs the package (editable mode with `[proxy]` extras), and adds `venv/bin` to your shell's PATH in both `.bashrc` and `.profile`. Source your profile or restart the terminal afterwards.

After setup, CLI commands work from any directory — no activation needed:

```bash
arkestra-server --config config.yaml --port 8080
arkestra-cli list
```

### Quick Start — Python API

```python
from model_arkestra.arkestra import ModelArkestra

async with ModelArkestra("config.yaml") as runner:
    await runner.start("qwen3-4b")                          # start a model
    result = await runner.ainvoke("qwen3-4b", "Explain quantum entanglement")
    print(result)                                           # → full response string

    async for chunk in runner.astream("qwen3-4b", {"prompt": "Write a haiku"}):
        if "token" in chunk:
            print(chunk["token"], end="", flush=True)      # streaming tokens
```

### Quick Start — Server

```bash
python -m model_arkestra.server --config config.yaml --port 8080
# or equivalently:
arkestra-server --config config.yaml --port 8080
```

Then hit `POST /v1/chat/completions` with any OpenAI-compatible client, or visit the admin dashboard at `http://localhost:8080/`.

## Architecture Overview

Model Arkestra routes models through a config-driven runner registry — each model selects a backend, which maps to a runner type (process, podman, or docker). A global port allocator distributes ports from a configured range. Port assignments are sticky: stopping and restarting a model reuses the same port.

For details see [Architecture](./docs/architecture.md) and [Lifecycle](./docs/lifecycle.md).

## Import Path

```python
from llm_config_manager.config_manager import ConfigManager    # data layer
from model_arkestra.arkestra import ModelArkestra              # orchestration (recommended)
from model_arkestra.base import BaseModelRunner                # abstract base class
from model_arkestra.process import ProcessModelRunner          # process runner
from model_arkestra.podman import PodmanModelRunner            # podman runner
from model_arkestra.docker import DockerModelRunner            # docker runner
from model_arkestra.container_runner import ContainerModelRunner  # container base class
from model_arkestra.http_client import ModelHttpClient         # lightweight HTTP client
from model_arkestra.langchain_adapter import LangChainModelAdapter  # LangChain LCEL wrapper
from model_arkestra.server import ArkestraServer             # OpenAI v1-compatible API server

# Convenience re-exports from __init__.py:
from model_arkestra import RunnerState, RunnerError, ServerReadyTimeout
from model_arkestra import ModelNotStarted, MaxRestartsExceeded, ModelShutdown
```

## Further Reading

- [API Reference — ModelArkestra](./docs/api/model-arkestra.md)
- [API Reference — Runners](./docs/api/runners.md)
- [LangChain Integration](./docs/langchain.md)
- [Error Hierarchy](./docs/errors.md)
- [HTTP Client](./docs/http-client.md)
- [Contributing & Tests](./docs/contributing.md)

## Running tests

Always use the wrapper script — it guarantees cleanup of ports, buildah dirs, and llama-server processes even if pytest is killed mid-run:

```bash
./tests/run-tests.sh -v              # unit + integration (excludes slow)
./tests/run-tests.sh --all           # includes slow tests  
./tests/run-tests.sh -m "not slow"   # same as default
```
