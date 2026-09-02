"""Unit tests for ModelArkestra core routing — port allocation, runner resolution, backend validation.

All tests use a mocked ConfigManager (MagicMock) so they do not require real models, binaries,
or container runtimes.  No non-test code is modified.
"""
from __future__ import annotations
import os

import pytest

from model_arkestra.arkestra import ModelArkestra
from model_arkestra.types import RunnerState, _ModelContext


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_cm(backend_cfg: dict | None = None, runner_cfg: dict | None = None) -> ModelArkestra:
    """Build a minimal ModelArkestra instance with mocked ConfigManager.

    The ConfigManager is constructed from an in-memory YAML file that contains only the keys
    used by the core routing logic — models, backends, runners, port range, and macros.
    """
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    os.makedirs(config_dir, exist_ok=True)
    cfg_path = os.path.join(config_dir, "arkestra_test_config.yaml")

    backend_section = ""
    if backend_cfg is not None:
        backend_lines = []
        for k, v in backend_cfg.items():
            if isinstance(v, dict):
                backend_lines.append(f"  {k}:")
                for kk, vv in v.items():
                    backend_lines.append(f"    {kk}: {vv}")
            else:
                backend_lines.append(f"  {k}: {v}")
        backend_section = "\n".join(backend_lines)

    runner_section = ""
    if runner_cfg is not None:
        for k, v in runner_cfg.items():
            if isinstance(v, dict):
                runner_section += f"  {k}:\n"
                for kk, vv in v.items():
                    runner_section += f"    {kk}: {vv}\n"
            else:
                runner_section += f"  {k}: {v}\n"

    if not runner_section:
        default_runners = (
            "  process:\n"
            "    class-name: ProcessModelRunner\n"
            "  podman:\n"
            "    class-name: PodmanModelRunner\n"
            "  docker:\n"
            "    class-name: DockerModelRunner\n"
            "  onnx:\n"
            "    class-name: OnnxRunner\n"
            "  remote:\n"
            "    class-name: RemoteModelRunner\n"
        )
    else:
        default_runners = runner_section

    if backend_cfg:
        backends_block = f"backends:\n{backend_section}"
    else:
        backends_block = "# backends disabled"

    yaml_content = f"""
default:
  model-start-port: 18000
  model-ports: 4

macros:
  ctx-size: 16384

{backends_block}

runners:
{default_runners}

models:
  test-model:
    model: dummy/dummy:Q4_K_M
    args:
      temp: 0.7
"""
    with open(cfg_path, "w") as f:
        f.write(yaml_content)

    return ModelArkestra(cfg_path, ready_timeout=2, warmup_delay=0)


# ── Tests: Port allocation ────────────────────────────────────────────────


class TestPortAllocation:
    def test_first_port_is_start_port(self):
        """worker_port() starts at models-start-port."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        assert arkestra._next_port == 18000
        port = arkestra.worker_port("m1")
        assert port == 18000

    def test_incrementing_allocation(self):
        """worker_port() increments sequentially."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        ports = [arkestra.worker_port(f"m{i}") for i in range(4)]
        assert ports == [18000, 18001, 18002, 18003]

    def test_exhaustion_raises(self):
        """worker_port() raises RuntimeError when pool is exhausted."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        # pool = 4 ports (18000–18003)
        ports = [arkestra.worker_port(f"m{i}") for i in range(4)]
        assert ports == [18000, 18001, 18002, 18003]
        with pytest.raises(RuntimeError, match="Port range exceeded"):
            arkestra.worker_port("exhausted")

    def test_shutdown_resets_port_counter(self):
        """shutdown() resets _next_port to models-start-port."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        arkestra.worker_port("m1")  # consume one port → next is 18001

        class FakeRunner:
            async def shutdown(self):
                pass

        arkestra._runners["fake"] = FakeRunner()
        arkestra._next_port = 18002  # simulate some consumption

        async def run():
            await arkestra.shutdown()

        import asyncio
        asyncio.run(run())

        assert arkestra._next_port == 18000


# ── Tests: Runner class registry ──────────────────────────────────────────


class TestRunnerClassMap:
    def test_built_in_runners_registered(self):
        """process, podman, docker built-ins are registered."""
        arkestra = _make_cm()
        assert "process" in ModelArkestra._RUNNER_CLASSES
        assert "podman" in ModelArkestra._RUNNER_CLASSES
        assert "docker" in ModelArkestra._RUNNER_CLASSES


# ── Tests: Runner instance factory ────────────────────────────────────────


class TestRunnerInstanceFactory:
    def test_same_runner_for_same_key(self):
        """_get_runner_instance returns the same object for the same (type, model) key."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        r1 = arkestra._get_runner_instance("process", "model-a")
        r2 = arkestra._get_runner_instance("process", "model-a")
        assert r1 is r2

    def test_different_keys_get_different_runners(self):
        """Different model names get separate runner instances."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        r_a = arkestra._get_runner_instance("process", "model-a")
        r_b = arkestra._get_runner_instance("process", "model-b")
        assert r_a is not r_b

    def test_unknown_runner_raises(self):
        """_get_runner_instance raises ValueError for unknown runner types."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        with pytest.raises(ValueError, match="Unknown runner type"):
            arkestra._get_runner_instance("nonexistent")



class TestStartValidation:
    def test_unknown_model_raises(self):
        """start() raises ValueError for a model not in config."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        async def run():
            await arkestra.start("nonexistent-model")
        import asyncio
        with pytest.raises(ValueError, match="Unknown model"):
            asyncio.run(run())

    def test_unknown_backend_raises(self):
        """start() raises ValueError for a backend not in config."""
        arkestra = _make_cm(
            backend_cfg={"default": "vulkan-radv"},
            runner_cfg={"default": "process"},
        )
        async def run():
            await arkestra.start("test-model", backend="ghost-backend")
        import asyncio
        with pytest.raises(ValueError, match="Unknown backend"):
            asyncio.run(run())


# ── Tests: back-compat shim properties ────────────────────────────────────

class TestBackCompatShims:
    def test_process_runner_property(self):
        """.process_runner creates and caches a process runner."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        pr = arkestra.process_runner
        assert pr is not None
        # Same instance on second access
        pr2 = arkestra.process_runner
        assert pr is pr2

    def test_podman_runner_property(self):
        """.podman_runner creates and caches a podman runner."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        pdm = arkestra.podman_runner
        assert pdm is not None
        pdm2 = arkestra.podman_runner
        assert pdm is pdm2

    def test_docker_runner_property(self):
        """.docker_runner creates and caches a docker runner."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        dkr = arkestra.docker_runner
        assert dkr is not None
        dkr2 = arkestra.docker_runner
        assert dkr is dkr2


# ── Tests: running_models aggregation ────────────────────────────────────

class TestRunningModelsProperty:
    def test_empty_when_no_runners(self):
        """running_models returns empty set when no runners exist."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        assert arkestra.running_models == set()

    def test_aggregates_across_runners(self):
        """running_models aggregates RUNNING models from all runner instances."""
        arkestra = _make_cm(runner_cfg={"default": "process"})

        # Manually create two runners with model contexts in RUNNING state
        r1 = arkestra._get_runner_instance("process", "model-a")
        r2 = arkestra._get_runner_instance("podman", "model-b")
        arkestra._runners["process:model-a"] = r1
        arkestra._runners["podman:model-b"] = r2

        # Fake RUNNING contexts
        c1 = _ModelContext("model-a", 18000)
        c1.state = RunnerState.RUNNING
        r1._models["model-a"] = c1

        c2 = _ModelContext("model-b", 18001)
        c2.state = RunnerState.RUNNING
        r2._models["model-b"] = c2

        models = arkestra.running_models
        assert "model-a" in models
        assert "model-b" in models


# ── Tests: cm delegation properties ───────────────────────────────────────

class TestCmDelegation:
    def test_cm_property_returns_config_manager(self):
        """.cm returns the internal ConfigManager."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        assert hasattr(arkestra, "cm")

    def test_get_model_delegates(self):
        """.get_model() delegates to ConfigManager."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        model = arkestra.get_model("test-model")
        assert model is not None
        assert isinstance(model, dict)

    def test_get_backend_delegates(self):
        """.get_backend() delegates to ConfigManager."""
        arkestra = _make_cm(
            backend_cfg={"default": "vulkan-radv", "vulkan-radv": {"args": {}}},
            runner_cfg={"default": "process"},
        )
        be = arkestra.get_backend("vulkan-radv")
        assert be is not None
