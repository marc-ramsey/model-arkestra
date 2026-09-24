"""Unit tests for ModelArkestra core routing — port allocation, runner resolution, backend validation.

All tests use a mocked ConfigManager (MagicMock) so they do not require real models, binaries,
or container runtimes.  No non-test code is modified.
"""
from __future__ import annotations
import os

import pytest

from model_arkestra.arkestra import ModelArkestra
from model_arkestra.types import RunnerState, _Model


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_cm(backend_cfg: dict | None = None, runner_cfg: dict | None = None,
             models_section: str = "") -> ModelArkestra:
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
            "    class-name: ProcessRunner\n"
            "  podman:\n"
            "    class-name: PodmanRunner\n"
            "  docker:\n"
            "    class-name: DockerRunner\n"
            "  onnx:\n"
            "    class-name: OnnxRunner\n"
            "  remote:\n"
            "    class-name: RemoteRunner\n"
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
  {models_section}
"""
    with open(cfg_path, "w") as f:
        f.write(yaml_content)

    return ModelArkestra(cfg_path, ready_timeout=2, warmup_delay=0)


# ── Tests: Port allocation ────────────────────────────────────────────────


class TestPortAllocation:
    def test_first_port_is_start_port(self):
        """worker_port() starts at models-start-port."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        port = arkestra.worker_port("m1")
        assert port == 18000

    def test_models_property_returns_live_model_map(self):
        """models is a name → _Model map backed by the registry (no .all() bug)."""
        arkestra = _make_cm(runner_cfg={"default": "process"},
                            models_section="m1:\n    model: dummy/x:Q4\n")
        models = arkestra.models
        assert isinstance(models, dict)
        assert "m1" in models
        assert models["m1"].name == "m1"

    def test_model_obj_lookup(self):
        """model_obj returns the live _Model, or None for unknown names."""
        arkestra = _make_cm(runner_cfg={"default": "process"},
                            models_section="m1:\n    model: dummy/x:Q4\n")
        assert arkestra.model_obj("m1") is not None
        assert arkestra.model_obj("nope") is None

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
        """shutdown() resets the registry port counter to models-start-port."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        arkestra.worker_port("m1")  # consume one port → next is 18001

        class FakeRunner:
            async def shutdown(self):
                pass

        arkestra._runners["fake"] = FakeRunner()
        arkestra._registry._next_port = 18002  # simulate some consumption

        async def run():
            await arkestra.shutdown()

        import asyncio
        asyncio.run(run())

        assert arkestra._registry._next_port == 18000


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
        r1 = arkestra.get_runner_instance("process", "model-a")
        r2 = arkestra.get_runner_instance("process", "model-a")
        assert r1 is r2

    def test_different_keys_get_different_runners(self):
        """Different model names get separate runner instances."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        r_a = arkestra.get_runner_instance("process", "model-a")
        r_b = arkestra.get_runner_instance("process", "model-b")
        assert r_a is not r_b

    def test_unknown_runner_raises(self):
        """_get_runner_instance raises ValueError for unknown runner types."""
        arkestra = _make_cm(runner_cfg={"default": "process"})
        with pytest.raises(ValueError, match="Unknown runner type"):
            arkestra.get_runner_instance("nonexistent")



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
            models_section="ghost-model:\n    model: dummy/x:Q4\n",
        )
        async def run():
            await arkestra.start("ghost-model", backend="ghost-backend")
        import asyncio
        with pytest.raises(ValueError, match="Unknown backend"):
            asyncio.run(run())

    def test_failed_start_records_last_error(self):
        """A start failure must record ctx.last_error so the UI can show why."""
        arkestra = _make_cm(
            backend_cfg={"default": "vulkan-radv", "vulkan-radv": {"runner": "process"}},
            runner_cfg={"default": "process"},
            models_section="ghost-model:\n    model: dummy/x:Q4\n",
        )

        from model_arkestra.base import BaseRunner

        class FailingRunner(BaseRunner):
            async def _start_model_process(self, ctx, model_data, model_name):
                raise RuntimeError("Port 12000 is already in use")

            async def _stop_model_process(self, ctx):
                pass

        runner = FailingRunner(arkestra._cm, arkestra=arkestra)
        arkestra._runners["process:ghost-model"] = runner

        import asyncio
        with pytest.raises(RuntimeError, match="already in use"):
            asyncio.run(arkestra.start("ghost-model"))

        ctx = arkestra.model_obj("ghost-model")
        assert ctx.state == RunnerState.STOPPED
        assert "Port 12000 is already in use" in (ctx.last_error or "")


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
        r1 = arkestra.get_runner_instance("process", "model-a")
        r2 = arkestra.get_runner_instance("podman", "model-b")
        arkestra._runners["process:model-a"] = r1
        arkestra._runners["podman:model-b"] = r2

        # Fake RUNNING contexts
        c1 = _Model("model-a", 18000)
        c1._state = RunnerState.RUNNING
        r1._ctx = c1
        c1._runner = r1

        c2 = _Model("model-b", 18001)
        c2._state = RunnerState.RUNNING
        r2._ctx = c2
        c2._runner = r2

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
        arkestra = _make_cm(
            runner_cfg={"default": "process"},
            models_section="my-model:\n    model: dummy/x:Q4\n",
        )
        model = arkestra.get_model("my-model")
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
