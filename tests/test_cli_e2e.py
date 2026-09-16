"""CLI end-to-end tests — arkestra models, arkestra chat, arkestra init.

Reuses the e2e server infrastructure from test_backend_e2e.py.
Each test starts a real ArkestraServer, runs the CLI as a subprocess,
and verifies the output.

Run:
    pytest tests/test_cli_e2e.py -v -m e2e --timeout=600
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.test_backend_e2e import (
    ADMIN_PORT,
    COMBOS,
    _build_e2e_config,
    _start_server,
    _stop_server,
    _stop_all_and_wait,
    _start_model,
)

CLI = sys.executable
SRC = str(Path(__file__).parent.parent / "src")


def _run_cli(*args: str, env_extra: dict | None = None,
             timeout: float = 30.0, stdin: str | None = None,
             module: str = "model_arkestra.cli") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC
    if env_extra:
        env.update(env_extra)
    cmd = [CLI, "-m", module, *args]
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env, input=stdin)


@pytest.fixture()
def e2e_cli_server(request):
    combo_id = request.param[0] if hasattr(request, "param") else "process-vulkan"
    backend_name = request.param[1] if hasattr(request, "param") else "vulkan-process"

    config = _build_e2e_config(combo_id, backend_name)
    proxy, client = _start_server(ADMIN_PORT, config, combo_id)

    try:
        yield {"server": proxy, "client": client,
               "base_url": f"http://127.0.0.1:{ADMIN_PORT}",
               "combo_id": combo_id}
    finally:
        try:
            _stop_all_and_wait(client, f"http://127.0.0.1:{ADMIN_PORT}")
        finally:
            _stop_server(proxy, client, ADMIN_PORT)


@pytest.mark.e2e
class TestCliModels:

    @pytest.mark.parametrize("e2e_cli_server", COMBOS, indirect=True)
    def test_list_models(self, e2e_cli_server):
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}
        result = _run_cli("models", env_extra=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert e2e_cli_server["combo_id"] in result.stdout

    @pytest.mark.parametrize("e2e_cli_server", COMBOS, indirect=True)
    def test_list_models_filter(self, e2e_cli_server):
        model_name = e2e_cli_server["combo_id"]
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}
        result = _run_cli("models", "-m", model_name, env_extra=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert model_name in result.stdout


@pytest.mark.e2e
class TestCliChat:

    @pytest.mark.parametrize("e2e_cli_server", COMBOS, indirect=True)
    def test_chat_single_message(self, e2e_cli_server):
        """Send a message via stdin, get a response, then /quit."""
        model_name = e2e_cli_server["combo_id"]
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}

        client = e2e_cli_server["client"]
        base_url = e2e_cli_server["base_url"]
        ok = _start_model(client, base_url, model_name)
        assert ok, f"Model {model_name} failed to start"

        stdin_text = "Say one word: hello\n/quit\n"
        result = _run_cli("chat", "-m", model_name, env_extra=env,
                          timeout=120, stdin=stdin_text)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Connected to" in result.stdout
        assert "Goodbye." in result.stdout

    @pytest.mark.parametrize("e2e_cli_server", COMBOS, indirect=True)
    def test_chat_with_inference_flags(self, e2e_cli_server):
        """Chat with -T and --max-tokens flags, send a message, quit."""
        model_name = e2e_cli_server["combo_id"]
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}

        client = e2e_cli_server["client"]
        base_url = e2e_cli_server["base_url"]
        ok = _start_model(client, base_url, model_name)
        assert ok, f"Model {model_name} failed to start"

        stdin_text = "Say one word: hello\n/quit\n"
        result = _run_cli("chat", "-m", model_name, "-T", "0.5",
                          "--max-tokens", "16", env_extra=env,
                          timeout=120, stdin=stdin_text)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Connected to" in result.stdout
        assert "Goodbye." in result.stdout

    @pytest.mark.parametrize("e2e_cli_server", COMBOS, indirect=True)
    def test_chat_slash_commands(self, e2e_cli_server):
        """Test /help, /system, /temperature in-loop commands."""
        model_name = e2e_cli_server["combo_id"]
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}

        client = e2e_cli_server["client"]
        base_url = e2e_cli_server["base_url"]
        ok = _start_model(client, base_url, model_name)
        assert ok, f"Model {model_name} failed to start"

        stdin_text = "/help\n/temperature 0.7\n/system You are a test bot.\n/quit\n"
        result = _run_cli("chat", "-m", model_name, env_extra=env,
                          timeout=120, stdin=stdin_text)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "/temperature" in result.stdout
        assert "temperature = 0.7" in result.stdout
        assert "System prompt updated" in result.stdout
        assert "Goodbye." in result.stdout


@pytest.mark.e2e
class TestCliInit:

    def test_init_creates_config(self, tmp_path):
        env = {"ARKESTRA_DIR": str(tmp_path)}
        result = _run_cli("init", env_extra=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "backends.yaml").exists()

    def test_init_refuses_overwrite(self, tmp_path):
        (tmp_path / "config.yaml").write_text("existing")
        env = {"ARKESTRA_DIR": str(tmp_path)}
        result = _run_cli("init", env_extra=env)
        assert result.returncode != 0
        assert "Refusing to overwrite" in result.stderr

    def test_init_force_overwrites(self, tmp_path):
        (tmp_path / "config.yaml").write_text("existing")
        env = {"ARKESTRA_DIR": str(tmp_path)}
        result = _run_cli("init", "--force", env_extra=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "existing" not in (tmp_path / "config.yaml").read_text()


@pytest.mark.e2e
class TestCliStatus:

    @pytest.mark.parametrize("e2e_cli_server", COMBOS[:1], indirect=True)
    def test_status_shows_banner(self, e2e_cli_server):
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}
        result = _run_cli("status", env_extra=env, module="model_arkestra.admin_cli")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "ModelArkestra v" in result.stdout
        assert "URL" in result.stdout
        assert "Hardware" in result.stdout
        assert "Cache" in result.stdout
        assert "Config" in result.stdout
        assert "Status" in result.stdout
        assert "ok" in result.stdout


@pytest.mark.e2e
class TestCliClusters:

    @pytest.mark.parametrize("e2e_cli_server", COMBOS[:1], indirect=True)
    def test_clusters_lifecycle(self, e2e_cli_server):
        env = {"ARKESTRA_URL": f"http://127.0.0.1:{ADMIN_PORT}"}

        # Add a cluster
        result = _run_cli("clusters", "add", "test-remote",
                          "http://10.99.99.99:8080", env_extra=env,
                          module="model_arkestra.admin_cli")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "added" in result.stdout

        # List should show it
        result = _run_cli("clusters", "list", env_extra=env,
                          module="model_arkestra.admin_cli")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "test-remote" in result.stdout
        assert "http://10.99.99.99:8080" in result.stdout

        # Delete it
        result = _run_cli("clusters", "delete", "test-remote", env_extra=env,
                          module="model_arkestra.admin_cli")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "removed" in result.stdout

        # List should no longer show it
        result = _run_cli("clusters", "list", env_extra=env,
                          module="model_arkestra.admin_cli")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "test-remote" not in result.stdout
