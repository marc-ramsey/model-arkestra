"""Tests for SdcppEngine — CLI arg building for stable-diffusion.cpp sd-server.

Mirrors the llama-cpp engine's contract: a merged config dict + port in, a list
of CLI tokens out. File keys become named file flags; everything else passes
through as --kebab flags; booleans are presence-only.
"""
from __future__ import annotations

from model_arkestra.engines import ENGINES, SdcppEngine


class TestBuildCliArgs:
    def test_port_carried_by_listen_port(self):
        cli = SdcppEngine.build_cli_args({"model": "m.gguf"}, 18018)
        assert cli[cli.index("--listen-port") + 1] == "18018"

    def test_listen_ip_emitted_when_set(self):
        cli = SdcppEngine.build_cli_args(
            {"model": "m.gguf", "listen-ip": "0.0.0.0"}, 18018)
        assert cli[cli.index("--listen-ip") + 1] == "0.0.0.0"

    def test_model_becomes_single_dash_m(self):
        cli = SdcppEngine.build_cli_args({"model": "/models/qwen-image.gguf"}, 1)
        assert "-m" in cli
        i = cli.index("-m")
        assert cli[i + 1] == "/models/qwen-image.gguf"

    def test_file_sidecars_map_to_flags(self):
        cli = SdcppEngine.build_cli_args({
            "model": "m.gguf",
            "llm": "te.gguf",
            "vae": "vae.gguf",
            "tokenizer": "tok.json",
        }, 2)
        assert cli[cli.index("--llm") + 1] == "te.gguf"
        assert cli[cli.index("--vae") + 1] == "vae.gguf"
        assert cli[cli.index("--tokenizer") + 1] == "tok.json"

    def test_empty_file_keys_dropped(self):
        cli = SdcppEngine.build_cli_args({"model": "m.gguf", "vae": ""}, 3)
        assert "--vae" not in cli
        # model still present
        assert "-m" in cli

    def test_port_key_never_emitted_as_flag(self):
        cli = SdcppEngine.build_cli_args({"model": "m.gguf", "port": 42}, 7)
        assert "--port" not in cli
        # port is carried by --listen-port only
        assert cli[cli.index("--listen-port") + 1] == "7"

    def test_integer_passthrough(self):
        # threads is a real sd-server startup flag (unlike width/height, which
        # are request-time params, not CLI flags).
        cli = SdcppEngine.build_cli_args(
            {"model": "m.gguf", "threads": 16}, 8)
        assert cli[cli.index("--threads") + 1] == "16"

    def test_bool_true_is_presence_only(self):
        cli = SdcppEngine.build_cli_args(
            {"model": "m.gguf", "eager-load": True}, 9)
        assert "--eager-load" in cli
        # no value token after a presence flag
        idx = cli.index("--eager-load")
        assert idx + 1 >= len(cli) or cli[idx + 1].startswith(("-"))

    def test_bool_false_dropped(self):
        cli = SdcppEngine.build_cli_args(
            {"model": "m.gguf", "eager-load": False}, 10)
        assert "--eager-load" not in cli

    def test_snake_case_key_kebabified(self):
        cli = SdcppEngine.build_cli_args(
            {"model": "m.gguf", "offload_to_cpu": True}, 11)
        # config key with underscore becomes a kebab flag
        assert "--offload-to-cpu" in cli

    def test_string_passthrough(self):
        cli = SdcppEngine.build_cli_args(
            {"model": "m.gguf", "backend": "diffusion=cpu"}, 12)
        assert cli[cli.index("--backend") + 1] == "diffusion=cpu"
