"""Shared connection surface for ``arkestra`` (client) and ``arkestra-server``.

Both entry points register the same flags (:func:`add_common_args`) and resolve
them the same way (:func:`resolve_conn`), always in this order::

    CLI flag  >  ARKESTRA_* env  >  config.yaml  >  default

Config-location env (``ARKESTRA_CONFIG`` / ``ARKESTRA_DIR``) lives in
:mod:`model_arkestra.common` so the core library and the CLI share one resolver;
the connection-side env (host / port / api-key / base-path) is defined here.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from model_arkestra.common import resolve_config_path, ENV_CONFIG, ENV_DIR

# ── Shared env-var names (connection side) ─────────────────────────────
ENV_HOST = "ARKESTRA_HOST"
ENV_PORT = "ARKESTRA_PORT"
ENV_API_KEY = "ARKESTRA_API_KEY"
ENV_BASE_PATH = "ARKESTRA_BASE_PATH"

DEFAULT_PORT = 8080
SERVER_HOST_DEFAULT = "0.0.0.0"    # server: bind all interfaces
CLIENT_HOST_DEFAULT = "127.0.0.1"  # client: target localhost


@dataclass(frozen=True)
class Conn:
    """Resolved connection coordinates shared by both entry points."""
    host: str
    port: int
    config_path: Path
    base_path: str = ""                 # "" or "/prefix"
    api_key: Optional[str] = None

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, route: str) -> str:
        """Absolute URL for a route path (e.g. ``"/v1/chat/completions"``)."""
        return f"{self.origin}{self.base_path}{route}"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared connection/config flags on *parser*."""
    parser.add_argument("--config", "-c", default=None,
                        help=f"Path to config.yaml (env: {ENV_CONFIG} / {ENV_DIR})")
    parser.add_argument("--host", "-H", default=None,
                        help=f"Host (env: {ENV_HOST})")
    parser.add_argument("--port", "-p", type=int, default=None,
                        help=f"Port (env: {ENV_PORT})")
    parser.add_argument("--api-key", default=None,
                        help=f"API key (env: {ENV_API_KEY})")


def _env_int(name: str) -> Optional[int]:
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def resolve_conn(
    args: Any,
    *,
    server: bool,
    cfg_get: Optional[Callable[..., Any]] = None,
) -> Conn:
    """Apply ``CLI > env > config > default`` to the shared surface.

    *args* is the parsed :class:`argparse.Namespace`. *cfg_get* is an optional
    ``getter(path, default)`` for consulting config.yaml (e.g.
    ``default/admin-port``); the entry point supplies it once config is loaded.
    """
    host = (getattr(args, "host", None)
            or os.environ.get(ENV_HOST)
            or (SERVER_HOST_DEFAULT if server else CLIENT_HOST_DEFAULT))

    port = getattr(args, "port", None) or _env_int(ENV_PORT)
    if port is None:
        port = cfg_get("default/admin-port", DEFAULT_PORT) if cfg_get else DEFAULT_PORT

    base_path = os.environ.get(ENV_BASE_PATH, "") or ""
    api_key = getattr(args, "api_key", None) or os.environ.get(ENV_API_KEY)
    config_path = resolve_config_path(getattr(args, "config", None))

    return Conn(host=host, port=int(port), config_path=config_path,
                base_path=base_path, api_key=api_key)
