"""Shared connection surface for ``arkestra`` (client), ``arkestra-admin``, and
``arkestra-server``.

All three entry points register the same flags (:func:`add_common_args`) and
resolve them the same way (:func:`resolve_conn`), always in this order::

    CLI flag  >  ARKESTRA_* env  >  config.yaml  >  default

The connection is expressed as a single **public URL** (``scheme://host:port/prefix``)
that carries the scheme, host, port, and any URL path prefix.  The server *bind*
address is a separate input because it can legitimately differ from the dialable
target (e.g. bind ``0.0.0.0`` for LAN exposure while clients dial the hostname).

Config-location env (``ARKESTRA_CONFIG`` / ``ARKESTRA_DIR``) lives in
:mod:`model_arkestra.common` so the core library and the CLI share one resolver;
the connection-side env (url / api-key) is defined here.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from model_arkestra.common import resolve_config_path, ENV_CONFIG, ENV_DIR

# ── Shared env-var names (connection side) ─────────────────────────────
ENV_URL = "ARKESTRA_URL"
ENV_API_KEY = "ARKESTRA_API_KEY"

DEFAULT_PORT = 8080
DEFAULT_SCHEME = "http"
DEFAULT_TARGET_HOST = "127.0.0.1"   # dialable default for clients
BIND_DEFAULT = "127.0.0.1"          # server bind default (loopback-only)


def parse_url(url: str, *, default_port: int = DEFAULT_PORT):
    """Parse a public URL into ``(scheme, host, port, base_path)``.

    *base_path* is the path component with no trailing slash ("" when absent).
    A missing scheme defaults to ``http``; a missing port defaults to
    *default_port*.  Hosts are lower-cased for consistency.
    """
    if not url:
        return DEFAULT_SCHEME, DEFAULT_TARGET_HOST, default_port, ""
    # Tolerate a bare "host:port" or "host" with no scheme.
    candidate = url if "://" in url else f"{DEFAULT_SCHEME}://{url}"
    parsed = urlparse(candidate)
    scheme = (parsed.scheme or DEFAULT_SCHEME).lower()
    host = (parsed.hostname or DEFAULT_TARGET_HOST).lower()
    port = parsed.port or default_port
    path = (parsed.path or "").rstrip("/")
    return scheme, host, port, path


@dataclass(frozen=True)
class Conn:
    """Resolved connection coordinates shared by all entry points.

    *host/port/base_path/scheme* describe the **public target** (dialable).
    *bind_host* is the server bind address (meaningful only when built with
    ``server=True``); it may differ from *host* in the LAN-exposure posture.
    """
    scheme: str
    host: str
    port: int
    base_path: str = ""                 # "" or "/prefix"
    config_path: Path = Path()
    api_key: Optional[str] = None
    bind_host: str = BIND_DEFAULT

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        """The full public URL including any prefix (no trailing slash)."""
        return f"{self.origin}{self.base_path}"

    def url_for(self, route: str) -> str:
        """Absolute URL for a route path (e.g. ``"/v1/chat/completions"``)."""
        if not route.startswith("/"):
            route = "/" + route
        return f"{self.origin}{self.base_path}{route}"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared connection/config flags on *parser*.

    The server additionally registers ``--bind`` (see :func:`add_server_args`).
    """
    parser.add_argument("--config", "-c", default=None,
                        help=f"Path to config.yaml (env: {ENV_CONFIG} / {ENV_DIR})")
    parser.add_argument("--url", default=None,
                        help=(f"Public server URL scheme://host:port/prefix "
                              f"(env: {ENV_URL}; default http://{DEFAULT_TARGET_HOST}:{DEFAULT_PORT})"))
    parser.add_argument("--api-key", default=None,
                        help=f"API key (env: {ENV_API_KEY})")


def add_server_args(parser: argparse.ArgumentParser) -> None:
    """Register server-only flags (bind address)."""
    parser.add_argument("--bind", default=None,
                        help="Address the server binds to — 127.0.0.1 for local "
                             "only, 0.0.0.0 to expose on the LAN (default: 127.0.0.1)")


def resolve_conn(
    args: Any,
    *,
    server: bool,
    cfg_get: Optional[Callable[..., Any]] = None,
) -> Conn:
    """Apply ``CLI > env > config > default`` to the shared surface.

    *args* is the parsed :class:`argparse.Namespace`.  *cfg_get* is an optional
    ``getter(path, default)`` for consulting config.yaml (e.g. ``default/url``);
    the entry point supplies it once config is loaded.

    The public URL is resolved from ``--url`` > ``ARKESTRA_URL`` >
    ``config default/url`` > default.  When *server* is True, the bind address is
    also resolved from ``--bind`` > ``config default/bind`` > ``127.0.0.1``.
    """
    raw_url = (getattr(args, "url", None)
               or os.environ.get(ENV_URL)
               or (cfg_get("default/url") if cfg_get else None)
               or f"{DEFAULT_SCHEME}://{DEFAULT_TARGET_HOST}:{DEFAULT_PORT}")
    scheme, host, port, base_path = parse_url(str(raw_url))

    api_key = getattr(args, "api_key", None) or os.environ.get(ENV_API_KEY)
    config_path = resolve_config_path(getattr(args, "config", None))

    bind_host = BIND_DEFAULT
    if server:
        bind_host = (getattr(args, "bind", None)
                     or (cfg_get("default/bind") if cfg_get else None)
                     or BIND_DEFAULT)

    return Conn(scheme=scheme, host=host, port=int(port), config_path=config_path,
                base_path=base_path, api_key=api_key, bind_host=str(bind_host))
