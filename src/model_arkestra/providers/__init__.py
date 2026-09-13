"""Inference providers — one per model, chosen by backend/runner type.

A provider answers "who provides inference for this model?" and is agnostic to
direction and transport:

    LlamaProvider   local llama.cpp, reached via its HTTP API
    OnnxProvider    in-process ONNX session (no transport)
    RemoteProvider  proxied to a cluster worker (engine-agnostic)

Runners never touch these; they only launch/stop/watch the underlying process
or container. ``Model.provider`` returns the active provider for a model.
"""
from model_arkestra.providers.base import Provider, NotSupported
from model_arkestra.providers.llama import LlamaProvider
from model_arkestra.providers.onnx import OnnxProvider
from model_arkestra.providers.remote import RemoteProvider

__all__ = ["Provider", "NotSupported", "LlamaProvider", "OnnxProvider", "RemoteProvider"]
