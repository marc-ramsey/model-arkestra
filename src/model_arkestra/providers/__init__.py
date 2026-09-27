"""Inference providers — one per model, chosen by engine.

A provider answers "who provides inference for this model?" and is agnostic to
direction and transport:

    LlamaProvider   local llama.cpp, reached via its HTTP API
    SdcppProvider   local stable-diffusion.cpp, reached via its HTTP API
    OnnxProvider    in-process ONNX session (no transport)
    RemoteProvider  proxied to a cluster worker (engine-agnostic)

PROVIDERS maps an engine name (backends.<id>.engine) to a provider class.
A container-run llama-cpp model is still a LlamaProvider — the provider
follows the engine, not the runner. New engine = one module + one entry.
"""
from model_arkestra.providers.base import Provider, NotSupported
from model_arkestra.providers.llama import LlamaProvider
from model_arkestra.providers.onnx import OnnxProvider
from model_arkestra.providers.remote import RemoteProvider
from model_arkestra.providers.sdcpp import SdcppProvider

PROVIDERS = {
    "llama-cpp": LlamaProvider,
    "sdcpp": SdcppProvider,
}

__all__ = [
    "Provider", "NotSupported", "PROVIDERS",
    "LlamaProvider", "OnnxProvider", "RemoteProvider", "SdcppProvider",
]
