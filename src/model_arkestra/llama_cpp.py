"""Llama.cpp engine — arg sets and CLI building for llama-server / llama serve."""
from __future__ import annotations

from typing import Any, Dict, List


class LlamaCppEngine:
    """Handles llama.cpp-specific argument filtering and CLI conversion."""

    # Inference params users can safely override via POST body.
    # Anything not in this set is silently dropped to prevent CLI crashes.
    LLAMA_INFER_ARGS: set[str] = {
        'temp', 'top-k', 'top-p', 'min-p', 'typical', 'top-nsigma',
        'frequency-penalty', 'presence-penalty', 'repeat-penalty', 'repeat-last-n',
        'rope-freq-base', 'rope-freq-scale', 'rope-scaling', 'rope-scale',
        'reasoning-budget', 'keep', 'ignore-eos', 'grammar', 'chat-template',
    }

    # Valid llama.cpp args that are meaningful at config/model level but
    # should never appear in POST inference overrides.  Kept for reference
    # and potential config validation; the engine uses LLAMA_INFER_ARGS as
    # the sole filter for inference kwargs.
    LLAMA_CONFIG_ARGS: set[str] = {
        # Server infra (managed by Arkestra)
        'port', 'host', 'sleep-idle-seconds', 'reuse-port',
        'threads-http', 'prio', 'poll', 'poll-batch',
        # Speculative decoding
        'spec-default', 'spec-type',
        'spec-draft-backend-sampling', 'spec-draft-cpu-mask', 'spec-draft-cpu-mask-batch',
        'spec-draft-cpu-moe', 'spec-draft-cpu-range', 'spec-draft-cpu-strict',
        'spec-draft-cpu-strict-batch', 'spec-draft-device', 'spec-draft-hf',
        'spec-draft-model', 'spec-draft-n-cpu-moe', 'spec-draft-ngl', 'spec-draft-n-max',
        'spec-draft-n-min', 'spec-draft-override-tensor', 'spec-draft-p-min',
        'spec-draft-p-split', 'spec-draft-poll', 'spec-draft-poll-batch',
        'spec-draft-prio', 'spec-draft-prio-batch', 'spec-draft-threads',
        'spec-draft-threads-batch', 'spec-draft-type-k', 'spec-draft-type-v',
        'spec-ngram-map-k-min-hits', 'spec-ngram-map-k-size-m', 'spec-ngram-map-k-size-n',
        'spec-ngram-map-k4v-min-hits', 'spec-ngram-map-k4v-size-m', 'spec-ngram-map-k4v-size-n',
        'spec-ngram-min-hits', 'spec-ngram-mod-n-match', 'spec-ngram-mod-n-max',
        'spec-ngram-mod-n-min', 'spec-ngram-simple-min-hits', 'spec-ngram-simple-size-m',
        'spec-ngram-simple-size-n', 'spec-ngram-size-m', 'spec-ngram-size-n',
        # Model downloaders (CLI presets — never inference params)
        'embd-gemma-default', 'fim-qwen-1.5b-default', 'fim-qwen-14b-spec',
        'fim-qwen-30b-default', 'fim-qwen-3b-default', 'fim-qwen-7b-default',
        'fim-qwen-7b-spec', 'gpt-oss-120b-default', 'gpt-oss-20b-default',
        'vision-gemma-12b-default', 'vision-gemma-4b-default',
        # Server lifecycle / cache
        'list-devices', 'check-tensors', 'completion-bash', 'version', 'offline',
        'cache-idle-slots', 'no-cache-idle-slots', 'cache-prompt', 'no-cache-prompt',
        'cache-reuse', 'warmup', 'no-warmup', 'adaptive-decay', 'adaptive-target',
        'swa-full', 'context-shift', 'no-context-shift',
        # Hardware
        'mmap', 'repack', 'op-offload', 'numa', 'cpu-strict', 'cpu-strict-batch', 'no-host',
        # LoRA / control vectors (rare)
        'lora', 'lora-init-without-apply', 'lora-scaled', 'control-vector',
        'control-vector-layer-range', 'control-vector-scaled',
        # Chat template files / mmproj
        'chat-template-file', 'mmproj-auto', 'no-mmproj', 'mmproj-offload',
        # Router/completion / multimedia
        'models-autoload', 'models-dir', 'models-max', 'models-preset',
        'mtmd-batch-max-tokens', 'pooling', 'prefill-assistant', 'skip-chat-parsing',
        'media-path', 'image-max-tokens', 'image-min-tokens',
        # Niche samplers (alt to temp/top-p)
        'mirostat', 'mirostat-ent', 'mirostat-lr', 'dynatemp-exp', 'dynatemp-range',
        'xtc-probability', 'xtc-threshold', 'samplers', 'sampler-seq',
        # DRY / YARN
        'dry-allowed-length', 'dry-base', 'dry-multiplier', 'dry-penalty-last-n',
        'dry-sequence-breaker', 'yarn-attn-factor', 'yarn-beta-fast',
        'yarn-beta-slow', 'yarn-ext-factor', 'yarn-orig-ctx',
        # Server UI / networking / logging (infra-level)
        'api-key', 'api-key-file', 'api-prefix', 'cors-credentials', 'cors-headers',
        'cors-methods', 'cors-origins', 'ssl-cert-file', 'ssl-key-file',
        'sse-ping-interval', 'ui', 'webui', 'ui-config', 'ui-config-file',
        'ui-mcp-proxy', 'mcp-servers-config', 'mcp-servers-json', 'tools',
        'metrics', 'props', 'rerank', 'slots',
        'log-colors', 'log-disable', 'log-file', 'log-prefix', 'log-timestamps',
        'log-prompts-dir', 'path', 'perf', 'prio-batch',
        # Advanced/niche
        'override-kv', 'reasoning-budget-message', 'reasoning-format',
        'reasoning-preserve', 'no-reasoning-preserve',
    }

    def filter_infer_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return only inference-safe keys from arbitrary kwargs."""
        return {k: v for k, v in kwargs.items() if k in self.LLAMA_INFER_ARGS}

    def build_args(
        self, config_args: Dict[str, Any], inference_kwargs: Dict[str, Any]
    ) -> List[str]:
        """Build CLI argument list from config args + filtered inference kwargs.

        *config_args* comes from the model's YAML definition (backend args, defaults,
        model args) — all of these pass through unchanged.

        *inference_kwargs* come from POST body overrides or runtime inference params.
        Only keys in ``LLAMA_INFER_ARGS`` are included; everything else is silently
        dropped to prevent the subprocess from crashing on bogus CLI flags.
        """
        cli: List[str] = []

        # Merge: config args first, then inference kwargs (last-wins)
        merged: Dict[str, Any] = dict(config_args)
        for key, value in inference_kwargs.items():
            if key not in self.LLAMA_INFER_ARGS:
                continue  # drop unknown inference params
            merged[key] = value

        # Convert to CLI flags
        for key, value in merged.items():
            if key == "hf":
                cli.extend(["-hf", str(value)])
            elif isinstance(value, bool):
                flag = f"--{key.replace('_', '-')}"
                if value:
                    cli.append(flag)
            else:
                flag = f"--{key.replace('_', '-')}"
                cli.extend([flag, str(value)])

        return cli
