"""Residual-stream activation collection for Qwen3 / Llama-family models."""

from .device import CUDA_ONLY_MODELS, LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
from .cache_io import ActivationCache, WindowActivationRecord
from .caching import cache_transcript_windows, load_transcripts
from .collect import ActivationBatch, collect_residual_stream
from .hooks import ResidualStreamHooks, resolve_decoder_layers
from .loader import LoadedModel, load_model_and_tokenizer
from .memory import MEMORY_PLANS, estimate_param_bytes, recommend_plan
from .registry import MODEL_REGISTRY, ModelSpec, get_model_spec
from .windows import TaggedAssistantTurn, TokenSpan, tag_transcript_assistant_turns

__all__ = [
    "ActivationBatch",
    "ActivationCache",
    "CUDA_ONLY_MODELS",
    "LOCAL_MODEL_KEY",
    "LoadedModel",
    "MEMORY_PLANS",
    "MODEL_REGISTRY",
    "ModelSpec",
    "ResidualStreamHooks",
    "TaggedAssistantTurn",
    "TokenSpan",
    "WindowActivationRecord",
    "assert_model_fits_machine",
    "cache_transcript_windows",
    "collect_residual_stream",
    "estimate_param_bytes",
    "get_model_spec",
    "load_model_and_tokenizer",
    "load_transcripts",
    "recommend_plan",
    "resolve_decoder_layers",
    "resolve_device_map",
    "tag_transcript_assistant_turns",
]

__version__ = "0.6.0"
