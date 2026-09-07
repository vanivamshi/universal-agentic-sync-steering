"""Local device / model defaults for Apple Silicon (no CUDA)."""

from __future__ import annotations

import torch

# MacBook / Apple Silicon default — do not use qwen3-32b or llama-70b here.
LOCAL_MODEL_KEY = "qwen3-0.6b"
CUDA_ONLY_MODELS = frozenset({"qwen3-32b", "llama-3.1-70b"})


def resolve_device_map(explicit: str | None = None) -> str:
    """Prefer MPS on Mac; never invent CUDA when unavailable."""
    if explicit:
        if explicit == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        if explicit == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def assert_model_fits_machine(model_key: str) -> None:
    """Refuse CUDA-only giants on this MacBook unless CUDA is actually present."""
    if model_key in CUDA_ONLY_MODELS and not torch.cuda.is_available():
        raise RuntimeError(
            f"Refusing {model_key!r} on this machine (no CUDA). "
            f"Use {LOCAL_MODEL_KEY!r} (Qwen3-0.6B on MPS) for local runs."
        )
