"""Model registry aligned with preregistration.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Open-weight model used for activation geometry work."""

    key: str
    hf_id: str
    role: str  # primary | replication | smoke
    n_layers: int
    hidden_size: int
    default_perturb_layer: int  # k
    default_measure_layer: int  # L
    architecture: str  # qwen2 | llama (layer path: model.model.layers)
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen3-32b": ModelSpec(
        key="qwen3-32b",
        hf_id="Qwen/Qwen3-32B",
        role="primary",
        n_layers=64,
        hidden_size=5120,
        default_perturb_layer=8,
        default_measure_layer=48,
        architecture="qwen2",
        notes="Primary model per preregistration. Hermes-style <tool_call> format.",
    ),
    "llama-3.1-70b": ModelSpec(
        key="llama-3.1-70b",
        hf_id="meta-llama/Llama-3.1-70B-Instruct",
        role="replication",
        n_layers=80,
        hidden_size=8192,
        default_perturb_layer=10,
        default_measure_layer=60,
        architecture="llama",
        notes="Cross-model replication (§10). Collect after Qwen RQ1 story.",
    ),
    # Local MacBook / Apple Silicon — not for published RQ1 numbers.
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        hf_id="Qwen/Qwen3-0.6B",
        role="local",
        n_layers=28,
        hidden_size=1024,
        default_perturb_layer=4,
        default_measure_layer=22,
        architecture="qwen3",
        notes=(
            "Default on MacBook (MPS, no CUDA). Same Qwen3 family + Hermes "
            "<tool_call> path as primary. Swap to qwen3-32b only on a CUDA host."
        ),
    ),
    "qwen2.5-0.5b-smoke": ModelSpec(
        key="qwen2.5-0.5b-smoke",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        role="smoke",
        n_layers=24,
        hidden_size=896,
        default_perturb_layer=4,
        default_measure_layer=20,
        architecture="qwen2",
        notes="Legacy; prefer qwen3-0.6b on Mac.",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as e:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model key {key!r}. Known: {known}") from e
