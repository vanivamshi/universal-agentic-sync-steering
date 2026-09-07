"""Device / VRAM plans for loading primary and replication models."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import MODEL_REGISTRY, ModelSpec, get_model_spec


@dataclass(frozen=True)
class MemoryPlan:
    """How to place a model for residual-stream collection."""

    model_key: str
    param_dtype: str
    approx_param_gib: float
    recommended_gpus: str
    device_map: str
    max_memory_example: dict[str | int, str]
    activation_batch_size: int
    activation_dtype: str
    layers_strategy: str
    notes: str


def estimate_param_bytes(spec: ModelSpec, dtype: str = "bfloat16") -> int:
    """Rough parameter footprint from hidden size × layers (order-of-magnitude).

    Prefer documented HF sizes when available; this is for planning only.
    """
    bytes_per = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1, "int4": 0.5}[dtype]
    # Dense transformer rough: ~12 * n_layers * hidden^2 (attn+mlp+norms+embed scale).
    # Calibrated constants from public model cards when possible.
    known = {
        ("qwen3-32b", "bfloat16"): 64.0 * (1 << 30),
        ("llama-3.1-70b", "bfloat16"): 140.0 * (1 << 30),
        ("qwen3-0.6b", "bfloat16"): 1.2 * (1 << 30),
        ("qwen2.5-0.5b-smoke", "bfloat16"): 1.0 * (1 << 30),
    }
    if (spec.key, dtype) in known:
        return int(known[(spec.key, dtype)])
    n = 12.0 * spec.n_layers * (spec.hidden_size**2)
    return int(n * bytes_per)


MEMORY_PLANS: dict[str, MemoryPlan] = {
    "qwen3-32b": MemoryPlan(
        model_key="qwen3-32b",
        param_dtype="bfloat16",
        approx_param_gib=64.0,
        recommended_gpus="≥2×80GB (A100/H100) or ≥4×48GB with offload",
        device_map="auto",
        max_memory_example={0: "75GiB", 1: "75GiB", "cpu": "64GiB"},
        activation_batch_size=1,
        activation_dtype="bfloat16",
        layers_strategy=(
            "Hook only selected layers (default {k=8, L=48} plus optional intermediates). "
            "Do not keep full hidden_states for all 64 layers unless needed for §9 sweeps."
        ),
        notes=(
            "This workstation has no NVIDIA GPU and ~8GB RAM — do not load Qwen3-32B here. "
            "Use a multi-GPU box or cloud. Prefer attn implementation sdpa/flash_attention_2. "
            "For long agentic contexts, batch_size=1 and truncate/pad carefully."
        ),
    ),
    "llama-3.1-70b": MemoryPlan(
        model_key="llama-3.1-70b",
        param_dtype="bfloat16",
        approx_param_gib=140.0,
        recommended_gpus="≥2×80GB tightly, preferably 4×80GB; or quantized 4-bit for bring-up only",
        device_map="auto",
        max_memory_example={0: "75GiB", 1: "75GiB", 2: "75GiB", 3: "75GiB", "cpu": "128GiB"},
        activation_batch_size=1,
        activation_dtype="bfloat16",
        layers_strategy="Hook {k=10, L=60} by default; expand for §9.",
        notes="Replication only after Qwen primary results. HF gated — set HF_TOKEN.",
    ),
    "qwen3-0.6b": MemoryPlan(
        model_key="qwen3-0.6b",
        param_dtype="float32",
        approx_param_gib=1.2,
        recommended_gpus="Apple Silicon MPS (MacBook default); CPU fallback",
        device_map="mps",
        max_memory_example={"mps": "8GiB", "cpu": "6GiB"},
        activation_batch_size=1,
        activation_dtype="float32",
        layers_strategy="Hook default {k=4, L=22}; logit-lens uses final layer.",
        notes=(
            "MacBook default model. Loader maps device_map='mps' via CPU load then .to('mps'). "
            "Do not attempt qwen3-32b / llama-3.1-70b without CUDA."
        ),
    ),
    "qwen2.5-0.5b-smoke": MemoryPlan(
        model_key="qwen2.5-0.5b-smoke",
        param_dtype="float32",
        approx_param_gib=1.0,
        recommended_gpus="CPU ok for short sequences; any single GPU",
        device_map="cpu",
        max_memory_example={"cpu": "4GiB"},
        activation_batch_size=2,
        activation_dtype="float32",
        layers_strategy="Hook a small layer set for smoke tests.",
        notes="Pipeline validation only. Not for publishing RQ1 numbers.",
    ),
}


def recommend_plan(model_key: str) -> MemoryPlan:
    key = get_model_spec(model_key).key
    try:
        return MEMORY_PLANS[key]
    except KeyError as e:
        raise KeyError(f"No memory plan for {model_key!r}") from e


def format_plan_report(model_key: str | None = None) -> str:
    keys = [model_key] if model_key else list(MEMORY_PLANS)
    lines: list[str] = []
    for key in keys:
        plan = recommend_plan(key)
        spec = MODEL_REGISTRY[plan.model_key]
        lines.append(f"## {plan.model_key} ({spec.hf_id})")
        lines.append(f"- role: {spec.role}")
        lines.append(f"- layers: {spec.n_layers}, hidden: {spec.hidden_size}")
        lines.append(f"- default (k, L): ({spec.default_perturb_layer}, {spec.default_measure_layer})")
        lines.append(f"- param dtype: {plan.param_dtype} (~{plan.approx_param_gib:.0f} GiB)")
        lines.append(f"- GPUs: {plan.recommended_gpus}")
        lines.append(f"- device_map: {plan.device_map}")
        lines.append(f"- max_memory example: {plan.max_memory_example}")
        lines.append(f"- activation batch_size: {plan.activation_batch_size}")
        lines.append(f"- activation dtype: {plan.activation_dtype}")
        lines.append(f"- layers: {plan.layers_strategy}")
        lines.append(f"- notes: {plan.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
