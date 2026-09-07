"""Load HF CausalLMs with device_map / dtype from the memory plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
from .memory import recommend_plan
from .registry import ModelSpec, get_model_spec


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    tokenizer: PreTrainedTokenizerBase
    device_map: str | dict[str, Any]
    dtype: torch.dtype


_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    model_key: str | None = None,
    *,
    device_map: str | dict[str, Any] | None = None,
    dtype: str | None = None,
    max_memory: dict[str | int, str] | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> LoadedModel:
    """Load model + tokenizer according to registry + memory plan defaults.

    On MacBook (no CUDA), omit ``model_key`` to get ``qwen3-0.6b`` on MPS.
    """
    key = model_key or LOCAL_MODEL_KEY
    assert_model_fits_machine(key)
    spec = get_model_spec(key)
    plan = recommend_plan(key)

    if isinstance(device_map, str) or device_map is None:
        resolved_map: str | dict[str, Any] = resolve_device_map(
            device_map if isinstance(device_map, str) else None
        )
        # Honor plan default for local model when caller left device unset and plan says mps/cpu
        if device_map is None and key == LOCAL_MODEL_KEY:
            resolved_map = resolve_device_map(None)
    else:
        resolved_map = device_map

    resolved_dtype_name = dtype or plan.param_dtype
    torch_dtype = _DTYPE[resolved_dtype_name]
    resolved_max_memory = max_memory if max_memory is not None else None
    if (
        resolved_max_memory is None
        and resolved_map == "auto"
        and torch.cuda.is_available()
    ):
        resolved_max_memory = plan.max_memory_example

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # accelerate device_map does not support "mps"; load then .to("mps").
    use_mps = resolved_map == "mps"
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "dtype": torch_dtype,
    }
    if use_mps or resolved_map == "cpu":
        model_kwargs["device_map"] = "cpu"
    else:
        model_kwargs["device_map"] = resolved_map
        if resolved_max_memory is not None:
            model_kwargs["max_memory"] = resolved_max_memory

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)
    if use_mps:
        model = model.to("mps")
    model.eval()

    return LoadedModel(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device_map=resolved_map,
        dtype=torch_dtype,
    )
