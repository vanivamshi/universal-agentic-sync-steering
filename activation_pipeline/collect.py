"""Batched residual-stream collection via forward hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from .hooks import ResidualStreamHooks, resolve_decoder_layers
from .loader import LoadedModel
from .memory import recommend_plan
from .registry import ModelSpec


@dataclass
class ActivationBatch:
    """Activations for one batched forward pass."""

    model_key: str
    hf_id: str
    layer_indices: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    activations: dict[int, torch.Tensor]  # layer -> (batch, seq, hidden)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


def default_layer_indices(spec: ModelSpec, extra: Sequence[int] | None = None) -> list[int]:
    layers = {spec.default_perturb_layer, spec.default_measure_layer}
    if extra:
        layers.update(extra)
    return sorted(layers)


@torch.inference_mode()
def collect_residual_stream(
    loaded: LoadedModel,
    texts: Sequence[str],
    *,
    layer_indices: Sequence[int] | None = None,
    batch_size: int | None = None,
    max_length: int = 2048,
    cast_dtype: torch.dtype | None = torch.bfloat16,
    store_cpu: bool = True,
) -> list[ActivationBatch]:
    """Tokenize ``texts``, run forward in batches, return hooked residual streams.

    Only selected layers are hooked (memory-safe vs ``output_hidden_states=True``
    for all layers).
    """
    if batch_size is None:
        try:
            bs = recommend_plan(loaded.spec.key).activation_batch_size
        except KeyError:
            bs = 1
    else:
        bs = batch_size
    indices = (
        list(layer_indices)
        if layer_indices is not None
        else default_layer_indices(loaded.spec)
    )

    # Validate against live module count (may differ from registry if revision changes).
    n_live = len(resolve_decoder_layers(loaded.model))
    for idx in indices:
        if idx < 0 or idx >= n_live:
            raise IndexError(f"layer {idx} out of range (model has {n_live} layers)")

    tokenizer: PreTrainedTokenizerBase = loaded.tokenizer
    results: list[ActivationBatch] = []

    for start in range(0, len(texts), bs):
        chunk = list(texts[start : start + bs])
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # Place inputs on the device of the first parameter (works with device_map).
        first_param = next(loaded.model.parameters())
        encoded = {k: v.to(first_param.device) for k, v in encoded.items()}

        hooks = ResidualStreamHooks(
            loaded.model,
            indices,
            cast_dtype=cast_dtype,
            store_cpu=store_cpu,
        )
        with hooks.capture():
            _ = loaded.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=False,
            )

        missing = [i for i in indices if i not in hooks.activations]
        if missing:
            raise RuntimeError(f"hooks did not fire for layers {missing}")

        results.append(
            ActivationBatch(
                model_key=loaded.spec.key,
                hf_id=loaded.spec.hf_id,
                layer_indices=indices,
                input_ids=encoded["input_ids"].detach().cpu(),
                attention_mask=encoded["attention_mask"].detach().cpu(),
                activations=dict(hooks.activations),
                meta={
                    "batch_start": start,
                    "texts": chunk,
                    "n_layers_live": n_live,
                    "dtype": str(cast_dtype),
                },
            )
        )

    return results
