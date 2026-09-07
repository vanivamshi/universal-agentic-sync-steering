"""Forward hooks that capture residual-stream activations at selected layers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


def resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder block list for Qwen2/3 and Llama HF CausalLMs.

    Path: ``model.model.layers`` (HF naming for Llama/Qwen2/Qwen3).
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        layers = inner.layers
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    # Some wrappers expose .transformer.layers / .layers directly
    for path in ("transformer.layers", "layers"):
        cur: nn.Module | None = model
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if isinstance(cur, nn.ModuleList) and len(cur) > 0:
            return cur
    raise AttributeError(
        "Could not resolve decoder layers. Expected model.model.layers "
        "(Llama / Qwen2 / Qwen3 HF CausalLM)."
    )


def _unwrap_hidden(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"Layer output is not a tensor: {type(output)}")
    return hidden


class ResidualStreamHooks:
    """Register forward hooks on selected decoder layers; store residual stream.

    Captures the **post-block** residual stream (module forward output), shape
    ``(batch, seq, hidden)``, detached and optionally cast.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        *,
        cast_dtype: torch.dtype | None = torch.bfloat16,
        store_cpu: bool = True,
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.n_layers = len(self.layers)
        self.layer_indices = sorted(set(layer_indices))
        self.cast_dtype = cast_dtype
        self.store_cpu = store_cpu
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for idx in self.layer_indices:
            if idx < 0 or idx >= self.n_layers:
                raise IndexError(
                    f"layer {idx} out of range for model with {self.n_layers} layers"
                )

    def _make_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
            hidden = _unwrap_hidden(output).detach()
            if self.cast_dtype is not None and hidden.dtype != self.cast_dtype:
                hidden = hidden.to(dtype=self.cast_dtype)
            if self.store_cpu:
                hidden = hidden.cpu()
            self.activations[layer_idx] = hidden

        return hook

    def clear(self) -> None:
        self.activations.clear()

    def register(self) -> None:
        self.remove()
        for idx in self.layer_indices:
            handle = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextmanager
    def capture(self) -> Iterator["ResidualStreamHooks"]:
        self.clear()
        self.register()
        try:
            yield self
        finally:
            self.remove()
