"""Minimal decoder-stack stand-in for hook smoke tests without HF weights."""

from __future__ import annotations

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin(x)


class ToyCausalLM(nn.Module):
    """Mimics ``model.model.layers`` path used by Llama/Qwen HF models."""

    def __init__(self, n_layers: int = 8, hidden: int = 32, vocab: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(hidden) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        del attention_mask, use_cache
        x = self.embed(input_ids)
        for block in self.model.layers:
            x = block(x)
        return self.lm_head(x)
