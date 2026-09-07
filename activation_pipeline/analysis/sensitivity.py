"""Directional sensitivity, plateau depth, top-k rank (plateau-literature metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from ..hooks import resolve_decoder_layers


def _as_unit(v: torch.Tensor) -> torch.Tensor:
    v = v.float().reshape(-1)
    n = torch.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero direction")
    return v / n


@dataclass
class SensitivityResult:
    direction_id: str
    epsilon_star: float
    blowup: float
    threshold: float
    layer_k: int
    layer_L: int
    converged: bool


def residual_l2_blowup(
    clean_L: torch.Tensor,
    pert_L: torch.Tensor,
    *,
    relative: bool = True,
    denom: float | None = None,
) -> float:
    """L2 distance at layer L; optional relative to ||clean|| or a fixed denom."""
    diff = (pert_L.float() - clean_L.float()).reshape(-1)
    blow = float(torch.linalg.norm(diff))
    if denom is not None:
        return blow / (float(denom) + 1e-8)
    if relative:
        base = float(torch.linalg.norm(clean_L.float().reshape(-1))) + 1e-8
        return blow / base
    return blow


def directional_sensitivity(
    forward_clean: Callable[[], dict[int, torch.Tensor]],
    forward_pert: Callable[[float], dict[int, torch.Tensor]],
    *,
    layer_k: int,
    layer_L: int,
    threshold: float = 0.5,
    eps_lo: float = 1e-4,
    eps_hi: float = 50.0,
    max_iter: int = 24,
    direction_id: str = "dir",
    pool: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> SensitivityResult:
    """Binary-search smallest ε along a direction that exceeds blowup threshold at L.

    ``forward_clean()`` → {layer: activation tensor for the measured window}.
    ``forward_pert(eps)`` → same after residual inject ε·d at layer k.
    Primary metric per preregistration (Heimersheim-style directional sensitivity).
    """
    pool_fn = pool or (lambda t: t.float().mean(dim=0) if t.dim() > 1 else t.float())

    clean = forward_clean()
    clean_L = pool_fn(clean[layer_L])

    def blow(eps: float) -> float:
        pert = forward_pert(eps)
        return residual_l2_blowup(clean_L, pool_fn(pert[layer_L]), relative=True)

    # Expand hi until threshold crossed or cap
    hi = eps_hi
    b_hi = blow(hi)
    if b_hi < threshold:
        return SensitivityResult(
            direction_id=direction_id,
            epsilon_star=hi,
            blowup=b_hi,
            threshold=threshold,
            layer_k=layer_k,
            layer_L=layer_L,
            converged=False,
        )

    lo = eps_lo
    b_lo = blow(lo)
    if b_lo >= threshold:
        return SensitivityResult(
            direction_id=direction_id,
            epsilon_star=lo,
            blowup=b_lo,
            threshold=threshold,
            layer_k=layer_k,
            layer_L=layer_L,
            converged=True,
        )

    best_eps = hi
    best_blow = b_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        b_mid = blow(mid)
        if b_mid >= threshold:
            hi = mid
            best_eps = mid
            best_blow = b_mid
        else:
            lo = mid
    return SensitivityResult(
        direction_id=direction_id,
        epsilon_star=best_eps,
        blowup=best_blow,
        threshold=threshold,
        layer_k=layer_k,
        layer_L=layer_L,
        converged=True,
    )


def default_epsilon_grid(
    *,
    eps_hi: float = 40.0,
    n: int = 20,
) -> list[float]:
    """Log-spaced ε grid for Heimersheim-style blowup-vs-ε curves (includes 0)."""
    if n < 2:
        return [0.0, float(eps_hi)]
    # denser near small ε where plateaus live
    import math

    lo = 1e-3
    log_lo, log_hi = math.log10(lo), math.log10(eps_hi)
    mid = [
        10 ** (log_lo + (log_hi - log_lo) * i / (n - 1))
        for i in range(n)
    ]
    return [0.0] + mid


def blowup_vs_epsilon_curve(
    *,
    clean_L: torch.Tensor,
    forward_pert: Callable[[float], dict[int, torch.Tensor]],
    layer_L: int,
    pool: Callable[[torch.Tensor], torch.Tensor],
    eps_grid: Sequence[float] | None = None,
    denom: float | None = None,
) -> dict[str, list[float]]:
    """Sweep ε and record absolute + relative L2 blowup at layer L.

    Heimersheim-style activation graph: late-layer residual change vs
    perturbation magnitude. Relative uses ||clean_L|| (or fixed ``denom``).
    """
    grid = list(eps_grid) if eps_grid is not None else default_epsilon_grid()
    abs_vals: list[float] = []
    rel_vals: list[float] = []
    for eps in grid:
        if eps <= 0.0:
            abs_vals.append(0.0)
            rel_vals.append(0.0)
            continue
        pert = forward_pert(float(eps))
        pert_L = pool(pert[layer_L])
        abs_b = residual_l2_blowup(clean_L, pert_L, relative=False)
        rel_b = residual_l2_blowup(clean_L, pert_L, relative=True, denom=denom)
        abs_vals.append(abs_b)
        rel_vals.append(rel_b)
    return {
        "epsilon": [float(e) for e in grid],
        "blowup_absolute": abs_vals,
        "blowup_relative": rel_vals,
    }


def plateau_depth(
    blowup_fn: Callable[[float], float],
    *,
    threshold: float = 0.5,
    eps_grid: Sequence[float] | None = None,
) -> float:
    """Largest ε on a grid with blowup < threshold (robustness check, not primary)."""
    grid = list(eps_grid) if eps_grid is not None else [
        1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0
    ]
    depth = 0.0
    for eps in grid:
        if blowup_fn(eps) < threshold:
            depth = eps
        else:
            break
    return depth


def top_k_sensitive_directions(
    sensitivities: Sequence[tuple[str, float]],
    k: int = 10,
) -> list[tuple[str, float]]:
    """Rank directions by sensitivity (smaller ε* = more privileged / less sensitive).

    Returns top-k **most sensitive** (largest vulnerability = largest ε needed inverse,
    i.e. smallest epsilon_star first for privilege; here we sort by ascending ε*).
    For "top-k sensitive" as highest blowup propensity, sort ascending ε*.
    """
    ranked = sorted(sensitivities, key=lambda x: x[1])
    return ranked[:k]


class LayerPerturbHooks:
    """Inject or replace residual at layer k on selected token indices.

    Default (Heim add mode): ``h <- h + ε·d``.
    With ``base_override`` (Heim replace mode): ``h <- base + ε·d`` at those tokens,
    used for random-base plateau tests.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_k: int,
        direction: torch.Tensor,
        token_indices: Sequence[int],
        read_layers: Sequence[int],
        base_override: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer_k = layer_k
        self.direction = _as_unit(direction)
        self.token_indices = list(token_indices)
        self.read_layers = sorted(set(read_layers) | {layer_k})
        self.base_override = None if base_override is None else base_override.float().reshape(-1)
        self.epsilon = 0.0
        self.captured: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _inject_hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h = hidden.clone()
        d = self.direction.to(device=h.device, dtype=h.dtype)
        for t in self.token_indices:
            if 0 <= t < h.shape[1]:
                if self.base_override is not None:
                    b = self.base_override.to(device=h.device, dtype=h.dtype)
                    h[:, t, :] = b + self.epsilon * d
                else:
                    h[:, t, :] = h[:, t, :] + self.epsilon * d
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def _read_hook(self, layer_idx: int):
        def hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured[layer_idx] = hidden.detach()

        return hook

    def register(self) -> None:
        self.remove()
        self._handles.append(
            self.layers[self.layer_k].register_forward_hook(self._inject_hook)
        )
        for idx in self.read_layers:
            # If same as k, read after inject by registering second hook
            self._handles.append(
                self.layers[idx].register_forward_hook(self._read_hook(idx))
            )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        epsilon: float,
    ) -> dict[int, torch.Tensor]:
        self.epsilon = float(epsilon)
        self.captured = {}
        self.register()
        try:
            with torch.inference_mode():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            self.remove()
        return dict(self.captured)
