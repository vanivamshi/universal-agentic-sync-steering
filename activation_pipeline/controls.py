"""Step 4 control-direction extraction and validation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from activation_pipeline.analysis.bootstrap_ci import bca_ci
from activation_pipeline.directions import (
    DirectionVector,
    cosine,
    last_token_residual,
    mean_difference_direction,
)
from activation_pipeline.loader import LoadedModel


@dataclass
class LearnedControlStability:
    direction_id: str
    kind: str
    domain: str | None
    n_pairs: int
    n_boot: int
    n_splits_per_stat: int
    threshold: float
    required_layers: int
    layers: list[dict[str, Any]]
    n_layers_passing: int
    passed: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RandomControlDiagnostics:
    layer: int
    seed: int
    n_random: int
    hidden_size: int
    min_norm: float
    max_norm: float
    max_abs_dot_with_fixed_basis: float
    max_abs_pairwise_cosine: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def collect_pair_activations(
    loaded: LoadedModel,
    pairs: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    system: str = "You are a helpful assistant.",
) -> tuple[list[dict[int, torch.Tensor]], list[dict[int, torch.Tensor]]]:
    """Cache positive and negative residuals once for a contrast set."""
    positive: list[dict[int, torch.Tensor]] = []
    negative: list[dict[int, torch.Tensor]] = []
    for pair in pairs:
        for key, bucket in (("positive", positive), ("negative", negative)):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": pair[key]},
            ]
            bucket.append(last_token_residual(loaded, messages, layers=layers))
    return positive, negative


def direction_from_indices(
    positive: Sequence[dict[int, torch.Tensor]],
    negative: Sequence[dict[int, torch.Tensor]],
    indices: Sequence[int],
    layer: int,
) -> torch.Tensor:
    return mean_difference_direction(
        [positive[i][layer] for i in indices],
        [negative[i][layer] for i in indices],
    )


def _split_half_stat(
    positive: Sequence[dict[int, torch.Tensor]],
    negative: Sequence[dict[int, torch.Tensor]],
    indices: Sequence[int],
    *,
    layer: int,
    n_splits: int,
    seed: int,
) -> float:
    """Mean cosine across deterministic random half splits of paired examples."""
    if len(indices) < 4:
        return float("nan")
    rng = random.Random(seed)
    half = len(indices) // 2
    values = []
    for _ in range(n_splits):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        a = shuffled[:half]
        b = shuffled[half : 2 * half]
        if not a or not b:
            continue
        da = direction_from_indices(positive, negative, a, layer)
        db = direction_from_indices(positive, negative, b, layer)
        values.append(cosine(da, db))
    return float(sum(values) / len(values)) if values else float("nan")


def learned_control_stability(
    positive: Sequence[dict[int, torch.Tensor]],
    negative: Sequence[dict[int, torch.Tensor]],
    *,
    layers: Sequence[int],
    direction_id: str,
    kind: str,
    domain: str | None,
    n_boot: int = 400,
    n_splits_per_stat: int = 20,
    seed: int = 0,
    threshold: float = 0.70,
    required_layers: int = 3,
) -> LearnedControlStability:
    """Split-half stability with paired-example BCa intervals."""
    if len(positive) != len(negative):
        raise ValueError("positive and negative activation counts must match")
    n = len(positive)
    if n < 4:
        raise ValueError("need at least four contrast pairs")
    all_indices = list(range(n))
    rows: list[dict[str, Any]] = []

    for layer in layers:
        point = _split_half_stat(
            positive,
            negative,
            all_indices,
            layer=layer,
            n_splits=n_splits_per_stat,
            seed=seed + layer,
        )
        boots = []
        rng = random.Random(seed + 10_000 + layer)
        for b in range(n_boot):
            sample = [rng.randrange(n) for _ in range(n)]
            boots.append(
                _split_half_stat(
                    positive,
                    negative,
                    sample,
                    layer=layer,
                    n_splits=n_splits_per_stat,
                    seed=seed + 20_000 + layer * 1_000 + b,
                )
            )
        jacks = []
        for leave_out in range(n):
            sample = [i for i in all_indices if i != leave_out]
            jacks.append(
                _split_half_stat(
                    positive,
                    negative,
                    sample,
                    layer=layer,
                    n_splits=n_splits_per_stat,
                    seed=seed + 30_000 + layer + leave_out,
                )
            )
        lo, hi, diagnostics = bca_ci(point, boots, jacks)
        rows.append(
            {
                "layer": int(layer),
                "split_half_cosine": point,
                "bca_ci95": [lo, hi],
                "bca_diagnostics": diagnostics,
                "meets_threshold": bool(point >= threshold),
                "ci_lower_meets_threshold": bool(lo >= threshold),
            }
        )

    n_passing = sum(1 for row in rows if row["meets_threshold"])
    passed = n_passing >= required_layers
    return LearnedControlStability(
        direction_id=direction_id,
        kind=kind,
        domain=domain,
        n_pairs=n,
        n_boot=n_boot,
        n_splits_per_stat=n_splits_per_stat,
        threshold=threshold,
        required_layers=required_layers,
        layers=rows,
        n_layers_passing=n_passing,
        passed=passed,
        status="eligible_for_step5" if passed else "unstable_rebuild_or_exclude",
    )


def full_directions_from_activations(
    positive: Sequence[dict[int, torch.Tensor]],
    negative: Sequence[dict[int, torch.Tensor]],
    *,
    layers: Sequence[int],
    direction_id: str,
    kind: str,
    domain: str | None,
) -> list[DirectionVector]:
    indices = list(range(len(positive)))
    return [
        DirectionVector(
            direction_id=f"{direction_id}_L{layer}",
            kind=kind,
            mode="prose",
            layer=int(layer),
            vector=direction_from_indices(positive, negative, indices, layer).tolist(),
            n_pos=len(positive),
            n_neg=len(negative),
            meta={"domain": domain, "role": "learned_control"},
        )
        for layer in layers
    ]


def _orthogonalize(vector: torch.Tensor, basis: Sequence[torch.Tensor]) -> torch.Tensor:
    out = vector.float().clone()
    for fixed in basis:
        unit = fixed.float()
        unit = unit / (torch.linalg.norm(unit) + 1e-12)
        out = out - torch.dot(out, unit) * unit
    norm = torch.linalg.norm(out)
    if norm < 1e-8:
        raise ValueError("candidate collapsed during orthogonalization")
    return out / norm


def build_random_orthogonal_controls(
    *,
    layer: int,
    hidden_size: int,
    fixed_basis: Sequence[torch.Tensor],
    n_random: int = 16,
    seed: int = 20260730,
    tolerance: float = 1e-5,
) -> tuple[list[DirectionVector], RandomControlDiagnostics]:
    """Sample deterministic random controls orthogonal to fixed and prior vectors."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer)
    # Build an orthonormal fixed basis first. Projecting sequentially against
    # merely normalized, non-orthogonal vectors can reintroduce earlier components.
    fixed_units: list[torch.Tensor] = []
    for vector in fixed_basis:
        try:
            fixed_units.append(_orthogonalize(vector.float(), fixed_units))
        except ValueError:
            # Skip linearly dependent fixed directions.
            continue
    random_units: list[torch.Tensor] = []
    attempts = 0
    while len(random_units) < n_random:
        attempts += 1
        if attempts > n_random * 20:
            raise RuntimeError("could not generate enough orthogonal random controls")
        candidate = torch.randn(hidden_size, generator=generator)
        try:
            candidate = _orthogonalize(candidate, [*fixed_units, *random_units])
        except ValueError:
            continue
        random_units.append(candidate)

    directions = [
        DirectionVector(
            direction_id=f"rand_{i:02d}_L{layer}",
            kind="random_control",
            mode="prose",
            layer=layer,
            vector=vector.tolist(),
            n_pos=0,
            n_neg=0,
            meta={"seed": seed, "role": "random_orthogonal_control"},
        )
        for i, vector in enumerate(random_units)
    ]

    norms = [float(torch.linalg.norm(vector)) for vector in random_units]
    fixed_dots = [
        abs(float(torch.dot(random, fixed)))
        for random in random_units
        for fixed in fixed_units
    ]
    pairwise = [
        abs(float(torch.dot(random_units[i], random_units[j])))
        for i in range(len(random_units))
        for j in range(i + 1, len(random_units))
    ]
    max_fixed = max(fixed_dots, default=0.0)
    max_pairwise = max(pairwise, default=0.0)
    diagnostics = RandomControlDiagnostics(
        layer=layer,
        seed=seed,
        n_random=n_random,
        hidden_size=hidden_size,
        min_norm=min(norms),
        max_norm=max(norms),
        max_abs_dot_with_fixed_basis=max_fixed,
        max_abs_pairwise_cosine=max_pairwise,
        passed=bool(
            max(abs(norm - 1.0) for norm in norms) <= tolerance
            and max_fixed <= tolerance
            and max_pairwise <= tolerance
        ),
    )
    return directions, diagnostics
