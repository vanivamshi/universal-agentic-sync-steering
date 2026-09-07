"""Orthogonal subspace utilities (Procrustes + cosine + principal angles)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SubspaceCompareResult:
    mean_cosine: float
    procrustes_disparity: float
    n_prose: int
    n_tool: int
    hidden: int
    principal_angles_deg: list[float] = field(default_factory=list)
    mean_principal_angle_deg: float | None = None
    max_principal_angle_deg: float | None = None
    subspace_rank: int = 0


def _stack_means(records_means: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([m.float().reshape(-1) for m in records_means], dim=0)


def mean_pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine between centroids of two activation point sets (unit-normalized)."""
    ma = a.mean(dim=0)
    mb = b.mean(dim=0)
    ma_n = ma / (torch.linalg.norm(ma) + 1e-8)
    mb_n = mb / (torch.linalg.norm(mb) + 1e-8)
    return float(torch.dot(ma_n, mb_n))


def orthogonal_procrustes(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Find orthonormal R minimizing ||a R - b||_F (rows = points).

    Returns (R, disparity = ||aR - b||_F^2 / ||b||_F^2).
    Uses SVD of A^T B (classic orthogonal Procrustes).
    """
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    n = min(a_c.shape[0], b_c.shape[0])
    a_c = a_c[:n]
    b_c = b_c[:n]
    m = a_c.T @ b_c
    u, _, vh = torch.linalg.svd(m, full_matrices=False)
    r = u @ vh
    aligned = a_c @ r
    num = torch.linalg.norm(aligned - b_c) ** 2
    den = torch.linalg.norm(b_c) ** 2 + 1e-8
    disparity = float(num / den)
    return r, disparity


def cloud_basis(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Orthonormal PCA basis for a point set (top-``rank`` right singular vectors).

    ``x``: (n, d) window-mean activations. Returns ``Q`` with shape (d, r), r ≤ rank.
    Used only for exploratory principal angles between prose and tool PCA subspaces.
    """
    if x.shape[0] < 2:
        raise ValueError("need ≥2 points for a subspace basis")
    x_c = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(x_c, full_matrices=False)
    nonzero = int((s > 1e-6).sum().item())
    r = max(1, min(rank, nonzero, vh.shape[0]))
    return vh[:r].T.contiguous()


def principal_angles_deg(qa: torch.Tensor, qb: torch.Tensor) -> list[float]:
    """Principal angles (degrees) between PCA subspaces (orthonormal column bases)."""
    r = min(qa.shape[1], qb.shape[1])
    qa = qa[:, :r]
    qb = qb[:, :r]
    qa, _ = torch.linalg.qr(qa, mode="reduced")
    qb, _ = torch.linalg.qr(qb, mode="reduced")
    s = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(s))
    return [float(a) for a in angles]


def compare_prose_tool_subspaces(
    prose_means: list[torch.Tensor],
    tool_means: list[torch.Tensor],
    *,
    subspace_rank: int | None = None,
) -> SubspaceCompareResult:
    if not prose_means or not tool_means:
        raise ValueError("need at least one prose and one tool mean vector")
    a = _stack_means(prose_means)
    b = _stack_means(tool_means)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"hidden mismatch {a.shape} vs {b.shape}")
    cos = mean_pairwise_cosine(a, b)
    _, disp = orthogonal_procrustes(a, b)

    # Default rank: min(n-1, 4) — small-n pilot; not a preregistered hypothesis metric
    max_r = min(a.shape[0] - 1, b.shape[0] - 1, 4)
    rank = subspace_rank if subspace_rank is not None else max(1, max_r)
    angles: list[float] = []
    mean_ang: float | None = None
    max_ang: float | None = None
    used_rank = 0
    if a.shape[0] >= 2 and b.shape[0] >= 2 and rank >= 1:
        qa = cloud_basis(a, rank)
        qb = cloud_basis(b, rank)
        angles = principal_angles_deg(qa, qb)
        used_rank = len(angles)
        if angles:
            mean_ang = float(sum(angles) / len(angles))
            max_ang = float(max(angles))

    return SubspaceCompareResult(
        mean_cosine=cos,
        procrustes_disparity=disp,
        n_prose=a.shape[0],
        n_tool=b.shape[0],
        hidden=a.shape[1],
        principal_angles_deg=angles,
        mean_principal_angle_deg=mean_ang,
        max_principal_angle_deg=max_ang,
        subspace_rank=used_rank,
    )
