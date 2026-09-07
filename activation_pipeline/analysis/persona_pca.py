"""Phase-1 persona PCA utilities (Lu et al.–style, numpy SVD — no sklearn)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class PcaResult:
    components: np.ndarray  # (n_components, hidden)
    explained_variance_ratio: np.ndarray  # (n_components,)
    singular_values: np.ndarray
    mean: np.ndarray  # (hidden,)
    n_samples: int
    n_components: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_variance": np.cumsum(self.explained_variance_ratio).tolist(),
            "dims_for_70": int(np.searchsorted(np.cumsum(self.explained_variance_ratio), 0.70) + 1),
            "dims_for_80": int(np.searchsorted(np.cumsum(self.explained_variance_ratio), 0.80) + 1),
            "dims_for_90": int(np.searchsorted(np.cumsum(self.explained_variance_ratio), 0.90) + 1),
        }


def fit_pca(X: np.ndarray, *, n_components: int | None = None) -> PcaResult:
    """Center rows of X (n_samples, hidden) and SVD → principal components."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2d, got {X.shape}")
    n, d = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD
    k = min(n - 1, d) if n_components is None else min(n_components, n - 1, d)
    if k < 1:
        raise ValueError("need ≥2 samples for PCA")
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    # variance explained from singular values
    var = (S[:k] ** 2) / max(n - 1, 1)
    total = var.sum() + 1e-12
    # include residual variance from discarded components
    if S.shape[0] > k:
        total = (S**2).sum() / max(n - 1, 1) + 1e-12
    ratio = var / total
    return PcaResult(
        components=components.astype(np.float64),
        explained_variance_ratio=ratio.astype(np.float64),
        singular_values=S[:k].astype(np.float64),
        mean=mean.astype(np.float64),
        n_samples=n,
        n_components=k,
    )


def principal_angles_deg(A: np.ndarray, B: np.ndarray, *, rank: int | None = None) -> dict[str, Any]:
    """Principal angles between row-spaces of A and B (each (k, d))."""
    r = rank or min(A.shape[0], B.shape[0], 4)
    A = A[:r]
    B = B[:r]
    # Orthonormalize
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    Qa = Qa[:, :r]
    Qb = Qb[:, :r]
    M = Qa.T @ Qb
    s = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "rank": r,
        "angles_deg": angles.tolist(),
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg": float(angles.max()),
        "min_angle_deg": float(angles.min()),
        "near_orthogonal_90": bool(float(angles.mean()) >= 70.0),
    }


def component_alignment(A: np.ndarray, B: np.ndarray, *, n: int = 5) -> list[dict[str, Any]]:
    """Per-PC absolute cosine between mode-A and mode-B components (matched by index)."""
    rows = []
    for i in range(min(n, A.shape[0], B.shape[0])):
        a = A[i]
        b = B[i]
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        cos = float(np.dot(a, b))
        # best match in B for A[i]
        B_u = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        best = B_u @ a
        j = int(np.argmax(np.abs(best)))
        rows.append(
            {
                "pc_index": i,
                "matched_index_cosine": cos,
                "best_match_index": j,
                "best_match_abs_cosine": float(abs(best[j])),
            }
        )
    return rows


def as_unit_torch(v: np.ndarray) -> torch.Tensor:
    t = torch.tensor(v, dtype=torch.float32)
    return t / (torch.linalg.norm(t) + 1e-8)
