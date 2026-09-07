"""Channel sync control directions and composition.

Factorized (legacy hypothesis, do not freeze without diagonal J):

  d(e) = e_C v_C + e_H v_H + e_O v_O

Hierarchical controller hypothesis (current Stage B lean):

  d = g_C(e_C) v_C + g_H(e_H) v_H
  # v_O is NOT an independent actuator — O is observed after tool execution

  C state → H intervention → tool execution → O state

Stage A: identification. Stage B: causal dose / H mediation.
Do not run 8-way until H mediation gate passes and hierarchy is designed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYER_DEFAULT = 4
ALPHA_DEFAULT = 0.25
CHANNEL_V = ROOT / "data" / "directions" / "sync_channel_V_L4.json"
CHANNEL_V_FROZEN = ROOT / "data" / "directions" / "sync_channel_V_L4.frozen.json"
CHANNELS = ("C", "H", "O")


def unit(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-12)


def error_e(m_star: list[int] | tuple[int, ...], S: list[int] | tuple[int, ...]) -> list[int]:
    return [int(m_star[i]) - int(S[i]) for i in range(3)]


def compose_direction(
    e: list[int] | tuple[int, ...] | np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """d(e) = e_C v_C + e_H v_H + e_O v_O. V shape (3, d) or (d, 3)."""
    e = np.asarray(e, dtype=np.float64).reshape(3)
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] == 3:
        d = e @ V
    elif V.shape[1] == 3:
        d = V @ e
    else:
        raise ValueError(f"V must be (3,d) or (d,3), got {V.shape}")
    return d  # keep magnitude from |e| composition; caller scales by alpha


def compose_direction_hierarchical(
    e: list[int] | tuple[int, ...] | np.ndarray,
    V: np.ndarray,
    *,
    g_C: float | None = None,
    g_H: float | None = None,
    scale_C: float = 0.0,
    scale_H: float = 1.0,
    c_deadzone: float = 0.15,
) -> np.ndarray:
    """H-only asymmetric controller by default: d = g_H(e_H) v_H.

    scale_C=0 freezes C out (observe-only). O never actuated.
    """
    e = np.asarray(e, dtype=np.float64).reshape(3)
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] != 3:
        V = V.T
    if g_H is None:
        g_H = float(scale_H) * float(e[1])
    if g_C is None:
        g_C = float(scale_C) * float(e[0]) if abs(float(e[0])) >= float(c_deadzone) else 0.0
    return float(g_C) * V[0] + float(g_H) * V[1]


def hierarchical_gains(
    e: list[float] | tuple[float, ...] | np.ndarray,
    *,
    scale_C: float = 0.0,
    scale_H: float = 1.0,
    c_deadzone: float = 0.15,
) -> dict[str, float]:
    """Return {g_C, g_H, e_C, e_H, e_O}; default g_C=0 (C frozen out)."""
    e = np.asarray(e, dtype=np.float64).reshape(3)
    g_H = float(scale_H) * float(e[1])
    g_C = float(scale_C) * float(e[0]) if abs(float(e[0])) >= float(c_deadzone) else 0.0
    return {
        "e_C": float(e[0]),
        "e_H": float(e[1]),
        "e_O": float(e[2]),
        "g_C": g_C,
        "g_H": g_H,
        "scale_C": float(scale_C),
        "scale_H": float(scale_H),
        "c_deadzone": float(c_deadzone),
    }


def continuous_q_from_proj(h: np.ndarray, V: np.ndarray) -> np.ndarray:
    """q_i = sigmoid(h · v_i). Preference margin, not binary S."""
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] != 3:
        V = V.T
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    scores = V @ h
    return 1.0 / (1.0 + np.exp(-scores))


def error_q(m_star: list[int] | tuple[int, ...], q: np.ndarray) -> np.ndarray:
    m = np.asarray(m_star, dtype=np.float64).reshape(3)
    return m - np.asarray(q, dtype=np.float64).reshape(3)


def project_out(v: np.ndarray, *basis: np.ndarray) -> np.ndarray:
    """v - sum_k Proj_{b_k}(v) for non-zero basis vectors."""
    out = np.asarray(v, dtype=np.float64).copy()
    for b in basis:
        b = np.asarray(b, dtype=np.float64)
        nb = float(np.linalg.norm(b))
        if nb < 1e-12:
            continue
        b = b / nb
        out = out - float(np.dot(out, b)) * b
    return unit(out)


def orthogonalize_channels(V: np.ndarray) -> np.ndarray:
    """Sequential Gram-Schmidt on rows of V (3, d)."""
    V = np.asarray(V, dtype=np.float64).copy()
    assert V.shape[0] == 3
    for i in range(3):
        v = V[i]
        for j in range(i):
            b = V[j]
            nb2 = float(np.dot(b, b))
            if nb2 < 1e-12:
                continue
            v = v - (float(np.dot(v, b)) / nb2) * b
        V[i] = unit(v)
    return V


def selectivity(delta_target: float, delta_collateral: np.ndarray, lam: float = 1e-3) -> float:
    """|Δ_i| / (λ + ||Δ_{-i}||)."""
    return abs(float(delta_target)) / (lam + float(np.linalg.norm(delta_collateral)))


@dataclass
class ChannelBank:
    """V = [v_C, v_H, v_O], shape (3, d)."""

    V: np.ndarray
    layer: int
    alpha: float
    meta: dict[str, Any]
    frozen: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "ChannelBank":
        path = path or (CHANNEL_V_FROZEN if CHANNEL_V_FROZEN.is_file() else CHANNEL_V)
        blob = json.loads(path.read_text())
        V = np.stack(
            [
                np.asarray(blob["v_C"], dtype=np.float64),
                np.asarray(blob["v_H"], dtype=np.float64),
                np.asarray(blob["v_O"], dtype=np.float64),
            ],
            axis=0,
        )
        return cls(
            V=V,
            layer=int(blob.get("layer", LAYER_DEFAULT)),
            alpha=float(blob.get("alpha", ALPHA_DEFAULT)),
            meta=dict(blob.get("meta") or {}),
            frozen=bool(blob.get("frozen", path == CHANNEL_V_FROZEN)),
        )

    def direction_for_error(self, e: list[int] | np.ndarray) -> np.ndarray:
        return compose_direction(e, self.V)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "alpha": self.alpha,
            "frozen": self.frozen,
            "v_C": self.V[0].astype(float).tolist(),
            "v_H": self.V[1].astype(float).tolist(),
            "v_O": self.V[2].astype(float).tolist(),
            "meta": self.meta,
            "equation": "d(e) = e_C v_C + e_H v_H + e_O v_O; h' = h + alpha d(e)",
        }


def save_bank(bank: ChannelBank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank.to_dict(), indent=2) + "\n")


def jacobian_from_deltas(
    deltas: dict[str, list[float]],
) -> np.ndarray:
    """Build 3x3 J with rows = affected channel (C,H,O), cols = steered (v_C,v_H,v_O).

    deltas keys: 'v_C', 'v_H', 'v_O' → [ΔC, ΔH, ΔO] mean effects at +α.
    """
    J = np.zeros((3, 3), dtype=np.float64)
    for j, name in enumerate(("v_C", "v_H", "v_O")):
        d = deltas.get(name)
        if d is None:
            continue
        J[:, j] = np.asarray(d, dtype=np.float64).reshape(3)
    return J


def summarize_jacobian(J: np.ndarray) -> dict[str, Any]:
    J = np.asarray(J, dtype=np.float64)
    diag = np.diag(J)
    off = J.copy()
    np.fill_diagonal(off, 0.0)
    return {
        "J": J.tolist(),
        "diag": diag.tolist(),
        "offdiag_l2": float(np.linalg.norm(off)),
        "diag_abs_mean": float(np.mean(np.abs(diag))),
        "approx_diagonal": bool(
            float(np.mean(np.abs(diag))) > 0.05
            and float(np.linalg.norm(off)) < float(np.mean(np.abs(diag))) * 2.0
        ),
        "channels": list(CHANNELS),
    }
