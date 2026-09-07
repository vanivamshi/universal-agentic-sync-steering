"""Alignment controller: m* conditions activation intervention.

Architecture (universal / controllable alignment):

  m*  ──►  alignment_controller  ──►  h' = h + α · d(m*)
  neutral task  ──►  free model (PLAN / tools / FINAL)  ──►  S
  compare S to m*

m* is an instruction to the *controller*, not text injected into PLAN/FINAL.

Preferred path (current): hierarchical hypothesis
  d = g_C(e_C) v_C + g_H(e_H) v_H   # no independent v_O actuator
  C state → H intervention → tool execution → O observe
Do not freeze factorized 3-way V without H-mediation gate.
Legacy: centroid / ±v_repair fallbacks below.
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
VREPAIR = ROOT / "data" / "directions" / "sync_v_repair_eq_L4.jsonl"
VDELTA = ROOT / "data" / "directions" / "sync_v_delta_L4.jsonl"
MSTAR_BANK = ROOT / "data" / "directions" / "sync_mstar_bank_L4.json"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _load_jsonl_vec(path: Path) -> tuple[np.ndarray, float]:
    row = json.loads(path.read_text().splitlines()[0])
    alpha = float(row.get("meta", {}).get("alpha_frozen", ALPHA_DEFAULT))
    return _unit(np.asarray(row["vector"], dtype=np.float64)), alpha


@dataclass(frozen=True)
class ControlDecision:
    m_star: tuple[int, int, int]
    intervention: str
    direction_id: str | None
    alpha: float
    sign: float
    layer: int
    reason: str
    coverage: str
    vector: tuple[float, ...] | None = None  # explicit d(m*) when from bank


@dataclass
class DirectionBank:
    v_repair: np.ndarray
    v_delta: np.ndarray | None
    alpha_repair: float
    layer: int = LAYER_DEFAULT
    # (h,o) -> centroid
    centroids_ho: dict[tuple[int, int], np.ndarray] | None = None
    v_out: np.ndarray | None = None  # disclose - hide under H=1

    @classmethod
    def load_default(cls) -> "DirectionBank":
        v_r, a = _load_jsonl_vec(VREPAIR)
        v_d = None
        if VDELTA.is_file():
            v_d, _ = _load_jsonl_vec(VDELTA)
        centroids = None
        v_out = None
        if MSTAR_BANK.is_file():
            blob = json.loads(MSTAR_BANK.read_text())
            centroids = {}
            for _k, c in (blob.get("centroids_ho") or {}).items():
                if c.get("mu") is not None:
                    centroids[(int(c["h"]), int(c["o"]))] = np.asarray(c["mu"], dtype=np.float64)
            if blob.get("v_out_disclose_minus_hide") is not None:
                v_out = _unit(np.asarray(blob["v_out_disclose_minus_hide"], dtype=np.float64))
        return cls(
            v_repair=v_r,
            v_delta=v_d,
            alpha_repair=a,
            layer=LAYER_DEFAULT,
            centroids_ho=centroids,
            v_out=v_out,
        )


def decide_control(
    m_star: list[int] | tuple[int, int, int],
    *,
    S_baseline: list[int] | None = None,
    private_loaded: bool | None = None,
    bank: DirectionBank | None = None,
    alpha: float | None = None,
) -> ControlDecision:
    """Map requested m* → steering decision without touching text."""
    bank = bank or DirectionBank.load_default()
    m = (int(m_star[0]), int(m_star[1]), int(m_star[2]))
    alpha_use = float(alpha if alpha is not None else bank.alpha_repair)

    S = None
    if S_baseline is not None and len(S_baseline) >= 3:
        S = (int(S_baseline[0]), int(S_baseline[1]), int(S_baseline[2]))

    if private_loaded is None:
        private_loaded = bool(S is not None and S[1] == 1)

    # Prefer centroid-difference d(m*) in (H,O) when bank + baseline exist
    if bank.centroids_ho and S is not None and S[1] in (0, 1) and S[2] in (0, 1):
        src = (S[1], S[2])
        tgt = (m[1], m[2])
        if src in bank.centroids_ho and tgt in bank.centroids_ho:
            if src == tgt:
                plan_gap = S[0] != m[0] and S[0] in (0, 1)
                return ControlDecision(
                    m_star=m,
                    intervention="none",
                    direction_id=None,
                    alpha=0.0,
                    sign=0.0,
                    layer=bank.layer,
                    reason=(
                        "HO already at target"
                        + ("; plan gap uncovered (plan not yet observed in geometry bank)" if plan_gap else "")
                    ),
                    coverage="partial" if plan_gap else "ho_matched",
                    vector=None,
                )
            vec = _unit(bank.centroids_ho[tgt] - bank.centroids_ho[src])
            plan_gap = S[0] != m[0] and S[0] in (0, 1)
            return ControlDecision(
                m_star=m,
                intervention="activation_steer",
                direction_id=f"mstar_bank_from_{src[0]}{src[1]}_to_{tgt[0]}{tgt[1]}",
                alpha=alpha_use,
                sign=1.0,
                layer=bank.layer,
                reason=(
                    f"d(m*) = μ[{tgt}]−μ[{src}] from geometry bank"
                    + ("; plan not yet in natural bank" if plan_gap else "")
                ),
                coverage="partial",
                vector=tuple(float(x) for x in vec.tolist()),
            )

    # Fallback: out-axis ±v_repair / ±v_out under private load
    if not private_loaded:
        return ControlDecision(
            m_star=m,
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=bank.layer,
            reason="no private execution; HO centroid steer not applicable without baseline HO",
            coverage="none",
        )

    m_out = m[2]
    sign = 1.0 if m_out == 1 else -1.0
    need = True
    if S is not None and S[2] in (0, 1) and S[2] == m_out:
        need = False
    if not need:
        return ControlDecision(
            m_star=m,
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=bank.layer,
            reason="out matches m*; plan/hook not controllable via Phase-0 fallback",
            coverage="partial",
        )

    axis = bank.v_out if bank.v_out is not None else bank.v_repair
    did = "v_out_from_bank" if bank.v_out is not None else "sync_v_repair_eq_L4"
    vec = _unit(sign * axis)
    return ControlDecision(
        m_star=m,
        intervention="activation_steer",
        direction_id=did,
        alpha=alpha_use,
        sign=sign,
        layer=bank.layer,
        reason=f"fallback steer out toward m*_out={m_out}",
        coverage="partial",
        vector=tuple(float(x) for x in vec.tolist()),
    )


def direction_vector(decision: ControlDecision, bank: DirectionBank | None = None) -> np.ndarray | None:
    if decision.intervention != "activation_steer":
        return None
    if decision.vector is not None:
        return _unit(np.asarray(decision.vector, dtype=np.float64))
    if decision.sign == 0.0:
        return None
    bank = bank or DirectionBank.load_default()
    axis = bank.v_out if bank.v_out is not None else bank.v_repair
    return _unit(decision.sign * axis)


def make_steer_hook(model, decision: ControlDecision, bank: DirectionBank | None = None):
    vec = direction_vector(decision, bank)
    if vec is None:
        return None
    import torch
    from activation_pipeline.steering import ActivationSteerHook

    return ActivationSteerHook(
        model,
        layer=decision.layer,
        direction=torch.tensor(vec, dtype=torch.float32),
        alpha=decision.alpha,
        pos_mode="last",
        collect_stats=True,
    )


def decision_dict(d: ControlDecision) -> dict[str, Any]:
    return {
        "m_star": list(d.m_star),
        "intervention": d.intervention,
        "direction_id": d.direction_id,
        "alpha": d.alpha,
        "sign": d.sign,
        "layer": d.layer,
        "reason": d.reason,
        "coverage": d.coverage,
        "has_explicit_vector": d.vector is not None,
    }



def decide_control_factorized(
    m_star: list[int] | tuple[int, int, int],
    *,
    S_baseline: list[int],
    alpha: float | None = None,
    require_frozen: bool = True,
) -> ControlDecision:
    """Closed-loop factorized controller: d = e · V. Default requires Stage-2 freeze."""
    from scripts.sync_channel_control import CHANNEL_V, CHANNEL_V_FROZEN, ChannelBank, compose_direction, error_e

    if require_frozen and not CHANNEL_V_FROZEN.is_file():
        return ControlDecision(
            m_star=(int(m_star[0]), int(m_star[1]), int(m_star[2])),
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=4,
            reason="factorized controller refused: no Stage-2 freeze (run causal J first)",
            coverage="none",
        )
    path = CHANNEL_V_FROZEN if CHANNEL_V_FROZEN.is_file() else CHANNEL_V
    if not path.is_file():
        return decide_control(m_star, S_baseline=S_baseline, private_loaded=True)
    bank = ChannelBank.load(path)
    m = (int(m_star[0]), int(m_star[1]), int(m_star[2]))
    e = error_e(m, S_baseline)
    if e == [0, 0, 0]:
        return ControlDecision(
            m_star=m,
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=bank.layer,
            reason="e=0 already",
            coverage="factorized",
        )
    d = compose_direction(e, bank.V)
    if float(np.linalg.norm(d)) < 1e-12:
        return ControlDecision(
            m_star=m,
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=bank.layer,
            reason="composed d≈0",
            coverage="factorized",
        )
    return ControlDecision(
        m_star=m,
        intervention="activation_steer",
        direction_id="factorized_e_dot_V",
        alpha=float(alpha if alpha is not None else bank.alpha),
        sign=1.0,
        layer=bank.layer,
        reason=f"d(e) with e={e}; frozen={bank.frozen}",
        coverage="factorized" if bank.frozen else "factorized_unfrozen_candidates",
        vector=tuple(float(x) for x in d.tolist()),
    )


def decide_control_hierarchical(
    m_star: list[int] | tuple[int, int, int],
    *,
    S_baseline: list[int] | None = None,
    q_baseline: list[float] | tuple[float, ...] | None = None,
    alpha: float | None = None,
    scale_C: float = 0.0,
    scale_H: float = 1.0,
    c_deadzone: float = 0.15,
) -> ControlDecision:
    """Frozen architecture: d = g_H(e_H) v_H (C observe-only by default).

    Prefer continuous e = m* - q. scale_C defaults to 0 (C frozen out after
    dose×rep ablation). O never actuated.
    """
    from scripts.sync_channel_control import (
        CHANNEL_V,
        ChannelBank,
        compose_direction_hierarchical,
        error_e,
        error_q,
        hierarchical_gains,
    )

    if not CHANNEL_V.is_file():
        return decide_control(
            m_star,
            S_baseline=S_baseline or [0, 0, 0],
            private_loaded=True,
        )
    bank = ChannelBank.load(CHANNEL_V)
    m = (int(m_star[0]), int(m_star[1]), int(m_star[2]))
    if q_baseline is not None:
        e = error_q(m, np.asarray(q_baseline, dtype=np.float64))
    else:
        e = np.asarray(error_e(m, S_baseline or [0, 0, 0]), dtype=np.float64)
    gains = hierarchical_gains(e, scale_C=scale_C, scale_H=scale_H, c_deadzone=c_deadzone)
    if abs(gains["g_C"]) < 1e-12 and abs(gains["g_H"]) < 1e-12:
        return ControlDecision(
            m_star=m,
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=bank.layer,
            reason=f"g_C=g_H=0 (e_O={gains['e_O']:.3f} observe-only)",
            coverage="hierarchical",
        )
    d = compose_direction_hierarchical(
        e,
        bank.V,
        g_C=gains["g_C"],
        g_H=gains["g_H"],
    )
    if float(np.linalg.norm(d)) < 1e-12:
        return ControlDecision(
            m_star=m,
            intervention="none",
            direction_id=None,
            alpha=0.0,
            sign=0.0,
            layer=bank.layer,
            reason="hierarchical d≈0",
            coverage="hierarchical",
        )
    return ControlDecision(
        m_star=m,
        intervention="activation_steer",
        direction_id="hierarchical_C_H",
        alpha=float(alpha if alpha is not None else bank.alpha),
        sign=1.0,
        layer=bank.layer,
        reason=(
            f"g_C={gains['g_C']:.3f} g_H={gains['g_H']:.3f} "
            f"(e_C={gains['e_C']:.3f} e_H={gains['e_H']:.3f}; O observe-only)"
        ),
        coverage="hierarchical_architecture",
        vector=tuple(float(x) for x in d.tolist()),
    )
