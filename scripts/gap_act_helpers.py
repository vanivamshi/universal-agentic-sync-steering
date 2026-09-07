"""GAP activation cache helpers (shared; n60 scripts removed)."""

from __future__ import annotations

from typing import Any

import numpy as np


def tid_mean(act: dict[str, Any], layer: int, window_kind: str) -> dict[str, np.ndarray]:
    """Mean layer activation per transcript for a window kind."""
    buckets: dict[str, list[np.ndarray]] = {}
    for rec in act.get("records") or []:
        if rec.get("window_kind") != window_kind:
            continue
        lm = (rec.get("layer_means") or {}).get(str(layer))
        if lm is None:
            continue
        buckets.setdefault(rec["transcript_id"], []).append(
            np.asarray(lm, dtype=np.float64)
        )
    return {t: np.mean(vs, axis=0) for t, vs in buckets.items()}


def strat_split(
    tids: list[str],
    labs: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    *,
    hold_frac: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Stratified train/hold by surface_gap label."""
    pos = [t for t in tids if labs[t].get("surface_gap")]
    neg = [t for t in tids if not labs[t].get("surface_gap")]
    pos = list(pos)
    neg = list(neg)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if not pos or not neg:
        raise RuntimeError("strat_split needs both classes")
    n_hold_pos = max(1, int(round(len(pos) * hold_frac)))
    n_hold_neg = max(1, int(round(len(neg) * hold_frac)))
    hold = pos[:n_hold_pos] + neg[:n_hold_neg]
    train = pos[n_hold_pos:] + neg[n_hold_neg:]
    if not train or not hold:
        raise RuntimeError("strat_split produced empty train or hold")
    return train, hold
