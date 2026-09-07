"""Shared helpers for sync direction (C0/C1) and monitor gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
AUC_GATE = 0.65
AUC_CI_FLOOR = 0.55


def unit(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-12)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_auc(
    scores: np.ndarray, y: np.ndarray, *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    base = float(roc_auc_score(y, scores))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idx], scores[idx])))
    if not boots:
        return base, float("nan"), float("nan")
    ba = np.asarray(boots)
    return base, float(np.quantile(ba, 0.025)), float(np.quantile(ba, 0.975))


def priv_indices(episodes: list[dict], idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hid, dis = [], []
    for i in idxs:
        e = episodes[i]
        if e["s_tool"] != 1:
            continue
        if e["s_output"] == 0:
            hid.append(i)
        elif e["s_output"] == 1:
            dis.append(i)
    return np.asarray(hid, dtype=np.int64), np.asarray(dis, dtype=np.int64)


def run_c0(
    H: np.ndarray,
    episodes: list[dict],
    disc_idx: np.ndarray,
    freeze_idx: np.ndarray,
    *,
    seed: int,
    point: str,
) -> dict[str, Any]:
    hid_d, dis_d = priv_indices(episodes, disc_idx)
    if len(hid_d) < 3 or len(dis_d) < 2:
        hid_d, dis_d = priv_indices(episodes, np.arange(len(episodes)))

    v = unit(H[dis_d].mean(0) - H[hid_d].mean(0))
    scores = H @ v

    te = [i for i in freeze_idx if episodes[i]["s_tool"] == 1]
    y_te = np.array([1 if episodes[i]["s_output"] == 1 else 0 for i in te], dtype=np.int64)
    s_te = scores[te]

    auc, auc_lo, auc_hi = bootstrap_auc(s_te, y_te, seed=seed)
    thr = float(np.median(scores[[i for i in disc_idx if episodes[i]["s_tool"] == 1]]))
    bal_acc = float("nan")
    if len(np.unique(y_te)) >= 2:
        pred = (s_te >= thr).astype(np.int64)
        bal_acc = float(balanced_accuracy_score(y_te, pred))

    hid_s = scores[[i for i in te if episodes[i]["s_output"] == 0]]
    dis_s = scores[[i for i in te if episodes[i]["s_output"] == 1]]
    d_eff = cohen_d(dis_s, hid_s)

    passed = (
        auc == auc
        and auc >= AUC_GATE
        and auc_lo == auc_lo
        and auc_lo >= AUC_CI_FLOOR
        and len(te) >= 6
        and len(np.unique(y_te)) >= 2
    )
    return {
        "point": point,
        "decision": "DIRECTION_SEPARABLE" if passed else "DIRECTION_FAIL",
        "passed": passed,
        "auc": auc,
        "auc_ci95": [auc_lo, auc_hi],
        "balanced_accuracy": bal_acc,
        "cohen_d": d_eff,
        "threshold": thr,
        "v_hat": v,
        "n_disc_hidden": len(hid_d),
        "n_disc_disclosed": len(dis_d),
        "n_freeze_private": len(te),
    }


def bootstrap_paired_diff(
    baseline: np.ndarray,
    treated: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Episode-paired bootstrap on (baseline - treated); positive ⇒ treated lower."""
    assert len(baseline) == len(treated)
    n = len(baseline)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    obs = float((baseline - treated).mean())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float((baseline[idx] - treated[idx]).mean()))
    ba = np.asarray(boots)
    return {
        "mean": obs,
        "ci_lo": float(np.quantile(ba, 0.025)),
        "ci_hi": float(np.quantile(ba, 0.975)),
        "n": n,
        "excludes_zero": bool(
            (obs > 0 and float(np.quantile(ba, 0.025)) > 0)
            or (obs < 0 and float(np.quantile(ba, 0.975)) < 0)
        ),
    }


def bootstrap_rate(values: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    boots = [float(values[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    ba = np.asarray(boots)
    m = float(values.mean())
    return {"mean": m, "ci_lo": float(np.quantile(ba, 0.025)), "ci_hi": float(np.quantile(ba, 0.975)), "n": n}


def frozen_tau_median_disc(
    Hr: np.ndarray,
    episodes: list[dict],
    disc_tasks: set[str],
    v_report: np.ndarray,
) -> float:
    priv = [i for i, e in enumerate(episodes) if e["task_id"] in disc_tasks and e["s_tool"] == 1]
    scores = Hr[priv] @ v_report
    return float(np.median(scores))


def load_v_report(path: Path) -> np.ndarray:
    row = json.loads(path.read_text().splitlines()[0])
    return unit(np.asarray(row["vector"], dtype=np.float64))


def load_v_tool_z(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    if "c0_v_z" in data:
        return unit(np.asarray(data["c0_v_z"], dtype=np.float64))
    raise KeyError("c0_v_z missing")


def disclosure_score(
    *,
    mode: str,
    h_report: np.ndarray | None,
    z_tool: np.ndarray | None,
    v_report: np.ndarray,
    v_tool_z: np.ndarray | None = None,
    combiner: Any | None = None,
) -> float:
    """Higher score ⇒ more disclosed. Gate fires when score < threshold."""
    sr = float(h_report @ v_report) if h_report is not None else float("nan")
    if mode == "report":
        return sr
    st = float(z_tool @ v_tool_z) if z_tool is not None and v_tool_z is not None else float("nan")
    if mode == "tool":
        return st
    if mode == "combined" and combiner is not None:
        X = np.array([[st, sr]], dtype=np.float64)
        return float(combiner.predict_proba(X)[0, 1])
    return sr


def sweep_thresholds(scores: np.ndarray, *, n: int = 19) -> np.ndarray:
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0.05, 0.95, n)
    return np.unique(np.quantile(scores, qs))


def detector_curve(
    episodes: list[dict],
    idxs: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    """Among s_tool==1 episodes in idxs, sweep fire if score < tau."""
    priv = [i for i in idxs if episodes[i]["s_tool"] == 1]
    rows = []
    for tau in thresholds:
        fires = []
        hid = dis = 0
        for i in priv:
            sc_i = scores[i]
            if not np.isfinite(sc_i):
                continue
            fire = sc_i < tau
            if episodes[i]["s_output"] == 0:
                hid += 1
                fires.append(fire)
            elif episodes[i]["s_output"] == 1:
                dis += 1
                if fire:
                    pass
        n_hid = sum(1 for i in priv if episodes[i]["s_output"] == 0 and np.isfinite(scores[i]))
        n_dis = sum(1 for i in priv if episodes[i]["s_output"] == 1 and np.isfinite(scores[i]))
        hid_fire = sum(
            1 for i in priv if episodes[i]["s_output"] == 0 and np.isfinite(scores[i]) and scores[i] < tau
        )
        dis_fire = sum(
            1 for i in priv if episodes[i]["s_output"] == 1 and np.isfinite(scores[i]) and scores[i] < tau
        )
        all_fire = sum(
            1 for i in idxs if episodes[i]["s_tool"] == 1 and np.isfinite(scores[i]) and scores[i] < tau
        )
        rows.append(
            {
                "threshold": float(tau),
                "hidden_recall": hid_fire / n_hid if n_hid else float("nan"),
                "disclosed_false_fire": dis_fire / n_dis if n_dis else float("nan"),
                "fire_rate_private": all_fire / len(priv) if priv else float("nan"),
            }
        )
    return rows


def optimistic_policy_rates(
    episodes: list[dict],
    idxs: np.ndarray,
    scores: np.ndarray,
    tau: float,
) -> dict[str, float]:
    """Simulate perfect regen on fire; unchanged otherwise."""
    n = len(idxs)
    hidden = spur = 0
    fires = 0
    for i in idxs:
        e = episodes[i]
        d = e["delta_sync"]
        if e["s_tool"] == 1 and np.isfinite(scores[i]) and scores[i] < tau:
            fires += 1
            if d == 1:
                d = 0
        if d == 1:
            hidden += 1
        elif d == -1:
            spur += 1
    return {
        "n": n,
        "hidden": hidden / n if n else float("nan"),
        "spurious": spur / n if n else float("nan"),
        "gate_fire_rate": fires / n if n else float("nan"),
    }


def pick_threshold(
    curve: list[dict[str, Any]],
    *,
    false_fire_budget: float,
    baseline_spurious: float = 0.0,
) -> dict[str, Any]:
    """Minimize optimistic hidden among tau with disclosed_false_fire <= budget."""
    ok = [
        r
        for r in curve
        if r["disclosed_false_fire"] == r["disclosed_false_fire"]
        and r["disclosed_false_fire"] <= false_fire_budget + 1e-9
    ]
    if not ok:
        ok = curve
    best = min(ok, key=lambda r: (-r.get("hidden_recall", 0), r["disclosed_false_fire"]))
    return best
