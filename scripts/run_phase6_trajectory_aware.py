#!/usr/bin/env python3
"""Phase 6 — Trajectory-aware gains on frozen converted actuators.

5C locked: sequential failure ≈ state-dependent cross-channel collateral
(esp. C→H), not actuator failure. Directions stay frozen.

Question:
  Can sequential sync be achieved by choosing α to trade off intended ΔM_k
  against downstream collateral, using the same v_c?

Arms (same prompts/seeds/sites/order C→H→O):
  1. fixed_α     — α_k = α0 for every stage
  2. aware_α     — α_k ∈ ALPHA_GRID maximizing
                   ΔM_k − λ Σ_{j≠k} w_j |ΔM_j|  (probe at current prior state)
  3. (optional) zero_C when collateral too high — not default

Also replicates C→H contrast (local |ΔM_C| + cross) for T diagnostic stability.

  .venv/bin/python scripts/run_phase6_trajectory_aware.py --reps 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts.sync_channel_margins import SPECS, margin_at_site, messages_for_channel, unit  # noqa: E402
from scripts.sync_eq import extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase6_trajectory_aware.json"
MD = ROOT / "data" / "results" / "sync_phase6_trajectory_aware.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260908
ALPHA0 = 1.5
ALPHA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ORDER = ("C", "H", "O")
SELECTED_MSTAR = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))
# weights: penalize H collateral most when choosing α_C (5C finding)
W = {"C": 1.0, "H": 1.5, "O": 1.0}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


def _compose(active: dict[str, tuple[float, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in CHANNELS:
        if k not in active:
            continue
        a, v = active[k]
        term = float(a) * v  # a already includes sign·|α|
        d = term if d is None else d + term
    return d


def measure_M(loaded, sc, active: dict, alpha_hook: float = 1.0) -> dict[str, float]:
    """Margins under cumulative active steers.

    active[k] = (signed_alpha, unit direction) meaning hook applies signed_alpha * dir.
    We pack as direction=sign(a)*v, alpha=|a| via make_steer_hook.
    """
    pd = _compose(active)
    out = {}
    for k in CHANNELS:
        msgs = messages_for_channel(sc, k)
        if pd is None or float(np.linalg.norm(pd)) < 1e-12:
            out[k] = float(margin_at_site(loaded, msgs, SPECS[k])["M"])
        else:
            # make_steer_hook units direction and uses abs(alpha); pass pd as direction with alpha=1
            hook = make_steer_hook(loaded, pd, float(np.linalg.norm(pd)))
            # Wait - make_steer_hook does sign(alpha)*direction with alpha=abs.
            # If we pass direction=pd and alpha=||pd||, we get sign(||pd||)*unit(pd)*||pd|| = pd. Good.
            out[k] = float(margin_at_site(loaded, msgs, SPECS[k], hook=hook)["M"])
    return out


def delta_M_for_alpha(
    loaded, sc, *, k: str, e_sign: float, v: np.ndarray, abs_alpha: float, prior: dict, base_M: dict
) -> dict[str, float]:
    """ΔM when adding e_sign * abs_alpha * v on top of prior."""
    trial = dict(prior)
    if abs_alpha > 1e-12:
        trial[k] = (float(e_sign) * float(abs_alpha), v)
    M1 = measure_M(loaded, sc, trial)
    return {j: float(M1[j] - base_M[j]) for j in CHANNELS}


def choose_alpha(
    loaded,
    sc,
    *,
    k: str,
    e_sign: float,
    v: np.ndarray,
    prior: dict,
    lam: float,
    grid: tuple[float, ...],
) -> tuple[float, dict[str, float], float]:
    """Return (abs_alpha, dM, score)."""
    base_M = measure_M(loaded, sc, prior)
    best_a, best_score, best_dM = 0.0, -1e18, {j: 0.0 for j in CHANNELS}
    for a in grid:
        dM = delta_M_for_alpha(
            loaded, sc, k=k, e_sign=e_sign, v=v, abs_alpha=a, prior=prior, base_M=base_M
        )
        # intended: e_sign * dM_k  (want positive movement in e_sign direction of M)
        intended = float(e_sign) * dM[k]
        collat = sum(float(W[j]) * abs(dM[j]) for j in CHANNELS if j != k)
        score = intended - float(lam) * collat
        if score > best_score:
            best_score = score
            best_a = float(a)
            best_dM = dM
    return best_a, best_dM, float(best_score)


def _S_from_row(row: dict) -> list[int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    plan = extract_plan(blob)
    if plan:
        c = score_plan(plan, H)
        C = int(c) if c is not None else 0
    else:
        C = 0
    return [C, H, O]


def _e(mstar, S):
    return [int(mstar[i]) - int(S[i]) for i in range(3)]


def _run_episode(sc, loaded, active: dict, seed: int):
    """active[k]=(signed_alpha, v). Tool phase: C+H packed; report: O."""
    tool_parts = []
    for k in ("C", "H"):
        if k in active:
            a, v = active[k]
            tool_parts.append(float(a) * v)
    tool_d = None if not tool_parts else sum(tool_parts)
    tool_hook = None
    if tool_d is not None and float(np.linalg.norm(tool_d)) > 1e-12:
        tool_hook = make_steer_hook(loaded, tool_d, float(np.linalg.norm(tool_d)))
    o_hook = None
    if "O" in active:
        a, v = active["O"]
        if abs(a) > 1e-12:
            o_hook = make_steer_hook(loaded, v, float(a))

    def hook_for_turn(phase: str, _t: int):
        if phase == "tool":
            return tool_hook
        if phase == "report":
            return o_hook
        return None

    row = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
        capture_activations=False,
    )
    if tool_hook is not None:
        tool_hook.remove()
    if o_hook is not None:
        o_hook.remove()
    return {"S": _S_from_row(row)}


def replicate_CH(loaded, sc, vc, alpha: float, reps: int) -> dict[str, Any]:
    """Replicate C→H contrast: |ΔM_C| and cross from h0 vs after H."""
    rows = []
    for r in range(reps):
        for e_sign in (+1.0, -1.0):
            base0 = measure_M(loaded, sc, {})
            d0 = delta_M_for_alpha(
                loaded, sc, k="C", e_sign=e_sign, v=vc["C"], abs_alpha=alpha, prior={}, base_M=base0
            )
            prior_H = {"H": (-alpha, vc["H"])}  # H suppress operating point
            baseH = measure_M(loaded, sc, prior_H)
            dH = delta_M_for_alpha(
                loaded, sc, k="C", e_sign=e_sign, v=vc["C"], abs_alpha=alpha, prior=prior_H, base_M=baseH
            )
            rows.append(
                {
                    "e_sign": e_sign,
                    "abs_dM_C_h0": abs(d0["C"]),
                    "abs_dM_C_hH": abs(dH["C"]),
                    "cross_h0": float(np.mean([abs(d0[j]) for j in ("H", "O")])),
                    "cross_hH": float(np.mean([abs(dH[j]) for j in ("H", "O")])),
                    "dM_H_from_C_h0": d0["H"],
                    "dM_H_from_C_hH": dH["H"],
                }
            )
    return {
        "retention": float(np.mean([r["abs_dM_C_hH"] for r in rows]) / max(np.mean([r["abs_dM_C_h0"] for r in rows]), 1e-9)),
        "cross_ratio": float(np.mean([r["cross_hH"] for r in rows]) / max(np.mean([r["cross_h0"] for r in rows]), 1e-9)),
        "mean_abs_dM_H_collateral_h0": float(np.mean([abs(r["dM_H_from_C_h0"]) for r in rows])),
        "mean_abs_dM_H_collateral_hH": float(np.mean([abs(r["dM_H_from_C_hH"]) for r in rows])),
        "rows": rows,
    }


def run_arm(
    sc,
    loaded,
    *,
    mode: str,
    mstar: tuple[int, int, int],
    vc: dict,
    alpha0: float,
    lam: float,
    grid: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    base = _run_episode(sc, loaded, {}, seed)
    S = base["S"]
    E_traj = [float(sync_error_norm(_e(mstar, S)))]
    active: dict[str, tuple[float, np.ndarray]] = {}
    stages = []
    for si, k in enumerate(ORDER):
        e = _e(mstar, S)
        ek = int(e[CH_IDX[k]])
        if ek != 0 and k not in active:
            e_sign = float(np.sign(ek))  # +1 need increase M toward 1; -1 need decrease
            # Note: M increase ↔ channel=1 side; e = m*-S so e=+1 means need S↑ → increase M
            if mode == "fixed":
                a = float(alpha0)
                base_M = measure_M(loaded, sc, active)
                dM = delta_M_for_alpha(
                    loaded, sc, k=k, e_sign=e_sign, v=vc[k], abs_alpha=a, prior=active, base_M=base_M
                )
                score = float(e_sign) * dM[k] - lam * sum(W[j] * abs(dM[j]) for j in CHANNELS if j != k)
            else:
                a, dM, score = choose_alpha(
                    loaded, sc, k=k, e_sign=e_sign, v=vc[k], prior=active, lam=lam, grid=grid
                )
            active[k] = (e_sign * a, vc[k])
            stages.append(
                {
                    "k": k,
                    "e": ek,
                    "alpha": a,
                    "dM": dM,
                    "intended": float(e_sign) * dM[k],
                    "collateral": float(sum(abs(dM[j]) for j in CHANNELS if j != k)),
                    "score": score,
                    "net": float(e_sign) * dM[k] - sum(abs(dM[j]) for j in CHANNELS if j != k),
                }
            )
        out = _run_episode(sc, loaded, active, seed + 100 * (si + 1))
        S = out["S"]
        E_traj.append(float(sync_error_norm(_e(mstar, S))))
    return {
        "mstar": list(mstar),
        "mode": mode,
        "E_traj": E_traj,
        "delta_E_total": float(E_traj[0] - E_traj[-1]),
        "delta_E_C": float(E_traj[0] - E_traj[1]),
        "delta_E_H": float(E_traj[1] - E_traj[2]),
        "delta_E_O": float(E_traj[2] - E_traj[3]),
        "hit": int(S == list(mstar)),
        "S_final": S,
        "stages": stages,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--alpha0", type=float, default=ALPHA0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    vc = _load_vc()

    print("=== replicate C→H contrast ===", flush=True)
    ch_rep = replicate_CH(loaded, sc, vc, args.alpha0, max(2, args.reps // 2))
    print(
        f"  retention={ch_rep['retention']:.2f} cross_ratio={ch_rep['cross_ratio']:.2f} "
        f"|dM_H| collat h0/hH={ch_rep['mean_abs_dM_H_collateral_h0']:.2f}/"
        f"{ch_rep['mean_abs_dM_H_collateral_hH']:.2f}",
        flush=True,
    )

    results: dict[str, list] = {"fixed": [], "aware": []}
    for mode in ("fixed", "aware"):
        print(f"=== sequential {mode} ===", flush=True)
        for mi, mstar in enumerate(SELECTED_MSTAR):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r
                print(f"  {mode} m*={mstar} r={r}", flush=True)
                torch.manual_seed(seed)
                tr = run_arm(
                    sc,
                    loaded,
                    mode=mode,
                    mstar=mstar,
                    vc=vc,
                    alpha0=args.alpha0,
                    lam=args.lam if mode == "aware" else 0.0,
                    grid=ALPHA_GRID,
                    seed=seed,
                )
                # for fixed, still record net using lam for secondary metric
                for st in tr["stages"]:
                    st["net_lam"] = st["intended"] - args.lam * st["collateral"]
                results[mode].append(tr)
                print(
                    f"    E:{tr['E_traj']} hit={tr['hit']} "
                    f"alphas={[s['alpha'] for s in tr['stages']]}",
                    flush=True,
                )

    def agg(trials: list[dict]) -> dict[str, Any]:
        c_stages = [s for t in trials for s in t["stages"] if s["k"] == "C"]
        return {
            "n": len(trials),
            "P_hit": float(np.mean([t["hit"] for t in trials])),
            "mean_delta_E_total": float(np.mean([t["delta_E_total"] for t in trials])),
            "mean_delta_E_C": float(np.mean([t["delta_E_C"] for t in trials])),
            "mean_delta_E_H": float(np.mean([t["delta_E_H"] for t in trials])),
            "mean_delta_E_O": float(np.mean([t["delta_E_O"] for t in trials])),
            "mean_E_traj": [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)],
            "mean_alpha_C": float(np.mean([s["alpha"] for s in c_stages])) if c_stages else float("nan"),
            "mean_C_intended": float(np.mean([s["intended"] for s in c_stages])) if c_stages else float("nan"),
            "mean_C_collateral": float(np.mean([s["collateral"] for s in c_stages])) if c_stages else float("nan"),
            "mean_C_net": float(np.mean([s["net"] for s in c_stages])) if c_stages else float("nan"),
        }

    ag = {m: agg(results[m]) for m in results}
    gate = {
        "question": "Can trajectory-aware α restore sequential sync with frozen v_c?",
        "CH_replication": {
            "retention": ch_rep["retention"],
            "cross_ratio": ch_rep["cross_ratio"],
            "cross_worsens": bool(ch_rep["cross_ratio"] > 1.15),
        },
        "aware_beats_fixed_delta_E": bool(
            ag["aware"]["mean_delta_E_total"] > ag["fixed"]["mean_delta_E_total"] + 0.05
        ),
        "aware_beats_fixed_hit": bool(ag["aware"]["P_hit"] >= ag["fixed"]["P_hit"]),
        "aware_reduces_C_collateral": bool(
            ag["aware"]["mean_C_collateral"] < ag["fixed"]["mean_C_collateral"] - 0.05
        ),
        "aware_improves_C_net": bool(ag["aware"]["mean_C_net"] > ag["fixed"]["mean_C_net"] + 0.05),
        "eight_way": "CLOSED",
    }

    payload = {
        "protocol": "Phase 6 trajectory-aware α",
        "alpha0": args.alpha0,
        "lam": args.lam,
        "grid": list(ALPHA_GRID),
        "CH_replication": ch_rep,
        "agg": ag,
        "trials": results,
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        r"# Phase 6 — Trajectory-aware gains (frozen \(v_c\))",
        "",
        "> Can sequential sync be restored by choosing α to penalize transition collateral?",
        "",
        f"α0={args.alpha0}, λ={args.lam}, grid={list(ALPHA_GRID)}, reps={args.reps}",
        "",
        "## C→H contrast replication",
        "",
        f"- retention: **{ch_rep['retention']:.2f}**",
        f"- cross ratio: **{ch_rep['cross_ratio']:.2f}**",
        f"- \\|ΔM_H\\| from C at h0 / h_H: "
        f"{ch_rep['mean_abs_dM_H_collateral_h0']:.2f} / {ch_rep['mean_abs_dM_H_collateral_hH']:.2f}",
        "",
        "## Sequential fixed vs aware",
        "",
        "| Arm | P(hit) | ΔE tot | ΔE_C | ΔE_H | ΔE_O | E0→E3 | ᾱ_C | C intended | C collat | C net |",
        "|-----|--------|--------|------|------|------|-------|------|------------|----------|-------|",
    ]
    for m in ("fixed", "aware"):
        a = ag[m]
        et = a["mean_E_traj"]
        lines.append(
            f"| {m} | {a['P_hit']:.2f} | {a['mean_delta_E_total']:+.2f} | "
            f"{a['mean_delta_E_C']:+.2f} | {a['mean_delta_E_H']:+.2f} | {a['mean_delta_E_O']:+.2f} | "
            f"{et[0]:.2f}→{et[1]:.2f}→{et[2]:.2f}→{et[3]:.2f} | "
            f"{a['mean_alpha_C']:.2f} | {a['mean_C_intended']:+.2f} | "
            f"{a['mean_C_collateral']:.2f} | {a['mean_C_net']:+.2f} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- Aware ΔE > fixed: **{gate['aware_beats_fixed_delta_E']}**",
        f"- Aware hit ≥ fixed: **{gate['aware_beats_fixed_hit']}**",
        f"- Aware reduces C collateral: **{gate['aware_reduces_C_collateral']}**",
        f"- Aware improves C net: **{gate['aware_improves_C_net']}**",
        f"- 8-way: **CLOSED**",
        "",
        r"Directions frozen. No new \(v\).",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "agg": ag}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
