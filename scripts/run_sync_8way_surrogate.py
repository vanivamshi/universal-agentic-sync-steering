#!/usr/bin/env python3
"""Diagnostic: continuous 8-way surrogate L_8way on frozen directions.

Channel-wise ΔM can look good while joint discrete hit is weak. Test whether
a target-signed soft-margin energy separates v_c from random/none better than
P(hit).

Convention: M_k > 0 ⇒ channel bit = 1.

  tilde M_k = (2 m*_k − 1) M_k     # larger ⇒ more correct for target
  L_8way    = Σ_k log(1 + exp(−β tilde M_k))
  ΔL        = L_after − L_before
  primary   = E[−ΔL]               # larger ⇒ more joint-target improvement

Also report corr(−ΔL, ΔE) and usual hit / ΔE.

Frozen: v_c, v_p, random construction, sites, α, C→H→O. No new v. No policy.

Fast diagnostic (~5–10× shorter than full sequential):
  .venv/bin/python scripts/run_sync_8way_surrogate.py --fast --reps 2 \\
      --arms none,converted,random

Full sequential (slow; 4 episodes/trial):
  .venv/bin/python scripts/run_sync_8way_surrogate.py --reps 4
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

from scripts.sync_channel_control import CHANNEL_V, ChannelBank  # noqa: E402
from scripts.sync_channel_margins import SPECS, margin_at_site, messages_for_channel, unit  # noqa: E402
from scripts.sync_eq import all_m_star_masks, extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_8way_surrogate.json"
MD = ROOT / "data" / "results" / "sync_8way_surrogate.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260910
ALPHA = 1.5
BETA = 1.0
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ORDER = ("C", "H", "O")
ARMS = ("none", "predictive", "converted", "random")


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_banks(seed: int) -> dict[str, dict[str, np.ndarray] | None]:
    bank = ChannelBank.load(CHANNEL_V)
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
        "none": None,
        "converted": {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)},
        "predictive": {ch: unit(bank.V[i]) for i, ch in enumerate(CHANNELS)},
        "random": {ch: unit(rng.standard_normal(Vc.shape[1])) for ch in CHANNELS},
    }


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


def _compose(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in CHANNELS:
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return d


_MSG_CACHE: dict[str, Any] = {}


def _msgs(sc, k: str):
    if k not in _MSG_CACHE:
        _MSG_CACHE[k] = messages_for_channel(sc, k)
    return _MSG_CACHE[k]


def measure_M(loaded, sc, *, active: dict, alpha: float) -> dict[str, float]:
    """M under cumulative steers. active[k]=(±1, v); hook scale = α."""
    pd = _compose(active)
    out: dict[str, float] = {}
    for k in CHANNELS:
        msgs = _msgs(sc, k)
        if pd is None:
            out[k] = float(margin_at_site(loaded, msgs, SPECS[k])["M"])
        else:
            if k == "O" and "O" in active and ("C" in active or "H" in active):
                ek, vk = active["O"]
                hook = make_steer_hook(loaded, vk, alpha * float(ek))
            elif k in ("C", "H"):
                tool = None
                for j in ("C", "H"):
                    if j not in active:
                        continue
                    ej, vj = active[j]
                    term = float(ej) * vj
                    tool = term if tool is None else tool + term
                if tool is None:
                    out[k] = float(margin_at_site(loaded, msgs, SPECS[k])["M"])
                    continue
                hook = make_steer_hook(loaded, tool, alpha)
            else:
                hook = make_steer_hook(loaded, pd, alpha)
            out[k] = float(margin_at_site(loaded, msgs, SPECS[k], hook=hook)["M"])
    return out


def softplus(x: float) -> float:
    # stable log(1+e^x)
    if x > 20:
        return float(x)
    if x < -20:
        return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def L_8way(M: dict[str, float], mstar: tuple[int, int, int], beta: float) -> float:
    total = 0.0
    for i, k in enumerate(CHANNELS):
        sign = 2 * int(mstar[i]) - 1  # +1 if m*=1, -1 if m*=0
        tilde = sign * float(M[k])
        total += softplus(-float(beta) * tilde)
    return float(total)


def Phi_soft(M: dict[str, float], mstar: tuple[int, int, int], beta: float) -> float:
    """Secondary potential Σ log(1+e^{β tilde M}) — larger = better."""
    total = 0.0
    for i, k in enumerate(CHANNELS):
        sign = 2 * int(mstar[i]) - 1
        tilde = sign * float(M[k])
        total += softplus(float(beta) * tilde)
    return float(total)


def _compose_tool(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in ("C", "H"):
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return d


def _run_episode(sc, loaded, *, active: dict, alpha: float, seed: int) -> dict[str, Any]:
    tool_d = _compose_tool(active)
    tool_hook = make_steer_hook(loaded, tool_d, alpha) if tool_d is not None else None
    o_hook = None
    if "O" in active:
        ek, vk = active["O"]
        o_hook = make_steer_hook(loaded, vk, alpha * float(ek))

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


def _active_from_error(
    mstar: tuple[int, int, int],
    S: list[int],
    dirs: dict[str, np.ndarray] | None,
    arm: str,
) -> tuple[dict[str, tuple[int, np.ndarray]], int]:
    """Pack all needed steers from one observed S (one-shot joint control)."""
    active: dict[str, tuple[int, np.ndarray]] = {}
    if arm == "none" or dirs is None:
        return active, 0
    e = _e(mstar, S)
    for k in CHANNELS:
        if e[CH_IDX[k]] != 0:
            active[k] = (int(e[CH_IDX[k]]), dirs[k])
    return active, len(active)


def run_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray] | None,
    alpha: float,
    beta: float,
    seed: int,
    arm: str,
    fast: bool = False,
    margins_only: bool = False,
) -> dict[str, Any]:
    """Full: S0 + 3 sequential reobserve stages (slow).

    fast: S0 free-run → pack all steers from e(S0) → optional one steered episode.
          Margins M0/M1 under {{}} vs full active. ~2–4× fewer episodes.
    margins_only: skip steered behavioral episode (ΔE/hit from free S0 only as
          E0 with S_final=S0 so ΔE=0); continuous −ΔL still valid. Fastest.
    """
    base = _run_episode(sc, loaded, active={}, alpha=alpha, seed=seed)
    S = base["S"]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    active: dict[str, tuple[int, np.ndarray]] = {}
    n_interv = 0
    mode = "sequential"

    M0 = measure_M(loaded, sc, active={}, alpha=alpha)
    L0 = L_8way(M0, mstar, beta)
    Phi0 = Phi_soft(M0, mstar, beta)

    if fast or margins_only:
        mode = "margins_only" if margins_only else "fast_oneshot"
        active, n_interv = _active_from_error(mstar, S, dirs, arm)
        if not margins_only and arm != "none" and active:
            out = _run_episode(sc, loaded, active=active, alpha=alpha, seed=seed + 100)
            S = out["S"]
            traj_E.append(float(sync_error_norm(_e(mstar, S))))
        else:
            traj_E.append(traj_E[0])
    else:
        for si, k in enumerate(ORDER):
            e = _e(mstar, S)
            if arm != "none" and dirs is not None and e[CH_IDX[k]] != 0 and k not in active:
                active[k] = (int(e[CH_IDX[k]]), dirs[k])
                n_interv += 1
            out = _run_episode(sc, loaded, active=active, alpha=alpha, seed=seed + 100 * (si + 1))
            S = out["S"]
            traj_E.append(float(sync_error_norm(_e(mstar, S))))

    M1 = measure_M(loaded, sc, active=active, alpha=alpha)
    L1 = L_8way(M1, mstar, beta)
    Phi1 = Phi_soft(M1, mstar, beta)

    dL = float(L1 - L0)
    dPhi = float(Phi1 - Phi0)
    dE = float(traj_E[0] - traj_E[-1])
    tilde0 = {k: (2 * int(mstar[CH_IDX[k]]) - 1) * M0[k] for k in CHANNELS}
    tilde1 = {k: (2 * int(mstar[CH_IDX[k]]) - 1) * M1[k] for k in CHANNELS}
    d_tilde = {k: float(tilde1[k] - tilde0[k]) for k in CHANNELS}

    return {
        "arm": arm,
        "mode": mode,
        "mstar": list(mstar),
        "S0": list(base["S"]),
        "S_final": list(S),
        "E_traj": traj_E,
        "delta_E": dE,
        "hit": int(list(S) == list(mstar)),
        "M0": M0,
        "M1": M1,
        "tilde_M0": tilde0,
        "tilde_M1": tilde1,
        "delta_tilde_M": d_tilde,
        "L0": L0,
        "L1": L1,
        "delta_L": dL,
        "neg_delta_L": float(-dL),
        "Phi0": Phi0,
        "Phi1": Phi1,
        "delta_Phi": dPhi,
        "n_interventions": n_interv,
        "active": {k: int(v[0]) for k, v in active.items()},
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _agg(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    neg_dL = [t["neg_delta_L"] for t in trials]
    dE = [t["delta_E"] for t in trials]
    return {
        "n": len(trials),
        "mean_neg_delta_L": m("neg_delta_L"),
        "mean_delta_L": m("delta_L"),
        "mean_delta_Phi": m("delta_Phi"),
        "mean_delta_E": m("delta_E"),
        "P_hit": m("hit"),
        "corr_neg_dL_delta_E": _corr(neg_dL, dE),
        "frac_L_improved": float(np.mean([t["delta_L"] < 0 for t in trials])) if trials else float("nan"),
        "mean_delta_tilde_C": float(np.mean([t["delta_tilde_M"]["C"] for t in trials])) if trials else float("nan"),
        "mean_delta_tilde_H": float(np.mean([t["delta_tilde_M"]["H"] for t in trials])) if trials else float("nan"),
        "mean_delta_tilde_O": float(np.mean([t["delta_tilde_M"]["O"] for t in trials])) if trials else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--arms",
        default="none,converted,random",
        help="comma-separated; drop predictive for speed (add back if needed)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="one-shot steers from S0 + ≤2 episodes/trial (recommended)",
    )
    ap.add_argument(
        "--margins-only",
        action="store_true",
        help="fastest: free-run + M0/M1 only (ΔE/hit not meaningful)",
    )
    args = ap.parse_args()
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if args.margins_only:
        args.fast = True

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    banks = _load_banks(args.seed)
    mstars = [tuple(m) for m in all_m_star_masks()]

    all_trials: dict[str, list[dict]] = {a: [] for a in arms}
    mode = "margins_only" if args.margins_only else ("fast" if args.fast else "sequential")
    print(
        f"=== 8-way surrogate L β={args.beta} α={args.alpha} reps={args.reps} "
        f"mode={mode} arms={arms} ===",
        flush=True,
    )

    for arm in arms:
        dirs = banks[arm]
        for mi, mstar in enumerate(mstars):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r + (hash(arm) % 97)
                torch.manual_seed(seed)
                print(f"  {arm} m*={mstar} r={r}", flush=True)
                tr = run_trial(
                    sc,
                    loaded,
                    mstar=mstar,
                    dirs=dirs,
                    alpha=args.alpha,
                    beta=args.beta,
                    seed=seed,
                    arm=arm,
                    fast=args.fast,
                    margins_only=args.margins_only,
                )
                all_trials[arm].append(tr)
                print(
                    f"    −ΔL={tr['neg_delta_L']:+.3f} ΔE={tr['delta_E']:+.1f} "
                    f"hit={tr['hit']} L:{tr['L0']:.2f}→{tr['L1']:.2f} "
                    f"d~M={{C:{tr['delta_tilde_M']['C']:+.2f},"
                    f"H:{tr['delta_tilde_M']['H']:+.2f},"
                    f"O:{tr['delta_tilde_M']['O']:+.2f}}}",
                    flush=True,
                )

    aggs = {a: _agg(all_trials[a]) for a in arms}
    # pooled corr over steered arms
    steered = [t for a in arms if a != "none" for t in all_trials[a]]
    pooled_corr = _corr([t["neg_delta_L"] for t in steered], [t["delta_E"] for t in steered])
    all_corr = _corr(
        [t["neg_delta_L"] for a in arms for t in all_trials[a]],
        [t["delta_E"] for a in arms for t in all_trials[a]],
    )

    pc = aggs.get("converted")
    pr = aggs.get("random")
    pn = aggs.get("none")
    pp = aggs.get("predictive")

    gate: dict[str, Any] = {
        "primary": "E[-ΔL_8way]",
        "secondary": "corr(-ΔL, ΔE), P(hit), ΔE",
        "mode": mode,
        "pooled_corr_neg_dL_delta_E_steered": pooled_corr,
        "pooled_corr_all": all_corr,
    }
    if pc and pr:
        gate["vc_beats_random_neg_dL"] = bool(pc["mean_neg_delta_L"] > pr["mean_neg_delta_L"])
        gate["vc_beats_random_hit"] = bool(pc["P_hit"] > pr["P_hit"])
        gate["vc_beats_random_delta_E"] = bool(pc["mean_delta_E"] > pr["mean_delta_E"])
        gate["surrogate_advantage_stronger_than_hit"] = bool(
            gate["vc_beats_random_neg_dL"] and not gate["vc_beats_random_hit"]
        )
    if pc and pn:
        gate["vc_beats_none_neg_dL"] = bool(pc["mean_neg_delta_L"] > pn["mean_neg_delta_L"])
    if pc and pp:
        gate["vc_beats_predictive_neg_dL"] = bool(pc["mean_neg_delta_L"] > pp["mean_neg_delta_L"])
    if pc and pr and pn and pp:
        gate["ranking_vc_ge_vp_ge_rand_none_on_neg_dL"] = bool(
            pc["mean_neg_delta_L"] >= pp["mean_neg_delta_L"]
            and pp["mean_neg_delta_L"] >= max(pr["mean_neg_delta_L"], pn["mean_neg_delta_L"]) - 1e-9
        )
    gate["corr_strong"] = bool((not np.isnan(pooled_corr)) and abs(pooled_corr) >= 0.3)
    gate["interpretation"] = (
        "If vc >> random on E[-ΔL] but not on hit: discrete eval understates causal joint control. "
        "If both weak: problem is downstream of converted representation."
    )

    payload = {
        "protocol": "8-way soft-margin surrogate L_8way",
        "formula": "L = Σ_k log(1+exp(-β(2m*_k-1)M_k))",
        "mode": mode,
        "alpha": args.alpha,
        "beta": args.beta,
        "reps": args.reps,
        "seed": args.seed,
        "arms": list(arms),
        "agg": aggs,
        "trials": {a: all_trials[a] for a in arms},
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 8-way surrogate — $\\mathcal{L}_{8\\mathrm{way}}$ diagnostic",
        "",
        r"> $\mathcal{L}=\sum_k\log(1+e^{-\beta(2m_k^*-1)M_k})$. Primary: $\mathbb{E}[-\Delta\mathcal{L}]$.",
        "",
        f"mode=**{mode}**, β={args.beta}, α={args.alpha}, reps={args.reps}/m*, arms={list(arms)}.",
        "",
        "## Aggregate",
        "",
        "| Arm | n | E[−ΔL] | frac L↓ | ΔE | P(hit) | corr(−ΔL,ΔE) | Δ~M_C | Δ~M_H | Δ~M_O |",
        "|-----|---|--------|---------|-----|--------|--------------|-------|-------|-------|",
    ]
    for a in arms:
        g = aggs[a]
        lines.append(
            f"| {a} | {g['n']} | {g['mean_neg_delta_L']:+.3f} | {g['frac_L_improved']:.2f} | "
            f"{g['mean_delta_E']:+.3f} | {g['P_hit']:.3f} | {g['corr_neg_dL_delta_E']:+.3f} | "
            f"{g['mean_delta_tilde_C']:+.2f} | {g['mean_delta_tilde_H']:+.2f} | "
            f"{g['mean_delta_tilde_O']:+.2f} |"
        )
    lines += [
        "",
        f"Pooled corr(−ΔL, ΔE) steered arms: **{pooled_corr:+.3f}**",
        f"Pooled corr all arms: **{all_corr:+.3f}**",
        "",
        "## Gate",
        "",
        f"- $v_c$ beats random on E[−ΔL]: **{gate.get('vc_beats_random_neg_dL', 'n/a')}**",
        f"- $v_c$ beats none on E[−ΔL]: **{gate.get('vc_beats_none_neg_dL', 'n/a')}**",
        f"- $v_c$ beats predictive on E[−ΔL]: **{gate.get('vc_beats_predictive_neg_dL', 'n/a')}**",
        f"- $v_c$ beats random on hit: **{gate.get('vc_beats_random_hit', 'n/a')}**",
        f"- Surrogate advantage stronger than hit: **{gate.get('surrogate_advantage_stronger_than_hit', 'n/a')}**",
        f"- corr strong (|r|≥0.3 steered): **{gate.get('corr_strong', 'n/a')}**",
        "",
        gate["interpretation"],
        "",
        "Directions frozen. No product objective. No policy.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "agg": aggs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
