#!/usr/bin/env python3
"""Phase 8K — High-repetition stratified 8-way validation (controller frozen).

Exact locked stack (no changes):

  order H → C → O
  C: stem-prefill \"PLAN: I will \", α_C = 5
  H: decision-token-only, α_H = 1.5
  O: early-FINAL, α_O = 1.5
  target-sign, same seed within trial, frozen v_c
  no skip, no new vectors, no new objective

Question: does the corrected-controller improvement survive enough
repetitions to support a claim about all eight m*?

Report for every m* ∈ {0,1}³:
  P(hit | m*),  ΔE(m*)
and aggregate P(hit), ΔE, P(C), P(H), P(O).

Do NOT call aggregate P(hit) \"universal 8-way sync\".
Supported claim if useful: frozen three-channel causal controller improves
8-way sync and removes C-stage regression vs legacy.

  .venv/bin/python scripts/run_phase8k_8way_validation.py --reps 8
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

from scripts.sync_eq import all_m_star_masks  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8k_8way_validation.json"
MD = ROOT / "data" / "results" / "sync_phase8k_8way_validation.md"

SEED = 20260920  # independent validation seed (≠ exploratory 8K)
DEFAULT_REPS = 8

# Legacy / prior refs
REF_LEGACY_CONV = {
    "label": "legacy converted (Phase8 validation / 8E-era)",
    "P_hit": 0.11,
    "mean_delta_E_total": 0.06,
    "P_C": 0.47,
    "P_H": 0.56,
    "P_O": 0.50,
    "per_mstar_P_hit": {
        "000": 0.00,
        "001": 0.00,
        "010": 0.25,
        "011": 0.50,
        "100": 0.12,
        "101": 0.00,
        "110": 0.00,
        "111": 0.00,
    },
}
REF_8E = {"P_hit": 0.06, "mean_delta_E_total": 0.31, "P_C": 0.56, "P_H": 0.62, "P_O": 0.44}
REF_8K_EXP = {"P_hit": 0.25, "mean_delta_E_total": 0.44, "mean_delta_E_C": 0.25, "P_C": 0.69, "P_H": 0.69, "P_O": 0.56}
REF_8H = {"P_hit": 0.31, "mean_delta_E_total": 0.38, "P_C": 0.56, "P_H": 0.69, "P_O": 0.75}


def _load_8k():
    path = ROOT / "scripts" / "run_phase8k_c_gain_compose.py"
    spec = importlib.util.spec_from_file_location("phase8k", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mkey(mstar: tuple[int, int, int] | list[int]) -> str:
    return "".join(str(int(x)) for x in mstar)


def _agg(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    et = (
        [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)]
        if trials
        else []
    )
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_C": m("bit_C"),
        "P_H": m("bit_H"),
        "P_O": m("bit_O"),
        "mean_delta_E_total": m("delta_E_total"),
        "mean_delta_E_C": m("delta_E_C"),
        "mean_delta_E_H": m("delta_E_H"),
        "mean_delta_E_O": m("delta_E_O"),
        "mean_E_traj": et,
    }


def _by_mstar(trials: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mstar in all_m_star_masks():
        key = _mkey(mstar)
        sub = [t for t in trials if _mkey(t["mstar"]) == key]
        a = _agg(sub)
        out[key] = {
            **a,
            "mstar": list(mstar),
            "n_hit": int(sum(t["hit"] for t in sub)),
            "reachable": bool(a["P_hit"] > 0.0),
            "meaningful": bool(a["P_hit"] >= 0.25),  # soft bar, not universal claim
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_8k()
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()
    mstars = [tuple(m) for m in all_m_star_masks()]

    print(
        f"=== Phase 8K 8-way VALIDATION (frozen) "
        f"α_C={p8k.ALPHA_C} α_H={p8k.ALPHA_H} α_O={p8k.ALPHA_O} "
        f"order=H→C→O reps={args.reps}/m* seed={args.seed} ===",
        flush=True,
    )
    print("  NO controller changes. Stratified over all 8 m*.", flush=True)

    trials: list[dict] = []
    for mi, mstar in enumerate(mstars):
        for r in range(args.reps):
            seed = args.seed + 1000 * mi + 10 * r
            torch.manual_seed(seed)
            print(f"  m*={_mkey(mstar)} r={r}/{args.reps - 1}", flush=True)
            tr = p8k.run_sequential_trial(
                sc, loaded, mstar=mstar, dirs=dirs, seed=seed
            )
            trials.append(tr)
            print(
                f"    E:{[round(x, 2) for x in tr['E_traj']]} "
                f"hit={tr['hit']} ΔE={tr['delta_E_total']:+.2f} "
                f"ΔE_C/H/O={tr['delta_E_C']:+.2f}/{tr['delta_E_H']:+.2f}/{tr['delta_E_O']:+.2f}",
                flush=True,
            )

    agg = _agg(trials)
    by_m = _by_mstar(trials)
    n_reachable = sum(1 for v in by_m.values() if v["reachable"])
    n_meaningful = sum(1 for v in by_m.values() if v["meaningful"])

    gate = {
        "controller_frozen": True,
        "stack": {
            "order": list(p8k.ORDER),
            "alpha_C": p8k.ALPHA_C,
            "alpha_H": p8k.ALPHA_H,
            "alpha_O": p8k.ALPHA_O,
            "c_stem": p8k.C_STEM,
        },
        "hypothesis": (
            "Corrected 8K controller improves 8-way sync vs legacy and keeps ΔE_C≥0; "
            "per-m* reachability is reported without claiming universal sync."
        ),
        "beats_legacy_P_hit": bool(agg["P_hit"] > REF_LEGACY_CONV["P_hit"] + 0.05),
        "beats_legacy_delta_E": bool(
            agg["mean_delta_E_total"] > REF_LEGACY_CONV["mean_delta_E_total"] + 0.05
        ),
        "beats_8E_P_hit": bool(agg["P_hit"] > REF_8E["P_hit"] + 0.05),
        "delta_E_C_nonneg": bool(agg["mean_delta_E_C"] >= -0.05),
        "n_mstar_reachable": n_reachable,
        "n_mstar_meaningful_ge_0p25": n_meaningful,
        "universal_claim": False,  # never auto-true from aggregate alone
        "universal_supported": bool(n_meaningful == 8),
        "supported_claim": (
            "frozen three-channel causal controller improves 8-way synchronization "
            "and removes C-stage regression"
            if (
                (agg["P_hit"] > REF_LEGACY_CONV["P_hit"] + 0.05 or agg["P_hit"] > REF_8E["P_hit"] + 0.05)
                and agg["mean_delta_E_C"] >= -0.05
            )
            else "insufficient for the improved-controller claim at this sample"
        ),
    }
    gate["improved_controller_claim"] = bool(
        gate["supported_claim"].startswith("frozen")
    )
    gate["read"] = (
        "Report per-m* P(hit|m*). Do not call aggregate P(hit) universal. "
        "Universal only if every m* is meaningfully reachable."
    )

    payload = {
        "protocol": "Phase 8K high-rep stratified 8-way validation",
        "reps": args.reps,
        "seed": args.seed,
        "n_trials": len(trials),
        "controller": gate["stack"],
        "aggregate": agg,
        "by_mstar": by_m,
        "gate": gate,
        "refs": {
            "legacy_converted": REF_LEGACY_CONV,
            "phase8e": REF_8E,
            "phase8k_exploratory": REF_8K_EXP,
            "phase8h": REF_8H,
        },
        "trials": trials,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8K — High-rep stratified 8-way validation",
        "",
        r"> **Frozen controller:** $H\to C\to O$; "
        r"$C$ stem $\alpha{=}5$, $H$ decision-token $\alpha{=}1.5$, "
        r"$O$ early-FINAL $\alpha{=}1.5$. No skip. No new $v$.",
        "",
        f"reps={args.reps}/m*, n={len(trials)}, seed={args.seed}.",
        "",
        "## Aggregate",
        "",
        "| Arm | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |",
        "|-----|--------|---:|-----:|-----:|-----:|------|------|------|",
        f"| **8K validated** | {agg['P_hit']:.2f} | {agg['mean_delta_E_total']:+.2f} | "
        f"{agg['mean_delta_E_C']:+.2f} | {agg['mean_delta_E_H']:+.2f} | "
        f"{agg['mean_delta_E_O']:+.2f} | {agg['P_C']:.2f} | {agg['P_H']:.2f} | "
        f"{agg['P_O']:.2f} |",
        f"| 8K exploratory (reps=2) | {REF_8K_EXP['P_hit']:.2f} | "
        f"{REF_8K_EXP['mean_delta_E_total']:+.2f} | {REF_8K_EXP['mean_delta_E_C']:+.2f} | "
        f"— | — | {REF_8K_EXP['P_C']:.2f} | {REF_8K_EXP['P_H']:.2f} | {REF_8K_EXP['P_O']:.2f} |",
        f"| legacy converted | {REF_LEGACY_CONV['P_hit']:.2f} | "
        f"{REF_LEGACY_CONV['mean_delta_E_total']:+.2f} | — | — | — | "
        f"{REF_LEGACY_CONV['P_C']:.2f} | {REF_LEGACY_CONV['P_H']:.2f} | "
        f"{REF_LEGACY_CONV['P_O']:.2f} |",
        f"| 8E corrected | {REF_8E['P_hit']:.2f} | {REF_8E['mean_delta_E_total']:+.2f} | "
        f"— | — | — | {REF_8E['P_C']:.2f} | {REF_8E['P_H']:.2f} | {REF_8E['P_O']:.2f} |",
        "",
        "## Per $m^*$",
        "",
        "| $m^*$ | P(hit) | n_hit/n | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) | legacy P(hit) |",
        "|------:|-------:|--------:|---:|-----:|-----:|-----:|------|------|------|-------------:|",
    ]
    for mstar in all_m_star_masks():
        key = _mkey(mstar)
        g = by_m[key]
        leg = REF_LEGACY_CONV["per_mstar_P_hit"].get(key, float("nan"))
        lines.append(
            f"| {key} | {g['P_hit']:.2f} | {g['n_hit']}/{g['n']} | "
            f"{g['mean_delta_E_total']:+.2f} | {g['mean_delta_E_C']:+.2f} | "
            f"{g['mean_delta_E_H']:+.2f} | {g['mean_delta_E_O']:+.2f} | "
            f"{g['P_C']:.2f} | {g['P_H']:.2f} | {g['P_O']:.2f} | {leg:.2f} |"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- Beats legacy P(hit): **{gate['beats_legacy_P_hit']}**",
        f"- Beats 8E P(hit): **{gate['beats_8E_P_hit']}**",
        f"- $\\Delta E_C$ non-negative: **{gate['delta_E_C_nonneg']}**",
        f"- $m^*$ reachable ($P>0$): **{n_reachable}/8**",
        f"- $m^*$ meaningful ($P\\ge0.25$): **{n_meaningful}/8**",
        f"- Improved-controller claim: **{gate['improved_controller_claim']}**",
        f"- Universal 8-way claim: **{gate['universal_claim']}** "
        f"(supported only if 8/8 meaningful: {gate['universal_supported']})",
        "",
        f"Supported wording: *{gate['supported_claim']}.*",
        "",
        gate["read"],
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(OUT),
                "md": str(MD),
                "aggregate": agg,
                "n_reachable": n_reachable,
                "n_meaningful": n_meaningful,
                "gate": {k: v for k, v in gate.items() if k not in ("read", "stack")},
                "by_mstar_P_hit": {k: v["P_hit"] for k, v in by_m.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
