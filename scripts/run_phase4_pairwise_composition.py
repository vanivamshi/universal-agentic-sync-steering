#!/usr/bin/env python3
"""Phase 4 — Pairwise composition matrix (no new discovery).

Hypothesis:
  If individual predictive→causal conversions are valid, their converted
  components should compose with limited cross-channel interference.

Pairs: C+H, H+O, C+O
Arms:  predictive (v_p) vs converted (v_c) vs random
Sites: frozen channel decision sites from sync_channel_margins.

Sparse compose: d = e_i α v_i + e_j α v_j  (only mismatched channels;
for the mechanistic matrix we use e∈{+1,-1} on the pair, 0 on the third).

Measure ΔM_C, ΔM_H, ΔM_O and diagonal dominance of J.

  .venv/bin/python scripts/run_phase4_pairwise_composition.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts.sync_channel_control import CHANNEL_V, ChannelBank  # noqa: E402
from scripts.sync_channel_margins import (  # noqa: E402
    SPECS,
    margin_at_site,
    messages_for_channel,
    safe_steer_dir,
    unit,
)
from scripts.sync_h_decision import LAYER_DEFAULT, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase4_pairwise_composition.json"
MD = ROOT / "data" / "results" / "sync_phase4_pairwise_composition.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
SEED = 20260904
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
PAIRS = (("C", "H"), ("H", "O"), ("C", "O"))
PAIR_Q = {
    ("C", "H"): "Can earlier plan-state intervention compose with downstream tool decision?",
    ("H", "O"): "Does controlling execution preserve/control subsequent disclosure? (H→O)",
    ("C", "O"): "Can non-adjacent channels compose without H mediation?",
}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_dirs(seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Return {arm: {C,H,O: v}} for predictive, converted, random."""
    bank = ChannelBank.load(CHANNEL_V)
    blob = json.loads(VC_PATH.read_text())
    Vc = np.asarray(blob["V_c"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = {
        "predictive": {ch: unit(bank.V[i]) for i, ch in enumerate(CHANNELS)},
        "converted": {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)},
        "random": {ch: unit(rng.standard_normal(Vc.shape[1])) for ch in CHANNELS},
    }
    return out


def _compose_dir(
    dirs: dict[str, np.ndarray],
    e: dict[str, float],
) -> np.ndarray:
    """Sparse d = Σ_k e_k v_k (skip e_k=0)."""
    d = None
    for k, ek in e.items():
        if abs(ek) < 1e-12:
            continue
        term = float(ek) * dirs[k]
        d = term if d is None else d + term
    if d is None:
        raise ValueError("empty compose")
    return safe_steer_dir(d, dirs[next(iter(dirs))])


def _delta_M_all(
    loaded,
    msgs: dict[str, list],
    v: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    """Apply +α v at each channel site separately; report ΔM_k (site-local)."""
    out: dict[str, float] = {}
    for k in CHANNELS:
        base = margin_at_site(loaded, msgs[k], SPECS[k])
        hook = make_steer_hook(loaded, v, alpha)
        s = margin_at_site(loaded, msgs[k], SPECS[k], hook=hook)
        out[f"dM_{k}"] = float(s["M"] - base["M"])
        out[f"M0_{k}"] = float(base["M"])
    return out


def _diag_dominance(J: np.ndarray) -> dict[str, float]:
    """DD_k = |J_kk| / sum_{j≠k}|J_kj|  (rows = affected)."""
    dd = {}
    for i, k in enumerate(CHANNELS):
        diag = abs(J[i, i])
        off = float(np.sum(np.abs(J[i, :])) - diag)
        dd[k] = float(diag / off) if off > 1e-9 else float("inf")
    return dd


def _classify_dd(dd: dict[str, float]) -> str:
    vals = [dd[k] for k in CHANNELS if np.isfinite(dd[k])]
    if not vals:
        return "unknown"
    med = float(np.median(vals))
    if med >= 2.0:
        return "independent_composition"
    if med >= 0.5:
        return "coupled_composition"
    return "unstable_composition"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not VC_PATH.is_file():
        raise SystemExit(f"missing {VC_PATH}; run run_phase4_channel_conversion_matrix.py first")

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    msgs = {k: messages_for_channel(sc, k) for k in CHANNELS}
    banks = _load_dirs(args.seed)

    # --- Full 3×3 J per arm (single-channel steers) ---
    print("=== single-channel Jacobians ===", flush=True)
    jacobians: dict[str, Any] = {}
    for arm, dirs in banks.items():
        J = np.zeros((3, 3), dtype=np.float64)
        for j, kj in enumerate(CHANNELS):
            for i, ki in enumerate(CHANNELS):
                base = margin_at_site(loaded, msgs[ki], SPECS[ki])
                hook = make_steer_hook(loaded, dirs[kj], args.alpha)
                s = margin_at_site(loaded, msgs[ki], SPECS[ki], hook=hook)
                J[i, j] = float(s["M"] - base["M"]) / args.alpha
        dd = _diag_dominance(J)
        jacobians[arm] = {
            "J": J.tolist(),
            "diagonal_dominance": dd,
            "composition_class": _classify_dd(dd),
        }
        print(f"  {arm}: class={jacobians[arm]['composition_class']} DD={dd}", flush=True)

    # --- Pairwise sparse compositions ---
    print("=== pairwise compositions ===", flush=True)
    rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        a, b = pair
        third = next(ch for ch in CHANNELS if ch not in pair)
        for arm, dirs in banks.items():
            for e_a, e_b, tag in (
                (+1.0, +1.0, "both_up"),
                (-1.0, -1.0, "both_down"),
                (+1.0, -1.0, "a_up_b_down"),
                (-1.0, +1.0, "a_down_b_up"),
            ):
                e = {a: e_a, b: e_b, third: 0.0}
                d = _compose_dir(dirs, e)
                # unit for steer magnitude parity with singles: scale so ||d||~1
                d = unit(d)
                deltas = _delta_M_all(loaded, msgs, d, alpha=args.alpha)
                intended = 0.5 * (abs(deltas[f"dM_{a}"]) + abs(deltas[f"dM_{b}"]))
                collateral = abs(deltas[f"dM_{third}"])
                # sign agreement: does ΔM_k move with e_k?
                agree_a = float(np.sign(deltas[f"dM_{a}"]) == np.sign(e_a)) if abs(deltas[f"dM_{a}"]) > 0.05 else float("nan")
                agree_b = float(np.sign(deltas[f"dM_{b}"]) == np.sign(e_b)) if abs(deltas[f"dM_{b}"]) > 0.05 else float("nan")
                row = {
                    "pair": f"{a}+{b}",
                    "question": PAIR_Q[pair],
                    "arm": arm,
                    "tag": tag,
                    "e": e,
                    "mean_abs_dM_intended": float(intended),
                    "abs_dM_collateral": float(collateral),
                    "sign_agree_a": agree_a,
                    "sign_agree_b": agree_b,
                    **deltas,
                }
                rows.append(row)
                print(
                    f"  {a}+{b} {arm} {tag}: "
                    f"|ΔM|_int={intended:.2f} collat={collateral:.2f}",
                    flush=True,
                )

    # Aggregate: converted > predictive > random on intended |ΔM|
    summary: dict[str, Any] = {}
    for pair in PAIRS:
        key = f"{pair[0]}+{pair[1]}"
        by_arm = {}
        for arm in ("predictive", "converted", "random"):
            sub = [r for r in rows if r["pair"] == key and r["arm"] == arm]
            by_arm[arm] = {
                "mean_abs_dM_intended": float(np.mean([r["mean_abs_dM_intended"] for r in sub])),
                "mean_abs_dM_collateral": float(np.mean([r["abs_dM_collateral"] for r in sub])),
                "mean_sign_agree": float(
                    np.nanmean([r["sign_agree_a"] for r in sub] + [r["sign_agree_b"] for r in sub])
                ),
            }
        ranking_ok = (
            by_arm["converted"]["mean_abs_dM_intended"]
            >= by_arm["predictive"]["mean_abs_dM_intended"]
            >= by_arm["random"]["mean_abs_dM_intended"] - 0.05  # tiny slack for noise
        )
        summary[key] = {
            "by_arm": by_arm,
            "converted_ge_predictive_ge_random": bool(
                by_arm["converted"]["mean_abs_dM_intended"]
                >= by_arm["predictive"]["mean_abs_dM_intended"]
                and by_arm["predictive"]["mean_abs_dM_intended"]
                >= by_arm["random"]["mean_abs_dM_intended"] - 0.05
            ),
            "converted_beats_predictive": bool(
                by_arm["converted"]["mean_abs_dM_intended"]
                > by_arm["predictive"]["mean_abs_dM_intended"]
            ),
            "question": PAIR_Q[pair],
        }

    n_rank = sum(1 for s in summary.values() if s["converted_beats_predictive"])
    gate = {
        "hypothesis": (
            "Converted components compose with limited cross-channel interference; "
            "converted composition > predictive composition > random."
        ),
        "n_pairs_converted_beats_predictive": n_rank,
        "jacobian_class_converted": jacobians["converted"]["composition_class"],
        "diagonal_dominance_converted": jacobians["converted"]["diagonal_dominance"],
        "ready_for_sequential": n_rank >= 2
        and jacobians["converted"]["composition_class"] != "unstable_composition",
        "ready_for_8way": False,
        "next_if_pass": "sequential C→H→O closed-loop with e recomputed each stage",
        "next_if_coupled": "characterize coupling; do not jump to 8-way",
    }

    payload = {
        "protocol": "Phase 4 pairwise composition",
        "alpha": args.alpha,
        "layer": LAYER_DEFAULT,
        "pairs": [f"{a}+{b}" for a, b in PAIRS],
        "arms": ["predictive", "converted", "random"],
        "jacobians": jacobians,
        "rows": rows,
        "summary": summary,
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 4 — Pairwise composition matrix",
        "",
        r"> **Hypothesis:** If individual predictive→causal conversions are valid,",
        r"> their converted components should compose with limited cross-channel interference.",
        "",
        "Controls: **converted** vs **predictive** vs **random**. Sites frozen.",
        "",
        "## Jacobian diagonal dominance",
        "",
        r"$$\mathrm{DD}_k=\frac{|J_{kk}|}{\sum_{j\neq k}|J_{kj}|}$$",
        "",
        "| Arm | class | DD_C | DD_H | DD_O |",
        "|-----|-------|------|------|------|",
    ]
    for arm in ("predictive", "converted", "random"):
        dd = jacobians[arm]["diagonal_dominance"]
        cls = jacobians[arm]["composition_class"]
        lines.append(
            f"| {arm} | {cls} | {dd['C']:.2f} | {dd['H']:.2f} | {dd['O']:.2f} |"
        )

    lines += [
        "",
        "### Converted J (dM/α)",
        "",
        "| affected\\\\steer | C | H | O |",
        "|------------------|---|---|---|",
    ]
    Jc = np.asarray(jacobians["converted"]["J"])
    for i, ki in enumerate(CHANNELS):
        cells = " | ".join(f"{Jc[i, j]:+.2f}" for j in range(3))
        lines.append(f"| **{ki}** | {cells} |")

    lines += [
        "",
        "## Pairwise intended |ΔM| (mean over e-patterns)",
        "",
        "| Pair | question | converted | predictive | random | conv>pred |",
        "|------|----------|-----------|------------|--------|-----------|",
    ]
    for pair in PAIRS:
        key = f"{pair[0]}+{pair[1]}"
        s = summary[key]
        ba = s["by_arm"]
        lines.append(
            f"| {key} | {s['question'][:48]}… | "
            f"{ba['converted']['mean_abs_dM_intended']:.2f} | "
            f"{ba['predictive']['mean_abs_dM_intended']:.2f} | "
            f"{ba['random']['mean_abs_dM_intended']:.2f} | "
            f"{s['converted_beats_predictive']} |"
        )

    lines += [
        "",
        "## Collateral |ΔM| on the held-out channel",
        "",
        "| Pair | converted | predictive | random |",
        "|------|-----------|------------|--------|",
    ]
    for pair in PAIRS:
        key = f"{pair[0]}+{pair[1]}"
        ba = summary[key]["by_arm"]
        lines.append(
            f"| {key} | {ba['converted']['mean_abs_dM_collateral']:.2f} | "
            f"{ba['predictive']['mean_abs_dM_collateral']:.2f} | "
            f"{ba['random']['mean_abs_dM_collateral']:.2f} |"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- Converted beats predictive on intended |ΔM|: **{n_rank}/3** pairs",
        f"- Converted Jacobian class: **{gate['jacobian_class_converted']}**",
        f"- Ready for sequential C→H→O: **{gate['ready_for_sequential']}**",
        f"- Ready for 8-way: **False**",
        "",
        "If coupled: characterize coupling next — do not jump to 8-state controller.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
