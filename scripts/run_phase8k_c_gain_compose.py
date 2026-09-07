#!/usr/bin/env python3
"""Phase 8K — Compose corrected actuators: H→C→O with α_C=5 stem-prefill.

Freeze:
  order H → C → O
  α_H = 1.5  (decision-token)
  α_C = 5.0  (stem-prefill \"PLAN: I will \")
  α_O = 1.5  (early-FINAL)
  target-sign, same seed, frozen v_c, no skip, no new v

Primary question vs Phase 8I H→C (ΔE_C=−0.25, ΔE_H=+0.62):
  Does ΔE_C ≥ 0 while preserving H/O gains?

If yes → full 8-way replication with this controller.

  .venv/bin/python scripts/run_phase8k_c_gain_compose.py --reps 2
  .venv/bin/python scripts/run_phase8k_c_gain_compose.py --reps 2 --eight-way
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

from scripts.sync_channel_margins import unit  # noqa: E402
from scripts.sync_eq import all_m_star_masks, extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8k_c_gain_compose.json"
MD = ROOT / "data" / "results" / "sync_phase8k_c_gain_compose.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260905  # match 8I/8D
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ORDER = ("H", "C", "O")

ALPHA_H = 1.5
ALPHA_C = 5.0
ALPHA_O = 1.5
C_STEM = "PLAN: I will "

SELECTED_MSTAR = (
    (0, 0, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)

# Phase 8I H→C→O reference (selected 4, reps=2)
REF_8I = {
    "P_hit": 0.38,
    "mean_delta_E_total": 0.75,
    "mean_delta_E_C": -0.25,
    "mean_delta_E_H": 0.62,
    "mean_delta_E_O": 0.375,
    "P_C": 0.62,
    "P_H": 0.75,
    "P_O": 0.88,
}
REF_8H = {"P_hit": 0.31, "mean_delta_E_total": 0.38, "P_C": 0.56, "P_H": 0.69, "P_O": 0.75}
REF_8E = {"P_hit": 0.06, "mean_delta_E_total": 0.31, "P_C": 0.56, "P_H": 0.62, "P_O": 0.44}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


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


def _s_star(mstar_k: int) -> int:
    return int(2 * int(mstar_k) - 1)


def _alphas() -> dict[str, float]:
    return {"C": ALPHA_C, "H": ALPHA_H, "O": ALPHA_O}


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    seed: int,
    h_plan_prefill: str | None = None,
) -> dict[str, Any]:
    """H decision-token + C stem-prefill (α_C=5) + O early-FINAL."""
    alphas = _alphas()
    h_hook = c_hook = o_hook = tool_c_hook = None
    c_stem = None

    if "H" in active:
        sH, vH = active["H"]
        h_hook = make_steer_hook(loaded, vH, alphas["H"] * float(sH))

    if "C" in active:
        sC, vC = active["C"]
        c_hook = make_steer_hook(loaded, vC, alphas["C"] * float(sC))
        c_stem = C_STEM
        # later tool turns: keep mild C (same α) if multi-turn
        tool_c_hook = make_steer_hook(loaded, vC, alphas["C"] * float(sC))

    if "O" in active:
        sO, vO = active["O"]
        o_hook = make_steer_hook(loaded, vO, alphas["O"] * float(sO))

    def hook_for_turn(phase: str, turn: int):
        if phase == "tool" and turn > 0:
            return tool_c_hook
        if phase == "report":
            return o_hook
        return None

    report_prefill = "FINAL: " if "O" in active else None

    row = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
        capture_activations=False,
        h_decision_hook=h_hook,
        h_plan_prefill=h_plan_prefill if ("H" in active and c_stem is None) else None,
        c_decision_hook=c_hook,
        c_stem_prefill=c_stem,
        report_prefill=report_prefill,
    )
    for h in (h_hook, c_hook, o_hook, tool_c_hook):
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
    S = _S_from_row(row)
    plan = extract_plan((row.get("final") or "") + "\n" + (row.get("messages_text") or ""))
    prefill = (plan if plan.endswith("\n") else plan + "\n") if plan else ""
    return {"S": S, "plan_prefill": prefill, "task_ok": int(bool((row.get("final") or "").strip()))}


def run_sequential_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    base = _run_episode(sc, loaded, active={}, seed=seed)
    S = base["S"]
    plan_prefill = base.get("plan_prefill") or ""
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    stage_E: dict[str, float] = {}
    n_interv = 0
    n_episodes = 1
    active: dict[str, tuple[int, np.ndarray]] = {}

    for k in ORDER:
        e = _e(mstar, S)
        E_pre = float(sync_error_norm(e))
        added = False
        if e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (_s_star(mstar[CH_IDX[k]]), dirs[k])
            n_interv += 1
            added = True
        if not added:
            traj_S.append(list(S))
            traj_E.append(traj_E[-1])
            stage_E[k] = 0.0
            continue
        # When C is active, stem path regenerates PLAN; do not force old H prefill
        # over C. When only H (C not yet), carry plan_prefill.
        h_pf = None
        if "H" in active and "C" not in active:
            h_pf = plan_prefill
        out = _run_episode(
            sc,
            loaded,
            active=active,
            seed=seed,
            h_plan_prefill=h_pf,
        )
        n_episodes += 1
        S = out["S"]
        if out.get("plan_prefill"):
            plan_prefill = out["plan_prefill"]
        E_post = float(sync_error_norm(_e(mstar, S)))
        traj_S.append(list(S))
        traj_E.append(E_post)
        stage_E[k] = float(E_pre - E_post)

    S0, Sf = traj_S[0], traj_S[-1]
    return {
        "mstar": list(mstar),
        "S0": S0,
        "S_final": Sf,
        "E_traj": traj_E,
        "delta_E_total": float(traj_E[0] - traj_E[-1]),
        "delta_E_H": float(stage_E.get("H", 0.0)),
        "delta_E_C": float(stage_E.get("C", 0.0)),
        "delta_E_O": float(stage_E.get("O", 0.0)),
        "hit": int(Sf == list(mstar)),
        "bit_C": int(Sf[0] == mstar[0]),
        "bit_H": int(Sf[1] == mstar[1]),
        "bit_O": int(Sf[2] == mstar[2]),
        "n_interventions": n_interv,
        "n_episodes": n_episodes,
    }


def _agg(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    et = [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)] if trials else []
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eight-way", action="store_true", help="all 8 m* (default: 8I selected 4)")
    ap.add_argument(
        "--force-eight-way",
        action="store_true",
        help="run 8-way even if ΔE_C gate fails",
    )
    args = ap.parse_args()
    mstars = (
        [tuple(m) for m in all_m_star_masks()]
        if args.eight_way
        else list(SELECTED_MSTAR)
    )

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = _load_vc()

    print(
        f"=== Phase 8K H→C→O α_C={ALPHA_C} α_H={ALPHA_H} α_O={ALPHA_O} "
        f"stem={C_STEM!r} n_m*={len(mstars)} reps={args.reps} ===",
        flush=True,
    )

    trials = []
    for mi, mstar in enumerate(mstars):
        for r in range(args.reps):
            seed = args.seed + 1000 * mi + 10 * r
            torch.manual_seed(seed)
            print(f"  m*={mstar} r={r}", flush=True)
            tr = run_sequential_trial(sc, loaded, mstar=mstar, dirs=dirs, seed=seed)
            trials.append(tr)
            print(
                f"    E:{[round(x, 2) for x in tr['E_traj']]} "
                f"ΔE_C={tr['delta_E_C']:+.2f} ΔE_H={tr['delta_E_H']:+.2f} "
                f"ΔE_O={tr['delta_E_O']:+.2f} hit={tr['hit']}",
                flush=True,
            )

    agg = _agg(trials)
    dEc = agg["mean_delta_E_C"]
    dEh = agg["mean_delta_E_H"]
    dEo = agg["mean_delta_E_O"]
    gate = {
        "hypothesis": "α_C=5 stem-prefill flips ΔE_C≥0 vs 8I (−0.25) while preserving H/O",
        "ref_8I_H_then_C": REF_8I,
        "delta_E_C": dEc,
        "delta_E_H": dEh,
        "delta_E_O": dEo,
        "delta_E_C_ge_0": bool(dEc >= 0.0),
        "delta_E_C_improves_vs_8I": bool(dEc > REF_8I["mean_delta_E_C"] + 0.05),
        "H_preserved": bool(dEh >= REF_8I["mean_delta_E_H"] - 0.15),
        "O_nonneg": bool(dEo >= -0.05),
        "P_hit": agg["P_hit"],
        "P_C": agg["P_C"],
        "P_H": agg["P_H"],
        "P_O": agg["P_O"],
    }
    gate["compose_pass"] = bool(
        gate["delta_E_C_ge_0"]
        and gate["delta_E_C_improves_vs_8I"]
        and gate["H_preserved"]
    )
    gate["eight_way_recommended"] = bool(gate["compose_pass"] or gate["delta_E_C_ge_0"])
    gate["read"] = (
        "If ΔE_C≥0 and H preserved: wire this controller to full 8-way. "
        "If ΔE_C still <0: C site/gain still insufficient — then mechanistic, not more α. "
        "No skip. No new v yet."
    )

    print(
        f"  >> P(hit)={agg['P_hit']:.2f} ΔE={agg['mean_delta_E_total']:+.2f} "
        f"ΔE_C={dEc:+.2f} ΔE_H={dEh:+.2f} ΔE_O={dEo:+.2f} "
        f"P(C/H/O)={agg['P_C']:.2f}/{agg['P_H']:.2f}/{agg['P_O']:.2f} "
        f"compose_pass={gate['compose_pass']}",
        flush=True,
    )

    # Optional immediate 8-way if selected-4 passed and user asked eight-way,
    # or auto-continue when compose_pass and --eight-way not yet set via flag.
    eight_agg = None
    eight_trials = None
    ran_eight = bool(args.eight_way)
    if (
        not args.eight_way
        and (gate["eight_way_recommended"] or args.force_eight_way)
        and (gate["delta_E_C_ge_0"] or args.force_eight_way)
    ):
        print("=== auto full 8-way (ΔE_C gate ok) ===", flush=True)
        ran_eight = True
        mstars8 = [tuple(m) for m in all_m_star_masks()]
        eight_trials = []
        for mi, mstar in enumerate(mstars8):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r
                torch.manual_seed(seed)
                print(f"  8way m*={mstar} r={r}", flush=True)
                tr = run_sequential_trial(sc, loaded, mstar=mstar, dirs=dirs, seed=seed)
                eight_trials.append(tr)
                print(
                    f"    E:{[round(x, 2) for x in tr['E_traj']]} hit={tr['hit']} "
                    f"ΔE_C={tr['delta_E_C']:+.2f}",
                    flush=True,
                )
        eight_agg = _agg(eight_trials)
        print(
            f"  >> 8-way P(hit)={eight_agg['P_hit']:.2f} "
            f"ΔE={eight_agg['mean_delta_E_total']:+.2f} "
            f"ΔE_C={eight_agg['mean_delta_E_C']:+.2f} "
            f"P(C/H/O)={eight_agg['P_C']:.2f}/{eight_agg['P_H']:.2f}/{eight_agg['P_O']:.2f}",
            flush=True,
        )
    elif args.eight_way:
        eight_agg = agg
        eight_trials = trials

    payload = {
        "protocol": "Phase 8K C gain compose H→C→O",
        "alphas": _alphas(),
        "c_stem": C_STEM,
        "order": list(ORDER),
        "reps": args.reps,
        "eight_way": ran_eight,
        "mstar_set": [list(m) for m in mstars],
        "selected": agg,
        "eight_way_agg": eight_agg,
        "gate": gate,
        "trials_selected": trials if not args.eight_way else None,
        "trials_eight": eight_trials,
        "trials": trials if args.eight_way else trials,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8K — C gain compose ($H\\to C\\to O$)",
        "",
        r"> $\alpha_H=1.5$ decision-token; $\alpha_C=5$ stem-prefill; "
        r"$\alpha_O=1.5$ early-FINAL. Target-sign. No skip. No new $v$.",
        "",
        f"m* n={len(mstars)}, reps={args.reps}, eight_way={ran_eight}.",
        "",
        "## Selected / primary",
        "",
        "| | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |",
        "|--|--------|---:|-----:|-----:|-----:|------|------|------|",
        f"| **8K** | {agg['P_hit']:.2f} | {agg['mean_delta_E_total']:+.2f} | "
        f"**{agg['mean_delta_E_C']:+.2f}** | {agg['mean_delta_E_H']:+.2f} | "
        f"{agg['mean_delta_E_O']:+.2f} | {agg['P_C']:.2f} | {agg['P_H']:.2f} | "
        f"{agg['P_O']:.2f} |",
        f"| 8I H→C | {REF_8I['P_hit']:.2f} | {REF_8I['mean_delta_E_total']:+.2f} | "
        f"{REF_8I['mean_delta_E_C']:+.2f} | {REF_8I['mean_delta_E_H']:+.2f} | "
        f"{REF_8I['mean_delta_E_O']:+.2f} | {REF_8I['P_C']:.2f} | {REF_8I['P_H']:.2f} | "
        f"{REF_8I['P_O']:.2f} |",
        "",
    ]
    if eight_agg is not None and ran_eight:
        lines += [
            "## Full 8-way",
            "",
            "| | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |",
            "|--|--------|---:|-----:|-----:|-----:|------|------|------|",
            f"| **8K 8-way** | {eight_agg['P_hit']:.2f} | "
            f"{eight_agg['mean_delta_E_total']:+.2f} | "
            f"{eight_agg['mean_delta_E_C']:+.2f} | {eight_agg['mean_delta_E_H']:+.2f} | "
            f"{eight_agg['mean_delta_E_O']:+.2f} | {eight_agg['P_C']:.2f} | "
            f"{eight_agg['P_H']:.2f} | {eight_agg['P_O']:.2f} |",
            f"| 8H ref | {REF_8H['P_hit']:.2f} | {REF_8H['mean_delta_E_total']:+.2f} | "
            f"— | — | — | {REF_8H['P_C']:.2f} | {REF_8H['P_H']:.2f} | {REF_8H['P_O']:.2f} |",
            f"| 8E conv | {REF_8E['P_hit']:.2f} | {REF_8E['mean_delta_E_total']:+.2f} | "
            f"— | — | — | {REF_8E['P_C']:.2f} | {REF_8E['P_H']:.2f} | {REF_8E['P_O']:.2f} |",
            "",
        ]
    lines += [
        "## Gate",
        "",
        f"- $\\Delta E_C \\ge 0$: **{gate['delta_E_C_ge_0']}** ({dEc:+.2f})",
        f"- $\\Delta E_C$ improves vs 8I (−0.25): **{gate['delta_E_C_improves_vs_8I']}**",
        f"- H preserved: **{gate['H_preserved']}** ({dEh:+.2f})",
        f"- Compose pass: **{gate['compose_pass']}**",
        f"- 8-way recommended: **{gate['eight_way_recommended']}**",
        "",
        gate["read"],
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "agg": agg, "eight": eight_agg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
