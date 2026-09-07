#!/usr/bin/env python3
"""Phase 8H — Controller integration: channel-specific α_O ∈ {5,8}.

Phase 8G: O under-actuated (not misaligned). Frozen v_c^O valid at higher gain.

  α_C = α_H = 1.5
  α_O ∈ {1.5 (8D baseline), 5, 8}

Path: C → H decision-token → early-FINAL O
  d_k = (2 m*_k − 1) v_c^k

Compare α_O=5 vs 8 vs 1.5 on same seeds. No new v. No policy.

  .venv/bin/python scripts/run_phase8h_o_gain.py --reps 2
  .venv/bin/python scripts/run_phase8h_o_gain.py --reps 2 --eight-way
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

OUT = ROOT / "data" / "results" / "sync_phase8h_o_gain.json"
MD = ROOT / "data" / "results" / "sync_phase8h_o_gain.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260905  # match 8D
ALPHA_C = 1.5
ALPHA_H = 1.5
ALPHA_O_SET = (1.5, 5.0, 8.0)
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ORDER = ("C", "H", "O")

SELECTED_MSTAR = (
    (0, 0, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)

# Published refs
REF_8D = {"P_hit": 0.25, "mean_delta_E_total": 0.50, "P_O": 0.88, "P_H": 0.62}
REF_8E_CONV = {"P_hit": 0.06, "mean_delta_E_total": 0.31, "P_O": 0.44, "P_H": 0.62}


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


def _alphas(alpha_O: float) -> dict[str, float]:
    return {"C": ALPHA_C, "H": ALPHA_H, "O": float(alpha_O)}


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    alphas: dict[str, float],
    seed: int,
    h_plan_prefill: str | None = None,
) -> dict[str, Any]:
    """8D path + per-channel α + early-FINAL when O active."""
    h_hook = plan_hook = o_hook = tool_c_hook = None

    if "H" in active:
        sH, vH = active["H"]
        h_hook = make_steer_hook(loaded, vH, alphas["H"] * float(sH))
        if "C" in active and not (h_plan_prefill and str(h_plan_prefill).strip()):
            sC, vC = active["C"]
            plan_hook = make_steer_hook(loaded, vC, alphas["C"] * float(sC))
        if "C" in active:
            sC, vC = active["C"]
            tool_c_hook = make_steer_hook(loaded, vC, alphas["C"] * float(sC))

        def hook_for_turn(phase: str, turn: int):
            if phase == "tool" and turn > 0:
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

    else:
        if "C" in active:
            sC, vC = active["C"]
            tool_c_hook = make_steer_hook(loaded, vC, alphas["C"] * float(sC))

        def hook_for_turn(phase: str, _turn: int):
            if phase == "tool":
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

    report_prefill = None
    if "O" in active:
        sO, vO = active["O"]
        o_hook = make_steer_hook(loaded, vO, alphas["O"] * float(sO))
        report_prefill = "FINAL: "  # early-FINAL decision tokens only

    row = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
        capture_activations=False,
        h_decision_hook=h_hook,
        plan_draft_hook=plan_hook,
        h_plan_prefill=h_plan_prefill if "H" in active else None,
        report_prefill=report_prefill,
    )
    for h in (h_hook, plan_hook, o_hook, tool_c_hook):
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
    S = _S_from_row(row)
    plan = extract_plan((row.get("final") or "") + "\n" + (row.get("messages_text") or ""))
    prefill = (plan if plan.endswith("\n") else plan + "\n") if plan else ""
    return {
        "S": S,
        "plan_prefill": prefill,
        "task_ok": int(bool((row.get("final") or "").strip())),
        "s_tool": int(row.get("s_tool") or 0),
        "s_output": int(row.get("s_output") or 0),
    }


def run_sequential_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray],
    alpha_O: float,
    seed: int,
) -> dict[str, Any]:
    alphas = _alphas(alpha_O)
    base = _run_episode(sc, loaded, active={}, alphas=alphas, seed=seed)
    S = base["S"]
    plan_prefill = base.get("plan_prefill") or ""
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    n_interv = 0
    n_episodes = 1
    active: dict[str, tuple[int, np.ndarray]] = {}

    for k in ORDER:
        e = _e(mstar, S)
        added = False
        if e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (_s_star(mstar[CH_IDX[k]]), dirs[k])
            n_interv += 1
            added = True
        if not added:
            traj_S.append(list(S))
            traj_E.append(traj_E[-1])
            continue
        out = _run_episode(
            sc,
            loaded,
            active=active,
            alphas=alphas,
            seed=seed,
            h_plan_prefill=plan_prefill if "H" in active else None,
        )
        n_episodes += 1
        S = out["S"]
        if out.get("plan_prefill"):
            plan_prefill = out["plan_prefill"]
        traj_S.append(list(S))
        traj_E.append(float(sync_error_norm(_e(mstar, S))))

    E0, E1, E2, E3 = traj_E
    S0, Sf = traj_S[0], traj_S[-1]
    return {
        "mstar": list(mstar),
        "alpha_O": alpha_O,
        "S0": S0,
        "S_final": Sf,
        "E_traj": traj_E,
        "delta_E_total": float(E0 - E3),
        "delta_E_C": float(E0 - E1),
        "delta_E_H": float(E1 - E2),
        "delta_E_O": float(E2 - E3),
        "hit": int(Sf == list(mstar)),
        "bit_C": int(Sf[0] == mstar[0]),
        "bit_H": int(Sf[1] == mstar[1]),
        "bit_O": int(Sf[2] == mstar[2]),
        "W2C_O": int(S0[2] != mstar[2] and Sf[2] == mstar[2]),
        "wrong0_O": int(S0[2] != mstar[2]),
        "W2C_H": int(S0[1] != mstar[1] and Sf[1] == mstar[1]),
        "wrong0_H": int(S0[1] != mstar[1]),
        "n_interventions": n_interv,
        "n_episodes": n_episodes,
    }


def _agg(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    wrong_o = [t for t in trials if t["wrong0_O"]]
    wrong_h = [t for t in trials if t["wrong0_H"]]
    et = [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)] if trials else []
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_C": m("bit_C"),
        "P_H": m("bit_H"),
        "P_O": m("bit_O"),
        "P_W2C_O": float(np.mean([t["W2C_O"] for t in wrong_o])) if wrong_o else float("nan"),
        "P_W2C_H": float(np.mean([t["W2C_H"] for t in wrong_h])) if wrong_h else float("nan"),
        "n_wrong0_O": len(wrong_o),
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
    ap.add_argument(
        "--alpha-O",
        default="1.5,5,8",
        help="comma-separated α_O values (include 1.5 as 8D baseline)",
    )
    ap.add_argument(
        "--eight-way",
        action="store_true",
        help="all 8 m* (default: 8D selected 4)",
    )
    args = ap.parse_args()
    alpha_Os = tuple(float(x) for x in args.alpha_O.split(",") if x.strip())
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
        f"=== Phase 8H α_C=α_H=1.5 α_O={alpha_Os} "
        f"n_m*={len(mstars)} reps={args.reps} early-FINAL ===",
        flush=True,
    )

    by_aO: dict[str, Any] = {}
    all_trials: dict[str, list] = {}

    for aO in alpha_Os:
        trials = []
        for mi, mstar in enumerate(mstars):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r  # paired across α_O
                torch.manual_seed(seed)
                print(f"  α_O={aO:g} m*={mstar} r={r}", flush=True)
                tr = run_sequential_trial(
                    sc, loaded, mstar=mstar, dirs=dirs, alpha_O=aO, seed=seed
                )
                trials.append(tr)
                print(
                    f"    E:{[round(x, 2) for x in tr['E_traj']]} hit={tr['hit']} "
                    f"S0={tr['S0']}→{tr['S_final']} W2C_O={tr['W2C_O']} eps={tr['n_episodes']}",
                    flush=True,
                )
        agg = _agg(trials)
        key = str(aO)
        by_aO[key] = agg
        all_trials[key] = trials
        print(
            f"  >> α_O={aO:g}: P(hit)={agg['P_hit']:.2f} ΔE={agg['mean_delta_E_total']:+.2f} "
            f"P(O)={agg['P_O']:.2f} P(W→C)_O={agg['P_W2C_O']:.2f}",
            flush=True,
        )

    # gates vs baseline α_O=1.5 if present
    base_key = "1.5" if "1.5" in by_aO else None
    gate: dict[str, Any] = {
        "hypothesis": "higher α_O improves P(O), P(W→C)_O, and 8-way/sequential sync",
        "ref_8D": REF_8D,
        "ref_8E_converted": REF_8E_CONV,
    }
    for aO in alpha_Os:
        k = str(aO)
        g = by_aO[k]
        gate[f"alpha_O_{k}"] = {
            "P_hit": g["P_hit"],
            "delta_E": g["mean_delta_E_total"],
            "P_O": g["P_O"],
            "P_W2C_O": g["P_W2C_O"],
        }
    if base_key and "5.0" in by_aO:
        gate["aO5_beats_1p5_PO"] = bool(by_aO["5.0"]["P_O"] > by_aO[base_key]["P_O"] + 0.05)
        gate["aO5_beats_1p5_W2C"] = bool(
            (by_aO["5.0"]["P_W2C_O"] or 0) > (by_aO[base_key]["P_W2C_O"] or 0) + 0.05
        )
        gate["aO5_beats_1p5_deltaE"] = bool(
            by_aO["5.0"]["mean_delta_E_total"] > by_aO[base_key]["mean_delta_E_total"] + 0.05
        )
    if base_key and "8.0" in by_aO:
        gate["aO8_beats_1p5_PO"] = bool(by_aO["8.0"]["P_O"] > by_aO[base_key]["P_O"] + 0.05)
        gate["aO8_beats_1p5_W2C"] = bool(
            (by_aO["8.0"]["P_W2C_O"] or 0) > (by_aO[base_key]["P_W2C_O"] or 0) + 0.05
        )
        gate["aO8_beats_1p5_deltaE"] = bool(
            by_aO["8.0"]["mean_delta_E_total"] > by_aO[base_key]["mean_delta_E_total"] + 0.05
        )
    if "5.0" in by_aO and "8.0" in by_aO:
        gate["aO8_vs_5_PO"] = float(by_aO["8.0"]["P_O"] - by_aO["5.0"]["P_O"])
        gate["aO8_vs_5_W2C"] = float(
            (by_aO["8.0"]["P_W2C_O"] or 0) - (by_aO["5.0"]["P_W2C_O"] or 0)
        )
    gate["O_gain_helps"] = bool(
        gate.get("aO5_beats_1p5_W2C")
        or gate.get("aO8_beats_1p5_W2C")
        or gate.get("aO5_beats_1p5_PO")
        or gate.get("aO8_beats_1p5_PO")
    )
    gate["read"] = (
        "If α_O=5/8 lifts P(O) and P(W→C)_O vs 1.5: gain calibration works. "
        "If O↑ but hit/ΔE flat: composition/order bottleneck. No new v."
    )

    payload = {
        "protocol": "Phase 8H O gain integration",
        "alpha_C": ALPHA_C,
        "alpha_H": ALPHA_H,
        "alpha_O_set": list(alpha_Os),
        "early_FINAL": True,
        "reps": args.reps,
        "eight_way": bool(args.eight_way),
        "mstar_set": [list(m) for m in mstars],
        "by_alpha_O": by_aO,
        "gate": gate,
        "trials": {k: v for k, v in all_trials.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8H — O gain integration ($\\alpha_O\\in\\{5,8\\}$)",
        "",
        r"> $\alpha_C=\alpha_H=1.5$; early-FINAL O; H decision-token; $d_k=(2m_k^*-1)v_c^k$. No new $v$.",
        "",
        f"m* n={len(mstars)}, reps={args.reps}, eight_way={args.eight_way}.",
        "",
        "## Aggregate by $\\alpha_O$",
        "",
        "| $\\alpha_O$ | P(hit) | ΔE | P(C) | P(H) | P(O) | P(W→C)$_O$ | E0→E3 |",
        "|-----------:|--------|---:|------|------|------|------------|-------|",
    ]
    for aO in alpha_Os:
        g = by_aO[str(aO)]
        et = "→".join(f"{x:.2f}" for x in g["mean_E_traj"]) if g["mean_E_traj"] else "—"
        lines.append(
            f"| {aO:g} | {g['P_hit']:.2f} | {g['mean_delta_E_total']:+.2f} | "
            f"{g['P_C']:.2f} | {g['P_H']:.2f} | {g['P_O']:.2f} | "
            f"{g['P_W2C_O']:.2f} | {et} |"
        )
    lines += [
        "",
        f"| 8D ref (α_O=1.5) | {REF_8D['P_hit']:.2f} | {REF_8D['mean_delta_E_total']:+.2f} | "
        f"— | {REF_8D['P_H']:.2f} | {REF_8D['P_O']:.2f} | — | — |",
        "",
        "## Gate",
        "",
        f"- O gain helps vs α_O=1.5: **{gate.get('O_gain_helps')}**",
        f"- α_O=5 beats 1.5 on P(O): **{gate.get('aO5_beats_1p5_PO', 'n/a')}**",
        f"- α_O=5 beats 1.5 on P(W→C)_O: **{gate.get('aO5_beats_1p5_W2C', 'n/a')}**",
        f"- α_O=8 beats 1.5 on P(O): **{gate.get('aO8_beats_1p5_PO', 'n/a')}**",
        f"- α_O=8 beats 1.5 on P(W→C)_O: **{gate.get('aO8_beats_1p5_W2C', 'n/a')}**",
        f"- α_O=8 − 5 on P(O): **{gate.get('aO8_vs_5_PO', 'n/a')}**",
        "",
        gate["read"],
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "by_alpha_O": by_aO}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
