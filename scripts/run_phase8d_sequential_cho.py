#!/usr/bin/env python3
"""Phase 8D — Sequential C→H→O with Phase-8C-corrected H hook.

Wire validated H chain into closed-loop composition:

  d_k = (2 m*_k − 1) v_c^k

  H: decision-token-only (PLAN draft → teacher-force PLAN → steer continuation)
  C: tool-phase target-sign (plan draft when H active; whole tool when H inactive)
  O: report-phase target-sign

Same seed for all stages within a trial (no seed+100*stage).
No new v, no surrogate, no policy, no full 8-way.

Compares against Phase 4C converted baseline (P(hit)=0.19, ΔE≈0, H-only ΔE gain).

  .venv/bin/python scripts/run_phase8d_sequential_cho.py --reps 4
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
from scripts.sync_eq import extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8d_sequential_cho.json"
MD = ROOT / "data" / "results" / "sync_phase8d_sequential_cho.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260905  # match Phase 4C for fair compare
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

SELECTED_MSTAR = (
    (0, 0, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)

PRIMARY_ORDER = ("C", "H", "O")

# Phase 4C converted reference (primary C→H→O, reps=4)
PHASE4_REF = {
    "P_hit": 0.19,
    "P_C": 0.50,
    "P_H": 0.62,
    "P_O": 0.50,
    "mean_delta_E_total": 0.00,
    "mean_delta_E_H": 0.50,
    "mean_E_traj": [1.38, 1.62, 1.12, 1.38],
}


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


def _e(mstar: tuple[int, int, int], S: list[int]) -> list[int]:
    return [int(mstar[i]) - int(S[i]) for i in range(3)]


def _s_star(mstar_k: int) -> int:
    """Target-sign s_k = 2 m*_k − 1."""
    return int(2 * int(mstar_k) - 1)


def _policy_ok(S: list[int]) -> int:
    return int(S[1] == S[2])


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    alpha: float,
    seed: int,
    h_plan_prefill: str | None = None,
) -> dict[str, Any]:
    """Episode with target-sign steers; H decision-token-only when H active.

    active[k] = (s_k, v_k) with s_k ∈ {−1,+1}.
    When H is active, teacher-force ``h_plan_prefill`` from the prior stage
    (Phase 8C same-episode site); draft only if prefill missing.
    """
    h_hook = None
    plan_hook = None
    o_hook = None
    tool_c_hook = None

    if "H" in active:
        sH, vH = active["H"]
        h_hook = make_steer_hook(loaded, vH, alpha * float(sH))
        # draft with C only if we must invent a PLAN (no carried prefill)
        if "C" in active and not (h_plan_prefill and str(h_plan_prefill).strip()):
            sC, vC = active["C"]
            plan_hook = make_steer_hook(loaded, vC, alpha * float(sC))
        if "C" in active:
            sC, vC = active["C"]
            tool_c_hook = make_steer_hook(loaded, vC, alpha * float(sC))

        def hook_for_turn(phase: str, turn: int):
            if phase == "tool" and turn > 0:
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

    else:
        tool_c_hook = None
        if "C" in active:
            sC, vC = active["C"]
            tool_c_hook = make_steer_hook(loaded, vC, alpha * float(sC))

        def hook_for_turn(phase: str, _turn: int):
            if phase == "tool":
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

    if "O" in active:
        sO, vO = active["O"]
        o_hook = make_steer_hook(loaded, vO, alpha * float(sO))

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
    )
    for h in (h_hook, plan_hook, o_hook, tool_c_hook):
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
    S = _S_from_row(row)
    plan = extract_plan(
        (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    )
    prefill = ""
    if plan:
        prefill = plan if plan.endswith("\n") else plan + "\n"
    return {
        "S": S,
        "plan_prefill": prefill,
        "task_ok": int(bool((row.get("final") or "").strip())),
        "policy_ok": _policy_ok(S),
        "s_tool": int(row.get("s_tool") or 0),
        "s_output": int(row.get("s_output") or 0),
    }


def run_sequential_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray],
    alpha: float,
    order: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    """S0 free → stages; same seed; carry PLAN into H decision-token steer.

    Skip episode re-runs when the stage adds no new channel (noop) — biggest
    speedup vs always doing free+C+H+O = 4 full episodes.
    """
    base = _run_episode(sc, loaded, active={}, alpha=alpha, seed=seed)
    S = base["S"]
    plan_prefill = base.get("plan_prefill") or ""
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    traj_e = [list(_e(mstar, S))]
    n_interv = 0
    n_episodes = 1
    active: dict[str, tuple[int, np.ndarray]] = {}
    stage_rows = []

    for k in order:
        e = _e(mstar, S)
        added = False
        if e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (_s_star(mstar[CH_IDX[k]]), dirs[k])
            n_interv += 1
            added = True
        if not added:
            # No new actuator — keep S/E; do not burn another full episode.
            E = traj_E[-1]
            stage_rows.append(
                {
                    "stage": k,
                    "S": list(S),
                    "e": list(e),
                    "E": E,
                    "active_s": {kk: int(vv[0]) for kk, vv in active.items()},
                    "n_active": len(active),
                    "skipped_noop": True,
                    "used_plan_prefill": False,
                    "task_ok": base["task_ok"] if not stage_rows else stage_rows[-1]["task_ok"],
                    "policy_ok": _policy_ok(S),
                }
            )
            traj_S.append(list(S))
            traj_E.append(E)
            traj_e.append(list(e))
            continue

        out = _run_episode(
            sc,
            loaded,
            active=active,
            alpha=alpha,
            seed=seed,
            h_plan_prefill=plan_prefill if "H" in active else None,
        )
        n_episodes += 1
        S = out["S"]
        if out.get("plan_prefill"):
            plan_prefill = out["plan_prefill"]
        E = float(sync_error_norm(_e(mstar, S)))
        stage_rows.append(
            {
                "stage": k,
                "S": list(S),
                "e": list(_e(mstar, S)),
                "E": E,
                "active_s": {kk: int(vv[0]) for kk, vv in active.items()},
                "n_active": len(active),
                "skipped_noop": False,
                "used_plan_prefill": bool(plan_prefill) and ("H" in active),
                "task_ok": out["task_ok"],
                "policy_ok": out["policy_ok"],
            }
        )
        traj_S.append(list(S))
        traj_E.append(E)
        traj_e.append(list(_e(mstar, S)))

    E0, E1, E2, E3 = traj_E[0], traj_E[1], traj_E[2], traj_E[3]
    S0, Sf = traj_S[0], traj_S[-1]
    return {
        "mstar": list(mstar),
        "order": list(order),
        "S0": S0,
        "S_final": Sf,
        "E_traj": traj_E,
        "e_traj": traj_e,
        "delta_E_C": float(E0 - E1),
        "delta_E_H": float(E1 - E2),
        "delta_E_O": float(E2 - E3),
        "delta_E_total": float(E0 - E3),
        "hit": int(Sf == list(mstar)),
        "bit_C": int(Sf[0] == mstar[0]),
        "bit_H": int(Sf[1] == mstar[1]),
        "bit_O": int(Sf[2] == mstar[2]),
        "W2C_H": int(S0[1] != mstar[1] and Sf[1] == mstar[1]),
        "wrong0_H": int(S0[1] != mstar[1]),
        "n_interventions": n_interv,
        "n_episodes": n_episodes,
        "task_ok_final": stage_rows[-1]["task_ok"] if stage_rows else base["task_ok"],
        "policy_ok_final": stage_rows[-1]["policy_ok"] if stage_rows else base["policy_ok"],
        "stages": stage_rows,
    }


def _agg(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([t[key] for t in trials]))

    wrong_h = [t for t in trials if t["wrong0_H"]]
    return {
        "n": len(trials),
        "P_hit": mean("hit"),
        "P_C": mean("bit_C"),
        "P_H": mean("bit_H"),
        "P_O": mean("bit_O"),
        "P_W2C_H": float(np.mean([t["W2C_H"] for t in wrong_h])) if wrong_h else float("nan"),
        "n_wrong0_H": len(wrong_h),
        "mean_E0": float(np.mean([t["E_traj"][0] for t in trials])),
        "mean_E_final": float(np.mean([t["E_traj"][-1] for t in trials])),
        "mean_delta_E_total": mean("delta_E_total"),
        "mean_delta_E_C": mean("delta_E_C"),
        "mean_delta_E_H": mean("delta_E_H"),
        "mean_delta_E_O": mean("delta_E_O"),
        "mean_n_interventions": mean("n_interventions"),
        "P_task_ok": mean("task_ok_final"),
        "P_policy_ok": mean("policy_ok_final"),
        "mean_E_traj": [
            float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2, help="reps per m* (default 2 for speed)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="reps=2 and only 2 m* masks: (0,0,0),(1,1,1)",
    )
    args = ap.parse_args()
    mstar_set = SELECTED_MSTAR
    if args.fast:
        args.reps = min(args.reps, 2)
        mstar_set = ((0, 0, 0), (1, 1, 1))

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not VC_PATH.is_file():
        raise SystemExit(f"missing {VC_PATH}")

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = _load_vc()

    print(
        f"=== Phase 8D sequential C→H→O "
        f"(reps={args.reps}, n_m*={len(mstar_set)}, skip-noop) ===",
        flush=True,
    )
    trials = []
    for mi, mstar in enumerate(mstar_set):
        for r in range(args.reps):
            seed = args.seed + 1000 * mi + 10 * r
            print(f"  m*={mstar} r={r} seed={seed}", flush=True)
            torch.manual_seed(seed)
            tr = run_sequential_trial(
                sc,
                loaded,
                mstar=mstar,
                dirs=dirs,
                alpha=args.alpha,
                order=PRIMARY_ORDER,
                seed=seed,
            )
            trials.append(tr)
            print(
                f"    E:{[round(x, 2) for x in tr['E_traj']]} "
                f"hit={tr['hit']} S0={tr['S0']}→{tr['S_final']} "
                f"W2C_H={tr['W2C_H']} eps={tr['n_episodes']}",
                flush=True,
            )

    agg = _agg(trials)
    gate = {
        "beats_phase4_hit": bool(agg["P_hit"] > PHASE4_REF["P_hit"] + 0.05),
        "beats_phase4_delta_E": bool(
            agg["mean_delta_E_total"] > PHASE4_REF["mean_delta_E_total"] + 0.05
        ),
        "H_stage_reduces_E": bool(agg["mean_delta_E_H"] > 0.05),
        "P_H_vs_phase4": float(agg["P_H"] - PHASE4_REF["P_H"]),
        "P_W2C_H": agg["P_W2C_H"],
        "sequential_composition_improved": False,
    }
    gate["sequential_composition_improved"] = bool(
        gate["beats_phase4_hit"]
        or gate["beats_phase4_delta_E"]
        or (gate["H_stage_reduces_E"] and agg["P_H"] >= PHASE4_REF["P_H"])
    )
    gate["read"] = (
        "Tests whether Phase-8C H chain survives C→H→O sequential composition. "
        "No new v. 8-way still paused."
    )

    payload = {
        "protocol": "Phase 8D sequential C→H→O target-sign + H decision-token-only",
        "alpha": args.alpha,
        "reps": args.reps,
        "seed": args.seed,
        "fast": bool(args.fast),
        "skip_noop_stages": True,
        "mstar_set": [list(m) for m in mstar_set],
        "formula": "d_k = (2 m*_k - 1) v_c^k ; H = PLAN prefill + steered continuation",
        "same_seed_stages": True,
        "phase4_ref": PHASE4_REF,
        "agg": agg,
        "gate": gate,
        "trials": trials,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    et = agg["mean_E_traj"]
    lines = [
        "# Phase 8D — Sequential C→H→O (corrected H)",
        "",
        r"> $d_k=(2m_k^*-1)v_c^k$. H = decision-token-only (Phase 8C). Same seed per trial.",
        "",
        "No new $v$. No policy. **8-way CLOSED.** Skip noop stages.",
        "",
        f"m* set: {list(mstar_set)} · reps={args.reps} · α={args.alpha} · fast={args.fast}",
        "",
        "## Primary order C→H→O",
        "",
        "| Arm | P(hit) | P(C) | P(H) | P(O) | P(W→C)_H | ΔE tot | E0→E3 |",
        "|-----|--------|------|------|------|----------|--------|-------|",
        (
            f"| corrected | {agg['P_hit']:.2f} | {agg['P_C']:.2f} | {agg['P_H']:.2f} | "
            f"{agg['P_O']:.2f} | {agg['P_W2C_H']:.2f} | {agg['mean_delta_E_total']:+.2f} | "
            f"{et[0]:.2f}→{et[1]:.2f}→{et[2]:.2f}→{et[3]:.2f} |"
        ),
        (
            f"| Phase4 converted (ref) | {PHASE4_REF['P_hit']:.2f} | {PHASE4_REF['P_C']:.2f} | "
            f"{PHASE4_REF['P_H']:.2f} | {PHASE4_REF['P_O']:.2f} | — | "
            f"{PHASE4_REF['mean_delta_E_total']:+.2f} | "
            f"{PHASE4_REF['mean_E_traj'][0]:.2f}→{PHASE4_REF['mean_E_traj'][1]:.2f}→"
            f"{PHASE4_REF['mean_E_traj'][2]:.2f}→{PHASE4_REF['mean_E_traj'][3]:.2f} |"
        ),
        "",
        "### Per-stage ΔE",
        "",
        "| Arm | ΔE_C | ΔE_H | ΔE_O |",
        "|-----|------|------|------|",
        (
            f"| corrected | {agg['mean_delta_E_C']:+.2f} | {agg['mean_delta_E_H']:+.2f} | "
            f"{agg['mean_delta_E_O']:+.2f} |"
        ),
        (
            f"| Phase4 converted | −0.25 | **+0.50** | −0.25 |"
        ),
        "",
        "## Gate",
        "",
        f"- Beats Phase4 P(hit): **{gate['beats_phase4_hit']}**",
        f"- Beats Phase4 ΔE tot: **{gate['beats_phase4_delta_E']}**",
        f"- H stage reduces E: **{gate['H_stage_reduces_E']}**",
        f"- ΔP(H) vs Phase4: **{gate['P_H_vs_phase4']:+.2f}**",
        f"- P(W→C)_H: **{gate['P_W2C_H']}**",
        f"- Sequential composition improved: **{gate['sequential_composition_improved']}**",
        "",
        gate["read"],
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "agg": agg, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
