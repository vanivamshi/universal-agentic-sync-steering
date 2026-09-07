#!/usr/bin/env python3
"""Phase 4C — Sequential C→H→O closed-loop composition.

Freeze directions/sites/gains. No discovery.

Protocol (primary order C→H→O):
  S0 = free episode
  for k in order:
      e_t = m* − S_t          # recompute every stage
      if e_k ≠ 0: lock steer_k = (e_k, v_k) into active set (cumulative)
      S_{t+1} = episode with site steers for all active channels
  → S_final

Arms: converted vs predictive vs random (identical targets/seeds/sites/α/logic).

Success:
  Converted sequential control → greater sync, lower interference than predictive.

  .venv/bin/python scripts/run_phase4_sequential_composition.py --reps 4
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
from scripts.sync_channel_margins import (  # noqa: E402
    SPECS,
    margin_at_site,
    messages_for_channel,
    unit,
)
from scripts.sync_eq import extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import LAYER_DEFAULT, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase4_sequential_composition.json"
MD = ROOT / "data" / "results" / "sync_phase4_sequential_composition.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260905
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

# Selected m* — not full 8-way yet
SELECTED_MSTAR = (
    (0, 0, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)

PRIMARY_ORDER = ("C", "H", "O")
DIAGNOSTIC_ORDERS = (("H", "C", "O"), ("O", "H", "C"))


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_dirs(seed: int) -> dict[str, dict[str, np.ndarray]]:
    bank = ChannelBank.load(CHANNEL_V)
    blob = json.loads(VC_PATH.read_text())
    Vc = np.asarray(blob["V_c"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
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


def _e(mstar: tuple[int, int, int], S: list[int]) -> list[int]:
    return [int(mstar[i]) - int(S[i]) for i in range(3)]


def _policy_ok(S: list[int]) -> int:
    """H==O is the sync policy bit used elsewhere."""
    return int(S[1] == S[2])


def _compose_tool_dir(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    """Compose C and/or H steers for the tool-phase window."""
    d = None
    for k in ("C", "H"):
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return None if d is None else d


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Full episode; tool-phase gets composed C/H; report-phase gets O."""

    tool_d = _compose_tool_dir(active)
    tool_hook = (
        make_steer_hook(loaded, tool_d, alpha) if tool_d is not None else None
    )
    o_hook = None
    if "O" in active:
        ek, vk = active["O"]
        o_hook = make_steer_hook(loaded, vk, alpha * float(ek))

    def hook_for_turn(phase: str, _turn: int):
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
    S = _S_from_row(row)
    return {
        "S": S,
        "E": sync_error_norm(_e((0, 0, 0), S)),  # placeholder overwritten by caller
        "task_ok": int(bool((row.get("final") or "").strip())),
        "policy_ok": _policy_ok(S),
        "s_tool": int(row.get("s_tool") or 0),
        "s_output": int(row.get("s_output") or 0),
        "final": (row.get("final") or "")[:200],
    }


def _margin_snapshot(loaded, sc, active: dict[str, tuple[int, np.ndarray]], alpha: float) -> dict[str, float]:
    """Site-local ΔM under current cumulative tool/report steers (mechanistic collateral)."""
    msgs = {k: messages_for_channel(sc, k) for k in CHANNELS}
    out: dict[str, float] = {}
    tool_d = _compose_tool_dir(active)
    for k in CHANNELS:
        base = margin_at_site(loaded, msgs[k], SPECS[k])
        if k in ("C", "H"):
            if tool_d is None:
                out[f"dM_{k}"] = 0.0
                continue
            hook = make_steer_hook(loaded, tool_d, alpha)
        else:
            if "O" not in active:
                out[f"dM_{k}"] = 0.0
                continue
            ek, vk = active["O"]
            hook = make_steer_hook(loaded, vk, alpha * float(ek))
        s = margin_at_site(loaded, msgs[k], SPECS[k], hook=hook)
        out[f"dM_{k}"] = float(s["M"] - base["M"])
    return out


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
    """S0 free → stages with e_t recomputed; cumulative active steers."""
    # Stage 0
    base = _run_episode(sc, loaded, active={}, alpha=alpha, seed=seed)
    S = base["S"]
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    traj_e = [list(_e(mstar, S))]
    n_interv = 0
    active: dict[str, tuple[int, np.ndarray]] = {}
    stage_rows = []

    for stage_i, k in enumerate(order):
        e = _e(mstar, S)
        if e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (int(e[CH_IDX[k]]), dirs[k])
            n_interv += 1
        # Always regenerate with current cumulative active (even if this stage adds nothing)
        out = _run_episode(
            sc, loaded, active=active, alpha=alpha, seed=seed + 100 * (stage_i + 1)
        )
        S = out["S"]
        E = float(sync_error_norm(_e(mstar, S)))
        dM = _margin_snapshot(loaded, sc, active, alpha)
        # collateral: margins for channels not the one just targeted this stage
        collat = float(
            np.mean([abs(dM[f"dM_{j}"]) for j in CHANNELS if j != k])
        ) if dM else float("nan")
        stage_rows.append(
            {
                "stage": k,
                "S": list(S),
                "e": list(_e(mstar, S)),
                "E": E,
                "active": {kk: int(vv[0]) for kk, vv in active.items()},
                "n_active": len(active),
                "task_ok": out["task_ok"],
                "policy_ok": out["policy_ok"],
                **dM,
                "mean_abs_dM_collateral_vs_stage": collat,
            }
        )
        traj_S.append(list(S))
        traj_E.append(E)
        traj_e.append(list(_e(mstar, S)))

    E0, E1, E2, E3 = traj_E[0], traj_E[1], traj_E[2], traj_E[3]
    return {
        "mstar": list(mstar),
        "order": list(order),
        "S0": traj_S[0],
        "S_final": traj_S[-1],
        "E_traj": traj_E,
        "e_traj": traj_e,
        "delta_E_C": float(E0 - E1),
        "delta_E_H": float(E1 - E2),
        "delta_E_O": float(E2 - E3),
        "delta_E_total": float(E0 - E3),
        "hit": int(traj_S[-1] == list(mstar)),
        "bit_C": int(traj_S[-1][0] == mstar[0]),
        "bit_H": int(traj_S[-1][1] == mstar[1]),
        "bit_O": int(traj_S[-1][2] == mstar[2]),
        "n_interventions": n_interv,
        "task_ok_final": stage_rows[-1]["task_ok"] if stage_rows else base["task_ok"],
        "policy_ok_final": stage_rows[-1]["policy_ok"] if stage_rows else base["policy_ok"],
        "stages": stage_rows,
    }


def _agg(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([t[key] for t in trials]))

    return {
        "n": len(trials),
        "P_hit": mean("hit"),
        "P_C": mean("bit_C"),
        "P_H": mean("bit_H"),
        "P_O": mean("bit_O"),
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
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-diagnostic-orders", action="store_true")
    args = ap.parse_args()

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
    banks = _load_dirs(args.seed)

    results: dict[str, Any] = {"primary": {}, "diagnostic_orders": {}}

    print("=== primary order C→H→O ===", flush=True)
    for arm, dirs in banks.items():
        trials = []
        for mi, mstar in enumerate(SELECTED_MSTAR):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r
                print(f"  {arm} m*={mstar} r={r}", flush=True)
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
                tr["arm"] = arm
                trials.append(tr)
                print(
                    f"    E:{tr['E_traj']} hit={tr['hit']} S={tr['S_final']}",
                    flush=True,
                )
        results["primary"][arm] = {"trials": trials, "agg": _agg(trials)}

    if not args.skip_diagnostic_orders:
        print("=== diagnostic orders (converted only, fewer) ===", flush=True)
        dirs_c = banks["converted"]
        for order in DIAGNOSTIC_ORDERS:
            key = "→".join(order)
            trials = []
            for mi, mstar in enumerate(SELECTED_MSTAR):
                for r in range(max(1, args.reps // 2)):
                    seed = args.seed + 5000 + 1000 * mi + 10 * r + hash(key) % 97
                    tr = run_sequential_trial(
                        sc,
                        loaded,
                        mstar=mstar,
                        dirs=dirs_c,
                        alpha=args.alpha,
                        order=order,
                        seed=seed,
                    )
                    trials.append(tr)
            results["diagnostic_orders"][key] = {"trials": trials, "agg": _agg(trials)}
            print(f"  {key}: P_hit={results['diagnostic_orders'][key]['agg']['P_hit']:.2f}", flush=True)

    pc = results["primary"]["converted"]["agg"]
    pp = results["primary"]["predictive"]["agg"]
    pr = results["primary"]["random"]["agg"]

    gate = {
        "hypothesis": (
            "Converted sequential control produces greater synchronization "
            "with lower cross-channel interference than predictive control."
        ),
        "converted_beats_predictive_delta_E": bool(
            pc["mean_delta_E_total"] > pp["mean_delta_E_total"]
        ),
        "converted_beats_predictive_hit": bool(pc["P_hit"] >= pp["P_hit"]),
        "converted_beats_random_delta_E": bool(
            pc["mean_delta_E_total"] > pr["mean_delta_E_total"]
        ),
        "ranking_delta_E_conv_ge_pred_ge_rand": bool(
            pc["mean_delta_E_total"] >= pp["mean_delta_E_total"] - 1e-9
            and pp["mean_delta_E_total"] >= pr["mean_delta_E_total"] - 0.05
        ),
        "ready_for_8way": False,  # set below
    }
    gate["ready_for_8way"] = bool(
        gate["converted_beats_predictive_delta_E"]
        and gate["converted_beats_random_delta_E"]
        and pc["mean_delta_E_total"] > 0
    )

    # Order dependence (converted)
    order_note = {}
    if results["diagnostic_orders"]:
        primary_hit = pc["P_hit"]
        for k, blob in results["diagnostic_orders"].items():
            order_note[k] = {
                "P_hit": blob["agg"]["P_hit"],
                "mean_delta_E": blob["agg"]["mean_delta_E_total"],
                "vs_primary_delta_E": float(
                    pc["mean_delta_E_total"] - blob["agg"]["mean_delta_E_total"]
                ),
            }

    payload = {
        "protocol": "Phase 4C sequential C→H→O",
        "alpha": args.alpha,
        "reps": args.reps,
        "mstar_set": [list(m) for m in SELECTED_MSTAR],
        "primary_order": list(PRIMARY_ORDER),
        "results": {
            "primary": {
                arm: {"agg": results["primary"][arm]["agg"]} for arm in results["primary"]
            },
            "diagnostic_orders": {
                k: {"agg": v["agg"]} for k, v in results["diagnostic_orders"].items()
            },
        },
        "primary_trials": {
            arm: results["primary"][arm]["trials"] for arm in results["primary"]
        },
        "order_dependence": order_note,
        "gate": gate,
        "claim": (
            "Converted directions compose with substantially greater independence "
            "than predictive; sequential closed-loop tests whether conversion enables "
            "composition along the agent causal trajectory."
        ),
    }
    # Drop huge trial dumps? Keep for analysis - might be large. Keep agg in md; trials in json.

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 4C — Sequential C→H→O composition",
        "",
        r"> **Hypothesis:** Converted sequential control produces greater synchronization",
        r"> with lower cross-channel interference than predictive control.",
        "",
        r"Closed-loop: \(e_t=m^*-S_t\) recomputed every stage. Cumulative sparse steers.",
        "",
        f"m* set: {list(SELECTED_MSTAR)} · reps={args.reps} · α={args.alpha}",
        "",
        "## Primary order C→H→O",
        "",
        "| Arm | P(hit) | P(C) | P(H) | P(O) | ΔE tot | E0→E3 | task✓ | policy✓ |",
        "|-----|--------|------|------|------|--------|-------|-------|---------|",
    ]
    for arm in ("converted", "predictive", "random"):
        a = results["primary"][arm]["agg"]
        et = a["mean_E_traj"]
        lines.append(
            f"| {arm} | {a['P_hit']:.2f} | {a['P_C']:.2f} | {a['P_H']:.2f} | {a['P_O']:.2f} | "
            f"{a['mean_delta_E_total']:+.2f} | "
            f"{et[0]:.2f}→{et[1]:.2f}→{et[2]:.2f}→{et[3]:.2f} | "
            f"{a['P_task_ok']:.2f} | {a['P_policy_ok']:.2f} |"
        )

    lines += [
        "",
        "### Per-stage ΔE (primary)",
        "",
        "| Arm | ΔE_C (0→1) | ΔE_H (1→2) | ΔE_O (2→3) |",
        "|-----|------------|------------|------------|",
    ]
    for arm in ("converted", "predictive", "random"):
        a = results["primary"][arm]["agg"]
        lines.append(
            f"| {arm} | {a['mean_delta_E_C']:+.2f} | {a['mean_delta_E_H']:+.2f} | {a['mean_delta_E_O']:+.2f} |"
        )

    if order_note:
        lines += [
            "",
            "## Order dependence (converted diagnostic)",
            "",
            "| Order | P(hit) | ΔE tot | vs primary ΔE |",
            "|-------|--------|--------|---------------|",
            f"| C→H→O (primary) | {pc['P_hit']:.2f} | {pc['mean_delta_E_total']:+.2f} | 0.00 |",
        ]
        for k, o in order_note.items():
            lines.append(
                f"| {k} | {o['P_hit']:.2f} | {o['mean_delta_E']:+.2f} | {o['vs_primary_delta_E']:+.2f} |"
            )

    lines += [
        "",
        "## Gate",
        "",
        f"- Converted ΔE > predictive: **{gate['converted_beats_predictive_delta_E']}**",
        f"- Converted hit ≥ predictive: **{gate['converted_beats_predictive_hit']}**",
        f"- Ranking conv ≥ pred ≥ rand (ΔE): **{gate['ranking_delta_E_conv_ge_pred_ge_rand']}**",
        f"- Ready for 8-way evaluation: **{gate['ready_for_8way']}**",
        "",
        "8-way remains a scaling demonstration of the composition principle — not the Phase 4 claim.",
        "",
        "> Converted directions exhibit strong diagonal dominance in pairwise composition;",
        "> non-adjacent interventions can still produce measurable downstream coupling (C+O→H).",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "primary_agg": results["primary"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
