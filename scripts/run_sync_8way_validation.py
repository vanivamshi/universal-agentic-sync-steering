#!/usr/bin/env python3
"""8-way validation — frozen causal conversion on all m* ∈ {0,1}^3.

Opens the full-state sync experiment after Phases 1–6 diagnosis.

Arms (identical targets / seeds / sites / α / C→H→O logic):
  none       — reobserve stages with no steers (natural + resampling baseline)
  predictive — v_p from sync_channel_V_L4.json
  converted  — v_c from sync_channel_Vc_L4.json  (intervention under test)
  random     — matched-dim random unit directions

Questions:
  A) Does converted improve sync vs baseline/random?  (primary)
  B) Reliable arbitrary m*?                           (hard; not required)

Do NOT invent new directions. Report aggregate + per-m* + per-bit.

  .venv/bin/python scripts/run_sync_8way_validation.py --reps 2
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
from scripts.sync_channel_margins import unit  # noqa: E402
from scripts.sync_eq import all_m_star_masks, extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_8way_validation.json"
MD = ROOT / "data" / "results" / "sync_8way_validation.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260909
ALPHA = 1.5
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


def run_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray] | None,
    alpha: float,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    base = _run_episode(sc, loaded, active={}, alpha=alpha, seed=seed)
    S = base["S"]
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    active: dict[str, tuple[int, np.ndarray]] = {}
    n_interv = 0

    for si, k in enumerate(ORDER):
        e = _e(mstar, S)
        if arm != "none" and dirs is not None and e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (int(e[CH_IDX[k]]), dirs[k])
            n_interv += 1
        out = _run_episode(sc, loaded, active=active, alpha=alpha, seed=seed + 100 * (si + 1))
        S = out["S"]
        traj_S.append(list(S))
        traj_E.append(float(sync_error_norm(_e(mstar, S))))

    E0, E1, E2, E3 = traj_E
    return {
        "arm": arm,
        "mstar": list(mstar),
        "S0": traj_S[0],
        "S_final": traj_S[-1],
        "E_traj": traj_E,
        "delta_E_total": float(E0 - E3),
        "delta_E_C": float(E0 - E1),
        "delta_E_H": float(E1 - E2),
        "delta_E_O": float(E2 - E3),
        "hit": int(traj_S[-1] == list(mstar)),
        "hit0": int(traj_S[0] == list(mstar)),
        "bit_C": int(traj_S[-1][0] == mstar[0]),
        "bit_H": int(traj_S[-1][1] == mstar[1]),
        "bit_O": int(traj_S[-1][2] == mstar[2]),
        "n_interventions": n_interv,
    }


def _agg(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_hit0": m("hit0"),
        "P_C": m("bit_C"),
        "P_H": m("bit_H"),
        "P_O": m("bit_O"),
        "mean_delta_E_total": m("delta_E_total"),
        "mean_delta_E_C": m("delta_E_C"),
        "mean_delta_E_H": m("delta_E_H"),
        "mean_delta_E_O": m("delta_E_O"),
        "mean_E0": float(np.mean([t["E_traj"][0] for t in trials])) if trials else float("nan"),
        "mean_E3": float(np.mean([t["E_traj"][-1] for t in trials])) if trials else float("nan"),
        "mean_E_traj": [
            float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)
        ]
        if trials
        else [],
        "mean_n_interventions": m("n_interventions"),
    }


def _per_mstar(trials: list[dict]) -> dict[str, dict[str, Any]]:
    out = {}
    for m in all_m_star_masks():
        key = "".join(str(x) for x in m)
        sub = [t for t in trials if t["mstar"] == m]
        out[key] = _agg(sub)
        if sub:
            out[key]["mstar"] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--arms",
        default="none,predictive,converted,random",
        help="comma-separated subset of arms",
    )
    args = ap.parse_args()
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {ARMS}")

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
    banks = _load_banks(args.seed)
    mstars = [tuple(m) for m in all_m_star_masks()]

    all_trials: dict[str, list[dict]] = {a: [] for a in arms}
    print(f"=== 8-way validation α={args.alpha} reps={args.reps} arms={arms} ===", flush=True)

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
                    seed=seed,
                    arm=arm,
                )
                all_trials[arm].append(tr)
                print(
                    f"    E:{tr['E_traj']} hit={tr['hit']} S0={tr['S0']}→{tr['S_final']}",
                    flush=True,
                )

    aggs = {a: _agg(all_trials[a]) for a in arms}
    per = {a: _per_mstar(all_trials[a]) for a in arms}

    pc = aggs.get("converted")
    pp = aggs.get("predictive")
    pr = aggs.get("random")
    pn = aggs.get("none")

    gate: dict[str, Any] = {
        "question_A": "Does converted improve sync vs baseline/random?",
        "question_B": "Reliable arbitrary m*? (not required)",
        "eight_way": "OPEN",
    }
    if pc and pn:
        gate["converted_beats_none_delta_E"] = bool(pc["mean_delta_E_total"] > pn["mean_delta_E_total"])
        gate["converted_beats_none_hit"] = bool(pc["P_hit"] > pn["P_hit"])
    if pc and pr:
        gate["converted_beats_random_delta_E"] = bool(pc["mean_delta_E_total"] > pr["mean_delta_E_total"])
        gate["converted_beats_random_hit"] = bool(pc["P_hit"] > pr["P_hit"])
    if pc and pp:
        gate["converted_beats_predictive_delta_E"] = bool(
            pc["mean_delta_E_total"] > pp["mean_delta_E_total"]
        )
        gate["converted_beats_predictive_hit"] = bool(pc["P_hit"] > pp["P_hit"])
    if pc and pp and pr and pn:
        gate["ranking_hypothesis_vc_gt_vp_gt_rand_none"] = bool(
            pc["mean_delta_E_total"] >= pp["mean_delta_E_total"]
            and pp["mean_delta_E_total"] >= max(pr["mean_delta_E_total"], pn["mean_delta_E_total"]) - 0.05
        )
    if pc:
        gate["converted_delta_E_positive"] = bool(pc["mean_delta_E_total"] > 0)
        gate["question_A_pass"] = bool(
            gate.get("converted_beats_none_delta_E")
            and gate.get("converted_beats_random_delta_E")
            and (pc["mean_delta_E_total"] > 0 or pc["P_hit"] > (pn or {}).get("P_hit", 0))
        )
        gate["question_B_pass"] = bool(pc["P_hit"] >= 0.75)

    payload = {
        "protocol": "8-way validation C→H→O",
        "alpha": args.alpha,
        "reps": args.reps,
        "seed": args.seed,
        "order": list(ORDER),
        "arms": list(arms),
        "mstar_set": [list(m) for m in mstars],
        "agg": aggs,
        "per_mstar": per,
        "trials": {a: all_trials[a] for a in arms},
        "gate": gate,
        "claim_frame": (
            "conversion produces causal control, but causal control does not "
            "automatically yield arbitrary sequential synchronization"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 8-way validation — frozen $v_c$ vs controls",
        "",
        "> **Question A:** Does converted improve sync vs baseline/random?",
        "> **Question B:** Reliable arbitrary $m^*$? (hard; not required)",
        "",
        f"α={args.alpha}, reps={args.reps}/m*, order={'→'.join(ORDER)}, task={NEUTRAL_TASK}",
        "",
        "## Aggregate",
        "",
        "| Arm | P(hit) | P(hit₀) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) | E₀→E₃ |",
        "|-----|--------|---------|-----|------|------|------|------|------|------|-------|",
    ]
    for a in arms:
        g = aggs[a]
        et_s = "→".join(f"{x:.2f}" for x in g["mean_E_traj"]) if g["mean_E_traj"] else "—"
        lines.append(
            f"| {a} | {g['P_hit']:.2f} | {g['P_hit0']:.2f} | {g['mean_delta_E_total']:+.2f} | "
            f"{g['mean_delta_E_C']:+.2f} | {g['mean_delta_E_H']:+.2f} | {g['mean_delta_E_O']:+.2f} | "
            f"{g['P_C']:.2f} | {g['P_H']:.2f} | {g['P_O']:.2f} | {et_s} |"
        )
    lines += [
        "",
        "## Per $m^*$ (converted vs none)",
        "",
        "| $m^*$ | none P(hit) | converted P(hit) | none ΔE | converted ΔE |",
        "|-------|-------------:|-----------------:|--------:|-------------:|",
    ]
    for m in mstars:
        key = "".join(str(x) for x in m)
        nn = per.get("none", {}).get(key, {})
        cc = per.get("converted", {}).get(key, {})
        lines.append(
            f"| {key} | {nn.get('P_hit', float('nan')):.2f} | {cc.get('P_hit', float('nan')):.2f} | "
            f"{nn.get('mean_delta_E_total', float('nan')):+.2f} | "
            f"{cc.get('mean_delta_E_total', float('nan')):+.2f} |"
        )
    lines += [
        "",
        "## Per $m^*$ (all arms P(hit))",
        "",
        "| $m^*$ | " + " | ".join(arms) + " |",
        "|-------|" + "|".join(["-----:" for _ in arms]) + "|",
    ]
    for m in mstars:
        key = "".join(str(x) for x in m)
        cells = [f"{per.get(a, {}).get(key, {}).get('P_hit', float('nan')):.2f}" for a in arms]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Gate",
        "",
        f"- Question A (converted useful): **{gate.get('question_A_pass', 'n/a')}**",
        f"- Question B (reliable arbitrary): **{gate.get('question_B_pass', 'n/a')}**",
        f"- converted ΔE > 0: **{gate.get('converted_delta_E_positive', 'n/a')}**",
        f"- converted beats none (ΔE): **{gate.get('converted_beats_none_delta_E', 'n/a')}**",
        f"- converted beats random (ΔE): **{gate.get('converted_beats_random_delta_E', 'n/a')}**",
        f"- converted beats predictive (ΔE): **{gate.get('converted_beats_predictive_delta_E', 'n/a')}**",
        f"- ranking $v_c ≥ v_p ≥$ rand/none: **{gate.get('ranking_hypothesis_vc_gt_vp_gt_rand_none', 'n/a')}**",
        "",
        "Directions frozen. No new $v$.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "agg": aggs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
