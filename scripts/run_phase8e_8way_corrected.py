#!/usr/bin/env python3
"""Phase 8E — Full 8-way validation with Phase-8D corrected controller.

Before/after test vs legacy 8-way converted:
  P(hit)=0.11, ΔE=+0.06

Controller (identical to 8D for all steered arms):
  d_k = (2 m*_k − 1) v_k
  C→H_decision-token→O
  same seed, carry PLAN into H, skip noop stages

Arms: none / predictive / converted / random — same path; only V differs.
No new v, no policy, no new objective.

  .venv/bin/python scripts/run_phase8e_8way_corrected.py --reps 2
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

OUT = ROOT / "data" / "results" / "sync_phase8e_8way_corrected.json"
MD = ROOT / "data" / "results" / "sync_phase8e_8way_corrected.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260909  # match legacy 8-way seed family
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ORDER = ("C", "H", "O")
ARMS = ("none", "predictive", "converted", "random")

# Legacy 8-way converted (reps=8) — before/after anchor
LEGACY_CONVERTED = {"P_hit": 0.11, "mean_delta_E_total": 0.06}


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


def _s_star(mstar_k: int) -> int:
    return int(2 * int(mstar_k) - 1)


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    alpha: float,
    seed: int,
    h_plan_prefill: str | None = None,
) -> dict[str, Any]:
    """8D episode: target-sign; H = PLAN prefill + steered continuation."""
    h_hook = plan_hook = o_hook = tool_c_hook = None

    if "H" in active:
        sH, vH = active["H"]
        h_hook = make_steer_hook(loaded, vH, alpha * float(sH))
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
    plan = extract_plan((row.get("final") or "") + "\n" + (row.get("messages_text") or ""))
    prefill = (plan if plan.endswith("\n") else plan + "\n") if plan else ""
    return {"S": S, "plan_prefill": prefill}


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
    """8D sequential path: same seed, skip noop, carry PLAN into H."""
    base = _run_episode(sc, loaded, active={}, alpha=alpha, seed=seed)
    S = base["S"]
    plan_prefill = base.get("plan_prefill") or ""
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    active: dict[str, tuple[int, np.ndarray]] = {}
    n_interv = 0
    n_episodes = 1

    for k in ORDER:
        e = _e(mstar, S)
        added = False
        if arm != "none" and dirs is not None and e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (_s_star(mstar[CH_IDX[k]]), dirs[k])
            n_interv += 1
            added = True
        if not added:
            E = traj_E[-1]
            traj_S.append(list(S))
            traj_E.append(E)
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
        "n_episodes": n_episodes,
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
        "mean_n_episodes": m("n_episodes"),
    }


def _per_mstar(trials: list[dict]) -> dict[str, dict[str, Any]]:
    out = {}
    for m in all_m_star_masks():
        key = "".join(str(x) for x in m)
        sub = [t for t in trials if t["mstar"] == list(m)]
        out[key] = _agg(sub)
        if sub:
            out[key]["mstar"] = list(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arms", default="none,predictive,converted,random")
    args = ap.parse_args()
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}")

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
    print(
        f"=== Phase 8E 8-way corrected α={args.alpha} reps={args.reps} arms={arms} ===",
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
                    sc, loaded, mstar=mstar, dirs=dirs, alpha=args.alpha, seed=seed, arm=arm
                )
                all_trials[arm].append(tr)
                print(
                    f"    E:{[round(x, 2) for x in tr['E_traj']]} hit={tr['hit']} "
                    f"S0={tr['S0']}→{tr['S_final']} eps={tr['n_episodes']}",
                    flush=True,
                )

    aggs = {a: _agg(all_trials[a]) for a in arms}
    per = {a: _per_mstar(all_trials[a]) for a in arms}
    pc, pp, pr, pn = (aggs.get(a) for a in ("converted", "predictive", "random", "none"))

    gate: dict[str, Any] = {
        "question": "Does 8D-corrected controller improve full 8-way vs legacy converted?",
        "legacy_converted": LEGACY_CONVERTED,
        "controller": "8D: target-sign + H decision-token + same-seed + skip-noop",
    }
    if pc:
        gate["converted_P_hit"] = pc["P_hit"]
        gate["converted_delta_E"] = pc["mean_delta_E_total"]
        gate["beats_legacy_hit"] = bool(pc["P_hit"] > LEGACY_CONVERTED["P_hit"] + 0.03)
        gate["beats_legacy_delta_E"] = bool(
            pc["mean_delta_E_total"] > LEGACY_CONVERTED["mean_delta_E_total"] + 0.05
        )
        gate["converted_delta_E_positive"] = bool(pc["mean_delta_E_total"] > 0)
    if pc and pn:
        gate["converted_beats_none_delta_E"] = bool(
            pc["mean_delta_E_total"] > pn["mean_delta_E_total"]
        )
        gate["converted_beats_none_hit"] = bool(pc["P_hit"] > pn["P_hit"])
    if pc and pr:
        gate["converted_beats_random_delta_E"] = bool(
            pc["mean_delta_E_total"] > pr["mean_delta_E_total"]
        )
        gate["converted_beats_random_hit"] = bool(pc["P_hit"] > pr["P_hit"])
    if pc and pp:
        gate["converted_beats_predictive_delta_E"] = bool(
            pc["mean_delta_E_total"] > pp["mean_delta_E_total"]
        )
        gate["converted_beats_predictive_hit"] = bool(pc["P_hit"] > pp["P_hit"])
    gate["question_A_pass"] = bool(
        gate.get("beats_legacy_delta_E")
        or (
            gate.get("converted_beats_none_delta_E")
            and gate.get("converted_delta_E_positive")
        )
    )
    gate["read"] = (
        "Before/after: legacy converted P(hit)=0.11 ΔE=+0.06. "
        "Corrected 8-way uses 8D controller only — no new v."
    )

    payload = {
        "protocol": "Phase 8E 8-way with 8D-corrected controller",
        "alpha": args.alpha,
        "reps": args.reps,
        "seed": args.seed,
        "order": list(ORDER),
        "arms": list(arms),
        "mstar_set": [list(m) for m in mstars],
        "legacy_converted_ref": LEGACY_CONVERTED,
        "agg": aggs,
        "per_mstar": per,
        "trials": {a: all_trials[a] for a in arms},
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8E — 8-way with corrected (8D) controller",
        "",
        "> Before/after vs legacy converted: $P(\\mathrm{hit})=0.11$, $\\Delta E=+0.06$.",
        "",
        r"Controller: $d_k=(2m_k^*-1)v_k$, H = decision-token-only, same seed, skip-noop.",
        "",
        f"α={args.alpha}, reps={args.reps}/m*, order=C→H→O, task={NEUTRAL_TASK}. No new $v$.",
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
        f"| **legacy converted** | **{LEGACY_CONVERTED['P_hit']:.2f}** | — | "
        f"**+{LEGACY_CONVERTED['mean_delta_E_total']:.2f}** | — | — | — | — | — | — | — |",
        "",
        "## Per $m^*$ $P(\\mathrm{hit}\\mid m^*)$",
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
        "## Per $m^*$ ΔE (converted vs none)",
        "",
        "| $m^*$ | none ΔE | converted ΔE | none P(hit) | converted P(hit) |",
        "|-------|--------:|-------------:|------------:|-----------------:|",
    ]
    for m in mstars:
        key = "".join(str(x) for x in m)
        nn = per.get("none", {}).get(key, {})
        cc = per.get("converted", {}).get(key, {})
        lines.append(
            f"| {key} | {nn.get('mean_delta_E_total', float('nan')):+.2f} | "
            f"{cc.get('mean_delta_E_total', float('nan')):+.2f} | "
            f"{nn.get('P_hit', float('nan')):.2f} | {cc.get('P_hit', float('nan')):.2f} |"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- Converted P(hit): **{gate.get('converted_P_hit', 'n/a')}** (legacy 0.11)",
        f"- Converted ΔE: **{gate.get('converted_delta_E', 'n/a')}** (legacy +0.06)",
        f"- Beats legacy hit: **{gate.get('beats_legacy_hit', 'n/a')}**",
        f"- Beats legacy ΔE: **{gate.get('beats_legacy_delta_E', 'n/a')}**",
        f"- Beats none (ΔE): **{gate.get('converted_beats_none_delta_E', 'n/a')}**",
        f"- Beats random (ΔE): **{gate.get('converted_beats_random_delta_E', 'n/a')}**",
        f"- Beats predictive (ΔE): **{gate.get('converted_beats_predictive_delta_E', 'n/a')}**",
        f"- Question A (corrected 8-way useful): **{gate.get('question_A_pass', 'n/a')}**",
        "",
        gate["read"],
        "",
        "Claim scope: reps=2 — directional before/after, not a stable rate claim.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "agg": aggs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
