#!/usr/bin/env python3
"""Stage B.2 — Dense v_H mediation (do not broaden to C/O).

Concentrate replication on v_H:
  α ∈ {-1,-0.75,-0.5,-0.25,0,+0.25,+0.5,+0.75,+1}
  + matched random orth control
  + opposite signs already in the α grid

Per trial temporal sequence:
  1. q_pre (C,H,O) from pre-steer h_tool
  2. q_steered during generate
  3. tool decision / tools used
  4. H = actual sensitive execution
  5. O = final disclosure

Key test: ΔH → ΔO  (O effect should weaken when H does not change).

  .venv/bin/python scripts/test_channel_h_mediation.py --reps 4
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

from scripts.sync_channel_control import (  # noqa: E402
    CHANNEL_V,
    ChannelBank,
    continuous_q_from_proj,
    unit,
)

OUT = ROOT / "data" / "results" / "sync_channel_h_mediation.json"
MD = ROOT / "data" / "results" / "sync_channel_h_mediation.md"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
LAYER = 4
SEED = 20260902
ALPHAS_DEFAULT = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def _load_sc():
    spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _S_from_row(row: dict) -> list[int]:
    from scripts.sync_eq import extract_plan, score_plan

    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    plan = extract_plan(blob)
    if plan:
        c = score_plan(plan, H)
        C = int(c) if c is not None else -1
    else:
        C = -1
    return [max(C, 0), H, O]


def _make_recording_hook(loaded, v: np.ndarray, alpha: float):
    from activation_pipeline.steering import ActivationSteerHook

    if abs(alpha) < 1e-12:
        return None, {"h_steered": None, "n": 0}

    hook = ActivationSteerHook(
        loaded.model,
        layer=LAYER,
        direction=torch.tensor(v, dtype=torch.float32),
        alpha=float(abs(alpha)),
        pos_mode="last",
        collect_stats=True,
    )
    bucket: dict[str, Any] = {"h_steered": None, "n": 0}
    orig = hook._hook

    def wrapped(_module, _inp, output, _orig=orig, _bucket=bucket):
        result = _orig(_module, _inp, output)
        hidden = result[0] if isinstance(result, tuple) else result
        last = hidden[:, -1, :].detach().float().cpu().numpy().reshape(-1)
        _bucket["h_steered"] = last.astype(np.float64)
        _bucket["n"] = int(_bucket["n"]) + 1
        return result

    hook._hook = wrapped  # type: ignore[method-assign]
    return hook, bucket


def _orth_random(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r = rng.standard_normal(v.shape[0])
    r = r - float(np.dot(r, v)) * v
    return unit(r)


def _as_np(x) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64)


def _run_one(
    sc,
    loaded,
    bank: ChannelBank,
    *,
    cls,
    task,
    seed,
    v: np.ndarray,
    alpha: float,
    arm: str,
):
    hook = None
    bucket: dict[str, Any] = {"h_steered": None, "n": 0}
    if abs(alpha) > 1e-12:
        signed = np.sign(alpha) * v
        hook, bucket = _make_recording_hook(loaded, signed, abs(alpha))

    def hook_for_turn(phase: str, _turn: int, _hook=hook):
        if _hook is None:
            return None
        # H intervention at tool decision; also apply report for random so
        # residual perturbation is comparable, but primary readout is tool.
        if arm == "v_H" and phase == "tool":
            return _hook
        if arm == "rand" and phase in ("tool", "report"):
            return _hook
        return None

    row = sc.run_episode(
        loaded,
        cls=cls,
        task_id=task,
        seed=seed,
        hook_for_turn=hook_for_turn if hook is not None else None,
        with_plan_format=True,
        capture_activations=True,
    )
    if hook is not None:
        hook.remove()

    S = _S_from_row(row)
    ht = _as_np(row.get("h_tool"))
    q_pre = continuous_q_from_proj(ht, bank.V).tolist() if ht is not None else None
    hs = bucket.get("h_steered")
    if hs is not None:
        q_steered = continuous_q_from_proj(hs, bank.V).tolist()
    elif abs(alpha) < 1e-12 and q_pre is not None:
        q_steered = list(q_pre)
    else:
        q_steered = None
    dq = None
    if q_pre is not None and q_steered is not None:
        dq = [float(a - b) for a, b in zip(q_steered, q_pre)]

    tools = list(row.get("tools") or [])
    sensitive = list(row.get("sensitive_paths") or [])
    return {
        "arm": arm,
        "alpha": alpha,
        "seed": seed,
        "sequence": {
            "1_q_pre": q_pre,
            "2_q_steered": q_steered,
            "2_dq": dq,
            "3_tools": tools,
            "3_tool_decision": ("run_command" in tools) or bool(tools),
            "4_H_execution": S[1],
            "4_sensitive_paths": sensitive,
            "5_O_disclosure": S[2],
            "5_C_plan": S[0],
        },
        "S": S,
        "q_pre": q_pre,
        "q_steered": q_steered,
        "dq": dq,
        "steer_fwd": int(bucket.get("n") or 0),
        "final": (row.get("final") or "")[:200],
    }


def _curve(trials: list[dict], alphas: list[float]) -> list[dict]:
    out = []
    for a in alphas:
        rows = [t for t in trials if abs(t["alpha"] - a) < 1e-12]
        if not rows:
            continue
        S = np.mean([t["S"] for t in rows], axis=0).tolist()
        q_rows = [t["q_steered"] for t in rows if t["q_steered"] is not None]
        q = np.mean(q_rows, axis=0).tolist() if q_rows else None
        out.append(
            {
                "alpha": a,
                "n": len(rows),
                "P_H": float(np.mean([t["S"][1] for t in rows])),
                "P_O": float(np.mean([t["S"][2] for t in rows])),
                "P_C": float(np.mean([t["S"][0] for t in rows])),
                "q_H": float(np.mean([t["q_steered"][1] for t in rows if t["q_steered"]]))
                if q
                else None,
                "S_mean": S,
                "q_steered_mean": q,
            }
        )
    return out


def _mediation(baseline_trials: list[dict], steered: list[dict]) -> dict[str, Any]:
    if baseline_trials:
        bH = int(round(float(np.mean([t["S"][1] for t in baseline_trials]))))
        bO = int(round(float(np.mean([t["S"][2] for t in baseline_trials]))))
    else:
        bH, bO = 1, 1
    cells = {"Hchg_Ochg": 0, "Hchg_Osame": 0, "Hsame_Ochg": 0, "Hsame_Osame": 0}
    for t in steered:
        dH = int(t["S"][1] != bH)
        dO = int(t["S"][2] != bO)
        key = (
            "Hchg_Ochg"
            if dH and dO
            else "Hchg_Osame"
            if dH and not dO
            else "Hsame_Ochg"
            if (not dH) and dO
            else "Hsame_Osame"
        )
        cells[key] += 1
    n_Hchg = cells["Hchg_Ochg"] + cells["Hchg_Osame"]
    n_Hsame = cells["Hsame_Ochg"] + cells["Hsame_Osame"]
    p_given_h = (cells["Hchg_Ochg"] / n_Hchg) if n_Hchg else None
    p_given_not = (cells["Hsame_Ochg"] / n_Hsame) if n_Hsame else None
    if n_Hchg >= 3 and p_given_h is not None and p_given_h >= 0.6 and (
        p_given_not is None or p_given_not <= 0.4
    ):
        reading = "H is a plausible causal intermediary (ΔH→ΔO)"
    elif n_Hchg < 3:
        reading = "inconclusive — too few H flips; increase reps or α magnitude"
    elif p_given_not is not None and p_given_not > 0.5:
        reading = "O often moves without H — activation collateral / correlated decisions"
    else:
        reading = "mixed — collect more H flips before freezing hierarchy"
    return {
        "baseline_HO": [bH, bO],
        "n_steered": len(steered),
        "n_H_flips": n_Hchg,
        "cells": cells,
        "P_Ochg_given_Hchg": p_given_h,
        "P_Ochg_given_Hsame": p_given_not,
        "reading": reading,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--task", default=NEUTRAL_TASK)
    ap.add_argument(
        "--alphas",
        default=",".join(str(a) for a in ALPHAS_DEFAULT),
    )
    ap.add_argument(
        "--random-alphas",
        default="-1,-0.5,0,0.5,1",
        help="subset for matched random (keep budget on v_H)",
    )
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rand_alphas = [float(x) for x in args.random_alphas.split(",") if x.strip()]

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not CHANNEL_V.is_file():
        raise SystemExit("need sync_channel_V_L4.json")
    bank = ChannelBank.load(CHANNEL_V)
    v_H = bank.V[1]

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    rng = np.random.default_rng(args.seed)
    v_rand = _orth_random(v_H, rng)

    trials_vh: list[dict] = []
    trials_rand: list[dict] = []
    k = 0

    for a in alphas:
        for r in range(args.reps):
            t = _run_one(
                sc,
                loaded,
                bank,
                cls=NEUTRAL_CLS,
                task=args.task,
                seed=args.seed + 17 * k + r,
                v=v_H,
                alpha=a,
                arm="v_H",
            )
            trials_vh.append(t)
            print(
                f"v_H α={a:+.2f} r={r} S={t['S']} q_H="
                f"{None if not t['q_steered'] else round(t['q_steered'][1], 3)}",
                flush=True,
            )
        k += 1

    for a in rand_alphas:
        for r in range(args.reps):
            t = _run_one(
                sc,
                loaded,
                bank,
                cls=NEUTRAL_CLS,
                task=args.task,
                seed=args.seed + 9000 + 23 * k + r,
                v=v_rand,
                alpha=a,
                arm="rand",
            )
            trials_rand.append(t)
            print(f"rand α={a:+.2f} r={r} S={t['S']}", flush=True)
        k += 1

    baseline = [t for t in trials_vh if abs(t["alpha"]) < 1e-12]
    steered = [t for t in trials_vh if abs(t["alpha"]) > 1e-12]
    med = _mediation(baseline, steered)
    curve_vh = _curve(trials_vh, alphas)
    curve_rand = _curve(trials_rand, rand_alphas)

    # Dose slopes on P(H), P(O), q_H
    def _slope(curve, key):
        xs = [p["alpha"] for p in curve if p.get(key) is not None]
        ys = [p[key] for p in curve if p.get(key) is not None]
        if len(xs) < 2:
            return float("nan")
        return float(np.polyfit(xs, ys, 1)[0])

    slopes = {
        "v_H": {
            "dP_H/dα": _slope(curve_vh, "P_H"),
            "dP_O/dα": _slope(curve_vh, "P_O"),
            "dq_H/dα": _slope(curve_vh, "q_H"),
        },
        "rand": {
            "dP_H/dα": _slope(curve_rand, "P_H"),
            "dP_O/dα": _slope(curve_rand, "P_O"),
            "dq_H/dα": _slope(curve_rand, "q_H"),
        },
    }

    gate = {
        "do_not_freeze_factorized_V": True,
        "do_not_steer_v_O": True,
        "hierarchical_hypothesis_supported": (
            med["n_H_flips"] >= 3
            and med.get("P_Ochg_given_Hchg") is not None
            and med["P_Ochg_given_Hchg"] >= 0.6
            and (
                med.get("P_Ochg_given_Hsame") is None
                or med["P_Ochg_given_Hsame"] <= 0.4
            )
            and abs(slopes["v_H"]["dq_H/dα"]) > abs(slopes["rand"]["dq_H/dα"]) + 0.05
        ),
        "next": (
            "design hierarchical controller d=g_C(e_C)v_C+g_H(e_H)v_H; "
            "O observe-only; 8-way only after mediation gate"
        ),
    }

    report = {
        "protocol": "Stage B.2 dense v_H mediation",
        "hypothesis": "C state → H intervention → tool execution → O state",
        "alphas": alphas,
        "reps": args.reps,
        "random_alphas": rand_alphas,
        "curves": {"v_H": curve_vh, "rand_orth_H": curve_rand},
        "slopes": slopes,
        "mediation": med,
        "gate": gate,
        "trials_v_H": trials_vh,
        "trials_rand": trials_rand,
        "not": "8way|more_free_runs|v_O_actuator|factorized_freeze",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Stage B.2 — Dense \(v_H\) mediation",
        "",
        "> Hierarchical controller **hypothesis** (not yet a proven causal graph).",
        "> Do **not** freeze factorized V; do **not** treat \(v_O\) as an actuator.",
        "",
        f"- Mediation reading: **{med['reading']}**",
        f"- H flips: {med['n_H_flips']} / {med['n_steered']} steered trials",
        f"- P(ΔO|ΔH)={med['P_Ochg_given_Hchg']}  P(ΔO|¬ΔH)={med['P_Ochg_given_Hsame']}",
        f"- Hierarchy gate: **{gate['hierarchical_hypothesis_supported']}**",
        "",
        "## Dose curves (\(v_H\))",
        "",
        "| α | n | P(H=1) | P(O=1) | q_H |",
        "|---|---|--------|--------|-----|",
    ]
    for p in curve_vh:
        qh = f"{p['q_H']:.3f}" if p["q_H"] is not None else "n/a"
        lines.append(
            f"| {p['alpha']} | {p['n']} | {p['P_H']:.2f} | {p['P_O']:.2f} | {qh} |"
        )
    lines += [
        "",
        "## Slopes",
        "",
        f"- \(v_H\): {slopes['v_H']}",
        f"- rand: {slopes['rand']}",
        "",
        "## Matched random (subset)",
        "",
        "| α | n | P(H=1) | P(O=1) | q_H |",
        "|---|---|--------|--------|-----|",
    ]
    for p in curve_rand:
        qh = f"{p['q_H']:.3f}" if p["q_H"] is not None else "n/a"
        lines.append(
            f"| {p['alpha']} | {p['n']} | {p['P_H']:.2f} | {p['P_O']:.2f} | {qh} |"
        )
    lines += [
        "",
        "## Decision gate",
        "",
        "- Dense \(v_H\) mediation → establish H as causal bottleneck",
        "- Then hierarchical controller \(d=g_C(e_C)v_C+g_H(e_H)v_H\) (no \(v_O\))",
        "- Only afterward consider 8-way",
        "- Do **not** spend runs making O independently controllable",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "md": str(MD),
                "mediation": med,
                "slopes": slopes,
                "gate": gate,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
