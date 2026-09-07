#!/usr/bin/env python3
"""Stage B.1–B.3 — Dose-response causal tests (continuous q + binary S).

Do NOT collect more free runs; use existing free-run V candidates.
Do NOT run 8-way until B.4 architecture choice.

For each channel direction v_i and matched random control:
  α ∈ alphas
  measure:
    q_steered = σ(h_steered · V)   continuous preference on *steered* residual
    q_pre     = σ(h_pre · V)       pre-steer baseline (scenario capture)
    S = (C,H,O)                    binary behavior
  For H steers: mediation table P(ΔO | ΔH) vs P(ΔO | ¬ΔH).

  .venv/bin/python scripts/test_channel_dose_response.py --reps 2
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
    CHANNELS,
    continuous_q_from_proj,
    unit,
)

OUT = ROOT / "data" / "results" / "sync_channel_dose_response.json"
MD = ROOT / "data" / "results" / "sync_channel_dose_response.md"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
LAYER = 4
SEED = 20260902
ALPHAS_DEFAULT = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)


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
    """Steer hook that records the last steered residual (post +αd)."""
    from activation_pipeline.steering import ActivationSteerHook

    if abs(alpha) < 1e-12:
        return None, {}

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


def _run_one(
    sc,
    loaded,
    bank: ChannelBank,
    *,
    cls,
    task,
    seed,
    ch: str | None,
    v: np.ndarray | None,
    alpha: float,
):
    hook = None
    bucket: dict[str, Any] = {"h_steered": None, "n": 0}
    if v is not None and abs(alpha) > 1e-12:
        signed = np.sign(alpha) * v
        hook, bucket = _make_recording_hook(loaded, signed, abs(alpha))

    def hook_for_turn(phase: str, _turn: int, _hook=hook, _ch=ch):
        if _hook is None:
            return None
        if _ch == "H" and phase == "tool":
            return _hook
        if _ch in ("C", "O") and phase == "report":
            return _hook
        if _ch == "rand" and phase in ("tool", "report"):
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
    ht = row.get("h_tool")
    hr = row.get("h_report")
    if isinstance(ht, list):
        ht = np.asarray(ht, dtype=np.float64)
    if isinstance(hr, list):
        hr = np.asarray(hr, dtype=np.float64)

    q_pre_tool = continuous_q_from_proj(ht, bank.V) if ht is not None else None
    q_pre_report = continuous_q_from_proj(hr, bank.V) if hr is not None else None
    # Phase-matched pre-steer baseline
    if ch == "H":
        q_pre = q_pre_tool
    else:
        q_pre = q_pre_report if q_pre_report is not None else q_pre_tool

    hs = bucket.get("h_steered")
    if hs is not None:
        q_steered = continuous_q_from_proj(hs, bank.V)
    elif abs(alpha) < 1e-12 and q_pre is not None:
        q_steered = q_pre
    else:
        q_steered = None

    q_pre_l = q_pre.tolist() if q_pre is not None else None
    q_st_l = q_steered.tolist() if q_steered is not None else None
    dq = None
    if q_pre_l is not None and q_st_l is not None:
        dq = [float(a - b) for a, b in zip(q_st_l, q_pre_l)]

    return {
        "S": S,
        "q_pre": q_pre_l,
        "q_steered": q_st_l,
        "dq": dq,
        "steer_fwd": int(bucket.get("n") or 0),
        "tools": list(row.get("tools") or []),
        "sensitive_paths": list(row.get("sensitive_paths") or []),
        "final": (row.get("final") or "")[:240],
    }


def _mean_lists(rows: list[list[float] | None]) -> list[float] | None:
    ok = [r for r in rows if r is not None]
    if not ok:
        return None
    return np.mean(ok, axis=0).tolist()


def _mediation_table(baseline_S: list[float], mediation: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify ΔH / ΔO vs baseline mean (rounded) for H arms."""
    bH = int(round(baseline_S[1]))
    bO = int(round(baseline_S[2]))
    cells = {"Hchg_Ochg": 0, "Hchg_Osame": 0, "Hsame_Ochg": 0, "Hsame_Osame": 0}
    for m in mediation:
        dH = int(m["H"] != bH)
        dO = int(m["O"] != bO)
        if dH and dO:
            cells["Hchg_Ochg"] += 1
        elif dH and not dO:
            cells["Hchg_Osame"] += 1
        elif (not dH) and dO:
            cells["Hsame_Ochg"] += 1
        else:
            cells["Hsame_Osame"] += 1
    n_Hchg = cells["Hchg_Ochg"] + cells["Hchg_Osame"]
    n_Hsame = cells["Hsame_Ochg"] + cells["Hsame_Osame"]
    return {
        "baseline_HO": [bH, bO],
        "cells": cells,
        "P_Ochg_given_Hchg": (cells["Hchg_Ochg"] / n_Hchg) if n_Hchg else None,
        "P_Ochg_given_Hsame": (cells["Hsame_Ochg"] / n_Hsame) if n_Hsame else None,
        "reading": (
            "behavioral mediation H→O favored"
            if n_Hchg
            and (cells["Hchg_Ochg"] / n_Hchg) > 0.5
            and (n_Hsame == 0 or (cells["Hsame_Ochg"] / max(n_Hsame, 1)) < 0.5)
            else (
                "activation collateral / direct O favored"
                if n_Hsame and cells["Hsame_Ochg"] / n_Hsame > 0.5
                else "inconclusive — need more H flips"
            )
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--task", default=NEUTRAL_TASK)
    ap.add_argument(
        "--alphas",
        default=",".join(str(a) for a in ALPHAS_DEFAULT),
        help="comma list of alpha values",
    )
    ap.add_argument(
        "--random-alphas",
        default="",
        help="alphas for random controls (default: same as --alphas). empty=all",
    )
    ap.add_argument("--skip-random", action="store_true")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if args.skip_random:
        rand_alphas: list[float] = []
    elif args.random_alphas.strip():
        rand_alphas = [float(x) for x in args.random_alphas.split(",") if x.strip()]
    else:
        rand_alphas = list(alphas)

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not CHANNEL_V.is_file():
        raise SystemExit("need sync_channel_V_L4.json from free-run learn")
    bank = ChannelBank.load(CHANNEL_V)
    if "proxy" in str(bank.meta.get("v_C_status") or "").lower():
        print("WARNING: V still marked proxy — prefer free-run candidates", flush=True)

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    rng = np.random.default_rng(args.seed)
    rand_dirs = {ch: _orth_random(bank.V[i], rng) for i, ch in enumerate(CHANNELS)}

    curves: dict[str, Any] = {}
    mediation: list[dict[str, Any]] = []
    baseline_S_acc: list[list[int]] = []
    k = 0

    for i, ch in enumerate(CHANNELS):
        v = bank.V[i]
        series = []
        for a in alphas:
            reps_S, reps_q, reps_dq = [], [], []
            for r in range(args.reps):
                out = _run_one(
                    sc,
                    loaded,
                    bank,
                    cls=NEUTRAL_CLS,
                    task=args.task,
                    seed=args.seed + 1000 * k + r,
                    ch=ch,
                    v=v,
                    alpha=a,
                )
                reps_S.append(out["S"])
                reps_q.append(out["q_steered"])
                reps_dq.append(out["dq"])
                if abs(a) < 1e-12:
                    baseline_S_acc.append(out["S"])
                if ch == "H" and abs(a) > 1e-12:
                    mediation.append(
                        {
                            "arm": f"{'+' if a > 0 else '-'}v_H@α={a}",
                            "alpha": a,
                            "S": out["S"],
                            "q_steered": out["q_steered"],
                            "dq": out["dq"],
                            "H": out["S"][1],
                            "O": out["S"][2],
                            "C": out["S"][0],
                            "sensitive_paths": out["sensitive_paths"],
                            "steer_fwd": out["steer_fwd"],
                        }
                    )
                k += 1
            S_mean = np.mean(reps_S, axis=0).tolist()
            q_mean = _mean_lists(reps_q)
            dq_mean = _mean_lists(reps_dq)
            series.append(
                {
                    "alpha": a,
                    "S_mean": S_mean,
                    "q_steered_mean": q_mean,
                    "dq_mean": dq_mean,
                    "n": args.reps,
                }
            )
            print(
                f"{ch} α={a:+.2f} S={S_mean} q={q_mean} dq={dq_mean}",
                flush=True,
            )
        curves[f"v_{ch}"] = series

        if rand_alphas:
            rv = rand_dirs[ch]
            rseries = []
            for a in rand_alphas:
                reps_S, reps_q, reps_dq = [], [], []
                for r in range(args.reps):
                    out = _run_one(
                        sc,
                        loaded,
                        bank,
                        cls=NEUTRAL_CLS,
                        task=args.task,
                        seed=args.seed + 5000 * (i + 1) + int(100 * a) + r,
                        ch="rand",
                        v=rv,
                        alpha=a,
                    )
                    reps_S.append(out["S"])
                    reps_q.append(out["q_steered"])
                    reps_dq.append(out["dq"])
                    k += 1
                rseries.append(
                    {
                        "alpha": a,
                        "S_mean": np.mean(reps_S, axis=0).tolist(),
                        "q_steered_mean": _mean_lists(reps_q),
                        "dq_mean": _mean_lists(reps_dq),
                        "n": args.reps,
                    }
                )
                print(
                    f"rand~{ch} α={a:+.2f} S={rseries[-1]['S_mean']} "
                    f"q={rseries[-1]['q_steered_mean']}",
                    flush=True,
                )
            curves[f"rand_orth_{ch}"] = rseries

    # Target-channel continuous slopes from steered q_i vs alpha
    slopes = {}
    for ch_i, ch in enumerate(CHANNELS):
        xs, ys, dys = [], [], []
        for pt in curves[f"v_{ch}"]:
            if pt["q_steered_mean"] is None:
                continue
            xs.append(pt["alpha"])
            ys.append(pt["q_steered_mean"][ch_i])
            if pt["dq_mean"] is not None:
                dys.append(pt["dq_mean"][ch_i])
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
        slopes[ch] = {
            "q_slope_vs_alpha": slope,
            "mean_dq_i": float(np.mean(dys)) if dys else None,
            "interpretation": (
                "monotonic preference shift" if abs(slope) > 0.05 else "flat ~ no continuous control"
            ),
        }

    # Binary dose: rate of target bit vs alpha
    binary_rates = {}
    for ch_i, ch in enumerate(CHANNELS):
        binary_rates[ch] = [
            {"alpha": pt["alpha"], "S_i": pt["S_mean"][ch_i], "S": pt["S_mean"]}
            for pt in curves[f"v_{ch}"]
        ]

    baseline_S = (
        np.mean(baseline_S_acc, axis=0).tolist() if baseline_S_acc else [0.0, 1.0, 0.5]
    )
    med = _mediation_table(baseline_S, mediation)

    # Architecture hint from slopes + mediation
    strong = [c for c, s in slopes.items() if abs(s["q_slope_vs_alpha"]) > 0.05]
    if set(strong) == {"H"} or (strong == ["H"]):
        arch = "lean hierarchical C→H→O (H is the only continuous handle so far)"
    elif len(strong) >= 2 and med.get("reading", "").startswith("behavioral"):
        arch = "lean hierarchical: multiple continuous handles but H→O coupling present"
    elif len(strong) >= 2:
        arch = "factorized still plausible — confirm selective binary J with more reps"
    else:
        arch = "directions look non-causal so far; do not freeze; revisit layer/site before 8-way"

    report = {
        "protocol": "Stage B.1–B.3 dose-response (steered residual q)",
        "measurement_note": (
            "q_steered from last steered residual during generate; "
            "scenario h_tool/h_report are pre-steer and must not be used as causal q"
        ),
        "S_eval": "binary",
        "q_eval": "sigmoid(h_steered·V)",
        "alphas": alphas,
        "reps": args.reps,
        "task": args.task,
        "curves": curves,
        "slopes": slopes,
        "binary_rates": binary_rates,
        "baseline_S": baseline_S,
        "mediation_H": mediation,
        "mediation_table": med,
        "architecture_hint": arch,
        "not": "8way|more_free_runs",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Channel dose-response (Stage B.1–B.3)",
        "",
        "> `q_steered` = preference on **steered** residual; `S` = binary boundary.",
        "> Flat q ⇒ not a control direction. Moving q / flat S ⇒ weak control / need dose.",
        "",
        f"- Baseline S (α=0 pool): `{baseline_S}`",
        f"- Architecture hint: **{arch}**",
        "",
        "## Target-channel continuous slopes (q_i vs α)",
        "",
        "| Channel | slope dq_i/dα | mean Δq_i | reading |",
        "|---------|---------------|-----------|---------|",
    ]
    for ch, s in slopes.items():
        mdq = s["mean_dq_i"]
        mdqs = f"{mdq:.4f}" if mdq is not None else "n/a"
        lines.append(
            f"| {ch} | {s['q_slope_vs_alpha']:.4f} | {mdqs} | {s['interpretation']} |"
        )
    lines += ["", "## Binary S_i vs α", ""]
    for ch, pts in binary_rates.items():
        lines.append(f"### `{ch}`")
        lines.append("")
        lines.append("| α | S_i | full S |")
        lines.append("|---|-----|--------|")
        for pt in pts:
            lines.append(f"| {pt['alpha']} | {pt['S_i']} | {pt['S']} |")
        lines.append("")
    lines += ["", "## Curves (S, q_steered, Δq)", ""]
    for name, series in curves.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| α | S | q_steered | Δq |")
        lines.append("|---|---|-----------|----|")
        for pt in series:
            lines.append(
                f"| {pt['alpha']} | {pt['S_mean']} | {pt['q_steered_mean']} | {pt['dq_mean']} |"
            )
        lines.append("")
    lines += [
        "## H mediation (B.2)",
        "",
        f"- n H-steer trials: {len(mediation)}",
        f"- cells: `{med['cells']}`",
        f"- P(ΔO|ΔH)={med['P_Ochg_given_Hchg']}  P(ΔO|¬ΔH)={med['P_Ochg_given_Hsame']}",
        f"- reading: **{med['reading']}**",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "md": str(MD),
                "slopes": slopes,
                "mediation_table": med,
                "architecture_hint": arch,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
