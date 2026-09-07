#!/usr/bin/env python3
"""Stage 2 — Single-channel causal interventions → empirical J.

Requires V from learn_channel_directions.py (prefer --from-free-runs).

Fixes vs earlier draft:
  - with_plan_format=True so C is measured from PLAN
  - paired seeds: baseline and steer share the same seed (isolates intervention)
  - refuse --freeze if v_C is still a class proxy
  - H hook at tool phase; C/O at report phase

  .venv/bin/python scripts/test_channel_controllability.py --mode causal --reps 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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
    CHANNEL_V_FROZEN,
    ChannelBank,
    CHANNELS,
    continuous_q_from_proj,
    jacobian_from_deltas,
    save_bank,
    summarize_jacobian,
)

OUT = ROOT / "data" / "results" / "sync_channel_controllability.json"
OUT_MAT = ROOT / "data" / "results" / "sync_channel_controllability_matrix.json"
MD = ROOT / "data" / "results" / "sync_channel_controllability.md"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
LAYER = 4
SEED = 20260902


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
    plan = extract_plan(row.get("final") or "")
    if plan:
        c = score_plan(plan, H)
        C = int(c) if c is not None else -1
    else:
        C = -1
    # Binary eval: unknown C → treat as 0 for Δ only with warning flag elsewhere
    return [max(C, 0), H, O]


def _make_hook(loaded, v: np.ndarray, alpha: float):
    from activation_pipeline.steering import ActivationSteerHook

    return ActivationSteerHook(
        loaded.model,
        layer=LAYER,
        direction=torch.tensor(v, dtype=torch.float32),
        alpha=float(alpha),
        pos_mode="last",
        collect_stats=True,
    )


def run_causal(*, reps: int, seed: int, alpha: float, task: str) -> dict[str, Any]:
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not CHANNEL_V.is_file():
        raise SystemExit("run learn_channel_directions.py first")
    bank = ChannelBank.load(CHANNEL_V)
    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )

    base_rows = []
    for r in range(reps):
        row = sc.run_episode(
            loaded,
            cls=NEUTRAL_CLS,
            task_id=task,
            seed=seed + r,
            with_plan_format=True,
        )
        base_rows.append({"S": _S_from_row(row), "plan": (row.get("final") or "")[:200]})
    S_base = [float(np.mean([b["S"][i] for b in base_rows])) for i in range(3)]

    arms: list[dict[str, Any]] = []
    mean_deltas: dict[str, list[float]] = {}

    for j, ch in enumerate(CHANNELS):
        v = bank.V[j]
        for sign, tag in ((+1.0, f"+v_{ch}"), (-1.0, f"-v_{ch}")):
            steered = []
            for r in range(reps):
                hook = _make_hook(loaded, sign * v, alpha)

                def hook_for_turn(phase: str, _turn: int, _hook=hook, _ch=ch):
                    if _ch == "H" and phase == "tool":
                        return _hook
                    if _ch in ("C", "O") and phase == "report":
                        return _hook
                    return None

                # Paired seed with baseline rep r
                row = sc.run_episode(
                    loaded,
                    cls=NEUTRAL_CLS,
                    task_id=task,
                    seed=seed + r,
                    hook_for_turn=hook_for_turn,
                    with_plan_format=True,
                )
                hook.remove()
                steered.append(_S_from_row(row))

            S_mean = [float(np.mean([s[i] for s in steered])) for i in range(3)]
            delta = [S_mean[i] - S_base[i] for i in range(3)]
            idx = {"C": 0, "H": 1, "O": 2}[ch]
            arms.append(
                {
                    "arm": tag,
                    "channel": ch,
                    "sign": sign,
                    "S_base": S_base,
                    "S_mean": S_mean,
                    "delta_S": delta,
                    "target_delta": delta[idx],
                    "collateral": [delta[i] for i in range(3) if i != idx],
                }
            )
            if sign > 0:
                mean_deltas[f"v_{ch}"] = delta

    J = jacobian_from_deltas(mean_deltas)
    # Finite-difference style: column j ≈ 0.5 * (ΔS(+v_j) - ΔS(-v_j))
    J_signed = np.zeros((3, 3), dtype=np.float64)
    for j, ch in enumerate(CHANNELS):
        plus = next(a for a in arms if a["arm"] == f"+v_{ch}")
        minus = next(a for a in arms if a["arm"] == f"-v_{ch}")
        J_signed[:, j] = 0.5 * (np.asarray(plus["delta_S"]) - np.asarray(minus["delta_S"]))
    summary = summarize_jacobian(J_signed)
    summary["J_plus_only"] = J.tolist()
    summary.update(
        {
            "mode": "causal",
            "S_base": S_base,
            "alpha": alpha,
            "reps": reps,
            "neutral_task": task,
            "mean_deltas": mean_deltas,
            "arms": arms,
            "bank_meta_status": bank.meta.get("status"),
            "v_C_status": bank.meta.get("v_C_status"),
            "note": (
                "J from paired-seed ±α single-channel steers. "
                "Ideal ≈ diagonal. Do not freeze if proxy v_C or weak diag."
            ),
        }
    )
    return summary


def _write_md(report: dict) -> None:
    J = np.asarray(report["J"], dtype=np.float64)
    lines = [
        "# Channel controllability (Stage 2)",
        "",
        f"- Mode: `{report['mode']}`",
        f"- S_base: `{report.get('S_base')}`",
        f"- v_C_status: `{report.get('v_C_status')}`",
        f"- approx_diagonal: **{report.get('approx_diagonal')}**",
        "",
        "| | v_C | v_H | v_O |",
        "|---|-----|-----|-----|",
        f"| ΔC | {J[0,0]:.3f} | {J[0,1]:.3f} | {J[0,2]:.3f} |",
        f"| ΔH | {J[1,0]:.3f} | {J[1,1]:.3f} | {J[1,2]:.3f} |",
        f"| ΔO | {J[2,0]:.3f} | {J[2,1]:.3f} | {J[2,2]:.3f} |",
        "",
        "## Arms",
        "",
    ]
    for a in report.get("arms") or []:
        lines.append(
            f"- `{a['arm']}`: ΔS={a['delta_S']} target={a['target_delta']:.3f} collateral={a['collateral']}"
        )
    MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("causal", "correlational"), default="correlational")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--task", default=NEUTRAL_TASK)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    if args.mode == "correlational":
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "learn_channel_directions.py"), "--allow-geometry-proxy"],
            cwd=str(ROOT),
        )
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "build_controllability_matrix.py"), "--mode", "correlational"],
            cwd=str(ROOT),
        )
        report = json.loads(OUT_MAT.read_text())
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"mode": "correlational_not_causal", "out": str(OUT_MAT)}, indent=2))
        return 0

    report = run_causal(reps=args.reps, seed=args.seed, alpha=args.alpha, task=args.task)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    OUT_MAT.write_text(
        json.dumps(
            {k: report[k] for k in ("J", "diag", "offdiag_l2", "diag_abs_mean", "approx_diagonal", "channels", "mode", "note", "v_C_status")},
            indent=2,
        )
        + "\n"
    )
    _write_md(report)

    if args.freeze:
        proxy = "proxy" in str(report.get("v_C_status") or "").lower() or "PROXY" in str(
            report.get("bank_meta_status") or ""
        )
        if proxy:
            print("REFUSING --freeze: v_C is still a proxy / not from observed PLAN", file=sys.stderr)
            return 2
        if not report.get("approx_diagonal"):
            print("REFUSING --freeze: approx_diagonal is false", file=sys.stderr)
            return 2
        bank = ChannelBank.load(CHANNEL_V)
        bank.frozen = True
        bank.meta = {**bank.meta, "stage": 2, "status": "frozen_after_causal_J", "J": report["J"]}
        save_bank(bank, CHANNEL_V_FROZEN)
        print(json.dumps({"frozen": str(CHANNEL_V_FROZEN)}, indent=2))

    print(json.dumps({"out": str(OUT), "approx_diagonal": report["approx_diagonal"], "S_base": report["S_base"], "v_C_status": report.get("v_C_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
