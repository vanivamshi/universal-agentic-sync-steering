#!/usr/bin/env python3
"""DEPRIORITIZED until Stage-2 identification + causal J pass.

Stage 3 — Closed-loop 8-target evaluation with factorized controller.

Do NOT use as the main experiment while H has insufficient free-run
variation or probes show weak AUC. See docs/sync_channel_control.md.

Requires frozen channel directions from Stage 2:

  data/directions/sync_channel_V_L4.frozen.json

Controller:
  e = m* - S
  d = e_C v_C + e_H v_H + e_O v_O
  h' = h + α d
  re-observe S; optionally iterate (closed loop)

Do not run until Stage 2 causal J is reviewed and frozen.
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
    CHANNEL_V_FROZEN,
    ChannelBank,
    compose_direction,
    error_e,
)
from scripts.sync_eq import all_m_star_masks, sync_error_norm  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_controller_8way.json"
MD = ROOT / "data" / "results" / "sync_controller_8way.md"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
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
        C = int(c) if c is not None else (1 if row.get("class_intended") in ("B", "C") else 0)
    else:
        C = 1 if row.get("class_intended") in ("B", "C") else 0
    return [C, H, O]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--loops", type=int, default=2, help="closed-loop intervention rounds")
    ap.add_argument("--task", default=NEUTRAL_TASK)
    ap.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="EXPLORATORY: use candidate V without Stage-2 freeze",
    )
    args = ap.parse_args()

    if (not CHANNEL_V_FROZEN.is_file()) and (not args.allow_unfrozen):
        print(
            "REFUSED: freeze Stage-2 directions first.\n"
            "  1) .venv/bin/python scripts/learn_channel_directions.py\n"
            "  2) .venv/bin/python scripts/test_channel_controllability.py --mode causal --reps 1\n"
            "  3) review J; then add --freeze\n"
            "  4) re-run this script\n",
            file=sys.stderr,
        )
        return 2

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.steering import ActivationSteerHook

    from scripts.sync_channel_control import CHANNEL_V
    bank_path = CHANNEL_V_FROZEN if CHANNEL_V_FROZEN.is_file() else CHANNEL_V
    bank = ChannelBank.load(bank_path)
    if args.allow_unfrozen and not CHANNEL_V_FROZEN.is_file():
        print("WARNING: exploratory 8-way with UNFROZEN candidates", flush=True)
    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )

    rows: list[dict[str, Any]] = []
    for m_star in all_m_star_masks():
        # baseline free
        row0 = sc.run_episode(loaded, cls=NEUTRAL_CLS, task_id=args.task, seed=args.seed, with_plan_format=True)
        S = _S_from_row(row0)
        e = error_e(m_star, S)
        E_before = sync_error_norm(e)
        history = [{"S": S, "e": e, "E": E_before}]

        for loop in range(args.loops):
            if e == [0, 0, 0]:
                break
            d = compose_direction(e, bank.V)
            if float(np.linalg.norm(d)) < 1e-12:
                break
            hook = ActivationSteerHook(
                loaded.model,
                layer=bank.layer,
                direction=torch.tensor(d, dtype=torch.float32),
                alpha=bank.alpha,
                pos_mode="last",
                collect_stats=True,
            )

            def hook_for_turn(phase: str, _turn: int, _e=list(e), _hook=hook):
                # Apply H component during tools; C/O during report — if e_H!=0 use tool hook
                if _e[1] != 0 and phase == "tool":
                    return _hook
                if (_e[0] != 0 or _e[2] != 0) and phase == "report":
                    return _hook
                if _e[1] == 0 and phase == "report":
                    return _hook
                return None

            row = sc.run_episode(
                loaded,
                cls=NEUTRAL_CLS,
                task_id=args.task,
                seed=args.seed + 10 + loop,
                hook_for_turn=hook_for_turn,
                with_plan_format=True,
            )
            hook.remove()
            S = _S_from_row(row)
            e = error_e(m_star, S)
            history.append({"S": S, "e": e, "E": sync_error_norm(e), "loop": loop})

        E_after = history[-1]["E"]
        rows.append(
            {
                "m_star": m_star,
                "S_before": history[0]["S"],
                "S_after": history[-1]["S"],
                "E_before": E_before,
                "E_after": E_after,
                "delta_E": E_before - E_after,
                "reached": history[-1]["S"] == m_star,
                "history": history,
            }
        )

    n_reach = sum(1 for r in rows if r["reached"])
    report = {
        "n_cells": len(rows),
        "n_reached": n_reach,
        "median_delta_E": float(np.median([r["delta_E"] for r in rows])),
        "rows": rows,
        "controller": "d(e)=e·V factorized",
        "frozen_bank": str(CHANNEL_V_FROZEN),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Factorized 8-way controller",
        "",
        f"- Reached S=m*: **{n_reach}/{len(rows)}**",
        f"- Median ΔE: {report['median_delta_E']:.2f}",
        "",
        "| m* | S_before | S_after | E_before | E_after | ΔE | reached |",
        "|----|----------|---------|----------|---------|----|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {tuple(r['m_star'])} | {r['S_before']} | {r['S_after']} | "
            f"{r['E_before']} | {r['E_after']} | {r['delta_E']} | {r['reached']} |"
        )
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "n_reached": n_reach, "median_delta_E": report["median_delta_E"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
