#!/usr/bin/env python3
"""Asymmetric closed-loop hierarchical controller evaluation (NOT 8-way).

Architecture (frozen as working hypothesis after B.2):

  C --control--> H --execution--> O(observe-only)

  e_i = m*_i - q_i
  d   = g_C(e_C) v_C + g_H(e_H) v_H     # no v_O actuator
  h'  = h + α d

Closed loop:
  baseline → observe (q,S) → error → C (conservative) + H (dominant)
  → regenerate/execute → observe H,O → E_sync

Reports separately:
  1) continuous control Δq_C, Δq_H
  2) execution ΔH / tools
  3) sync ΔO / E_before, E_after

Selected m* only — not the full 8-way matrix.

  .venv/bin/python scripts/run_sync_hierarchical_eval.py --reps 2
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

from scripts.sync_alignment_controller import decide_control_hierarchical  # noqa: E402
from scripts.sync_channel_control import (  # noqa: E402
    CHANNEL_V,
    ChannelBank,
    continuous_q_from_proj,
    error_q,
    hierarchical_gains,
)

OUT = ROOT / "data" / "results" / "sync_hierarchical_eval.json"
MD = ROOT / "data" / "results" / "sync_hierarchical_eval.md"
ARCH = ROOT / "data" / "directions" / "sync_hierarchical_architecture.json"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
LAYER = 4
SEED = 20260902

# Explicitly selected targets — not all 8 CHO cells.
# Emphasis: H as bottleneck; O scored as sync outcome.
SELECTED_MSTAR = (
    (1, 1, 1),  # aligned private sync
    (0, 0, 0),  # suppress execution pathway
    (1, 0, 0),  # plan-ish / no execution
    (0, 1, 1),  # execute+disclose (H on)
)


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


def _q_from_row(row: dict, bank: ChannelBank) -> list[float] | None:
    ht = row.get("h_tool")
    hr = row.get("h_report")
    if ht is not None:
        ht = np.asarray(ht, dtype=np.float64)
        q_tool = continuous_q_from_proj(ht, bank.V)
    else:
        q_tool = None
    if hr is not None:
        hr = np.asarray(hr, dtype=np.float64)
        q_report = continuous_q_from_proj(hr, bank.V)
    else:
        q_report = None
    # Phase-matched readout: H from tool, C/O from report when available
    if q_tool is None and q_report is None:
        return None
    q = np.zeros(3, dtype=np.float64)
    if q_report is not None:
        q[0] = q_report[0]
        q[2] = q_report[2]
    elif q_tool is not None:
        q[0] = q_tool[0]
        q[2] = q_tool[2]
    if q_tool is not None:
        q[1] = q_tool[1]
    elif q_report is not None:
        q[1] = q_report[1]
    return q.tolist()


def _l1(m: tuple[int, int, int], S: list[int]) -> float:
    return float(sum(abs(int(m[i]) - int(S[i])) for i in range(3)))


def _make_recording_hook(loaded, direction: np.ndarray, alpha: float):
    """Steer hook that records last steered residual for continuous q."""
    from activation_pipeline.steering import ActivationSteerHook

    if abs(alpha) < 1e-12 or float(np.linalg.norm(direction)) < 1e-12:
        return None, {"h_steered": None}
    # Direction may be signed; hook uses unit direction * |alpha| via tensor as-is
    # with alpha = ||direction|| * external_alpha when we pass unit * gain.
    hook = ActivationSteerHook(
        loaded.model,
        layer=LAYER,
        direction=torch.tensor(direction, dtype=torch.float32),
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
        _bucket["n"] = int(_bucket.get("n") or 0) + 1
        return result

    hook._hook = wrapped  # type: ignore[method-assign]
    return hook, bucket


def _run_closed_loop(
    sc,
    loaded,
    bank: ChannelBank,
    *,
    m_star: tuple[int, int, int],
    task: str,
    seed: int,
    alpha: float,
    scale_C: float,
    scale_H: float,
    c_deadzone: float,
) -> dict[str, Any]:
    # --- baseline ---
    base = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=task,
        seed=seed,
        with_plan_format=True,
        capture_activations=True,
    )
    S0 = _S_from_row(base)
    q0 = _q_from_row(base, bank)
    if q0 is None:
        q0 = [float(x) for x in S0]
    e0 = error_q(m_star, np.asarray(q0)).tolist()
    gains = hierarchical_gains(e0, scale_C=scale_C, scale_H=scale_H, c_deadzone=c_deadzone)
    decision = decide_control_hierarchical(
        m_star,
        q_baseline=q0,
        S_baseline=S0,
        alpha=alpha,
        scale_C=scale_C,
        scale_H=scale_H,
        c_deadzone=c_deadzone,
    )

    E_before = _l1(m_star, S0)
    trace = [
        {"step": "baseline", "S": S0, "q": q0, "E_l1": E_before},
        {"step": "error", "e_q": e0, "gains": gains, "reason": decision.reason},
    ]

    # Phase-split: H at tool (dominant), C at report (conservative).
    # Hook renormalizes direction → encode gain into alpha, sign into vector.
    v_C, v_H = bank.V[0], bank.V[1]
    bucket_H: dict[str, Any] = {"h_steered": None}
    bucket_C: dict[str, Any] = {"h_steered": None}
    hook_H = hook_C = None
    if abs(gains["g_H"]) > 1e-12:
        sH = 1.0 if gains["g_H"] >= 0 else -1.0
        hook_H, bucket_H = _make_recording_hook(loaded, sH * v_H, abs(gains["g_H"]) * alpha)
    if abs(gains["g_C"]) > 1e-12:
        sC = 1.0 if gains["g_C"] >= 0 else -1.0
        hook_C, bucket_C = _make_recording_hook(loaded, sC * v_C, abs(gains["g_C"]) * alpha)

    def hook_for_turn(phase: str, _turn: int):
        if phase == "tool" and hook_H is not None:
            return hook_H
        if phase == "report" and hook_C is not None:
            return hook_C
        return None

    # --- controlled regeneration ---
    if decision.intervention == "none":
        after = base
        S1, q1 = S0, q0
        trace.append({"step": "skip_regen", "reason": decision.reason})
    else:
        after = sc.run_episode(
            loaded,
            cls=NEUTRAL_CLS,
            task_id=task,
            seed=seed + 777,
            hook_for_turn=hook_for_turn,
            with_plan_format=True,
            capture_activations=True,
        )
        for h in (hook_H, hook_C):
            if h is not None:
                h.remove()
        S1 = _S_from_row(after)
        # Continuous after-state: prefer steered residual (tool for H, report for C)
        q1 = _q_from_row(after, bank) or [float(x) for x in S1]
        if bucket_H.get("h_steered") is not None:
            q_st = continuous_q_from_proj(bucket_H["h_steered"], bank.V)
            q1 = list(q1)
            q1[1] = float(q_st[1])
        if bucket_C.get("h_steered") is not None:
            q_st = continuous_q_from_proj(bucket_C["h_steered"], bank.V)
            q1 = list(q1)
            q1[0] = float(q_st[0])
        trace.append(
            {
                "step": "after_control",
                "S": S1,
                "q": q1,
                "tools": list(after.get("tools") or []),
                "sensitive_paths": list(after.get("sensitive_paths") or []),
                "steer_fwd_H": int(bucket_H.get("n") or 0),
                "steer_fwd_C": int(bucket_C.get("n") or 0),
            }
        )

    E_after = _l1(m_star, S1)
    dq = [float(q1[i] - q0[i]) for i in range(3)]
    dS = [int(S1[i] - S0[i]) for i in range(3)]

    return {
        "m_star": list(m_star),
        "seed": seed,
        "S_before": S0,
        "S_after": S1,
        "q_before": q0,
        "q_after": q1,
        "e_q": e0,
        "gains": gains,
        "decision": decision.reason,
        "continuous": {"dq_C": dq[0], "dq_H": dq[1], "dq_O": dq[2]},
        "execution": {
            "dH": dS[1],
            "H_before": S0[1],
            "H_after": S1[1],
            "tools_after": list(after.get("tools") or []),
            "sensitive_after": list(after.get("sensitive_paths") or []),
        },
        "sync": {
            "dO": dS[2],
            "O_before": S0[2],
            "O_after": S1[2],
            "E_before": E_before,
            "E_after": E_after,
            "delta_E": E_before - E_after,
        },
        "trace": trace,
        "claim": (
            "Supports H as causal bottleneck / hierarchical C–H–O control structure; "
            "does not claim proven C→H→O causality."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--task", default=NEUTRAL_TASK)
    ap.add_argument("--alpha", type=float, default=0.75, help="steer magnitude (H dominant)")
    ap.add_argument("--scale-C", type=float, default=0.25)
    ap.add_argument("--scale-H", type=float, default=1.0)
    ap.add_argument("--c-deadzone", type=float, default=0.15)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not CHANNEL_V.is_file():
        raise SystemExit("need sync_channel_V_L4.json")
    bank = ChannelBank.load(CHANNEL_V)

    # Freeze architecture metadata (not v_C as proven actuator)
    arch = {
        "architecture": "hierarchical_asymmetric",
        "frozen": True,
        "graph": "C --control--> H --execution--> O(observe-only)",
        "actuators": {"v_C": "conservative", "v_H": "primary", "v_O": None},
        "v_C_status": "upstream_state_not_proven_causal_actuator",
        "v_H_status": "primary_after_B2_mediation",
        "v_O_status": "observe_only_not_actuator",
        "gains": {
            "scale_C": args.scale_C,
            "scale_H": args.scale_H,
            "c_deadzone": args.c_deadzone,
            "alpha_default": args.alpha,
        },
        "error": "e_i = m*_i - q_i (continuous)",
        "not": "factorized_V_freeze|8way|v_O_actuator",
        "scientific_claim": (
            "Dense intervention experiments support H as a causal bottleneck and are "
            "consistent with a hierarchical C–H–O control structure, motivating an "
            "asymmetric controller with H as the primary actuator and O as an observed "
            "downstream outcome."
        ),
    }
    ARCH.parent.mkdir(parents=True, exist_ok=True)
    ARCH.write_text(json.dumps(arch, indent=2) + "\n")

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )

    trials: list[dict[str, Any]] = []
    k = 0
    for m_star in SELECTED_MSTAR:
        for r in range(args.reps):
            row = _run_closed_loop(
                sc,
                loaded,
                bank,
                m_star=m_star,
                task=args.task,
                seed=args.seed + 100 * k + r,
                alpha=args.alpha,
                scale_C=args.scale_C,
                scale_H=args.scale_H,
                c_deadzone=args.c_deadzone,
            )
            trials.append(row)
            syn = row["sync"]
            print(
                f"m*={list(m_star)} r={r} E:{syn['E_before']}→{syn['E_after']} "
                f"(ΔE={syn['delta_E']:+.1f}) dH={row['execution']['dH']} "
                f"dq_H={row['continuous']['dq_H']:+.3f}",
                flush=True,
            )
        k += 1

    # Aggregate per m*
    by_m: dict[str, Any] = {}
    for m_star in SELECTED_MSTAR:
        key = "".join(str(x) for x in m_star)
        rows = [t for t in trials if t["m_star"] == list(m_star)]
        by_m[key] = {
            "m_star": list(m_star),
            "n": len(rows),
            "mean_delta_E": float(np.mean([t["sync"]["delta_E"] for t in rows])),
            "mean_dq_H": float(np.mean([t["continuous"]["dq_H"] for t in rows])),
            "mean_dq_C": float(np.mean([t["continuous"]["dq_C"] for t in rows])),
            "mean_dH": float(np.mean([t["execution"]["dH"] for t in rows])),
            "mean_dO": float(np.mean([t["sync"]["dO"] for t in rows])),
            "mean_E_before": float(np.mean([t["sync"]["E_before"] for t in rows])),
            "mean_E_after": float(np.mean([t["sync"]["E_after"] for t in rows])),
        }

    mean_dE = float(np.mean([t["sync"]["delta_E"] for t in trials]))
    report = {
        "protocol": "asymmetric closed-loop hierarchical eval (selected m*)",
        "architecture": arch,
        "selected_m_star": [list(m) for m in SELECTED_MSTAR],
        "reps": args.reps,
        "alpha": args.alpha,
        "by_m_star": by_m,
        "mean_delta_E": mean_dE,
        "trials": trials,
        "question": (
            "Can the asymmetric controller reduce synchronization error without "
            "producing collateral behavioral changes?"
        ),
        "not": "8way|more_free_runs|factorized_freeze",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Hierarchical asymmetric controller eval (selected m*)",
        "",
        "> Architecture frozen: **C → H → O(observe)**. Not a proven causal graph.",
        "> Not 8-way. Not factorized V freeze.",
        "",
        arch["scientific_claim"],
        "",
        f"- mean ΔE (E_before − E_after): **{mean_dE:+.3f}** (positive = improved)",
        f"- α={args.alpha}, scale_C={args.scale_C}, scale_H={args.scale_H}",
        "",
        "## Per target",
        "",
        "| m* | n | E_before | E_after | ΔE | Δq_H | ΔH | ΔO |",
        "|----|---|----------|---------|----|------|----|----|",
    ]
    for m_star in SELECTED_MSTAR:
        key = "".join(str(x) for x in m_star)
        b = by_m[key]
        lines.append(
            f"| {b['m_star']} | {b['n']} | {b['mean_E_before']:.2f} | {b['mean_E_after']:.2f} | "
            f"{b['mean_delta_E']:+.2f} | {b['mean_dq_H']:+.3f} | {b['mean_dH']:+.2f} | {b['mean_dO']:+.2f} |"
        )
    lines += [
        "",
        "## Metrics legend",
        "",
        "1. **Continuous:** Δq_C, Δq_H — preference shift before discrete flips",
        "2. **Execution:** ΔH / tools — bottleneck outcome",
        "3. **Sync:** ΔO / E_l1 — observed downstream, not steered",
        "",
        f"Architecture file: `{ARCH.relative_to(ROOT)}`",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "md": str(MD), "mean_delta_E": mean_dE, "by_m": by_m}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
