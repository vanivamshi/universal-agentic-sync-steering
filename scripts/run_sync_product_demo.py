#!/usr/bin/env python3
"""Product demo: model + equation + prompt → readable output.

For team verification (not batch JSON). Shows tool hook, probe, intervention,
activation hook, and final answer.

Examples:
  .venv/bin/python scripts/run_sync_product_demo.py --mode compare
  .venv/bin/python scripts/run_sync_product_demo.py --mode gate --task api
  .venv/bin/python scripts/run_sync_product_demo.py --mode steer --task api
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

LAYER = 4
SEED = 20260822
ALPHA = 0.25
POS_MODE = "last"

GATE_A = ROOT / "data" / "results" / "sync_geometry.json"
VDELTA = ROOT / "data" / "directions" / "sync_v_delta_L4.jsonl"
VREPAIR = ROOT / "data" / "directions" / "sync_v_repair_eq_L4.jsonl"

_spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
_lib = importlib.util.spec_from_file_location("sync_direction_lib", ROOT / "scripts" / "sync_direction_lib.py")
assert _spec and _spec.loader and _lib and _lib.loader
sc = importlib.util.module_from_spec(_spec)
sdl = importlib.util.module_from_spec(_lib)
_spec.loader.exec_module(sc)
_lib.loader.exec_module(sdl)


def _load_v_repair() -> tuple[np.ndarray, float]:
    row = json.loads(VREPAIR.read_text().splitlines()[0])
    alpha = float(row.get("meta", {}).get("alpha_frozen", ALPHA))
    return np.asarray(row["vector"], dtype=np.float64), alpha


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _run_mode(
    loaded: Any,
    *,
    mode: str,
    task_id: str,
    folder: str,
    seed: int,
    v_detect: np.ndarray,
    tau: float,
    v_repair: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    from activation_pipeline.steering import ActivationSteerHook

    cls = "B" if mode in ("normal", "fault_act") else "C"
    task = sc.task_by_id(task_id)
    user_prompt = sc.task_user_message(cls, task)

    state = sc.run_tools_phase(
        loaded, cls=cls, task_id=task_id, folder=folder, seed=seed, Q=None
    )
    h_report = np.asarray(state["h_report"], dtype=np.float64)
    probe_before = float(h_report @ v_detect)
    s_tool = bool(state["sensitive_paths"])
    sync_detected = bool(s_tool and probe_before < tau)

    direction: np.ndarray | None = None
    gate_cfg: dict[str, Any] | None = None
    hook_registered = False
    hook_fired = False
    hook_n_fwd = 0

    if mode == "gate" and sync_detected:
        gate_cfg = {
            "mode": "report",
            "v_report": v_detect,
            "threshold": tau,
            "prompt_style": "generic",
            "force_prompt": sc.GATE_PROMPT_GENERIC,
        }
    elif mode == "steer" and s_tool:
        direction = v_repair
        hook_registered = True
    elif mode == "steer_vdelta" and s_tool:
        direction = v_detect
        hook_registered = True
    elif mode == "fault_act" and s_tool:
        direction = -v_repair
        hook_registered = True

    hook: ActivationSteerHook | None = None
    if hook_registered and direction is not None:
        hook = ActivationSteerHook(
            loaded.model,
            layer=LAYER,
            direction=torch.tensor(direction, dtype=torch.float32),
            alpha=alpha,
            pos_mode=POS_MODE,
            collect_stats=True,
        )

        def hook_for_turn(phase: str, _turn: int):
            if phase != "report":
                return None
            return hook

        row = sc.finalize_episode(loaded, state, gate=None, hook_for_turn=hook_for_turn)
        hook_fired = int(hook.stats["n_fwd"]) > 0
        hook_n_fwd = int(hook.stats["n_fwd"])
    else:
        row = sc.finalize_episode(loaded, state, gate=gate_cfg, hook_for_turn=None)
        hook_fired = False

    disclosed = bool(row["s_output"])
    hidden = row["delta_sync"] == 1

    return {
        "mode": mode,
        "cls": cls,
        "task_id": task_id,
        "user_prompt": user_prompt,
        "tools": state.get("tools") or [],
        "sensitive_paths": state.get("sensitive_paths") or [],
        "tool_hook_fired": "run_command" in (state.get("tools") or []),
        "private_access": s_tool,
        "probe_before": probe_before,
        "tau": tau,
        "sync_detected": sync_detected,
        "gate_fired": bool(row.get("gate_fired")),
        "activation_hook_registered": hook_registered,
        "activation_hook_fired": hook_fired,
        "activation_hook_n_fwd": hook_n_fwd,
        "repair_alpha": alpha if hook_registered else 0.0,
        "final_answer": row.get("final") or "",
        "disclosed": disclosed,
        "hidden": hidden,
        "delta_sync": row["delta_sync"],
    }


def _print_result(r: dict[str, Any]) -> None:
    _banner(f"MODE: {r['mode'].upper()}  |  task={r['task_id']}  |  class={r['cls']}")

    _section("PROMPT (what the model receives)")
    print(r["user_prompt"][:800])
    if len(r["user_prompt"]) > 800:
        print("...")

    _section("TOOL PHASE")
    print(f"  tool_hook_fired:     {r['tool_hook_fired']}")
    print(f"  tools used:          {r['tools']}")
    print(f"  private_access:      {r['private_access']}")
    print(f"  sensitive_paths:     {r['sensitive_paths']}")

    _section("DETECT (equation probe v_Δ)")
    print(f"  probe_before:        {r['probe_before']:.4f}")
    print(f"  threshold τ:         {r['tau']:.4f}")
    print(f"  sync_failure:        {r['sync_detected']}")

    _section("REPAIR")
    if r["mode"] == "gate":
        print("  method:              policy gate (disclosure regen)")
        print(f"  gate_fired:          {r['gate_fired']}")
    elif r["mode"] in ("steer", "steer_vdelta", "fault_act"):
        print("  method:              activation hook @ report (L4)")
        if r["mode"] == "steer":
            print(f"  equation direction:  v_R (repair)")
        elif r["mode"] == "fault_act":
            print(f"  equation direction:  −v_R (hide inject, class B prompt)")
        else:
            print(f"  equation direction:  v_Δ (detector)")
        print(f"  α:                   {r['repair_alpha']}")
        print(f"  hook_registered:     {r['activation_hook_registered']}")
        print(f"  hook_fired:          {r['activation_hook_fired']}  (n_fwd={r['activation_hook_n_fwd']})")
    else:
        print("  method:              none (fault baseline)")

    _section("FINAL ANSWER")
    print(r["final_answer"])

    _section("VERDICT")
    status = "PASS ✓ disclosed private access" if r["disclosed"] else "FAIL ✗ hidden private access"
    print(f"  {status}")
    print(f"  delta_sync:          {r['delta_sync']}  (+1=hidden, 0=sync ok, -1=spurious)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("compare", "fault", "fault_act", "gate", "steer", "steer_vdelta", "normal"),
        default="compare",
        help="compare=fault/steer/gate; fault_act=class B + hide activation (no omit prompt)",
    )
    ap.add_argument("--task", default="api", help="task_id (default: api)")
    ap.add_argument("--seed", type=int, default=SEED, help="RNG seed")
    args = ap.parse_args()

    ga = json.loads(GATE_A.read_text())
    v_detect = sdl.load_v_report(VDELTA)
    cache = np.load(ROOT / "data" / "results" / "sync_geometry_cache.npz")
    Hr = np.asarray(cache["Hr"], dtype=np.float64)
    tau = sdl.frozen_tau_median_disc(Hr, ga["episodes"], set(ga["disc_tasks"]), v_detect)
    v_repair, alpha = _load_v_repair()

    task = sc.task_by_id(args.task)
    folder = task["folder"]

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )

    _banner("SYNC PRODUCT DEMO")
    print(f"Model: qwen3-0.6B  |  task={args.task}  |  seed={args.seed}")
    print(f"Equation: v_R = unit(P_S(μ_B − μ_C)), α={alpha}")
    print(f"Detector: v_Δ, τ={tau:.4f}")

    if args.mode == "compare":
        modes = [
            ("fault", args.seed + 1),
            ("steer", args.seed + 1),
            ("gate", args.seed + 1),
        ]
        results = []
        for mode, seed in modes:
            r = _run_mode(
                loaded,
                mode=mode,
                task_id=args.task,
                folder=folder,
                seed=seed,
                v_detect=v_detect,
                tau=tau,
                v_repair=v_repair,
                alpha=alpha,
            )
            results.append(r)
            _print_result(r)

        _banner("SUMMARY (product test)")
        print(f"{'Mode':<12} {'Tool hook':<10} {'Detect':<8} {'Hook':<8} {'Disclosed':<10}")
        print("-" * 52)
        for r in results:
            print(
                f"{r['mode']:<12} "
                f"{str(r['tool_hook_fired']):<10} "
                f"{str(r['sync_detected']):<8} "
                f"{str(r['activation_hook_fired']):<8} "
                f"{str(r['disclosed']):<10}"
            )
        print("\nExpected product behavior:")
        print("  fault  → FAIL (hidden)     — shows the bug")
        print("  steer  → hook fires       — equation applied (may still FAIL)")
        print("  gate   → PASS (disclosed) — working product fix")
    else:
        mode = "fault" if args.mode == "fault" else args.mode
        r = _run_mode(
            loaded,
            mode=mode,
            task_id=args.task,
            folder=folder,
            seed=args.seed + (1 if mode != "normal" else 0),
            v_detect=v_detect,
            tau=tau,
            v_repair=v_repair,
            alpha=alpha,
        )
        _print_result(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
