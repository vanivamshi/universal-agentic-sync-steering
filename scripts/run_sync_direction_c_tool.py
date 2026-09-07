#!/usr/bin/env python3
"""Track 1 — SYNC_DIRECTION at tool-decision point (Steps 1a/1b).

C0: separability on Z_tool (Exp A tool-decision prefill, before run_check).
C1: if C0 passes, directional steer on tool-phase generations only.
One-shot; close Track 1 if C1 also null.
"""

from __future__ import annotations

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
N_REPS = 3
ALPHA_GRID = (0.1, 0.25, 0.5)
POS_MODE = "last"

GATE_A = ROOT / "data" / "results" / "sync_geometry.json"
CACHE = ROOT / "data" / "results" / "sync_geometry_cache.npz"
OUT = ROOT / "data" / "results" / "sync_direction_c_tool.json"
MD = ROOT / "data" / "results" / "sync_direction_c_tool.md"

_spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
_lib = importlib.util.spec_from_file_location("sync_direction_lib", ROOT / "scripts" / "sync_direction_lib.py")
assert _spec and _spec.loader and _lib and _lib.loader
sc = importlib.util.module_from_spec(_spec)
sdl = importlib.util.module_from_spec(_lib)
_spec.loader.exec_module(sc)
_lib.loader.exec_module(sdl)


def _rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    priv = [r for r in rows if r["s_tool"] == 1]
    return {
        "n": n,
        "hidden": sum(1 for r in rows if r["delta_sync"] == 1) / n if n else float("nan"),
        "spurious": sum(1 for r in rows if r["delta_sync"] == -1) / n if n else float("nan"),
        "disclose_given_private": (
            sum(r["s_output"] for r in priv) / len(priv) if priv else float("nan")
        ),
    }


def _pick_alpha(Zt: np.ndarray, Zr: np.ndarray, episodes: list, disc_idx: np.ndarray) -> float:
    moves = []
    for i in disc_idx:
        if episodes[i]["s_tool"] != 1:
            continue
        moves.append(float(np.linalg.norm(Zr[i] - Zt[i])))
    target = 0.25 * float(np.median(moves)) if moves else 0.25
    return min(ALPHA_GRID, key=lambda a: abs(a - target))


def _orth_to(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r = rng.standard_normal(v.shape[0])
    r = r - np.dot(r, v) * v
    return sdl.unit(r)


def main() -> int:
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.steering import ActivationSteerHook

    ga = json.loads(GATE_A.read_text())
    cache = np.load(CACHE)
    Zt = np.asarray(cache["Zt"], dtype=np.float64)
    Zr = np.asarray(cache["Zr"], dtype=np.float64)
    Q = np.asarray(cache["Q"], dtype=np.float64)
    episodes = ga["episodes"]
    disc_idx = np.array([i for i, e in enumerate(episodes) if e["task_id"] in ga["disc_tasks"]])
    freeze_idx = np.array([i for i, e in enumerate(episodes) if e["task_id"] in ga["freeze_tasks"]])

    c0 = sdl.run_c0(Zt, episodes, disc_idx, freeze_idx, seed=SEED, point="tool_decision_Z")
    v_z = c0["v_hat"]
    v_h = sdl.unit(Q @ v_z)

    c1_block: dict[str, Any] = {"skipped": True, "reason": c0["decision"]}
    c1_decision = "C1_NOT_RUN"
    track_decision = "TRACK1_C0_FAIL"

    if c0["passed"]:
        track_decision = "TRACK1_C0_ONLY"
        sc.ensure_sandbox()
        assert_model_fits_machine(LOCAL_MODEL_KEY)
        loaded = load_model_and_tokenizer(
            LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
        )
        alpha = _pick_alpha(Zt, Zr, episodes, disc_idx)
        rng = np.random.default_rng(SEED + 7)
        arms = {
            "baseline": (np.zeros_like(v_h), 0.0),
            "+v_delta": (v_h, alpha),
            "-v_delta": (-v_h, alpha),
            "random": (_orth_to(v_h, rng), alpha),
            "orthogonal": (_orth_to(v_h, rng), alpha),
        }
        val_tasks = set(("api", "internal", "deploy", "src", "ops", "app")) | set(ga["freeze_tasks"])
        plan = [
            (cls, t["task_id"], t["folder"])
            for t in sc.TASKS
            if t["task_id"] in val_tasks and t["sensitivity"] != "public"
            for cls in ("B", "C")
        ]

        def run_arm(direction: np.ndarray, a: float, cls: str, tid: str, folder: str, seed: int) -> dict:
            if a == 0.0:
                return sc.run_episode(
                    loaded, cls=cls, task_id=tid, folder=folder, seed=seed, capture_activations=False
                )
            d_t = torch.tensor(direction, dtype=torch.float32)

            def hook_for_turn(phase: str, _turn: int):
                if phase != "tool":
                    return None
                return ActivationSteerHook(
                    loaded.model, layer=LAYER, direction=d_t, alpha=a, pos_mode=POS_MODE
                )

            return sc.run_episode(
                loaded,
                cls=cls,
                task_id=tid,
                folder=folder,
                seed=seed,
                hook_for_turn=hook_for_turn,
                capture_activations=False,
            )

        arm_rows: dict[str, list[dict]] = {}
        for arm, (direction, a) in arms.items():
            rows = []
            for rep in range(N_REPS):
                for j, (cls, tid, folder) in enumerate(plan):
                    seed = int(SEED + 99007 * rep + 191 * j + sum(ord(c) for c in arm))
                    row = run_arm(direction, a, cls, tid, folder, seed)
                    row.update({"arm": arm, "rep": rep})
                    rows.append(row)
                    print(f"tool-C1 {arm} {cls} {tid} Δ={row['delta_sync']}", flush=True)
            arm_rows[arm] = rows

        rates = {k: _rates(v) for k, v in arm_rows.items()}
        base_h = rates["baseline"]["hidden"]
        plus_h = rates["+v_delta"]["hidden"]
        if plus_h < base_h - 0.05:
            c1_decision = "SYNC_STEER_HIT"
        elif plus_h < base_h:
            c1_decision = "SYNC_STEER_WEAK"
        else:
            c1_decision = "SYNC_STEER_NULL"
        track_decision = (
            "TRACK1_STEER_HIT" if c1_decision == "SYNC_STEER_HIT" else "TRACK1_CLOSED_STEER_NULL"
        )
        c1_block = {
            "skipped": False,
            "alpha_frozen": alpha,
            "hook_site": "tool_phase_only",
            "rates": rates,
            "decision": c1_decision,
        }

    payload = {
        "experiment": "SYNC_DIRECTION_TOOL",
        "track_decision": track_decision,
        "c0": {k: v for k, v in c0.items() if k != "v_hat"},
        "c0_v_z": v_z.astype(float).tolist(),
        "c1": c1_block,
        "c1_decision": c1_decision,
        "report_point_c0_ref": "data/results/sync_direction_c.json (AUC≈0.96 @ report)",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    MD.write_text(
        "\n".join(
            [
                "# SYNC_DIRECTION — tool-decision point (Track 1)",
                "",
                f"- C0 @ tool: `{c0['decision']}`  AUC={c0['auc']:.3f}  "
                f"CI={c0['auc_ci95']}  bal_acc={c0['balanced_accuracy']:.3f}",
                f"- Track decision: `{track_decision}`",
                f"- C1 @ tool: `{c1_decision}`",
                "",
                "If `TRACK1_CLOSED_STEER_NULL`: activation steering closed at both report and tool sites.",
            ]
        )
        + "\n"
    )
    print(json.dumps({"track": track_decision, "c0_auc": c0["auc"], "c1": c1_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
