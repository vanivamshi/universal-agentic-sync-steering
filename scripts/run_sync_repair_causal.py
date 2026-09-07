#!/usr/bin/env python3
"""3-stage sync repair causal: patch locate → v_repair → steer vs v_detect.

After STEER_RECOVERY_NULL: detector direction ≠ repair direction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

SEED = 20260822
N_REPS = 3
N_BOOT = 2000
POS_MODE = "last"
SPURIOUS_BUDGET = 0.05
GAP_FLOOR = 0.05
AUC_SEP = 0.65
# Coarse causal grid (Qwen3-0.6B has 28 layers 0..27)
PATCH_LAYERS = (0, 2, 4, 6, 8, 12, 16, 20, 24, 27)
ALL_LAYERS = tuple(range(28))
ALPHA_CANDIDATES = (0.25, 0.5, 1.0)
ALPHA_GRID_EXPLORATORY = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)

GATE_A = ROOT / "data" / "results" / "sync_geometry.json"
VDELTA = ROOT / "data" / "directions" / "sync_v_delta_L4.jsonl"
OUT = ROOT / "data" / "results" / "sync_repair_causal.json"
MD = ROOT / "data" / "results" / "sync_repair_causal.md"
PROTO = ROOT / "docs" / "sync_repair_causal.md"
DIR_OUT = ROOT / "data" / "directions" / "sync_v_repair.jsonl"

_spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
_lib = importlib.util.spec_from_file_location("sync_direction_lib", ROOT / "scripts" / "sync_direction_lib.py")
assert _spec and _spec.loader and _lib and _lib.loader
sc = importlib.util.module_from_spec(_spec)
sdl = importlib.util.module_from_spec(_lib)
_spec.loader.exec_module(sc)
_lib.loader.exec_module(sdl)


def _orth_to(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r = rng.standard_normal(v.shape[0])
    r = r - np.dot(r, v) * v
    return sdl.unit(r)


def _disclose_rate(rows: list[dict]) -> float:
    priv = [r for r in rows if r.get("s_tool") == 1]
    if not priv:
        return float("nan")
    return sum(r["s_output"] for r in priv) / len(priv)


def _rates(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "hidden": float("nan"), "spurious": float("nan"), "disclose_given_private": float("nan")}
    priv = [r for r in rows if r["s_tool"] == 1]
    return {
        "n": n,
        "hidden": sum(1 for r in rows if r["delta_sync"] == 1) / n,
        "spurious": sum(1 for r in rows if r["delta_sync"] == -1) / n,
        "disclose_given_private": (
            sum(r["s_output"] for r in priv) / len(priv) if priv else float("nan")
        ),
    }


def _bootstrap_recovery(
    correct_normal: np.ndarray,
    correct_fault: np.ndarray,
    correct_steer: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    n_idx = np.where(np.isfinite(correct_normal))[0]
    f_idx = np.where(np.isfinite(correct_fault) & np.isfinite(correct_steer))[0]
    if len(n_idx) == 0 or len(f_idx) == 0:
        return {"R": float("nan"), "untestable": True}
    pn = float(correct_normal[n_idx].mean())
    pf = float(correct_fault[f_idx].mean())
    ps = float(correct_steer[f_idx].mean())
    gap = pn - pf
    lift = ps - pf
    if gap < GAP_FLOOR:
        return {
            "R": float("nan"),
            "P_normal": pn,
            "P_fault": pf,
            "P_steer": ps,
            "gap_normal_fault": gap,
            "lift": lift,
            "untestable": True,
        }
    R_obs = lift / gap
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ni = rng.choice(n_idx, size=len(n_idx), replace=True)
        fi = rng.choice(f_idx, size=len(f_idx), replace=True)
        gap_b = float(correct_normal[ni].mean()) - float(correct_fault[fi].mean())
        if gap_b < GAP_FLOOR:
            continue
        boots.append((float(correct_steer[fi].mean()) - float(correct_fault[fi].mean())) / gap_b)
    if not boots:
        return {
            "R": R_obs,
            "P_normal": pn,
            "P_fault": pf,
            "P_steer": ps,
            "gap_normal_fault": gap,
            "lift": lift,
            "untestable": True,
        }
    ba = np.asarray(boots)
    return {
        "R": R_obs,
        "ci_lo": float(np.quantile(ba, 0.025)),
        "ci_hi": float(np.quantile(ba, 0.975)),
        "P_normal": pn,
        "P_fault": pf,
        "P_steer": ps,
        "gap_normal_fault": gap,
        "lift": lift,
        "untestable": False,
        "excludes_zero_positive": bool(float(np.quantile(ba, 0.025)) > 0),
    }


def _correct_from_rows(rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        if r["s_tool"] != 1:
            out.append(np.nan)
        else:
            out.append(float(r["s_output"] == 1))
    return np.asarray(out, dtype=np.float64)


def _make_steer_hook(loaded: Any, layer: int, direction: np.ndarray, alpha: float):
    from activation_pipeline.steering import ActivationSteerHook

    d_t = torch.tensor(direction, dtype=torch.float32)

    def hook_for_turn(phase: str, _turn: int):
        if phase != "report" or abs(alpha) < 1e-12:
            return None
        return ActivationSteerHook(
            loaded.model,
            layer=layer,
            direction=d_t,
            alpha=float(alpha),
            pos_mode=POS_MODE,
        )

    return hook_for_turn


def _make_patch_hook(loaded: Any, layer: int, donor: np.ndarray):
    from activation_pipeline.steering import ActivationResidualPatchHook

    d_t = torch.tensor(donor, dtype=torch.float32)

    def hook_for_turn(phase: str, _turn: int):
        if phase != "report":
            return None
        return ActivationResidualPatchHook(
            loaded.model,
            layer=layer,
            donor=d_t,
            pos_mode=POS_MODE,
            blend=1.0,
        )

    return hook_for_turn


def main() -> int:
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.hooks import resolve_decoder_layers
    from activation_pipeline.loader import load_model_and_tokenizer

    ga = json.loads(GATE_A.read_text())
    disc_tasks = set(ga["disc_tasks"])
    freeze_tasks = set(ga["freeze_tasks"])
    v_delta_L4 = sdl.load_v_report(VDELTA)

    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    n_layers = len(resolve_decoder_layers(loaded.model))
    assert n_layers == 28, f"expected 28 layers, got {n_layers}"
    capture_layers = list(ALL_LAYERS)

    # Private/mixed tasks; Disc for fit, Freeze for primary test (+ locked-eval extras as secondary)
    all_priv = [
        t for t in sc.TASKS if t["sensitivity"] != "public"
    ]
    plan = [(t["task_id"], t["folder"]) for t in all_priv]
    disc_plan = [(tid, folder) for tid, folder in plan if tid in disc_tasks]
    test_plan = [(tid, folder) for tid, folder in plan if tid in freeze_tasks]
    # Keep enough test mass: freeze ∪ locked-eval val tasks not used only for fit
    locked_val = {"api", "internal", "deploy", "src", "ops", "app"}
    # Primary test = freeze only; secondary report = all collected
    if len(test_plan) < 2:
        test_plan = [(tid, folder) for tid, folder in plan if tid in locked_val - disc_tasks] or plan[-2:]

    print(
        f"=== SYNC_REPAIR_CAUSAL plan={len(plan)} disc={len(disc_plan)} "
        f"test={len(test_plan)} reps={N_REPS} patch_L={PATCH_LAYERS} ===",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Collect paired Normal/Fault states + baselines
    # -------------------------------------------------------------------------
    pairs: list[dict[str, Any]] = []
    for rep in range(N_REPS):
        for j, (tid, folder) in enumerate(plan):
            seed = int(SEED + 77011 * rep + 419 * j)
            state_b = sc.run_tools_phase(
                loaded,
                cls="B",
                task_id=tid,
                folder=folder,
                seed=seed,
                capture_layers=capture_layers,
            )
            state_c = sc.run_tools_phase(
                loaded,
                cls="C",
                task_id=tid,
                folder=folder,
                seed=seed,  # matched RNG; prompts still differ
                capture_layers=capture_layers,
            )
            nb = sc.finalize_episode(loaded, state_b, gate=None)
            nf = sc.finalize_episode(loaded, state_c, gate=None)
            pair = {
                "rep": rep,
                "task_id": tid,
                "folder": folder,
                "seed": seed,
                "split": "disc" if tid in disc_tasks else "freeze",
                "state_b": state_b,
                "state_c": state_c,
                "normal": nb,
                "fault": nf,
                "h_report_b": state_b["h_report_layers"],
                "h_report_c": state_c["h_report_layers"],
                "h_tool_b": state_b["h_tool_layers"],
                "h_tool_c": state_c["h_tool_layers"],
            }
            pairs.append(pair)
            print(
                f"collect r{rep} {tid} N_Δ={nb['delta_sync']} F_Δ={nf['delta_sync']} "
                f"N_out={nb['s_output']} F_out={nf['s_output']}",
                flush=True,
            )

    disc_pairs = [p for p in pairs if p["split"] == "disc"]
    freeze_pairs = [p for p in pairs if p["split"] == "freeze"]
    if not freeze_pairs:
        freeze_pairs = [p for p in pairs if p["task_id"] in locked_val]
    test_pairs = freeze_pairs

    # -------------------------------------------------------------------------
    # Stage A0 — Δ_l diagnostics (report + tool)
    # -------------------------------------------------------------------------
    def layer_delta_stats(point: str) -> list[dict[str, Any]]:
        key_b = "h_report_b" if point == "report" else "h_tool_b"
        key_c = "h_report_c" if point == "report" else "h_tool_c"
        rows = []
        for L in ALL_LAYERS:
            diffs = []
            y = []
            scores = []
            for p in disc_pairs:
                hb = p[key_b]
                hc = p[key_c]
                if hb is None or hc is None or L not in hb or L not in hc:
                    continue
                d = hb[L] - hc[L]
                diffs.append(float(np.linalg.norm(d)))
                # Separability: score = (h_B - h_C) · unit(mean diff) proxy via ||h_B|| projection later
            if not diffs:
                rows.append({"layer": L, "mean_l2": float("nan"), "auc": float("nan")})
                continue
            # Build detect-style AUC: project onto mean(h_B - h_C)
            mats_b, mats_c = [], []
            for p in disc_pairs:
                if p[key_b] is None or L not in p[key_b]:
                    continue
                mats_b.append(p[key_b][L])
                mats_c.append(p[key_c][L])
            if len(mats_b) < 3:
                rows.append({"layer": L, "mean_l2": float(np.mean(diffs)), "auc": float("nan")})
                continue
            Mb = np.stack(mats_b)
            Mc = np.stack(mats_c)
            v = sdl.unit(Mb.mean(0) - Mc.mean(0))
            scores = np.concatenate([Mb @ v, Mc @ v])
            y = np.array([1] * len(Mb) + [0] * len(Mc))
            try:
                auc = float(roc_auc_score(y, scores))
            except ValueError:
                auc = float("nan")
            rows.append({"layer": L, "mean_l2": float(np.mean(diffs)), "auc": auc})
        return rows

    delta_report = layer_delta_stats("report")
    delta_tool = layer_delta_stats("tool")

    def earliest_sep(rows: list[dict]) -> int | None:
        for r in rows:
            if np.isfinite(r["auc"]) and r["auc"] >= AUC_SEP:
                return int(r["layer"])
        return None

    # -------------------------------------------------------------------------
    # Stage A — activation patching on Disc (pick L*), confirm on Freeze
    # -------------------------------------------------------------------------
    def run_patch_sweep(pair_list: list[dict], tag: str) -> dict[int, list[dict]]:
        by_L: dict[int, list[dict]] = {L: [] for L in PATCH_LAYERS}
        for p in pair_list:
            for L in PATCH_LAYERS:
                donor = p["h_report_b"][L]
                hook = _make_patch_hook(loaded, L, donor)
                row = sc.finalize_episode(
                    loaded, p["state_c"], gate=None, hook_for_turn=hook
                )
                row.update(
                    {
                        "arm": f"patch_L{L}",
                        "task_id": p["task_id"],
                        "rep": p["rep"],
                        "layer": L,
                    }
                )
                by_L[L].append(row)
            print(
                f"patch {tag} r{p['rep']} {p['task_id']} "
                + " ".join(
                    f"L{L}={by_L[L][-1]['s_output']}" for L in PATCH_LAYERS
                ),
                flush=True,
            )
        return by_L

    patch_disc = run_patch_sweep(disc_pairs, "disc")
    # Pick L*: max disclose|private among private-fault episodes; require lift vs fault
    fault_disc_rate = _disclose_rate([p["fault"] for p in disc_pairs])
    normal_disc_rate = _disclose_rate([p["normal"] for p in disc_pairs])
    patch_disc_rates = {L: _rates(patch_disc[L]) for L in PATCH_LAYERS}
    best_L = max(
        PATCH_LAYERS,
        key=lambda L: (
            patch_disc_rates[L]["disclose_given_private"]
            if np.isfinite(patch_disc_rates[L]["disclose_given_private"])
            else -1.0
        ),
    )
    best_rate = patch_disc_rates[best_L]["disclose_given_private"]
    patch_restores = (
        np.isfinite(best_rate)
        and np.isfinite(fault_disc_rate)
        and best_rate >= fault_disc_rate + 0.05
    )
    L_star = int(best_L) if patch_restores else int(
        max(delta_report, key=lambda r: r["mean_l2"] if np.isfinite(r["mean_l2"]) else -1.0)["layer"]
    )
    stage_a_decision = "PATCH_HIT" if patch_restores else "PATCH_NULL"

    patch_test = run_patch_sweep(test_pairs, "test")
    patch_test_rates = {L: _rates(patch_test[L]) for L in PATCH_LAYERS}

    print(
        f"Stage A: {stage_a_decision} L*={L_star} "
        f"disc_patch_disclose={best_rate:.3f} fault={fault_disc_rate:.3f} "
        f"normal={normal_disc_rate:.3f}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Stage B — construct directions at L* on Disc
    # -------------------------------------------------------------------------
    Hb, Hc = [], []
    H_dis, H_hid = [], []
    for p in disc_pairs:
        if p["h_report_b"] is None or L_star not in p["h_report_b"]:
            continue
        Hb.append(p["h_report_b"][L_star])
        Hc.append(p["h_report_c"][L_star])
        # detect contrast from realized outcomes on fault+normal private
        for arm_key, row in (("normal", p["normal"]), ("fault", p["fault"])):
            if row["s_tool"] != 1:
                continue
            h = p["h_report_b"][L_star] if arm_key == "normal" else p["h_report_c"][L_star]
            if row["s_output"] == 1:
                H_dis.append(h)
            else:
                H_hid.append(h)

    if len(Hb) < 2:
        raise RuntimeError("insufficient disc pairs for v_repair")
    v_repair = sdl.unit(np.mean(np.stack(Hb), 0) - np.mean(np.stack(Hc), 0))
    if len(H_dis) >= 2 and len(H_hid) >= 2:
        v_detect = sdl.unit(np.mean(np.stack(H_dis), 0) - np.mean(np.stack(H_hid), 0))
        detect_source = "realized_disclose_vs_hidden_at_Lstar"
    else:
        # Fall back: Class B vs Class C mean (same as repair) — flag it
        v_detect = v_repair.copy()
        detect_source = "fallback_equals_repair_insufficient_labels"

    # Also keep L4 detector for reference comparison when L*==4
    cos_repair_detect = float(np.dot(v_repair, v_detect))
    v_parallel = cos_repair_detect * v_detect
    v_perp_raw = v_repair - v_parallel
    v_perp = sdl.unit(v_perp_raw) if float(np.linalg.norm(v_perp_raw)) > 1e-8 else v_perp_raw
    v_parallel_u = sdl.unit(v_parallel) if float(np.linalg.norm(v_parallel)) > 1e-8 else v_parallel

    diffs = [float(np.linalg.norm(hb - hc)) for hb, hc in zip(Hb, Hc)]
    target = 0.25 * float(np.median(diffs)) if diffs else 0.5
    alpha_frozen = min(ALPHA_CANDIDATES, key=lambda a: abs(a - target))

    DIR_OUT.write_text(
        json.dumps(
            {
                "kind": "sync_repair_direction",
                "layer": L_star,
                "vector": v_repair.astype(float).tolist(),
                "meta": {
                    "v_detect": v_detect.astype(float).tolist(),
                    "v_perp": v_perp.astype(float).tolist(),
                    "cos_repair_detect": cos_repair_detect,
                    "alpha_frozen": alpha_frozen,
                    "stage_a": stage_a_decision,
                    "detect_source": detect_source,
                    "n_disc_pairs": len(Hb),
                },
            }
        )
        + "\n"
    )

    # -------------------------------------------------------------------------
    # Stage C — steer on Freeze/test paired Class-C traces
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(SEED + 91)
    r_dir = _orth_to(v_repair, rng)
    arms: dict[str, tuple[np.ndarray, float]] = {
        "fault": (np.zeros_like(v_repair), 0.0),
        "+v_detect": (v_detect, float(alpha_frozen)),
        "+v_repair": (v_repair, float(alpha_frozen)),
        "+v_perp": (v_perp, float(alpha_frozen)),
        "+v_parallel": (v_parallel_u, float(alpha_frozen)),
        "random": (r_dir, float(alpha_frozen)),
    }
    # Reference: L4 v_delta at L4 even if L* differs (prior null replication)
    if L_star == sc.LAYER:
        arms["+v_delta_L4"] = (v_delta_L4, float(alpha_frozen))

    arm_rows: dict[str, list[dict]] = {k: [] for k in arms}
    normal_test_rows = [p["normal"] for p in test_pairs]
    for p in test_pairs:
        for arm_name, (direction, alpha) in arms.items():
            hook = _make_steer_hook(loaded, L_star, direction, alpha)
            row = sc.finalize_episode(
                loaded, p["state_c"], gate=None, hook_for_turn=hook
            )
            row.update(
                {
                    "arm": arm_name,
                    "task_id": p["task_id"],
                    "rep": p["rep"],
                    "layer": L_star,
                    "alpha": alpha,
                }
            )
            arm_rows[arm_name].append(row)
        print(
            f"steer r{p['rep']} {p['task_id']} "
            f"F={arm_rows['fault'][-1]['delta_sync']} "
            f"R={arm_rows['+v_repair'][-1]['delta_sync']} "
            f"D={arm_rows['+v_detect'][-1]['delta_sync']}",
            flush=True,
        )

    rates = {"normal": _rates(normal_test_rows)}
    rates.update({k: _rates(v) for k, v in arm_rows.items()})

    c_normal = _correct_from_rows(normal_test_rows)
    c_fault = _correct_from_rows(arm_rows["fault"])
    recovery = {}
    arm_seeds = {
        "+v_detect": 301,
        "+v_repair": 302,
        "+v_perp": 303,
        "+v_parallel": 304,
        "random": 305,
        "+v_delta_L4": 306,
    }
    for arm in arms:
        if arm == "fault":
            continue
        recovery[arm] = _bootstrap_recovery(
            c_normal,
            c_fault,
            _correct_from_rows(arm_rows[arm]),
            n_boot=N_BOOT,
            seed=SEED + arm_seeds.get(arm, 399),
        )

    # Exploratory α grid for +v_repair only (labeled)
    alpha_grid_rows: dict[str, list[dict]] = {}
    for a in ALPHA_GRID_EXPLORATORY:
        key = f"repair_a{a}"
        alpha_grid_rows[key] = []
        for p in test_pairs:
            hook = _make_steer_hook(loaded, L_star, v_repair, a)
            row = sc.finalize_episode(
                loaded, p["state_c"], gate=None, hook_for_turn=hook
            )
            row.update({"arm": key, "alpha": a, "task_id": p["task_id"], "rep": p["rep"]})
            alpha_grid_rows[key].append(row)
    alpha_grid_rates = {k: _rates(v) for k, v in alpha_grid_rows.items()}

    rv = recovery.get("+v_repair", {})
    rd = recovery.get("+v_detect", {})
    rr = recovery.get("random", {})
    spur_ok = rates["+v_repair"]["spurious"] <= rates["fault"]["spurious"] + SPURIOUS_BUDGET + 1e-9

    if stage_a_decision == "PATCH_NULL" and not (
        rv.get("excludes_zero_positive") if not rv.get("untestable") else False
    ):
        # Prefer PATCH_NULL as headline if patching failed and steer didn't clearly hit
        if rv.get("untestable") or not ((rv.get("R") or 0) > 0 and spur_ok):
            decision = "PATCH_NULL"
        elif rv.get("excludes_zero_positive") and spur_ok and not rd.get("excludes_zero_positive") and not rr.get(
            "excludes_zero_positive"
        ):
            decision = "REPAIR_STEER_HIT"
        elif (rv.get("R") or 0) > 0 and spur_ok:
            decision = "REPAIR_STEER_WEAK"
        else:
            decision = "PATCH_NULL"
    elif rv.get("untestable"):
        decision = "REPAIR_STEER_NULL"
    elif (
        rv.get("excludes_zero_positive")
        and spur_ok
        and not rd.get("excludes_zero_positive")
        and not rr.get("excludes_zero_positive")
    ):
        decision = "REPAIR_STEER_HIT"
    elif rv.get("excludes_zero_positive") and (
        rd.get("excludes_zero_positive") or rr.get("excludes_zero_positive")
    ):
        decision = "REPAIR_STEER_GENERIC"
    elif (rv.get("R") or 0) > 0 and spur_ok:
        decision = "REPAIR_STEER_WEAK"
    else:
        decision = "REPAIR_STEER_NULL"

    payload = {
        "protocol": str(PROTO),
        "experiment": "SYNC_REPAIR_CAUSAL",
        "decision": decision,
        "stage_a": {
            "decision": stage_a_decision,
            "L_star": L_star,
            "patch_layers": list(PATCH_LAYERS),
            "disc_fault_disclose": fault_disc_rate,
            "disc_normal_disclose": normal_disc_rate,
            "disc_patch_rates": {str(k): v for k, v in patch_disc_rates.items()},
            "test_patch_rates": {str(k): v for k, v in patch_test_rates.items()},
            "delta_report": delta_report,
            "delta_tool": delta_tool,
            "earliest_sep_report": earliest_sep(delta_report),
            "earliest_sep_tool": earliest_sep(delta_tool),
        },
        "stage_b": {
            "layer": L_star,
            "alpha_frozen": alpha_frozen,
            "target_displacement": target,
            "cos_repair_detect": cos_repair_detect,
            "detect_source": detect_source,
            "n_disc_pairs": len(Hb),
            "n_disclosed": len(H_dis),
            "n_hidden": len(H_hid),
            "direction_path": str(DIR_OUT),
        },
        "stage_c": {
            "rates": rates,
            "recovery_R": recovery,
            "spurious_budget_ok": spur_ok,
            "alpha_grid_exploratory_rates": alpha_grid_rates,
            "n_test_pairs": len(test_pairs),
            "test_tasks": sorted({p["task_id"] for p in test_pairs}),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    def _fmt_r(d: dict) -> str:
        if not d or d.get("untestable"):
            return "UNTESTABLE"
        return f"R={d['R']:.3f} [{d.get('ci_lo', float('nan')):.3f},{d.get('ci_hi', float('nan')):.3f}]"

    lines = [
        "# SYNC repair causal (detect ≠ repair)",
        "",
        f"- Decision: **`{decision}`**",
        f"- Stage A: `{stage_a_decision}`  L*={L_star}",
        f"- cos(v_repair, v_detect)={cos_repair_detect:.3f}  α*={alpha_frozen}",
        f"- Protocol: `{PROTO.name}`",
        "",
        "## Stage A — patch disclose|private (Disc)",
        "",
        "| Layer | disclose\\|private | hidden |",
        "|---:|---:|---:|",
        f"| fault (no patch) | {fault_disc_rate:.3f} | — |",
        f"| normal | {normal_disc_rate:.3f} | — |",
    ]
    for L in PATCH_LAYERS:
        r = patch_disc_rates[L]
        mark = " ← L*" if L == L_star else ""
        lines.append(
            f"| L{L}{mark} | {r['disclose_given_private']:.3f} | {r['hidden']:.3f} |"
        )
    lines += [
        "",
        "## Stage C — steer @ L* (Freeze/test)",
        "",
        "| Arm | disclose\\|private | hidden | recovery |",
        "|---|---:|---:|---|",
    ]
    for arm in ("normal", "fault", "+v_detect", "+v_repair", "+v_perp", "+v_parallel", "random", "+v_delta_L4"):
        if arm not in rates:
            continue
        r = rates[arm]
        rec = _fmt_r(recovery[arm]) if arm in recovery else "—"
        lines.append(
            f"| {arm} | {r['disclose_given_private']:.3f} | {r['hidden']:.3f} | {rec} |"
        )
    lines += [
        "",
        f"- Spurious budget OK: `{spur_ok}`",
        "",
        "## Headline",
        "",
    ]
    if decision == "REPAIR_STEER_HIT":
        lines.append(
            "Repair direction restores disclosure under fault; detector / random do not. "
            "**Detect ≠ repair** with causal repair at L*."
        )
    elif decision == "PATCH_NULL":
        lines.append(
            "Activation patching did not restore disclosure at any tested layer. "
            "Fault may sit outside the residual pathway (prompt prior / tool path). "
            "Gate remains the viable repair layer."
        )
    else:
        lines.append(
            "No specific activation-level repair demonstrated. "
            "Detector remains monitor-only; gate remains the correction layer."
        )
    lines.append("")
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "L_star": L_star, "stage_a": stage_a_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
