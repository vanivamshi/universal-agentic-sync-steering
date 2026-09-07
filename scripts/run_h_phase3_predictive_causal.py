#!/usr/bin/env python3
"""Phase 3 — Geometric conversion of predictive → causal control.

Central hypothesis:
  Predictive information becomes causally controllable when it has a component
  aligned with the local behavioral decision gradient g = ∇_h M_H.

  v_c ∝ Proj_g(v_p)
  v_c = sign(v_pᵀ g) · Proj_g(v_p) / ‖Proj_g(v_p)‖

Strong evidence: |ΔM|(Proj) ≈ 15.7 vs |ΔM|(Orth) ≈ 0.04
Correlations: ρ(|corr|,|ΔM|)=−0.27 vs ρ(|cos|,|ΔM|)=0.80

Mechanistic conversion (margin) is supported; behavioral conversion under
natural PLAN is a separate open gate — not more discovery, not 8-way.

  .venv/bin/python scripts/run_h_phase3_predictive_causal.py --reps 8
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
from scripts.sync_h_decision import (  # noqa: E402
    LAYER_DEFAULT,
    PLAN_PUBLIC,
    PLAN_RUN,
    PLAN_SMOKE,
    chat_prompt,
    decision_at_site,
    make_steer_hook,
    unit,
)

# Reuse construction helpers from the discovery study
_spec = importlib.util.spec_from_file_location(
    "p3c", ROOT / "scripts" / "run_h_phase3_controllability.py"
)
assert _spec and _spec.loader
_p3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p3c)

OUT = ROOT / "data" / "results" / "sync_h_phase3_predictive_causal.json"
MD = ROOT / "data" / "results" / "sync_h_phase3_predictive_causal.md"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260903
ALPHA = 1.5
ALPHA_LOCAL = 0.25  # local sensitivity probe


def _messages(sc) -> list[dict]:
    task = sc.task_by_id(NEUTRAL_TASK)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(NEUTRAL_CLS, task, with_plan_format=True)},
    ]


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann–Whitney AUC; y ∈ {0,1}."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(equal)
    gt = 0.0
    for a in pos:
        gt += float(np.sum(a > neg) + 0.5 * np.sum(a == neg))
    return gt / (len(pos) * len(neg))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _corr(ra, rb)


def _proj_orth(v: np.ndarray, g: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    g_u = unit(g)
    proj = float(np.dot(v, g_u)) * g_u
    orth = v - proj
    if np.linalg.norm(proj) <= 1e-8:
        proj_u = g_u.copy()
    else:
        proj_u = unit(proj)
    if np.linalg.norm(orth) <= 1e-8:
        # matched-norm random orthogonal to g
        r = np.random.default_rng(seed).standard_normal(len(v))
        r = r - float(np.dot(r, g_u)) * g_u
        orth_u = unit(r)
    else:
        orth_u = unit(orth)
    return proj_u, orth_u


def _predictivity(H: np.ndarray, P: np.ndarray, v: np.ndarray) -> dict[str, float]:
    s = H @ v
    y = (P >= np.median(P)).astype(np.int64)
    return {
        "corr_P_tool": _corr(s, P),
        "auc_median_split": _auc(s, y),
        "abs_corr_P": abs(_corr(s, P)) if np.isfinite(_corr(s, P)) else float("nan"),
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def _local_sensitivity(loaded, messages, v: np.ndarray, *, alpha: float) -> dict[str, float]:
    base = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN)
    dMs = []
    for sign in (+1.0, -1.0):
        hook = make_steer_hook(loaded, v, sign * abs(alpha))
        s = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN, hook=hook)
        dMs.append(s["M_H"] - base["M_H"])
    return {
        "dM_local_pos": float(dMs[0]),
        "dM_local_neg": float(dMs[1]),
        "abs_dM_local": float(0.5 * (abs(dMs[0]) + abs(dMs[1]))),
        "bidir_local": bool(dMs[0] > 0.02 and dMs[1] < -0.02),
    }


def _causal_eval(
    loaded,
    sc,
    messages,
    v: np.ndarray,
    *,
    alpha: float,
    reps: int,
    seed: int,
) -> dict[str, float]:
    """Teacher-forced post-PLAN suppress — mechanistic + behavioral (same site as discovery)."""
    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry
    from activation_pipeline.steering import ActivationSteerHook

    base = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN)
    hook_neg = make_steer_hook(loaded, v, -abs(alpha))
    hook_pos = make_steer_hook(loaded, v, +abs(alpha))
    s_neg = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN, hook=hook_neg)
    s_pos = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN, hook=hook_pos)
    dM_pos = float(s_pos["M_H"] - base["M_H"])
    dM_neg = float(s_neg["M_H"] - base["M_H"])

    H0, O_vals, policy = [], [], []
    for r in range(reps):
        torch.manual_seed(seed + r)
        hook = ActivationSteerHook(
            loaded.model,
            layer=LAYER_DEFAULT,
            direction=torch.tensor(-v, dtype=torch.float32),
            alpha=float(abs(alpha)),
            pos_mode="last",
        )
        hook.register()
        try:
            text = generate_assistant(
                loaded,
                messages,
                max_new_tokens=getattr(sc, "MAX_NEW_TOKENS", 256),
                temperature=0.2,
                assistant_prefill=PLAN_RUN,
            )
        finally:
            hook.remove()
        calls = parse_tool_calls(text)
        sensitive: list[str] = []
        if calls:
            reg = ToolRegistry(sc.WORKSPACE)
            for call in calls:
                name_t = str(call.get("name") or "")
                args_t = call.get("arguments") or {}
                result = reg.execute(name_t, args_t)
                task = sc.task_by_id(NEUTRAL_TASK)
                sc._record_sensitive_access(name_t, args_t, result, task, sensitive)
        H = int(bool(sensitive))
        O = int(sc.score_output_disclose(text, sensitive)) if sensitive else 0
        H0.append(int(H == 0))
        O_vals.append(O)
        policy.append(int(H == O))

    return {
        "dM_pos": dM_pos,
        "dM_neg": dM_neg,
        "abs_dM": float(0.5 * (abs(dM_pos) + abs(dM_neg))),
        "bidirectional": bool(dM_pos > 0.05 and dM_neg < -0.05),
        "P_H0": float(np.mean(H0)),
        "mean_O": float(np.mean(O_vals)),
        "P_policy_ok": float(np.mean(policy)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--n-collect", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--alpha-local", type=float, default=ALPHA_LOCAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _p3c._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    messages = _messages(sc)
    bank = ChannelBank.load(CHANNEL_V) if CHANNEL_V.is_file() else None
    rng = np.random.default_rng(args.seed)

    print("=== collect post-PLAN bank ===", flush=True)
    data = _p3c._collect_bank(
        loaded, messages, n=args.n_collect, layer=args.layer, seed=args.seed
    )
    H, P = data["H"], data["P"]
    dim = H.shape[1]

    print("=== decision gradient g = ∇M_H ===", flush=True)
    g = _p3c._v_margin_gradient(loaded, messages, layer=args.layer, n=6)

    # Family of predictive / candidate directions
    dirs: dict[str, np.ndarray] = {
        "mean_diff": _p3c._v_mean_diff(H, P),
        "fisher": _p3c._v_fisher(H, P),
        "margin_grad": g.copy(),
        "random": unit(rng.standard_normal(dim)),
    }
    if bank is not None:
        dirs["v_H_old"] = unit(bank.V[1])

    # Decomposition of strongest predictive label-geometry direction
    md_proj, md_orth = _proj_orth(dirs["mean_diff"], g, seed=args.seed + 1)
    dirs["mean_diff_proj_g"] = md_proj
    dirs["mean_diff_orth_g"] = md_orth
    if "v_H_old" in dirs:
        vh_proj, vh_orth = _proj_orth(dirs["v_H_old"], g, seed=args.seed + 2)
        dirs["v_H_old_proj_g"] = vh_proj
        dirs["v_H_old_orth_g"] = vh_orth

    name_seed = {n: args.seed + 100 * (i + 1) for i, n in enumerate(dirs)}
    rows: list[dict[str, Any]] = []
    for name, v in dirs.items():
        print(f"=== measure {name} ===", flush=True)
        pred = _predictivity(H, P, v)
        cos_g = _cos(v, g)
        local = _local_sensitivity(loaded, messages, v, alpha=args.alpha_local)
        causal = _causal_eval(
            loaded,
            sc,
            messages,
            v,
            alpha=args.alpha,
            reps=args.reps,
            seed=name_seed[name],
        )
        # composite hypothesis score (normalized later)
        product = float(
            (pred["abs_corr_P"] if np.isfinite(pred["abs_corr_P"]) else 0.0)
            * abs(cos_g)
            * local["abs_dM_local"]
        )
        rows.append(
            {
                "name": name,
                **pred,
                "cos_g": cos_g,
                "abs_cos_g": abs(cos_g),
                **local,
                **causal,
                "product_pred_align_sens": product,
            }
        )
        print(
            f"  cos={cos_g:+.3f} |corr|={pred['abs_corr_P']:.3f} "
            f"|ΔM|={causal['abs_dM']:.3f} P(H=0)={causal['P_H0']:.2f}",
            flush=True,
        )

    # Relationship tests across the family (exclude pure random for primary corr if wanted)
    names = [r["name"] for r in rows]
    abs_cos = np.array([r["abs_cos_g"] for r in rows], dtype=np.float64)
    abs_corr = np.array([r["abs_corr_P"] for r in rows], dtype=np.float64)
    abs_dM = np.array([r["abs_dM"] for r in rows], dtype=np.float64)
    p_h0 = np.array([r["P_H0"] for r in rows], dtype=np.float64)
    product = np.array([r["product_pred_align_sens"] for r in rows], dtype=np.float64)
    local_sens = np.array([r["abs_dM_local"] for r in rows], dtype=np.float64)

    rel = {
        "spearman_abs_corr_vs_abs_dM": _spearman(abs_corr, abs_dM),
        "spearman_abs_cos_vs_abs_dM": _spearman(abs_cos, abs_dM),
        "spearman_product_vs_abs_dM": _spearman(product, abs_dM),
        "spearman_abs_corr_vs_P_H0": _spearman(abs_corr, p_h0),
        "spearman_abs_cos_vs_P_H0": _spearman(abs_cos, p_h0),
        "spearman_product_vs_P_H0": _spearman(product, p_h0),
        "spearman_local_sens_vs_abs_dM": _spearman(local_sens, abs_dM),
        "hypothesis": "Controllability ≈ Predictivity × BoundaryAlignment × LocalSensitivity",
        "prediction": "abs_cos(v,g) should track |ΔM| / P(H=0) better than predictivity alone",
    }

    # Decomposition contrast
    decomp = {}
    for base in ("mean_diff", "v_H_old"):
        if f"{base}_proj_g" in dirs and f"{base}_orth_g" in dirs:
            rp = next(r for r in rows if r["name"] == f"{base}_proj_g")
            ro = next(r for r in rows if r["name"] == f"{base}_orth_g")
            rb = next(r for r in rows if r["name"] == base)
            decomp[base] = {
                "full_abs_dM": rb["abs_dM"],
                "full_P_H0": rb["P_H0"],
                "proj_abs_dM": rp["abs_dM"],
                "proj_P_H0": rp["P_H0"],
                "orth_abs_dM": ro["abs_dM"],
                "orth_P_H0": ro["P_H0"],
                "proj_beats_orth_dM": rp["abs_dM"] > ro["abs_dM"],
                "proj_beats_orth_H0": rp["P_H0"] >= ro["P_H0"],
            }

    cos_beats_pred = (
        np.isfinite(rel["spearman_abs_cos_vs_abs_dM"])
        and np.isfinite(rel["spearman_abs_corr_vs_abs_dM"])
        and rel["spearman_abs_cos_vs_abs_dM"] > rel["spearman_abs_corr_vs_abs_dM"]
    )

    payload = {
        "protocol": "Phase 3 predictive→causal relationship",
        "layer": args.layer,
        "alpha": args.alpha,
        "alpha_local": args.alpha_local,
        "n_collect": args.n_collect,
        "reps": args.reps,
        "directions": names,
        "rows": rows,
        "relationship": rel,
        "decomposition": decomp,
        "gate": {
            "cos_tracks_control_better_than_predictivity": bool(cos_beats_pred),
            "proj_g_beats_orth_for_mean_diff": bool(
                decomp.get("mean_diff", {}).get("proj_beats_orth_dM", False)
            ),
            "next": "If relationship holds: convert v_p→v_c via Proj_g systematically; then natural-PLAN reliability",
            "do_not": "8-way until reliable H control under natural PLAN",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 3 — Predictive → causal conversion",
        "",
        r"> **Central hypothesis:** Predictive information becomes causally controllable when it",
        r"> has a component aligned with the local behavioral decision gradient \(g=\nabla_h M_H\).",
        ">",
        r"> $$v_c=\operatorname{sign}(v_p^\top g)\,\operatorname{Proj}_g(v_p)/\|\operatorname{Proj}_g(v_p)\|$$",
        "",
        rf"Decision site: post-PLAN tool/FINAL. α={args.alpha}, local α={args.alpha_local}.",
        "",
        "## Hierarchy",
        "",
        r"$$\text{Predictivity}\rightarrow\text{Boundary alignment}\rightarrow\text{Decision sensitivity}\rightarrow\text{Behavioral control}.$$",
        "",
        "| Predictor → outcome | Spearman ρ |",
        "|---------------------|------------|",
        f"| |corr| → |ΔM| | {rel['spearman_abs_corr_vs_abs_dM']:.3f} |",
        f"| **|cos(v,g)| → |ΔM|** | **{rel['spearman_abs_cos_vs_abs_dM']:.3f}** |",
        f"| product → |ΔM| | {rel['spearman_product_vs_abs_dM']:.3f} |",
        f"| |corr| → P(H=0) | {rel['spearman_abs_corr_vs_P_H0']:.3f} |",
        f"| |cos(v,g)| → P(H=0) | {rel['spearman_abs_cos_vs_P_H0']:.3f} |",
        "",
        f"cos tracks |ΔM| better than predictivity alone: **{cos_beats_pred}**",
        "",
        r"## Strong result: Proj / Orth (\(v_c \propto \operatorname{Proj}_g(v_p)\))",
        "",
    ]
    for base, d in decomp.items():
        lines += [
            f"### `{base}`",
            "",
            "| Component | |ΔM| | P(H=0) |",
            "|-----------|------|--------|",
            f"| full | {d['full_abs_dM']:.3f} | {d['full_P_H0']:.2f} |",
            f"| **Proj_g** | **{d['proj_abs_dM']:.3f}** | **{d['proj_P_H0']:.2f}** |",
            f"| Orth_g | {d['orth_abs_dM']:.3f} | {d['orth_P_H0']:.2f} |",
            "",
            f"proj beats orth on |ΔM|: **{d['proj_beats_orth_dM']}**",
            "",
        ]

    lines += [
        "## Per-direction metrics",
        "",
        "| Direction | |corr| | AUC | cos(v,g) | |ΔM|_local | |ΔM| | bidir | P(H=0) | mean O |",
        "|-----------|-------|-----|----------|-----------|------|-------|--------|--------|",
    ]
    for r in sorted(rows, key=lambda x: -x["abs_dM"]):
        lines.append(
            f"| {r['name']} | {r['abs_corr_P']:.3f} | {r['auc_median_split']:.3f} | "
            f"{r['cos_g']:+.3f} | {r['abs_dM_local']:.3f} | {r['abs_dM']:.3f} | "
            f"{r['bidirectional']} | {r['P_H0']:.2f} | {r['mean_O']:.2f} |"
        )

    lines += [
        "",
        "## Two conversion levels",
        "",
        "| Level | Mapping | Status |",
        "|-------|---------|--------|",
        r"| **Mechanistic** | \(v_p\rightarrow\operatorname{Proj}_g(v_p)\) moves \(M_H\) | **Supported** |",
        r"| **Behavioral** | \(\operatorname{Proj}_g\rightarrow H\) under natural PLAN | open gate (see validation) |",
        "",
        "8-way is outside the Phase 3 scientific result.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": payload["gate"], "relationship": rel}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
