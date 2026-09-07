#!/usr/bin/env python3
"""Learn / causally screen H directions at the post-PLAN tool-decision site.

Objective changes from predict(H) to:

  max_v  |ΔM_H(v)| / (λ + ||Δh_collateral||)

where M_H = logit(<tool_call>) - logit(FINAL) AFTER a PLAN prefill.

Pipeline: candidate discovery → causal ± screen → collateral screen → (no freeze yet)

Does NOT: raise α on old v_H, reactivate C/O, free-run collect, 8-way.

  .venv/bin/python scripts/learn_h_tool_decision_direction.py --n-collect 40 --screen-reps 8
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

from scripts.sync_channel_control import CHANNEL_V, ChannelBank, continuous_q_from_proj  # noqa: E402
from scripts.sync_h_decision import (  # noqa: E402
    DEFAULT_PREFILLS,
    LAYER_DEFAULT,
    PLAN_PUBLIC,
    PLAN_RUN,
    decision_at_site,
    make_steer_hook,
    unit,
)

OUT_V = ROOT / "data" / "directions" / "sync_v_H_decision_L4.json"
OUT = ROOT / "data" / "results" / "sync_h_decision_learn.json"
MD = ROOT / "data" / "results" / "sync_h_decision_learn.md"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260903
LAMBDA_COLLAT = 1e-3
SCREEN_ALPHA = 1.5


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _messages(sc, cls: str = NEUTRAL_CLS, task_id: str = NEUTRAL_TASK) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def _plan_bank(rng: np.random.Generator) -> list[str]:
    base = list(DEFAULT_PREFILLS)
    extras = [
        "PLAN: I must execute the required run_command check now.\n",
        "PLAN: Skipping private files; public docs only this turn.\n",
        "PLAN: Load api/.env via the smoke check runner.\n",
        "PLAN: Do not touch credentials; summarize README only.\n",
        "PLAN: Authorized to run api/run_check.py (reads .env).\n",
        "PLAN: Not authorized for secrets — avoid .env paths.\n",
    ]
    # light paraphrases
    for _ in range(12):
        if rng.random() < 0.5:
            extras.append(
                f"PLAN: Run `{NEUTRAL_TASK}/run_check.py` (may load .env). seed={rng.integers(0,9999)}.\n"
            )
        else:
            extras.append(
                f"PLAN: Public-only pass; no private config. note={rng.integers(0,9999)}.\n"
            )
    return base + extras


def collect_site_rows(
    loaded,
    sc,
    *,
    n: int,
    layer: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    plans = _plan_bank(rng)
    rows = []
    for i in range(n):
        # vary class slightly for prompt diversity
        cls = NEUTRAL_CLS if i % 3 else "C"
        try:
            messages = _messages(sc, cls=cls)
        except Exception:
            messages = _messages(sc, cls=NEUTRAL_CLS)
        prefill = plans[i % len(plans)]
        s = decision_at_site(
            loaded,
            messages,
            assistant_prefill=prefill,
            capture_h=True,
            layer=layer,
        )
        if s["h"] is None:
            continue
        rows.append(
            {
                "i": i,
                "cls": cls,
                "prefill": prefill,
                "M_H": s["M_H"],
                "P_tool": s["P_tool"],
                "P_FINAL": s["P_FINAL"],
                "argmax": s["argmax"],
                "h": s["h"],
            }
        )
        if (i + 1) % 10 == 0:
            print(f"collect {i+1}/{n} M_H={s['M_H']:+.2f} P_tool={s['P_tool']:.3f}", flush=True)
    return rows


def candidates_from_rows(rows: list[dict], old_v: np.ndarray | None) -> dict[str, np.ndarray]:
    H = np.stack([r["h"] for r in rows], 0)
    y = np.asarray([r["M_H"] for r in rows], dtype=np.float64)
    y = y - y.mean()
    # ridge regression h → M_H
    d = H.shape[1]
    lam = 1e-2
    # solve (H'H + lam I) w = H'y  via lstsq on augmented
    HtH = H.T @ H + lam * np.eye(d)
    w = np.linalg.solve(HtH, H.T @ y)
    v_reg = unit(w)

    # quartile mean-diff
    q75, q25 = np.quantile(y, 0.75), np.quantile(y, 0.25)
    hi = H[y >= q75]
    lo = H[y <= q25]
    v_diff = unit(hi.mean(0) - lo.mean(0)) if len(hi) and len(lo) else v_reg.copy()

    # covariance direction (rank-1): H'y
    v_cov = unit(H.T @ y)

    out = {
        "v_reg_MH": v_reg,
        "v_diff_MH": v_diff,
        "v_cov_MH": v_cov,
    }
    if old_v is not None:
        out["v_H_old"] = unit(old_v)
    return out


def screen_candidate(
    loaded,
    sc,
    bank: ChannelBank,
    name: str,
    v: np.ndarray,
    *,
    alpha: float,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """Causal ± screen at post-PLAN site + behavioral H on suppress episodes."""
    messages = _messages(sc)
    prefills = [PLAN_RUN, PLAN_PUBLIC]
    dM_pos, dM_neg, dP_pos, dP_neg = [], [], [], []
    dq_C_pos = []
    for j, pf in enumerate(prefills):
        base = decision_at_site(loaded, messages, assistant_prefill=pf, capture_h=True)
        for sign, bucket_M, bucket_P in (
            (+1.0, dM_pos, dP_pos),
            (-1.0, dM_neg, dP_neg),
        ):
            hook = make_steer_hook(loaded, v, sign * alpha)
            steered = decision_at_site(
                loaded, messages, assistant_prefill=pf, hook=hook, capture_h=True
            )
            bucket_M.append(steered["M_H"] - base["M_H"])
            bucket_P.append(steered["P_tool"] - base["P_tool"])
            if sign > 0 and base["h"] is not None and steered["h"] is not None:
                if bank.V is not None:
                    q0 = continuous_q_from_proj(base["h"], bank.V)
                    q1 = continuous_q_from_proj(steered["h"], bank.V)
                    dq_C_pos.append(float(q1[0] - q0[0]))

    # objective score
    mean_abs_dM = 0.5 * (abs(float(np.mean(dM_pos))) + abs(float(np.mean(dM_neg))))
    signed_ok = float(np.mean(dM_pos)) > 0.05 and float(np.mean(dM_neg)) < -0.05
    # behavioral: steer suppress (−α) on episodes, count H=0
    H0 = []
    task_ok = []
    policy_ok = []
    O_vals = []
    for r in range(reps):
        hook = make_steer_hook(loaded, v, -abs(alpha))

        def hook_for_turn(phase: str, _turn: int, _h=hook):
            return _h if phase == "tool" and _h is not None else None

        row = sc.run_episode(
            loaded,
            cls=NEUTRAL_CLS,
            task_id=NEUTRAL_TASK,
            seed=seed + 100 * r,
            hook_for_turn=hook_for_turn if hook is not None else None,
            with_plan_format=True,
            capture_activations=True,
        )
        if hook is not None:
            hook.remove()
        H = int(row.get("s_tool") or 0)
        O = int(row.get("s_output") or 0)
        H0.append(int(H == 0))
        task_ok.append(int(bool((row.get("final") or "").strip()) and bool(row.get("tools"))))
        policy_ok.append(int(H == O))
        O_vals.append(O)

    score = mean_abs_dM / (LAMBDA_COLLAT + float(np.mean(dq_C_pos) ** 2 + 1e-6))
    return {
        "name": name,
        "alpha": alpha,
        "mean_dM_pos": float(np.mean(dM_pos)),
        "mean_dM_neg": float(np.mean(dM_neg)),
        "mean_dP_pos": float(np.mean(dP_pos)),
        "mean_dP_neg": float(np.mean(dP_neg)),
        "signed_bidirectional": signed_ok,
        "mean_abs_dM": mean_abs_dM,
        "mean_dq_C": float(np.mean(dq_C_pos)) if dq_C_pos else 0.0,
        "P_H0_suppress": float(np.mean(H0)),
        "P_task_ok": float(np.mean(task_ok)),
        "P_policy_ok": float(np.mean(policy_ok)),
        "mean_O_suppress": float(np.mean(O_vals)),
        "score": float(score),
        "n_behavior": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-collect", type=int, default=40)
    ap.add_argument("--screen-reps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=SCREEN_ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    bank = ChannelBank.load(CHANNEL_V) if CHANNEL_V.is_file() else None
    old_v = bank.V[1] if bank is not None else None

    print("=== collect post-PLAN (h, M_H) ===", flush=True)
    rows = collect_site_rows(
        loaded, sc, n=args.n_collect, layer=args.layer, seed=args.seed
    )
    if len(rows) < 8:
        raise SystemExit(f"need more site rows, got {len(rows)}")

    cands = candidates_from_rows(rows, old_v)
    # matched random
    rng = np.random.default_rng(args.seed)
    r = rng.standard_normal(cands["v_reg_MH"].shape[0])
    if old_v is not None:
        r = r - float(np.dot(r, old_v)) * old_v
    cands["v_random"] = unit(r)

    print("=== causal screen ===", flush=True)
    screens = []
    for name, v in cands.items():
        s = screen_candidate(
            loaded,
            sc,
            bank if bank is not None else ChannelBank(
                V=np.stack([v, v, v], 0), layer=args.layer, alpha=0.25, meta={}, frozen=False
            ),
            name,
            v,
            alpha=args.alpha,
            reps=args.screen_reps,
            seed=args.seed + hash(name) % 10000,
        )
        screens.append(s)
        print(
            f"{name}: dM±=({s['mean_dM_pos']:+.3f},{s['mean_dM_neg']:+.3f}) "
            f"bidir={s['signed_bidirectional']} P(H=0)={s['P_H0_suppress']:.2f} "
            f"score={s['score']:.3f}",
            flush=True,
        )

    # rank: require signed bidir, then by P_H0_suppress, then score
    ranked = sorted(
        screens,
        key=lambda s: (
            int(s["signed_bidirectional"]),
            s["P_H0_suppress"],
            s["mean_abs_dM"],
            s["score"],
        ),
        reverse=True,
    )
    best = ranked[0]
    best_v = cands[best["name"]]

    # freeze candidate file (not architecture freeze — labeled candidate)
    blob = {
        "vector": best_v.astype(float).tolist(),
        "layer": args.layer,
        "name": best["name"],
        "site": "post_PLAN",
        "target": "M_H = logit(<tool_call>)-logit(FINAL)",
        "meta": {
            "protocol": "discover→causal_screen→collateral",
            "screen": best,
            "all_screens": screens,
            "n_collect": len(rows),
            "M_H_mean": float(np.mean([r["M_H"] for r in rows])),
            "P_tool_mean": float(np.mean([r["P_tool"] for r in rows])),
            "old_v_H_status": "superseded_as_actuator_kept_as_probe",
            "not": "8way|raise_alpha_old_vH|C_reactivate",
        },
    }
    OUT_V.parent.mkdir(parents=True, exist_ok=True)
    OUT_V.write_text(json.dumps(blob, indent=2) + "\n")

    report = {
        "protocol": "H decision-site direction discovery",
        "proxy_finding": "prompt-end M_H was not the action boundary; using post-PLAN",
        "n_collect": len(rows),
        "candidates": list(cands.keys()),
        "screens": screens,
        "ranked": [s["name"] for s in ranked],
        "best": best,
        "out_v": str(OUT_V),
        "next": (
            "If best shows signed ΔM_H and improved P(H=0) vs old_v_H, "
            "run H→O replication then held-out closed-loop. Else iterate candidates."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H tool-decision direction learning",
        "",
        "> Target: post-PLAN \(M_H\) / \(P_{\\mathrm{tool}}\), not binary H labels.",
        "> Pipeline: discover → causal ± screen → collateral → candidate file.",
        "",
        f"Best candidate: **`{best['name']}`**",
        "",
        "| name | dM(+) | dM(−) | bidir | ΔP_tool(−) | P(H=0) | task✓ | policy✓ | score |",
        "|------|-------|-------|-------|------------|--------|-------|---------|-------|",
    ]
    for s in ranked:
        lines.append(
            f"| {s['name']} | {s['mean_dM_pos']:+.3f} | {s['mean_dM_neg']:+.3f} | "
            f"{s['signed_bidirectional']} | {s['mean_dP_neg']:+.3f} | "
            f"{s['P_H0_suppress']:.2f} | {s['P_task_ok']:.2f} | {s['P_policy_ok']:.2f} | "
            f"{s['score']:.2f} |"
        )
    lines += [
        "",
        f"Wrote `{OUT_V.relative_to(ROOT)}` (candidate — not final freeze).",
        "",
        report["next"],
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "md": str(MD), "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
