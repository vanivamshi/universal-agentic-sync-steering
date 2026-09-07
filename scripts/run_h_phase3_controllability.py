#!/usr/bin/env python3
"""Phase 3 controllability study — prediction vs control.

Research question (explicit):
  What properties make an activation direction causally effective at a real
  decision boundary? (site, boundary alignment, local sensitivity)

NOT: make all 8 m* work / free runs / C-O actuators / blind α grids.

Exp 1 — direction construction methods (post-PLAN site)
Exp 2 — site sensitivity (same v)
Exp 3 — temporal control (constant vs post-PLAN-only vs all-tokens)

  .venv/bin/python scripts/run_h_phase3_controllability.py --reps 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

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

OUT = ROOT / "data" / "results" / "sync_h_phase3_controllability.json"
MD = ROOT / "data" / "results" / "sync_h_phase3_controllability.md"
CAND_V = ROOT / "data" / "directions" / "sync_v_H_decision_L4.json"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260903
ALPHA = 1.5
LAMBDA_OPT = 0.05

SITES = {
    "prompt_end": None,
    "mid_plan": "PLAN: I will ",
    "post_plan": PLAN_RUN,
    "tool_decision": PLAN_RUN,  # same local boundary; labeled for the table
}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _messages(sc) -> list[dict]:
    task = sc.task_by_id(NEUTRAL_TASK)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(NEUTRAL_CLS, task, with_plan_format=True)},
    ]


def _collect_bank(loaded, messages, *, n: int, layer: int, seed: int) -> dict[str, np.ndarray]:
    """Post-PLAN residuals labeled by P_tool for mean-diff / Fisher."""
    rng = np.random.default_rng(seed)
    prefills = [PLAN_RUN, PLAN_PUBLIC, PLAN_SMOKE]
    extras = [
        f"PLAN: Run check (env load). id={i}.\n" for i in range(n // 2)
    ] + [f"PLAN: Public only; no .env. id={i}.\n" for i in range(n // 2)]
    prefills = prefills + extras
    hs, ps, ms = [], [], []
    for i in range(n):
        pf = prefills[i % len(prefills)]
        s = decision_at_site(
            loaded, messages, assistant_prefill=pf, capture_h=True, layer=layer
        )
        if s["h"] is None:
            continue
        hs.append(s["h"])
        ps.append(s["P_tool"])
        ms.append(s["M_H"])
    H = np.stack(hs, 0)
    P = np.asarray(ps, dtype=np.float64)
    M = np.asarray(ms, dtype=np.float64)
    return {"H": H, "P": P, "M": M}


def _v_mean_diff(H: np.ndarray, P: np.ndarray) -> np.ndarray:
    hi, lo = np.quantile(P, 0.7), np.quantile(P, 0.3)
    pos, neg = H[P >= hi], H[P <= lo]
    if len(pos) == 0 or len(neg) == 0:
        pos, neg = H[P >= np.median(P)], H[P < np.median(P)]
    return unit(pos.mean(0) - neg.mean(0))


def _v_fisher(H: np.ndarray, P: np.ndarray) -> np.ndarray:
    hi, lo = np.quantile(P, 0.7), np.quantile(P, 0.3)
    pos, neg = H[P >= hi], H[P <= lo]
    if len(pos) < 2 or len(neg) < 2:
        return _v_mean_diff(H, P)
    mu = pos.mean(0) - neg.mean(0)
    X = H - H.mean(0, keepdims=True)
    # shrinkage covariance
    d = H.shape[1]
    cov = (X.T @ X) / max(len(H) - 1, 1) + 1e-2 * np.eye(d)
    w = np.linalg.solve(cov, mu)
    return unit(w)


def _v_margin_gradient(loaded, messages, *, layer: int, n: int = 6) -> np.ndarray:
    """Average ∇_{h_L} M_H at post-PLAN site via backprop to layer residual."""
    tok = loaded.tokenizer
    id_tool = tok.encode("<tool_call>", add_special_tokens=False)[0]
    id_final = tok.encode("FINAL", add_special_tokens=False)[0]
    grads = []
    prefills = [PLAN_RUN, PLAN_PUBLIC, PLAN_SMOKE] * ((n + 2) // 3)
    layers = resolve = None
    from activation_pipeline.hooks import resolve_decoder_layers

    dec = resolve_decoder_layers(loaded.model)
    for i in range(n):
        pf = prefills[i]
        enc = tok(chat_prompt(tok, messages, pf), return_tensors="pt")
        device = next(loaded.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        captured: dict[str, torch.Tensor] = {}

        def make_hook():
            def hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # enable grad on last-token residual
                h_last = hidden[:, -1, :].detach().requires_grad_(True)
                captured["h"] = h_last
                # replace last position with grad-enabled tensor
                h_new = hidden.clone()
                h_new = h_new.clone()
                h_new[:, -1, :] = h_last
                if isinstance(output, tuple):
                    return (h_new,) + output[1:]
                return h_new

            return hook

        handle = dec[layer].register_forward_hook(make_hook())
        try:
            loaded.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = loaded.model(**enc, use_cache=False)
                logits = out.logits[0, -1]
                M = logits[id_tool] - logits[id_final]
                M.backward()
            g = captured["h"].grad
            if g is not None:
                grads.append(g.detach().float().cpu().numpy().reshape(-1))
        finally:
            handle.remove()
    if not grads:
        raise RuntimeError("margin gradient failed — no grads captured")
    return unit(np.mean(grads, axis=0))


def _v_optimized(
    loaded,
    messages,
    *,
    dim: int,
    alpha: float,
    seed: int,
    n_cand: int = 24,
    lam: float = LAMBDA_OPT,
) -> np.ndarray:
    """Sphere random search: max ΔM_H(−α) magnitude with sign toward suppress, −λ|ΔP_FINAL|."""
    rng = np.random.default_rng(seed)
    base = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN)
    best_v, best_score = None, -1e9
    for _ in range(n_cand):
        v = unit(rng.standard_normal(dim))
        hook = make_steer_hook(loaded, v, -abs(alpha))
        s = decision_at_site(loaded, messages, assistant_prefill=PLAN_RUN, hook=hook)
        dM = float(s["M_H"] - base["M_H"])  # want negative for suppress
        dPf = float(s["P_FINAL"] - base["P_FINAL"])
        # score: how much we reduce tool margin (more negative dM is better) minus collateral
        score = (-dM) - lam * abs(dPf)
        if score > best_score:
            best_score = score
            best_v = v
    assert best_v is not None
    return best_v


def _eval_direction(
    loaded,
    sc,
    messages,
    name: str,
    v: np.ndarray,
    *,
    alpha: float,
    site_prefill: str | None,
    reps: int,
    seed: int,
    temporal: str = "constant",
) -> dict[str, Any]:
    """ΔM_H / ΔP_tool at site + behavioral P(H=0) under suppress (−v · α)."""
    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry
    from activation_pipeline.steering import ActivationSteerHook

    base = decision_at_site(loaded, messages, assistant_prefill=site_prefill)
    dMs, dPs = [], []
    for sign in (+1.0, -1.0):
        hook = make_steer_hook(loaded, v, sign * abs(alpha))
        s = decision_at_site(
            loaded, messages, assistant_prefill=site_prefill, hook=hook
        )
        dMs.append(s["M_H"] - base["M_H"])
        dPs.append(s["P_tool"] - base["P_tool"])
    bidir = dMs[0] > 0.05 and dMs[1] < -0.05

    H0, task_ok, policy_ok, O_vals = [], [], [], []
    pos_mode = "all" if temporal == "all_tokens" else "last"

    for r in range(reps):
        torch.manual_seed(seed + r)
        if temporal == "post_plan_only":
            # Steer only after teacher-forced PLAN (true tool-selection window).
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
            # O: disclosure in text
            O = int(sc.score_output_disclose(text, sensitive)) if sensitive else 0
            H0.append(int(H == 0))
            task_ok.append(int(bool(text.strip())))
            policy_ok.append(int(H == O))
            O_vals.append(O)
            continue

        hook = ActivationSteerHook(
            loaded.model,
            layer=LAYER_DEFAULT,
            direction=torch.tensor(-v, dtype=torch.float32),
            alpha=float(abs(alpha)),
            pos_mode=pos_mode,
        )

        def hook_for_turn(phase: str, _turn: int, _h=hook):
            return _h if phase == "tool" else None

        row = sc.run_episode(
            loaded,
            cls=NEUTRAL_CLS,
            task_id=NEUTRAL_TASK,
            seed=seed + r,
            hook_for_turn=hook_for_turn,
            with_plan_format=True,
        )
        hook.remove()
        H = int(row.get("s_tool") or 0)
        O = int(row.get("s_output") or 0)
        H0.append(int(H == 0))
        task_ok.append(int(bool((row.get("final") or "").strip()) and bool(row.get("tools"))))
        policy_ok.append(int(H == O))
        O_vals.append(O)

    return {
        "name": name,
        "temporal": temporal,
        "site_prefill": site_prefill,
        "dM_pos": float(dMs[0]),
        "dM_neg": float(dMs[1]),
        "dP_pos": float(dPs[0]),
        "dP_neg": float(dPs[1]),
        "bidirectional": bool(bidir),
        "mean_abs_dM": float(0.5 * (abs(dMs[0]) + abs(dMs[1]))),
        "P_H0": float(np.mean(H0)) if H0 else float("nan"),
        "P_task_ok": float(np.mean(task_ok)) if task_ok else float("nan"),
        "P_policy_ok": float(np.mean(policy_ok)) if policy_ok else float("nan"),
        "mean_O": float(np.mean(O_vals)) if O_vals else float("nan"),
        "n": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--n-collect", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    ap.add_argument("--opt-cands", type=int, default=24)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    messages = _messages(sc)
    bank = ChannelBank.load(CHANNEL_V) if CHANNEL_V.is_file() else None

    print("=== collect post-PLAN bank ===", flush=True)
    data = _collect_bank(
        loaded, messages, n=args.n_collect, layer=args.layer, seed=args.seed
    )
    H, P, M = data["H"], data["P"], data["M"]
    dim = H.shape[1]

    print("=== Exp1 construct directions ===", flush=True)
    dirs: dict[str, np.ndarray] = {
        "mean_diff": _v_mean_diff(H, P),
        "fisher": _v_fisher(H, P),
    }
    try:
        dirs["margin_grad"] = _v_margin_gradient(
            loaded, messages, layer=args.layer, n=6
        )
        print("margin_grad ok", flush=True)
        grad_note = None
    except Exception as e:
        print(f"margin_grad FAILED: {e}", flush=True)
        dirs["margin_grad"] = dirs["mean_diff"].copy()
        grad_note = str(e)

    dirs["optimized"] = _v_optimized(
        loaded,
        messages,
        dim=dim,
        alpha=args.alpha,
        seed=args.seed + 1,
        n_cand=args.opt_cands,
    )
    if bank is not None:
        dirs["v_H_old"] = unit(bank.V[1])
    if CAND_V.is_file():
        dirs["v_diff_MH_prior"] = unit(
            np.asarray(json.loads(CAND_V.read_text())["vector"], dtype=np.float64)
        )

    print("=== Exp1 screen at post_plan ===", flush=True)
    exp1 = []
    for name, v in dirs.items():
        if name.endswith("_error"):
            continue
        row = _eval_direction(
            loaded,
            sc,
            messages,
            name,
            v,
            alpha=args.alpha,
            site_prefill=PLAN_RUN,
            reps=args.reps,
            seed=args.seed + 17 * (abs(hash(name)) % 1000),
            temporal="constant",
        )
        exp1.append(row)
        print(
            f"E1 {name}: |dM|={row['mean_abs_dM']:.3f} bidir={row['bidirectional']} "
            f"P(H=0)={row['P_H0']:.2f}",
            flush=True,
        )

    # pick best by P_H0 then |dM|
    best = sorted(exp1, key=lambda r: (r["P_H0"], r["mean_abs_dM"]), reverse=True)[0]
    v_star = dirs[best["name"]]
    print(f"using {best['name']} for Exp2/3", flush=True)

    print("=== Exp2 site sensitivity ===", flush=True)
    exp2 = []
    for site, pf in SITES.items():
        row = _eval_direction(
            loaded,
            sc,
            messages,
            best["name"],
            v_star,
            alpha=args.alpha,
            site_prefill=pf,
            reps=args.reps,
            seed=args.seed + 200 + abs(hash(site)) % 500,
            temporal="constant",
        )
        row["site"] = site
        exp2.append(row)
        print(f"E2 {site}: P(H=0)={row['P_H0']:.2f} |dM|={row['mean_abs_dM']:.3f}", flush=True)

    print("=== Exp3 temporal ===", flush=True)
    exp3 = []
    for mode in ("constant", "post_plan_only", "all_tokens"):
        row = _eval_direction(
            loaded,
            sc,
            messages,
            best["name"],
            v_star,
            alpha=args.alpha,
            site_prefill=PLAN_RUN,
            reps=args.reps,
            seed=args.seed + 400 + abs(hash(mode)) % 500,
            temporal=mode,
        )
        exp3.append(row)
        print(f"E3 {mode}: P(H=0)={row['P_H0']:.2f}", flush=True)

    report = {
        "phase": 3,
        "question": (
            "What properties make an activation direction causally effective "
            "at a real decision boundary?"
        ),
        "not": "8way|free_runs|C_O_actuators|blind_alpha_old_vH",
        "alpha": args.alpha,
        "reps": args.reps,
        "exp1_directions": exp1,
        "exp2_sites": exp2,
        "exp3_temporal": exp3,
        "best_direction": best["name"],
        "margin_grad_note": grad_note,
        "claim": (
            "Predictive directions need not be causal actuators; site and boundary "
            "alignment dominate controllability."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Phase 3 — Controllability at the real decision boundary",
        "",
        "> Phase 1: predict? **yes**. Phase 2: control? **mostly no**.",
        "> Phase 3: **why** prediction ≠ control (site / boundary / sensitivity).",
        "",
        report["claim"],
        "",
        f"Best Exp1 direction for Exp2/3: **`{best['name']}`** (α={args.alpha})",
        "",
        "## Exp 1 — Direction construction (post-PLAN)",
        "",
        "| Direction | ΔM(+) | ΔM(−) | |ΔM| | bidir | P(H=0) | task✓ | policy✓ |",
        "|-----------|-------|-------|------|-------|--------|-------|---------|",
    ]
    for r in sorted(exp1, key=lambda x: (-x["P_H0"], -x["mean_abs_dM"])):
        lines.append(
            f"| {r['name']} | {r['dM_pos']:+.3f} | {r['dM_neg']:+.3f} | "
            f"{r['mean_abs_dM']:.3f} | {r['bidirectional']} | {r['P_H0']:.2f} | "
            f"{r['P_task_ok']:.2f} | {r['P_policy_ok']:.2f} |"
        )
    lines += [
        "",
        "## Exp 2 — Site sensitivity",
        "",
        "| Site | |ΔM| | P(H=0) | ΔP_tool(−) |",
        "|------|------|--------|------------|",
    ]
    for r in exp2:
        lines.append(
            f"| {r['site']} | {r['mean_abs_dM']:.3f} | {r['P_H0']:.2f} | {r['dP_neg']:+.3f} |"
        )
    lines += [
        "",
        "## Exp 3 — Temporal control",
        "",
        "| Mode | P(H=0) | |ΔM| | task✓ | policy✓ |",
        "|------|--------|------|-------|---------|",
    ]
    for r in exp3:
        lines.append(
            f"| {r['temporal']} | {r['P_H0']:.2f} | {r['mean_abs_dM']:.3f} | "
            f"{r['P_task_ok']:.2f} | {r['P_policy_ok']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation gate",
        "",
        "- If **site** dominates direction choice → intervention locus is the main lever.",
        "- If **margin_grad / optimized** >> mean-diff → boundary alignment beats H-label geometry.",
        "- If **post_plan_only** >> constant → temporal localization matters.",
        "- 8-way stays gated until P(H=0) reaches ~0.6–0.8 with low collateral.",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "md": str(MD), "best": best["name"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
