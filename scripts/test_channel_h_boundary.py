#!/usr/bin/env python3
"""H tool-decision boundary experiment (suppression targets only).

Closed-loop eval failed to improve sync. Do NOT run another generic α grid
or C+H ablation. Question:

  Does increasing v_H displacement move the actual tool-selection margin M_H,
  and is there a finite dose where H reliably flips to 0?

  M_H = logit(<tool_call>) - logit(FINAL)   at first generation token

Primary endpoint:  P(H'=0 | m_H*=0)
Secondary:         ΔM_H, Δq_H, ΔO, ΔE

Arms: v_H (signed α), matched random, α=0, opposite +v_H at |α|=2.
α ∈ {0, -1, -1.5, -2, -2.5, -3} on m* with m_H*=0 only.
C is observe-only (frozen out).

  .venv/bin/python scripts/test_channel_h_boundary.py --reps 10
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
    continuous_q_from_proj,
    unit,
)

OUT = ROOT / "data" / "results" / "sync_channel_h_boundary.json"
MD = ROOT / "data" / "results" / "sync_channel_h_boundary.md"
ARCH = ROOT / "data" / "directions" / "sync_hierarchical_architecture.json"
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
LAYER = 4
SEED = 20260902
# Suppression population only — do not dilute with m_H*=1
SUPPRESS_MSTAR = ((0, 0, 0), (1, 0, 0))
ALPHAS_DEFAULT = (0.0, -1.0, -1.5, -2.0, -2.5, -3.0)


def _load_sc():
    spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _chat_prompt(tok, messages: list[dict]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def tool_decision_margin(
    loaded,
    messages: list[dict],
    *,
    hook=None,
) -> dict[str, float]:
    """M_H = logit(<tool_call>) − logit(FINAL) at the first generation position."""
    tok = loaded.tokenizer
    prompt = _chat_prompt(tok, messages)
    enc = tok(prompt, return_tensors="pt")
    device = next(loaded.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    if hook is not None:
        hook.register()
    try:
        out = loaded.model(**enc, use_cache=False)
    finally:
        if hook is not None:
            hook.remove()

    logits = out.logits[0, -1].float()
    id_tool = tok.encode("<tool_call>", add_special_tokens=False)[0]
    id_final = tok.encode("FINAL", add_special_tokens=False)[0]
    logit_tool = float(logits[id_tool].item())
    logit_final = float(logits[id_final].item())
    # also log Softmax probs for readability
    logp = torch.log_softmax(logits, dim=-1)
    return {
        "M_H": logit_tool - logit_final,
        "logit_tool": logit_tool,
        "logit_final": logit_final,
        "logp_tool": float(logp[id_tool].item()),
        "logp_final": float(logp[id_final].item()),
        "id_tool": int(id_tool),
        "id_final": int(id_final),
    }


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


def _make_hook(loaded, direction: np.ndarray, alpha: float):
    from activation_pipeline.steering import ActivationSteerHook

    if abs(alpha) < 1e-12:
        return None
    return ActivationSteerHook(
        loaded.model,
        layer=LAYER,
        direction=torch.tensor(np.sign(alpha) * direction, dtype=torch.float32),
        alpha=float(abs(alpha)),
        pos_mode="last",
        collect_stats=True,
    )


def _orth_random(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r = rng.standard_normal(v.shape[0])
    r = r - float(np.dot(r, v)) * v
    return unit(r)


def _build_tool_messages(sc, cls: str, task_id: str) -> list[dict]:
    """Same prompt the episode uses at the first tool-decision point."""
    task = sc.task_by_id(task_id)
    system = sc.system_prompt()
    user = sc.task_user_message(cls, task, with_plan_format=True)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_one(
    sc,
    loaded,
    bank: ChannelBank,
    *,
    m_star: tuple[int, int, int],
    seed: int,
    alpha: float,
    arm: str,
    v_H: np.ndarray,
    v_rand: np.ndarray,
) -> dict[str, Any]:
    messages = _build_tool_messages(sc, NEUTRAL_CLS, NEUTRAL_TASK)

    if arm == "random":
        direction = v_rand
    else:
        direction = v_H  # signed α carries suppress (−) or opposite (+)

    hook_m = _make_hook(loaded, direction, alpha) if abs(alpha) > 1e-12 else None
    margin0 = tool_decision_margin(loaded, messages, hook=None)
    margin1 = tool_decision_margin(loaded, messages, hook=hook_m)
    dM = float(margin1["M_H"] - margin0["M_H"])

    # Analytical q_H shift on a free residual at tool site (pre-episode)
    from activation_pipeline.directions import last_token_residual

    acts = last_token_residual(loaded, messages, layers=[LAYER], cast_dtype=torch.float32)
    h = acts[LAYER].numpy().astype(np.float64)
    score = float(h @ v_H)
    q_pre = 1.0 / (1.0 + np.exp(-score))
    # Effective add along v_H: for arm v_H, delta_score = alpha; for random ≈ 0 on v_H
    if arm == "random":
        delta_score = 0.0
    else:
        delta_score = float(alpha)  # hook: sign(α)*v * |α| = α * v
    q_post = 1.0 / (1.0 + np.exp(-(score + delta_score)))
    dq_H = float(q_post - q_pre)

    # Behavioral episode with same steer at tool phase
    hook_ep = _make_hook(loaded, direction, alpha) if abs(alpha) > 1e-12 else None

    def hook_for_turn(phase: str, _turn: int, _h=hook_ep):
        if _h is not None and phase == "tool":
            return _h
        return None

    row = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn if hook_ep is not None else None,
        with_plan_format=True,
        capture_activations=True,
    )
    if hook_ep is not None:
        hook_ep.remove()

    S = _S_from_row(row)
    H = S[1]
    O = S[2]
    E = float(sum(abs(int(m_star[i]) - int(S[i])) for i in range(3)))
    return {
        "m_star": list(m_star),
        "alpha": alpha,
        "arm": arm,
        "seed": seed,
        "M_H_base": margin0["M_H"],
        "M_H_steer": margin1["M_H"],
        "dM_H": dM,
        "logit_tool_steer": margin1["logit_tool"],
        "logit_final_steer": margin1["logit_final"],
        "q_H_pre": q_pre,
        "q_H_post": q_post,
        "dq_H": dq_H,
        "S": S,
        "H": H,
        "O": O,
        "H_suppressed": int(H == 0),
        "E_l1": E,
        "tools": list(row.get("tools") or []),
        "sensitive_paths": list(row.get("sensitive_paths") or []),
        "final": (row.get("final") or "")[:160],
    }


def _curve(trials: list[dict], alphas: list[float], arm: str) -> list[dict]:
    out = []
    for a in alphas:
        rows = [t for t in trials if t["arm"] == arm and abs(t["alpha"] - a) < 1e-12]
        if not rows:
            continue
        out.append(
            {
                "alpha": a,
                "n": len(rows),
                "mean_dM_H": float(np.mean([t["dM_H"] for t in rows])),
                "mean_M_H_steer": float(np.mean([t["M_H_steer"] for t in rows])),
                "P_M_H_neg": float(np.mean([t["M_H_steer"] < 0 for t in rows])),
                "mean_dq_H": float(np.mean([t["dq_H"] for t in rows])),
                "P_H0": float(np.mean([t["H_suppressed"] for t in rows])),
                "mean_O": float(np.mean([t["O"] for t in rows])),
                "mean_E": float(np.mean([t["E_l1"] for t in rows])),
            }
        )
    return out


def _diagnose(curve_vh: list[dict], curve_rand: list[dict]) -> dict[str, Any]:
    """A: margin crosses / B: moves no cross / C: margin flat."""
    if len(curve_vh) < 2:
        return {"outcome": "inconclusive", "detail": "too few α points"}
    xs = [p["alpha"] for p in curve_vh if abs(p["alpha"]) > 1e-12]
    ys = [p["mean_dM_H"] for p in curve_vh if abs(p["alpha"]) > 1e-12]
    p_h0 = [p["P_H0"] for p in curve_vh if abs(p["alpha"]) > 1e-12]
    m_steer = [p["mean_M_H_steer"] for p in curve_vh if abs(p["alpha"]) > 1e-12]
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    crosses = any(m < 0 for m in m_steer)
    # also require P_H0 rising with more negative α
    if len(p_h0) >= 2:
        p_slope = float(np.polyfit(xs, p_h0, 1)[0])  # more neg α → want higher P_H0 → neg slope of P vs α? α↓ P↑ ⇒ slope neg
    else:
        p_slope = float("nan")
    rand_dM = float(np.mean([p["mean_dM_H"] for p in curve_rand if abs(p["alpha"]) > 1e-12] or [0.0]))

    if abs(slope) < 0.05 and abs(ys[-1] if ys else 0) < 0.2:
        outcome = "C"
        detail = "Margin barely moves — q_H may not be the tool-decision variable; move site/remeasure."
    elif crosses and (p_h0 and max(p_h0) >= 0.5):
        outcome = "A"
        # estimate crossing α where mean M_H_steer ≈ 0
        cross_alpha = None
        for p in curve_vh:
            if p["mean_M_H_steer"] < 0:
                cross_alpha = p["alpha"]
                break
        detail = (
            f"Margin moves monotonically and crosses; use minimum effective α≈{cross_alpha}. "
            f"dM/dα={slope:.3f}, max P(H=0)={max(p_h0):.2f}."
        )
    elif abs(slope) >= 0.05 or (ys and abs(ys[0] - ys[-1]) > 0.5):
        outcome = "B"
        detail = (
            "Margin moves with α but does not yield reliable H=0 — weakly aligned direction; "
            "relearn H at tool-selection site rather than increasing α indefinitely."
        )
    else:
        outcome = "inconclusive"
        detail = "No clean pattern yet."

    return {
        "outcome": outcome,
        "detail": detail,
        "dM_H_d_alpha": slope,
        "P_H0_d_alpha": p_slope,
        "rand_mean_dM_H": rand_dM,
        "crosses_zero_M_H": crosses,
        "max_P_H0": float(max(p_h0)) if p_h0 else None,
        "labels": {
            "A": "margin crosses → finite effective dose",
            "B": "margin moves, no reliable flip → relearn direction",
            "C": "margin flat → wrong site / correlated q_H",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS_DEFAULT))
    ap.add_argument("--opp-alpha", type=float, default=2.0, help="opposite-sign control |α|")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not CHANNEL_V.is_file():
        raise SystemExit("need sync_channel_V_L4.json")
    bank = ChannelBank.load(CHANNEL_V)
    v_H = bank.V[1]
    rng = np.random.default_rng(args.seed)
    v_rand = _orth_random(v_H, rng)

    # Freeze C out of architecture
    arch = {
        "architecture": "hierarchical_H_only",
        "frozen": True,
        "graph": "C(observe) → H(actuator) → execution → O(observe)",
        "controller": "d = g_H(e_H) v_H",
        "actuators": {"v_C": None, "v_H": "primary", "v_O": None},
        "v_C_status": "observe_only_frozen_out",
        "scientific_claim": (
            "Dense intervention supports H as causal bottleneck. Closed-loop sync not yet "
            "effective. Next: measure whether v_H displaces the tool-selection margin M_H."
        ),
        "status_pipeline": [
            "A PASS",
            "B.1 H pass / C weak / O null",
            "B.2 H mediation SUPPORT",
            "Architecture FROZEN (H-only)",
            "Closed-loop FAIL / not yet effective",
            "YOU ARE HERE: H decision-boundary experiment",
            "then: reliable H suppression → H→O propagation → held-out sync → 8-way",
        ],
        "not": "8way|C+H|generic_dose|more_free_runs",
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

    # α=0 once per m*×rep (shared baseline arm)
    for m_star in SUPPRESS_MSTAR:
        for r in range(args.reps):
            t = _run_one(
                sc, loaded, bank,
                m_star=m_star, seed=args.seed + 11 * k + r,
                alpha=0.0, arm="alpha0", v_H=v_H, v_rand=v_rand,
            )
            trials.append(t)
            print(
                f"α=0 m*={list(m_star)} r={r} M_H={t['M_H_steer']:+.2f} H={t['H']} dM={t['dM_H']:+.3f}",
                flush=True,
            )
        k += 1

    signed = [a for a in alphas if abs(a) > 1e-12]
    for arm in ("v_H", "random"):
        for a in signed:
            for m_star in SUPPRESS_MSTAR:
                for r in range(args.reps):
                    t = _run_one(
                        sc, loaded, bank,
                        m_star=m_star, seed=args.seed + 1000 * k + 17 * r,
                        alpha=a, arm=arm, v_H=v_H, v_rand=v_rand,
                    )
                    trials.append(t)
                    print(
                        f"{arm} α={a:+.1f} m*={list(m_star)} r={r} "
                        f"dM={t['dM_H']:+.2f} M={t['M_H_steer']:+.2f} "
                        f"H={t['H']} P0_run",
                        flush=True,
                    )
                k += 1

    # Opposite-sign control at +opp_alpha
    opp = float(args.opp_alpha)
    if abs(opp) > 1e-12:
        for m_star in SUPPRESS_MSTAR:
            for r in range(args.reps):
                t = _run_one(
                    sc, loaded, bank,
                    m_star=m_star, seed=args.seed + 5000 + 13 * r + k,
                    alpha=opp, arm="opp_v_H", v_H=v_H, v_rand=v_rand,
                )
                trials.append(t)
                print(
                    f"opp_v_H α={opp:+.1f} m*={list(m_star)} r={r} "
                    f"dM={t['dM_H']:+.2f} H={t['H']}",
                    flush=True,
                )
            k += 1

    # Primary endpoint pooled over suppress m*
    def _p_h0(arm: str, a: float | None = None) -> float:
        rows = [t for t in trials if t["arm"] == arm]
        if a is not None:
            rows = [t for t in rows if abs(t["alpha"] - a) < 1e-12]
        return float(np.mean([t["H_suppressed"] for t in rows])) if rows else float("nan")

    curve_vh = _curve(trials, alphas, "v_H")
    # include α=0 on v_H curve for plotting continuity
    zrows = [t for t in trials if t["arm"] == "alpha0"]
    if zrows:
        curve_vh = [
            {
                "alpha": 0.0,
                "n": len(zrows),
                "mean_dM_H": 0.0,
                "mean_M_H_steer": float(np.mean([t["M_H_steer"] for t in zrows])),
                "P_M_H_neg": float(np.mean([t["M_H_steer"] < 0 for t in zrows])),
                "mean_dq_H": 0.0,
                "P_H0": float(np.mean([t["H_suppressed"] for t in zrows])),
                "mean_O": float(np.mean([t["O"] for t in zrows])),
                "mean_E": float(np.mean([t["E_l1"] for t in zrows])),
            }
        ] + curve_vh
    curve_rand = _curve(trials, signed, "random")
    curve_opp = _curve(trials, [opp], "opp_v_H") if abs(opp) > 1e-12 else []
    diag = _diagnose(curve_vh, curve_rand)

    # Paired O given H flip vs not (on v_H arms only)
    vh_trials = [t for t in trials if t["arm"] == "v_H"]
    base_O = float(np.mean([t["O"] for t in zrows])) if zrows else 0.5
    flips = [t for t in vh_trials if t["H"] == 0]
    noflip = [t for t in vh_trials if t["H"] == 1]
    prop = {
        "P_H0_v_H": _p_h0("v_H"),
        "P_H0_random": _p_h0("random"),
        "P_H0_alpha0": _p_h0("alpha0"),
        "P_H0_opp": _p_h0("opp_v_H"),
        "mean_O_when_H0": float(np.mean([t["O"] for t in flips])) if flips else None,
        "mean_O_when_H1": float(np.mean([t["O"] for t in noflip])) if noflip else None,
        "n_H0": len(flips),
        "n_H1": len(noflip),
        "note": "Full H→O propagation test waits until reliable H suppression.",
    }

    report = {
        "protocol": "H tool-decision boundary (suppress m_H*=0 only)",
        "controller": "d = g_H(e_H) v_H  (C frozen out)",
        "M_H": "logit(<tool_call>) - logit(FINAL)",
        "primary_endpoint": "P(H'=0 | m_H*=0)",
        "alphas": alphas,
        "reps": args.reps,
        "m_star": [list(m) for m in SUPPRESS_MSTAR],
        "curves": {"v_H": curve_vh, "random": curve_rand, "opp_v_H": curve_opp},
        "diagnosis": diag,
        "propagation_preview": prop,
        "architecture": arch,
        "trials": trials,
        "not": "8way|C+H|generic_end_to_end_dose",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# H tool-decision boundary experiment",
        "",
        "> Controller frozen: `d=g_H(e_H)v_H`. C observe-only.",
        "> Population: m_H*=0 only. Primary: P(H'=0 | m_H*=0).",
        "> M_H = logit(<tool_call>) - logit(FINAL).",
        "",
        f"## Diagnosis: **{diag['outcome']}**",
        "",
        diag["detail"],
        "",
        f"- dM_H/dα = {diag['dM_H_d_alpha']}",
        f"- crosses zero M_H: {diag['crosses_zero_M_H']}",
        f"- max P(H=0) on v_H: {diag['max_P_H0']}",
        f"- random mean ΔM_H: {diag['rand_mean_dM_H']}",
        "",
        "## Dose curves (v_H)",
        "",
        "| α | n | ΔM_H | M_H | P(M_H<0) | Δq_H | P(H=0) | mean O | E |",
        "|---|---|------|-----|----------|------|--------|--------|---|",
    ]
    for p in curve_vh:
        lines.append(
            f"| {p['alpha']} | {p['n']} | {p['mean_dM_H']:+.3f} | {p['mean_M_H_steer']:+.2f} | "
            f"{p['P_M_H_neg']:.2f} | {p['mean_dq_H']:+.3f} | {p['P_H0']:.2f} | "
            f"{p['mean_O']:.2f} | {p['mean_E']:.2f} |"
        )
    lines += [
        "",
        "## Random control",
        "",
        "| α | n | ΔM_H | P(H=0) |",
        "|---|---|------|--------|",
    ]
    for p in curve_rand:
        lines.append(
            f"| {p['alpha']} | {p['n']} | {p['mean_dM_H']:+.3f} | {p['P_H0']:.2f} |"
        )
    if curve_opp:
        lines += ["", "## Opposite +v_H", "", "| α | n | ΔM_H | P(H=0) |", "|---|---|------|--------|"]
        for p in curve_opp:
            lines.append(
                f"| {p['alpha']} | {p['n']} | {p['mean_dM_H']:+.3f} | {p['P_H0']:.2f} |"
            )
    lines += [
        "",
        "## Primary endpoint summary",
        "",
        f"- P(H=0|α=0) = {prop['P_H0_alpha0']:.3f}",
        f"- P(H=0|v_H) = {prop['P_H0_v_H']:.3f}",
        f"- P(H=0|random) = {prop['P_H0_random']:.3f}",
        f"- P(H=0|opp) = {prop['P_H0_opp']:.3f}",
        "",
        "## Next (gated)",
        "",
        "- A → use min effective α; then H→O on same trajectories",
        "- B → relearn H direction at tool-selection site (do not keep raising α)",
        "- C → move intervention/measurement site",
        "- 8-way still blocked",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "md": str(MD), "diagnosis": diag, "prop": prop}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
