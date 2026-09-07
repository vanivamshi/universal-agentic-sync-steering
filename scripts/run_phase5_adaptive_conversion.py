#!/usr/bin/env python3
"""Phase 5 — Static vc(h0) vs adaptive vc(ht) sequential composition.

Question:
  Can causal conversion be made trajectory-valid by recomputing g_k(h_t)
  immediately before each intervention?

Arms (identical prompts/seeds/sites/α/closed-loop logic):
  static   — frozen V_c from Phase 4 (g at h0)
  adaptive — v_c^k(h_t) = convert(v_p^k, g_k(h_t)) with prior steers active
  predictive — bank v_p (control)
  random     — matched random (control)

Primary success: restore E0→E1→E2→E3 so C/O no longer systematically regress
(ΔE_C, ΔE_O ≥ 0 on average, or total ΔE clearly > static).

Do NOT open 8-way from a partial win.

  .venv/bin/python scripts/run_phase5_adaptive_conversion.py --reps 4
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
from scripts.sync_channel_margins import (  # noqa: E402
    SPECS,
    convert_direction,
    margin_at_site,
    margin_gradient,
    messages_for_channel,
    unit,
)
from scripts.sync_eq import extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import LAYER_DEFAULT, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase5_adaptive_conversion.json"
MD = ROOT / "data" / "results" / "sync_phase5_adaptive_conversion.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260906
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
SELECTED_MSTAR = (
    (0, 0, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)
PRIMARY_ORDER = ("C", "H", "O")


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _S_from_row(row: dict) -> list[int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    plan = extract_plan(blob)
    if plan:
        c = score_plan(plan, H)
        C = int(c) if c is not None else 0
    else:
        C = 0
    return [C, H, O]


def _e(mstar: tuple[int, int, int], S: list[int]) -> list[int]:
    return [int(mstar[i]) - int(S[i]) for i in range(3)]


def _policy_ok(S: list[int]) -> int:
    return int(S[1] == S[2])


def _compose_tool_dir(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in ("C", "H"):
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return None if d is None else d


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    tool_d = _compose_tool_dir(active)
    tool_hook = make_steer_hook(loaded, tool_d, alpha) if tool_d is not None else None
    o_hook = None
    if "O" in active:
        ek, vk = active["O"]
        o_hook = make_steer_hook(loaded, vk, alpha * float(ek))

    def hook_for_turn(phase: str, _turn: int):
        if phase == "tool":
            return tool_hook
        if phase == "report":
            return o_hook
        return None

    row = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
        capture_activations=False,
    )
    if tool_hook is not None:
        tool_hook.remove()
    if o_hook is not None:
        o_hook.remove()
    S = _S_from_row(row)
    return {
        "S": S,
        "task_ok": int(bool((row.get("final") or "").strip())),
        "policy_ok": _policy_ok(S),
    }


def _g_at_state(
    loaded,
    sc,
    k: str,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    alpha: float,
    layer: int,
) -> tuple[np.ndarray, float]:
    """g_k(h_t), M_k(h_t) with currently active prior steers applied at k's site."""
    msgs = messages_for_channel(sc, k)
    spec = SPECS[k]
    # Build a composite prior steer for the forward used to read g_k
    # Use all active channels' directions summed (state after prior interventions)
    prior_d = None
    for j, (ej, vj) in active.items():
        term = float(ej) * vj
        prior_d = term if prior_d is None else prior_d + term

    if prior_d is None:
        g = margin_gradient(loaded, msgs, spec, layer=layer, n=3)
        M = float(margin_at_site(loaded, msgs, spec)["M"])
        return g, M

    # Steered margin
    hook = make_steer_hook(loaded, prior_d, alpha)
    M = float(margin_at_site(loaded, msgs, spec, hook=hook)["M"])
    # Steered gradient: one-shot backprop with steer active (reuse phase4 diag approach)
    g = _grad_with_prior(loaded, msgs, spec, prior_d=prior_d, alpha=alpha, layer=layer)
    return g, M


def _grad_with_prior(loaded, messages, spec, *, prior_d: np.ndarray, alpha: float, layer: int) -> np.ndarray:
    from activation_pipeline.hooks import resolve_decoder_layers
    from scripts.sync_channel_margins import token_id
    from scripts.sync_h_decision import chat_prompt

    tok = loaded.tokenizer
    id_pos = token_id(tok, spec.pos_token)
    id_neg = token_id(tok, spec.neg_token)
    enc = tok(chat_prompt(tok, messages, spec.prefill), return_tensors="pt")
    device = next(loaded.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    steer = make_steer_hook(loaded, prior_d, alpha, layer=layer)
    dec = resolve_decoder_layers(loaded.model)
    captured: dict[str, torch.Tensor] = {}

    def make_hook():
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h_last = hidden[:, -1, :].detach().requires_grad_(True)
            captured["h"] = h_last
            h_new = hidden.clone()
            h_new[:, -1, :] = h_last
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        return hook

    handle = dec[layer].register_forward_hook(make_hook())
    if steer is not None:
        steer.register()
    try:
        loaded.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = loaded.model(**enc, use_cache=False)
            M = out.logits[0, -1][id_pos] - out.logits[0, -1][id_neg]
            M.backward()
        g = captured["h"].grad
        if g is None:
            raise RuntimeError("no grad under prior")
        return unit(g.detach().float().cpu().numpy().reshape(-1))
    finally:
        handle.remove()
        if steer is not None:
            steer.remove()


def _margin_recovery(
    loaded,
    sc,
    k: str,
    v: np.ndarray,
    e_k: int,
    *,
    active_prior: dict[str, tuple[int, np.ndarray]],
    alpha: float,
) -> dict[str, float]:
    """ΔM_k when applying e_k·v at site k given prior active steers."""
    msgs = messages_for_channel(sc, k)
    # baseline under prior
    prior_d = None
    for j, (ej, vj) in active_prior.items():
        term = float(ej) * vj
        prior_d = term if prior_d is None else prior_d + term
    if prior_d is None:
        M0 = float(margin_at_site(loaded, msgs, SPECS[k])["M"])
    else:
        hook0 = make_steer_hook(loaded, prior_d, alpha)
        M0 = float(margin_at_site(loaded, msgs, SPECS[k], hook=hook0)["M"])
    # after adding current
    if prior_d is None:
        d = float(e_k) * v
    else:
        d = prior_d + float(e_k) * v
    hook1 = make_steer_hook(loaded, d, alpha)
    M1 = float(margin_at_site(loaded, msgs, SPECS[k], hook=hook1)["M"])
    dM = M1 - M0
    # intended: e_k > 0 should increase M; e_k < 0 decrease
    agree = float(np.sign(dM) == np.sign(e_k)) if abs(dM) > 0.05 else float("nan")
    return {"M0": M0, "M1": M1, "dM": dM, "sign_agree": agree}


def run_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    mode: str,
    v_static: dict[str, np.ndarray],
    v_p: dict[str, np.ndarray],
    alpha: float,
    seed: int,
    layer: int,
) -> dict[str, Any]:
    base = _run_episode(sc, loaded, active={}, alpha=alpha, seed=seed)
    S = base["S"]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    traj_S = [list(S)]
    active: dict[str, tuple[int, np.ndarray]] = {}
    stages = []
    cos_adapt_static: list[float] = []

    for stage_i, k in enumerate(PRIMARY_ORDER):
        e = _e(mstar, S)
        ek = int(e[CH_IDX[k]])
        if ek != 0 and k not in active:
            if mode == "static":
                vk = v_static[k]
            elif mode == "adaptive":
                g_t, M_t = _g_at_state(
                    loaded, sc, k, active=active, alpha=alpha, layer=layer
                )
                vk = convert_direction(v_p[k], g_t)
                cos_adapt_static.append(float(np.dot(vk, v_static[k])))
                # store M_t for logging
            elif mode == "predictive":
                vk = v_p[k]
            elif mode == "random":
                vk = v_static[k]  # placeholder replaced below
            else:
                raise ValueError(mode)
            if mode == "random":
                rng = np.random.default_rng(seed + 17 * stage_i + 3)
                vk = unit(rng.standard_normal(v_static[k].shape[0]))

            # margin recovery diagnostic before locking
            rec = _margin_recovery(
                loaded, sc, k, vk, ek, active_prior=dict(active), alpha=alpha
            )
            active[k] = (ek, vk)
            stage_meta = {"stage": k, "ek": ek, "mode": mode, **rec}
            if mode == "adaptive" and cos_adapt_static:
                stage_meta["cos_vc_ht_vs_h0"] = cos_adapt_static[-1]
        else:
            stage_meta = {"stage": k, "ek": ek, "skipped": True}

        out = _run_episode(
            sc, loaded, active=active, alpha=alpha, seed=seed + 100 * (stage_i + 1)
        )
        S = out["S"]
        E = float(sync_error_norm(_e(mstar, S)))
        stage_meta.update({"S": list(S), "E": E, "task_ok": out["task_ok"], "policy_ok": out["policy_ok"]})
        stages.append(stage_meta)
        traj_E.append(E)
        traj_S.append(list(S))

    E0, E1, E2, E3 = traj_E
    return {
        "mstar": list(mstar),
        "mode": mode,
        "S0": traj_S[0],
        "S_final": traj_S[-1],
        "E_traj": traj_E,
        "delta_E_C": float(E0 - E1),
        "delta_E_H": float(E1 - E2),
        "delta_E_O": float(E2 - E3),
        "delta_E_total": float(E0 - E3),
        "hit": int(traj_S[-1] == list(mstar)),
        "bit_C": int(traj_S[-1][0] == mstar[0]),
        "bit_H": int(traj_S[-1][1] == mstar[1]),
        "bit_O": int(traj_S[-1][2] == mstar[2]),
        "task_ok_final": stages[-1].get("task_ok", 1),
        "policy_ok_final": stages[-1].get("policy_ok", 0),
        "mean_cos_adapt_static": float(np.mean(cos_adapt_static)) if cos_adapt_static else float("nan"),
        "stages": stages,
    }


def _agg(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials]))

    # margin sign agreement where present
    agrees = []
    for t in trials:
        for st in t["stages"]:
            if "sign_agree" in st and np.isfinite(st["sign_agree"]):
                agrees.append(st["sign_agree"])
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_C": m("bit_C"),
        "P_H": m("bit_H"),
        "P_O": m("bit_O"),
        "mean_E0": float(np.mean([t["E_traj"][0] for t in trials])),
        "mean_E_final": float(np.mean([t["E_traj"][-1] for t in trials])),
        "mean_E_traj": [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)],
        "mean_delta_E_total": m("delta_E_total"),
        "mean_delta_E_C": m("delta_E_C"),
        "mean_delta_E_H": m("delta_E_H"),
        "mean_delta_E_O": m("delta_E_O"),
        "P_task_ok": m("task_ok_final"),
        "P_policy_ok": m("policy_ok_final"),
        "mean_margin_sign_agree": float(np.mean(agrees)) if agrees else float("nan"),
        "C_no_regression": bool(m("delta_E_C") >= -0.05),
        "O_no_regression": bool(m("delta_E_O") >= -0.05),
        "H_helps": bool(m("delta_E_H") > 0.05),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=ALPHA)
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
    bank = ChannelBank.load(CHANNEL_V)
    blob = json.loads(VC_PATH.read_text())
    Vc = np.asarray(blob["V_c"], dtype=np.float64)
    v_static = {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}
    v_p = {ch: unit(bank.V[i]) for i, ch in enumerate(CHANNELS)}

    modes = ("static", "adaptive", "predictive", "random")
    results: dict[str, Any] = {}
    for mode in modes:
        print(f"=== mode {mode} ===", flush=True)
        trials = []
        for mi, mstar in enumerate(SELECTED_MSTAR):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r
                print(f"  {mode} m*={mstar} r={r}", flush=True)
                torch.manual_seed(seed)
                tr = run_trial(
                    sc,
                    loaded,
                    mstar=mstar,
                    mode=mode,
                    v_static=v_static,
                    v_p=v_p,
                    alpha=args.alpha,
                    seed=seed,
                    layer=args.layer,
                )
                trials.append(tr)
                print(
                    f"    E:{tr['E_traj']} dE={tr['delta_E_total']:+.2f} hit={tr['hit']}",
                    flush=True,
                )
        results[mode] = {"agg": _agg(trials), "trials": trials}

    sa = results["static"]["agg"]
    aa = results["adaptive"]["agg"]
    gate = {
        "question": "Does recomputing g_k(h_t) fix sequential sync vs static vc(h0)?",
        "adaptive_beats_static_delta_E": bool(
            aa["mean_delta_E_total"] > sa["mean_delta_E_total"] + 0.05
        ),
        "adaptive_beats_static_hit": bool(aa["P_hit"] >= sa["P_hit"]),
        "C_regression_fixed": bool(
            aa["mean_delta_E_C"] >= sa["mean_delta_E_C"] and aa["C_no_regression"]
        ),
        "O_regression_fixed": bool(
            aa["mean_delta_E_O"] >= sa["mean_delta_E_O"] and aa["O_no_regression"]
        ),
        "phase4c_failure_restored": False,  # set below
        "ready_for_8way": False,
    }
    gate["phase4c_failure_restored"] = bool(
        gate["adaptive_beats_static_delta_E"]
        and aa["mean_delta_E_total"] > 0.15
        and (aa["C_no_regression"] or aa["mean_delta_E_C"] > sa["mean_delta_E_C"] + 0.1)
        and (aa["O_no_regression"] or aa["mean_delta_E_O"] > sa["mean_delta_E_O"] + 0.1)
    )
    # 8-way only if failure clearly restored — still not automatic
    gate["ready_for_8way"] = False
    gate["note"] = "8-way stays closed until Phase 5 restores C/O trajectory explicitly"

    payload = {
        "protocol": "Phase 5 static vs adaptive conversion",
        "alpha": args.alpha,
        "reps": args.reps,
        "mstar_set": [list(m) for m in SELECTED_MSTAR],
        "results": {m: {"agg": results[m]["agg"]} for m in modes},
        "trials": {m: results[m]["trials"] for m in modes},
        "gate": gate,
        "phase4_frozen": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 5 — Static \(v_c(h_0)\) vs adaptive \(v_c(h_t)\)",
        "",
        r"> **Question:** Can causal conversion be made trajectory-valid by recomputing \(g_k(h_t)\)?",
        "",
        f"m*={list(SELECTED_MSTAR)} · reps={args.reps} · α={args.alpha} · order C→H→O",
        "",
        "## Sequential sync",
        "",
        "| Arm | P(hit) | ΔE tot | E0→E3 | ΔE_C | ΔE_H | ΔE_O | margin agree |",
        "|-----|--------|--------|-------|------|------|------|--------------|",
    ]
    for mode in modes:
        a = results[mode]["agg"]
        et = a["mean_E_traj"]
        lines.append(
            f"| {mode} | {a['P_hit']:.2f} | {a['mean_delta_E_total']:+.2f} | "
            f"{et[0]:.2f}→{et[1]:.2f}→{et[2]:.2f}→{et[3]:.2f} | "
            f"{a['mean_delta_E_C']:+.2f} | {a['mean_delta_E_H']:+.2f} | {a['mean_delta_E_O']:+.2f} | "
            f"{a['mean_margin_sign_agree']:.2f} |"
        )

    lines += [
        "",
        "## Gate (restore Phase 4C failure before 8-way)",
        "",
        f"- Adaptive ΔE > static: **{gate['adaptive_beats_static_delta_E']}**",
        f"- Adaptive hit ≥ static: **{gate['adaptive_beats_static_hit']}**",
        f"- C regression improved/fixed: **{gate['C_regression_fixed']}**",
        f"- O regression improved/fixed: **{gate['O_regression_fixed']}**",
        f"- Phase 4C failure restored: **{gate['phase4c_failure_restored']}**",
        f"- Ready for 8-way: **False**",
        "",
        "Phase 4 remains frozen. 8-way stays closed until C/O sequential regressions are fixed.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "aggs": {m: results[m]["agg"] for m in modes}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
