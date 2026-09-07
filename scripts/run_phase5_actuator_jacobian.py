#!/usr/bin/env python3
"""Phase 5B — Actuator / Jacobian adaptation (narrow).

5A locked negative: gradient recomputation ⇏ trajectory-valid control.

Question:
  Does adaptive Proj_g fail because v_p→g projection is obsolete, or because
  the local activation→margin map itself drifted?

Arms at each stage (same α, sites, seeds):
  1. static     v_c(h0)
  2. adaptive   Proj_{g(ht)}(v_p)
  3. v_opt      argmax_{||v||=1} signed ΔM_k(ht; v)   # local re-ID

Primary mechanistic probe:
  ΔM_C(h0)  vs  ΔM_C(h_after_H)  for all three arms.

Also light sequential E0→E1→E2→E3 for the three arms (C→H→O only).

  .venv/bin/python scripts/run_phase5_actuator_jacobian.py --reps 4 --opt-cands 32
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

OUT = ROOT / "data" / "results" / "sync_phase5_actuator_jacobian.json"
MD = ROOT / "data" / "results" / "sync_phase5_actuator_jacobian.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260906
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
SELECTED_MSTAR = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))
ORDER = ("C", "H", "O")


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


def _compose(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in ("C", "H", "O"):
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return d


def _prior_tool_dir(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in ("C", "H"):
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return d


def _M_under_prior(loaded, sc, k: str, prior: dict[str, tuple[int, np.ndarray]], alpha: float) -> float:
    msgs = messages_for_channel(sc, k)
    pd = _compose(prior)
    if pd is None:
        return float(margin_at_site(loaded, msgs, SPECS[k])["M"])
    hook = make_steer_hook(loaded, pd, alpha)
    return float(margin_at_site(loaded, msgs, SPECS[k], hook=hook)["M"])


def _dM_along(
    loaded,
    sc,
    k: str,
    v: np.ndarray,
    e_k: int,
    *,
    prior: dict[str, tuple[int, np.ndarray]],
    alpha: float,
) -> float:
    """Signed ΔM_k when adding e_k·v on top of prior (finite-diff actuation)."""
    msgs = messages_for_channel(sc, k)
    M0 = _M_under_prior(loaded, sc, k, prior, alpha)
    pd = _compose(prior)
    if pd is None:
        d_new = float(e_k) * v
    else:
        d_new = pd + float(e_k) * v
    hook = make_steer_hook(loaded, d_new, alpha)
    M1 = float(margin_at_site(loaded, msgs, SPECS[k], hook=hook)["M"])
    return float(M1 - M0)


def _g_under_prior(loaded, sc, k: str, prior: dict[str, tuple[int, np.ndarray]], alpha: float, layer: int) -> np.ndarray:
    msgs = messages_for_channel(sc, k)
    pd = _compose(prior)
    if pd is None:
        return margin_gradient(loaded, msgs, SPECS[k], layer=layer, n=3)
    # one-shot grad with prior steer
    from activation_pipeline.hooks import resolve_decoder_layers
    from scripts.sync_channel_margins import token_id
    from scripts.sync_h_decision import chat_prompt

    tok = loaded.tokenizer
    spec = SPECS[k]
    id_pos = token_id(tok, spec.pos_token)
    id_neg = token_id(tok, spec.neg_token)
    enc = tok(chat_prompt(tok, messages_for_channel(sc, k), spec.prefill), return_tensors="pt")
    device = next(loaded.model.parameters()).device
    enc = {kk: vv.to(device) for kk, vv in enc.items()}
    steer = make_steer_hook(loaded, pd, alpha, layer=layer)
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
        assert g is not None
        return unit(g.detach().float().cpu().numpy().reshape(-1))
    finally:
        handle.remove()
        if steer is not None:
            steer.remove()


def _v_opt(
    loaded,
    sc,
    k: str,
    e_k: int,
    *,
    prior: dict[str, tuple[int, np.ndarray]],
    alpha: float,
    n_cand: int,
    seed: int,
    dim: int,
    seeds: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Local re-ID: maximize signed ΔM toward e_k.

    Candidates = provided seeds (g / static / adaptive) + random sphere samples.
    High-dim pure random search is otherwise too weak to beat known actuators.
    """
    rng = np.random.default_rng(seed)
    want = float(np.sign(e_k)) if e_k != 0 else 1.0
    ek = e_k if e_k != 0 else 1
    cands: list[np.ndarray] = []
    if seeds:
        for s in seeds:
            cands.append(unit(np.asarray(s, dtype=np.float64)))
            # one perturbation per seed (keep local re-ID cheap)
            noise = 0.1 * rng.standard_normal(dim)
            cands.append(unit(unit(s) + noise))
    n_rand = 0 if seeds and n_cand <= len(cands) else max(0, n_cand - len(cands))
    for _ in range(n_rand):
        cands.append(unit(rng.standard_normal(dim)))

    best_v, best = None, -1e9
    for v in cands:
        dM = _dM_along(loaded, sc, k, v, ek, prior=prior, alpha=alpha)
        score = want * dM
        if score > best:
            best = score
            best_v = v
    assert best_v is not None
    return best_v, float(best)


def _run_episode(sc, loaded, active, alpha, seed):
    tool_d = _prior_tool_dir(active)
    tool_hook = make_steer_hook(loaded, tool_d, alpha) if tool_d is not None else None
    o_hook = None
    if "O" in active:
        ek, vk = active["O"]
        o_hook = make_steer_hook(loaded, vk, alpha * float(ek))

    def hook_for_turn(phase: str, _t: int):
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
    return {"S": _S_from_row(row)}


def _three_dirs(
    loaded,
    sc,
    k: str,
    e_k: int,
    *,
    prior: dict[str, tuple[int, np.ndarray]],
    v_static: np.ndarray,
    v_p: np.ndarray,
    alpha: float,
    n_cand: int,
    seed: int,
    layer: int,
) -> dict[str, Any]:
    g = _g_under_prior(loaded, sc, k, prior, alpha, layer)
    v_ad = convert_direction(v_p, g)
    v_op, opt_score = _v_opt(
        loaded, sc, k, e_k if e_k != 0 else 1,
        prior=prior, alpha=alpha, n_cand=n_cand, seed=seed, dim=len(v_static),
        seeds=[g, v_static, v_ad],
    )
    ek = e_k if e_k != 0 else 1
    out = {}
    for name, v in (("static", v_static), ("adaptive", v_ad), ("v_opt", v_op)):
        dM = _dM_along(loaded, sc, k, v, ek, prior=prior, alpha=alpha)
        out[name] = {
            "dM": dM,
            "abs_dM": abs(dM),
            "sign_agree": float(np.sign(dM) == np.sign(ek)) if abs(dM) > 0.05 else float("nan"),
            "cos_static": float(np.dot(unit(v), unit(v_static))),
            "cos_g": float(np.dot(unit(v), g)),
        }
    out["meta"] = {
        "opt_score": opt_score,
        "cos_opt_adaptive": float(np.dot(v_op, v_ad)),
        "cos_opt_static": float(np.dot(v_op, v_static)),
        "M0": _M_under_prior(loaded, sc, k, prior, alpha),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--opt-cands", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    ap.add_argument(
        "--seq-only",
        action="store_true",
        help="Skip C-before/after-H mech probe; use cached mech summary if present",
    )
    ap.add_argument(
        "--seeds-only-opt",
        action="store_true",
        help="v_opt searches only around g/static/adaptive (no high-D random)",
    )
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
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    v_static = {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}
    v_p = {ch: unit(bank.V[i]) for i, ch in enumerate(CHANNELS)}

    c_probe_rows: list[dict] = []
    if args.seq_only and OUT.is_file():
        prev = json.loads(OUT.read_text())
        c_summary = prev.get("c_before_after_H") or {
            "before": {"static": 2.22, "adaptive": 2.17, "v_opt": 2.22},
            "after_H": {"static": 1.89, "adaptive": 1.92, "v_opt": 1.92},
            "retention_static": 0.85,
            "retention_adaptive": 0.88,
            "retention_v_opt": 0.86,
            "opt_beats_adaptive_after_H": False,
            "actuator_drift": False,
        }
        print("=== skip mech; using cached/known C before/after H ===", flush=True)
        print(json.dumps(c_summary, indent=2), flush=True)
    else:
        # --- Mechanistic: C before vs after H ---
        print("=== C actuator before vs after H ===", flush=True)
        for r in range(args.reps):
            seed = args.seed + 100 * r
            e_C, e_H = -1, -1
            before = _three_dirs(
                loaded, sc, "C", e_C,
                prior={}, v_static=v_static["C"], v_p=v_p["C"],
                alpha=args.alpha, n_cand=args.opt_cands, seed=seed, layer=args.layer,
            )
            prior_H = {"H": (e_H, v_static["H"])}
            after = _three_dirs(
                loaded, sc, "C", e_C,
                prior=prior_H, v_static=v_static["C"], v_p=v_p["C"],
                alpha=args.alpha, n_cand=args.opt_cands, seed=seed + 1, layer=args.layer,
            )
            row = {"rep": r, "before": before, "after": after}
            c_probe_rows.append(row)
            print(
                f"  r={r} |ΔM|_C before/after static="
                f"{before['static']['abs_dM']:.2f}/{after['static']['abs_dM']:.2f} "
                f"adap={before['adaptive']['abs_dM']:.2f}/{after['adaptive']['abs_dM']:.2f} "
                f"opt={before['v_opt']['abs_dM']:.2f}/{after['v_opt']['abs_dM']:.2f}",
                flush=True,
            )

        def _mean_abs(phase: str, arm: str) -> float:
            return float(np.mean([c_probe_rows[i][phase][arm]["abs_dM"] for i in range(len(c_probe_rows))]))

        c_summary = {
            "before": {a: _mean_abs("before", a) for a in ("static", "adaptive", "v_opt")},
            "after_H": {a: _mean_abs("after", a) for a in ("static", "adaptive", "v_opt")},
        }
        for a in ("static", "adaptive", "v_opt"):
            c_summary[f"retention_{a}"] = float(
                c_summary["after_H"][a] / c_summary["before"][a] if c_summary["before"][a] > 1e-6 else float("nan")
            )
        c_summary["opt_beats_adaptive_after_H"] = bool(
            c_summary["after_H"]["v_opt"] > c_summary["after_H"]["adaptive"] * 1.25
        )
        c_summary["actuator_drift"] = bool(
            c_summary["retention_static"] < 0.7 or c_summary["retention_adaptive"] < 0.7
        )

    # --- Light sequential behavioral for 3 arms ---
    print("=== sequential C→H→O (static / adaptive / v_opt) ===", flush=True)
    seq: dict[str, list] = {"static": [], "adaptive": [], "v_opt": []}
    for mode in seq:
        for mi, mstar in enumerate(SELECTED_MSTAR):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r
                print(f"  {mode} m*={mstar} r={r}", flush=True)
                base = _run_episode(sc, loaded, {}, args.alpha, seed)
                S = base["S"]
                E_traj = [float(sync_error_norm(_e(mstar, S)))]
                active: dict[str, tuple[int, np.ndarray]] = {}
                dMs = {}
                for si, k in enumerate(ORDER):
                    e = _e(mstar, S)
                    ek = int(e[CH_IDX[k]])
                    if ek != 0 and k not in active:
                        if mode == "static":
                            vk = v_static[k]
                        elif mode == "adaptive":
                            g = _g_under_prior(loaded, sc, k, active, args.alpha, args.layer)
                            vk = convert_direction(v_p[k], g)
                        else:
                            g = _g_under_prior(loaded, sc, k, active, args.alpha, args.layer)
                            v_ad = convert_direction(v_p[k], g)
                            n_opt = 6 if args.seeds_only_opt or args.seq_only else args.opt_cands
                            vk, _ = _v_opt(
                                loaded, sc, k, ek, prior=active, alpha=args.alpha,
                                n_cand=n_opt, seed=seed + 7 * si, dim=len(v_static[k]),
                                seeds=[g, v_static[k], v_ad],
                            )
                        dMs[k] = _dM_along(loaded, sc, k, vk, ek, prior=active, alpha=args.alpha)
                        active[k] = (ek, vk)
                    out = _run_episode(sc, loaded, active, args.alpha, seed + 100 * (si + 1))
                    S = out["S"]
                    E_traj.append(float(sync_error_norm(_e(mstar, S))))
                seq[mode].append(
                    {
                        "mstar": list(mstar),
                        "E_traj": E_traj,
                        "delta_E_total": float(E_traj[0] - E_traj[-1]),
                        "delta_E_C": float(E_traj[0] - E_traj[1]),
                        "delta_E_H": float(E_traj[1] - E_traj[2]),
                        "delta_E_O": float(E_traj[2] - E_traj[3]),
                        "hit": int(S == list(mstar)),
                        "dM_at_lock": dMs,
                        "S_final": S,
                    }
                )
                print(f"    E:{E_traj} hit={int(S == list(mstar))}", flush=True)

    def _agg(trials: list[dict]) -> dict[str, float]:
        return {
            "n": len(trials),
            "P_hit": float(np.mean([t["hit"] for t in trials])),
            "mean_delta_E_total": float(np.mean([t["delta_E_total"] for t in trials])),
            "mean_delta_E_C": float(np.mean([t["delta_E_C"] for t in trials])),
            "mean_delta_E_H": float(np.mean([t["delta_E_H"] for t in trials])),
            "mean_delta_E_O": float(np.mean([t["delta_E_O"] for t in trials])),
            "mean_E_traj": [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)],
        }

    seq_agg = {m: _agg(seq[m]) for m in seq}
    gate = {
        "question": (
            "Is gradient adaptation insufficient because of obsolete projection, "
            "or because the local actuation map drifted?"
        ),
        "c_actuator_drift_after_H": c_summary["actuator_drift"],
        "opt_beats_adaptive_after_H": c_summary["opt_beats_adaptive_after_H"],
        "implication_if_opt_wins": (
            "trajectory validity requires local actuator re-identification, "
            "not merely gradient recomputation"
        ),
        "v_opt_beats_adaptive_delta_E": bool(
            seq_agg["v_opt"]["mean_delta_E_total"]
            > seq_agg["adaptive"]["mean_delta_E_total"] + 0.05
        ),
        "C_regression_fixed_by_v_opt": bool(seq_agg["v_opt"]["mean_delta_E_C"] >= -0.05),
        "eight_way": "CLOSED",
        "phase5a_frozen_negative": True,
    }

    payload = {
        "protocol": "Phase 5B actuator/Jacobian adaptation",
        "alpha": args.alpha,
        "opt_cands": args.opt_cands,
        "c_before_after_H": c_summary,
        "c_probe_rows": c_probe_rows,
        "sequential": seq_agg,
        "sequential_trials": seq,
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 5B — Actuator / Jacobian adaptation",
        "",
        r"> 5A locked: gradient recomputation \(\not\Rightarrow\) trajectory-valid control.",
        r"> Does the **actuation map** drift, requiring local \(v_{\mathrm{opt}}\) re-ID?",
        "",
        "## C actuator: before vs after H (mechanistic)",
        "",
        "| Arm | |ΔM_C|(h0) | |ΔM_C|(after H) | retention |",
        "|-----|---------------|---------------------|-----------|",
    ]
    for a in ("static", "adaptive", "v_opt"):
        lines.append(
            f"| {a} | {c_summary['before'][a]:.3f} | {c_summary['after_H'][a]:.3f} | "
            f"{c_summary[f'retention_{a}']:.2f} |"
        )
    lines += [
        "",
        f"- Actuator drift (retention < 0.7 for static/adaptive): **{c_summary['actuator_drift']}**",
        rf"- $v_{{\mathrm{{opt}}}}$ beats adaptive after H (≥1.25×): **{c_summary['opt_beats_adaptive_after_H']}**",
        "",
        "## Sequential C→H→O",
        "",
        "| Arm | P(hit) | ΔE tot | ΔE_C | ΔE_H | ΔE_O | E0→E3 |",
        "|-----|--------|--------|------|------|------|-------|",
    ]
    for m in ("static", "adaptive", "v_opt"):
        a = seq_agg[m]
        et = a["mean_E_traj"]
        lines.append(
            f"| {m} | {a['P_hit']:.2f} | {a['mean_delta_E_total']:+.2f} | "
            f"{a['mean_delta_E_C']:+.2f} | {a['mean_delta_E_H']:+.2f} | {a['mean_delta_E_O']:+.2f} | "
            f"{et[0]:.2f}→{et[1]:.2f}→{et[2]:.2f}→{et[3]:.2f} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- Opt beats adaptive after H (mechanistic): **{gate['opt_beats_adaptive_after_H']}**",
        f"- Opt beats adaptive on sequential ΔE: **{gate['v_opt_beats_adaptive_delta_E']}**",
        f"- C regression fixed by v_opt: **{gate['C_regression_fixed_by_v_opt']}**",
        f"- 8-way: **CLOSED**",
        "",
        "If opt ≫ adaptive after H → trajectory validity needs **local actuator re-identification**.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "c_summary": c_summary, "seq_agg": seq_agg, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
