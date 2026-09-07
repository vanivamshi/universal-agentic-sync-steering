#!/usr/bin/env python3
"""Phase 8A — 8-way validation with live-boundary objective (frozen v_c).

Replaces template M_k with Phase-7 episode-conditioned scores B_k = w_kᵀ h + b_k.

  s_k = 2 m*_k − 1
  L_live = Σ_k softplus(−β s_k B_k)
  G_k    = s_k (B_after − B_before)   # >0 ⇒ moved live boundary toward target

Arms (same protocol as prior 8-way): none / predictive / converted / random.
Primary behavioral: ΔE, P(hit). Mechanistic: E[−ΔL_live], E[G_k], P(W→C).

  .venv/bin/python scripts/run_phase8_live_8way.py --fast --reps 2
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
from scripts.sync_channel_margins import SPECS, margin_at_site, unit  # noqa: E402
from scripts.sync_eq import all_m_star_masks, extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import decision_at_site, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8_live_8way.json"
MD = ROOT / "data" / "results" / "sync_phase8_live_8way.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260913
ALPHA = 1.5
BETA = 1.0
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ORDER = ("C", "H", "O")
ARMS = ("none", "predictive", "converted", "random")


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_banks(seed: int) -> dict[str, dict[str, np.ndarray] | None]:
    bank = ChannelBank.load(CHANNEL_V)
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
        "none": None,
        "converted": {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)},
        "predictive": {ch: unit(bank.V[i]) for i, ch in enumerate(CHANNELS)},
        "random": {ch: unit(rng.standard_normal(Vc.shape[1])) for ch in CHANNELS},
    }


def _load_boundaries() -> dict[str, dict[str, Any]]:
    if not P7_PATH.is_file():
        raise SystemExit(f"missing Phase 7 boundaries: {P7_PATH}")
    blob = json.loads(P7_PATH.read_text())
    out = {}
    for k in CHANNELS:
        b = blob["boundaries"][k]
        if not b.get("ok") or b.get("w") is None:
            raise SystemExit(f"Phase 7 boundary for {k} not ok")
        out[k] = {
            "w": np.asarray(b["w"], dtype=np.float64),
            "b": float(b["b"]),
            "auc": float(b.get("auc_heldout", float("nan"))),
        }
    return out


def softplus(x: float) -> float:
    x = float(x)
    if x > 20:
        return x
    if x < -20:
        return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


def L_live(B: dict[str, float], mstar: tuple[int, int, int], beta: float) -> float:
    total = 0.0
    for i, k in enumerate(CHANNELS):
        if k not in B or B[k] is None or (isinstance(B[k], float) and np.isnan(B[k])):
            continue
        s = 2 * int(mstar[i]) - 1
        total += softplus(-float(beta) * s * float(B[k]))
    return float(total)


def _base_messages(sc, *, cls: str = NEUTRAL_CLS, task_id: str = NEUTRAL_TASK) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def _S_from_row(row: dict, messages: list[dict] | None = None) -> list[int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    if messages is not None:
        asst = [str(m.get("content") or "") for m in messages if m.get("role") == "assistant"]
        blob = (row.get("final") or "") + "\n" + "\n\n".join(asst)
    else:
        blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    plan = extract_plan(blob)
    if plan:
        c = score_plan(plan, H)
        C = int(c) if c is not None else 0
    else:
        C = 0
    return [C, H, O]


def _e(mstar, S):
    return [int(mstar[i]) - int(S[i]) for i in range(3)]


def capture_live_B(
    loaded,
    sc,
    *,
    state: dict,
    boundaries: dict[str, dict[str, Any]],
    cls: str = NEUTRAL_CLS,
    task_id: str = NEUTRAL_TASK,
) -> dict[str, Any]:
    """Episode-conditioned B_k = wᵀh+b at live decision sites."""
    base = _base_messages(sc, cls=cls, task_id=task_id)
    msgs = state["messages_snapshot"]
    asst = [str(m.get("content") or "") for m in msgs if m.get("role") == "assistant"]
    plan = extract_plan("\n\n".join(asst)) or ""

    out_B: dict[str, float] = {}
    out_M: dict[str, float] = {}
    out_ok: dict[str, bool] = {}

    sC = margin_at_site(loaded, base, SPECS["C"], capture_h=True)
    hC = np.asarray(sC["h"], dtype=np.float64)
    out_B["C"] = B_score(hC, boundaries["C"]["w"], boundaries["C"]["b"])
    out_M["C"] = float(sC["M"])
    out_ok["C"] = True

    if plan.strip():
        plan_pf = plan if plan.endswith("\n") else plan + "\n"
        sH = decision_at_site(loaded, base, assistant_prefill=plan_pf, capture_h=True)
        hH = np.asarray(sH["h"], dtype=np.float64)
        out_B["H"] = B_score(hH, boundaries["H"]["w"], boundaries["H"]["b"])
        out_M["H"] = float(sH["M_H"])
        out_ok["H"] = True
    else:
        out_B["H"] = float("nan")
        out_M["H"] = float("nan")
        out_ok["H"] = False

    sO = margin_at_site(loaded, msgs, SPECS["O"], capture_h=True)
    hO = np.asarray(sO["h"], dtype=np.float64)
    out_B["O"] = B_score(hO, boundaries["O"]["w"], boundaries["O"]["b"])
    out_M["O"] = float(sO["M"])
    out_ok["O"] = True

    return {"B": out_B, "M": out_M, "ok": out_ok, "plan": plan[:200]}


def _compose_tool(active: dict[str, tuple[int, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in ("C", "H"):
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return d


def _run_episode_state(sc, loaded, *, active: dict, alpha: float, seed: int):
    """Run tools+finalize; return (S, state, row)."""
    tool_d = _compose_tool(active)
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

    state = sc.run_tools_phase(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
    )
    row = sc.finalize_episode(loaded, state, hook_for_turn=hook_for_turn)
    if tool_hook is not None:
        tool_hook.remove()
    if o_hook is not None:
        o_hook.remove()
    S = _S_from_row(row, state["messages_snapshot"])
    return S, state, row


def _active_from_error(mstar, S, dirs, arm):
    active: dict[str, tuple[int, np.ndarray]] = {}
    if arm == "none" or dirs is None:
        return active, 0
    e = _e(mstar, S)
    for k in CHANNELS:
        if e[CH_IDX[k]] != 0:
            active[k] = (int(e[CH_IDX[k]]), dirs[k])
    return active, len(active)


def run_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray] | None,
    boundaries: dict,
    alpha: float,
    beta: float,
    seed: int,
    arm: str,
    fast: bool,
) -> dict[str, Any]:
    # free-run
    S0, state0, _ = _run_episode_state(sc, loaded, active={}, alpha=alpha, seed=seed)
    live0 = capture_live_B(loaded, sc, state=state0, boundaries=boundaries)
    B0 = live0["B"]
    L0 = L_live(B0, mstar, beta)
    E0 = float(sync_error_norm(_e(mstar, S0)))

    active, n_interv = _active_from_error(mstar, S0, dirs, arm)

    if arm == "none" or not active:
        S1, state1 = S0, state0
        if fast:
            # reobserve once without steers for fair ΔE noise baseline
            S1, state1, _ = _run_episode_state(
                sc, loaded, active={}, alpha=alpha, seed=seed + 100
            )
        live1 = capture_live_B(loaded, sc, state=state1, boundaries=boundaries)
    else:
        if fast:
            S1, state1, _ = _run_episode_state(
                sc, loaded, active=active, alpha=alpha, seed=seed + 100
            )
            live1 = capture_live_B(loaded, sc, state=state1, boundaries=boundaries)
        else:
            # sequential C→H→O cumulative (slower)
            S = list(S0)
            state = state0
            active_seq: dict[str, tuple[int, np.ndarray]] = {}
            for si, k in enumerate(ORDER):
                e = _e(mstar, S)
                if e[CH_IDX[k]] != 0 and k not in active_seq and dirs is not None:
                    active_seq[k] = (int(e[CH_IDX[k]]), dirs[k])
                S, state, _ = _run_episode_state(
                    sc, loaded, active=active_seq, alpha=alpha, seed=seed + 100 * (si + 1)
                )
            S1, state1 = S, state
            live1 = capture_live_B(loaded, sc, state=state1, boundaries=boundaries)
            n_interv = len(active_seq)

    B1 = live1["B"]
    L1 = L_live(B1, mstar, beta)
    E1 = float(sync_error_norm(_e(mstar, S1)))

    G = {}
    for i, k in enumerate(CHANNELS):
        s = 2 * int(mstar[i]) - 1
        if live0["ok"].get(k) and live1["ok"].get(k):
            G[k] = float(s * (B1[k] - B0[k]))
        else:
            G[k] = float("nan")

    # bit transitions
    cls = {}
    for i, k in enumerate(CHANNELS):
        w0 = S0[i] != mstar[i]
        c1 = S1[i] == mstar[i]
        if w0 and c1:
            cls[k] = "W2C"
        elif (not w0) and c1:
            cls[k] = "C2C"
        elif w0 and (not c1):
            cls[k] = "W2W"
        else:
            cls[k] = "C2W"

    return {
        "arm": arm,
        "mstar": list(mstar),
        "S0": list(S0),
        "S1": list(S1),
        "E0": E0,
        "E1": E1,
        "delta_E": float(E0 - E1),
        "hit": int(list(S1) == list(mstar)),
        "hit0": int(list(S0) == list(mstar)),
        "B0": B0,
        "B1": B1,
        "L0": L0,
        "L1": L1,
        "delta_L_live": float(L1 - L0),
        "neg_delta_L_live": float(-(L1 - L0)),
        "G": G,
        "cls": cls,
        "n_interventions": n_interv,
        "active": {k: int(v[0]) for k, v in active.items()},
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _agg(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    neg_dL = [t["neg_delta_L_live"] for t in trials]
    dE = [t["delta_E"] for t in trials]
    out: dict[str, Any] = {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_hit0": m("hit0"),
        "mean_delta_E": m("delta_E"),
        "mean_neg_delta_L_live": m("neg_delta_L_live"),
        "frac_L_improved": float(np.mean([t["delta_L_live"] < 0 for t in trials]))
        if trials
        else float("nan"),
        "corr_neg_dL_live_delta_E": _corr(neg_dL, dE),
    }
    for k in CHANNELS:
        gs = [t["G"][k] for t in trials if not np.isnan(t["G"][k])]
        out[f"mean_G_{k}"] = float(np.mean(gs)) if gs else float("nan")
        # P(W→C)
        w2c = sum(1 for t in trials if t["cls"][k] == "W2C")
        w2w = sum(1 for t in trials if t["cls"][k] == "W2W")
        n_w = w2c + w2w
        out[f"P_W2C_{k}"] = float(w2c / n_w) if n_w else float("nan")
        out[f"n_wrong0_{k}"] = n_w
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arms", default="none,predictive,converted,random")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--sequential", action="store_true", help="full C→H→O (slow)")
    args = ap.parse_args()
    fast = not args.sequential
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    boundaries = _load_boundaries()
    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    banks = _load_banks(args.seed)
    mstars = [tuple(m) for m in all_m_star_masks()]

    all_trials: dict[str, list] = {a: [] for a in arms}
    mode = "fast" if fast else "sequential"
    print(
        f"=== Phase 8A live-B 8-way mode={mode} α={args.alpha} β={args.beta} "
        f"reps={args.reps} arms={arms} ===",
        flush=True,
    )
    print(
        "  B aucs: " + ", ".join(f"{k}={boundaries[k]['auc']:.3f}" for k in CHANNELS),
        flush=True,
    )

    for arm in arms:
        dirs = banks[arm]
        for mi, mstar in enumerate(mstars):
            for r in range(args.reps):
                seed = args.seed + 1000 * mi + 10 * r + (hash(arm) % 97)
                torch.manual_seed(seed)
                print(f"  {arm} m*={mstar} r={r}", flush=True)
                tr = run_trial(
                    sc,
                    loaded,
                    mstar=mstar,
                    dirs=dirs,
                    boundaries=boundaries,
                    alpha=args.alpha,
                    beta=args.beta,
                    seed=seed,
                    arm=arm,
                    fast=fast,
                )
                all_trials[arm].append(tr)
                print(
                    f"    −ΔL_live={tr['neg_delta_L_live']:+.3f} ΔE={tr['delta_E']:+.1f} "
                    f"hit={tr['hit']} G={{C:{tr['G']['C']:+.2f},H:{tr['G']['H']:+.2f},"
                    f"O:{tr['G']['O']:+.2f}}}",
                    flush=True,
                )

    aggs = {a: _agg(all_trials[a]) for a in arms}
    pc, pr, pn, pp = (aggs.get(x) for x in ("converted", "random", "none", "predictive"))

    gate: dict[str, Any] = {
        "question": "Does live-B 8-way objective close the gap?",
        "mode": mode,
    }
    if pc and pr:
        gate["vc_beats_random_neg_dL_live"] = bool(
            pc["mean_neg_delta_L_live"] > pr["mean_neg_delta_L_live"]
        )
        gate["vc_beats_random_delta_E"] = bool(pc["mean_delta_E"] > pr["mean_delta_E"])
        gate["vc_beats_random_hit"] = bool(pc["P_hit"] > pr["P_hit"])
        gate["vc_beats_random_G"] = {
            k: bool(pc[f"mean_G_{k}"] > pr[f"mean_G_{k}"])
            for k in CHANNELS
            if not np.isnan(pc[f"mean_G_{k}"]) and not np.isnan(pr[f"mean_G_{k}"])
        }
    if pc and pn:
        gate["vc_beats_none_neg_dL_live"] = bool(
            pc["mean_neg_delta_L_live"] > pn["mean_neg_delta_L_live"]
        )
        gate["vc_beats_none_delta_E"] = bool(pc["mean_delta_E"] > pn["mean_delta_E"])
    if pc and pp:
        gate["vc_beats_predictive_neg_dL_live"] = bool(
            pc["mean_neg_delta_L_live"] > pp["mean_neg_delta_L_live"]
        )
    if pc:
        gate["corr_neg_dL_live_delta_E"] = pc["corr_neg_dL_live_delta_E"]
        gate["corr_strong"] = bool(
            (not np.isnan(pc["corr_neg_dL_live_delta_E"]))
            and abs(pc["corr_neg_dL_live_delta_E"]) >= 0.3
        )
        gate["diagnostic_closed"] = bool(
            gate.get("vc_beats_random_neg_dL_live")
            and gate.get("corr_strong")
            and pc["corr_neg_dL_live_delta_E"] > 0
        )
        gate["read"] = (
            "If E[-ΔL_live]_vc ≫ controls AND corr(-ΔL_live, ΔE)≫0: chain closed. "
            "If L_live moves but corr weak: bottleneck is post-boundary behavioral realization."
        )

    payload = {
        "protocol": "Phase 8A live-boundary 8-way",
        "formula": "L_live = Σ softplus(-β s_k B_k), G_k = s_k (B_after-B_before)",
        "mode": mode,
        "alpha": args.alpha,
        "beta": args.beta,
        "reps": args.reps,
        "seed": args.seed,
        "phase7_auc": {k: boundaries[k]["auc"] for k in CHANNELS},
        "agg": aggs,
        "trials": {a: all_trials[a] for a in arms},
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8A — Live-boundary 8-way",
        "",
        r"> $\mathcal{L}_{\mathrm{live}}=\sum_k\mathrm{softplus}(-\beta s_k B_k)$ with Phase-7 $B_k=w_k^\top h+b_k$.",
        "",
        f"mode=**{mode}**, α={args.alpha}, β={args.beta}, reps={args.reps}/m*, frozen $v_c$.",
        "",
        "## Aggregate",
        "",
        "| Arm | P(hit) | ΔE | E[−ΔL_live] | frac L↓ | corr(−ΔL,ΔE) | G_C | G_H | G_O |",
        "|-----|--------|-----|-------------|---------|--------------|-----|-----|-----|",
    ]
    for a in arms:
        g = aggs[a]
        lines.append(
            f"| {a} | {g['P_hit']:.3f} | {g['mean_delta_E']:+.3f} | "
            f"{g['mean_neg_delta_L_live']:+.3f} | {g['frac_L_improved']:.2f} | "
            f"{g['corr_neg_dL_live_delta_E']:+.3f} | "
            f"{g['mean_G_C']:+.2f} | {g['mean_G_H']:+.2f} | {g['mean_G_O']:+.2f} |"
        )

    lines += [
        "",
        "## P(W→C) at live trajectory",
        "",
        "| Arm | C | H | O |",
        "|-----|---|---|---|",
    ]
    for a in arms:
        g = aggs[a]
        cells = []
        for k in CHANNELS:
            p = g[f"P_W2C_{k}"]
            cells.append(f"{p:.2f}" if not np.isnan(p) else "—")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Gate",
        "",
        f"- $v_c$ beats random on E[−ΔL_live]: **{gate.get('vc_beats_random_neg_dL_live', 'n/a')}**",
        f"- $v_c$ beats random on ΔE: **{gate.get('vc_beats_random_delta_E', 'n/a')}**",
        f"- $v_c$ beats random on hit: **{gate.get('vc_beats_random_hit', 'n/a')}**",
        f"- corr(−ΔL_live, ΔE) strong & positive: **{gate.get('corr_strong', 'n/a')}** "
        f"(r={gate.get('corr_neg_dL_live_delta_E', float('nan'))})",
        f"- Diagnostic closed: **{gate.get('diagnostic_closed', 'n/a')}**",
        "",
        gate.get("read", ""),
        "",
        "No new $v$. No policy.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "agg": aggs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
