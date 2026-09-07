#!/usr/bin/env python3
"""Phase 8I — Corrected C/H composition (O frozen at 8H success path).

O locked:
  α_O = 1.5, early-FINAL

Compare only order:
  C → H → O    vs    H → C → O

Target-sign d_k = (2 m*_k − 1) v_c^k; same seeds; H decision-token;
C tool/plan path as in 8H. No new v. No 8-way reopen. No policy.

Primary behavioral metrics (per stage):
  ΔE_C, ΔE_H, P(C correct), P(H correct)

Crucial act-space cross-ratios at corrected live sites (Phase-7 B):
  R_{C|H} = |ΔB_C(h_H)| / |ΔB_C(h_0)|
  R_{H|C} = |ΔB_H(h_C)| / |ΔB_H(h_0)|

  .venv/bin/python scripts/run_phase8i_ch_composition.py --reps 2
  .venv/bin/python scripts/run_phase8i_ch_composition.py --reps 2 --n-cross 12
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

from scripts.sync_channel_margins import SPECS, margin_at_site, unit  # noqa: E402
from scripts.sync_eq import extract_plan, score_plan, sync_error_norm  # noqa: E402
from scripts.sync_h_decision import decision_at_site, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8i_ch_composition.json"
MD = ROOT / "data" / "results" / "sync_phase8i_ch_composition.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260905  # match 8D/8H
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

SELECTED_MSTAR = (
    (0, 0, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 1),
)

ORDERS = {
    "C_then_H": ("C", "H", "O"),
    "H_then_C": ("H", "C", "O"),
}

# 5C replicate refs
REF_5C = {"C_cross_ratio": 1.33, "H_cross_ratio": 0.66, "C_retention": 0.83}
REF_8H = {"P_hit": 0.31, "mean_delta_E_total": 0.38, "P_C": 0.56, "P_H": 0.69, "P_O": 0.75}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


def _load_boundaries() -> dict[str, dict[str, Any]]:
    blob = json.loads(P7_PATH.read_text())["boundaries"]
    out = {}
    for k in ("C", "H"):
        b = blob[k]
        if not b.get("ok") or not b.get("w"):
            raise SystemExit(f"need Phase-7 boundary for {k}")
        out[k] = {"w": np.asarray(b["w"], dtype=np.float64), "b": float(b["b"])}
    return out


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


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


def _e(mstar, S):
    return [int(mstar[i]) - int(S[i]) for i in range(3)]


def _s_star(mstar_k: int) -> int:
    return int(2 * int(mstar_k) - 1)


def _compose(active: dict[str, tuple[float, np.ndarray]]) -> np.ndarray | None:
    d = None
    for _k, (s, v) in active.items():
        term = float(s) * v
        d = term if d is None else d + term
    return d


def _run_episode(
    sc,
    loaded,
    *,
    active: dict[str, tuple[int, np.ndarray]],
    seed: int,
    h_plan_prefill: str | None = None,
) -> dict[str, Any]:
    """8H path: H decision-token, early-FINAL O, α=1.5 all channels."""
    h_hook = plan_hook = o_hook = tool_c_hook = None
    alpha = ALPHA

    if "H" in active:
        sH, vH = active["H"]
        h_hook = make_steer_hook(loaded, vH, alpha * float(sH))
        if "C" in active and not (h_plan_prefill and str(h_plan_prefill).strip()):
            sC, vC = active["C"]
            plan_hook = make_steer_hook(loaded, vC, alpha * float(sC))
        if "C" in active:
            sC, vC = active["C"]
            tool_c_hook = make_steer_hook(loaded, vC, alpha * float(sC))

        def hook_for_turn(phase: str, turn: int):
            if phase == "tool" and turn > 0:
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

    else:
        if "C" in active:
            sC, vC = active["C"]
            tool_c_hook = make_steer_hook(loaded, vC, alpha * float(sC))

        def hook_for_turn(phase: str, _turn: int):
            if phase == "tool":
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

    report_prefill = None
    if "O" in active:
        sO, vO = active["O"]
        o_hook = make_steer_hook(loaded, vO, alpha * float(sO))
        report_prefill = "FINAL: "

    row = sc.run_episode(
        loaded,
        cls=NEUTRAL_CLS,
        task_id=NEUTRAL_TASK,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
        capture_activations=False,
        h_decision_hook=h_hook,
        plan_draft_hook=plan_hook,
        h_plan_prefill=h_plan_prefill if "H" in active else None,
        report_prefill=report_prefill,
    )
    for h in (h_hook, plan_hook, o_hook, tool_c_hook):
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
    S = _S_from_row(row)
    plan = extract_plan((row.get("final") or "") + "\n" + (row.get("messages_text") or ""))
    prefill = (plan if plan.endswith("\n") else plan + "\n") if plan else ""
    return {
        "S": S,
        "plan_prefill": prefill,
        "task_ok": int(bool((row.get("final") or "").strip())),
    }


def run_sequential_trial(
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict[str, np.ndarray],
    order: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    base = _run_episode(sc, loaded, active={}, seed=seed)
    S = base["S"]
    plan_prefill = base.get("plan_prefill") or ""
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(_e(mstar, S)))]
    stage_E: dict[str, float] = {}
    stage_bits: dict[str, dict[str, int]] = {}
    n_interv = 0
    n_episodes = 1
    active: dict[str, tuple[int, np.ndarray]] = {}
    E_before_ch = traj_E[0]

    for k in order:
        e = _e(mstar, S)
        E_pre = float(sync_error_norm(e))
        added = False
        if e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (_s_star(mstar[CH_IDX[k]]), dirs[k])
            n_interv += 1
            added = True
        if not added:
            traj_S.append(list(S))
            traj_E.append(traj_E[-1])
            stage_E[k] = 0.0
            stage_bits[k] = {
                "bit_C": int(S[0] == mstar[0]),
                "bit_H": int(S[1] == mstar[1]),
                "bit_O": int(S[2] == mstar[2]),
                "skipped_noop": 1,
            }
            continue
        out = _run_episode(
            sc,
            loaded,
            active=active,
            seed=seed,
            h_plan_prefill=plan_prefill if "H" in active else None,
        )
        n_episodes += 1
        S = out["S"]
        if out.get("plan_prefill"):
            plan_prefill = out["plan_prefill"]
        E_post = float(sync_error_norm(_e(mstar, S)))
        traj_S.append(list(S))
        traj_E.append(E_post)
        stage_E[k] = float(E_pre - E_post)
        stage_bits[k] = {
            "bit_C": int(S[0] == mstar[0]),
            "bit_H": int(S[1] == mstar[1]),
            "bit_O": int(S[2] == mstar[2]),
            "skipped_noop": 0,
        }

    # after C+H stages (before final O effect attribution uses full traj)
    # find S after both C and H have been considered (2nd non-O stage index)
    ch_stages = [k for k in order if k in ("C", "H")]
    # traj indices: 0=free, 1=after first, 2=after second, 3=after third
    idx_after_ch = len(ch_stages)  # after 2 C/H stages
    S_after_ch = traj_S[idx_after_ch]
    E_after_ch = traj_E[idx_after_ch]

    S0, Sf = traj_S[0], traj_S[-1]
    return {
        "mstar": list(mstar),
        "order": list(order),
        "S0": S0,
        "S_after_CH": S_after_ch,
        "S_final": Sf,
        "E_traj": traj_E,
        "delta_E_total": float(traj_E[0] - traj_E[-1]),
        "delta_E_CH": float(E_before_ch - E_after_ch),
        "delta_E_C": float(stage_E.get("C", 0.0)),
        "delta_E_H": float(stage_E.get("H", 0.0)),
        "delta_E_O": float(stage_E.get("O", 0.0)),
        "after_C": stage_bits.get("C"),
        "after_H": stage_bits.get("H"),
        "after_O": stage_bits.get("O"),
        "bit_C_after_CH": int(S_after_ch[0] == mstar[0]),
        "bit_H_after_CH": int(S_after_ch[1] == mstar[1]),
        "both_CH_after_CH": int(S_after_ch[0] == mstar[0] and S_after_ch[1] == mstar[1]),
        "hit": int(Sf == list(mstar)),
        "bit_C": int(Sf[0] == mstar[0]),
        "bit_H": int(Sf[1] == mstar[1]),
        "bit_O": int(Sf[2] == mstar[2]),
        "n_interventions": n_interv,
        "n_episodes": n_episodes,
    }


def _agg_beh(trials: list[dict]) -> dict[str, Any]:
    def m(key: str) -> float:
        return float(np.mean([t[key] for t in trials])) if trials else float("nan")

    def m_after(stage: str, bit: str) -> float:
        vals = []
        for t in trials:
            row = t.get(f"after_{stage}")
            if row and bit in row:
                vals.append(row[bit])
        return float(np.mean(vals)) if vals else float("nan")

    et = [float(np.mean([t["E_traj"][i] for t in trials])) for i in range(4)] if trials else []
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_C": m("bit_C"),
        "P_H": m("bit_H"),
        "P_O": m("bit_O"),
        "P_C_after_C": m_after("C", "bit_C"),
        "P_H_after_C": m_after("C", "bit_H"),
        "P_C_after_H": m_after("H", "bit_C"),
        "P_H_after_H": m_after("H", "bit_H"),
        "P_C_after_CH": m("bit_C_after_CH"),
        "P_H_after_CH": m("bit_H_after_CH"),
        "P_both_CH_after_CH": m("both_CH_after_CH"),
        "mean_delta_E_total": m("delta_E_total"),
        "mean_delta_E_CH": m("delta_E_CH"),
        "mean_delta_E_C": m("delta_E_C"),
        "mean_delta_E_H": m("delta_E_H"),
        "mean_delta_E_O": m("delta_E_O"),
        "mean_E_traj": et,
    }


def _base_messages(sc, *, cls: str, task_id: str) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def _stratified_schedule(sc, n: int, seed: int) -> list[tuple[str, str]]:
    plan = sc.iter_episode_plan()
    priv = [(c, tid) for c, tid, _f, sens in plan if sens != "public"]
    pub = [(c, tid) for c, tid, _f, sens in plan if sens == "public"]
    rng = np.random.default_rng(seed)
    rng.shuffle(priv)
    rng.shuffle(pub)
    out: list[tuple[str, str]] = []
    i_p = i_u = 0
    for i in range(n):
        if i % 2 == 0 and priv:
            out.append(priv[i_p % len(priv)])
            i_p += 1
        elif pub:
            out.append(pub[i_u % len(pub)])
            i_u += 1
        elif priv:
            out.append(priv[i_p % len(priv)])
            i_p += 1
    return out


def _capture_h_C(loaded, base, *, prior: dict[str, tuple[float, np.ndarray]], alpha: float):
    """h at mid-PLAN C site under optional prior hooks (composed)."""
    pd = _compose(prior)
    hook = make_steer_hook(loaded, pd, alpha) if pd is not None else None
    try:
        s = margin_at_site(loaded, base, SPECS["C"], hook=hook, capture_h=True)
        return np.asarray(s["h"], dtype=np.float64)
    finally:
        if hook is not None:
            hook.remove()


def _capture_h_H(loaded, base, plan_pf: str, *, prior: dict[str, tuple[float, np.ndarray]], alpha: float):
    """h at live post-PLAN H decision site under optional prior."""
    pd = _compose(prior)
    hook = make_steer_hook(loaded, pd, alpha) if pd is not None else None
    try:
        s = decision_at_site(
            loaded, base, assistant_prefill=plan_pf, hook=hook, capture_h=True
        )
        return np.asarray(s["h"], dtype=np.float64)
    finally:
        if hook is not None:
            hook.remove()


def _delta_B_hooked(
    *,
    capture_fn,
    w: np.ndarray,
    b: float,
    prior: dict[str, tuple[float, np.ndarray]],
    add: dict[str, tuple[float, np.ndarray]],
    alpha: float,
) -> float:
    """|B(h_{prior+add}) − B(h_prior)| via real hooked forwards (not analytical)."""
    h0 = capture_fn(prior=prior, alpha=alpha)
    h1 = capture_fn(prior={**prior, **add}, alpha=alpha)
    return abs(B_score(h1, w, b) - B_score(h0, w, b))


def run_cross_ratios(
    sc,
    loaded,
    *,
    dirs: dict[str, np.ndarray],
    boundaries: dict[str, dict[str, Any]],
    n: int,
    seed: int,
) -> dict[str, Any]:
    """R_C|H and R_H|C on corrected live sites (hooked ΔB; nonlinear retention)."""
    schedule = _stratified_schedule(sc, n, seed + 777)
    rows_C: list[dict] = []
    rows_H: list[dict] = []

    print(f"=== live-site cross ratios n={n} (hooked ΔB) ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        ep_seed = seed + 17 * i
        torch.manual_seed(ep_seed)
        state = sc.run_tools_phase(
            loaded, cls=cls, task_id=task_id, seed=ep_seed, with_plan_format=True
        )
        row = sc.finalize_episode(loaded, state)
        blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
        plan = extract_plan(blob) or ""
        if not plan.strip():
            print(f"  [{i}] skip (no PLAN)", flush=True)
            continue
        plan_pf = plan if plan.endswith("\n") else plan + "\n"
        base = _base_messages(sc, cls=cls, task_id=task_id)
        wC, bC = boundaries["C"]["w"], boundaries["C"]["b"]
        wH, bH = boundaries["H"]["w"], boundaries["H"]["b"]

        def cap_C(*, prior, alpha):
            return _capture_h_C(loaded, base, prior=prior, alpha=alpha)

        def cap_H(*, prior, alpha):
            return _capture_h_H(loaded, base, plan_pf, prior=prior, alpha=alpha)

        # --- C contrasts: |ΔB_C| alone vs under H prior ---
        for sC in (+1.0, -1.0):
            add_C = {"C": (sC, dirs["C"])}
            prior_H = {"H": (sC, dirs["H"])}  # matched-sign prior (5C style)
            dB0 = _delta_B_hooked(
                capture_fn=cap_C, w=wC, b=bC, prior={}, add=add_C, alpha=ALPHA
            )
            dBH = _delta_B_hooked(
                capture_fn=cap_C, w=wC, b=bC, prior=prior_H, add=add_C, alpha=ALPHA
            )
            # cross of C→H: |ΔB_H| from applying C alone, and under H prior
            cross0 = _delta_B_hooked(
                capture_fn=cap_H, w=wH, b=bH, prior={}, add=add_C, alpha=ALPHA
            )
            crossH = _delta_B_hooked(
                capture_fn=cap_H, w=wH, b=bH, prior=prior_H, add=add_C, alpha=ALPHA
            )
            rows_C.append(
                {
                    "i": i,
                    "s_C": sC,
                    "abs_dB_C_h0": dB0,
                    "abs_dB_C_hH": dBH,
                    "R_C_given_H": float(dBH / max(dB0, 1e-9)),
                    "cross_H_from_C_alone": cross0,
                    "cross_H_from_C_given_H": crossH,
                    "cross_ratio_C": float(crossH / max(cross0, 1e-9)),
                }
            )

        # --- H contrasts: |ΔB_H| alone vs under C prior ---
        for sH in (+1.0, -1.0):
            add_H = {"H": (sH, dirs["H"])}
            prior_C = {"C": (sH, dirs["C"])}
            dB0 = _delta_B_hooked(
                capture_fn=cap_H, w=wH, b=bH, prior={}, add=add_H, alpha=ALPHA
            )
            dBC = _delta_B_hooked(
                capture_fn=cap_H, w=wH, b=bH, prior=prior_C, add=add_H, alpha=ALPHA
            )
            cross0 = _delta_B_hooked(
                capture_fn=cap_C, w=wC, b=bC, prior={}, add=add_H, alpha=ALPHA
            )
            crossC = _delta_B_hooked(
                capture_fn=cap_C, w=wC, b=bC, prior=prior_C, add=add_H, alpha=ALPHA
            )
            rows_H.append(
                {
                    "i": i,
                    "s_H": sH,
                    "abs_dB_H_h0": dB0,
                    "abs_dB_H_hC": dBC,
                    "R_H_given_C": float(dBC / max(dB0, 1e-9)),
                    "cross_C_from_H_alone": cross0,
                    "cross_C_from_H_given_C": crossC,
                    "cross_ratio_H": float(crossC / max(cross0, 1e-9)),
                }
            )

        print(
            f"  [{i}] R_C|H≈{rows_C[-1]['R_C_given_H']:.2f} "
            f"cross_C≈{rows_C[-1]['cross_ratio_C']:.2f} "
            f"R_H|C≈{rows_H[-1]['R_H_given_C']:.2f}",
            flush=True,
        )

    def _sum_R(
        rows: list[dict], rkey: str, a0: str, a1: str, x0: str, x1: str, xr: str
    ) -> dict[str, float]:
        if not rows:
            return {
                "n": 0,
                "R": float("nan"),
                "R_from_means": float("nan"),
                "abs_dB_h0": float("nan"),
                "abs_dB_h_prior": float("nan"),
                "cross_alone": float("nan"),
                "cross_given_prior": float("nan"),
                "cross_ratio": float("nan"),
            }
        a0s = [r[a0] for r in rows]
        a1s = [r[a1] for r in rows]
        return {
            "n": len(rows),
            "R": float(np.mean([r[rkey] for r in rows])),
            "R_from_means": float(np.mean(a1s) / max(np.mean(a0s), 1e-9)),
            "abs_dB_h0": float(np.mean(a0s)),
            "abs_dB_h_prior": float(np.mean(a1s)),
            "cross_alone": float(np.mean([r[x0] for r in rows])),
            "cross_given_prior": float(np.mean([r[x1] for r in rows])),
            "cross_ratio": float(
                np.mean([r[x1] for r in rows]) / max(np.mean([r[x0] for r in rows]), 1e-9)
            ),
            "cross_ratio_mean_of_ratios": float(np.mean([r[xr] for r in rows])),
        }

    c_sum = _sum_R(
        rows_C,
        "R_C_given_H",
        "abs_dB_C_h0",
        "abs_dB_C_hH",
        "cross_H_from_C_alone",
        "cross_H_from_C_given_H",
        "cross_ratio_C",
    )
    h_sum = _sum_R(
        rows_H,
        "R_H_given_C",
        "abs_dB_H_h0",
        "abs_dB_H_hC",
        "cross_C_from_H_alone",
        "cross_C_from_H_given_C",
        "cross_ratio_H",
    )

    return {
        "C_given_H": c_sum,
        "H_given_C": h_sum,
        "R_C_given_H": c_sum["R_from_means"],
        "R_H_given_C": h_sum["R_from_means"],
        "cross_ratio_C": c_sum["cross_ratio"],
        "cross_ratio_H": h_sum["cross_ratio"],
        "rows_C": rows_C,
        "rows_H": rows_H,
        "ref_5C": REF_5C,
        "note": "ΔB from hooked forwards (not analytical); R≠1 only if nonlinear/site coupling",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-cross", type=int, default=8, help="free-run n for live R ratios")
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--skip-beh", action="store_true")
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = _load_vc()
    boundaries = _load_boundaries()
    mstars = list(SELECTED_MSTAR)

    by_order: dict[str, Any] = {}
    all_trials: dict[str, list] = {}

    if not args.skip_beh:
        print(
            f"=== Phase 8I behavioral C↔H order "
            f"n_m*={len(mstars)} reps={args.reps} O frozen early-FINAL α=1.5 ===",
            flush=True,
        )
        for okey, order in ORDERS.items():
            trials = []
            for mi, mstar in enumerate(mstars):
                for r in range(args.reps):
                    seed = args.seed + 1000 * mi + 10 * r
                    torch.manual_seed(seed)
                    print(f"  {okey} m*={mstar} r={r}", flush=True)
                    tr = run_sequential_trial(
                        sc, loaded, mstar=mstar, dirs=dirs, order=order, seed=seed
                    )
                    trials.append(tr)
                    print(
                        f"    E:{[round(x, 2) for x in tr['E_traj']]} "
                        f"ΔE_C={tr['delta_E_C']:+.2f} ΔE_H={tr['delta_E_H']:+.2f} "
                        f"CH_ok={tr['both_CH_after_CH']} hit={tr['hit']}",
                        flush=True,
                    )
            agg = _agg_beh(trials)
            by_order[okey] = agg
            all_trials[okey] = trials
            print(
                f"  >> {okey}: P(hit)={agg['P_hit']:.2f} ΔE={agg['mean_delta_E_total']:+.2f} "
                f"ΔE_C={agg['mean_delta_E_C']:+.2f} ΔE_H={agg['mean_delta_E_H']:+.2f} "
                f"P(both_CH)={agg['P_both_CH_after_CH']:.2f}",
                flush=True,
            )

    cross: dict[str, Any] = {}
    if not args.skip_cross:
        cross = run_cross_ratios(
            sc,
            loaded,
            dirs=dirs,
            boundaries=boundaries,
            n=args.n_cross,
            seed=args.seed,
        )
        # drop bulky rows from printed gate; keep in payload
        print(
            f"  >> R_C|H={cross['R_C_given_H']:.3f} R_H|C={cross['R_H_given_C']:.3f} "
            f"(5C ref C cross≈{REF_5C['C_cross_ratio']})",
            flush=True,
        )

    # gates
    gate: dict[str, Any] = {
        "hypothesis": "H→C or C→H order matters under corrected live sites; R captures coupling",
        "O_frozen": {"alpha_O": 1.5, "early_FINAL": True},
        "ref_8H_C_then_H": REF_8H,
        "ref_5C": REF_5C,
    }
    if by_order:
        a = by_order.get("C_then_H", {})
        b = by_order.get("H_then_C", {})
        gate["C_then_H"] = {
            "P_hit": a.get("P_hit"),
            "delta_E": a.get("mean_delta_E_total"),
            "delta_E_C": a.get("mean_delta_E_C"),
            "delta_E_H": a.get("mean_delta_E_H"),
            "P_both_CH": a.get("P_both_CH_after_CH"),
            "P_C_after_CH": a.get("P_C_after_CH"),
            "P_H_after_CH": a.get("P_H_after_CH"),
        }
        gate["H_then_C"] = {
            "P_hit": b.get("P_hit"),
            "delta_E": b.get("mean_delta_E_total"),
            "delta_E_C": b.get("mean_delta_E_C"),
            "delta_E_H": b.get("mean_delta_E_H"),
            "P_both_CH": b.get("P_both_CH_after_CH"),
            "P_C_after_CH": b.get("P_C_after_CH"),
            "P_H_after_CH": b.get("P_H_after_CH"),
        }
        d_hit = float(b.get("P_hit", 0) - a.get("P_hit", 0))
        d_ch = float(b.get("P_both_CH_after_CH", 0) - a.get("P_both_CH_after_CH", 0))
        d_E = float(b.get("mean_delta_E_total", 0) - a.get("mean_delta_E_total", 0))
        gate["H_then_C_beats_C_then_H"] = bool(d_ch > 0.05 or d_E > 0.05 or d_hit > 0.05)
        gate["C_then_H_beats_H_then_C"] = bool(d_ch < -0.05 or d_E < -0.05 or d_hit < -0.05)
        gate["orders_similar"] = bool(
            not gate["H_then_C_beats_C_then_H"] and not gate["C_then_H_beats_H_then_C"]
        )
        preferred = "tie"
        if gate["H_then_C_beats_C_then_H"]:
            preferred = "H_then_C"
        elif gate["C_then_H_beats_H_then_C"]:
            preferred = "C_then_H"
        gate["preferred_order"] = preferred
    if cross:
        gate["R_C_given_H"] = cross["R_C_given_H"]
        gate["R_H_given_C"] = cross["R_H_given_C"]
        gate["cross_ratio_C"] = cross.get("cross_ratio_C")
        gate["cross_ratio_H"] = cross.get("cross_ratio_H")
        gate["C_self_amplified_by_H"] = bool(cross["R_C_given_H"] > 1.1)
        gate["C_cross_worsens"] = bool((cross.get("cross_ratio_C") or 0) > 1.25)
        gate["H_attenuated_by_C"] = bool(cross["R_H_given_C"] < 0.9)
    gate["read"] = (
        "If H→C outperforms C→H: prefer H first. "
        "If orders similar/weak: next is conditional C skip (small policy), not re-convert. "
        "O stays frozen. No 8-way reopen."
    )

    payload = {
        "protocol": "Phase 8I C/H composition (O frozen early-FINAL α=1.5)",
        "alpha": ALPHA,
        "reps": args.reps,
        "mstar_set": [list(m) for m in mstars],
        "orders": {k: list(v) for k, v in ORDERS.items()},
        "by_order": by_order,
        "cross": {
            k: v
            for k, v in cross.items()
            if k not in ("rows_C", "rows_H")
        }
        if cross
        else {},
        "cross_rows": {"C": cross.get("rows_C"), "H": cross.get("rows_H")} if cross else {},
        "gate": gate,
        "trials": all_trials,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8I — C/H composition (O frozen)",
        "",
        r"> O locked: $\alpha_O=1.5$, early-FINAL. Compare $C\to H$ vs $H\to C$. "
        r"Target-sign; H decision-token. No new $v$; no 8-way reopen.",
        "",
        f"m* n={len(mstars)}, reps={args.reps}.",
        "",
        "## Behavioral order",
        "",
        "| order | P(hit) | ΔE | ΔE_C | ΔE_H | P(C)|_CH | P(H)|_CH | P(both CH) |",
        "|-------|--------|---:|-----:|-----:|----------:|----------:|-----------:|",
    ]
    for okey in ORDERS:
        if okey not in by_order:
            continue
        g = by_order[okey]
        label = "C→H→O" if okey == "C_then_H" else "H→C→O"
        lines.append(
            f"| {label} | {g['P_hit']:.2f} | {g['mean_delta_E_total']:+.2f} | "
            f"{g['mean_delta_E_C']:+.2f} | {g['mean_delta_E_H']:+.2f} | "
            f"{g['P_C_after_CH']:.2f} | {g['P_H_after_CH']:.2f} | "
            f"{g['P_both_CH_after_CH']:.2f} |"
        )
    lines += [
        "",
        f"| 8H ref (C→H→O) | {REF_8H['P_hit']:.2f} | {REF_8H['mean_delta_E_total']:+.2f} | "
        f"— | — | {REF_8H['P_C']:.2f} | {REF_8H['P_H']:.2f} | — |",
        "",
        "### After each stage (bit correct)",
        "",
        "| order | P(C) after C | P(H) after C | P(C) after H | P(H) after H |",
        "|-------|-------------:|-------------:|-------------:|-------------:|",
    ]
    for okey in ORDERS:
        if okey not in by_order:
            continue
        g = by_order[okey]
        label = "C→H→O" if okey == "C_then_H" else "H→C→O"
        lines.append(
            f"| {label} | {g['P_C_after_C']:.2f} | {g['P_H_after_C']:.2f} | "
            f"{g['P_C_after_H']:.2f} | {g['P_H_after_H']:.2f} |"
        )
    if cross:
        lines += [
            "",
            "## Live-site cross ratios (Phase-7 $B$, hooked $\\Delta B$)",
            "",
            r"$R_{C|H}=|\Delta B_C(h_H)|/|\Delta B_C(h_0)|$, "
            r"$R_{H|C}=|\Delta B_H(h_C)|/|\Delta B_H(h_0)|$; "
            r"cross_ratio = cross-after-prior / cross-alone.",
            "",
            "| | $R$ (self) | cross_ratio | $|\\Delta B|$ $h_0$ | $|\\Delta B|$ prior |",
            "|--|------------:|------------:|-------------------:|-------------------:|",
            f"| $C|H$ | **{cross['R_C_given_H']:.2f}** | "
            f"**{cross['cross_ratio_C']:.2f}** | "
            f"{cross['C_given_H']['abs_dB_h0']:.3f} | "
            f"{cross['C_given_H']['abs_dB_h_prior']:.3f} |",
            f"| $H|C$ | **{cross['R_H_given_C']:.2f}** | "
            f"**{cross['cross_ratio_H']:.2f}** | "
            f"{cross['H_given_C']['abs_dB_h0']:.3f} | "
            f"{cross['H_given_C']['abs_dB_h_prior']:.3f} |",
            "",
            f"5C ref: C cross_ratio={REF_5C['C_cross_ratio']}, H={REF_5C['H_cross_ratio']}.",
            "",
        ]
    lines += [
        "## Gate",
        "",
        f"- Preferred order: **{gate.get('preferred_order', 'n/a')}**",
        f"- H→C beats C→H: **{gate.get('H_then_C_beats_C_then_H', 'n/a')}**",
        f"- Orders similar: **{gate.get('orders_similar', 'n/a')}**",
        f"- $R_{{C|H}}$: **{gate.get('R_C_given_H', 'n/a')}**",
        f"- $R_{{H|C}}$: **{gate.get('R_H_given_C', 'n/a')}**",
        f"- C cross_ratio (5C-style): **{gate.get('cross_ratio_C', 'n/a')}**",
        f"- C cross worsens (>1.25): **{gate.get('C_cross_worsens', 'n/a')}**",
        "",
        gate["read"],
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "md": str(MD),
                "gate": {k: v for k, v in gate.items() if k != "read"},
                "by_order": by_order,
                "R_C_given_H": cross.get("R_C_given_H") if cross else None,
                "R_H_given_C": cross.get("R_H_given_C") if cross else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
