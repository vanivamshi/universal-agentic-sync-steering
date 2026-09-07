#!/usr/bin/env python3
"""Phase 9M — Target retention: hold vs planner after acquire.

9L showed P_acq>0 for most m* but P_ret≈0 under stay-seek. Diagnostic:

  Once S_t = m*, either
    HOLD:    no further steering (active={}) for n_hold steps
    PLANNER: continue stay-seeking interventions for n_hold steps

Acquisition uses frozen hybrid (Γ' soft / all-channel sink-seek hard).
No new v. No new acquisition objective. No extra memory.

Measures per m*:
  P_acq
  P_stay | acq, no_steer
  P_stay | acq, planner
  P_final under each retention mode

If P_stay|hold ≫ P_stay|planner, the planner destroys targets.
If both ≈0, intrinsic instability → need a retention mechanism.

After diagnostic, optional --full-8way runs acquire+hold for all eight
with success criterion P_final(m*)>0 ∀m*.

  .venv/bin/python -u scripts/run_phase9m_target_hold.py --reps 4
  .venv/bin/python -u scripts/run_phase9m_target_hold.py --full-8way --reps 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9m_target_hold.json"
MD = ROOT / "data" / "results" / "sync_phase9m_target_hold.md"
SEED = 20261001
ORDER = ("H", "C", "O")
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
HARD = {(0, 0, 0), (0, 0, 1), (1, 1, 0)}


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _E(mstar, S) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def apply_free(p8k, sc, loaded, *, seed, plan_prefill, cache):
    """No-steer continuation."""
    key = (int(seed), "FREE", plan_prefill or "")
    if cache is not None and key in cache:
        return cache[key], True
    out = p8k._run_episode(
        sc, loaded, active={}, seed=seed, h_plan_prefill=None
    )
    # free runs ignore h_plan; keep plan string for bookkeeping only
    if cache is not None:
        cache[key] = out
    return out, False


def choose_stay(s, mstar, lookup, *, a_prev, stagnated):
    meta: dict[str, Any] = {"mode": "stay_seek", "by_a": {}, "blocked": None}
    cands = []
    for a in ORDER:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "P_hit": None}
        ph = st.get("P_hit")
        meta["by_a"][a] = {"P_hit": ph, "n": st.get("n")}
        if stagnated and a_prev is not None and a == a_prev:
            meta["blocked"] = a
            continue
        if int(st.get("n") or 0) >= 1 and ph is not None:
            cands.append((a, float(ph), int(st["n"])))
    if cands:
        cands.sort(key=lambda t: (t[1], t[2], -ORDER.index(t[0])), reverse=True)
        meta["a_star"] = cands[0][0]
        meta["P_hit_star"] = cands[0][1]
        return cands[0][0], meta
    for a in ORDER:
        if stagnated and a == a_prev:
            continue
        meta["fallback"] = True
        meta["a_star"] = a
        return a, meta
    meta["a_star"] = "O"
    return "O", meta


def acquire(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    mstar,
    seed,
    S0,
    plan0,
    max_steps_acq,
    lam,
    cache,
) -> dict[str, Any]:
    mk = tuple(mstar)
    S = list(S0)
    plan = plan0 or ""
    traj = [list(S)]
    steps = []
    a_prev = None
    stagnated = False
    acquired = int(S == list(mstar))
    land_t = -1 if acquired else None
    t = 0
    while (not acquired) and t < max_steps_acq:
        if mk not in HARD:
            a, meta = p9j.choose_gamma_prime(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated
            )
        else:
            a, meta = p9j.choose_sink_seek(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, lam=lam
            )
        if a is None:
            break
        out, hit = p9j.apply_channel(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            mstar=mstar,
            channel=a,
            seed=seed + 1 + t,
            plan_prefill=plan,
            cache=cache,
        )
        Sa = list(out["S"])
        if out.get("plan_prefill"):
            plan = out["plan_prefill"]
        steps.append(
            {
                "t": t,
                "a": a,
                "S_before": list(S),
                "S_after": Sa,
                "mode": meta.get("mode"),
                "cache_hit": hit,
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj.append(list(S))
        t += 1
        if S == list(mstar):
            acquired = 1
            land_t = t - 1
            break
    return {
        "acquired": acquired,
        "land_t": land_t,
        "S": list(S),
        "plan": plan,
        "traj_S": traj,
        "steps": steps,
        "t_next": t,
        "path": [_key(s) for s in traj],
    }


def retain_phase(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    mstar,
    seed,
    S_start,
    plan0,
    t0,
    n_hold,
    mode: str,  # "hold" | "planner"
    cache,
) -> dict[str, Any]:
    """After acquire: hold (no steer) or planner (stay-seek)."""
    S = list(S_start)
    plan = plan0 or ""
    traj = [list(S)]
    steps = []
    a_prev = None
    stagnated = False
    stayed_all = int(S == list(mstar))
    t = t0
    for k in range(n_hold):
        if mode == "hold":
            out, hit = apply_free(
                p8k, sc, loaded, seed=seed + 1 + t, plan_prefill=plan, cache=cache
            )
            a = None
            meta = {"mode": "hold"}
        else:
            a, meta = choose_stay(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated
            )
            out, hit = p9j.apply_channel(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                mstar=mstar,
                channel=a,
                seed=seed + 1 + t,
                plan_prefill=plan,
                cache=cache,
            )
            if out.get("plan_prefill"):
                plan = out["plan_prefill"]
        Sa = list(out["S"])
        stayed = int(Sa == list(mstar))
        steps.append(
            {
                "k": k,
                "a": a,
                "S_before": list(S),
                "S_after": Sa,
                "stayed": stayed,
                "mode": meta.get("mode"),
                "cache_hit": hit,
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj.append(list(S))
        t += 1
        if not stayed:
            stayed_all = 0
            # continue remaining steps for full trajectory logging
    stayed_final = int(S == list(mstar))
    # primary retain: still at m* after all n_hold steps
    return {
        "mode": mode,
        "stay": stayed_final,
        "stay_all_steps": stayed_all and stayed_final,
        "traj_S": traj,
        "path": [_key(s) for s in traj],
        "steps": steps,
        "S_final": list(S),
        "actions": [s["a"] for s in steps],
    }


def run_paired_trial(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    mstar,
    seed,
    S0,
    plan0,
    max_steps_acq,
    n_hold,
    lam,
) -> dict[str, Any]:
    cache: dict = {}
    acq = acquire(
        p9j,
        p8k,
        sc,
        loaded,
        dirs=dirs,
        lookup=lookup,
        mstar=mstar,
        seed=seed,
        S0=S0,
        plan0=plan0,
        max_steps_acq=max_steps_acq,
        lam=lam,
        cache=cache,
    )
    out: dict[str, Any] = {
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "seed": seed,
        "S0": list(S0),
        "acquire": acq["acquired"],
        "acq_path": acq["path"],
        "acq_actions": [s["a"] for s in acq["steps"]],
        "land_t": acq["land_t"],
    }
    if not acq["acquired"]:
        out["hold"] = None
        out["planner"] = None
        return out

    # paired retain from same landing state/plan; separate caches for free vs steer
    # but share acq cache — use same seed offsets; hold uses FREE key
    for mode in ("hold", "planner"):
        # fresh retain cache keys differ by mode (FREE vs channel)
        ret = retain_phase(
            p9j,
            p8k,
            sc,
            loaded,
            dirs=dirs,
            lookup=lookup,
            mstar=mstar,
            seed=seed,
            S_start=acq["S"],
            plan0=acq["plan"],
            t0=acq["t_next"],
            n_hold=n_hold,
            mode=mode,
            cache=cache,
        )
        out[mode] = ret
    return out


def run_hold_controller_trial(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    mstar,
    seed,
    S0,
    plan0,
    max_steps_acq,
    n_hold,
    lam,
) -> dict[str, Any]:
    """Canonical π: acquire while s≠m*, hold while s=m*."""
    cache: dict = {}
    acq = acquire(
        p9j,
        p8k,
        sc,
        loaded,
        dirs=dirs,
        lookup=lookup,
        mstar=mstar,
        seed=seed,
        S0=S0,
        plan0=plan0,
        max_steps_acq=max_steps_acq,
        lam=lam,
        cache=cache,
    )
    if not acq["acquired"]:
        return {
            "mstar_key": _key(mstar),
            "acquire": 0,
            "final_ok": 0,
            "stay": None,
            "path": acq["path"],
            "actions": [s["a"] for s in acq["steps"]],
        }
    ret = retain_phase(
        p9j,
        p8k,
        sc,
        loaded,
        dirs=dirs,
        lookup=lookup,
        mstar=mstar,
        seed=seed,
        S_start=acq["S"],
        plan0=acq["plan"],
        t0=acq["t_next"],
        n_hold=n_hold,
        mode="hold",
        cache=cache,
    )
    return {
        "mstar_key": _key(mstar),
        "acquire": 1,
        "stay": ret["stay"],
        "final_ok": int(ret["stay"] == 1),
        "path": acq["path"] + ret["path"][1:],
        "acq_actions": [s["a"] for s in acq["steps"]],
        "hold_path": ret["path"],
    }


def _agg_diag(trials: list[dict]) -> dict[str, Any]:
    n = len(trials)
    n_acq = sum(t["acquire"] for t in trials)
    p_acq = n_acq / n if n else float("nan")
    hold_rows = [t["hold"] for t in trials if t["acquire"] and t.get("hold")]
    plan_rows = [t["planner"] for t in trials if t["acquire"] and t.get("planner")]
    p_hold = (
        float(np.mean([r["stay"] for r in hold_rows])) if hold_rows else float("nan")
    )
    p_plan = (
        float(np.mean([r["stay"] for r in plan_rows])) if plan_rows else float("nan")
    )
    p_final_hold = p_acq * p_hold if hold_rows else (0.0 if n_acq == 0 else float("nan"))
    p_final_plan = p_acq * p_plan if plan_rows else (0.0 if n_acq == 0 else float("nan"))
    return {
        "n": n,
        "n_acq": n_acq,
        "P_acquire": p_acq,
        "P_stay_hold": p_hold,
        "P_stay_planner": p_plan,
        "P_final_hold": p_final_hold,
        "P_final_planner": p_final_plan,
        "delta_stay_hold_minus_planner": (
            p_hold - p_plan if hold_rows and plan_rows else float("nan")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-steps-acq", type=int, default=7)
    ap.add_argument("--n-hold", type=int, default=2)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--full-8way",
        action="store_true",
        help="also run acquire+hold controller for all 8",
    )
    ap.add_argument(
        "--full-only",
        action="store_true",
        help="skip diagnostic; only acquire+hold 8-way",
    )
    ap.add_argument(
        "--auto-full",
        action="store_true",
        help="run full-8way if mean P_stay|hold > mean P_stay|planner",
    )
    args = ap.parse_args()

    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")

    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    dirs = p8k._load_vc()
    lookup, _, _ = p9j.build_lookup(p9e, p9g, None)

    diag_by_m: dict[str, list] = defaultdict(list)
    diag_paired = []
    per_diag = {}

    if not args.full_only:
        n_tot = 8 * args.reps
        idx = 0
        print(
            f"=== Phase 9M diagnostic hold vs planner "
            f"reps={args.reps} n_hold={args.n_hold} ===",
            flush=True,
        )
        for mi, mstar in enumerate(ALL_MSTAR):
            for r in range(args.reps):
                seed0 = args.seed + 200 * mi + 19 * r
                S0, plan0, seed = p9j.free_S0(
                    p8k, sc, loaded, mstar=mstar, seed0=seed0
                )
                idx += 1
                print(
                    f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)}",
                    flush=True,
                )
                if S0 == list(mstar):
                    print("    skip S0==m*", flush=True)
                    continue
                trial = run_paired_trial(
                    p9j,
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    lookup=lookup,
                    mstar=mstar,
                    seed=seed,
                    S0=S0,
                    plan0=plan0,
                    max_steps_acq=args.max_steps_acq,
                    n_hold=args.n_hold,
                    lam=args.lam,
                )
                diag_by_m[_key(mstar)].append(trial)
                diag_paired.append(trial)
                h = trial.get("hold")
                p = trial.get("planner")
                print(
                    f"    acq={trial['acquire']} "
                    f"hold_stay={None if not h else h['stay']} "
                    f"plan_stay={None if not p else p['stay']} "
                    f"acq_path={trial['acq_path']}",
                    flush=True,
                )
                if h:
                    print(f"      hold→{h['path']}", flush=True)
                if p:
                    print(f"      plan→{p['path']} acts={p['actions']}", flush=True)

        for m in ALL_MSTAR:
            mk = _key(m)
            per_diag[mk] = _agg_diag(diag_by_m[mk])
            per_diag[mk]["hard"] = tuple(m) in HARD

    # decide full
    stay_holds = [
        st["P_stay_hold"]
        for st in per_diag.values()
        if st.get("n_acq", 0) > 0 and st.get("P_stay_hold") == st.get("P_stay_hold")
    ]
    stay_plans = [
        st["P_stay_planner"]
        for st in per_diag.values()
        if st.get("n_acq", 0) > 0 and st.get("P_stay_planner") == st.get("P_stay_planner")
    ]
    mean_hold = float(np.mean(stay_holds)) if stay_holds else float("nan")
    mean_plan = float(np.mean(stay_plans)) if stay_plans else float("nan")
    hold_wins = (
        mean_hold == mean_hold
        and mean_plan == mean_plan
        and mean_hold > mean_plan + 1e-9
    )
    run_full = args.full_8way or args.full_only or (args.auto_full and hold_wins)

    full_by_m: dict[str, list] = defaultdict(list)
    full_trials = []
    per_full = {}

    if run_full:
        print(
            f"\n=== Phase 9M full acquire+hold "
            f"(hold_wins={hold_wins} mean_hold={mean_hold:.2f} "
            f"mean_plan={mean_plan:.2f}) ===",
            flush=True,
        )
        n_tot = 8 * args.reps
        idx = 0
        for mi, mstar in enumerate(ALL_MSTAR):
            for r in range(args.reps):
                # offset seed from diagnostic to avoid exact replay
                seed0 = args.seed + 9000 + 200 * mi + 19 * r
                S0, plan0, seed = p9j.free_S0(
                    p8k, sc, loaded, mstar=mstar, seed0=seed0
                )
                idx += 1
                print(
                    f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)}",
                    flush=True,
                )
                if S0 == list(mstar):
                    continue
                trial = run_hold_controller_trial(
                    p9j,
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    lookup=lookup,
                    mstar=mstar,
                    seed=seed,
                    S0=S0,
                    plan0=plan0,
                    max_steps_acq=args.max_steps_acq,
                    n_hold=args.n_hold,
                    lam=args.lam,
                )
                full_by_m[_key(mstar)].append(trial)
                full_trials.append(trial)
                print(
                    f"    acq={trial['acquire']} stay={trial['stay']} "
                    f"final_ok={trial['final_ok']} path={trial['path']}",
                    flush=True,
                )
        for m in ALL_MSTAR:
            mk = _key(m)
            rows = full_by_m[mk]
            n = len(rows)
            n_acq = sum(t["acquire"] for t in rows)
            ret = [t for t in rows if t["acquire"]]
            p_acq = n_acq / n if n else float("nan")
            p_stay = (
                float(np.mean([t["stay"] for t in ret])) if ret else float("nan")
            )
            p_final = sum(t["final_ok"] for t in rows) / n if n else float("nan")
            per_full[mk] = {
                "n": n,
                "P_acquire": p_acq,
                "P_stay_hold": p_stay,
                "P_final": p_final,
                "hard": tuple(m) in HARD,
            }

    n_acq_gt0 = sum(1 for st in per_diag.values() if (st.get("P_acquire") or 0) > 0)
    n_final_hold = sum(
        1 for st in per_diag.values() if (st.get("P_final_hold") or 0) > 0
    )
    n_final_plan = sum(
        1 for st in per_diag.values() if (st.get("P_final_planner") or 0) > 0
    )
    n_full_final = sum(
        1 for st in per_full.values() if (st.get("P_final") or 0) > 0
    )

    verdict = (
        "planner destroys targets (hold ≫ planner)"
        if hold_wins and mean_hold > 0.05
        else (
            "intrinsic instability (both near 0)"
            if (mean_hold != mean_hold or mean_hold < 0.05)
            and (mean_plan != mean_plan or mean_plan < 0.05)
            else "mixed / inconclusive"
        )
    )

    gate = {
        "hypothesis": (
            "After acquire, no-steer hold vs continued planner separates "
            "controller-induced loss from intrinsic instability"
        ),
        "mean_P_stay_hold": mean_hold,
        "mean_P_stay_planner": mean_plan,
        "hold_wins": hold_wins,
        "verdict": verdict,
        "n_mstar_acq_gt0": n_acq_gt0,
        "n_mstar_final_hold": n_final_hold,
        "n_mstar_final_planner": n_final_plan,
        "ran_full_8way": run_full,
        "n_mstar_final_full_hold": n_full_final if run_full else None,
        "universal_final_full": (n_full_final == 8) if run_full else None,
        "controller": (
            r"π(s,m*)=acquire if s≠m*; hold if s=m*"
        ),
        "read": (
            "If hold≫planner: stop acting at target. "
            "If both≈0: need retention mechanism beyond no-steer."
        ),
    }

    def _nan_safe(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {k: _nan_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_safe(x) for x in obj]
        return obj

    payload = {
        "protocol": "Phase 9M target hold vs planner retention",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps_acq": args.max_steps_acq,
        "n_hold": args.n_hold,
        "lambda": args.lam,
        "diagnostic_per_mstar": per_diag,
        "diagnostic_paired": diag_paired,
        "full_per_mstar": per_full,
        "full_trials": full_trials,
        "gate": gate,
    }
    OUT.write_text(
        json.dumps(_nan_safe(payload), indent=2, default=str), encoding="utf-8"
    )

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9M — Target hold vs planner",
        "",
        r"> After acquire: **hold** (no steer) vs **planner** (stay-seek). "
        r"Frozen hybrid acquisition. No new $v$.",
        "",
        f"reps={args.reps}, acq_steps={args.max_steps_acq}, "
        f"n_hold={args.n_hold}, seed={args.seed}.",
        "",
        r"$$\pi(s,m^*)=\mathrm{acquire}\ [s\neq m^*];\quad \mathrm{hold}\ [s=m^*]$$",
        "",
    ]
    if per_diag:
        lines += [
            "## Diagnostic: $P_{stay}\\mid acq$",
            "",
            "| $m^*$ | $P_{acq}$ | $P_{stay}$ hold | $P_{stay}$ planner | "
            "$\\Delta$ (hold$-$plan) | $P_{final}$ hold | $P_{final}$ plan |",
            "|-------|----------:|-----------------:|-------------------:|"
            "-----------------------:|-----------------:|-----------------:|",
        ]
        for m in ALL_MSTAR:
            mk = _key(m)
            st = per_diag[mk]
            lines.append(
                f"| `{mk}` | {fmt(st.get('P_acquire'))} | "
                f"**{fmt(st.get('P_stay_hold'))}** | {fmt(st.get('P_stay_planner'))} | "
                f"{fmt(st.get('delta_stay_hold_minus_planner'))} | "
                f"{fmt(st.get('P_final_hold'))} | {fmt(st.get('P_final_planner'))} |"
            )
        lines += [
            "",
            f"mean $P_{{stay}}$ hold / planner: **{fmt(mean_hold)}** / {fmt(mean_plan)}",
            f"Verdict: **{verdict}**",
            "",
        ]
    if run_full and per_full:
        lines += [
            "## Full 8-way: acquire + hold",
            "",
            "| $m^*$ | $P_{acq}$ | $P_{stay}$ | $P_{final}$ |",
            "|-------|----------:|-----------:|------------:|",
        ]
        for m in ALL_MSTAR:
            mk = _key(m)
            st = per_full[mk]
            lines.append(
                f"| `{mk}` | {fmt(st.get('P_acquire'))} | "
                f"{fmt(st.get('P_stay_hold'))} | **{fmt(st.get('P_final'))}** |"
            )
        lines += [
            "",
            f"$m^*$ with $P_{{\\mathrm{{final}}}}>0$: **{n_full_final}/8**",
            f"Universal final: **{gate['universal_final_full']}**",
            "",
        ]
    lines += [
        "## Gate",
        "",
        f"- hold ≫ planner: **{hold_wins}**",
        f"- ran acquire+hold 8-way: **{run_full}**",
        "",
        gate["read"],
        "",
        r"$$\boxed{s=m^*\Rightarrow\mathrm{hold};\quad s\neq m^*\Rightarrow\mathrm{acquire}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": _nan_safe(gate)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
