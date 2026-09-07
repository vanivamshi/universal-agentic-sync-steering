#!/usr/bin/env python3
"""Phase 9L — Reliable attainment: P_acquire × P_retain for all 8 m*.

Milestone so far: all 8 states reachable. This phase measures the proper
8-way objective under frozen actuators + canonical hybrid planner
(Γ' on soft, all-channel sink-seek λ=0 on hard):

  P_final(m*) = P_acquire(m*) · P_retain(m*|acquire)

Acquire: ∃t S_t = m* within max_steps_acq.
Retain: after first landing, run n_retain stay-seeking steps
  a = argmax_a P(S'=m*|s,a); success if still at m* after those steps.

No new v.

  .venv/bin/python -u scripts/run_phase9l_acquire_retain.py --reps 4
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

OUT = ROOT / "data" / "results" / "sync_phase9l_acquire_retain.json"
MD = ROOT / "data" / "results" / "sync_phase9l_acquire_retain.md"
SEED = 20260931
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


def choose_stay(s, mstar, lookup, *, a_prev, stagnated):
    """Stay-seek: argmax P(S'=m*|s,a) over all channels, with anti-stag."""
    return choose_stay_or_fallback(
        s, mstar, lookup, a_prev=a_prev, stagnated=stagnated, p9j=None
    )


def choose_stay_or_fallback(s, mstar, lookup, *, a_prev, stagnated, p9j):
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
    # fallback: prefer no-op-ish — try blocked skip then O
    for a in ORDER:
        if stagnated and a == a_prev:
            continue
        meta["fallback"] = True
        meta["a_star"] = a
        return a, meta
    meta["a_star"] = ORDER[0]
    return ORDER[0], meta


def run_acquire_retain(
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
    max_steps_acq: int,
    n_retain: int,
    lam: float,
    cache: dict,
) -> dict[str, Any]:
    """Hybrid acquire, then stay-seek retain for n_retain steps."""
    hard_set = HARD
    mk = tuple(mstar)

    S = list(S0)
    plan = plan0 or ""
    traj_S = [list(S)]
    steps = []
    a_prev = None
    stagnated = False
    acquired = int(S == list(mstar))
    land_t = -1 if acquired else None
    a_arrive = None

    t = 0
    # --- acquire ---
    while (not acquired) and t < max_steps_acq:
        if mk not in hard_set:
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
        first_land = Sa == list(mstar)
        steps.append(
            {
                "t": t,
                "phase": "acquire",
                "a": a,
                "S_before": list(S),
                "S_after": Sa,
                "dS": int(Sa != S),
                "mode": meta.get("mode"),
                "first_land": first_land,
                "cache_hit": hit,
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj_S.append(list(S))
        t += 1
        if first_land:
            acquired = 1
            land_t = t - 1
            a_arrive = a
            a_prev = None
            stagnated = False
            break

    # --- retain ---
    retained = None
    if acquired:
        for _ in range(n_retain):
            if S != list(mstar):
                retained = 0
                break
            a, meta = choose_stay_or_fallback(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, p9j=p9j
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
            Sa = list(out["S"])
            if out.get("plan_prefill"):
                plan = out["plan_prefill"]
            stayed = int(Sa == list(mstar))
            steps.append(
                {
                    "t": t,
                    "phase": "retain",
                    "a": a,
                    "S_before": list(S),
                    "S_after": Sa,
                    "dS": int(Sa != S),
                    "stayed": stayed,
                    "mode": meta.get("mode"),
                    "P_hit_star": meta.get("P_hit_star"),
                    "cache_hit": hit,
                }
            )
            stagnated = Sa == S
            a_prev = a
            S = Sa
            traj_S.append(list(S))
            t += 1
            if not stayed:
                retained = 0
                break
        else:
            retained = int(S == list(mstar))

    path = [_key(s) for s in traj_S]
    return {
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "policy": "sink_seek" if mk in hard_set else "gamma_prime",
        "seed": seed,
        "S0": list(S0),
        "S_final": list(S),
        "path": path,
        "steps": steps,
        "actions": [s["a"] for s in steps],
        "acquire": acquired,
        "retain": retained,
        "land_t": land_t,
        "a_arrive": a_arrive,
        "n_retain_steps": n_retain,
        "hit_final": int(S == list(mstar)),
        "final_ok": int(bool(acquired) and retained == 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-steps-acq", type=int, default=7)
    ap.add_argument("--n-retain", type=int, default=2)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
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

    by_m: dict[str, list] = defaultdict(list)
    all_trials = []
    n_tot = 8 * args.reps
    idx = 0

    print(
        f"=== Phase 9L acquire×retain reps={args.reps} "
        f"acq_steps={args.max_steps_acq} n_retain={args.n_retain} λ={args.lam} ===",
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
                f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                f"E0={_E(mstar, S0)}",
                flush=True,
            )
            if S0 == list(mstar):
                print("    skip S0==m*", flush=True)
                continue
            cache: dict = {}
            trial = run_acquire_retain(
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
                n_retain=args.n_retain,
                lam=args.lam,
                cache=cache,
            )
            by_m[_key(mstar)].append(trial)
            all_trials.append(trial)
            print(
                f"    acq={trial['acquire']} ret={trial['retain']} "
                f"final_ok={trial['final_ok']} "
                f"path={trial['path']} acts={trial['actions']}",
                flush=True,
            )

    per = {}
    for m in ALL_MSTAR:
        mk = _key(m)
        rows = by_m[mk]
        n = len(rows)
        n_acq = sum(t["acquire"] for t in rows)
        ret_rows = [t for t in rows if t["acquire"]]
        n_ret = sum(1 for t in ret_rows if t["retain"] == 1)
        p_acq = n_acq / n if n else float("nan")
        p_ret = n_ret / len(ret_rows) if ret_rows else float("nan")
        p_final = (
            p_acq * p_ret
            if ret_rows and p_acq == p_acq and p_ret == p_ret
            else (0.0 if n_acq == 0 and n else float("nan"))
        )
        # also empirical P(final_ok)
        p_final_emp = (
            sum(t["final_ok"] for t in rows) / n if n else float("nan")
        )
        per[mk] = {
            "n": n,
            "hard": tuple(m) in HARD,
            "policy": "sink_seek" if tuple(m) in HARD else "gamma_prime",
            "P_acquire": p_acq,
            "P_retain_given_acq": p_ret,
            "P_final_product": p_final,
            "P_final_empirical": p_final_emp,
            "n_acquire": n_acq,
            "n_retain_ok": n_ret,
            "path_examples": [
                {
                    "acq": t["acquire"],
                    "ret": t["retain"],
                    "path": t["path"],
                    "acts": t["actions"],
                }
                for t in rows[:4]
            ],
        }

    n_reach = sum(1 for mk, st in per.items() if (st.get("P_acquire") or 0) > 0)
    n_stable = sum(
        1
        for mk, st in per.items()
        if (st.get("P_final_empirical") or 0) > 0
        or (st.get("P_final_product") or 0) > 0
    )
    hard_acq = {mk: per[mk]["P_acquire"] for mk in ("000", "001", "110")}
    hard_ret = {mk: per[mk]["P_retain_given_acq"] for mk in ("000", "001", "110")}
    hard_fin = {mk: per[mk]["P_final_empirical"] for mk in ("000", "001", "110")}

    gate = {
        "hypothesis": (
            "P_final = P_acquire · P_retain is the proper 8-way objective; "
            "hard sinks fail on retain (and low acquire), not missing actuators"
        ),
        "n_mstar_acquire_gt0": n_reach,
        "n_mstar_final_gt0": n_stable,
        "universal_acquire": n_reach == 8,
        "universal_final": n_stable == 8,
        "hard_P_acquire": hard_acq,
        "hard_P_retain": hard_ret,
        "hard_P_final": hard_fin,
        "mean_P_acquire": float(
            np.mean([st["P_acquire"] for st in per.values() if st["n"]])
        ),
        "mean_P_final": float(
            np.nanmean(
                [
                    st["P_final_empirical"]
                    for st in per.values()
                    if st["n"] and st["P_final_empirical"] == st["P_final_empirical"]
                ]
            )
        ),
        "read": (
            "Primary: per-m* P_acquire, P_retain|acq, P_final. "
            "Reliable 8-way needs ∀m* P_final>0 (ideally ≫0)."
        ),
    }

    payload = {
        "protocol": "Phase 9L P_acquire × P_retain (hybrid planner)",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps_acq": args.max_steps_acq,
        "n_retain": args.n_retain,
        "lambda": args.lam,
        "per_mstar": per,
        "trials": all_trials,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9L — $P_{\\mathrm{acquire}}\\times P_{\\mathrm{retain}}$",
        "",
        "> Hybrid: $\\Gamma'$ soft, all-channel sink-seek ($\\lambda=0$) hard. "
        f"Retain = {args.n_retain}-step stay-seek after first landing. Frozen 8K.",
        "",
        f"reps={args.reps}, acq_steps={args.max_steps_acq}, "
        f"n_retain={args.n_retain}, seed={args.seed}.",
        "",
        "## Per $m^*$",
        "",
        "| $m^*$ | pol | n | $P_{acq}$ | $P_{ret}\\mid acq$ | "
        "$P_{final}$ (prod) | $P_{final}$ (emp) |",
        "|-------|-----|--:|----------:|-------------------:|"
        "-------------------:|-------------------:|",
    ]
    for m in ALL_MSTAR:
        mk = _key(m)
        st = per[mk]
        tag = "SS" if st["hard"] else "Γ'"
        lines.append(
            f"| `{mk}` | {tag} | {st['n']} | **{fmt(st['P_acquire'])}** | "
            f"{fmt(st['P_retain_given_acq'])} | {fmt(st['P_final_product'])} | "
            f"**{fmt(st['P_final_empirical'])}** |"
        )
    lines += [
        "",
        f"$m^*$ with $P_{{acq}}>0$: **{n_reach}/8**. "
        f"$m^*$ with $P_{{\\mathrm{{final}}}}>0$: **{n_stable}/8**.",
        "",
        "### Hard sinks",
        "",
        f"- $P_{{acq}}$: `{hard_acq}`",
        f"- $P_{{ret}}\\mid acq$: `{hard_ret}`",
        f"- $P_{{\\mathrm{{final}}}}$: `{hard_fin}`",
        "",
        "## Gate",
        "",
        f"- Universal acquire: **{gate['universal_acquire']}**",
        f"- Universal final (retain): **{gate['universal_final']}**",
        f"- mean $P_{{acq}}$ / mean $P_{{\\mathrm{{final}}}}$: "
        f"{fmt(gate['mean_P_acquire'])} / {fmt(gate['mean_P_final'])}",
        "",
        gate["read"],
        "",
        r"$$\boxed{P_{\mathrm{final}}=P_{\mathrm{acquire}}\cdot P_{\mathrm{retain}}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
