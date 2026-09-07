#!/usr/bin/env python3
"""Phase 9R — Hybrid Γ' + bidirectional beam fallback; full 8-way acq×ret.

9Q showed hard sinks are reachable via ±C/±H/±O rollouts when polarity is
not forced to target-sign. This phase builds:

  π(s,m*) = Γ'(s,m*)           if max Γ' ≥ τ and action not predicted no-op
           beam(±C/±H/±O)      otherwise (or after observed no-op)

Arms (paired same S0):
  A) gamma_prime only
  B) gamma_prime_beam (hybrid)

Metrics: P_acq, P_ret|acq, P_final = P_acq·P_ret for all 8 m*.
Also: P(successful transition uses target sign) vs opposite sign.

No new v. Frozen 8K actuators.

  .venv/bin/python -u scripts/run_phase9r_hybrid.py --reps 2 --tau-gamma 0.0
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

OUT = ROOT / "data" / "results" / "sync_phase9r_hybrid.json"
MD = ROOT / "data" / "results" / "sync_phase9r_hybrid.md"
SEED = 20261006
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
HARD = {(0, 0, 0), (0, 0, 1), (1, 1, 0)}
ARMS = ("gamma_prime", "gamma_prime_beam")


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required before dataclass exec
    spec.loader.exec_module(mod)
    return mod


def _key(s) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _E(mstar, S) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _parse_act(act: str) -> tuple[str, int]:
    """'+C' → ('C', +1)."""
    return act[1], (+1 if act[0] == "+" else -1)


def _target_sign(p8k, mstar, channel: str) -> int:
    return int(p8k._s_star(mstar[CH_IDX[channel]]))


def choose_stay_or_fallback(s, mstar, lookup, *, a_prev, stagnated):
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
    meta["a_star"] = ORDER[0]
    return ORDER[0], meta


def should_beam(
    *,
    a: str | None,
    meta: dict,
    lookup,
    S,
    mstar,
    tau: float,
    stagnated: bool,
) -> tuple[bool, str]:
    """Return (need_beam, reason)."""
    if stagnated:
        return True, "observed_noop"
    if a is None:
        return True, "no_gamma_action"
    if meta.get("fallback"):
        return True, "gamma_fallback"
    g = meta.get("Gamma_star")
    if g is None:
        return True, "gamma_missing"
    if float(g) < float(tau):
        return True, "gamma_below_tau"
    st = lookup.get((_key(S), a, _key(mstar))) or {}
    n = int(st.get("n") or 0)
    ps = st.get("P_shift")
    if n >= 1 and ps is not None and float(ps) <= 0.0:
        return True, "predicted_noop"
    return False, ""


def _sign_record(p8k, mstar, channel: str, applied_sign: int, moved: bool, land: bool):
    if not (moved or land):
        return None
    ts = _target_sign(p8k, mstar, channel)
    return {
        "channel": channel,
        "applied_sign": int(applied_sign),
        "target_sign": int(ts),
        "matches_target_sign": int(applied_sign == ts),
        "opposite_sign": int(applied_sign == -ts),
        "land": int(land),
        "moved": int(moved),
    }


def run_trial(
    p9j,
    p9q,
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
    arm: str,
    tau: float,
    beam: int,
    horizon: int,
    max_steps_acq: int,
    n_retain: int,
    cache: dict,
    beam_cache: dict,
) -> dict[str, Any]:
    S = list(S0)
    plan = plan0 or ""
    traj_S = [list(S)]
    steps: list[dict] = []
    sign_rows: list[dict] = []
    a_prev = None
    stagnated = False
    acquired = int(S == list(mstar))
    land_t = -1 if acquired else None
    a_arrive = None
    n_beam_calls = 0
    beam_reasons: list[str] = []

    t = 0
    while (not acquired) and t < max_steps_acq:
        a, meta = p9j.choose_gamma_prime(
            S, mstar, lookup, a_prev=a_prev, stagnated=stagnated
        )
        use_beam = False
        reason = ""
        if arm == "gamma_prime_beam":
            use_beam, reason = should_beam(
                a=a,
                meta=meta,
                lookup=lookup,
                S=S,
                mstar=mstar,
                tau=tau,
                stagnated=stagnated,
            )

        if use_beam:
            n_beam_calls += 1
            beam_reasons.append(reason)
            # Budget remaining as horizon cap
            H = min(horizon, max_steps_acq - t)
            if H < 1:
                break
            bs = p9q.beam_search(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                mstar=mstar,
                S0=S,
                plan0=plan,
                seed0=seed + 100 + t,
                beam=beam,
                horizon=H,
                cache=beam_cache,
            )
            acts = list(bs.get("acts") or [])
            traj = [list(x) for x in (bs.get("traj") or [])]
            if not traj:
                traj = [list(S)]
                if bs.get("S_final") is not None:
                    traj.append(list(bs["S_final"]))
            if not acts:
                steps.append(
                    {
                        "t": t,
                        "phase": "acquire",
                        "mode": "beam",
                        "beam_reason": reason,
                        "a": None,
                        "S_before": list(S),
                        "S_after": list(S),
                        "dS": 0,
                        "first_land": False,
                        "empty_beam": True,
                        "n_expansions": bs.get("n_expansions"),
                    }
                )
                stagnated = True
                t += 1
                continue

            # Consume beam rollouts directly (already executed inside beam_search)
            for i, act in enumerate(acts):
                if i + 1 >= len(traj):
                    break
                ch, sign = _parse_act(act)
                S_before = list(traj[i])
                Sa = list(traj[i + 1])
                first_land = Sa == list(mstar)
                moved = Sa != S_before
                rec = _sign_record(p8k, mstar, ch, sign, moved, first_land)
                if rec:
                    rec["source"] = "beam"
                    sign_rows.append(rec)
                steps.append(
                    {
                        "t": t,
                        "phase": "acquire",
                        "mode": "beam",
                        "beam_reason": reason,
                        "a": act,
                        "channel": ch,
                        "sign": sign,
                        "S_before": S_before,
                        "S_after": Sa,
                        "dS": int(moved),
                        "first_land": first_land,
                        "matches_target_sign": None
                        if not rec
                        else rec["matches_target_sign"],
                        "n_expansions": bs.get("n_expansions"),
                    }
                )
                traj_S.append(list(Sa))
                t += 1
                if first_land:
                    acquired = 1
                    land_t = t - 1
                    a_arrive = act
                    a_prev = None
                    stagnated = False
                    break
                if t >= max_steps_acq:
                    break

            S = list(traj[min(len(acts), len(traj) - 1)])
            if bs.get("found") and list(bs["S_final"]) == list(mstar):
                S = list(bs["S_final"])
                acquired = 1
                if land_t is None:
                    land_t = t - 1
                a_arrive = acts[-1] if acts else a_arrive
                a_prev = None
                stagnated = False
            plan = bs.get("plan_final") or plan
            if not acquired:
                stagnated = bool(steps) and steps[-1].get("dS", 1) == 0
                a_prev = acts[-1] if acts else a_prev
            continue

        # --- Γ' step (Arm A always; Arm B when evidence OK) ---
        if a is None:
            break
        applied_sign = _target_sign(p8k, mstar, a)
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
        moved = Sa != S
        rec = _sign_record(p8k, mstar, a, applied_sign, moved, first_land)
        if rec:
            rec["source"] = "gamma_prime"
            sign_rows.append(rec)
        steps.append(
            {
                "t": t,
                "phase": "acquire",
                "mode": "gamma_prime",
                "a": a,
                "channel": a,
                "sign": applied_sign,
                "Gamma_star": meta.get("Gamma_star"),
                "S_before": list(S),
                "S_after": Sa,
                "dS": int(moved),
                "first_land": first_land,
                "cache_hit": hit,
                "matches_target_sign": 1 if rec else None,
            }
        )
        stagnated = not moved
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

    # --- retain (same stay-seek for both arms; target-signed via apply_channel) ---
    retained = None
    if acquired:
        for _ in range(n_retain):
            if S != list(mstar):
                retained = 0
                break
            a, meta = choose_stay_or_fallback(
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
            Sa = list(out["S"])
            if out.get("plan_prefill"):
                plan = out["plan_prefill"]
            stayed = int(Sa == list(mstar))
            steps.append(
                {
                    "t": t,
                    "phase": "retain",
                    "mode": "stay_seek",
                    "a": a,
                    "S_before": list(S),
                    "S_after": Sa,
                    "dS": int(Sa != S),
                    "stayed": stayed,
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

    n_match = sum(1 for r in sign_rows if r.get("matches_target_sign") == 1)
    n_opp = sum(1 for r in sign_rows if r.get("opposite_sign") == 1)
    n_succ = len(sign_rows)

    return {
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "arm": arm,
        "hard": tuple(mstar) in HARD,
        "seed": seed,
        "S0": list(S0),
        "S_final": list(S),
        "path": [_key(s) for s in traj_S],
        "steps": steps,
        "actions": [s.get("a") for s in steps],
        "acquire": acquired,
        "retain": retained,
        "land_t": land_t,
        "a_arrive": a_arrive,
        "hit_final": int(S == list(mstar)),
        "final_ok": int(bool(acquired) and retained == 1),
        "n_beam_calls": n_beam_calls,
        "beam_reasons": beam_reasons,
        "sign_rows": sign_rows,
        "n_succ_trans": n_succ,
        "n_target_sign": n_match,
        "n_opposite_sign": n_opp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-steps-acq", type=int, default=7)
    ap.add_argument("--n-retain", type=int, default=2)
    ap.add_argument("--tau-gamma", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
    p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
    p9q = _load_mod("phase9q", ROOT / "scripts" / "run_phase9q_bidir_beam.py")

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
    lookup, _ker, _rows = p9j.build_lookup(p9e, p9g, None)

    by_arm_m: dict[str, dict[str, list]] = {arm: defaultdict(list) for arm in ARMS}
    paired = []
    n_tot = len(ALL_MSTAR) * args.reps
    idx = 0

    print(
        f"=== Phase 9R hybrid Γ'+beam τ={args.tau_gamma} "
        f"K={args.beam} T={args.horizon} reps={args.reps} ===",
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
            beam_cache: dict = {}
            trials = {}
            for arm in ARMS:
                print(f"    arm={arm} ...", flush=True)
                tr = run_trial(
                    p9j,
                    p9q,
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    lookup=lookup,
                    mstar=mstar,
                    seed=seed,
                    S0=S0,
                    plan0=plan0,
                    arm=arm,
                    tau=args.tau_gamma,
                    beam=args.beam,
                    horizon=args.horizon,
                    max_steps_acq=args.max_steps_acq,
                    n_retain=args.n_retain,
                    cache=cache,
                    beam_cache=beam_cache,
                )
                trials[arm] = tr
                by_arm_m[arm][_key(mstar)].append(tr)
                print(
                    f"      acq={tr['acquire']} ret={tr['retain']} "
                    f"final={tr['final_ok']} beam_calls={tr['n_beam_calls']} "
                    f"path={tr['path']}",
                    flush=True,
                )
            paired.append(
                {
                    "mstar": list(mstar),
                    "mstar_key": _key(mstar),
                    "seed": seed,
                    "S0": list(S0),
                    "plan0": plan0,
                    "trials": {
                        arm: {
                            k: trials[arm][k]
                            for k in (
                                "acquire",
                                "retain",
                                "final_ok",
                                "path",
                                "actions",
                                "n_beam_calls",
                                "beam_reasons",
                                "a_arrive",
                                "n_target_sign",
                                "n_opposite_sign",
                                "n_succ_trans",
                            )
                        }
                        for arm in ARMS
                    },
                }
            )

    def summarize(arm: str) -> dict:
        per = {}
        all_sign = []
        for m in ALL_MSTAR:
            mk = _key(m)
            rows = by_arm_m[arm][mk]
            n = len(rows)
            n_acq = sum(int(t["acquire"]) for t in rows)
            ret_rows = [t for t in rows if t["acquire"]]
            n_ret = sum(1 for t in ret_rows if t["retain"] == 1)
            p_acq = n_acq / n if n else float("nan")
            p_ret = n_ret / len(ret_rows) if ret_rows else float("nan")
            p_final = (
                (p_acq * p_ret)
                if (n and ret_rows and p_ret == p_ret)
                else (0.0 if n_acq == 0 else float("nan"))
            )
            if n_acq == 0:
                p_final = 0.0
            p_final_emp = sum(int(t["final_ok"]) for t in rows) / n if n else float("nan")
            for t in rows:
                all_sign.extend(t.get("sign_rows") or [])
            per[mk] = {
                "n": n,
                "P_acq": p_acq,
                "P_ret": p_ret,
                "P_final": p_final,
                "P_final_empirical": p_final_emp,
                "n_acq": n_acq,
                "n_ret": n_ret,
                "mean_beam_calls": float(
                    np.mean([t["n_beam_calls"] for t in rows])
                )
                if rows
                else 0.0,
            }
        n_succ = len(all_sign)
        n_ts = sum(1 for r in all_sign if r.get("matches_target_sign") == 1)
        n_os = sum(1 for r in all_sign if r.get("opposite_sign") == 1)
        beam_sign = [r for r in all_sign if r.get("source") == "beam"]
        return {
            "per_mstar": per,
            "n_mstar_acq": sum(
                1 for mk, st in per.items() if (st.get("P_acq") or 0) > 0
            ),
            "n_mstar_final": sum(
                1 for mk, st in per.items() if (st.get("P_final") or 0) > 0
            ),
            "sign_stats": {
                "n_successful_transitions": n_succ,
                "P_target_sign": n_ts / n_succ if n_succ else float("nan"),
                "P_opposite_sign": n_os / n_succ if n_succ else float("nan"),
                "n_beam_successful": len(beam_sign),
                "P_beam_opposite_sign": (
                    sum(1 for r in beam_sign if r.get("opposite_sign") == 1)
                    / len(beam_sign)
                    if beam_sign
                    else float("nan")
                ),
            },
        }

    summaries = {arm: summarize(arm) for arm in ARMS}

    # Paired deltas
    paired_delta = []
    for p in paired:
        mk = p["mstar_key"]
        a = p["trials"]["gamma_prime"]
        b = p["trials"]["gamma_prime_beam"]
        paired_delta.append(
            {
                "mstar_key": mk,
                "d_acq": int(b["acquire"]) - int(a["acquire"]),
                "d_final": int(b["final_ok"]) - int(a["final_ok"]),
            }
        )

    gate = {
        "hypothesis": (
            "Γ'+beam fallback converts 9Q hard reachability into "
            "8/8 target-conditioned acquisition under frozen actuators"
        ),
        "tau_gamma": args.tau_gamma,
        "beam": args.beam,
        "horizon": args.horizon,
        "n_acq_states": {
            arm: summaries[arm]["n_mstar_acq"] for arm in ARMS
        },
        "n_final_states": {
            arm: summaries[arm]["n_mstar_final"] for arm in ARMS
        },
        "acq_gate_hybrid": summaries["gamma_prime_beam"]["n_mstar_acq"] == 8,
        "final_gate_hybrid": summaries["gamma_prime_beam"]["n_mstar_final"] == 8,
        "read": (
            "Primary: hybrid P_acq>0 on all 8. Secondary: P_final>0 on all 8. "
            "Contrast Γ' alone. Report opposite-sign rate on successful transitions."
        ),
    }

    payload = {
        "protocol": "Phase 9R hybrid Γ' + beam fallback; paired 8-way acq×ret",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps_acq": args.max_steps_acq,
        "n_retain": args.n_retain,
        "tau_gamma": args.tau_gamma,
        "beam": args.beam,
        "horizon": args.horizon,
        "arms": list(ARMS),
        "summaries": summaries,
        "paired": paired,
        "paired_delta": paired_delta,
        "gate": gate,
        "by_arm_trials": {
            arm: {mk: by_arm_m[arm][mk] for mk in [_key(m) for m in ALL_MSTAR]}
            for arm in ARMS
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{float(x):.{nd}f}"

    lines = [
        "# Phase 9R — Hybrid $\\Gamma'$ + beam fallback (8-way acq×ret)",
        "",
        "> Paired $S_0$. No new $v$. Beam over $\\pm C,\\pm H,\\pm O$ when "
        f"$\\max\\Gamma'<\\tau$ ($\\tau={args.tau_gamma}$), predicted no-op, "
        "or observed no-op.",
        "",
        f"reps={args.reps}, max_steps_acq={args.max_steps_acq}, "
        f"n_retain={args.n_retain}, beam K={args.beam} T={args.horizon}, "
        f"seed={args.seed}.",
        "",
        "## $\\Gamma'$ alone",
        "",
        "| $m^*$ | $P_{acq}$ | $P_{ret}$ | $P_{final}$ |",
        "|-------|----------:|----------:|------------:|",
    ]
    for m in ALL_MSTAR:
        mk = _key(m)
        st = summaries["gamma_prime"]["per_mstar"][mk]
        lines.append(
            f"| `{mk}` | {fmt(st['P_acq'])} | {fmt(st['P_ret'])} | "
            f"**{fmt(st['P_final'])}** |"
        )
    sg = summaries["gamma_prime"]["sign_stats"]
    lines += [
        "",
        f"States with $P_{{acq}}>0$: **{summaries['gamma_prime']['n_mstar_acq']}/8**; "
        f"$P_{{final}}>0$: **{summaries['gamma_prime']['n_mstar_final']}/8**.",
        f"Successful transitions target-sign rate: {fmt(sg.get('P_target_sign'))}.",
        "",
        "## $\\Gamma'$ + beam fallback",
        "",
        "| $m^*$ | $P_{acq}$ | $P_{ret}$ | $P_{final}$ | mean beam calls |",
        "|-------|----------:|----------:|------------:|-----------------:|",
    ]
    for m in ALL_MSTAR:
        mk = _key(m)
        st = summaries["gamma_prime_beam"]["per_mstar"][mk]
        lines.append(
            f"| `{mk}` | {fmt(st['P_acq'])} | {fmt(st['P_ret'])} | "
            f"**{fmt(st['P_final'])}** | {fmt(st['mean_beam_calls'])} |"
        )
    sb = summaries["gamma_prime_beam"]["sign_stats"]
    lines += [
        "",
        f"States with $P_{{acq}}>0$: **{summaries['gamma_prime_beam']['n_mstar_acq']}/8**; "
        f"$P_{{final}}>0$: **{summaries['gamma_prime_beam']['n_mstar_final']}/8**.",
        "",
        "### Sign vs target-sign (successful transitions)",
        "",
        f"- $P(\\text{{target sign}})$ = **{fmt(sb.get('P_target_sign'))}**",
        f"- $P(\\text{{opposite sign}})$ = **{fmt(sb.get('P_opposite_sign'))}**",
        f"- Among beam-sourced successes, "
        f"$P(\\text{{opposite}})$ = **{fmt(sb.get('P_beam_opposite_sign'))}** "
        f"(n={sb.get('n_beam_successful')})",
        "",
        f"Acq gate (hybrid 8/8): **{gate['acq_gate_hybrid']}**. "
        f"Final gate (hybrid 8/8): **{gate['final_gate_hybrid']}**.",
        "",
        gate["read"],
        "",
        r"$$\boxed{\pi=\Gamma'\ \text{or}\ \mathrm{beam}(\pm v_C,\pm v_H,\pm v_O)}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
