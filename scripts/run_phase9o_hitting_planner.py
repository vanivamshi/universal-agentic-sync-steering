#!/usr/bin/env python3
"""Phase 9O — Finite-horizon target-hitting planner (hard sinks).

9N JIT-RH optimizes local remaining bits; hard targets need temporary E↑ moves.
Use empirical kernel to maximize hitting probability:

  V_0(s) = 1[s=m*]
  V_k(s) = 1[s=m*]  if s=m*
           max_a E_{s'|s,a}[V_{k-1}(s')]  else

  a* = argmax_a E[V_{T-1}(s')]   (receding horizon)

Hard targets only: {000,001,110}. Compare live vs Γ'.
No new v. Temporary E↑ allowed when it raises hit prob.

  .venv/bin/python -u scripts/run_phase9o_hitting_planner.py --reps 4 --T 3
  .venv/bin/python -u scripts/run_phase9o_hitting_planner.py --offline-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9o_hitting_planner.json"
MD = ROOT / "data" / "results" / "sync_phase9o_hitting_planner.md"
SEED = 20261003
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
STATES = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
HARD = [(0, 0, 0), (0, 0, 1), (1, 1, 0)]
MIN_N = 1


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


def _parse(sk: str) -> tuple[int, int, int]:
    return (int(sk[0]), int(sk[1]), int(sk[2]))


def build_trans_probs(ker) -> dict[tuple[str, str], dict[str, float]]:
    """(s_key, a) -> {s'_key: p}."""
    out = {}
    for (sk, a), outs in ker.items():
        n = len(outs)
        if n < MIN_N:
            continue
        hist = Counter(_key(sa) for sa in outs)
        out[(sk, a)] = {spk: c / n for spk, c in hist.items()}
    return out


def compute_V(trans, mstar, T: int):
    """Value iteration for P(hit m* within k steps), k=0..T."""
    mk = _key(mstar)
    V = [{_key(s): (1.0 if _key(s) == mk else 0.0) for s in STATES}]
    policy = []  # policy[k][s] = a* using V[k] after one step toward V[k]... 
    # V[k] = value with k steps remaining
    for k in range(1, T + 1):
        Vk = {}
        pk = {}
        for s in STATES:
            sk = _key(s)
            if sk == mk:
                Vk[sk] = 1.0
                pk[sk] = None
                continue
            best_a, best_v = None, -1.0
            for a in ORDER:
                dist = trans.get((sk, a))
                if not dist:
                    continue
                ev = sum(p * V[k - 1].get(spk, 0.0) for spk, p in dist.items())
                p_hit = dist.get(mk, 0.0)
                better = False
                if best_a is None or ev > best_v + 1e-12:
                    better = True
                elif abs(ev - best_v) <= 1e-12:
                    p_hit_b = (trans.get((sk, best_a)) or {}).get(mk, 0.0)
                    if p_hit > p_hit_b + 1e-12:
                        better = True
                    elif abs(p_hit - p_hit_b) <= 1e-12 and ORDER.index(a) < ORDER.index(best_a):
                        better = True
                if better:
                    best_v, best_a = ev, a
            if best_a is None:
                Vk[sk] = 0.0
                pk[sk] = None
            else:
                Vk[sk] = float(best_v)
                pk[sk] = best_a
        V.append(Vk)
        policy.append(pk)
    return V, policy


def extract_best_path(trans, mstar, s0, T: int, V, policy):
    """Greedy open-loop path using a* from V at each state (deterministic mode)."""
    mk = _key(mstar)
    s = _key(s0)
    path = [s]
    acts = []
    for k in range(T, 0, -1):
        if s == mk:
            break
        # policy[k-1] is for V_k (index: policy[0] <-> V_1)
        a = policy[k - 1].get(s)
        if a is None:
            break
        dist = trans.get((s, a)) or {}
        if not dist:
            break
        # take most likely next, prefer hitting m* if present
        if mk in dist and dist[mk] > 0:
            sp = mk
        else:
            sp = max(dist.items(), key=lambda kv: kv[1])[0]
        acts.append(a)
        path.append(sp)
        s = sp
    p_est = V[T].get(_key(s0), 0.0)
    return {"s0": _key(s0), "path": path, "acts": acts, "V": p_est, "hit_openloop": path[-1] == mk}


def offline_report(trans, horizons=(2, 3, 4)):
    report = {}
    for mstar in HARD:
        mk = _key(mstar)
        report[mk] = {"horizons": {}}
        for T in horizons:
            V, policy = compute_V(trans, mstar, T)
            # best s0 among soft / all non-target
            cands = []
            for s in STATES:
                sk = _key(s)
                if sk == mk:
                    continue
                cands.append((V[T][sk], sk))
            cands.sort(reverse=True)
            best = []
            for v, sk in cands[:5]:
                bp = extract_best_path(trans, mstar, _parse(sk), T, V, policy)
                best.append(bp)
            # one-step P(land) producers
            producers = []
            for s in STATES:
                sk = _key(s)
                for a in ORDER:
                    dist = trans.get((sk, a)) or {}
                    ph = dist.get(mk, 0.0)
                    if ph > 0:
                        producers.append(
                            {"from": sk, "a": a, "P_land": ph, "n_support": len(dist)}
                        )
            producers.sort(key=lambda r: -r["P_land"])
            report[mk]["horizons"][str(T)] = {
                "best_V_by_s0": [{ "s0": sk, "V": v} for v, sk in cands[:8]],
                "best_paths": best[:3],
                "top_producers": producers[:8],
                "V_table": V[T],
                "a_star_T": policy[T - 1] if T >= 1 else {},
            }
        # recommend T=3 path from best soft s0
        T = 3
        V, policy = compute_V(trans, mstar, T)
        soft = ["010", "011", "100", "101"]
        soft_ranked = sorted(
            ((V[T][sk], sk) for sk in soft if sk != mk), reverse=True
        )
        if soft_ranked:
            v, sk = soft_ranked[0]
            report[mk]["recommended"] = extract_best_path(
                trans, mstar, _parse(sk), T, V, policy
            )
            report[mk]["recommended"]["T"] = T
        else:
            report[mk]["recommended"] = None
    return report


def choose_hitting(s, mstar, trans, V_remain, *, a_prev, stagnated):
    """a* = argmax_a E[V_{k-1}(s')] with k steps remaining = len related."""
    sk = _key(s)
    mk = _key(mstar)
    meta: dict[str, Any] = {"mode": "hitting", "by_a": {}, "blocked": None}
    if sk == mk:
        return None, meta
    # V_remain is the V table for steps-1 after action (k-1)
    best_a, best_v = None, -1.0
    for a in ORDER:
        if stagnated and a_prev is not None and a == a_prev:
            meta["blocked"] = a
            # still allow if all blocked — don't skip entirely from cands with -inf
            continue
        dist = trans.get((sk, a))
        if not dist:
            meta["by_a"][a] = {"EV": None, "n": 0}
            continue
        ev = sum(p * V_remain.get(spk, 0.0) for spk, p in dist.items())
        meta["by_a"][a] = {"EV": ev, "P_land": dist.get(mk, 0.0)}
        if ev > best_v + 1e-12:
            best_v, best_a = ev, a
    if best_a is None:
        # fallback: any action with data, ignore anti-stag
        for a in ORDER:
            dist = trans.get((sk, a))
            if not dist:
                continue
            ev = sum(p * V_remain.get(spk, 0.0) for spk, p in dist.items())
            if ev > best_v + 1e-12:
                best_v, best_a = ev, a
        meta["fallback"] = True
    meta["a_star"] = best_a
    meta["EV_star"] = best_v if best_a else None
    return best_a, meta


def run_live_trial(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    trans,
    V_stack,
    mstar,
    seed,
    S0,
    plan0,
    policy: str,
    max_steps: int,
    cache: dict,
):
    """policy in {gamma_prime, hitting}."""
    S = list(S0)
    plan = plan0 or ""
    traj = [list(S)]
    steps = []
    a_prev = None
    stagnated = False
    ever = int(S == list(mstar))
    T = max_steps

    for t in range(max_steps):
        if S == list(mstar):
            ever = 1
            break
        remain = max_steps - t  # steps left including this one
        if policy == "gamma_prime":
            a, meta = p9j.choose_gamma_prime(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated
            )
        else:
            # use V_{remain-1} after this action
            k_after = remain - 1
            V_next = V_stack[k_after]
            a, meta = choose_hitting(
                S, mstar, trans, V_next, a_prev=a_prev, stagnated=stagnated
            )
        if a is None:
            break
        Eb = _E(mstar, S)
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
        Ea = _E(mstar, Sa)
        if Sa == list(mstar):
            ever = 1
        steps.append(
            {
                "t": t,
                "a": a,
                "S_before": list(S),
                "S_after": Sa,
                "E_before": Eb,
                "E_after": Ea,
                "dE": Ea - Eb,
                "temp_worse": int(Ea > Eb),
                "mode": meta.get("mode"),
                "EV": meta.get("EV_star"),
                "by_a": meta.get("by_a"),
                "cache_hit": hit,
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj.append(list(S))
        if ever:
            break

    return {
        "policy": policy,
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "seed": seed,
        "S0": list(S0),
        "S_final": list(S),
        "path": [_key(s) for s in traj],
        "steps": steps,
        "actions": [s["a"] for s in steps],
        "hit": int(S == list(mstar)),
        "ever": ever,
        "E0": _E(mstar, S0),
        "E_final": _E(mstar, S),
        "delta_E": _E(mstar, S0) - _E(mstar, S),
        "n_temp_worse": sum(s["temp_worse"] for s in steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--T", type=int, default=3, help="live horizon / max steps")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--offline-only", action="store_true")
    args = ap.parse_args()

    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
    p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")

    lookup, ker, rows = p9j.build_lookup(p9e, p9g, None)
    trans = build_trans_probs(ker)
    print(
        f"=== Phase 9O kernel cells={len(trans)} rows≈{len(rows)} ===",
        flush=True,
    )

    offline = offline_report(trans, horizons=(2, 3, 4))
    print("=== Offline best paths (T=3) ===", flush=True)
    for mk in ("000", "001", "110"):
        rec = offline[mk].get("recommended")
        print(f"  m*={mk} rec={rec}", flush=True)
        top = offline[mk]["horizons"]["3"]["top_producers"][:3]
        print(f"    producers={top}", flush=True)

    live_paired = []
    by_pol: dict[str, list] = defaultdict(list)
    by_m: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    V_live, _pol_live = compute_V(trans, HARD[0], args.T)  # placeholder replaced per m*

    if not args.offline_only:
        from activation_pipeline.device import (
            LOCAL_MODEL_KEY,
            assert_model_fits_machine,
            resolve_device_map,
        )
        from activation_pipeline.loader import load_model_and_tokenizer

        p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
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

        n_tot = len(HARD) * args.reps
        idx = 0
        print(
            f"=== Live hard Γ' vs hitting T={args.T} reps={args.reps} ===",
            flush=True,
        )
        for mi, mstar in enumerate(HARD):
            V_stack, _ = compute_V(trans, mstar, args.T)
            for r in range(args.reps):
                seed0 = args.seed + 300 * mi + 19 * r
                S0, plan0, seed = p9j.free_S0(
                    p8k, sc, loaded, mstar=mstar, seed0=seed0
                )
                idx += 1
                print(
                    f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                    f"E0={_E(mstar, S0)} V={V_stack[args.T].get(_key(S0))}",
                    flush=True,
                )
                if S0 == list(mstar):
                    print("    skip", flush=True)
                    continue
                cache: dict = {}
                trials = {}
                for pol in ("gamma_prime", "hitting"):
                    trials[pol] = run_live_trial(
                        p9j,
                        p8k,
                        sc,
                        loaded,
                        dirs=dirs,
                        lookup=lookup,
                        trans=trans,
                        V_stack=V_stack,
                        mstar=mstar,
                        seed=seed,
                        S0=S0,
                        plan0=plan0,
                        policy=pol,
                        max_steps=args.T,
                        cache=cache,
                    )
                    t = trials[pol]
                    print(
                        f"    {pol}: hit={t['hit']} ever={t['ever']} "
                        f"worse={t['n_temp_worse']} path={t['path']} acts={t['actions']}",
                        flush=True,
                    )
                    by_pol[pol].append(t)
                    by_m[_key(mstar)][pol].append(t)
                live_paired.append(
                    {
                        "mstar": list(mstar),
                        "mstar_key": _key(mstar),
                        "seed": seed,
                        "S0": S0,
                        **trials,
                    }
                )

    def agg(trials):
        if not trials:
            return {"n": 0}

        def m(k):
            return float(np.mean([t[k] for t in trials]))

        return {
            "n": len(trials),
            "P_hit": m("hit"),
            "P_ever": m("ever"),
            "mean_delta_E": m("delta_E"),
            "mean_temp_worse": m("n_temp_worse"),
        }

    live_agg = {p: agg(by_pol[p]) for p in ("gamma_prime", "hitting")}
    live_per = {
        _key(m): {p: agg(by_m[_key(m)][p]) for p in ("gamma_prime", "hitting")}
        for m in HARD
    }
    n_hit_h = sum(
        1 for m in HARD if (live_per[_key(m)]["hitting"].get("P_ever") or 0) > 0
    )
    n_hit_g = sum(
        1 for m in HARD if (live_per[_key(m)]["gamma_prime"].get("P_ever") or 0) > 0
    )

    gate = {
        "hypothesis": (
            "Finite-horizon P(hit) lookahead accepts temporary E↑ to reach "
            "rare hard sinks; Γ' does not"
        ),
        "T": args.T,
        "offline_recommended": {
            mk: offline[mk].get("recommended") for mk in ("000", "001", "110")
        },
        "live_P_ever": {
            "hitting": {mk: live_per[mk]["hitting"].get("P_ever") for mk in live_per},
            "gamma_prime": {
                mk: live_per[mk]["gamma_prime"].get("P_ever") for mk in live_per
            },
        },
        "n_hard_ever_hitting": n_hit_h,
        "n_hard_ever_gamma_prime": n_hit_g,
        "hard_gate_hitting": n_hit_h == 3,
        "delta_P_ever": (live_agg.get("hitting", {}).get("P_ever") or 0)
        - (live_agg.get("gamma_prime", {}).get("P_ever") or 0),
        "read": (
            "Primary: ∀ hard m* P(ever)>0 under hitting. "
            "Secondary: temporary E↑ usage; vs Γ'."
        ),
    }

    payload = {
        "protocol": "Phase 9O finite-horizon target-hitting planner",
        "seed": args.seed,
        "T": args.T,
        "n_kernel_cells": len(trans),
        "offline": offline,
        "live_agg": live_agg,
        "live_per_mstar": live_per,
        "live_paired": live_paired,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9O — Finite-horizon target-hitting planner",
        "",
        r"> $V_k(s)=\max_a\mathbb E[V_{k-1}(s')]$; $V_0=1[s=m^*]$. "
        "Hard $m^*\\in\\{000,001,110\\}$. Temporary $E\\uparrow$ allowed. No new $v$.",
        "",
        f"T={args.T}, reps={args.reps}, seed={args.seed}, kernel_cells={len(trans)}.",
        "",
        "## Offline recommended paths (T=3)",
        "",
    ]
    for mk in ("000", "001", "110"):
        rec = offline[mk].get("recommended")
        lines.append(f"### $m^*={mk}$")
        if not rec:
            lines.append("- no recommendation")
        else:
            lines.append(
                f"- from `{rec['s0']}`: `{' → '.join(rec['path'])}` "
                f"acts={rec['acts']} $V$={fmt(rec['V'])}"
            )
        prods = offline[mk]["horizons"]["3"]["top_producers"][:4]
        if prods:
            lines.append(
                "- producers: "
                + ", ".join(
                    f"`{p['from']}|{p['a']}`→{fmt(p['P_land'])}" for p in prods
                )
            )
        lines.append("")

    if live_paired:
        lines += [
            "## Live: $\\Gamma'$ vs hitting",
            "",
            "| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\\Delta E$ | mean # $E\\uparrow$ |",
            "|--------|--:|----------:|-----------:|-----------------:|--------------------:|",
        ]
        for p, lab in (("hitting", "**hitting**"), ("gamma_prime", "$\\Gamma'$")):
            a = live_agg[p]
            lines.append(
                f"| {lab} | {a['n']} | {fmt(a.get('P_hit'))} | **{fmt(a.get('P_ever'))}** | "
                f"{fmt(a.get('mean_delta_E'))} | {fmt(a.get('mean_temp_worse'))} |"
            )
        lines += [
            "",
            "| $m^*$ | $P_{ever}$ hit | $P_{ever}$ $\\Gamma'$ | $P_{hit}$ hit |",
            "|-------|---------------:|---------------------:|-------------:|",
        ]
        for m in HARD:
            mk = _key(m)
            st = live_per[mk]
            lines.append(
                f"| `{mk}` | **{fmt(st['hitting'].get('P_ever'))}** | "
                f"{fmt(st['gamma_prime'].get('P_ever'))} | "
                f"{fmt(st['hitting'].get('P_hit'))} |"
            )
        lines += [
            "",
            "## Gate",
            "",
            f"- Hard ever under hitting: **{n_hit_h}/3** (Γ' {n_hit_g}/3)",
            f"- Hard gate (all 3): **{gate['hard_gate_hitting']}**",
            f"- $\\Delta P_{{\\mathrm{{ever}}}}$(hit$-\\Gamma'$): {fmt(gate['delta_P_ever'])}",
            "",
            gate["read"],
            "",
        ]
    lines += [
        r"$$\boxed{a^*=\arg\max_a\mathbb E[V_{T-1}(S')]\quad\text{(hitting, not }\Delta E\text{)}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
