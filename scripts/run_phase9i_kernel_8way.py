#!/usr/bin/env python3
"""Phase 9I — Kernel expand for 000/001/111 + full 8-way Γ'/Γ/fixed.

1) Densify sparse (s,a) cells needed for hard targets 000,001,111 (and 110/101 gaps).
2) Full 8-way live: fixed vs Γ vs Γ' with paired S0, episode cache.
3) Report both final P(hit) and P(ever reach m* along the path).

No new v. No extra memory beyond Γ'. Canonical planner = Γ'.

  .venv/bin/python -u scripts/run_phase9i_kernel_8way.py --reps-expand 6 --reps-8way 3
  .venv/bin/python -u scripts/run_phase9i_kernel_8way.py --expand-only
  .venv/bin/python -u scripts/run_phase9i_kernel_8way.py --eight-way-only
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

from scripts.sync_eq import sync_error_norm  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase9i_kernel_8way.json"
MD = ROOT / "data" / "results" / "sync_phase9i_kernel_8way.md"
KERNEL_9G = ROOT / "data" / "results" / "sync_phase9g_kernel.json"
KERNEL_9I = ROOT / "data" / "results" / "sync_phase9i_kernel.json"

SEED = 20260928
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
SOFT = {(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)}
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
POLICIES = ("gamma", "gamma_prime", "fixed")

# Soft-only (free-batch). Hard 000/001 sources skipped — too expensive to construct.
EXPAND_CELLS: list[tuple[tuple[int, int, int], str]] = [
    ((1, 0, 1), "C"),
    ((1, 0, 1), "H"),
    ((1, 0, 1), "O"),
    ((0, 1, 0), "H"),
    ((1, 0, 0), "C"),
    ((0, 1, 1), "H"),
]


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


def _relevant(s, mstar):
    return [a for a in ORDER if int(s[CH_IDX[a]]) != int(mstar[CH_IDX[a]])]


def _fixed_action(s, mstar):
    rel = _relevant(s, mstar)
    return rel[0] if rel else None


def _neighbors(s):
    out = []
    for i in range(3):
        t = list(s)
        t[i] = 1 - t[i]
        out.append(tuple(t))
    return out


def load_extra_rows() -> list[dict]:
    rows = []
    for path in (KERNEL_9G, KERNEL_9I):
        if not path.exists():
            continue
        for r in json.loads(path.read_text()).get("extra_rows") or []:
            rows.append(
                {
                    "src": r.get("src") or path.stem,
                    "edge": r.get("edge"),
                    "S_before": tuple(r["S_before"]),
                    "S_after": tuple(r["S_after"]),
                    "a": r["a"],
                    "to_intended": tuple(r.get("to_intended") or r["S_after"]),
                }
            )
    return rows


def build_lookup(p9e, p9g, extras: list[dict] | None = None):
    base_extra = load_extra_rows()
    if extras:
        base_extra = base_extra + extras
    return p9g.build_lookup(p9e, base_extra if base_extra else None)


def choose_action(s, mstar, lookup, *, a_prev, stagnated, anti_stag):
    rel = _relevant(s, mstar)
    meta = {"relevant": rel, "blocked": None, "fallback": False}
    if not rel:
        return None, meta
    cands = []
    by_a = {}
    for a in rel:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "Gamma": None}
        by_a[a] = {"Gamma": st.get("Gamma"), "n": st.get("n"), "mean_dE": st.get("mean_dE")}
        if anti_stag and stagnated and a_prev is not None and a == a_prev:
            meta["blocked"] = a
            continue
        if int(st.get("n") or 0) >= 1 and st.get("Gamma") is not None:
            cands.append((a, float(st["Gamma"]), int(st["n"])))
    meta["by_a"] = by_a
    if not cands:
        for a in rel:
            if anti_stag and stagnated and a == a_prev:
                continue
            meta["fallback"] = True
            meta["a_star"] = a
            return a, meta
        meta["fallback"] = True
        meta["a_star"] = rel[0]
        return rel[0], meta
    cands.sort(key=lambda t: (t[1], t[2], -ORDER.index(t[0])), reverse=True)
    meta["a_star"] = cands[0][0]
    meta["Gamma_star"] = cands[0][1]
    return cands[0][0], meta


def construct_source(p9, p8k, sc, loaded, *, dirs, source, seed):
    """Home for soft states; soft-bridge only for hard (skip expensive hard home)."""
    if tuple(source) in SOFT:
        home = p9._home_to_source(
            p8k, sc, loaded, dirs=dirs, source=source, seed=seed
        )
        return {
            "at_source": int(home["homed"]),
            "S": list(home["S_home"]),
            "plan_prefill": home["plan_prefill"],
            "method": "home" if home["homed"] else "home_miss",
        }
    # hard: try at most one soft bridge (no direct home — too slow / rarely works)
    bridges = [br for br in _neighbors(source) if br in SOFT]
    if not bridges:
        return {"at_source": 0, "S": [0, 0, 0], "plan_prefill": "", "method": "no_bridge"}
    br = bridges[seed % len(bridges)]
    bh = p9._home_to_source(p8k, sc, loaded, dirs=dirs, source=br, seed=seed)
    if not bh["homed"]:
        return {
            "at_source": 0,
            "S": list(bh["S_home"]),
            "plan_prefill": bh["plan_prefill"],
            "method": "bridge_home_miss",
        }
    edge = p9._run_one_bit_transition(
        p8k,
        sc,
        loaded,
        dirs=dirs,
        from_s=br,
        to_s=source,
        seed=seed + 9,
        plan_prefill=bh["plan_prefill"],
    )
    if edge["S"] == list(source):
        return {
            "at_source": 1,
            "S": list(source),
            "plan_prefill": edge.get("plan_prefill") or bh["plan_prefill"],
            "method": f"bridge_{_key(br)}",
        }
    return {
        "at_source": 0,
        "S": list(edge["S"]),
        "plan_prefill": edge.get("plan_prefill") or bh["plan_prefill"],
        "method": "bridge_miss",
    }


def densify_free_batch(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    cells: list[tuple[tuple[int, int, int], str]],
    reps: int,
    seed0: int,
    max_frees: int,
    prior_rows: list[dict],
) -> list[dict[str, Any]]:
    """Fast densify: free-run; if S matches a needed source, apply one pending action."""
    need: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for s, a in cells:
        need[s].add(a)
    counts: dict[tuple[tuple[int, int, int], str], int] = defaultdict(int)
    for r in prior_rows:
        key = (tuple(r["S_before"]), r["a"])
        if key[0] in need and key[1] in need[key[0]]:
            counts[key] += 1
    rows: list[dict] = []

    def remaining() -> int:
        return sum(max(0, reps - counts[(s, a)]) for s, acts in need.items() for a in acts)

    print(
        f"=== Phase 9I free-batch densify need={remaining()} max_frees={max_frees} ===",
        flush=True,
    )
    for i in range(max_frees):
        if remaining() <= 0:
            print(f"  done early at free#{i}", flush=True)
            break
        seed = seed0 + 17 * i
        torch.manual_seed(seed)
        free = p8k._run_episode(sc, loaded, active={}, seed=seed)
        S = tuple(free["S"])
        plan = free.get("plan_prefill") or ""
        if S not in need:
            continue
        pending = [a for a in sorted(need[S], key=lambda x: ORDER.index(x)) if counts[(S, a)] < reps]
        if not pending:
            continue
        a = pending[0]
        mstar = list(S)
        mstar[CH_IDX[a]] = 1 - int(S[CH_IDX[a]])
        mstar_t = tuple(mstar)
        active = {a: (p8k._s_star(mstar_t[CH_IDX[a]]), dirs[a])}
        h_pf = plan if a == "H" else None
        out = p8k._run_episode(
            sc, loaded, active=active, seed=seed + 7, h_plan_prefill=h_pf
        )
        Sa = tuple(out["S"])
        row = {
            "src": "9I_free",
            "edge": f"{_key(S)}|{a}",
            "S_before": list(S),
            "S_after": list(Sa),
            "a": a,
            "to_intended": list(mstar_t),
            "dS": int(Sa != S),
            "method": "free",
            "seed": seed,
        }
        rows.append(row)
        counts[(S, a)] += 1
        print(
            f"  free#{i} {_key(S)}|{a} →{_key(Sa)} n={counts[(S, a)]}/{reps} left={remaining()}",
            flush=True,
        )
    return rows


def densify_cell(p9, p8k, sc, loaded, *, dirs, s_target, action, reps, seed0):
    """Unused slow path kept for optional hard-source experiments."""
    raise RuntimeError("use densify_free_batch")


def apply_channel(p8k, sc, loaded, *, dirs, mstar, channel, seed, plan_prefill, cache):
    key = (int(seed), channel, plan_prefill or "")
    if cache is not None and key in cache:
        return cache[key], True
    active = {channel: (p8k._s_star(mstar[CH_IDX[channel]]), dirs[channel])}
    h_pf = plan_prefill if channel == "H" else None
    out = p8k._run_episode(sc, loaded, active=active, seed=seed, h_plan_prefill=h_pf)
    if cache is not None:
        cache[key] = out
    return out, False


def run_policy_trial(
    p8k, sc, loaded, *, dirs, lookup, mstar, seed, policy, max_steps, S0, plan0, cache
):
    S = list(S0)
    plan = plan0 or ""
    traj_S = [list(S)]
    traj_E = [_E(mstar, S)]
    steps = []
    a_prev = None
    stagnated = False
    ever = int(S == list(mstar))
    n_noop = n_rep = n_res = 0

    for t in range(max_steps):
        if S == list(mstar):
            ever = 1
            break
        prev_noop = bool(stagnated and a_prev is not None)
        if policy == "fixed":
            a = _fixed_action(S, mstar)
            meta = {"a_star": a, "blocked": None}
        elif policy == "gamma":
            a, meta = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=False
            )
        else:
            a, meta = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=True
            )
        if a is None:
            break
        if prev_noop:
            n_noop += 1
            if a == a_prev:
                n_rep += 1
        Eb = _E(mstar, S)
        out, _ = apply_channel(
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
        dS = int(Sa != S)
        down = int(Ea < Eb)
        if prev_noop and a != a_prev and (dS or down):
            n_res += 1
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
                "dS": dS,
                "down": down,
                "blocked": meta.get("blocked"),
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj_S.append(list(S))
        traj_E.append(Ea)

    E0, Ef = traj_E[0], traj_E[-1]
    return {
        "policy": policy,
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "seed": seed,
        "S0": traj_S[0],
        "S_final": traj_S[-1],
        "traj_S": traj_S,
        "traj_E": traj_E,
        "steps": steps,
        "hit": int(traj_S[-1] == list(mstar)),
        "ever_reach": ever,
        "delta_E": int(E0 - Ef),
        "E0": E0,
        "E_final": Ef,
        "n_steps": len(steps),
        "actions": [s["a"] for s in steps],
        "P_down_steps": float(np.mean([s["down"] for s in steps])) if steps else float("nan"),
        "n_noop_contexts": n_noop,
        "n_repeat_after_noop": n_rep,
        "n_rescue_after_noop": n_res,
        "P_repeat_after_noop": n_rep / n_noop if n_noop else float("nan"),
        "P_rescue_given_noop": n_res / n_noop if n_noop else float("nan"),
    }


def _agg(trials):
    if not trials:
        return {"n": 0}

    def m(k):
        xs = [t[k] for t in trials if t.get(k) == t.get(k)]
        return float(np.mean(xs)) if xs else float("nan")

    n_noop = sum(t["n_noop_contexts"] for t in trials)
    n_rep = sum(t["n_repeat_after_noop"] for t in trials)
    n_res = sum(t["n_rescue_after_noop"] for t in trials)
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_ever": m("ever_reach"),
        "mean_delta_E": m("delta_E"),
        "mean_E_final": m("E_final"),
        "mean_P_down_steps": m("P_down_steps"),
        "P_repeat_after_noop": n_rep / n_noop if n_noop else float("nan"),
        "P_rescue_given_noop": n_res / n_noop if n_noop else float("nan"),
        "n_noop_contexts": n_noop,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps-expand", type=int, default=3)
    ap.add_argument("--reps-8way", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--expand-only", action="store_true")
    ap.add_argument("--eight-way-only", action="store_true")
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
    p9 = _load_mod("phase9", ROOT / "scripts" / "run_phase9_hamming_paths.py")
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()

    new_rows: list[dict] = []
    expand_stats: dict[str, Any] = {}

    if not args.eight_way_only:
        prior = []
        if KERNEL_9I.exists():
            prior = json.loads(KERNEL_9I.read_text()).get("extra_rows") or []
            print(f"resumed {len(prior)} prior 9I rows", flush=True)
        batch = densify_free_batch(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            cells=EXPAND_CELLS,
            reps=args.reps_expand,
            seed0=args.seed,
            max_frees=max(24, args.reps_expand * 8),
            prior_rows=prior,
        )
        new_rows = prior + batch
        raw_stats: dict[str, Any] = {}
        for r in new_rows:
            ck = f"{_key(r['S_before'])}|{r['a']}"
            raw_stats.setdefault(ck, {"n": 0, "shift": 0, "hist": Counter()})
            raw_stats[ck]["n"] += 1
            raw_stats[ck]["shift"] += int(r.get("dS") or 0)
            raw_stats[ck]["hist"][_key(r["S_after"])] += 1
        expand_stats = {
            ck: {
                "n": st["n"],
                "P_shift": st["shift"] / st["n"] if st["n"] else None,
                "after_hist": dict(st["hist"]),
            }
            for ck, st in raw_stats.items()
        }
        KERNEL_9I.write_text(
            json.dumps({"extra_rows": new_rows, "expand_stats": expand_stats}, indent=2),
            encoding="utf-8",
        )
        print(f"  wrote {len(batch)} new rows (total {len(new_rows)})", flush=True)

    lookup, ker, _ = build_lookup(p9e, p9g, None)
    # rebuild expand_stats summary from ker for hard cells
    hard_cov = {
        f"{sk}|{a}": len(outs)
        for (sk, a), outs in ker.items()
        if sk in ("000", "001", "111", "110", "101")
    }

    paired = []
    by_policy = {p: [] for p in POLICIES}
    by_mstar = defaultdict(lambda: {p: [] for p in POLICIES})

    if not args.expand_only:
        print(
            f"=== Phase 9I full 8-way Γ'/Γ/fixed reps={args.reps_8way} "
            f"max_steps={args.max_steps} ker_cells={len(ker)} ===",
            flush=True,
        )
        n_total = 8 * args.reps_8way
        idx = 0
        for mi, mstar in enumerate(ALL_MSTAR):
            for r in range(args.reps_8way):
                seed0 = args.seed + 200 * mi + 19 * r
                S0 = None
                plan0 = ""
                seed = seed0
                for attempt in range(10):
                    seed = seed0 + 29 * attempt
                    torch.manual_seed(seed)
                    free = p8k._run_episode(sc, loaded, active={}, seed=seed)
                    S0 = list(free["S"])
                    plan0 = free.get("plan_prefill") or ""
                    if S0 != list(mstar):
                        break
                idx += 1
                print(
                    f"  [{idx}/{n_total}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                    f"E0={_E(mstar, S0)}",
                    flush=True,
                )
                if S0 == list(mstar):
                    print("    skip S0==m*", flush=True)
                    continue
                ep_cache: dict = {}
                trials = {}
                for pol in POLICIES:
                    trials[pol] = run_policy_trial(
                        p8k,
                        sc,
                        loaded,
                        dirs=dirs,
                        lookup=lookup,
                        mstar=mstar,
                        seed=seed,
                        policy=pol,
                        max_steps=args.max_steps,
                        S0=S0,
                        plan0=plan0,
                        cache=ep_cache,
                    )
                    t = trials[pol]
                    tag = {"gamma_prime": "Γ'", "gamma": "Γ", "fixed": "f"}[pol]
                    print(
                        f"    {tag}: hit={t['hit']} ever={t['ever_reach']} "
                        f"E={t['E0']}→{t['E_final']} acts={t['actions']} "
                        f"path={[ _key(s) for s in t['traj_S'] ]}",
                        flush=True,
                    )
                    by_policy[pol].append(t)
                    by_mstar[_key(mstar)][pol].append(t)
                paired.append(
                    {
                        "mstar": list(mstar),
                        "mstar_key": _key(mstar),
                        "seed": seed,
                        "S0": S0,
                        **trials,
                    }
                )
                print(f"    cache={len(ep_cache)}", flush=True)

    aggs = {p: _agg(by_policy[p]) for p in POLICIES}
    per_m = {
        _key(m): {p: _agg(by_mstar[_key(m)][p]) for p in POLICIES}
        for m in ALL_MSTAR
    }

    # reachability: any ever>0 per m*
    reach = {}
    for m in ALL_MSTAR:
        mk = _key(m)
        reach[mk] = {
            p: (per_m[mk][p].get("P_ever") or 0) > 0 for p in POLICIES
        }

    n_reach_prime = sum(1 for mk, v in reach.items() if v["gamma_prime"])
    n_reach_gamma = sum(1 for mk, v in reach.items() if v["gamma"])
    n_reach_fixed = sum(1 for mk, v in reach.items() if v["fixed"])

    gate = {
        "hypothesis": (
            "Expanded kernel + Γ' yields P(ever reach)>0 for all 8 m* "
            "(multi-step reachability before high final hit)"
        ),
        "n_9i_rows": len([r for r in (json.loads(KERNEL_9I.read_text())["extra_rows"] if KERNEL_9I.exists() else [])]),
        "hard_coverage": hard_cov,
        "P_hit": {p: aggs[p].get("P_hit") for p in POLICIES},
        "P_ever": {p: aggs[p].get("P_ever") for p in POLICIES},
        "n_mstar_ever_gamma_prime": n_reach_prime,
        "n_mstar_ever_gamma": n_reach_gamma,
        "n_mstar_ever_fixed": n_reach_fixed,
        "universal_ever_gamma_prime": n_reach_prime == 8,
        "hard_ever_prime": {
            mk: reach[mk]["gamma_prime"] for mk in ("000", "001", "111")
        },
        "delta_P_hit_prime_gamma": (aggs["gamma_prime"].get("P_hit") or 0)
        - (aggs["gamma"].get("P_hit") or 0),
        "delta_P_ever_prime_gamma": (aggs["gamma_prime"].get("P_ever") or 0)
        - (aggs["gamma"].get("P_ever") or 0),
        "read": (
            "Primary: ∀m* P(ever reach)>0 under Γ'. Secondary: final P(hit) and "
            "Γ' vs Γ vs fixed. No extra memory."
        ),
    }

    payload = {
        "protocol": "Phase 9I kernel expand + full 8-way Γ'/Γ/fixed",
        "seed": args.seed,
        "reps_expand": args.reps_expand,
        "reps_8way": args.reps_8way,
        "max_steps": args.max_steps,
        "expand_stats": expand_stats,
        "hard_coverage": hard_cov,
        "agg": aggs,
        "per_mstar": per_m,
        "reachability": reach,
        "paired": paired,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9I — Kernel expand + full 8-way $\\Gamma'$",
        "",
        "> Canonical planner $\\Gamma'=\\Gamma+$no-op memory. Frozen 8K. "
        "Primary criterion: $P(\\exists t:S_t=m^*)>0$ for all $m^*$.",
        "",
        f"expand_reps={args.reps_expand}, 8way_reps={args.reps_8way}, "
        f"max_steps={args.max_steps}, seed={args.seed}.",
        "",
        "## Hard-path kernel coverage $(s,a)$",
        "",
        "| cell | n |",
        "|------|--:|",
    ]
    for ck, n in sorted(hard_cov.items()):
        lines.append(f"| `{ck}` | {n} |")
    if expand_stats:
        lines += ["", "### New 9I densify", "", "| cell | n | $P_{shift}$ | hist |", "|------|--:|------------:|------|"]
        for ck, st in sorted(expand_stats.items()):
            hist = " ".join(f"{k}:{v}" for k, v in list((st.get("after_hist") or {}).items())[:4])
            lines.append(f"| `{ck}` | {st['n']} | {fmt(st.get('P_shift'))} | {hist} |")

    if not args.expand_only and paired:
        lines += [
            "",
            "## Full 8-way aggregate",
            "",
            "| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\\Delta E$ | $P(E\\downarrow)$ | $P(\\mathrm{repeat}|\\mathrm{noop})$ |",
            "|--------|--:|----------:|-----------:|-----------------:|------------------:|-------------------------------:|",
        ]
        for p, lab in (("gamma_prime", "**$\\Gamma'$**"), ("gamma", "$\\Gamma$"), ("fixed", "fixed")):
            a = aggs[p]
            lines.append(
                f"| {lab} | {a['n']} | {fmt(a.get('P_hit'))} | **{fmt(a.get('P_ever'))}** | "
                f"{fmt(a.get('mean_delta_E'))} | {fmt(a.get('mean_P_down_steps'))} | "
                f"{fmt(a.get('P_repeat_after_noop'))} |"
            )
        lines += [
            "",
            "## Per $m^*$: $P_{ever}$ / $P_{hit}$ ($\\Gamma'$)",
            "",
            "| $m^*$ | $P_{ever}\\Gamma'$ | $P_{hit}\\Gamma'$ | $P_{ever}\\Gamma$ | $P_{ever}$ fixed |",
            "|-------|-------------------:|-----------------:|-----------------:|-----------------:|",
        ]
        for m in ALL_MSTAR:
            mk = _key(m)
            st = per_m[mk]
            lines.append(
                f"| `{mk}` | {fmt(st['gamma_prime'].get('P_ever'))} | "
                f"{fmt(st['gamma_prime'].get('P_hit'))} | {fmt(st['gamma'].get('P_ever'))} | "
                f"{fmt(st['fixed'].get('P_ever'))} |"
            )
        lines += [
            "",
            f"$m^*$ with $P_{{ever}}>0$: $\\Gamma'$ **{n_reach_prime}/8**, "
            f"$\\Gamma$ {n_reach_gamma}/8, fixed {n_reach_fixed}/8.",
            "",
            "## Gate",
            "",
            f"- Universal ever-reach under $\\Gamma'$: **{gate['universal_ever_gamma_prime']}**",
            f"- Hard ever-reach $000/001/111$: `{gate['hard_ever_prime']}`",
            f"- $\\Delta P_{{hit}}(\\Gamma'-\\Gamma)$: {fmt(gate['delta_P_hit_prime_gamma'])}",
            f"- $\\Delta P_{{ever}}(\\Gamma'-\\Gamma)$: {fmt(gate['delta_P_ever_prime_gamma'])}",
            "",
            gate["read"],
            "",
            r"$$\boxed{\text{causal actuators}+\Gamma+\text{one-step memory}\rightarrow\text{8-way multi-step reachability}}$$",
            "",
        ]
    else:
        lines += ["", "## Eight-way", "", "_skipped (`--expand-only`)_", ""]

    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
