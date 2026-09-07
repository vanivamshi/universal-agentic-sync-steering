#!/usr/bin/env python3
"""Phase 9J — Sink-seeking for hard targets {000,001,110}.

Evidence from 9I: rare sink-producing transitions exist, but Γ' systematically
deprioritizes them. Test whether target-aware landing probability recovers them.

  a* = argmax_a [ P(S'=m*|s,a) + λ Γ'(a|s,m*) ]

First hard-target test uses λ=0 (pure sink seeking) vs Γ'-only.
Sink-seek pools over ALL channels (not Hamming-relevant only): producers like
111|C→110 and 011|C→001 flip bits that already match m*.

1) Densify only producer cells: 011|H, 011|C, 100|O, 111|C, 111|H
2) Hard-target live: m*∈{000,001,110}, max_steps=6–8
3) If ∀ hard P(ever)>0 under sink-seek, full 8-way with hybrid Γ'+hard sink-seek

No new v. Frozen 8K. No extra memory beyond Γ'.

  .venv/bin/python -u scripts/run_phase9j_sink_seeking.py
  .venv/bin/python -u scripts/run_phase9j_sink_seeking.py --densify-only
  .venv/bin/python -u scripts/run_phase9j_sink_seeking.py --hard-only
  .venv/bin/python -u scripts/run_phase9j_sink_seeking.py --eight-way-only
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

OUT = ROOT / "data" / "results" / "sync_phase9j_sink_seeking.json"
MD = ROOT / "data" / "results" / "sync_phase9j_sink_seeking.md"
KERNEL_9G = ROOT / "data" / "results" / "sync_phase9g_kernel.json"
KERNEL_9I = ROOT / "data" / "results" / "sync_phase9i_kernel.json"
KERNEL_9J = ROOT / "data" / "results" / "sync_phase9j_kernel.json"

SEED = 20260929
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
SOFT = {(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)}
HARD_MSTAR = [(0, 0, 0), (0, 0, 1), (1, 1, 0)]
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]

# Only the empirically observed sink producers
PRODUCER_CELLS: list[tuple[tuple[int, int, int], str]] = [
    ((0, 1, 1), "H"),  # →000
    ((0, 1, 1), "C"),  # →001
    ((1, 0, 0), "O"),  # →110
    ((1, 1, 1), "C"),  # →110
    ((1, 1, 1), "H"),  # →110 / 000
]

FOCUS_P = {
    ("011", "H"): "000",
    ("011", "C"): "001",
    ("100", "O"): "110",
    ("111", "C"): "110",
    ("111", "H"): "110",
}

HARD_POLICIES = ("gamma_prime", "sink_seek")
EIGHT_POLICIES = ("gamma_prime", "hybrid")  # hybrid = Γ' soft + sink-seek hard


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


def _neighbors(s):
    out = []
    for i in range(3):
        t = list(s)
        t[i] = 1 - t[i]
        out.append(tuple(t))
    return out


def load_extra_rows(include_9j: bool = True) -> list[dict]:
    rows = []
    paths = [KERNEL_9G, KERNEL_9I]
    if include_9j:
        paths.append(KERNEL_9J)
    for path in paths:
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
    base = load_extra_rows(include_9j=True)
    if extras:
        base = base + extras
    return p9g.build_lookup(p9e, base if base else None)


def choose_gamma_prime(s, mstar, lookup, *, a_prev, stagnated):
    """Γ' = Γ with anti-stagnation block."""
    rel = _relevant(s, mstar)
    meta: dict[str, Any] = {
        "relevant": rel,
        "blocked": None,
        "fallback": False,
        "mode": "gamma_prime",
    }
    if not rel:
        return None, meta
    cands = []
    by_a = {}
    for a in rel:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "Gamma": None}
        by_a[a] = {
            "Gamma": st.get("Gamma"),
            "P_hit": st.get("P_hit"),
            "n": st.get("n"),
            "mean_dE": st.get("mean_dE"),
        }
        if stagnated and a_prev is not None and a == a_prev:
            meta["blocked"] = a
            continue
        if int(st.get("n") or 0) >= 1 and st.get("Gamma") is not None:
            cands.append((a, float(st["Gamma"]), int(st["n"])))
    meta["by_a"] = by_a
    if not cands:
        for a in rel:
            if stagnated and a == a_prev:
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


def choose_sink_seek(s, mstar, lookup, *, a_prev, stagnated, lam: float):
    """a* = argmax [P(S'=m*|s,a) + λ Γ'] over ALL channels (not only relevant).

    Hard sinks are often produced by side-effect flips of bits that already
    match m* (e.g. 111|C→110, 011|C→001). Restricting to Hamming-relevant
    actions makes those producers unreachable by construction.
    """
    rel = _relevant(s, mstar)
    meta: dict[str, Any] = {
        "relevant": rel,
        "blocked": None,
        "fallback": False,
        "mode": "sink_seek",
        "lambda": lam,
        "pool": "all_channels",
    }
    if s == list(mstar) or tuple(s) == tuple(mstar):
        return None, meta
    cands = []
    by_a = {}
    for a in ORDER:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "Gamma": None, "P_hit": None}
        ph = st.get("P_hit")
        g = st.get("Gamma")
        by_a[a] = {
            "Gamma": g,
            "P_hit": ph,
            "n": st.get("n"),
            "mean_dE": st.get("mean_dE"),
            "score": None,
            "relevant": a in rel,
        }
        if stagnated and a_prev is not None and a == a_prev:
            meta["blocked"] = a
            continue
        n = int(st.get("n") or 0)
        if n < 1 or ph is None:
            continue
        score = float(ph) + lam * (float(g) if g is not None else 0.0)
        by_a[a]["score"] = score
        # Prefer positive landing mass; allow 0 only if all are 0 (handled below)
        cands.append((a, score, float(ph), n, int(a in rel)))
    meta["by_a"] = by_a
    pos = [c for c in cands if c[2] > 0]
    use = pos if pos else cands
    if not use:
        a, gmeta = choose_gamma_prime(s, mstar, lookup, a_prev=a_prev, stagnated=stagnated)
        meta["fallback"] = True
        meta["fallback_to"] = "gamma_prime"
        meta["a_star"] = a
        meta["by_a"] = {**by_a, **(gmeta.get("by_a") or {})}
        return a, meta
    # score, P_hit, n; prefer empirically landing >0; mild preference for relevant on ties
    use.sort(key=lambda t: (t[1], t[2], t[3], t[4], -ORDER.index(t[0])), reverse=True)
    meta["a_star"] = use[0][0]
    meta["score_star"] = use[0][1]
    meta["P_hit_star"] = use[0][2]
    meta["used_nonrelevant"] = use[0][0] not in rel
    return use[0][0], meta


def choose_action(s, mstar, lookup, *, policy, a_prev, stagnated, lam, hard_set):
    mk = tuple(mstar) if not isinstance(mstar, tuple) else mstar
    if policy == "gamma_prime":
        return choose_gamma_prime(s, mstar, lookup, a_prev=a_prev, stagnated=stagnated)
    if policy == "sink_seek":
        return choose_sink_seek(
            s, mstar, lookup, a_prev=a_prev, stagnated=stagnated, lam=lam
        )
    if policy == "hybrid":
        if mk in hard_set:
            return choose_sink_seek(
                s, mstar, lookup, a_prev=a_prev, stagnated=stagnated, lam=lam
            )
        return choose_gamma_prime(s, mstar, lookup, a_prev=a_prev, stagnated=stagnated)
    raise ValueError(policy)


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
        f"=== 9J free-batch densify need={remaining()} max_frees={max_frees} ===",
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
        pending = [
            a
            for a in sorted(need[S], key=lambda x: ORDER.index(x))
            if counts[(S, a)] < reps
        ]
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
            "src": "9J_free",
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


def densify_111_bridge(
    p9,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    actions: list[str],
    reps: int,
    seed0: int,
    prior_rows: list[dict],
    max_att_per: int,
) -> list[dict[str, Any]]:
    """Construct ≈111 via soft bridge, then apply C/H."""
    source = (1, 1, 1)
    counts = {a: 0 for a in actions}
    for r in prior_rows:
        if tuple(r["S_before"]) == source and r["a"] in counts:
            counts[r["a"]] += 1
    rows: list[dict] = []
    bridges = [br for br in _neighbors(source) if br in SOFT]
    print(
        f"=== 9J bridge-densify 111|{actions} need="
        f"{sum(max(0, reps - counts[a]) for a in actions)} ===",
        flush=True,
    )
    for ai, a in enumerate(actions):
        attempts = 0
        while counts[a] < reps and attempts < max_att_per:
            seed = seed0 + 1000 * ai + 23 * attempts
            attempts += 1
            br = bridges[seed % len(bridges)]
            bh = p9._home_to_source(p8k, sc, loaded, dirs=dirs, source=br, seed=seed)
            if not bh["homed"]:
                print(f"  111|{a} att#{attempts} bridge_home_miss from {_key(br)}", flush=True)
                continue
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
            if edge["S"] != list(source):
                print(
                    f"  111|{a} att#{attempts} bridge_miss {_key(br)}→{_key(edge['S'])}",
                    flush=True,
                )
                continue
            plan = edge.get("plan_prefill") or bh["plan_prefill"] or ""
            mstar = list(source)
            mstar[CH_IDX[a]] = 1 - int(source[CH_IDX[a]])
            mstar_t = tuple(mstar)
            active = {a: (p8k._s_star(mstar_t[CH_IDX[a]]), dirs[a])}
            h_pf = plan if a == "H" else None
            out = p8k._run_episode(
                sc, loaded, active=active, seed=seed + 11, h_plan_prefill=h_pf
            )
            Sa = tuple(out["S"])
            row = {
                "src": "9J_bridge",
                "edge": f"111|{a}",
                "S_before": list(source),
                "S_after": list(Sa),
                "a": a,
                "to_intended": list(mstar_t),
                "dS": int(Sa != source),
                "method": f"bridge_{_key(br)}",
                "seed": seed,
            }
            rows.append(row)
            counts[a] += 1
            print(
                f"  111|{a} →{_key(Sa)} n={counts[a]}/{reps} via {_key(br)}",
                flush=True,
            )
    return rows


def producer_landing_stats(rows: list[dict]) -> dict[str, Any]:
    """Estimate focus P(land|s,a) from all rows matching producer cells."""
    by_cell: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        sk = _key(r["S_before"])
        a = r["a"]
        ck = f"{sk}|{a}"
        if (sk, a) in FOCUS_P:
            by_cell[ck].append(_key(r["S_after"]))
    out = {}
    for (sk, a), land in FOCUS_P.items():
        ck = f"{sk}|{a}"
        hist = Counter(by_cell.get(ck, []))
        n = sum(hist.values())
        out[ck] = {
            "n": n,
            "target_land": land,
            "P_land": (hist.get(land, 0) / n) if n else None,
            "after_hist": dict(hist),
        }
    return out


def choose_retention_probe(s, mstar, lookup, *, a_arrive: str | None):
    """One post-landing action: prefer max P(stay)=P(S'=m*|s,a); else a_arrive else O."""
    meta: dict[str, Any] = {"mode": "retention_probe", "by_a": {}}
    cands = []
    for a in ORDER:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "P_hit": None}
        ph = st.get("P_hit")
        meta["by_a"][a] = {"P_hit": ph, "n": st.get("n")}
        if int(st.get("n") or 0) >= 1 and ph is not None:
            cands.append((a, float(ph), int(st["n"])))
    if cands:
        cands.sort(key=lambda t: (t[1], t[2], -ORDER.index(t[0])), reverse=True)
        meta["a_star"] = cands[0][0]
        meta["P_hit_star"] = cands[0][1]
        return cands[0][0], meta
    a = a_arrive if a_arrive in ORDER else "O"
    meta["a_star"] = a
    meta["fallback"] = True
    return a, meta


def run_policy_trial(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    mstar,
    seed,
    policy,
    max_steps,
    S0,
    plan0,
    cache,
    lam,
    hard_set,
    continue_after_land: bool = True,
):
    """Live trial. After first landing, one retention probe then stop."""
    S = list(S0)
    plan = plan0 or ""
    traj_S = [list(S)]
    traj_E = [_E(mstar, S)]
    steps = []
    a_prev = None
    stagnated = False
    ever = int(S == list(mstar))
    land_t = -1 if ever else None  # -1 = started at target
    a_arrive = None
    retention = None
    n_noop = n_rep = n_res = 0
    need_retention = bool(ever and continue_after_land)

    for t in range(max_steps):
        at_target = S == list(mstar)
        if at_target and land_t is not None and not need_retention:
            break
        if at_target and need_retention:
            a, meta = choose_retention_probe(
                S, mstar, lookup, a_arrive=a_arrive
            )
        else:
            prev_noop = bool(stagnated and a_prev is not None)
            a, meta = choose_action(
                S,
                mstar,
                lookup,
                policy=policy,
                a_prev=a_prev,
                stagnated=stagnated,
                lam=lam,
                hard_set=hard_set,
            )
            if a is None:
                break
            if prev_noop:
                n_noop += 1
                if a == a_prev:
                    n_rep += 1
        if a is None:
            break
        prev_noop = bool(stagnated and a_prev is not None)
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
        if (not at_target) and prev_noop and a != a_prev and (dS or down):
            n_res += 1
        first_land = False
        if Sa == list(mstar) and land_t is None:
            ever = 1
            land_t = t
            a_arrive = a
            first_land = True
            need_retention = bool(continue_after_land)
        step = {
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
            "mode": meta.get("mode"),
            "score": meta.get("score_star"),
            "P_hit_star": meta.get("P_hit_star"),
            "Gamma_star": meta.get("Gamma_star"),
            "fallback": meta.get("fallback"),
            "by_a": meta.get("by_a"),
            "first_land": first_land,
            "retention_probe": bool(at_target and need_retention),
        }
        steps.append(step)
        if at_target and need_retention:
            retention = {
                "land_t": land_t,
                "exit_a": a,
                "S_land": list(mstar),
                "S_after_exit": Sa,
                "stayed": int(Sa == list(mstar)),
                "dS": dS,
            }
            need_retention = False
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj_S.append(list(S))
        traj_E.append(Ea)
        if retention is not None and not need_retention and land_t is not None:
            # stop after retention probe
            if steps and steps[-1].get("retention_probe"):
                break

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
        "land_t": land_t,
        "retention": retention,
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
    ret = [t["retention"] for t in trials if t.get("retention")]
    n_land = sum(1 for t in trials if t.get("ever_reach"))
    n_stay = sum(1 for r in ret if r.get("stayed"))
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "P_ever": m("ever_reach"),
        "mean_delta_E": m("delta_E"),
        "mean_E_final": m("E_final"),
        "mean_P_down_steps": m("P_down_steps"),
        "P_repeat_after_noop": n_rep / n_noop if n_noop else float("nan"),
        "P_rescue_given_noop": n_res / n_noop if n_noop else float("nan"),
        "n_landings": n_land,
        "n_retention_logged": len(ret),
        "P_stay_after_land": (n_stay / len(ret)) if ret else float("nan"),
        "retention_exits": [
            {
                "exit_a": r["exit_a"],
                "S_after": _key(r["S_after_exit"]),
                "stayed": r["stayed"],
            }
            for r in ret[:12]
        ],
    }


def free_S0(p8k, sc, loaded, *, mstar, seed0):
    S0 = None
    plan0 = ""
    seed = seed0
    for attempt in range(12):
        seed = seed0 + 29 * attempt
        torch.manual_seed(seed)
        free = p8k._run_episode(sc, loaded, active={}, seed=seed)
        S0 = list(free["S"])
        plan0 = free.get("plan_prefill") or ""
        if S0 != list(mstar):
            break
    return S0, plan0, seed


def write_kernel(rows, stats):
    KERNEL_9J.write_text(
        json.dumps({"extra_rows": rows, "expand_stats": stats, "landing": producer_landing_stats(rows)}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps-expand", type=int, default=8)
    ap.add_argument("--reps-hard", type=int, default=4)
    ap.add_argument("--reps-8way", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=7)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-frees", type=int, default=0, help="0 → auto")
    ap.add_argument("--densify-only", action="store_true")
    ap.add_argument("--hard-only", action="store_true")
    ap.add_argument("--eight-way-only", action="store_true")
    ap.add_argument("--skip-bridge", action="store_true", help="skip 111 bridge densify")
    ap.add_argument("--force-8way", action="store_true", help="run 8-way even if hard gate fails")
    args = ap.parse_args()

    hard_set = set(HARD_MSTAR)

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

    expand_stats: dict[str, Any] = {}
    landing: dict[str, Any] = {}
    new_rows: list[dict] = []

    if not args.hard_only and not args.eight_way_only:
        prior = []
        if KERNEL_9J.exists():
            prior = json.loads(KERNEL_9J.read_text()).get("extra_rows") or []
            print(f"resumed {len(prior)} prior 9J rows", flush=True)

        soft_cells = [(s, a) for s, a in PRODUCER_CELLS if s in SOFT]
        hard_cells = [(s, a) for s, a in PRODUCER_CELLS if s not in SOFT]
        max_frees = args.max_frees or max(48, args.reps_expand * 12)
        batch = densify_free_batch(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            cells=soft_cells + (hard_cells if not args.skip_bridge else []),
            reps=args.reps_expand,
            seed0=args.seed,
            max_frees=max_frees,
            prior_rows=prior,
        )
        new_rows = prior + batch
        write_kernel(new_rows, {})

        if hard_cells and not args.skip_bridge:
            # Fill remaining 111|* via soft bridge
            need_111 = []
            counts_111 = Counter(
                (tuple(r["S_before"]), r["a"])
                for r in new_rows
                if tuple(r["S_before"]) == (1, 1, 1)
            )
            for s, a in hard_cells:
                if counts_111[(s, a)] < args.reps_expand:
                    need_111.append(a)
            if need_111:
                bridge = densify_111_bridge(
                    p9,
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    actions=sorted(set(need_111), key=lambda x: ORDER.index(x)),
                    reps=args.reps_expand,
                    seed0=args.seed + 5000,
                    prior_rows=new_rows,
                    max_att_per=max(16, args.reps_expand * 4),
                )
                new_rows = new_rows + bridge

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
        # Landing estimates use 9J + prior kernels for producer cells
        all_for_land = load_extra_rows(include_9j=False) + [
            {
                "S_before": tuple(r["S_before"]),
                "S_after": tuple(r["S_after"]),
                "a": r["a"],
            }
            for r in new_rows
        ]
        landing = producer_landing_stats(all_for_land)
        write_kernel(new_rows, expand_stats)
        print("=== Producer landing estimates ===", flush=True)
        for ck, st in sorted(landing.items()):
            print(
                f"  {ck} →{st['target_land']}: n={st['n']} "
                f"P={st['P_land']} hist={st['after_hist']}",
                flush=True,
            )

    if args.densify_only:
        payload = {
            "protocol": "Phase 9J densify-only",
            "expand_stats": expand_stats,
            "landing": landing or (
                json.loads(KERNEL_9J.read_text()).get("landing") if KERNEL_9J.exists() else {}
            ),
        }
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        MD.write_text(
            "# Phase 9J — densify only\n\nSee JSON landing estimates.\n",
            encoding="utf-8",
        )
        print(json.dumps({"out": str(OUT), "landing": payload["landing"]}, indent=2))
        return 0

    if KERNEL_9J.exists() and not landing:
        landing = json.loads(KERNEL_9J.read_text()).get("landing") or {}
        expand_stats = json.loads(KERNEL_9J.read_text()).get("expand_stats") or {}

    lookup, ker, _ = build_lookup(p9e, p9g, None)
    # Refresh landing from full merged kernel for focus cells
    land_rows = []
    for (sk, a), outs in ker.items():
        if (sk, a) in FOCUS_P:
            for sa in outs:
                land_rows.append({"S_before": sk, "S_after": sa, "a": a})
    landing = producer_landing_stats(land_rows)
    print("=== Kernel focus P(land) (merged) ===", flush=True)
    for ck, st in sorted(landing.items()):
        print(
            f"  {ck} →{st['target_land']}: n={st['n']} P={st['P_land']} hist={st['after_hist']}",
            flush=True,
        )

    hard_paired = []
    hard_by_pol = {p: [] for p in HARD_POLICIES}
    hard_by_m = defaultdict(lambda: {p: [] for p in HARD_POLICIES})
    prior_hard = {}
    if args.eight_way_only:
        # Preserve hard-gate block from a previous hard-only run
        for path in (OUT, ROOT / "data" / "results" / "sync_phase9j_sink_seeking_hard_r2.json"):
            if not path.exists():
                continue
            prior = json.loads(path.read_text())
            if prior.get("hard_paired"):
                prior_hard = prior
                hard_paired = prior.get("hard_paired") or []
                hard_aggs_prior = prior.get("hard_agg") or {}
                hard_per_prior = prior.get("hard_per_mstar") or {}
                for pair in hard_paired:
                    mk = pair.get("mstar_key") or _key(pair["mstar"])
                    for pol in HARD_POLICIES:
                        if pol in pair and isinstance(pair[pol], dict):
                            hard_by_pol[pol].append(pair[pol])
                            hard_by_m[mk][pol].append(pair[pol])
                print(
                    f"reloaded {len(hard_paired)} hard pairs from {path.name}",
                    flush=True,
                )
                break

    if not args.eight_way_only:
        print(
            f"=== 9J hard-target Γ' vs sink-seek λ={args.lam} "
            f"reps={args.reps_hard} max_steps={args.max_steps} ===",
            flush=True,
        )
        n_tot = len(HARD_MSTAR) * args.reps_hard
        idx = 0
        for mi, mstar in enumerate(HARD_MSTAR):
            for r in range(args.reps_hard):
                seed0 = args.seed + 300 * mi + 19 * r
                S0, plan0, seed = free_S0(p8k, sc, loaded, mstar=mstar, seed0=seed0)
                idx += 1
                print(
                    f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                    f"E0={_E(mstar, S0)}",
                    flush=True,
                )
                if S0 == list(mstar):
                    print("    skip S0==m*", flush=True)
                    continue
                ep_cache: dict = {}
                trials = {}
                for pol in HARD_POLICIES:
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
                        lam=args.lam,
                        hard_set=hard_set,
                        continue_after_land=True,
                    )
                    t = trials[pol]
                    tag = {"gamma_prime": "Γ'", "sink_seek": "SS"}[pol]
                    ret = t.get("retention")
                    ret_s = (
                        f" exit={ret['exit_a']}→{_key(ret['S_after_exit'])} stay={ret['stayed']}"
                        if ret
                        else ""
                    )
                    print(
                        f"    {tag}: hit={t['hit']} ever={t['ever_reach']} "
                        f"E={t['E0']}→{t['E_final']} acts={t['actions']} "
                        f"path={[ _key(s) for s in t['traj_S'] ]}{ret_s}",
                        flush=True,
                    )
                    hard_by_pol[pol].append(t)
                    hard_by_m[_key(mstar)][pol].append(t)
                hard_paired.append(
                    {
                        "mstar": list(mstar),
                        "mstar_key": _key(mstar),
                        "seed": seed,
                        "S0": S0,
                        **trials,
                    }
                )

    hard_aggs = {p: _agg(hard_by_pol[p]) for p in HARD_POLICIES}
    hard_per = {
        _key(m): {p: _agg(hard_by_m[_key(m)][p]) for p in HARD_POLICIES}
        for m in HARD_MSTAR
    }
    hard_reach = {
        _key(m): {
            p: (hard_per[_key(m)][p].get("P_ever") or 0) > 0 for p in HARD_POLICIES
        }
        for m in HARD_MSTAR
    }
    n_ss = sum(1 for mk, v in hard_reach.items() if v.get("sink_seek"))
    n_gp = sum(1 for mk, v in hard_reach.items() if v.get("gamma_prime"))
    hard_gate_ok = n_ss == 3

    eight_paired = []
    eight_by_pol = {p: [] for p in EIGHT_POLICIES}
    eight_by_m = defaultdict(lambda: {p: [] for p in EIGHT_POLICIES})
    ran_8way = False

    if (hard_gate_ok or args.force_8way) and not args.hard_only and not args.densify_only:
        ran_8way = True
        print(
            f"=== 9J full 8-way hybrid vs Γ' (gate_ok={hard_gate_ok}) "
            f"reps={args.reps_8way} ===",
            flush=True,
        )
        n_tot = 8 * args.reps_8way
        idx = 0
        for mi, mstar in enumerate(ALL_MSTAR):
            for r in range(args.reps_8way):
                seed0 = args.seed + 900 + 200 * mi + 19 * r
                S0, plan0, seed = free_S0(p8k, sc, loaded, mstar=mstar, seed0=seed0)
                idx += 1
                print(
                    f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                    f"E0={_E(mstar, S0)}",
                    flush=True,
                )
                if S0 == list(mstar):
                    continue
                ep_cache = {}
                trials = {}
                for pol in EIGHT_POLICIES:
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
                        lam=args.lam,
                        hard_set=hard_set,
                        continue_after_land=True,
                    )
                    t = trials[pol]
                    tag = {"gamma_prime": "Γ'", "hybrid": "hyb"}[pol]
                    print(
                        f"    {tag}: hit={t['hit']} ever={t['ever_reach']} "
                        f"E={t['E0']}→{t['E_final']} acts={t['actions']} "
                        f"path={[ _key(s) for s in t['traj_S'] ]}",
                        flush=True,
                    )
                    eight_by_pol[pol].append(t)
                    eight_by_m[_key(mstar)][pol].append(t)
                eight_paired.append(
                    {
                        "mstar": list(mstar),
                        "mstar_key": _key(mstar),
                        "seed": seed,
                        "S0": S0,
                        **trials,
                    }
                )

    eight_aggs = {p: _agg(eight_by_pol[p]) for p in EIGHT_POLICIES}
    eight_per = {
        _key(m): {p: _agg(eight_by_m[_key(m)][p]) for p in EIGHT_POLICIES}
        for m in ALL_MSTAR
    }
    eight_reach = {
        _key(m): {
            p: (eight_per[_key(m)][p].get("P_ever") or 0) > 0 for p in EIGHT_POLICIES
        }
        for m in ALL_MSTAR
    }
    n_hyb = sum(1 for mk, v in eight_reach.items() if v.get("hybrid")) if ran_8way else 0
    n_gp8 = sum(1 for mk, v in eight_reach.items() if v.get("gamma_prime")) if ran_8way else 0

    gate = {
        "hypothesis": (
            "Sink-seeking λ=0 recovers P(ever)>0 for {000,001,110} where Γ' fails; "
            "gap is action-selection, not missing actuators"
        ),
        "lambda": args.lam,
        "landing": landing,
        "hard_P_ever": {
            p: {mk: hard_per[mk][p].get("P_ever") for mk in hard_per} for p in HARD_POLICIES
        },
        "hard_P_hit": {
            p: {mk: hard_per[mk][p].get("P_hit") for mk in hard_per} for p in HARD_POLICIES
        },
        "n_hard_ever_sink_seek": n_ss,
        "n_hard_ever_gamma_prime": n_gp,
        "hard_gate_all_three": hard_gate_ok,
        "hard_reach": hard_reach,
        "ran_8way": ran_8way,
        "n_mstar_ever_hybrid": n_hyb,
        "n_mstar_ever_gamma_prime_8way": n_gp8,
        "universal_ever_hybrid": n_hyb == 8 if ran_8way else None,
        "read": (
            "Primary hard gate: ∀m*∈{000,001,110} P(ever)>0 under sink-seek. "
            "Retention P(stay|land) is secondary. Full 8-way only if hard gate holds."
        ),
    }

    payload = {
        "protocol": "Phase 9J sink-seeking hard targets + optional 8-way",
        "seed": args.seed,
        "lambda": args.lam,
        "reps_expand": args.reps_expand,
        "reps_hard": args.reps_hard,
        "reps_8way": args.reps_8way,
        "max_steps": args.max_steps,
        "expand_stats": expand_stats,
        "landing": landing,
        "hard_agg": hard_aggs,
        "hard_per_mstar": hard_per,
        "hard_paired": hard_paired,
        "eight_agg": eight_aggs if ran_8way else {},
        "eight_per_mstar": eight_per if ran_8way else {},
        "eight_paired": eight_paired,
        "eight_reachability": eight_reach if ran_8way else {},
        "gate": gate,
        "sync_error_norm_ref": sync_error_norm,
    }
    # drop huge by_a from steps in saved paired for readability? keep full for analysis
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9J — Sink-seeking for hard targets",
        "",
        r"> $a^*=\arg\max_a\big[P(S'=m^*\mid s,a)+\lambda\Gamma'(a)\big]$. "
        f"First test $\\lambda={args.lam}$. Frozen 8K. No new $v$.",
        "",
        f"expand_reps={args.reps_expand}, hard_reps={args.reps_hard}, "
        f"max_steps={args.max_steps}, seed={args.seed}.",
        "",
        "## Producer landing estimates",
        "",
        "| cell | target | n | $P(\\mathrm{land})$ | hist |",
        "|------|--------|--:|--------------------:|------|",
    ]
    for ck, st in sorted(landing.items()):
        hist = " ".join(f"{k}:{v}" for k, v in list((st.get("after_hist") or {}).items())[:5])
        lines.append(
            f"| `{ck}` | `{st['target_land']}` | {st['n']} | "
            f"**{fmt(st.get('P_land'))}** | {hist} |"
        )

    if hard_paired:
        lines += [
            "",
            "## Hard-target live: $\\Gamma'$ vs sink-seek",
            "",
            "| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\\Delta E$ | $P(\\mathrm{stay}\\mid\\mathrm{land})$ |",
            "|--------|--:|----------:|-----------:|-----------------:|---------------------------------------:|",
        ]
        for p, lab in (("sink_seek", "**sink-seek**"), ("gamma_prime", "$\\Gamma'$")):
            a = hard_aggs[p]
            lines.append(
                f"| {lab} | {a['n']} | {fmt(a.get('P_hit'))} | **{fmt(a.get('P_ever'))}** | "
                f"{fmt(a.get('mean_delta_E'))} | {fmt(a.get('P_stay_after_land'))} |"
            )
        lines += [
            "",
            "### Per hard $m^*$",
            "",
            "| $m^*$ | $P_{ever}$ SS | $P_{hit}$ SS | $P_{ever}$ $\\Gamma'$ | $P_{hit}$ $\\Gamma'$ | stay|land SS |",
            "|-------|---------------:|-------------:|---------------------:|---------------------:|-------------------:|",
        ]
        for m in HARD_MSTAR:
            mk = _key(m)
            st = hard_per[mk]
            lines.append(
                f"| `{mk}` | **{fmt(st['sink_seek'].get('P_ever'))}** | "
                f"{fmt(st['sink_seek'].get('P_hit'))} | {fmt(st['gamma_prime'].get('P_ever'))} | "
                f"{fmt(st['gamma_prime'].get('P_hit'))} | "
                f"{fmt(st['sink_seek'].get('P_stay_after_land'))} |"
            )
        lines += [
            "",
            "### Retention exits (sink-seek landings)",
            "",
        ]
        for mk in (_key(m) for m in HARD_MSTAR):
            exits = hard_per[mk]["sink_seek"].get("retention_exits") or []
            if not exits:
                lines.append(f"- `{mk}`: no landings logged")
                continue
            bits = ", ".join(
                f"{e['exit_a']}→`{e['S_after']}` stay={e['stayed']}" for e in exits[:6]
            )
            lines.append(f"- `{mk}`: {bits}")

        lines += [
            "",
            "## Hard gate",
            "",
            f"- $P_{{\\mathrm{{ever}}}}>0$ for all three under sink-seek: **{hard_gate_ok}** "
            f"({n_ss}/3; $\\Gamma'$ {n_gp}/3)",
            f"- $\\Delta$ aggregate $P_{{\\mathrm{{ever}}}}$(SS$-\\Gamma'$): "
            f"{fmt((hard_aggs['sink_seek'].get('P_ever') or 0) - (hard_aggs['gamma_prime'].get('P_ever') or 0))}",
            "",
        ]

    if ran_8way:
        lines += [
            "## Full 8-way: hybrid ($\\Gamma'$ soft + sink-seek hard) vs $\\Gamma'$",
            "",
            "| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\\Delta E$ | ever $m^*$ |",
            "|--------|--:|----------:|-----------:|-----------------:|-----------:|",
        ]
        for p, lab in (("hybrid", "**hybrid**"), ("gamma_prime", "$\\Gamma'$")):
            a = eight_aggs[p]
            n_r = sum(1 for mk, v in eight_reach.items() if v.get(p))
            lines.append(
                f"| {lab} | {a['n']} | {fmt(a.get('P_hit'))} | **{fmt(a.get('P_ever'))}** | "
                f"{fmt(a.get('mean_delta_E'))} | **{n_r}/8** |"
            )
        lines += [
            "",
            "| $m^*$ | $P_{ever}$ hyb | $P_{hit}$ hyb | $P_{ever}$ $\\Gamma'$ |",
            "|-------|---------------:|--------------:|---------------------:|",
        ]
        for m in ALL_MSTAR:
            mk = _key(m)
            st = eight_per[mk]
            lines.append(
                f"| `{mk}` | {fmt(st['hybrid'].get('P_ever'))} | "
                f"{fmt(st['hybrid'].get('P_hit'))} | {fmt(st['gamma_prime'].get('P_ever'))} |"
            )
        lines += [
            "",
            f"Universal ever-reach (hybrid): **{gate['universal_ever_hybrid']}**",
            "",
        ]
    elif hard_paired and not hard_gate_ok:
        lines += [
            "## Full 8-way",
            "",
            "_Skipped — hard gate failed (use `--force-8way` to override)._",
            "",
        ]

    lines += [
        gate["read"],
        "",
        r"$$\boxed{a^*=\arg\max_a P(S'=m^*\mid s,a)\quad(\lambda=0)\text{ for hard }m^*}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
