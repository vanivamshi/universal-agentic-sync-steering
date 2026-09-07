#!/usr/bin/env python3
"""Phase 9C — Densify weak one-bit edges; rebuild planner without Laplace.

Protocol for edge s → s':
  1) free-run
  2) explicitly construct source s via greedy one-bit flips toward s
     (not rare natural visits; not Laplace fill)
  3) only if S == s, apply one-bit transition to s' and count

Collect until n_at_s >= --target-n (default 8) or --max-attempts.

Estimates (no planner Laplace):
  p_hat = k/n
  p ~ Beta(k+1, n-k+1)
  p_plan = 5th percentile of Beta

Merge densified counts with Phase 9B raw counts for other edges.
Test strong connectivity of the digraph with edges p_plan ≥ τ
for τ ∈ {0.5, 0.7}.

Critical budget: edges into/out of 000 and 110.

  .venv/bin/python scripts/run_phase9c_densify_edges.py --target-n 8
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9c_densify_edges.json"
MD = ROOT / "data" / "results" / "sync_phase9c_densify_edges.md"
P9B = ROOT / "data" / "results" / "sync_phase9_transition_graph.json"

SEED = 20260923
CHANNELS = ("C", "H", "O")
STATES = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]

# Full sink neighborhood (reference). Prefer --edges remaining for budget.
CRITICAL_EDGES: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((0, 0, 1), (0, 0, 0)),  # 001→000 O  — source hard; usually abort
    ((0, 1, 0), (0, 0, 0)),  # 010→000 H  — DONE 0/8
    ((1, 0, 0), (0, 0, 0)),  # 100→000 C  — DONE 0/8
    ((0, 1, 0), (1, 1, 0)),  # 010→110 C  — DONE 0/8
    ((1, 0, 0), (1, 1, 0)),  # 100→110 H  — last soft→110 hope
    ((1, 1, 1), (1, 1, 0)),  # 111→110 O  — last non-soft in to 110
    ((1, 0, 1), (1, 0, 0)),  # 101→100 O  — soft reverse (user-named)
]

# Skip already-settled / already-strong; only unknowns that still matter for SCC
REMAINING_EDGES: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((1, 0, 0), (1, 1, 0)),  # 100→110 H
    ((1, 1, 1), (1, 1, 0)),  # 111→110 O
    ((1, 0, 1), (1, 0, 0)),  # 101→100 O
]


def _load_p9():
    path = ROOT / "scripts" / "run_phase9_hamming_paths.py"
    spec = importlib.util.spec_from_file_location("phase9", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s: tuple[int, ...] | list[int]) -> str:
    return "".join(str(int(x)) for x in s)


def _ek(a, b) -> str:
    return f"{_key(a)}→{_key(b)}"


def _neighbors(s: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    out = []
    for i in range(3):
        t = list(s)
        t[i] = 1 - t[i]
        out.append(tuple(t))
    return out


def _all_directed_edges():
    return [(s, t) for s in STATES for t in _neighbors(s)]


def _beta_p05(k: int, n: int) -> float:
    """5th percentile of Beta(k+1, n-k+1)."""
    if n <= 0:
        return 0.0
    k = int(max(0, min(k, n)))
    from scipy.stats import beta as beta_dist

    a, b = k + 1.0, (n - k) + 1.0
    return float(beta_dist.ppf(0.05, a, b))


# Soft states from 8K validation (reachable); use as bridges into hard sources
SOFT_STATES = {(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)}  # 010,011,100,101


def construct_source(
    p9,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    source: tuple[int, int, int],
    seed: int,
    max_flips: int = 3,
) -> dict[str, Any]:
    """Explicitly build source: free → 8K home → optional soft-bridge → flips."""
    home = p9._home_to_source(
        p8k, sc, loaded, dirs=dirs, source=source, seed=seed
    )
    S = list(home["S_home"])
    plan = home.get("plan_prefill") or ""
    flips = []
    n_eps = int(home.get("n_episodes") or 1)
    method = "home" if home["homed"] else "home_miss"

    def _done() -> bool:
        return S == list(source)

    if _done():
        return {
            "S_free": home["S_free"],
            "S": S,
            "at_source": 1,
            "plan_prefill": plan,
            "flips": flips,
            "n_episodes": n_eps,
            "method": method,
        }

    # Soft-bridge: if source is hard, home to a Hamming-1 soft neighbor then one-bit in
    if tuple(source) not in SOFT_STATES:
        bridges = [n for n in _neighbors(source) if n in SOFT_STATES]
        for br in bridges:
            br_home = p9._home_to_source(
                p8k, sc, loaded, dirs=dirs, source=br, seed=seed + 7 + bridges.index(br)
            )
            n_eps += int(br_home.get("n_episodes") or 1)
            if not br_home["homed"]:
                continue
            edge = p9._run_one_bit_transition(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                from_s=br,
                to_s=source,
                seed=seed + 33 + bridges.index(br),
                plan_prefill=br_home["plan_prefill"],
            )
            n_eps += 1
            S = edge["S"]
            plan = edge.get("plan_prefill") or br_home["plan_prefill"]
            flips.append(
                {
                    "bridge_from": list(br),
                    "to_target": list(source),
                    "S": list(S),
                    "hit": int(S == list(source)),
                    "channel": edge["channel"],
                }
            )
            method = f"soft_bridge_{_key(br)}"
            if _done():
                return {
                    "S_free": home["S_free"],
                    "S": S,
                    "at_source": 1,
                    "plan_prefill": plan,
                    "flips": flips,
                    "n_episodes": n_eps,
                    "method": method,
                }

    # residual: greedy one-bit flips toward source from current S
    for step in range(max_flips):
        if _done():
            break
        i = next(j for j in range(3) if int(S[j]) != int(source[j]))
        to = list(S)
        to[i] = int(source[i])
        from_s = tuple(S)
        to_s = tuple(to)
        edge = p9._run_one_bit_transition(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            from_s=from_s,
            to_s=to_s,
            seed=seed + 11 * (step + 1),
            plan_prefill=plan,
        )
        n_eps += 1
        S = edge["S"]
        if edge.get("plan_prefill"):
            plan = edge["plan_prefill"]
        flips.append(
            {
                "from": list(from_s),
                "to_target": list(to_s),
                "S": list(S),
                "hit": int(S == list(to_s)),
                "channel": edge["channel"],
            }
        )
    if flips and method.startswith("home"):
        method = "home+flips"
    elif flips and "bridge" not in method:
        method = "flips"
    return {
        "S_free": home["S_free"],
        "S": S,
        "at_source": int(_done()),
        "plan_prefill": plan,
        "flips": flips,
        "n_episodes": n_eps,
        "method": method,
    }


def densify_edge(
    p9,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    s: tuple[int, int, int],
    t: tuple[int, int, int],
    seed0: int,
    target_n: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Collect conditional transitions from explicitly constructed s."""
    rows = []
    n_at = 0
    n_hit = 0
    attempts = 0
    construct_fail = 0
    streak_fail = 0
    abort_after_fail_streak = max(8, target_n)
    while n_at < target_n and attempts < max_attempts:
        seed = seed0 + 17 * attempts
        attempts += 1
        print(
            f"    att={attempts}/{max_attempts} construct {_key(s)} …",
            flush=True,
        )
        cons = construct_source(
            p9, p8k, sc, loaded, dirs=dirs, source=s, seed=seed
        )
        if not cons["at_source"]:
            construct_fail += 1
            streak_fail += 1
            print(
                f"    att={attempts} construct_fail S={_key(cons['S'])} "
                f"method={cons.get('method')} flips={len(cons.get('flips') or [])} "
                f"streak_fail={streak_fail}",
                flush=True,
            )
            rows.append(
                {
                    "attempt": attempts,
                    "seed": seed,
                    "constructed": False,
                    "S_after_construct": cons["S"],
                    "method": cons.get("method"),
                    "flips": cons.get("flips"),
                    "hit": None,
                }
            )
            if n_at == 0 and streak_fail >= abort_after_fail_streak:
                print(
                    f"    abort: {streak_fail} construct fails with n_at=0 "
                    f"(source {_key(s)} not reproducible under 8K)",
                    flush=True,
                )
                break
            continue
        streak_fail = 0
        n_at += 1
        edge = p9._run_one_bit_transition(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            from_s=s,
            to_s=t,
            seed=seed + 99,
            plan_prefill=cons["plan_prefill"],
        )
        hit = int(edge["S"] == list(t))
        n_hit += hit
        rows.append(
            {
                "attempt": attempts,
                "seed": seed,
                "constructed": True,
                "S_after_construct": cons["S"],
                "S_after_edge": edge["S"],
                "hit": hit,
                "channel": edge["channel"],
                "method": cons.get("method"),
                "flips": cons["flips"],
            }
        )
        print(
            f"    att={attempts} constructed=1 n_at={n_at}/{target_n} "
            f"S→{_key(edge['S'])} hit={hit} method={cons.get('method')}",
            flush=True,
        )

    p_hat = float(n_hit / n_at) if n_at > 0 else float("nan")
    p_plan = _beta_p05(n_hit, n_at) if n_at > 0 else 0.0
    return {
        "edge": _ek(s, t),
        "from": list(s),
        "to": list(t),
        "channel": p9._bit_diff(s, t),
        "n": n_at,
        "k": n_hit,
        "p_hat": p_hat,
        "p_plan_p05": p_plan,
        "attempts": attempts,
        "construct_fail": construct_fail,
        "construct_rate": float((attempts - construct_fail) / attempts) if attempts else 0.0,
        "rows": rows,
    }


def _best_paths(P: dict[str, dict[str, float]], *, min_p: float) -> dict:
    nodes = [_key(s) for s in STATES]
    result = {u: {} for u in nodes}
    for src in nodes:
        dist = {v: float("inf") for v in nodes}
        prev: dict[str, str | None] = {v: None for v in nodes}
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, p in (P.get(u) or {}).items():
                if not (p == p) or p < min_p:
                    continue
                nd = d - math.log(max(p, 1e-12))
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        for tgt in nodes:
            if dist[tgt] == float("inf"):
                result[src][tgt] = {
                    "reachable": False,
                    "path": None,
                    "P_path": 0.0,
                    "n_hops": None,
                }
                continue
            path = [tgt]
            cur = tgt
            while prev[cur] is not None:
                cur = prev[cur]
                path.append(cur)
            path.reverse()
            prod = 1.0
            for a, b in zip(path, path[1:]):
                prod *= float(P[a][b])
            result[src][tgt] = {
                "reachable": True,
                "path": path,
                "P_path": float(1.0 if src == tgt else prod),
                "n_hops": 0 if src == tgt else len(path) - 1,
            }
    return result


def _strongly_connected(adj: dict[str, set[str]]) -> bool:
    """Tarjan / Kosaraju: check one SCC covering all 8 nodes."""
    nodes = [_key(s) for s in STATES]
    # Kosaraju
    def dfs(u, graph, seen, order):
        seen.add(u)
        for v in graph.get(u, ()):
            if v not in seen:
                dfs(v, graph, seen, order)
        order.append(u)

    order: list[str] = []
    seen: set[str] = set()
    for u in nodes:
        if u not in seen:
            dfs(u, adj, seen, order)
    # transpose
    radj: dict[str, set[str]] = {u: set() for u in nodes}
    for u, vs in adj.items():
        for v in vs:
            radj[v].add(u)
    seen2: set[str] = set()
    comps = 0

    def dfs2(u):
        seen2.add(u)
        for v in radj.get(u, ()):
            if v not in seen2:
                dfs2(v)

    for u in reversed(order):
        if u not in seen2:
            dfs2(u)
            comps += 1
    return comps == 1 and len(seen2) == 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-n", type=int, default=8, help="conditional trials per edge")
    ap.add_argument("--max-attempts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--edges",
        default="remaining",
        help="remaining | critical | all | comma list like 100→110,111→110",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="merge prior densified stats from .partial.json",
    )
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p9 = _load_p9()
    p8k = p9._load_8k()
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()

    if args.edges == "critical":
        edge_list = list(CRITICAL_EDGES)
    elif args.edges == "remaining":
        edge_list = list(REMAINING_EDGES)
    elif args.edges == "all":
        edge_list = _all_directed_edges()
    else:
        edge_list = []
        for tok in args.edges.split(","):
            a, b = tok.strip().split("→")
            edge_list.append((tuple(int(x) for x in a), tuple(int(x) for x in b)))

    print(
        f"=== Phase 9C densify n_edges={len(edge_list)} target_n={args.target_n} "
        f"max_att={args.max_attempts} (explicit source construct) ===",
        flush=True,
    )
    print(f"  edges: {[_ek(a, b) for a, b in edge_list]}", flush=True)

    densified: dict[str, Any] = {}
    checkpoint = ROOT / "data" / "results" / "sync_phase9c_densify_edges.partial.json"
    if args.resume and checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        for ek, st in prior.items():
            densified[ek] = dict(st)
            densified[ek].setdefault("rows", [])
        print(f"  resumed {len(prior)} edges from {checkpoint.name}", flush=True)
    for i, (s, t) in enumerate(edge_list):
        ek = _ek(s, t)
        prior = densified.get(ek)
        if prior and int(prior.get("n") or 0) >= args.target_n:
            print(
                f"  [{i+1}/{len(edge_list)}] skip {ek} "
                f"(already n={prior['n']} k={prior['k']})",
                flush=True,
            )
            continue
        print(f"  [{i+1}/{len(edge_list)}] densify {ek}", flush=True)
        densified[ek] = densify_edge(
            p9,
            p8k,
            sc,
            loaded,
            dirs=dirs,
            s=s,
            t=t,
            seed0=args.seed + 1000 * i,
            target_n=args.target_n,
            max_attempts=args.max_attempts,
        )
        d = densified[ek]
        print(
            f"    >> n={d['n']} k={d['k']} p_hat={d['p_hat']} "
            f"p_plan05={d['p_plan_p05']:.3f} construct_rate={d['construct_rate']:.2f}",
            flush=True,
        )
        checkpoint.write_text(
            json.dumps(
                {
                    kk: {a: b for a, b in vv.items() if a != "rows"}
                    for kk, vv in densified.items()
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # Merge with 9B raw counts for non-densified edges
    merged: dict[str, dict[str, Any]] = {}
    p9b = {}
    if P9B.exists():
        p9b = json.loads(P9B.read_text())
        for ek, st in (p9b.get("edge_stats") or {}).items():
            # reconstruct raw from hits/n (unconditional) — prefer homed if present
            n_h = int(st.get("n_homed") or 0)
            k_h = int(st.get("hits_homed") or 0)
            n_u = int(st.get("n") or 0)
            k_u = int(st.get("hits") or 0)
            if n_h > 0:
                n, k = n_h, k_h
                source = "9B_homed"
            else:
                n, k = n_u, k_u
                source = "9B_uncond"
            merged[ek] = {
                "n": n,
                "k": k,
                "p_hat": float(k / n) if n else float("nan"),
                "p_plan_p05": _beta_p05(k, n) if n else 0.0,
                "source": source,
                "channel": st.get("channel"),
            }

    for ek, d in densified.items():
        merged[ek] = {
            "n": d["n"],
            "k": d["k"],
            "p_hat": d["p_hat"],
            "p_plan_p05": d["p_plan_p05"],
            "source": "9C_conditional",
            "channel": d["channel"],
            "construct_rate": d["construct_rate"],
            "attempts": d["attempts"],
        }

    # Ensure all 24 edges exist
    for s, t in _all_directed_edges():
        ek = _ek(s, t)
        if ek not in merged:
            merged[ek] = {
                "n": 0,
                "k": 0,
                "p_hat": float("nan"),
                "p_plan_p05": 0.0,
                "source": "missing",
                "channel": p9._bit_diff(s, t),
            }

    # Build planner graphs at τ
    results_tau: dict[str, Any] = {}
    for tau in (0.5, 0.7):
        P: dict[str, dict[str, float]] = {_key(s): {} for s in STATES}
        adj: dict[str, set[str]] = {_key(s): set() for s in STATES}
        kept = []
        for s, t in _all_directed_edges():
            ek = _ek(s, t)
            p = float(merged[ek]["p_plan_p05"])
            if p >= tau:
                P[_key(s)][_key(t)] = p
                adj[_key(s)].add(_key(t))
                kept.append((ek, p))
        paths = _best_paths(P, min_p=tau)
        n_pairs = 56
        n_reach = sum(
            1
            for u in paths
            for v, info in paths[u].items()
            if u != v and info["reachable"]
        )
        scc = _strongly_connected(adj)
        results_tau[str(tau)] = {
            "tau": tau,
            "n_edges_kept": len(kept),
            "edges_kept": kept,
            "n_reachable_pairs": n_reach,
            "n_pairs": n_pairs,
            "strongly_connected": scc,
            "path_000_111": paths.get("000", {}).get("111"),
            "paths": paths,
        }
        print(
            f"  τ={tau}: edges={len(kept)} reach={n_reach}/{n_pairs} SCC={scc} "
            f"000→111={paths.get('000', {}).get('111', {}).get('path')}",
            flush=True,
        )

    gate = {
        "hypothesis": (
            "With empirical conditional edges + Beta p05 planning, "
            "the 8-cube becomes strongly connected at practical τ"
        ),
        "densified_edges": list(densified.keys()),
        "target_n": args.target_n,
        "tau_0.5_SCC": results_tau["0.5"]["strongly_connected"],
        "tau_0.7_SCC": results_tau["0.7"]["strongly_connected"],
        "tau_0.5_reach": results_tau["0.5"]["n_reachable_pairs"],
        "tau_0.7_reach": results_tau["0.7"]["n_reachable_pairs"],
        "universal_via_paths_0.5": bool(
            results_tau["0.5"]["strongly_connected"]
            and results_tau["0.5"]["n_reachable_pairs"] == 56
        ),
        "universal_via_paths_0.7": bool(
            results_tau["0.7"]["strongly_connected"]
            and results_tau["0.7"]["n_reachable_pairs"] == 56
        ),
        "path_000_111_at_0.5": results_tau["0.5"]["path_000_111"],
        "read": (
            "No Laplace planner. p_plan = Beta(k+1,n-k+1) 5th pct. "
            "Universal multi-step claim only if SCC at τ=0.5 (practical) or 0.7 (reliable)."
        ),
    }

    def _jsonable(x: Any) -> Any:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_jsonable(v) for v in x]
        return x

    payload = {
        "protocol": "Phase 9C densify weak edges + Beta-p05 planner",
        "seed": args.seed,
        "target_n": args.target_n,
        "max_attempts": args.max_attempts,
        "critical_edges": [_ek(a, b) for a, b in CRITICAL_EDGES],
        "densified": {
            k: {kk: vv for kk, vv in d.items() if kk != "rows"} for k, d in densified.items()
        },
        "densified_trials": {k: d.get("rows") for k, d in densified.items()},
        "merged_edges": merged,
        "by_tau": {
            t: {
                **{k: v for k, v in blob.items() if k != "paths"},
                "path_examples": {
                    "000→111": blob["paths"].get("000", {}).get("111"),
                    "111→000": blob["paths"].get("111", {}).get("000"),
                    "100→110": blob["paths"].get("100", {}).get("110"),
                    "110→100": blob["paths"].get("110", {}).get("100"),
                },
            }
            for t, blob in results_tau.items()
        },
        "all_pairs_paths_0.5": results_tau["0.5"]["paths"],
        "all_pairs_paths_0.7": results_tau["0.7"]["paths"],
        "gate": gate,
    }
    OUT.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    lines = [
        "# Phase 9C — Densify weak edges (Beta $p_{05}$ planner)",
        "",
        r"> Explicit source construction + conditional one-bit trials. "
        r"No Laplace. $p_{\mathrm{plan}}=\mathrm{Beta}(k{+}1,n{-}k{+}1)$ 5th pct.",
        "",
        f"target_n={args.target_n}, max_attempts={args.max_attempts}, seed={args.seed}.",
        "",
        "## Densified edges (000 / 110 neighborhood)",
        "",
        "| edge | ch | n | k | $\\hat p$ | $p_{plan,05}$ | construct_rate |",
        "|------|----|--:|--:|--------:|--------------:|---------------:|",
    ]
    for s, t in edge_list:
        ek = _ek(s, t)
        d = densified[ek]
        ph = d["p_hat"]
        lines.append(
            f"| `{ek}` | {d['channel']} | {d['n']} | {d['k']} | "
            f"{ph if ph==ph else float('nan'):.2f} | {d['p_plan_p05']:.3f} | "
            f"{d['construct_rate']:.2f} |"
        )

    lines += [
        "",
        "## Strong connectivity",
        "",
        "| $\\tau$ | edges kept | reachable pairs | strongly connected |",
        "|--------:|-----------:|----------------:|:------------------:|",
    ]
    for tau in ("0.5", "0.7"):
        b = results_tau[tau]
        lines.append(
            f"| {tau} | {b['n_edges_kept']} | {b['n_reachable_pairs']}/56 | "
            f"**{b['strongly_connected']}** |"
        )

    p05 = results_tau["0.5"]["path_000_111"] or {}
    lines += [
        "",
        f"$000\\to111$ at $\\tau=0.5$: `{p05.get('path')}` P≈{p05.get('P_path')}",
        "",
        "## Gate",
        "",
        f"- SCC at $\\tau=0.5$: **{gate['tau_0.5_SCC']}**",
        f"- SCC at $\\tau=0.7$: **{gate['tau_0.7_SCC']}**",
        f"- Universal via paths @0.5: **{gate['universal_via_paths_0.5']}**",
        f"- Universal via paths @0.7: **{gate['universal_via_paths_0.7']}**",
        "",
        gate["read"],
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": {k: v for k, v in gate.items() if k != "read"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
