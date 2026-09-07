#!/usr/bin/env python3
"""Phase 9B — Empirical 8-state one-bit transition graph (frozen 8K).

For every directed Hamming edge s → s' (‖s−s'‖₁ = 1), estimate:

  P(s → s')

Protocol per trial:
  free → 8K home toward s → one-bit actuator toward s'

Primary estimate: P(hit s' | S_home == s)
Secondary: P(hit s') after home attempt (even if not at s)

Then all-pairs best paths under edge weights −log P (Laplace-smoothed),
asking whether the cube is mutually reachable via multi-step transitions.

No new v. No aggregate 8-way. Final path success = product of one-bit edges.

  .venv/bin/python scripts/run_phase9_transition_graph.py --reps 2
  .venv/bin/python scripts/run_phase9_transition_graph.py --reps 3
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

OUT = ROOT / "data" / "results" / "sync_phase9_transition_graph.json"
MD = ROOT / "data" / "results" / "sync_phase9_transition_graph.md"

SEED = 20260922
CHANNELS = ("C", "H", "O")
STATES = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
EPS = 1e-6
LAPLACE = 0.5  # (hits + α) / (n + 2α)


def _load_p9():
    path = ROOT / "scripts" / "run_phase9_hamming_paths.py"
    spec = importlib.util.spec_from_file_location("phase9", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s: tuple[int, ...] | list[int]) -> str:
    return "".join(str(int(x)) for x in s)


def _neighbors(s: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    out = []
    for i in range(3):
        t = list(s)
        t[i] = 1 - t[i]
        out.append(tuple(t))
    return out


def _all_directed_edges() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    edges = []
    for s in STATES:
        for t in _neighbors(s):
            edges.append((s, t))
    return edges  # 24


def _smooth(hits: int, n: int, alpha: float = LAPLACE) -> float:
    if n <= 0:
        return float("nan")
    return float((hits + alpha) / (n + 2 * alpha))


def _best_paths(
    P: dict[str, dict[str, float]],
    *,
    min_p: float = EPS,
) -> dict[str, dict[str, Any]]:
    """All-pairs max-product paths via Dijkstra on −log P."""
    nodes = [_key(s) for s in STATES]
    result: dict[str, dict[str, Any]] = {u: {} for u in nodes}
    for src in nodes:
        # max log-prob = min -log
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
                nd = d - math.log(max(p, min_p))
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
            # reconstruct
            path = [tgt]
            cur = tgt
            while prev[cur] is not None:
                cur = prev[cur]
                path.append(cur)
            path.reverse()
            # product along path
            prod = 1.0
            for a, b in zip(path, path[1:]):
                prod *= float(P[a][b])
            result[src][tgt] = {
                "reachable": True,
                "path": path,
                "P_path": float(prod if src != tgt else 1.0),
                "n_hops": 0 if src == tgt else len(path) - 1,
                "neglog": float(dist[tgt]),
            }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2, help="trials per directed edge")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--min-edge-p",
        type=float,
        default=0.05,
        help="edges below this smoothed P are dropped from path search",
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
    edges = _all_directed_edges()

    print(
        f"=== Phase 9B transition graph n_edges={len(edges)} reps={args.reps} "
        f"(free→home→one-bit) ===",
        flush=True,
    )

    # edge_key -> list of trial rows
    trials: dict[str, list[dict]] = {}
    edge_i = 0
    for s, t in edges:
        ek = f"{_key(s)}→{_key(t)}"
        ch = p9._bit_diff(s, t)
        trials[ek] = []
        for r in range(args.reps):
            seed = args.seed + 1000 * edge_i + 10 * r
            print(f"  [{edge_i+1}/{len(edges)}] {ek} ({ch}) r={r}", flush=True)
            home = p9._home_to_source(p8k, sc, loaded, dirs=dirs, source=s, seed=seed)
            at_s = int(home["S_home"] == list(s))
            edge = p9._run_one_bit_transition(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                from_s=s,
                to_s=t,
                seed=seed + 77,
                plan_prefill=home["plan_prefill"],
            )
            hit = int(edge["S"] == list(t))
            row = {
                "edge": ek,
                "channel": ch,
                "from": list(s),
                "to": list(t),
                "S_free": home["S_free"],
                "S_home": home["S_home"],
                "homed": home["homed"],
                "at_from": at_s,
                "S_after": edge["S"],
                "hit": hit,
                "hit_if_homed": int(at_s and hit),
                "seed": seed,
            }
            trials[ek].append(row)
            print(
                f"    free={_key(home['S_free'])} home={_key(home['S_home'])} "
                f"homed={at_s} → {_key(edge['S'])} hit={hit}",
                flush=True,
            )
        edge_i += 1

    # summarize edges
    edge_stats: dict[str, Any] = {}
    P_cond: dict[str, dict[str, float]] = {_key(s): {} for s in STATES}
    P_uncond: dict[str, dict[str, float]] = {_key(s): {} for s in STATES}
    n_cond_total = 0
    n_cond_pos = 0

    for s, t in edges:
        ek = f"{_key(s)}→{_key(t)}"
        rows = trials[ek]
        hits = sum(r["hit"] for r in rows)
        n = len(rows)
        homed_rows = [r for r in rows if r["at_from"]]
        hits_h = sum(r["hit"] for r in homed_rows)
        n_h = len(homed_rows)
        n_cond_total += n_h
        n_cond_pos += hits_h
        p_u = _smooth(hits, n)
        p_c = _smooth(hits_h, n_h) if n_h > 0 else float("nan")
        # Prefer conditional when we have data; else uncond for path search
        p_use = p_c if n_h > 0 and p_c == p_c else p_u
        ch = p9._bit_diff(s, t)
        edge_stats[ek] = {
            "channel": ch,
            "n": n,
            "hits": hits,
            "P": p_u,
            "n_homed": n_h,
            "hits_homed": hits_h,
            "P_if_homed": p_c,
            "P_for_paths": p_use,
            "home_rate": float(n_h / n) if n else 0.0,
        }
        P_uncond[_key(s)][_key(t)] = p_u
        if n_h > 0:
            P_cond[_key(s)][_key(t)] = p_c

    # Path matrix: use P_for_paths (conditional when available)
    P_path_graph: dict[str, dict[str, float]] = {_key(s): {} for s in STATES}
    for s, t in edges:
        ek = f"{_key(s)}→{_key(t)}"
        p = edge_stats[ek]["P_for_paths"]
        if p == p:
            P_path_graph[_key(s)][_key(t)] = float(p)

    paths = _best_paths(P_path_graph, min_p=max(args.min_edge_p, EPS))

    # mutual reachability (excluding self)
    n_pairs = 8 * 7
    n_reach = sum(
        1
        for u in paths
        for v, info in paths[u].items()
        if u != v and info["reachable"] and info["P_path"] >= args.min_edge_p
    )
    # stronger: path product >= 0.05
    n_reach_strong = sum(
        1
        for u in paths
        for v, info in paths[u].items()
        if u != v and info["reachable"] and info["P_path"] >= 0.05
    )

    # highlight 000→111
    p_000_111 = paths.get("000", {}).get("111", {})
    # best outgoing / hardest edges
    ranked = sorted(
        edge_stats.items(),
        key=lambda kv: (
            -(kv[1]["P_if_homed"] if kv[1]["P_if_homed"] == kv[1]["P_if_homed"] else kv[1]["P"]),
            -kv[1]["home_rate"],
        ),
    )

    gate = {
        "hypothesis": (
            "8-state cube is mutually reachable via reliable one-bit transitions "
            "even when some direct multi-bit jumps fail"
        ),
        "n_directed_edges": 24,
        "reps": args.reps,
        "n_pairs": n_pairs,
        "n_reachable_pairs": n_reach,
        "n_reachable_pairs_Ppath_ge_0.05": n_reach_strong,
        "mutual_reachability_fraction": float(n_reach / n_pairs),
        "strong_mutual_fraction": float(n_reach_strong / n_pairs),
        "P_000_to_111_path": p_000_111.get("P_path"),
        "path_000_to_111": p_000_111.get("path"),
        "universal_via_paths": bool(n_reach_strong == n_pairs),
        "min_edge_p": args.min_edge_p,
        "read": (
            "Claim universal multi-step reachability only if all 56 ordered pairs "
            "have a path with non-trivial product probability. "
            "One-bit P(s→s'|at s) is the graph; intermediates are control path only."
        ),
    }

    payload = {
        "protocol": "Phase 9B 8-state one-bit transition graph",
        "reps": args.reps,
        "seed": args.seed,
        "laplace_alpha": LAPLACE,
        "controller": "frozen 8K",
        "edge_stats": edge_stats,
        "P_uncond": P_uncond,
        "P_cond_if_homed": P_cond,
        "P_for_paths": P_path_graph,
        "all_pairs_paths": paths,
        "gate": gate,
        "trials": trials,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # markdown
    lines = [
        "# Phase 9B — 8-state one-bit transition graph",
        "",
        r"> Frozen 8K. For each directed Hamming edge: free $\to$ home to $s$ $\to$ one-bit to $s'$. "
        r"Then all-pairs max-product paths. No new $v$.",
        "",
        f"reps={args.reps}/edge, n_edges=24, seed={args.seed}, "
        f"Laplace α={LAPLACE}, min_edge_p={args.min_edge_p}.",
        "",
        "## Edge matrix $P(s\\to s')$ (smoothed; prefer $|$homed when $n_{homed}{>}0$)",
        "",
        "| from\\to | "
        + " | ".join(_key(t) for t in STATES)
        + " |",
        "|------|"
        + "|".join(["------:" for _ in STATES])
        + "|",
    ]
    for s in STATES:
        row = [f"| **{_key(s)}** |"]
        for t in STATES:
            if s == t:
                row.append(" — |")
            elif _key(t) in P_path_graph[_key(s)]:
                p = P_path_graph[_key(s)][_key(t)]
                ek = f"{_key(s)}→{_key(t)}"
                nh = edge_stats[ek]["n_homed"]
                mark = "" if nh > 0 else "*"
                row.append(f" {p:.2f}{mark} |")
            else:
                row.append(" · |")
        lines.append("".join(row))
    lines += [
        "",
        "\\* = no successful homes; unconditional P after home attempt.",
        "",
        "## Strongest / weakest edges",
        "",
        "| edge | ch | P_homed | n_homed | P_uncond | home_rate |",
        "|------|----|--------:|--------:|---------:|----------:|",
    ]
    def _pf(x):
        return f"{x:.2f}" if x == x else "nan"

    for ek, st in ranked[:8]:
        lines.append(
            f"| `{ek}` | {st['channel']} | {_pf(st['P_if_homed'])} | "
            f"{st['n_homed']} | {st['P']:.2f} | {st['home_rate']:.2f} |"
        )
    lines.append("| … | | | | | |")
    for ek, st in ranked[-6:]:
        lines.append(
            f"| `{ek}` | {st['channel']} | {_pf(st['P_if_homed'])} | "
            f"{st['n_homed']} | {st['P']:.2f} | {st['home_rate']:.2f} |"
        )

    lines += [
        "",
        "## All-pairs reachability",
        "",
        f"- Reachable pairs (edge P≥{args.min_edge_p}): "
        f"**{n_reach}/{n_pairs}** ({100*n_reach/n_pairs:.0f}%)",
        f"- Strong paths (product P≥0.05): "
        f"**{n_reach_strong}/{n_pairs}** ({100*n_reach_strong/n_pairs:.0f}%)",
        f"- $000\\to111$ best path: `{p_000_111.get('path')}` "
        f"P≈{p_000_111.get('P_path')}",
        "",
        "### Selected path examples",
        "",
        "| source | target | path | P_path | hops |",
        "|-------:|-------:|------|-------:|-----:|",
    ]
    examples = [
        ("000", "111"),
        ("000", "110"),
        ("111", "000"),
        ("100", "111"),
        ("011", "100"),
        ("001", "110"),
    ]
    for u, v in examples:
        info = paths[u][v]
        path_s = "→".join(info["path"]) if info.get("path") else "—"
        lines.append(
            f"| {u} | {v} | `{path_s}` | {info.get('P_path', 0):.3f} | "
            f"{info.get('n_hops')} |"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- Mutual reachability (weak): **{n_reach}/{n_pairs}**",
        f"- Universal via paths (strong): **{gate['universal_via_paths']}**",
        f"- $000\\to111$ path P: **{gate['P_000_to_111_path']}**",
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
                "top_edges": [
                    (ek, edge_stats[ek]["P_for_paths"], edge_stats[ek]["channel"])
                    for ek, _ in ranked[:5]
                ],
                "path_000_111": p_000_111,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
