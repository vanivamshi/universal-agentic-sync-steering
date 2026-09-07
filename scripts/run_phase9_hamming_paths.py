#!/usr/bin/env python3
"""Phase 9 — Hamming-path reachability (frozen 8K actuators).

Hypothesis: hard corners (e.g. 111) may be unreachable in one shot but
reachable via intermediate states that change one bit at a time.

Protocol:
  free → 8K home to 000 → (direct one-shot to 111 | Hamming path to 111)

Paths:
  direct:  000 → 111
  OHC:     000 → 001 → 011 → 111
  HOC:     000 → 010 → 011 → 111
  COH:     000 → 100 → 101 → 111

Frozen 8K sites/gains only (no new v). Final success = S_final == 111 only.

  .venv/bin/python scripts/run_phase9_hamming_paths.py --reps 4
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

from scripts.sync_eq import sync_error_norm  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase9_hamming_paths.json"
MD = ROOT / "data" / "results" / "sync_phase9_hamming_paths.md"

SEED = 20260921
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

SOURCE = (0, 0, 0)
TARGET = (1, 1, 1)

# Three shortest Hamming paths 000 → 111 (and direct)
PATHS: dict[str, list[tuple[int, int, int]]] = {
    "direct": [(0, 0, 0), (1, 1, 1)],
    "OHC": [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)],  # O then H then C
    "HOC": [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],  # H then O then C
    "COH": [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)],  # C then O then H
}

REF_8K_VAL = {
    "P_hit_111": 0.00,
    "P_hit_agg": 0.20,
    "note": "8K validation: 111 unreachable in one-shot (0/8)",
}


def _load_8k():
    path = ROOT / "scripts" / "run_phase8k_c_gain_compose.py"
    spec = importlib.util.spec_from_file_location("phase8k", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s: tuple[int, ...] | list[int]) -> str:
    return "".join(str(int(x)) for x in s)


def _bit_diff(a: tuple[int, int, int], b: tuple[int, int, int]) -> str | None:
    diffs = [CHANNELS[i] for i in range(3) if int(a[i]) != int(b[i])]
    if len(diffs) != 1:
        return None
    return diffs[0]


def _hamming(a, b) -> int:
    return sum(int(a[i]) != int(b[i]) for i in range(3))


def _run_one_bit_transition(
    p8k,
    sc,
    loaded,
    *,
    dirs: dict,
    from_s: tuple[int, int, int],
    to_s: tuple[int, int, int],
    seed: int,
    plan_prefill: str,
) -> dict[str, Any]:
    """Apply only the single differing channel toward to_s (8K site for that ch)."""
    ch = _bit_diff(from_s, to_s)
    if ch is None:
        raise ValueError(f"not a Hamming-1 edge: {from_s} → {to_s}")
    active = {ch: (p8k._s_star(to_s[CH_IDX[ch]]), dirs[ch])}
    h_pf = plan_prefill if (ch == "H") else None
    out = p8k._run_episode(
        sc, loaded, active=active, seed=seed, h_plan_prefill=h_pf
    )
    S = list(out["S"])
    return {
        "channel": ch,
        "from": list(from_s),
        "to_target": list(to_s),
        "S": S,
        "hit_waypoint": int(S == list(to_s)),
        "plan_prefill": out.get("plan_prefill") or plan_prefill,
        "E_to_final": float(sync_error_norm(p8k._e(TARGET, S))),
        "E_to_waypoint": float(sync_error_norm(p8k._e(to_s, S))),
    }


def run_sequential_continue(
    p8k,
    sc,
    loaded,
    *,
    mstar: tuple[int, int, int],
    dirs: dict,
    seed: int,
    S: list[int],
    plan_prefill: str,
) -> dict[str, Any]:
    """8K H→C→O stages from an existing state (no free-run). Same seed for episodes."""
    ORDER = p8k.ORDER
    CH_IDX = p8k.CH_IDX
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(p8k._e(mstar, S)))]
    stage_E: dict[str, float] = {}
    n_interv = 0
    n_episodes = 0
    active: dict[str, tuple[int, np.ndarray]] = {}
    plan_prefill = plan_prefill or ""

    for k in ORDER:
        e = p8k._e(mstar, S)
        E_pre = float(sync_error_norm(e))
        added = False
        if e[CH_IDX[k]] != 0 and k not in active:
            active[k] = (p8k._s_star(mstar[CH_IDX[k]]), dirs[k])
            n_interv += 1
            added = True
        if not added:
            traj_S.append(list(S))
            traj_E.append(traj_E[-1])
            stage_E[k] = 0.0
            continue
        h_pf = None
        if "H" in active and "C" not in active:
            h_pf = plan_prefill
        out = p8k._run_episode(
            sc, loaded, active=active, seed=seed, h_plan_prefill=h_pf
        )
        n_episodes += 1
        S = out["S"]
        if out.get("plan_prefill"):
            plan_prefill = out["plan_prefill"]
        E_post = float(sync_error_norm(p8k._e(mstar, S)))
        traj_S.append(list(S))
        traj_E.append(E_post)
        stage_E[k] = float(E_pre - E_post)

    return {
        "S0": traj_S[0],
        "S_final": list(S),
        "E_traj": traj_E,
        "plan_prefill": plan_prefill,
        "delta_E_total": float(traj_E[0] - traj_E[-1]),
        "n_interventions": n_interv,
        "n_episodes": n_episodes,
        "hit": int(list(S) == list(mstar)),
    }


def _home_to_source(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    source: tuple[int, int, int],
    seed: int,
) -> dict[str, Any]:
    """Free-run then 8K sequential toward source (homing)."""
    torch.manual_seed(seed)
    free = p8k._run_episode(sc, loaded, active={}, seed=seed)
    S = list(free["S"])
    plan = free.get("plan_prefill") or ""
    home = run_sequential_continue(
        p8k,
        sc,
        loaded,
        mstar=source,
        dirs=dirs,
        seed=seed,
        S=S,
        plan_prefill=plan,
    )
    return {
        "S_free": S,
        "S_home": home["S_final"],
        "homed": int(home["S_final"] == list(source)),
        "plan_prefill": home["plan_prefill"],
        "E_free": float(sync_error_norm(p8k._e(TARGET, S))),
        "E_home": float(sync_error_norm(p8k._e(TARGET, home["S_final"]))),
        "n_episodes": 1 + home["n_episodes"],
        "home_delta_E_to_source": float(
            sync_error_norm(p8k._e(source, S)) - sync_error_norm(p8k._e(source, home["S_final"]))
        ),
    }


def run_direct_trial(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    seed0: int,
    **_kwargs,
) -> dict[str, Any]:
    """Home to source, then one-shot 8K continue toward target."""
    seed = seed0
    home = _home_to_source(p8k, sc, loaded, dirs=dirs, source=source, seed=seed)
    cont = run_sequential_continue(
        p8k,
        sc,
        loaded,
        mstar=target,
        dirs=dirs,
        seed=seed + 50,
        S=list(home["S_home"]),
        plan_prefill=home["plan_prefill"],
    )
    E_traj = [home["E_free"], home["E_home"]] + cont["E_traj"][1:]
    return {
        "arm": "direct",
        "path": "direct",
        "waypoints": [list(source), list(target)],
        "S_free": home["S_free"],
        "S_home": home["S_home"],
        "homed": home["homed"],
        "source_matched": bool(home["homed"]),
        "n_free_attempts": 1,
        "S0": home["S_home"],
        "S_final": cont["S_final"],
        "E_traj": E_traj,
        "hit_final": int(cont["S_final"] == list(target)),
        "hit_final_if_homed": int(home["homed"] and cont["S_final"] == list(target)),
        "hit_all_waypoints": int(cont["S_final"] == list(target)),
        "edges": [],
        "delta_E_total": float(E_traj[0] - E_traj[-1]),
        "n_episodes": home["n_episodes"] + cont["n_episodes"],
    }


def run_path_trial(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    path_name: str,
    waypoints: list[tuple[int, int, int]],
    seed0: int,
    **_kwargs,
) -> dict[str, Any]:
    """Home to source, then successive one-bit Hamming edges to target."""
    source = waypoints[0]
    target = waypoints[-1]
    seed = seed0
    home = _home_to_source(p8k, sc, loaded, dirs=dirs, source=source, seed=seed)
    S = list(home["S_home"])
    plan_prefill = home["plan_prefill"]
    traj_S = [list(S)]
    traj_E = [float(sync_error_norm(p8k._e(target, S)))]
    edges = []
    all_wp = True
    n_episodes = home["n_episodes"]

    for i in range(len(waypoints) - 1):
        wp_from = waypoints[i]
        wp_to = waypoints[i + 1]
        at_from = int(S == list(wp_from))
        edge = _run_one_bit_transition(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            from_s=wp_from,
            to_s=wp_to,
            seed=seed + 100 * (i + 1),
            plan_prefill=plan_prefill,
        )
        n_episodes += 1
        S = edge["S"]
        if edge.get("plan_prefill"):
            plan_prefill = edge["plan_prefill"]
        traj_S.append(list(S))
        traj_E.append(edge["E_to_final"])
        hit_wp = int(S == list(wp_to))
        if not hit_wp:
            all_wp = False
        edges.append({**edge, "at_from_before": at_from, "hit_waypoint": hit_wp})

    hit_final = int(S == list(target))
    return {
        "arm": "path",
        "path": path_name,
        "waypoints": [list(w) for w in waypoints],
        "S_free": home["S_free"],
        "S_home": home["S_home"],
        "homed": home["homed"],
        "source_matched": bool(home["homed"]),
        "n_free_attempts": 1,
        "S0": traj_S[0],
        "S_final": S,
        "traj_S": traj_S,
        "E_traj": [home["E_free"], home["E_home"]] + traj_E[1:],
        "hit_final": hit_final,
        "hit_final_if_homed": int(home["homed"] and hit_final),
        "hit_all_waypoints": int(all_wp and hit_final),
        "edges": edges,
        "delta_E_total": float(traj_E[0] - traj_E[-1]),
        "n_episodes": n_episodes,
    }


def _agg_arm(trials: list[dict]) -> dict[str, Any]:
    if not trials:
        return {"n": 0}
    homed = [t for t in trials if t.get("homed")]
    def m(key, rows=trials):
        return float(np.mean([t[key] for t in rows])) if rows else float("nan")

    edge_stats: dict[str, list[float]] = {}
    edge_stats_homed: dict[str, list[float]] = {}
    for t in trials:
        for e in t.get("edges") or []:
            label = f"{_key(e['from'])}→{_key(e['to_target'])}"
            edge_stats.setdefault(label, []).append(float(e["hit_waypoint"]))
            if t.get("homed"):
                edge_stats_homed.setdefault(label, []).append(float(e["hit_waypoint"]))
    return {
        "n": len(trials),
        "P_homed": m("homed"),
        "n_homed": len(homed),
        "P_hit_final": m("hit_final"),
        "P_hit_final_if_homed": m("hit_final_if_homed", homed) if homed else float("nan"),
        "P_hit_all_waypoints": m("hit_all_waypoints"),
        "mean_delta_E": m("delta_E_total"),
        "mean_E_free": float(np.mean([t["E_traj"][0] for t in trials])),
        "mean_Ef": float(np.mean([t["E_traj"][-1] for t in trials])),
        "P_edge": {k: float(np.mean(v)) for k, v in edge_stats.items()},
        "P_edge_if_homed": {k: float(np.mean(v)) for k, v in edge_stats_homed.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--paths",
        default="direct,OHC,HOC,COH",
        help="comma-separated path keys",
    )
    args = ap.parse_args()
    path_names = [p.strip() for p in args.paths.split(",") if p.strip()]

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_8k()
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()

    print(
        f"=== Phase 9 Hamming paths {_key(SOURCE)}→{_key(TARGET)} "
        f"(home→path) reps={args.reps} ===",
        flush=True,
    )
    print(
        "  Protocol: free → 8K home to 000 → direct one-shot OR one-bit Hamming path to 111",
        flush=True,
    )

    by_arm: dict[str, list[dict]] = {name: [] for name in path_names}

    for r in range(args.reps):
        seed0 = args.seed + 1000 * r
        for name in path_names:
            wps = PATHS[name]
            print(f"  r={r} path={name} wps={[_key(w) for w in wps]}", flush=True)
            if name == "direct":
                tr = run_direct_trial(
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    source=SOURCE,
                    target=TARGET,
                    seed0=seed0,
                )
            else:
                tr = run_path_trial(
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    path_name=name,
                    waypoints=wps,
                    seed0=seed0,
                )
            by_arm[name].append(tr)
            print(
                f"    free={_key(tr['S_free'])} home={_key(tr['S_home'])} "
                f"homed={tr['homed']} → final={_key(tr['S_final'])} "
                f"hit={tr['hit_final']} hit|homed={tr['hit_final_if_homed']} "
                f"E:{[round(x, 2) for x in tr['E_traj']]}",
                flush=True,
            )

    summary = {name: _agg_arm(rows) for name, rows in by_arm.items()}
    p_direct = summary.get("direct", {}).get("P_hit_final", float("nan"))
    p_direct_h = summary.get("direct", {}).get("P_hit_final_if_homed", float("nan"))
    best_path = None
    best_p = -1.0
    best_p_h = -1.0
    for name, s in summary.items():
        if name == "direct":
            continue
        p = s.get("P_hit_final", float("nan"))
        ph = s.get("P_hit_final_if_homed", float("nan"))
        if p == p and p > best_p:
            best_p = p
            best_path = name
        if ph == ph and ph > best_p_h:
            best_p_h = ph

    gate = {
        "hypothesis": (
            "After homing to 000, one-bit Hamming paths reach 111 more often than "
            "direct 000→111 one-shot"
        ),
        "protocol": "free → home(000) → path|direct to 111",
        "source": list(SOURCE),
        "target": list(TARGET),
        "P_direct": p_direct,
        "P_direct_if_homed": p_direct_h,
        "best_path": best_path,
        "P_best_path": best_p if best_path else float("nan"),
        "P_best_path_if_homed": best_p_h,
        "path_beats_direct": bool(
            best_path is not None
            and best_p > (p_direct if p_direct == p_direct else -1) + 0.05
        ),
        "path_beats_direct_if_homed": bool(
            best_p_h > (p_direct_h if p_direct_h == p_direct_h else -1) + 0.05
        ),
        "any_path_reaches": bool(best_p > 0),
        "ref_8K_val_111": REF_8K_VAL,
        "claim_if_pass": (
            "direct controllability ≠ global reachability; "
            "local controllability + intermediate states ⇒ global reachability"
        ),
        "read": (
            "Final success = S_final==111 only. Homing/intermediates are path, not credit. "
            "No new v."
        ),
    }

    payload = {
        "protocol": "Phase 9 Hamming-path reachability (home then path)",
        "reps": args.reps,
        "seed": args.seed,
        "controller": "frozen 8K (C stem α=5, H decision-token α=1.5, O early-FINAL α=1.5)",
        "paths": {k: [_key(w) for w in PATHS[k]] for k in path_names},
        "summary": summary,
        "gate": gate,
        "trials": by_arm,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 9 — Hamming-path reachability",
        "",
        r"> Frozen 8K. Protocol: free $\to$ home to $000$ $\to$ direct or Hamming path to $111$. "
        r"Final success $=S_{\mathrm{final}}{=}111$ only.",
        "",
        f"reps={args.reps}, seed={args.seed}.",
        "",
        "## Aggregate",
        "",
        "| arm | path | n | P(homed) | P(hit 111) | P(hit\\|homed) | P(all wp) | ΔE |",
        "|-----|------|--:|---------:|-----------:|--------------:|----------:|---:|",
    ]
    for name in path_names:
        s = summary[name]
        path_str = "→".join(_key(w) for w in PATHS[name])
        lines.append(
            f"| {name} | `{path_str}` | {s.get('n', 0)} | "
            f"{s.get('P_homed', float('nan')):.2f} | "
            f"**{s.get('P_hit_final', float('nan')):.2f}** | "
            f"{s.get('P_hit_final_if_homed', float('nan')):.2f} | "
            f"{s.get('P_hit_all_waypoints', float('nan')):.2f} | "
            f"{s.get('mean_delta_E', float('nan')):+.2f} |"
        )
    lines += [
        "",
        f"| 8K val one-shot (111) | — | 8 | — | **{REF_8K_VAL['P_hit_111']:.2f}** | — | — | — |",
        "",
        "## Per-edge $P(\\mathrm{waypoint})$",
        "",
    ]
    for name in path_names:
        if name == "direct":
            continue
        s = summary[name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| edge | P(hit) | P(hit\\|homed) |")
        lines.append("|------|-------:|--------------:|")
        edges = s.get("P_edge") or {}
        edges_h = s.get("P_edge_if_homed") or {}
        for edge in edges:
            lines.append(
                f"| `{edge}` | {edges[edge]:.2f} | {edges_h.get(edge, float('nan')):.2f} |"
            )
        lines.append("")

    lines += [
        "## Gate",
        "",
        f"- Path beats direct: **{gate['path_beats_direct']}**",
        f"- Path beats direct | homed: **{gate['path_beats_direct_if_homed']}**",
        f"- Best path: **{gate['best_path']}** (P={gate['P_best_path']})",
        f"- Direct P(hit): **{gate['P_direct']}**",
        f"- Any path reaches 111: **{gate['any_path_reaches']}**",
        "",
        gate["read"],
        "",
        f"If pass: *{gate['claim_if_pass']}.*",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "summary": summary, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
