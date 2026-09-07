#!/usr/bin/env python3
"""Phase 9E — Offline Γ-planner simulation on existing 9B/9C trajectories.

No new model runs. No new v. Frozen actuators; only changes *which* channel
to apply given (s, m*).

  Γ(a | s, m*) = P(E↓) - P(E↑),  E = ||m* - S||_1
  a* = argmax_a Γ(a | s, m*)

Compare a* vs fixed H→C→O bit-order using measured empirical transitions.

  .venv/bin/python scripts/run_phase9e_gamma_planner.py
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
P9B = ROOT / "data" / "results" / "sync_phase9_transition_graph.json"
P9C = ROOT / "data" / "results" / "sync_phase9c_densify_edges.json"
OUT = ROOT / "data" / "results" / "sync_phase9e_gamma_planner.json"
MD = ROOT / "data" / "results" / "sync_phase9e_gamma_planner.md"

CHANNELS = ("H", "C", "O")  # fixed 8K order
CH_IDX = {"C": 0, "H": 1, "O": 2}
STATES = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]

TERM_LOGS = [
    Path.home()
    / ".cursor/projects/Users-vamshisunku-mohan-multidim-steering-agentic-alignment/terminals/38457.txt",
    Path.home()
    / ".cursor/projects/Users-vamshisunku-mohan-multidim-steering-agentic-alignment/terminals/38458.txt",
]

SEED = 20260924
N_ROLLOUTS = 64
MAX_STEPS = 6
MIN_N_ACTION = 1  # allow sparse cells; report n alongside Γ


def _key(s: list[int] | tuple[int, ...] | str) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _parse(s: str) -> tuple[int, int, int]:
    return (int(s[0]), int(s[1]), int(s[2]))


def _E(mstar: tuple[int, int, int], S: tuple[int, int, int] | list[int]) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _load_transitions() -> list[dict[str, Any]]:
    """Empirical one-step outcomes: S_before --a--> S_after."""
    rows: list[dict[str, Any]] = []

    if P9B.exists():
        j = json.loads(P9B.read_text())
        for ek, trs in (j.get("trials") or {}).items():
            for r in trs:
                rows.append(
                    {
                        "src": "9B",
                        "edge": ek,
                        "S_before": tuple(r["S_home"]),
                        "S_after": tuple(r["S_after"]),
                        "a": r["channel"],
                        "to_intended": tuple(r["to"]),
                    }
                )

    if P9C.exists():
        j = json.loads(P9C.read_text())
        for ek, trs in (j.get("densified_trials") or {}).items():
            if "→" not in ek:
                continue
            src_s, tgt_s = ek.split("→")
            src, tgt = _parse(src_s), _parse(tgt_s)
            ch = {"001→000": "O", "010→000": "H", "100→000": "C", "010→110": "C",
                  "100→110": "H", "111→110": "O", "101→100": "O"}.get(ek)
            for r in trs or []:
                if not r or not r.get("constructed"):
                    continue
                Sb = tuple(r["S_after_construct"])
                Sa = tuple(r["S_after_edge"])
                if Sb != src:
                    continue
                rows.append(
                    {
                        "src": "9C",
                        "edge": ek,
                        "S_before": Sb,
                        "S_after": Sa,
                        "a": r.get("channel") or ch,
                        "to_intended": tgt,
                    }
                )

    # Recover early 9C from logs (rows not persisted)
    json_edges = {r["edge"] for r in rows if r.get("edge") and r["src"] == "9C"}
    ch_of = {
        "001→000": "O", "010→000": "H", "100→000": "C", "010→110": "C",
        "100→110": "H", "111→110": "O", "101→100": "O",
    }
    for path in TERM_LOGS:
        if not path.exists():
            continue
        cur = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"densify\s+(\d{3}→\d{3})", line)
            if m:
                cur = m.group(1)
                continue
            m = re.search(r"constructed=1.*S→(\d{3})", line)
            if not m or not cur or cur in json_edges:
                continue
            src_s, tgt_s = cur.split("→")
            rows.append(
                {
                    "src": "9C_log",
                    "edge": cur,
                    "S_before": _parse(src_s),
                    "S_after": _parse(m.group(1)),
                    "a": ch_of.get(cur),
                    "to_intended": _parse(tgt_s),
                }
            )
    return [r for r in rows if r.get("a") in CHANNELS]


def _build_kernel(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    """(s_key, a) -> list of observed S_after."""
    ker: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    for r in rows:
        ker[(_key(r["S_before"]), r["a"])].append(tuple(r["S_after"]))  # type: ignore
    return ker


def _gamma_from_outcomes(
    s: tuple[int, int, int],
    outcomes: list[tuple[int, int, int]],
    mstar: tuple[int, int, int],
) -> dict[str, Any]:
    n = len(outcomes)
    if n == 0:
        return {"n": 0, "Gamma": None, "P_down": None, "P_up": None, "P_neutral": None,
                "P_shift": None, "mean_dE": None, "P_hit": None}
    Eb = _E(mstar, s)
    n_down = n_up = n_neu = n_shift = n_hit = 0
    dE_sum = 0
    for sa in outcomes:
        Ea = _E(mstar, sa)
        dE = Ea - Eb
        dE_sum += dE
        if sa != s:
            n_shift += 1
        if Ea < Eb:
            n_down += 1
        elif Ea > Eb:
            n_up += 1
        else:
            n_neu += 1
        if sa == mstar:
            n_hit += 1
    pd, pu = n_down / n, n_up / n
    return {
        "n": n,
        "Gamma": pd - pu,
        "P_down": pd,
        "P_up": pu,
        "P_neutral": n_neu / n,
        "P_shift": n_shift / n,
        "mean_dE": dE_sum / n,
        "P_hit": n_hit / n,
        "after_hist": dict(Counter(_key(x) for x in outcomes)),
    }


def _gamma_table(
    ker: dict[tuple[str, str], list[tuple[int, int, int]]],
    mstar: tuple[int, int, int],
) -> dict[str, dict[str, Any]]:
    """s_key -> {a: gamma_stats, a_star, gammas}."""
    out: dict[str, dict[str, Any]] = {}
    for s in STATES:
        sk = _key(s)
        by_a = {}
        for a in CHANNELS:
            outs = ker.get((sk, a), [])
            by_a[a] = _gamma_from_outcomes(s, outs, mstar)
        # argmax among actions with n >= MIN_N_ACTION
        cands = [(a, by_a[a]["Gamma"], by_a[a]["n"]) for a in CHANNELS if by_a[a]["n"] >= MIN_N_ACTION and by_a[a]["Gamma"] is not None]
        if not cands:
            a_star = None
            g_star = None
        else:
            # tie-break: higher n, then fixed order H preference among equal Γ
            cands.sort(key=lambda t: (t[1], t[2], -CHANNELS.index(t[0])), reverse=True)
            a_star, g_star, _ = cands[0]
        out[sk] = {
            "by_a": by_a,
            "a_star": a_star,
            "Gamma_star": g_star,
            "n_actions_available": len(cands),
        }
    return out


def _fixed_action(s: tuple[int, int, int], mstar: tuple[int, int, int]) -> str | None:
    """Next channel under frozen H→C→O for bits that still differ."""
    for a in CHANNELS:
        i = CH_IDX[a]
        if int(s[i]) != int(mstar[i]):
            return a
    return None


def _sample_next(
    ker: dict[tuple[str, str], list[tuple[int, int, int]]],
    s: tuple[int, int, int],
    a: str,
    rng: random.Random,
) -> tuple[int, int, int] | None:
    outs = ker.get((_key(s), a), [])
    if not outs:
        return None
    return outs[rng.randrange(len(outs))]


def _rollout(
    ker: dict[tuple[str, str], list[tuple[int, int, int]]],
    *,
    s0: tuple[int, int, int],
    mstar: tuple[int, int, int],
    policy: str,
    gamma_tab: dict[str, dict[str, Any]],
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    s = s0
    path = [_key(s)]
    actions = []
    dE_traj = [_E(mstar, s)]
    stuck = False
    for _ in range(max_steps):
        if s == mstar:
            break
        if policy == "gamma":
            a = gamma_tab[_key(s)]["a_star"]
        elif policy == "fixed":
            a = _fixed_action(s, mstar)
        else:
            raise ValueError(policy)
        if a is None:
            stuck = True
            break
        nxt = _sample_next(ker, s, a, rng)
        if nxt is None:
            stuck = True
            actions.append({"a": a, "ok": False})
            break
        actions.append({"a": a, "ok": True, "S_after": _key(nxt)})
        # track ΔS and ΔE
        dS = int(nxt != s)
        dE = _E(mstar, nxt) - _E(mstar, s)
        actions[-1]["dS"] = dS
        actions[-1]["dE"] = dE
        s = nxt
        path.append(_key(s))
        dE_traj.append(_E(mstar, s))
    return {
        "hit": int(s == mstar),
        "path": path,
        "actions": actions,
        "E_final": _E(mstar, s),
        "E0": dE_traj[0],
        "stuck": stuck,
        "n_steps": len(actions),
        "mean_dE_step": (
            sum(a.get("dE", 0) for a in actions if a.get("ok")) / max(1, sum(1 for a in actions if a.get("ok")))
        ),
    }


def _simulate_pair(
    ker,
    gamma_tab,
    s0: tuple[int, int, int],
    mstar: tuple[int, int, int],
    *,
    n_rollouts: int,
    seed: int,
) -> dict[str, Any]:
    hits_g = hits_f = 0
    stuck_g = stuck_f = 0
    e_g = []
    e_f = []
    paths_g = []
    paths_f = []
    for i in range(n_rollouts):
        rng = random.Random(seed + 17 * i + 3 * int(_key(s0), 2) + int(_key(mstar), 2))
        rg = _rollout(ker, s0=s0, mstar=mstar, policy="gamma", gamma_tab=gamma_tab, rng=rng)
        rf = _rollout(ker, s0=s0, mstar=mstar, policy="fixed", gamma_tab=gamma_tab, rng=random.Random(seed + 1000 + i))
        hits_g += rg["hit"]
        hits_f += rf["hit"]
        stuck_g += int(rg["stuck"])
        stuck_f += int(rf["stuck"])
        e_g.append(rg["E_final"])
        e_f.append(rf["E_final"])
        if i < 3:
            paths_g.append(rg)
            paths_f.append(rf)
    return {
        "s0": _key(s0),
        "mstar": _key(mstar),
        "n_rollouts": n_rollouts,
        "gamma": {
            "P_hit": hits_g / n_rollouts,
            "P_stuck": stuck_g / n_rollouts,
            "mean_E_final": sum(e_g) / n_rollouts,
            "examples": paths_g,
        },
        "fixed": {
            "P_hit": hits_f / n_rollouts,
            "P_stuck": stuck_f / n_rollouts,
            "mean_E_final": sum(e_f) / n_rollouts,
            "examples": paths_f,
        },
        "delta_P_hit": (hits_g - hits_f) / n_rollouts,
        "delta_mean_E": (sum(e_f) - sum(e_g)) / n_rollouts,  # positive => gamma closer
    }


def _retrospective(
    rows: list[dict[str, Any]],
    ker: dict[tuple[str, str], list[tuple[int, int, int]]],
    mstars: list[tuple[int, int, int]],
) -> dict[str, Any]:
    """For each historical step, compare a_taken vs a_Γ on expected ΔE / Γ."""
    n = 0
    n_agree = 0
    n_gamma_better_dE = 0
    n_gamma_worse_dE = 0
    n_gamma_better_G = 0
    details = []
    for r in rows:
        s = r["S_before"]
        a_taken = r["a"]
        sa = r["S_after"]
        for mstar in mstars:
            # skip if already at target
            if s == mstar:
                continue
            tab = {}
            for a in CHANNELS:
                outs = ker.get((_key(s), a), [])
                tab[a] = _gamma_from_outcomes(s, outs, mstar)
            cands = [(a, tab[a]["Gamma"], tab[a]["n"]) for a in CHANNELS if tab[a]["n"] >= MIN_N_ACTION and tab[a]["Gamma"] is not None]
            if len(cands) < 2:
                continue  # need choice
            cands.sort(key=lambda t: (t[1], t[2], -CHANNELS.index(t[0])), reverse=True)
            a_g = cands[0][0]
            n += 1
            if a_g == a_taken:
                n_agree += 1
            # expected ΔE under empirical kernel
            def mean_dE(a: str) -> float | None:
                outs = ker.get((_key(s), a), [])
                if not outs:
                    return None
                Eb = _E(mstar, s)
                return sum(_E(mstar, x) - Eb for x in outs) / len(outs)

            md_g = mean_dE(a_g)
            md_t = mean_dE(a_taken)
            # also realized ΔE of taken
            dE_taken = _E(mstar, sa) - _E(mstar, s)
            if md_g is not None and md_t is not None:
                if md_g < md_t - 1e-12:
                    n_gamma_better_dE += 1
                elif md_g > md_t + 1e-12:
                    n_gamma_worse_dE += 1
            if tab[a_g]["Gamma"] is not None and tab[a_taken]["Gamma"] is not None:
                if tab[a_g]["Gamma"] > tab[a_taken]["Gamma"] + 1e-12:
                    n_gamma_better_G += 1
            if len(details) < 40 and a_g != a_taken:
                details.append(
                    {
                        "s": _key(s),
                        "mstar": _key(mstar),
                        "a_taken": a_taken,
                        "a_gamma": a_g,
                        "Gamma_taken": tab[a_taken]["Gamma"],
                        "Gamma_gamma": tab[a_g]["Gamma"],
                        "mean_dE_taken": md_t,
                        "mean_dE_gamma": md_g,
                        "realized_dE_taken": dE_taken,
                    }
                )
    return {
        "n_comparisons": n,
        "P_agree": n_agree / n if n else None,
        "P_gamma_better_mean_dE": n_gamma_better_dE / n if n else None,
        "P_gamma_worse_mean_dE": n_gamma_worse_dE / n if n else None,
        "P_gamma_strictly_higher_G": n_gamma_better_G / n if n else None,
        "disagreement_examples": details,
    }


def _fmt(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    rows = _load_transitions()
    ker = _build_kernel(rows)
    coverage = {
        f"{sk}|{a}": len(outs)
        for (sk, a), outs in sorted(ker.items())
    }

    # Build Γ tables for each m*
    tables = {_key(m): _gamma_table(ker, m) for m in STATES}

    # Rollouts: all s0 with ≥1 action observed, all m*
    s0_usable = sorted(
        {
            sk
            for (sk, a), outs in ker.items()
            if outs
        }
    )
    pair_results = []
    for s0k in s0_usable:
        s0 = _parse(s0k)
        for mstar in STATES:
            if s0 == mstar:
                continue
            tab = tables[_key(mstar)]
            # need at least one action at s0 for gamma policy
            if tab[s0k]["a_star"] is None and _fixed_action(s0, mstar) is None:
                continue
            pair_results.append(
                _simulate_pair(
                    ker, tab, s0, mstar, n_rollouts=N_ROLLOUTS, seed=SEED
                )
            )

    # Aggregate: for each m*, max P_hit over s0 under each policy
    by_mstar: dict[str, Any] = {}
    for m in STATES:
        mk = _key(m)
        sub = [p for p in pair_results if p["mstar"] == mk]
        if not sub:
            by_mstar[mk] = {"reachable_gamma": False, "reachable_fixed": False}
            continue
        best_g = max(sub, key=lambda p: p["gamma"]["P_hit"])
        best_f = max(sub, key=lambda p: p["fixed"]["P_hit"])
        by_mstar[mk] = {
            "best_gamma": {
                "s0": best_g["s0"],
                "P_hit": best_g["gamma"]["P_hit"],
                "mean_E": best_g["gamma"]["mean_E_final"],
            },
            "best_fixed": {
                "s0": best_f["s0"],
                "P_hit": best_f["fixed"]["P_hit"],
                "mean_E": best_f["fixed"]["mean_E_final"],
            },
            "reachable_gamma": best_g["gamma"]["P_hit"] > 0,
            "reachable_fixed": best_f["fixed"]["P_hit"] > 0,
            "mean_delta_P_hit": sum(p["delta_P_hit"] for p in sub) / len(sub),
            "n_pairs": len(sub),
        }

    # Soft starts → each m* (exclude hard starts we can't initialize)
    soft = {"010", "011", "100", "101"}
    soft_to_m = []
    for p in pair_results:
        if p["s0"] in soft:
            soft_to_m.append(p)

    mean_delta = (
        sum(p["delta_P_hit"] for p in pair_results) / len(pair_results)
        if pair_results
        else 0.0
    )
    n_gamma_wins = sum(1 for p in pair_results if p["delta_P_hit"] > 1e-9)
    n_fixed_wins = sum(1 for p in pair_results if p["delta_P_hit"] < -1e-9)
    n_tie = len(pair_results) - n_gamma_wins - n_fixed_wins

    retro = _retrospective(rows, ker, STATES)

    # Example Γ(a|s,m*) tables for the striking case
    examples = {
        "010_vs_000": tables["000"]["010"],
        "010_vs_111": tables["111"]["010"],
        "100_vs_000": tables["000"]["100"],
        "100_vs_111": tables["111"]["100"],
        "101_vs_111": tables["111"]["101"],
    }

    n_m_reach_g = sum(1 for m, st in by_mstar.items() if st.get("reachable_gamma"))
    n_m_reach_f = sum(1 for m, st in by_mstar.items() if st.get("reachable_fixed"))

    gate = {
        "hypothesis": (
            "Target-conditioned a*=argmax Γ beats fixed H→C→O on offline "
            "empirical rollouts without new actuators"
        ),
        "n_transitions": len(rows),
        "n_kernel_cells": len(ker),
        "n_pair_sims": len(pair_results),
        "mean_delta_P_hit_gamma_minus_fixed": mean_delta,
        "n_pairs_gamma_wins": n_gamma_wins,
        "n_pairs_fixed_wins": n_fixed_wins,
        "n_pairs_tie": n_tie,
        "n_mstar_reachable_gamma": n_m_reach_g,
        "n_mstar_reachable_fixed": n_m_reach_f,
        "retro_P_agree": retro["P_agree"],
        "retro_P_gamma_better_mean_dE": retro["P_gamma_better_mean_dE"],
        "coverage_gap_states": sorted(
            {_key(s) for s in STATES} - set(s0_usable)
        ),
        "read": (
            "Offline only. Positive mean ΔP_hit favors deploying Γ-choice in-loop next; "
            "coverage gaps (e.g. 000/001 as s0) limit claims of universal 8-way."
        ),
    }

    payload = {
        "protocol": "Phase 9E offline Γ-planner",
        "seed": SEED,
        "n_rollouts": N_ROLLOUTS,
        "max_steps": MAX_STEPS,
        "definitions": {
            "Gamma": "P(E↓)-P(E↑) from empirical kernel",
            "a_star": "argmax_a Gamma(a|s,m*)",
            "fixed": "H→C→O first differing bit",
        },
        "coverage": coverage,
        "examples_gamma": examples,
        "by_mstar": by_mstar,
        "pair_results_summary": [
            {
                "s0": p["s0"],
                "mstar": p["mstar"],
                "P_hit_gamma": p["gamma"]["P_hit"],
                "P_hit_fixed": p["fixed"]["P_hit"],
                "delta_P_hit": p["delta_P_hit"],
                "delta_mean_E": p["delta_mean_E"],
                "mean_E_gamma": p["gamma"]["mean_E_final"],
                "mean_E_fixed": p["fixed"]["mean_E_final"],
            }
            for p in sorted(pair_results, key=lambda x: -x["delta_P_hit"])
        ],
        "retrospective": retro,
        "gate": gate,
        # keep a few full pair examples
        "pair_examples": [p for p in pair_results if p["s0"] in soft and p["mstar"] in ("111", "000", "110")][:12],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 9E — Offline $\\Gamma$-planner simulation",
        "",
        "> Existing 9B/9C transitions only. No new $v$. "
        r"$a^*=\arg\max_a\Gamma(a\mid s,m^*)$ vs fixed $H\to C\to O$.",
        "",
        f"Kernel: **{len(rows)}** transitions across **{len(ker)}** $(s,a)$ cells. "
        f"Rollouts: {N_ROLLOUTS}/pair, max_steps={MAX_STEPS}.",
        "",
        f"Coverage gap as $s_0$: `{gate['coverage_gap_states']}`",
        "",
        "## Striking $\\Gamma$ sign flip (same $s$, different $m^*$)",
        "",
        "| $s$ | $m^*$ | $\\Gamma_C$ | $\\Gamma_H$ | $\\Gamma_O$ | $a^*$ |",
        "|-----|-------|-----------:|-----------:|-----------:|------|",
    ]
    for label, blk in (
        ("010→ local 000", examples["010_vs_000"]),
        ("010→ global 111", examples["010_vs_111"]),
        ("100→ local 000", examples["100_vs_000"]),
        ("100→ global 111", examples["100_vs_111"]),
    ):
        s_label = label.split("→")[0].strip()
        m_label = label.split()[-1]
        by = blk["by_a"]

        def g(a):
            return _fmt(by[a]["Gamma"])

        lines.append(
            f"| `{s_label}` | `{m_label}` | {g('C')} (n={by['C']['n']}) | "
            f"{g('H')} (n={by['H']['n']}) | {g('O')} (n={by['O']['n']}) | "
            f"**{blk['a_star']}** |"
        )

    lines += [
        "",
        "## Offline reachability by $m^*$ (best $s_0$ in kernel)",
        "",
        "| $m^*$ | best $P_{hit}$ $\\Gamma$ | from $s_0$ | best $P_{hit}$ fixed | from $s_0$ | "
        r"mean $\Delta P_{hit}$ |",
        "|-------|-------------------------:|-----------:|---------------------:|-----------:|----------------------:|",
    ]
    for m in STATES:
        mk = _key(m)
        st = by_mstar.get(mk) or {}
        bg = st.get("best_gamma") or {}
        bf = st.get("best_fixed") or {}
        lines.append(
            f"| `{mk}` | {_fmt(bg.get('P_hit'))} | `{bg.get('s0','—')}` | "
            f"{_fmt(bf.get('P_hit'))} | `{bf.get('s0','—')}` | "
            f"{_fmt(st.get('mean_delta_P_hit'))} |"
        )

    lines += [
        "",
        f"Targets with any offline hit: $\\Gamma$ **{n_m_reach_g}/8**, fixed **{n_m_reach_f}/8**.",
        "",
        "## Policy comparison (all simulated pairs)",
        "",
        f"- mean $\\Delta P_{{\\mathrm{{hit}}}}(\\Gamma - \\mathrm{{fixed}})$ = **{_fmt(mean_delta, 3)}**",
        f"- pairs $\\Gamma$ wins / fixed wins / ties: "
        f"**{n_gamma_wins}** / **{n_fixed_wins}** / **{n_tie}** (of {len(pair_results)})",
        "",
        "### Top $\\Gamma$ improvements",
        "",
        "| $s_0$ | $m^*$ | $P_{hit}\\Gamma$ | $P_{hit}$ fixed | $\\Delta$ |",
        "|-------|-------|----------------:|----------------:|---------:|",
    ]
    top = sorted(pair_results, key=lambda p: -p["delta_P_hit"])[:8]
    for p in top:
        lines.append(
            f"| `{p['s0']}` | `{p['mstar']}` | {_fmt(p['gamma']['P_hit'])} | "
            f"{_fmt(p['fixed']['P_hit'])} | **{_fmt(p['delta_P_hit'])}** |"
        )

    lines += [
        "",
        "### Top fixed wins (Γ worse)",
        "",
        "| $s_0$ | $m^*$ | $P_{hit}\\Gamma$ | $P_{hit}$ fixed | $\\Delta$ |",
        "|-------|-------|----------------:|----------------:|---------:|",
    ]
    bot = sorted(pair_results, key=lambda p: p["delta_P_hit"])[:5]
    for p in bot:
        lines.append(
            f"| `{p['s0']}` | `{p['mstar']}` | {_fmt(p['gamma']['P_hit'])} | "
            f"{_fmt(p['fixed']['P_hit'])} | {_fmt(p['delta_P_hit'])} |"
        )

    lines += [
        "",
        "## Retrospective: would $\\Gamma$ have chosen differently?",
        "",
        f"- comparisons: {retro['n_comparisons']}",
        f"- $P(\\mathrm{{agree}})$: {_fmt(retro['P_agree'])}",
        f"- $P(\\Gamma$ better mean $\\Delta E)$: **{_fmt(retro['P_gamma_better_mean_dE'])}**",
        f"- $P(\\Gamma$ worse mean $\\Delta E)$: {_fmt(retro['P_gamma_worse_mean_dE'])}",
        "",
        "## Gate",
        "",
        f"- mean $\\Delta P_{{hit}}$: **{_fmt(mean_delta, 3)}**",
        f"- $m^*$ reachable offline ($\\Gamma$ / fixed): **{n_m_reach_g}/8** / **{n_m_reach_f}/8**",
        f"- coverage gaps as $s_0$: `{gate['coverage_gap_states']}`",
        "",
        gate["read"],
        "",
        r"$$\boxed{a^*=\arg\max_a\Gamma(a\mid s,m^*)\text{ — planner change, not a new actuator}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
