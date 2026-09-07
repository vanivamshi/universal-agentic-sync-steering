#!/usr/bin/env python3
"""Phase 9D — Target-aware shift probabilities from existing 9B/9C trajectories.

No new model runs. No new v. Reanalyze stored trials:

  E_t = ||m* - S_t||_1
  P↓ = P(E_after < E_before)
  P↑ = P(E_after > E_before)
  P0 = P(E_after = E_before)
  Γ  = P↓ - P↑
  P_r = P(d_H(S_after, m*) = r)

Primary: Phase 9C constructed trials (S_before = source).
Also: Phase 9B one-bit trials (S_before = S_home).

  .venv/bin/python scripts/analyze_phase9d_shift.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
P9B = ROOT / "data" / "results" / "sync_phase9_transition_graph.json"
P9C = ROOT / "data" / "results" / "sync_phase9c_densify_edges.json"
OUT = ROOT / "data" / "results" / "sync_phase9d_shift.json"
MD = ROOT / "data" / "results" / "sync_phase9d_shift.md"

TERM_LOGS = [
    Path.home()
    / ".cursor/projects/Users-vamshisunku-mohan-multidim-steering-agentic-alignment/terminals/38457.txt",
    Path.home()
    / ".cursor/projects/Users-vamshisunku-mohan-multidim-steering-agentic-alignment/terminals/38458.txt",
]


def _key(s: list[int] | tuple[int, ...] | str) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _parse(s: str) -> tuple[int, int, int]:
    return (int(s[0]), int(s[1]), int(s[2]))


def _E(mstar: tuple[int, int, int], S: tuple[int, int, int] | list[int]) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _stats(rows: list[dict[str, Any]], mstar: tuple[int, int, int]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "P_shift": None,
            "P_down": None,
            "P_up": None,
            "P_neutral": None,
            "Gamma": None,
            "P_hit_mstar": None,
            "P_r": {str(r): None for r in range(4)},
            "mean_dE": None,
            "transitions": [],
        }
    n_shift = n_down = n_up = n_neu = n_hit = 0
    dE_sum = 0
    pr = [0, 0, 0, 0]
    transitions = []
    for r in rows:
        Sb = tuple(r["S_before"])
        Sa = tuple(r["S_after"])
        Eb, Ea = _E(mstar, Sb), _E(mstar, Sa)
        dE = Ea - Eb
        dE_sum += dE
        if Sa != Sb:
            n_shift += 1
        if Ea < Eb:
            n_down += 1
        elif Ea > Eb:
            n_up += 1
        else:
            n_neu += 1
        if Sa == mstar:
            n_hit += 1
        pr[Ea] += 1
        transitions.append(
            {
                "S_before": list(Sb),
                "S_after": list(Sa),
                "E_before": Eb,
                "E_after": Ea,
                "dE": dE,
            }
        )
    p_down = n_down / n
    p_up = n_up / n
    return {
        "n": n,
        "P_shift": n_shift / n,
        "P_down": p_down,
        "P_up": p_up,
        "P_neutral": n_neu / n,
        "Gamma": p_down - p_up,
        "P_hit_mstar": n_hit / n,
        "P_r": {str(r): pr[r] / n for r in range(4)},
        "mean_dE": dE_sum / n,
        "counts": {"down": n_down, "up": n_up, "neutral": n_neu, "shift": n_shift},
        "transitions": transitions,
    }


def _load_9c_from_json() -> list[dict[str, Any]]:
    if not P9C.exists():
        return []
    j = json.loads(P9C.read_text())
    out = []
    for ek, rows in (j.get("densified_trials") or {}).items():
        if "→" not in ek:
            continue
        src_s, tgt_s = ek.split("→")
        src, tgt = _parse(src_s), _parse(tgt_s)
        for row in rows or []:
            if not row or not row.get("constructed"):
                continue
            Sb = row.get("S_after_construct")
            Sa = row.get("S_after_edge")
            if Sb is None or Sa is None:
                continue
            # only count true conditional-at-source
            if tuple(Sb) != src:
                continue
            out.append(
                {
                    "source": "9C",
                    "edge": ek,
                    "channel": row.get("channel"),
                    "S_before": list(Sb),
                    "S_after": list(Sa),
                    "from": list(src),
                    "to": list(tgt),
                    "hit_edge": int(tuple(Sa) == tgt),
                }
            )
    return out


def _load_9c_from_logs() -> list[dict[str, Any]]:
    """Recover early 9C trials whose rows were not persisted."""
    out = []
    ch_of = {
        "001→000": "O",
        "010→000": "H",
        "100→000": "C",
        "010→110": "C",
        "100→110": "H",
        "111→110": "O",
        "101→100": "O",
        "000→001": "O",
        "000→010": "H",
        "000→100": "C",
        "110→010": "C",
        "110→100": "H",
        "110→111": "O",
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
            m = re.search(
                r"constructed=1.*S→(\d{3})\s+hit=(\d)",
                line,
            )
            if not m or not cur:
                continue
            src_s, tgt_s = cur.split("→")
            src, tgt = _parse(src_s), _parse(tgt_s)
            Sa = _parse(m.group(1))
            out.append(
                {
                    "source": "9C_log",
                    "edge": cur,
                    "channel": ch_of.get(cur),
                    "S_before": list(src),
                    "S_after": list(Sa),
                    "from": list(src),
                    "to": list(tgt),
                    "hit_edge": int(Sa == tgt),
                }
            )
    return out


def _dedupe_9c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer JSON rows; fill missing edges from logs."""
    by_edge: dict[str, list] = defaultdict(list)
    for r in rows:
        by_edge[r["edge"]].append(r)
    # if JSON already has an edge, drop log duplicates for that edge
    json_edges = {r["edge"] for r in rows if r["source"] == "9C"}
    out = [r for r in rows if r["source"] == "9C"]
    for r in rows:
        if r["source"] == "9C_log" and r["edge"] not in json_edges:
            out.append(r)
    return out


def _load_9b(conditioned: bool) -> list[dict[str, Any]]:
    if not P9B.exists():
        return []
    j = json.loads(P9B.read_text())
    out = []
    for ek, rows in (j.get("trials") or {}).items():
        for row in rows:
            if conditioned and not row.get("at_from"):
                continue
            Sb = row["S_home"] if not conditioned else row["from"]
            # when conditioned, S_before is source; when not, use actual home state
            if conditioned:
                Sb = row["from"]
            else:
                Sb = row["S_home"]
            out.append(
                {
                    "source": "9B_cond" if conditioned else "9B_all",
                    "edge": ek,
                    "channel": row["channel"],
                    "S_before": list(Sb),
                    "S_after": list(row["S_after"]),
                    "from": list(row["from"]),
                    "to": list(row["to"]),
                    "hit_edge": int(row.get("hit") or 0),
                }
            )
    return out


def _by_edge(rows: list[dict[str, Any]], mstar_mode: str) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["edge"]].append(r)
    out = {}
    for ek, rs in sorted(groups.items()):
        if mstar_mode == "local":
            mstar = tuple(rs[0]["to"])
        elif mstar_mode == "111":
            mstar = (1, 1, 1)
        elif mstar_mode == "000":
            mstar = (0, 0, 0)
        else:
            raise ValueError(mstar_mode)
        st = _stats(rs, mstar)
        st["mstar"] = _key(mstar)
        st["channel"] = rs[0].get("channel")
        st["P_hit_edge"] = sum(r["hit_edge"] for r in rs) / len(rs)
        # drop bulky transitions in by-edge summary? keep compact
        trans = st.pop("transitions")
        st["n_unique_after"] = len({_key(t["S_after"]) for t in trans})
        st["after_hist"] = dict(
            sorted(
                Counter(_key(t["S_after"]) for t in trans).items(),
                key=lambda kv: -kv[1],
            )
        )
        out[ek] = st
    return out


def _by_channel(rows: list[dict[str, Any]], mstar: tuple[int, int, int]) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        ch = r.get("channel") or "?"
        groups[ch].append(r)
    return {ch: _stats(rs, mstar) | {"channel": ch, "mstar": _key(mstar)} for ch, rs in sorted(groups.items())}


def _fmt(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    rows_9c = _dedupe_9c(_load_9c_from_json() + _load_9c_from_logs())
    rows_9b_cond = _load_9b(conditioned=True)
    rows_9b_all = _load_9b(conditioned=False)

    # For 9C sink analysis, primary m* = local edge target
    by_edge_local = _by_edge(rows_9c, "local")
    by_edge_111 = _by_edge(rows_9c, "111")
    by_edge_000 = _by_edge(rows_9c, "000")

    # Aggregate Gamma ranking for 9C local
    ranking = sorted(
        (
            (ek, st["Gamma"], st["P_down"], st["P_up"], st["P_shift"], st["P_hit_edge"], st["n"])
            for ek, st in by_edge_local.items()
            if st["n"] > 0
        ),
        key=lambda x: (x[1] is not None, x[1]),
        reverse=True,
    )

    # Channel aggregates vs 111 (8-way corner) on 9C + 9B_all
    ch_9c_111 = _by_channel(rows_9c, (1, 1, 1))
    ch_9b_111 = _by_channel(rows_9b_all, (1, 1, 1))

    # Strip transitions from channel stats for JSON size
    def strip_trans(d):
        out = {}
        for k, v in d.items():
            vv = dict(v)
            vv.pop("transitions", None)
            out[k] = vv
        return out

    gate = {
        "hypothesis": (
            "Edge hit rates understate control; Γ=P↓-P↑ reveals directional tendency"
        ),
        "n_9c_trials": len(rows_9c),
        "n_9b_cond": len(rows_9b_cond),
        "n_9b_all": len(rows_9b_all),
        "any_positive_Gamma_local_9c": any(
            (st.get("Gamma") or 0) > 0 for st in by_edge_local.values() if st.get("n")
        ),
        "worst_Gamma_local": ranking[-1] if ranking else None,
        "best_Gamma_local": ranking[0] if ranking else None,
        "read": (
            "If Γ≪0 on sink-in interventions, actuators push away from the intended "
            "sink even while 'failing' the exact edge — binary P(s→s') hides this."
        ),
    }

    payload = {
        "protocol": "Phase 9D shift / Γ reanalysis of 9B+9C trajectories",
        "definitions": {
            "E": "||m*-S||_1",
            "P_down": "P(E_after < E_before)",
            "P_up": "P(E_after > E_before)",
            "P_neutral": "P(E_after = E_before)",
            "Gamma": "P_down - P_up",
            "P_shift": "P(S_after != S_before)",
            "P_r": "P(d_H(S_after,m*)=r)",
        },
        "gate": gate,
        "9c_by_edge_local_mstar": strip_trans(by_edge_local),
        "9c_by_edge_mstar_111": strip_trans(by_edge_111),
        "9c_by_edge_mstar_000": strip_trans(by_edge_000),
        "9c_by_channel_mstar_111": strip_trans(ch_9c_111),
        "9b_by_edge_local": strip_trans(_by_edge(rows_9b_all, "local")),
        "9b_cond_by_edge_local": strip_trans(_by_edge(rows_9b_cond, "local")),
        "9b_by_channel_mstar_111": strip_trans(ch_9b_111),
        "9c_Gamma_ranking_local": [
            {
                "edge": ek,
                "Gamma": g,
                "P_down": pd,
                "P_up": pu,
                "P_shift": ps,
                "P_hit_edge": ph,
                "n": n,
            }
            for ek, g, pd, pu, ps, ph, n in ranking
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 9D — Target-aware shift / $\\Gamma$ (reanalysis)",
        "",
        "> Offline on Phase 9B/9C trajectories. No new $v$, no controller change.",
        "",
        r"$E=\|m^*-S\|_1$,\quad $P_\downarrow=P(E_{\mathrm{after}}<E_{\mathrm{before}})$,\quad "
        r"$P_\uparrow=P(E_{\mathrm{after}}>E_{\mathrm{before}})$,\quad "
        r"$\Gamma=P_\downarrow-P_\uparrow$.",
        "",
        f"9C conditional trials: **{len(rows_9c)}**. "
        f"9B (all home→edge): {len(rows_9b_all)}; 9B at-source: {len(rows_9b_cond)}.",
        "",
        "## 9C edges vs local $m^*=s'$ (intended one-bit target)",
        "",
        "| edge | ch | n | $P_{hit}$ | $P_{shift}$ | $P_\\downarrow$ | $P_\\uparrow$ | $P_0$ | $\\Gamma$ | mean $\\Delta E$ | after hist |",
        "|------|----|--:|----------:|------------:|---------------:|-------------:|------:|----------:|-----------------:------------|",
    ]
    for ek, g, pd, pu, ps, ph, n in ranking:
        st = by_edge_local[ek]
        hist = " ".join(f"{k}:{v}" for k, v in list(st["after_hist"].items())[:4])
        lines.append(
            f"| `{ek}` | {st.get('channel')} | {n} | {_fmt(ph)} | {_fmt(ps)} | "
            f"{_fmt(pd)} | {_fmt(pu)} | {_fmt(st['P_neutral'])} | **{_fmt(g)}** | "
            f"{_fmt(st['mean_dE'])} | {hist} |"
        )

    lines += [
        "",
        "## Same 9C trials vs global $m^*=111$",
        "",
        "| edge | n | $P_\\downarrow$ | $P_\\uparrow$ | $\\Gamma$ | $P_r$ (r=0..3) |",
        "|------|--:|---------------:|-------------:|----------:|----------------|",
    ]
    for ek, *_ in ranking:
        st = by_edge_111[ek]
        pr = st["P_r"]
        prs = ",".join(_fmt(pr[str(r)]) for r in range(4))
        lines.append(
            f"| `{ek}` | {st['n']} | {_fmt(st['P_down'])} | {_fmt(st['P_up'])} | "
            f"**{_fmt(st['Gamma'])}** | {prs} |"
        )

    lines += [
        "",
        "## Channel aggregate vs $m^*=111$",
        "",
        "### 9C sink-neighborhood interventions",
        "",
        "| ch | n | $P_{shift}$ | $P_\\downarrow$ | $P_\\uparrow$ | $\\Gamma$ |",
        "|----|--:|------------:|---------------:|-------------:|----------:|",
    ]
    for ch, st in ch_9c_111.items():
        st2 = {k: v for k, v in st.items() if k != "transitions"}
        lines.append(
            f"| {ch} | {st2['n']} | {_fmt(st2['P_shift'])} | {_fmt(st2['P_down'])} | "
            f"{_fmt(st2['P_up'])} | **{_fmt(st2['Gamma'])}** |"
        )

    lines += [
        "",
        "### 9B all edges (S_before = S_home)",
        "",
        "| ch | n | $P_{shift}$ | $P_\\downarrow$ | $P_\\uparrow$ | $\\Gamma$ |",
        "|----|--:|------------:|---------------:|-------------:|----------:|",
    ]
    for ch, st in ch_9b_111.items():
        lines.append(
            f"| {ch} | {st['n']} | {_fmt(st['P_shift'])} | {_fmt(st['P_down'])} | "
            f"{_fmt(st['P_up'])} | **{_fmt(st['Gamma'])}** |"
        )

    # Highlight sink story
    sink_notes = []
    for ek in ("010→000", "100→000", "010→110", "100→110", "111→110"):
        if ek not in by_edge_local:
            continue
        st = by_edge_local[ek]
        sink_notes.append(
            f"- `{ek}`: $P_{{hit}}={_fmt(st['P_hit_edge'])}$ but "
            f"$P_{{shift}}={_fmt(st['P_shift'])}$, $\\Gamma={_fmt(st['Gamma'])}$ "
            f"(mean $\\Delta E={_fmt(st['mean_dE'])}$)"
        )

    lines += [
        "",
        "## Sink read",
        "",
        *sink_notes,
        "",
        "## Gate",
        "",
        f"- Any $\\Gamma>0$ (local) on 9C densified edges: **{gate['any_positive_Gamma_local_9c']}**",
        f"- Best local $\\Gamma$: `{gate['best_Gamma_local']}`",
        f"- Worst local $\\Gamma$: `{gate['worst_Gamma_local']}`",
        "",
        gate["read"],
        "",
        r"$$\boxed{\Gamma\text{ measures control tendency; edge }P(s\to s')\text{ alone is too brittle}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
