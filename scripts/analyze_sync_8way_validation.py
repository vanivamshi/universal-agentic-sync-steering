#!/usr/bin/env python3
"""Enrich 8-way JSON with per-m* tables + C/O-hard strata; rewrite MD + decision.

  .venv/bin/python scripts/analyze_sync_8way_validation.py [path.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "results" / "sync_8way_validation.json"
ARMS = ("none", "predictive", "converted", "random")
MKEYS = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]


def _agg(trials: list[dict]) -> dict:
    if not trials:
        return {"n": 0, "P_hit": float("nan"), "mean_delta_E_total": float("nan"),
                "mean_delta_E_C": float("nan"), "mean_delta_E_H": float("nan"),
                "mean_delta_E_O": float("nan"), "P_C": float("nan"), "P_H": float("nan"),
                "P_O": float("nan")}
    def m(k):
        return float(np.mean([t[k] for t in trials]))
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "mean_delta_E_total": m("delta_E_total"),
        "mean_delta_E_C": m("delta_E_C"),
        "mean_delta_E_H": m("delta_E_H"),
        "mean_delta_E_O": m("delta_E_O"),
        "P_C": m("bit_C"),
        "P_H": m("bit_H"),
        "P_O": m("bit_O"),
    }


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    p = json.loads(path.read_text())
    trials = p["trials"]
    arms = [a for a in ARMS if a in trials]
    aggs = {a: _agg(trials[a]) for a in arms}

    # per m*
    per = {}
    for a in arms:
        per[a] = {}
        for key in MKEYS:
            mstar = [int(c) for c in key]
            sub = [t for t in trials[a] if t["mstar"] == mstar]
            per[a][key] = _agg(sub)

    # hard strata: based on whether free-run S0 already matches bits — but user asked
    # about *targets* where C/O must change relative to typical baseline.
    # Stratify by target bits: C*=1, O*=1, C*≠O* etc., and by whether S0 needs ΔC/ΔO.
    def needs_flip(trials_arm: list, bit_idx: int) -> list:
        return [t for t in trials_arm if t["S0"][bit_idx] != t["mstar"][bit_idx]]

    strata = {}
    for a in arms:
        ta = trials[a]
        strata[a] = {
            "all": _agg(ta),
            "need_C": _agg(needs_flip(ta, 0)),
            "need_H": _agg(needs_flip(ta, 1)),
            "need_O": _agg(needs_flip(ta, 2)),
            "need_C_or_O": _agg(
                [t for t in ta if t["S0"][0] != t["mstar"][0] or t["S0"][2] != t["mstar"][2]]
            ),
            "need_C_and_O": _agg(
                [t for t in ta if t["S0"][0] != t["mstar"][0] and t["S0"][2] != t["mstar"][2]]
            ),
            "mstar_011": _agg([t for t in ta if t["mstar"] == [0, 1, 1]]),
            "mstar_not_011": _agg([t for t in ta if t["mstar"] != [0, 1, 1]]),
        }

    pc, pr, pn, pp = (aggs.get(x) for x in ("converted", "random", "none", "predictive"))
    decision = "inconclusive"
    if pc and pr and pn:
        de_ok = pc["mean_delta_E_total"] > pr["mean_delta_E_total"] and pc["mean_delta_E_total"] > pn["mean_delta_E_total"]
        hit_ok = pc["P_hit"] > pr["P_hit"] and pc["P_hit"] > pn["P_hit"]
        if de_ok and hit_ok:
            decision = "vc_wins_both → policy optimization justified"
        elif de_ok and not hit_ok:
            decision = "vc_wins_ΔE_only → state/decision policy justified"
        elif (not de_ok) and pr["P_hit"] > pc["P_hit"]:
            decision = "random_confound → diagnose; no policy yet"
        else:
            decision = "mixed → inspect per-m* / hard strata"

    out_json = path.with_name(path.stem + "_analysis.json")
    blob = {
        "source": str(path),
        "reps": p.get("reps"),
        "agg": aggs,
        "per_mstar": per,
        "strata": strata,
        "decision": decision,
    }
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    md = path.with_suffix(".md")
    # also write enriched companion
    enrich = path.with_name(path.stem + "_enriched.md")
    lines = [
        f"# 8-way enriched analysis (reps={p.get('reps')})",
        "",
        f"Source: `{path.name}`",
        "",
        f"**Decision:** {decision}",
        "",
        "## Aggregate (primary ΔE, secondary hit)",
        "",
        "| Arm | n | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |",
        "|-----|---|--------|-----|------|------|------|------|------|------|",
    ]
    for a in arms:
        g = aggs[a]
        lines.append(
            f"| {a} | {g['n']} | {g['P_hit']:.3f} | {g['mean_delta_E_total']:+.3f} | "
            f"{g['mean_delta_E_C']:+.3f} | {g['mean_delta_E_H']:+.3f} | {g['mean_delta_E_O']:+.3f} | "
            f"{g['P_C']:.3f} | {g['P_H']:.3f} | {g['P_O']:.3f} |"
        )

    lines += ["", "## Per $m^*$ — ΔE / P(hit)", ""]
    hdr = "| $m^*$ | " + " | ".join(f"{a} ΔE" for a in arms) + " | " + " | ".join(f"{a} hit" for a in arms) + " |"
    sep = "|-------|" + "|".join(["------:" for _ in arms]) + "|" + "|".join(["------:" for _ in arms]) + "|"
    lines += [hdr, sep]
    for key in MKEYS:
        de = [f"{per[a][key]['mean_delta_E_total']:+.2f}" if per[a][key]["n"] else "—" for a in arms]
        ht = [f"{per[a][key]['P_hit']:.2f}" if per[a][key]["n"] else "—" for a in arms]
        lines.append(f"| {key} | " + " | ".join(de) + " | " + " | ".join(ht) + " |")

    lines += ["", "## Hard strata (converted vs random vs none)", ""]
    lines.append("| Stratum | none ΔE | conv ΔE | rand ΔE | none hit | conv hit | rand hit | n_conv |")
    lines.append("|---------|--------:|--------:|--------:|---------:|---------:|---------:|-------:|")
    for sk in ("all", "need_C", "need_H", "need_O", "need_C_or_O", "need_C_and_O", "mstar_011", "mstar_not_011"):
        cells = []
        for a in ("none", "converted", "random"):
            s = strata.get(a, {}).get(sk, {})
            cells.append(f"{s.get('mean_delta_E_total', float('nan')):+.2f}")
        for a in ("none", "converted", "random"):
            s = strata.get(a, {}).get(sk, {})
            cells.append(f"{s.get('P_hit', float('nan')):.2f}")
        n = strata.get("converted", {}).get(sk, {}).get("n", 0)
        lines.append(f"| {sk} | " + " | ".join(cells) + f" | {n} |")

    lines += ["", "## Decision rule", "", f"- {decision}", ""]
    enrich.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"analysis": str(out_json), "enriched_md": str(enrich), "decision": decision, "agg": aggs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
