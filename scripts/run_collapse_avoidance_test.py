#!/usr/bin/env python3
"""Collapse-avoidance: Fisher/binomial on existing Phase 1b draws.

Locked: docs/collapse_avoidance_protocol.md
No new generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P1B = ROOT / "data" / "results" / "dimension_selection_phase1b.json"
P1A = ROOT / "data" / "results" / "dimension_selection_phase1.json"
OUT = ROOT / "data" / "results" / "collapse_avoidance_test.json"
MD = ROOT / "data" / "results" / "collapse_avoidance_test.md"
ALPHA = 0.10


def _collapse(rec: dict) -> bool:
    return float(rec["control_correct"]) == 0.0


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def fisher_exact_p(table: np.ndarray, alternative: str) -> tuple[float, float]:
    """Return (odds_ratio, p). Hypergeometric, 2×2."""
    a, b = int(table[0, 0]), int(table[0, 1])
    c, d = int(table[1, 0]), int(table[1, 1])
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c
    odds = (a * d) / (b * c) if b * c else (math.inf if a * d else float("nan"))

    def p_table(aa: int) -> float:
        bb = r1 - aa
        cc = c1 - aa
        dd = r2 - cc
        if min(aa, bb, cc, dd) < 0:
            return 0.0
        return (
            _comb(r1, aa) * _comb(r2, cc) / _comb(n, c1)
            if _comb(n, c1)
            else 0.0
        )

    lo = max(0, c1 - r2)
    hi = min(c1, r1)
    p_obs = p_table(a)
    if alternative == "less":
        p = sum(p_table(x) for x in range(lo, a + 1))
    elif alternative == "greater":
        p = sum(p_table(x) for x in range(a, hi + 1))
    else:
        p = sum(p_table(x) for x in range(lo, hi + 1) if p_table(x) <= p_obs + 1e-15)
    return float(odds), float(min(1.0, p))


def binom_pmf(k: int, n: int, p: float) -> float:
    return _comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def _ci(k: int, n: int) -> tuple[float, float]:
    """Clopper–Pearson 95% CI by inverting the binomial CDF."""

    def p_le(p: float) -> float:
        return sum(binom_pmf(i, n, p) for i in range(k + 1))

    def p_ge(p: float) -> float:
        return sum(binom_pmf(i, n, p) for i in range(k, n + 1))

    def bisect(fn, target: float, *, increasing: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            hit = fn(mid) < target
            if increasing:
                if hit:
                    lo = mid
                else:
                    hi = mid
            else:
                if hit:
                    hi = mid
                else:
                    lo = mid
        return 0.5 * (lo + hi)

    lo = 0.0 if k == 0 else bisect(p_ge, 0.025, increasing=True)
    hi = 1.0 if k == n else bisect(p_le, 0.025, increasing=False)
    return lo, hi


def main() -> int:
    p1b = json.loads(P1B.read_text())
    ident = p1b["identity"]
    sel = []
    for name, rec in p1b["locked_conditions"].items():
        row = {
            "tag": name,
            "k": rec["n_channels"],
            "control_correct": rec["control_correct"],
            "unique_token_ratio": rec["unique_token_ratio"],
            "heldout_ppl": rec["heldout_ppl"],
            "gen_ppl": rec["gen_ppl"],
            "collapse": _collapse(rec),
        }
        sel.append(row)
    rnd = []
    for d in p1b["draws"]:
        row = {
            "tag": f"random_{d['k']}_seed_{d['seed']}",
            "k": d["k"],
            "seed": d["seed"],
            "control_correct": d["control_correct"],
            "unique_token_ratio": d["unique_token_ratio"],
            "heldout_ppl": d["heldout_ppl"],
            "gen_ppl": d["gen_ppl"],
            "collapse": _collapse(d),
        }
        rnd.append(row)

    n_sel, k_sel = len(sel), sum(r["collapse"] for r in sel)
    n_rnd, k_rnd = len(rnd), sum(r["collapse"] for r in rnd)
    table = np.array([[k_sel, n_sel - k_sel], [k_rnd, n_rnd - k_rnd]], dtype=int)
    odds, p_two = fisher_exact_p(table, alternative="two-sided")
    _, p_less = fisher_exact_p(table, alternative="less")
    p_hat_r = k_rnd / n_rnd
    p_zero = binom_pmf(0, n_sel, p_hat_r)

    # sensitivity: 2 methods (collapse if either k)
    by_method = {"nll": False, "gxa": False}
    for r in sel:
        m = "nll" if r["tag"].startswith("nll") else "gxa"
        by_method[m] = by_method[m] or r["collapse"]
    n_m, k_m = 2, sum(by_method.values())
    table_m = np.array([[k_m, n_m - k_m], [k_rnd, n_rnd - k_rnd]], dtype=int)
    _, p_m = fisher_exact_p(table_m, alternative="less")

    sel64 = [r for r in sel if r["k"] == 64]
    rnd64 = [r for r in rnd if r["k"] == 64]
    table64 = np.array(
        [
            [sum(r["collapse"] for r in sel64), len(sel64) - sum(r["collapse"] for r in sel64)],
            [sum(r["collapse"] for r in rnd64), len(rnd64) - sum(r["collapse"] for r in rnd64)],
        ],
        dtype=int,
    )
    _, p_64 = fisher_exact_p(table64, alternative="less")

    decision = "AVOID_PROMISING" if p_less < ALPHA else "AVOID_UNRESOLVED"
    note = (
        "one-sided Fisher p < 0.10 — license powered follow-up."
        if decision == "AVOID_PROMISING"
        else (
            "existing n cannot reject equal collapse rates. Nested selector "
            "cells overstate n=4. Do not expand. Do not claim a safety margin."
        )
    )

    p1a_note = None
    if P1A.is_file():
        p1a = json.loads(P1A.read_text())
        k_a = sum(1 for d in p1a["draws"] if float(d["control_correct"]) == 0.0)
        p1a_note = {
            "n_random": len(p1a["draws"]),
            "n_collapse": k_a,
            "rate": k_a / len(p1a["draws"]),
            "not_in_primary": True,
        }

    sel_ci = _ci(k_sel, n_sel)
    rnd_ci = _ci(k_rnd, n_rnd)
    report = {
        "stage": "COLLAPSE_AVOIDANCE_EXISTING",
        "collapse_rule": "control_correct == 0",
        "identity_correct": ident["control_correct"],
        "identity_uniq": ident["unique_token_ratio"],
        "identity_held_ppl": ident["heldout_ppl"],
        "selector": sel,
        "random": [
            {k: r[k] for k in ("tag", "k", "collapse", "control_correct", "unique_token_ratio", "heldout_ppl")}
            for r in rnd
        ],
        "primary": {
            "selector_collapses": f"{k_sel}/{n_sel}",
            "random_collapses": f"{k_rnd}/{n_rnd}",
            "table_collapse_ok": table.tolist(),
            "fisher_odds_ratio": float(odds),
            "fisher_p_two_sided": float(p_two),
            "fisher_p_one_sided_less": float(p_less),
            "clopper_pearson_95_selector": sel_ci,
            "clopper_pearson_95_random": rnd_ci,
            "p_hat_random": p_hat_r,
            "binom_P_zero_given_p_hat_random": p_zero,
        },
        "sensitivity_two_methods": {
            "collapses": f"{k_m}/{n_m}",
            "fisher_p_one_sided_less": float(p_m),
        },
        "sensitivity_k64_only": {
            "selector": f"{int(table64[0, 0])}/{len(sel64)}",
            "random": f"{int(table64[1, 0])}/{len(rnd64)}",
            "fisher_p_one_sided_less": float(p_64),
        },
        "alpha": ALPHA,
        "decision": decision,
        "note": note,
        "phase1a_random_robustness": p1a_note,
        "gain_question": "NOISE_STOP unchanged",
        "causal_claim": False,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    def _fmt_ci(ci: tuple[float, float]) -> str:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    md = [
        "# Collapse-avoidance test (existing Phase 1b, no new gens)",
        "",
        "Collapse = `control_correct == 0`. Gain question stays `NOISE_STOP`.",
        "",
        f"## Decision: `{decision}`",
        note,
        "",
        "| arm | collapses | rate | 95% CI (Clopper–Pearson) |",
        "|---|---:|---:|---|",
        f"| selector (4 locked bottoms) | {k_sel}/{n_sel} | {k_sel / n_sel:.3f} | {_fmt_ci(sel_ci)} |",
        f"| random (10×16 + 10×64) | {k_rnd}/{n_rnd} | {k_rnd / n_rnd:.3f} | {_fmt_ci(rnd_ci)} |",
        "",
        f"Fisher exact one-sided (H1: p_sel < p_rand): **p = {p_less:.3f}** "
        f"(two-sided p = {p_two:.3f}).",
        f"P(0 collapses | Binomial(n=4, p={p_hat_r:.2f})) = {p_zero:.3f}.",
        "",
        "Collapsed random draws:",
    ]
    for r in rnd:
        if r["collapse"]:
            md.append(
                f"- `{r['tag']}` correct=0 uniq={r['unique_token_ratio']:.3f} "
                f"held_ppl={r['heldout_ppl']:.2f}"
            )
    md += [
        "",
        "All four selector bottoms: no collapse "
        + ", ".join(
            f"{r['tag']} correct={r['control_correct']:.3f} uniq={r['unique_token_ratio']:.3f}"
            for r in sel
        )
        + ".",
        "",
        "## Sensitivity (dependence)",
        f"- 2 methods (either-k): {k_m}/{n_m} vs {k_rnd}/{n_rnd}, one-sided Fisher p = {p_m:.3f}",
        f"- k=64 only: {int(table64[0,0])}/{len(sel64)} vs {int(table64[1,0])}/{len(rnd64)}, "
        f"one-sided Fisher p = {p_64:.3f}",
        "",
        "Phase 1a n=12 random (not pooled): "
        + (
            f"{p1a_note['n_collapse']}/{p1a_note['n_random']} collapse"
            if p1a_note
            else "n/a"
        )
        + ".",
        "",
        f"Artifact: `{OUT}`",
    ]
    MD.write_text("\n".join(md) + "\n")
    print(json.dumps({"decision": decision, "p_one_sided": p_less, "table": table.tolist()}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
