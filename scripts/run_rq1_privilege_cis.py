#!/usr/bin/env python3
"""BCa + joint CIs for Step 5 / RQ1 privilege contrasts (existing window rows).

Joint: each replicate uses paired prose/tool within transcript to form Δ, then
aggregates directions into the contrast — same spirit as §3 joint gap bootstrap.
BCa: bias-corrected accelerated interval with jackknife over transcripts.
Percentile intervals are retained only as a pathology check (§3 precedent).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pilot",
        type=Path,
        default=ROOT / "data" / "results" / "rq1_privilege_pilot.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "rq1_privilege_pilot_cis.json",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--update-pilot", action="store_true", default=True)
    ap.add_argument("--no-update-pilot", action="store_false", dest="update_pilot")
    args = ap.parse_args()

    from activation_pipeline.analysis.bootstrap_ci import bca_ci, percentile_ci

    pilot = json.loads(args.pilot.read_text())
    rows = pilot["window_rows"]
    meta = {d["direction_id"]: d for d in pilot["per_direction"]}

    # Window → transcript-level mean ε* by mode, then joint Δ = tool − prose
    by_dir_tr: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"prose": [], "tool_call": []})
    )
    for r in rows:
        by_dir_tr[r["direction_id"]][r["transcript_id"]][r["window_kind"]].append(
            r["epsilon_star"]
        )

    deltas_by_dir: dict[str, dict[str, float]] = {}
    for did, tr_map in by_dir_tr.items():
        dmap: dict[str, float] = {}
        for tid, modes in tr_map.items():
            if not modes["prose"] or not modes["tool_call"]:
                continue
            p = sum(modes["prose"]) / len(modes["prose"])
            t = sum(modes["tool_call"]) / len(modes["tool_call"])
            dmap[tid] = t - p  # joint within transcript
        deltas_by_dir[did] = dmap

    all_tr = sorted({tid for m in deltas_by_dir.values() for tid in m})
    n_tr = len(all_tr)
    if n_tr < 3:
        raise SystemExit(f"need ≥3 transcripts for jackknife; got {n_tr}")

    safety_ids = [
        d
        for d, m in meta.items()
        if m["direction_type"] == "safety" and m["kind"] != "refusal"
    ]
    learned_ids = [d for d, m in meta.items() if m["direction_type"] == "learned_control"]
    random_ids = [d for d, m in meta.items() if m["direction_type"] == "random_control"]
    pooled_ids = learned_ids + random_ids

    def mean_delta(direction_ids: list[str], tr_ids: list[str]) -> float | None:
        vals = []
        for did in direction_ids:
            dmap = deltas_by_dir[did]
            sampled = [dmap[t] for t in tr_ids if t in dmap]
            if sampled:
                vals.append(sum(sampled) / len(sampled))
        if not vals:
            return None
        return sum(vals) / len(vals)

    def interaction(
        safety: list[str], controls: list[str], tr_ids: list[str]
    ) -> float | None:
        s = mean_delta(safety, tr_ids)
        c = mean_delta(controls, tr_ids)
        if s is None or c is None:
            return None
        return s - c

    def bca_for_stat(stat_fn, label: str) -> dict:
        point = stat_fn(all_tr)
        if point is None:
            return {"label": label, "point": None}

        rng = random.Random(args.seed)
        boots: list[float] = []
        for b in range(args.n_boot):
            brng = random.Random(args.seed + 10_000 + b)
            sample = [brng.choice(all_tr) for _ in all_tr]
            val = stat_fn(sample)
            if val is not None:
                boots.append(val)

        jacks: list[float] = []
        for leave in all_tr:
            jack_ids = [t for t in all_tr if t != leave]
            val = stat_fn(jack_ids)
            if val is not None:
                jacks.append(val)

        lo_bca, hi_bca, diag = bca_ci(point, boots, jacks)
        lo_pct, hi_pct = percentile_ci(boots)
        return {
            "label": label,
            "point": point,
            "boot_mean": sum(boots) / len(boots) if boots else None,
            "n_boot": len(boots),
            "n_jack": len(jacks),
            "n_transcripts": n_tr,
            "ci_method": "bca_joint",
            "bca_ci95": [lo_bca, hi_bca],
            "percentile_ci95": [lo_pct, hi_pct],
            "bca_width": hi_bca - lo_bca,
            "percentile_width": hi_pct - lo_pct,
            "bca_excludes_zero": bool(lo_bca > 0 or hi_bca < 0),
            "percentile_excludes_zero": bool(lo_pct > 0 or hi_pct < 0),
            "bca_diagnostics": diag,
            "note": (
                "Joint: each replicate forms transcript-level Δ=ε*_tool−ε*_prose "
                "from paired modes, then aggregates directions. BCa with jackknife "
                "over transcripts. Percentile retained as pathology check (§3)."
            ),
        }

    group_specs = [
        ("safety_excluding_blocked_refusal", safety_ids),
        ("learned_controls", learned_ids),
        ("random_controls", random_ids),
        ("pooled_controls", pooled_ids),
    ]
    interaction_specs = [
        ("safety_minus_learned", safety_ids, learned_ids),
        ("safety_minus_pooled", safety_ids, pooled_ids),
        ("safety_minus_random", safety_ids, random_ids),
    ]

    out = {
        "ci_method": "bca_joint",
        "bootstrap_n": args.n_boot,
        "seed": args.seed,
        "n_transcripts": n_tr,
        "cluster": "transcript",
        "rationale": (
            "§3 moved from percentile to BCa+joint after small-n undercoverage "
            "(L14 flip). §5 uses the same cluster count order (n=8 transcripts), "
            "so interaction/group CIs use BCa+joint; percentile kept for pathology."
        ),
        "group_mean_delta": {},
        "interaction": {},
        "per_direction_delta": {},
    }

    print(f"BCa+joint RQ1 CIs n_tr={n_tr} n_boot={args.n_boot}")
    for name, ids in group_specs:
        res = bca_for_stat(lambda tr, ids=ids: mean_delta(ids, tr), name)
        out["group_mean_delta"][name] = res
        print(
            f"  group {name}: point={res['point']:+.3f} "
            f"BCa={res['bca_ci95']} pct={res['percentile_ci95']} "
            f"excl0_bca={res['bca_excludes_zero']}"
        )

    for name, sids, cids in interaction_specs:
        res = bca_for_stat(
            lambda tr, s=sids, c=cids: interaction(s, c, tr), name
        )
        out["interaction"][name] = res
        print(
            f"  inter {name}: point={res['point']:+.3f} "
            f"BCa={res['bca_ci95']} pct={res['percentile_ci95']} "
            f"excl0_bca={res['bca_excludes_zero']}"
        )

    for did in meta:
        res = bca_for_stat(lambda tr, d=did: mean_delta([d], tr), did)
        res["kind"] = meta[did]["kind"]
        res["direction_type"] = meta[did]["direction_type"]
        out["per_direction_delta"][did] = res
        print(
            f"  dir {did}: point={res['point']:+.3f} "
            f"BCa={res['bca_ci95']} excl0={res['bca_excludes_zero']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")

    if args.update_pilot:
        li = out["interaction"]["safety_minus_learned"]
        agg = pilot["aggregates"]
        agg["ci_method"] = "bca_joint"
        agg["bootstrap"] = {
            "n": args.n_boot,
            "seed": args.seed,
            "method": "bca_joint_transcript_clusters",
        }
        agg["group_mean_delta_ci95"] = out["group_mean_delta"]
        agg["interaction_ci95"] = out["interaction"]
        agg["per_direction_delta_ci95"] = out["per_direction_delta"]
        agg["interaction_safety_minus_learned"] = li["point"]
        agg["interaction_learned_bca_ci95"] = li["bca_ci95"]
        agg["interaction_learned_percentile_ci95"] = li["percentile_ci95"]
        agg["interaction_learned_excludes_zero"] = li["bca_excludes_zero"]
        # Keep legacy key pointing at BCa now
        agg["interaction_learned_ci95"] = li["bca_ci95"]
        agg["h1_supported_vs_learned_controls"] = bool(li["bca_ci95"][0] > 0)
        pooled = out["interaction"]["safety_minus_pooled"]
        agg["interaction_safety_minus_controls"] = pooled["point"]
        agg["interaction_percentile_ci95"] = pooled["percentile_ci95"]
        agg["interaction_bca_ci95"] = pooled["bca_ci95"]
        agg["h1_supported_by_pilot_ci"] = bool(pooled["bca_ci95"][0] > 0)
        agg["pilot_interpretation"] = (
            "Learned-control interaction point≈+0.02 with BCa+joint CI95 "
            f"[{li['bca_ci95'][0]:+.2f},{li['bca_ci95'][1]:+.2f}] "
            f"(percentile was [{li['percentile_ci95'][0]:+.2f},{li['percentile_ci95'][1]:+.2f}]). "
            "Consistent with null; interval still wide at n=8 transcripts. "
            "CI method aligned with §3 (BCa+joint; percentile pathology-only). "
            "Pooled positivity remains random-driven. Not confirmatory on 0.6B."
        )
        if "random_delta_investigation" in pilot:
            pilot["random_delta_investigation"]["phrasing_note"] = (
                "Type×mode divergence redescribes the pattern and justifies "
                "keeping random as its own estimand; it is not a mechanism for "
                "why isotropic residuals gain privilege in tool mode. Mechanism "
                "remains open. Norm-matched / absolute blowup is a required 32B "
                "analysis, not optional."
            )
            pilot["random_delta_investigation"]["design_implication_32b"] = (
                "REQUIRED at 32B: (1) primary H1 = safety vs learned with BCa+joint "
                "transcript-cluster CI; (2) norm-matched relative and/or absolute "
                "blowup sensitivity to decompose random negative Δ; (3) report "
                "random mean Δ and ||h_L|| mode ratio as separate diagnostics."
            )
        pilot["status"] = "pilot_complete_bca_joint_cis"
        pilot["ci_method"] = "bca_joint"
        args.pilot.write_text(json.dumps(pilot, indent=2))
        print(f"updated {args.pilot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
