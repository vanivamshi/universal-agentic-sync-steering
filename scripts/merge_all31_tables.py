#!/usr/bin/env python3
"""Merge Layer 1 characterize + Layer 2 probe into one all-31 table."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}"


def main() -> int:
    l1 = json.loads((ROOT / "data/results/all31_layer1_characterize.json").read_text())
    l2 = json.loads((ROOT / "data/results/all31_layer2_probe.json").read_text())
    rows_l1 = {r["pc_index"]: r for r in l1["rows"]}
    by_pc = l2.get("by_pc") or {}

    reduce_pcs = []
    increase_pcs = []
    null_pcs = []
    merged = []
    for i in range(31):
        a = rows_l1[i]
        blk = by_pc.get(f"pc{i}") or {}
        curves = blk.get("curves") or {}
        c_m2 = curves.get("-2.0") or curves.get("-2") or {}
        c_p2 = curves.get("2.0") or curves.get("2") or {}
        status = blk.get("status") or "missing"
        chosen = blk.get("chosen_alpha")
        d_s = None
        if chosen is not None:
            key = str(float(chosen))
            d_s = (curves.get(key) or {}).get("delta_surface_gap")
        if status == "PROBE_NULL":
            bucket = "NULL"
            null_pcs.append(i)
        elif d_s is not None and d_s < 0:
            bucket = "REDUCE"
            reduce_pcs.append(i)
        elif status == "PROBE_PASS":
            bucket = "INCREASE"
            increase_pcs.append(i)
        else:
            bucket = status
        merged.append(
            {
                "pc_index": i,
                "var": a.get("explained_variance_ratio"),
                "r_surface": a.get("r_surface_gap"),
                "r_viol": a.get("r_tool_violation"),
                "ridge_surface": a.get("ridge_coef_surface_gap"),
                "logit_pos": a.get("logit_top_pos") or [],
                "logit_neg": a.get("logit_top_neg") or [],
                "d_surf_m2": c_m2.get("delta_surface_gap"),
                "d_task_m2": c_m2.get("delta_task_attempted"),
                "d_surf_p2": c_p2.get("delta_surface_gap"),
                "d_task_p2": c_p2.get("delta_task_attempted"),
                "probe_status": status,
                "probe_bucket": bucket,
                "chosen_alpha": chosen,
            }
        )

    out = {
        "n": 31,
        "layer2_base_surface_rate": l2.get("baselines", {}).get("surface_gap_rate"),
        "n_probe_pass": l2.get("n_probe_pass"),
        "n_probe_null": l2.get("n_probe_null"),
        "reduce_pcs": reduce_pcs,
        "increase_pcs": increase_pcs,
        "null_pcs": null_pcs,
        "layer3_rule": (
            "Full J1–J4 only on REDUCE bucket (chosen α lowers surface_gap). "
            "INCREASE = probe moved jailbreak *up* — not a control lever. "
            "NULL = |Δ|<0.10 or task drop too large."
        ),
        "rows": merged,
    }
    json_path = ROOT / "data/results/all31_combined.json"
    md_path = ROOT / "data/results/all31_combined.md"
    json_path.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# All 31 persona PCs — Layer 1 + Layer 2",
        "",
        f"Screen n=18, base surface_gap rate={out['layer2_base_surface_rate']:.3f} "
        f"(~5/18). A 2-trajectory flip is Δ≈0.11, so the |Δ|≥0.10 probe gate is "
        f"**coarse** — many PROBE_PASS are two-count flips.",
        "",
        f"| Bucket | n | Meaning |",
        f"|---|---:|---|",
        f"| REDUCE | {len(reduce_pcs)} | probe α *lowers* surface_gap → Layer 3 |",
        f"| INCREASE | {len(increase_pcs)} | probe α *raises* jailbreak rate — not a control lever |",
        f"| NULL | {len(null_pcs)} | no |Δ|≥0.10 with Δ_task≥−0.15 |",
        "",
        f"REDUCE PCs: {reduce_pcs or '(none)'}",
        f"NULL PCs: {null_pcs}",
        "",
        "| PC | var | r_surf | ridge | α=−2 Δs | α=+2 Δs | bucket | logit + |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in merged:
        lp = " ".join(f"`{t}`" for t in r["logit_pos"][:3])
        lines.append(
            f"| {r['pc_index']} | {(r['var'] or 0):.3f} | {fmt(r['r_surface'])} | "
            f"{fmt(r['ridge_surface'])} | {fmt(r['d_surf_m2'], 3)} | {fmt(r['d_surf_p2'], 3)} | "
            f"{r['probe_bucket']} | {lp} |"
        )
    lines.append("")
    lines.append("Logit-lens = persona/style tokens only. Layer 3 = J1–J4 on REDUCE only.")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("REDUCE", reduce_pcs, "NULL", null_pcs, "INCREASE", len(increase_pcs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
