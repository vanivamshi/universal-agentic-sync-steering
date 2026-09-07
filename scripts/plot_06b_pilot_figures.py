#!/usr/bin/env python3
"""Plot remaining 0.6B pilot figures from already-collected result JSONs.

Produces:
  docs/figures/step_5_rq1_blowup_mode_compare.png
  docs/figures/step_5_rq1_norm_shrinkage_check.png
  docs/figures/step_1_logit_lens_layer_sweep.png
  docs/figures/step_3_direction_stability_layers.png
  docs/figures/step_4_control_split_half_l4.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    # ------------------------------------------------------------------
    # 1. Blowup-mode Δ comparison (relative / absolute / norm_matched)
    # ------------------------------------------------------------------
    var = json.loads((ROOT / "data/results/rq1_blowup_variants_gap_full.json").read_text())
    modes = ["relative", "absolute", "norm_matched"]
    mode_colors = {"relative": "#7a7a7a", "absolute": "#2c7fb8", "norm_matched": "#2ca25f"}

    order = [
        ("assistant_axis_L4", "Assistant Axis", "safety"),
        ("harmlessness_L4", "harmlessness", "safety"),
        ("refusal_L4", "refusal*", "safety"),
        ("syntax_L4", "syntax", "learned"),
        ("domain_content_coding_L4", "coding", "learned"),
        ("domain_content_writing_L4", "writing", "learned"),
        ("domain_content_therapy_L4", "therapy", "learned"),
        ("rand_00_L4", "rand_00", "random"),
        ("rand_01_L4", "rand_01", "random"),
        ("rand_02_L4", "rand_02", "random"),
        ("rand_03_L4", "rand_03", "random"),
    ]
    # per-direction only has mean_delta in variants file; use aggregate CIs for right panel
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    y = np.arange(len(order))
    height = 0.22
    offsets = {"relative": -height, "absolute": 0.0, "norm_matched": height}
    for mode in modes:
        pd = var["blowup_variants"][mode]["per_direction"]
        vals = [pd[did]["mean_delta"] if did in pd else np.nan for did, _, _ in order]
        ax.barh(
            y + offsets[mode],
            vals,
            height=height,
            color=mode_colors[mode],
            label=mode,
            alpha=0.9,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in order])
    ax.set_xlabel(r"$\Delta\varepsilon^\star$ (tool − prose)")
    ax.set_title("Per-direction Δ by blowup mode")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    # mark Assistant Axis sign flip
    ax.annotate(
        "AA sign flip",
        xy=(1.48, 0),
        xytext=(3.2, -0.8),
        fontsize=7,
        color="#a50f15",
        arrowprops=dict(arrowstyle="->", color="#a50f15", lw=0.8),
    )

    ax = axes[1]
    agg_keys = [
        ("safety_mean_delta", "safety mean\n(excl. refusal)"),
        ("learned_mean_delta", "learned mean"),
        ("random_mean_delta", "random mean"),
        ("learned_interaction", "safety − learned\n(interaction)"),
    ]
    # variants may store interaction under learned_interaction
    x = np.arange(len(agg_keys))
    width = 0.25
    for i, mode in enumerate(modes):
        blk = var["blowup_variants"][mode]
        pts, los, his = [], [], []
        for key, _ in agg_keys:
            d = blk[key]
            pts.append(d["point"])
            lo, hi = d["bca_ci95"]
            los.append(d["point"] - lo)
            his.append(hi - d["point"])
        ax.bar(
            x + (i - 1) * width,
            pts,
            width,
            color=mode_colors[mode],
            label=mode,
            yerr=np.vstack([los, his]),
            capsize=3,
            error_kw={"lw": 0.9},
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in agg_keys], fontsize=8)
    ax.set_ylabel(r"$\Delta\varepsilon^\star$ / interaction")
    ax.set_title("Aggregates with BCa CI95")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "RQ1 blowup-mode sensitivity (Qwen3-0.6B GAP) — relative vs absolute / norm-matched",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Prefer absolute/norm_matched for directional claims. Interaction stays near null; "
        "Assistant Axis flips sign under corrected metrics; relative random−Δ is metric-amplified.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = FIG / "step_5_rq1_blowup_mode_compare.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)

    # ------------------------------------------------------------------
    # 2. Norm-shrinkage magnitude check
    # ------------------------------------------------------------------
    mag = json.loads((ROOT / "data/results/rq1_norm_shrinkage_magnitude_check.json").read_text())
    rows = mag["per_direction"]
    # exclude refusal from main visual or mark it
    labels = [r["direction_id"].replace("_L4", "") for r in rows]
    pred = [r["predicted_shift_abs_minus_rel"] for r in rows]
    obs = [r["observed_shift_abs_minus_rel"] for r in rows]
    groups = [r["group"] for r in rows]
    gcol = {"safety": "#e6550d", "learned": "#3182bd", "random": "#636363"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    ax = axes[0]
    y = np.arange(len(rows))
    ax.barh(y - 0.15, pred, height=0.3, color="#9ecae1", label="predicted (9% formula)")
    ax.barh(y + 0.15, obs, height=0.3, color="#fd8d3c", label="observed (abs−rel)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$\Delta_{abs}-\Delta_{rel}$ shift")
    ax.set_title("Predicted vs observed metric shift")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1]
    for r in rows:
        ax.scatter(
            r["predicted_shift_abs_minus_rel"],
            r["observed_shift_abs_minus_rel"],
            c=gcol[r["group"]],
            s=55,
            zorder=3,
        )
        ax.annotate(
            r["direction_id"].replace("_L4", "").replace("domain_content_", ""),
            (r["predicted_shift_abs_minus_rel"], r["observed_shift_abs_minus_rel"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
        )
    lims = [0, max(max(pred), max(obs)) * 1.15]
    ax.plot(lims, lims, "k--", lw=1.0, label="y = x (perfect match)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("predicted shift")
    ax.set_ylabel("observed shift")
    ax.set_title("9% norm-shrinkage magnitude check")
    ax.legend(
        handles=[
            Patch(color=gcol["safety"], label="safety"),
            Patch(color=gcol["learned"], label="learned"),
            Patch(color=gcol["random"], label="random"),
            plt.Line2D([0], [0], color="k", ls="--", label="y=x"),
        ],
        fontsize=8,
    )
    ax.grid(True, alpha=0.25)
    s = mag["summary"]
    fig.suptitle(
        f"Does 9% tool-norm shrinkage explain the ~+1.7 uniform shift?  "
        f"(~{100*s['fraction_of_shift_explained']:.0f}% explained; mean |resid|={s['mean_abs_residual_excl_refusal']:.2f})",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        r"Formula: $(\Delta_{abs}-\Delta_{rel})\approx(1-N_{tool}/N_{prose})\cdot\varepsilon^\star_{prose}$ "
        f"with N_t/N_p≈{mag['norm_ratio_tool_over_prose']:.3f}. Random residual opposite sign vs safety/learned.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out = FIG / "step_5_rq1_norm_shrinkage_check.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)

    # ------------------------------------------------------------------
    # 3. Logit-lens layer sweep (§1)
    # ------------------------------------------------------------------
    sweep = json.loads((ROOT / "data/results/logit_lens_layer_sweep_gap.json").read_text())
    layers = sweep["layers"]
    xs = [r["layer"] for r in layers]
    deltas = [r["mean_delta_tool_minus_prose"] for r in layers]
    p_bh = [r["p_bh_fdr"] for r in layers]
    p_raw = [r["p_value_one_sided_raw"] for r in layers]
    preg = sweep.get("preregistered_final_layer")

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.axhline(0, color="black", lw=0.8)
    colors = ["#2ca25f" if (r["direction_ok"] and r["p_bh_fdr"] < 0.05) else "#bdbdbd" for r in layers]
    ax.scatter(xs, deltas, c=colors, s=28, zorder=3)
    ax.plot(xs, deltas, color="#636363", lw=1.0, alpha=0.7)
    if preg is not None:
        ax.axvline(preg, color="#e34a33", ls="--", lw=1.2, label=f"preregistered L{preg}")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta$ entropy (tool − prose)")
    ax.set_title("§1 logit-lens mode gate — GAP layer sweep (Qwen3-0.6B)")
    # annotate pass status
    status = "NOT VALIDATED" if not sweep.get("preregistered_gate_pass") else "PASS"
    ax.text(
        0.99,
        0.02,
        f"prereg gate: {status}\nBH-FDR any-layer: {sweep.get('corrected_any_layer_pass_bh_fdr')}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = FIG / "step_1_logit_lens_layer_sweep.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)

    # ------------------------------------------------------------------
    # 4. Direction stability vs layer (§3)
    # ------------------------------------------------------------------
    stab = json.loads((ROOT / "data/results/direction_stability_pilot.json").read_text())
    # refusal layer cosines; also try other dirs if present
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    thr = stab["refusal_stability"].get("threshold", 0.7)
    ax.axhline(thr, color="#666666", ls=":", lw=1.0, label=f"threshold {thr}")

    def plot_cos(name, layer_cosines, color):
        if isinstance(layer_cosines, dict):
            xs = sorted(int(k) for k in layer_cosines)
            ys = [layer_cosines[str(k)] if str(k) in layer_cosines else layer_cosines[k] for k in xs]
        else:
            # list of {layer, cosine} or pairs
            xs, ys = [], []
            for item in layer_cosines:
                if isinstance(item, dict):
                    xs.append(int(item["layer"]))
                    ys.append(float(item.get("cosine", item.get("split_half_cosine", np.nan))))
                else:
                    continue
        ax.plot(xs, ys, marker="o", color=color, lw=1.8, label=name)

    plot_cos("refusal (prose↔tool)", stab["refusal_stability"]["layer_cosines"], "#e6550d")

    # assistant / harmlessness if in artifacts or same file
    for key, label, color in [
        ("assistant_axis_stability", "Assistant Axis", "#3182bd"),
        ("harmlessness_stability", "harmlessness", "#31a354"),
    ]:
        if key in stab and isinstance(stab[key], dict) and "layer_cosines" in stab[key]:
            plot_cos(label, stab[key]["layer_cosines"], color)

    # band highlight if present
    band = stab["refusal_stability"].get("band_layers")
    if band:
        ax.axvspan(min(band), max(band), color="#fee6ce", alpha=0.35, label="mid/late band")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine (prose-extracted vs tool-extracted)")
    ax.set_title("§3 direction stability across layers (Qwen3-0.6B)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    passed = stab["refusal_stability"].get("passed")
    ax.text(
        0.99,
        0.02,
        f"refusal stability gate: {'PASS' if passed else 'FAIL / blocked'}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#a50f15" if not passed else "#006d2c",
    )
    fig.tight_layout()
    out = FIG / "step_3_direction_stability_layers.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)

    # ------------------------------------------------------------------
    # 5. Control split-half L4 (§4)
    # ------------------------------------------------------------------
    ctrl = json.loads((ROOT / "data/results/control_directions_l4.json").read_text())
    learned = ctrl["learned_controls"]
    randoms = ctrl.get("random_controls") or []
    # random may be summary only
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    names, cos, lo, hi, passed_flags = [], [], [], [], []
    for d in learned:
        L = d["layers"][0]
        names.append(d["direction_id"])
        cos.append(L["split_half_cosine"])
        a, b = L["bca_ci95"]
        lo.append(L["split_half_cosine"] - a)
        hi.append(b - L["split_half_cosine"])
        passed_flags.append(bool(L.get("ci_lower_meets_threshold")))

    # if random controls have per-dir structure
    if randoms and isinstance(randoms, list) and randoms and "layers" in randoms[0]:
        for d in randoms[:4]:
            L = d["layers"][0]
            names.append(d["direction_id"])
            cos.append(L["split_half_cosine"])
            a, b = L["bca_ci95"]
            lo.append(L["split_half_cosine"] - a)
            hi.append(b - L["split_half_cosine"])
            passed_flags.append(bool(L.get("ci_lower_meets_threshold")))

    x = np.arange(len(names))
    colors = ["#2ca25f" if p else "#e34a33" for p in passed_flags]
    ax.bar(x, cos, color=colors, yerr=np.vstack([lo, hi]), capsize=3, alpha=0.9)
    ax.axhline(0.7, color="#666666", ls="--", lw=1.2, label="gate 0.70")

    # Overlay writing v2 from watch-list close-out (canonical PASS)
    wl = json.loads((ROOT / "data/results/rq1_06b_watchlist.json").read_text())
    wv2 = wl.get("writing_control_l4", {}).get("v2")
    if wv2 and "domain_content_writing" in names:
        wi = names.index("domain_content_writing")
        # ghost v1 already plotted; annotate v2 as the closed result
        v2_cos = wv2["split_half_cosine"]
        v2_lo, v2_hi = wv2["bca_ci95"]
        ax.errorbar(
            [wi + 0.18],
            [v2_cos],
            yerr=[[v2_cos - v2_lo], [v2_hi - v2_cos]],
            fmt="D",
            color="#006d2c",
            ms=7,
            capsize=3,
            label="writing v2 (PASS)",
            zorder=5,
        )
        ax.annotate(
            "v1 grazed\nv2 PASS",
            xy=(wi + 0.18, v2_cos),
            xytext=(wi + 0.55, 0.55),
            fontsize=7,
            color="#006d2c",
            arrowprops=dict(arrowstyle="->", color="#006d2c", lw=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("split-half cosine")
    ax.set_ylim(0, 1.05)
    ax.set_title("§4 control direction stability at L4 (BCa CI95)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG / "step_4_control_split_half_l4.png"
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
