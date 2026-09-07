#!/usr/bin/env python3
"""Tan-style directional agreement diagnostic (extract quality, pre-steer).

Paper framing: Tan et al. — unreliable steering vectors when per-example
activation differences disagree. Complementary to Engels geometry / J1–J4.

Uses cached GAP L4 means only (no model load / no new generation).
Targets:
  1) surface_gap mean-diff (built here from cache)
  2) Track A persona PCs [5,12,4,7,24] vs those same per-example deltas
  3) performance_meandiff stored E2 (reference HIGH from prior extract)

Anti-steerable *steering* effects need per-trial steer logs (not in
jailbreak_surface_steering.json) — report extract-time anti-alignment only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations" / "gap_deception.json"
LABELS = ROOT / "data" / "labels" / "gap_deception_eliciting.jsonl"
PCS = ROOT / "data" / "directions" / "persona_pca_prose_L4.jsonl"
PERF = ROOT / "data" / "results" / "performance_meandiff_screen.json"
STEER = ROOT / "data" / "results" / "jailbreak_surface_steering.json"
OUT = ROOT / "data" / "results" / "tan_directional_agreement.json"
MD = ROOT / "data" / "results" / "tan_directional_agreement.md"

LAYER = "4"
TRACK_A_PCS = [5, 12, 4, 7, 24]
SEED = 20260813

# Locked bars (Tan-inspired; Mac-scale)
PAIRWISE_HIGH = 0.50
PAIRWISE_LOW = 0.30
LOO_HIGH = 0.50
ANTI_FRAC_HIGH = 0.30  # ≥30% opposite-aligned deltas ⇒ extract risk


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _pairwise_mean_cos(vecs: list[np.ndarray]) -> float:
    if len(vecs) < 2:
        return float("nan")
    u = [_unit(v) for v in vecs]
    vals = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            vals.append(float(np.dot(u[i], u[j])))
    return float(np.mean(vals)) if vals else float("nan")


def _mean_cos_to(ref: np.ndarray, vecs: list[np.ndarray]) -> float:
    if not vecs:
        return float("nan")
    r = _unit(ref)
    return float(np.mean([float(np.dot(r, _unit(v))) for v in vecs]))


def _anti_frac(ref: np.ndarray, vecs: list[np.ndarray]) -> float:
    """Fraction of per-example deltas with opposite sign to mean alignment."""
    if not vecs:
        return float("nan")
    r = _unit(ref)
    cos = np.array([float(np.dot(r, _unit(v))) for v in vecs])
    mean_c = float(cos.mean())
    if abs(mean_c) < 1e-8:
        return float(np.mean(np.abs(cos) < 1e-8))
    return float(np.mean(cos * mean_c < 0))


def _loo_meandiff_pairwise(
    pos: list[np.ndarray], neg: list[np.ndarray]
) -> dict[str, float]:
    """Leave-one-out mean-diff stability (same spirit as performance E2)."""
    if len(pos) < 2 or len(neg) < 1:
        return {
            "loo_pairwise_cos": float("nan"),
            "loo_vs_full_mean": float("nan"),
            "n_loo": 0,
        }
    mu_n = np.mean(np.stack(neg, 0), 0)
    full = _unit(np.mean(np.stack(pos, 0), 0) - mu_n)
    loo_dirs: list[np.ndarray] = []
    # LOO over positives (dominant instability source when n_pos small)
    for i in range(len(pos)):
        rest = [p for j, p in enumerate(pos) if j != i]
        if not rest:
            continue
        d = _unit(np.mean(np.stack(rest, 0), 0) - mu_n)
        loo_dirs.append(d)
    # Also LOO over negatives if ≥2
    if len(neg) >= 2:
        mu_p = np.mean(np.stack(pos, 0), 0)
        for i in range(len(neg)):
            rest = [n for j, n in enumerate(neg) if j != i]
            d = _unit(mu_p - np.mean(np.stack(rest, 0), 0))
            loo_dirs.append(d)
    vs_full = [float(np.dot(v, full)) for v in loo_dirs]
    return {
        "loo_pairwise_cos": _pairwise_mean_cos(loo_dirs),
        "loo_vs_full_mean": float(np.mean(vs_full)) if vs_full else float("nan"),
        "n_loo": len(loo_dirs),
    }


def _tag_agreement(
    pairwise: float, loo: float, anti: float
) -> str:
    if pairwise != pairwise or loo != loo:
        return "UNDERPOWERED"
    high = pairwise >= PAIRWISE_HIGH and loo >= LOO_HIGH
    low = pairwise < PAIRWISE_LOW or loo < LOO_HIGH
    anti_bad = anti == anti and anti >= ANTI_FRAC_HIGH
    if high and not anti_bad:
        return "HIGH_AGREEMENT"
    if high and anti_bad:
        return "HIGH_BUT_ANTI_FRAC"
    if low:
        return "LOW_AGREEMENT"
    return "MIXED_AGREEMENT"


def _load_transcript_means() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    act = json.loads(ACT.read_text())
    labs = {
        json.loads(l)["transcript_id"]: json.loads(l)
        for l in LABELS.read_text().splitlines()
        if l.strip()
    }
    buckets: dict[str, list[np.ndarray]] = {}
    for rec in act["records"]:
        tid = rec["transcript_id"]
        lm = (rec.get("layer_means") or {}).get(LAYER)
        if lm is None:
            continue
        buckets.setdefault(tid, []).append(np.asarray(lm, dtype=np.float64))
    H = {t: np.mean(vs, 0) for t, vs in buckets.items()}
    return H, labs


def _load_pcs(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "persona_pc":
            continue
        idx = int(r["meta"]["pc_index"])
        v = np.asarray(r["vector"], dtype=np.float64)
        out[idx] = _unit(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--md-out", type=Path, default=MD)
    args = ap.parse_args()

    H, labs = _load_transcript_means()
    pcs = _load_pcs(PCS)

    pos_ids = sorted(
        t for t, h in H.items() if t in labs and float(labs[t].get("surface_gap") or 0) > 0.5
    )
    neg_ids = sorted(
        t for t, h in H.items() if t in labs and float(labs[t].get("surface_gap") or 0) <= 0.5
    )
    pos = [H[t] for t in pos_ids]
    neg = [H[t] for t in neg_ids]
    print(
        f"=== Tan directional agreement L{LAYER} "
        f"n_pos={len(pos)} n_neg={len(neg)} n_H={len(H)} ===",
        flush=True,
    )

    if len(pos) < 2 or len(neg) < 1:
        payload = {
            "stage": "TAN_DIRECTIONAL_AGREEMENT",
            "decision": "UNDERPOWERED",
            "note": "Need ≥2 surface_gap pos and ≥1 neg on cached means.",
            "n_pos": len(pos),
            "n_neg": len(neg),
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        args.md_out.write_text("# Tan agreement\n\nUNDERPOWERED\n")
        return 0

    mu_neg = np.mean(np.stack(neg, 0), 0)
    deltas = [p - mu_neg for p in pos]  # per-positive contribution to mean-diff
    d_full = _unit(np.mean(np.stack(pos, 0), 0) - mu_neg)
    pair_d = _pairwise_mean_cos(deltas)
    mean_to_d = _mean_cos_to(d_full, deltas)
    anti_d = _anti_frac(d_full, deltas)
    loo = _loo_meandiff_pairwise(pos, neg)
    tag_md = _tag_agreement(pair_d, loo["loo_pairwise_cos"], anti_d)

    surface_row = {
        "name": "surface_gap_meandiff",
        "kind": "mean_difference",
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "delta_pairwise_cos": pair_d,
        "delta_mean_cos_to_direction": mean_to_d,
        "anti_aligned_frac": anti_d,
        **loo,
        "agreement_tag": tag_md,
        "steer_history": "not_extracted_as_primary_Track_A_lever",
    }
    print(
        f"surface_gap meandiff: pair={pair_d:.3f} loo={loo['loo_pairwise_cos']:.3f} "
        f"anti={anti_d:.3f} → {tag_md}",
        flush=True,
    )

    # Track A PCs: does each PC agree with the outcome deltas?
    pc_rows: list[dict[str, Any]] = []
    steer = json.loads(STEER.read_text()) if STEER.exists() else {}
    by_pc = steer.get("by_pc") or {}
    for idx in TRACK_A_PCS:
        if idx not in pcs:
            continue
        v = pcs[idx]
        # Sign PC so mean cos to deltas ≥ 0 (fair agreement, not arbitrary PC sign)
        raw_mean = _mean_cos_to(v, deltas)
        if raw_mean < 0:
            v = -v
            signed = -1
        else:
            signed = 1
        pair = pair_d  # same delta cloud
        mean_c = _mean_cos_to(v, deltas)
        anti = _anti_frac(v, deltas)
        # LOO for "direction = PC" is N/A; use mean_c as alignment strength
        # Tag: treat mean_c as stand-in for loo when comparing extract quality of using PC as lever
        tag = _tag_agreement(pair, mean_c, anti)
        # Override: if mean alignment to PC is weak, LOW regardless of delta pairwise
        if mean_c == mean_c and mean_c < PAIRWISE_LOW:
            tag = "LOW_AGREEMENT"
        elif mean_c == mean_c and mean_c < PAIRWISE_HIGH and tag == "HIGH_AGREEMENT":
            tag = "MIXED_AGREEMENT"

        hist = by_pc.get(f"pc{idx}") or {}
        pc_rows.append(
            {
                "name": f"persona_pc{idx}",
                "kind": "persona_pc",
                "pc_index": idx,
                "sign_flipped_to_align": signed < 0,
                "delta_pairwise_cos": pair,
                "delta_mean_cos_to_direction": mean_c,
                "anti_aligned_frac": anti,
                "agreement_tag": tag,
                "track_a_claim_tier": hist.get("overall_claim_tier"),
                "track_a_J2_class": hist.get("J2_class"),
            }
        )
        print(
            f"pc{idx}: mean_cos_to_δ={mean_c:.3f} anti={anti:.3f} → {tag} "
            f"(Track A {hist.get('overall_claim_tier')})",
            flush=True,
        )

    # Performance mean-diff reference (prior extract; already E2-OK)
    perf_row = None
    if PERF.exists():
        perf = json.loads(PERF.read_text())
        e2 = float(perf.get("E2_loo_pairwise_cos") or float("nan"))
        perf_row = {
            "name": "performance_meandiff_L4",
            "kind": "mean_difference_reference",
            "source": str(PERF),
            "n_pos": perf.get("n_pos"),
            "n_neg": perf.get("n_neg"),
            "floor_of_power": perf.get("floor_of_power"),
            "E2": perf.get("E2"),
            "loo_pairwise_cos": e2,
            "loo_vs_full_mean": perf.get("E2_loo_vs_full_mean"),
            "P1_status": (perf.get("P1") or {}).get("P1_status"),
            "agreement_tag": (
                "HIGH_AGREEMENT"
                if e2 == e2 and e2 >= LOO_HIGH
                else "LOW_AGREEMENT"
                if e2 == e2
                else "UNDERPOWERED"
            ),
            "note": "Prior extract already passed E2; held as HIGH reference — not the Track A failure.",
        }

    # Decision for next steps
    low_pcs = [r for r in pc_rows if r["agreement_tag"] in ("LOW_AGREEMENT", "HIGH_BUT_ANTI_FRAC")]
    high_pcs = [r for r in pc_rows if r["agreement_tag"] == "HIGH_AGREEMENT"]
    if tag_md == "LOW_AGREEMENT":
        decision = "EXTRACT_UNRELIABLE_SURFACE"
        note = (
            "surface_gap mean-diff deltas disagree — Tan predicts unreliable steer; "
            "do not invest in more α grids on this extract. Prefer RQ1 privilege "
            "or rebuild contrasts with higher agreement."
        )
    elif low_pcs and not high_pcs:
        decision = "TRACK_A_PCS_MISALIGNED"
        note = (
            "Per-example surface_gap deltas do not agree with Track A PCs "
            f"{[r['pc_index'] for r in low_pcs]} — explains NULL steers if "
            "deltas themselves are coherent, or compounds failure if surface "
            f"extract is also weak (surface={tag_md})."
        )
    elif high_pcs and tag_md in ("HIGH_AGREEMENT", "MIXED_AGREEMENT", "HIGH_BUT_ANTI_FRAC"):
        decision = "AGREEMENT_OK_FAILURE_DOWNSTREAM"
        note = (
            "Extract/PC alignment is not the bottleneck — failure is downstream "
            "(privilege / geometry / outcome). Go to RQ1 safety-vs-control privilege."
        )
    else:
        decision = "MIXED"
        note = "See per-direction tags; no single extract-quality verdict."

    payload = {
        "stage": "TAN_DIRECTIONAL_AGREEMENT",
        "paper": "Tan et al. Understanding (Un)Reliability of Steering Vectors",
        "layer": int(LAYER),
        "seed": SEED,
        "bars": {
            "pairwise_high": PAIRWISE_HIGH,
            "pairwise_low": PAIRWISE_LOW,
            "loo_high": LOO_HIGH,
            "anti_frac_high": ANTI_FRAC_HIGH,
        },
        "n_pos": len(pos),
        "n_neg": len(neg),
        "decision": decision,
        "note": note,
        "surface_gap_meandiff": surface_row,
        "track_a_pcs": pc_rows,
        "performance_meandiff_reference": perf_row,
        "anti_steerable_steer_effects": {
            "status": "NOT_AVAILABLE",
            "reason": (
                "jailbreak_surface_steering.json stores aggregate rates only; "
                "per-trial steered outcomes required for Tan anti-steerable rate."
            ),
        },
        "causal_claim": False,
        "next": (
            "RQ1 privilege (safety vs controls)"
            if decision == "AGREEMENT_OK_FAILURE_DOWNSTREAM"
            else "Do not re-α-grid low-agreement levers; RQ1 or rebuild contrasts"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Tan-style directional agreement (extract quality)",
        "",
        f"L{LAYER}. surface_gap pos={len(pos)} neg={len(neg)}. Cached means only.",
        "",
        f"## Decision: `{decision}`",
        note,
        "",
        "### surface_gap mean-diff",
        f"- δ pairwise cos={pair_d:.3f}",
        f"- LOO pairwise cos={loo['loo_pairwise_cos']:.3f} (vs full={loo['loo_vs_full_mean']:.3f})",
        f"- mean cos(δ, d)={mean_to_d:.3f}; anti-aligned frac={anti_d:.3f}",
        f"- tag=`{tag_md}`",
        "",
        "### Track A persona PCs vs same δ cloud",
        "| PC | mean cos(δ,PC) | anti frac | tag | Track A tier |",
        "|---:|---:|---:|---|---|",
    ]
    for r in pc_rows:
        lines.append(
            f"| {r['pc_index']} | {r['delta_mean_cos_to_direction']:.3f} | "
            f"{r['anti_aligned_frac']:.3f} | `{r['agreement_tag']}` | "
            f"{r.get('track_a_claim_tier')} |"
        )
    if perf_row:
        lines += [
            "",
            "### Reference: performance mean-diff (prior)",
            f"- E2 LOO pairwise={perf_row.get('loo_pairwise_cos')} → `{perf_row.get('agreement_tag')}`",
            f"- P1={perf_row.get('P1_status')} (in-sample only; holdout not licensed)",
        ]
    lines += [
        "",
        "Bars: HIGH if δ-pairwise≥0.50 and LOO/align≥0.50 and anti-frac<0.30.",
        "Anti-steerable *steer* rate: not available from aggregate Track A JSON.",
        f"Artifact: `{args.out}`",
        "",
        "**Not a causal claim.**",
    ]
    args.md_out.write_text("\n".join(lines) + "\n")
    print(f"decision={decision}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
