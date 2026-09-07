#!/usr/bin/env python3
"""§1 multi-layer logit-lens sweep with multiple-comparison correction.

Does NOT treat any post-hoc single-layer p < α as a gate PASS.
Reports: preregistered final-layer result + per-layer raw p + Bonferroni + BH-FDR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    prev = 1.0
    for i in reversed(range(n)):
        idx = order[i]
        rank = i + 1
        val = min(prev, p_values[idx] * n / rank)
        prev = val
        adj[idx] = min(1.0, val)
    return adj


def bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcripts",
        type=Path,
        default=ROOT / "data" / "transcripts" / "real" / "gap_agentic.jsonl",
    )
    ap.add_argument("--source-filter", default="mind_the_gap_tool_replay,mind_the_gap_scenarios")
    ap.add_argument("--evaluation-dataset", default="mind_the_gap")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--pair-mode", choices=["auto", "turn", "transcript"], default="transcript")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers to sweep (default: all decoder layers)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "logit_lens_layer_sweep_gap.json",
    )
    args = ap.parse_args()

    from activation_pipeline.analysis.logit_lens import (
        collect_window_entropies,
        evaluate_logit_lens_gate,
    )
    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.hooks import resolve_decoder_layers
    from activation_pipeline.loader import load_model_and_tokenizer

    model_key = args.model or LOCAL_MODEL_KEY
    assert_model_fits_machine(model_key)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    n_layers = len(resolve_decoder_layers(loaded.model))
    layers = args.layers if args.layers is not None else list(range(n_layers))
    final_layer = n_layers - 1

    transcripts = load_transcripts(args.transcripts)
    if args.source_filter.strip():
        keep = {s.strip() for s in args.source_filter.split(",") if s.strip()}
        transcripts = [t for t in transcripts if t.get("source") in keep]
    print(f"sweep model={model_key} n_traj={len(transcripts)} layers={layers}")

    per_layer = []
    raw_ps: list[float] = []
    for layer in layers:
        rows = collect_window_entropies(loaded, transcripts, layer=layer)
        gate = evaluate_logit_lens_gate(
            rows,
            alpha=args.alpha,
            layer=layer,
            pair_mode=args.pair_mode,
            layer_choice="sweep",
            evaluation_dataset=args.evaluation_dataset,
        )
        p = gate.p_value_one_sided
        # For correction, missing p → 1.0
        raw_ps.append(1.0 if p is None else float(p))
        per_layer.append(
            {
                "layer": layer,
                "is_preregistered_final": layer == final_layer,
                "n_pairs": gate.n_pairs,
                "mean_prose_entropy": gate.mean_prose_entropy,
                "mean_tool_entropy": gate.mean_tool_entropy,
                "mean_delta_tool_minus_prose": gate.mean_delta_tool_minus_prose,
                "cohens_d_paired": gate.cohens_d_paired,
                "paired_t": gate.paired_t,
                "p_value_one_sided_raw": p,
                "direction_ok": (
                    gate.mean_delta_tool_minus_prose == gate.mean_delta_tool_minus_prose
                    and gate.mean_delta_tool_minus_prose < 0
                ),
            }
        )
        print(
            f"L{layer:02d} Δ={gate.mean_delta_tool_minus_prose:+.4f} "
            f"d={gate.cohens_d_paired} p_raw={p}"
        )

    bon = bonferroni(raw_ps)
    fdr = bh_fdr(raw_ps)
    for i, row in enumerate(per_layer):
        row["p_bonferroni"] = bon[i]
        row["p_bh_fdr"] = fdr[i]
        row["sig_raw_alpha"] = bool(
            row["direction_ok"]
            and row["p_value_one_sided_raw"] is not None
            and row["p_value_one_sided_raw"] < args.alpha
        )
        row["sig_bonferroni"] = bool(row["direction_ok"] and bon[i] < args.alpha)
        row["sig_bh_fdr"] = bool(row["direction_ok"] and fdr[i] < args.alpha)

    final_row = next(r for r in per_layer if r["layer"] == final_layer)
    any_fdr = any(r["sig_bh_fdr"] for r in per_layer)
    any_bon = any(r["sig_bonferroni"] for r in per_layer)

    # Preregistered gate verdict (final layer only, uncorrected α)
    prereg_pass = bool(
        final_row["direction_ok"]
        and final_row["p_value_one_sided_raw"] is not None
        and final_row["p_value_one_sided_raw"] < args.alpha
    )

    payload = {
        "evaluation_dataset": args.evaluation_dataset,
        "model": model_key,
        "n_transcripts": len(transcripts),
        "pair_mode": args.pair_mode,
        "alpha": args.alpha,
        "n_layers_tested": len(layers),
        "preregistered_final_layer": final_layer,
        "preregistered_final_layer_result": final_row,
        "preregistered_gate_pass": prereg_pass,
        "corrected_any_layer_pass_bonferroni": any_bon,
        "corrected_any_layer_pass_bh_fdr": any_fdr,
        "status": "not_validated",
        "methodological_note": (
            "Post-hoc selection of a single mid-layer that clears uncorrected α "
            "is a multiple-comparisons violation and must NOT be labeled a gate PASS. "
            "This sweep reports raw and corrected p-values. Correction unit (frozen): "
            "Bonferroni + BH-FDR within each dataset across layers; GAP and τ are "
            "separate families. Primary confirmatory gate remains final-layer per "
            "dataset with no multiplicity. Same-layer GAP↔τ replication is a "
            "consistency check, not a joint FDR. Mode distinction remains unvalidated "
            "until 32B close-out under that protocol."
        ),
        "layers": per_layer,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print("---")
    print(f"preregistered final L{final_layer}: pass={prereg_pass} p_raw={final_row['p_value_one_sided_raw']}")
    print(f"any layer Bonferroni α={args.alpha}: {any_bon}")
    print(f"any layer BH-FDR α={args.alpha}: {any_fdr}")
    print(f"STATUS: not_validated — wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
