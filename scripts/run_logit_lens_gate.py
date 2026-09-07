#!/usr/bin/env python3
"""§1 logit-lens mode gate: tool windows should have lower vocab entropy than prose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcripts",
        type=Path,
        default=ROOT / "data" / "transcripts" / "agentic" / "coding.jsonl",
    )
    ap.add_argument("--model", default=None, help="Default: qwen3-0.6b on MacBook (MPS)")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--pair-mode",
        choices=["auto", "turn", "transcript"],
        default="auto",
        help="How to pair prose vs tool (auto: turn if ≥3 else transcript)",
    )
    ap.add_argument("--device", default=None, help="Default: mps on MacBook")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "logit_lens_gate.json",
    )
    ap.add_argument(
        "--evaluation-dataset",
        default="",
        help="Label written into results JSON (e.g. mind_the_gap, tau_bench)",
    )
    ap.add_argument(
        "--layer-choice",
        default="",
        help="preregistered_final | exploratory_mid (auto if empty)",
    )
    ap.add_argument(
        "--source-filter",
        default="",
        help="Comma-separated transcript sources to keep (empty = preferred real set)",
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
    print(f"gate model={model_key} device={device_map}")
    layer = args.layer
    n_layers = len(resolve_decoder_layers(loaded.model))
    if layer is None:
        layer = n_layers - 1
    layer_choice = args.layer_choice or (
        "preregistered_final" if layer == n_layers - 1 else "exploratory_mid"
    )
    transcripts = load_transcripts(args.transcripts)
    if args.source_filter.strip():
        keep = {s.strip() for s in args.source_filter.split(",") if s.strip()}
        transcripts = [t for t in transcripts if t.get("source") in keep]
        print(f"using {len(transcripts)} transcripts with source in {sorted(keep)}")
    else:
        preferred = {
            "mind_the_gap_tool_replay",
            "mind_the_gap_scenarios",
            "tau_bench_historical",
            "agentic_loop",
        }
        real = [t for t in transcripts if t.get("source") in preferred]
        if real:
            transcripts = real
            print(f"using {len(transcripts)} real/agentic transcripts")
        else:
            print(f"using {len(transcripts)} transcripts from {args.transcripts}")

    eval_name = args.evaluation_dataset
    if not eval_name:
        sources = {t.get("source") for t in transcripts}
        if sources <= {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}:
            eval_name = "mind_the_gap"
        elif sources <= {"tau_bench_historical"}:
            eval_name = "tau_bench"
        else:
            eval_name = "mixed"

    rows = collect_window_entropies(loaded, transcripts, layer=layer)
    gate = evaluate_logit_lens_gate(
        rows,
        alpha=args.alpha,
        layer=layer,
        pair_mode=args.pair_mode,
        layer_choice=layer_choice,
        evaluation_dataset=eval_name,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = gate.to_dict()
    payload["reporting"] = {
        "emphasis": (
            "§1 validates window definitions (mode distinction), not effect magnitude. "
            "Do not compare Δ to synthetic/smoke runs. Report Cohen's d; treat p as secondary."
        ),
        "layer_note": (
            "Preregistration did not freeze a logit-lens layer. "
            "Default on this machine is final residual layer; exploratory mid-layer "
            "is allowed when final fails to separate modes on the smoke model, and "
            "must be re-checked on Qwen3-32B for primary claims."
            if layer_choice == "exploratory_mid"
            else "Used final residual layer (default)."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2))

    print(
        f"dataset={gate.evaluation_dataset} layer={gate.layer} ({gate.layer_choice}) "
        f"pairs={gate.n_pairs} "
        f"prose_H={gate.mean_prose_entropy:.4f} tool_H={gate.mean_tool_entropy:.4f} "
        f"delta(tool-prose)={gate.mean_delta_tool_minus_prose:.4f}"
    )
    if gate.cohens_d_paired is not None:
        print(f"cohens_d_paired={gate.cohens_d_paired:.4f}")
    if gate.paired_t is not None and gate.p_value_one_sided is not None:
        print(f"paired_t={gate.paired_t:.4f} p_one_sided={gate.p_value_one_sided:.4g}")
    print(f"{'PASS' if gate.passed else 'FAIL'}: {gate.notes}")
    print(f"wrote {args.out}")
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
