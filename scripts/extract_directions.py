#!/usr/bin/env python3
"""§3 Extract safety directions (mean-diff) + refusal stability gate (pilot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=[4, 14, 22],
        help="Layers for mean-diff extraction (0.6B defaults: k=4, mid=14, L=22)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Refusal stability cosine threshold (preregistered)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "directions",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data" / "results" / "direction_stability_pilot.json",
    )
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.directions import (
        extract_mean_diff_from_pairs,
        extract_tool_mean_diff,
        save_directions,
        stability_gate,
    )
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
    print(f"directions model={model_key} device={device_map} layers={args.layers}")

    pairs_dir = ROOT / "data" / "contrast_pairs"
    refusal_pairs = _load_jsonl(pairs_dir / "refusal_pilot.jsonl")
    assistant_pairs = _load_jsonl(pairs_dir / "assistant_axis_pilot.jsonl")
    harm_pairs = _load_jsonl(pairs_dir / "harmlessness_pilot.jsonl")
    refusal_tool = _load_jsonl(pairs_dir / "refusal_tool_pilot.jsonl")

    refusal_prose = extract_mean_diff_from_pairs(
        loaded,
        refusal_pairs,
        layers=args.layers,
        direction_id="refusal",
        kind="refusal",
        mode="prose",
    )
    print(f"extracted refusal prose: {len(refusal_prose)} layer vectors")

    refusal_tool_dirs = extract_tool_mean_diff(
        loaded,
        refusal_tool,
        layers=args.layers,
        direction_id="refusal",
        kind="refusal",
    )
    print(f"extracted refusal tool: {len(refusal_tool_dirs)} layer vectors")

    assistant_dirs = extract_mean_diff_from_pairs(
        loaded,
        assistant_pairs,
        layers=args.layers,
        direction_id="assistant_axis",
        kind="assistant_axis",
        mode="prose",
        system="Follow the persona instruction in the user message.",
    )
    print(f"extracted assistant_axis prose: {len(assistant_dirs)} layer vectors")

    harm_dirs = extract_mean_diff_from_pairs(
        loaded,
        harm_pairs,
        layers=args.layers,
        direction_id="harmlessness",
        kind="harmlessness",
        mode="prose",
    )
    print(f"extracted harmlessness prose: {len(harm_dirs)} layer vectors")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_directions(args.out_dir / "refusal_prose.jsonl", refusal_prose)
    save_directions(args.out_dir / "refusal_tool.jsonl", refusal_tool_dirs)
    save_directions(args.out_dir / "assistant_axis_prose.jsonl", assistant_dirs)
    save_directions(args.out_dir / "harmlessness_prose.jsonl", harm_dirs)

    gate = stability_gate(
        refusal_prose,
        refusal_tool_dirs,
        kind="refusal",
        threshold=args.threshold,
        band_layers=args.layers,
    )
    print(gate.notes)
    for layer, c in sorted(gate.layer_cosines.items(), key=lambda x: int(x[0])):
        print(f"  refusal stability L{layer}: cos={c:.4f}")

    payload = {
        "model": model_key,
        "pilot": True,
        "note": (
            "Qwen3-0.6B bring-up. Assistant Axis HF vectors are Qwen3-32B-only; "
            "this run uses local mean-diff proxies. Full AdvBench / HF axes on 32B."
        ),
        "refusal_stability": gate.to_dict(),
        "artifacts": {
            "refusal_prose": str(args.out_dir / "refusal_prose.jsonl"),
            "refusal_tool": str(args.out_dir / "refusal_tool.jsonl"),
            "assistant_axis_prose": str(args.out_dir / "assistant_axis_prose.jsonl"),
            "harmlessness_prose": str(args.out_dir / "harmlessness_prose.jsonl"),
            "contrast_pairs": str(pairs_dir),
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.results}")
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
