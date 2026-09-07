#!/usr/bin/env python3
"""§3 same-mode split-half baseline for refusal direction (prose↔prose)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--layers", type=int, nargs="*", default=[4, 14, 22])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data" / "contrast_pairs" / "refusal_pilot.jsonl",
    )
    ap.add_argument(
        "--prose-tool-results",
        type=Path,
        default=ROOT / "data" / "results" / "direction_stability_pilot.json",
        help="Existing prose↔tool result to annotate with split-half baseline",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "refusal_split_half_prose.json",
    )
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.directions import split_half_prose_stability
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
    pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    print(
        f"split-half model={model_key} pairs={len(pairs)} "
        f"splits={args.n_splits} layers={args.layers}"
    )
    result = split_half_prose_stability(
        loaded,
        pairs,
        layers=args.layers,
        kind="refusal",
        n_splits=args.n_splits,
        seed=args.seed,
    )
    print(result.notes)
    for layer, m in sorted(result.layer_mean_cosine.items(), key=lambda x: int(x[0])):
        s = result.layer_std_cosine[layer]
        print(f"  L{layer}: mean_cos={m:.4f} std={s:.4f}")

    prose_tool_mean = None
    if args.prose_tool_results.exists():
        prev = json.loads(args.prose_tool_results.read_text())
        prose_tool_mean = prev.get("refusal_stability", {}).get("mean_cosine")

    interpretation = (
        "If prose↔prose split-half cosine is also low, the prose↔tool FAIL is not "
        "interpretable as a mode-specific effect at this scale — extraction noise / "
        "missing linear refusal direction on 0.6B. Re-run both checks on Qwen3-32B "
        "before any refusal privilege arm."
    )
    if prose_tool_mean is not None:
        interpretation += (
            f" Observed: split-half band mean={result.band_mean_cosine:.3f}; "
            f"prose↔tool band mean={float(prose_tool_mean):.3f}."
        )

    payload = {
        "model": model_key,
        "pilot": True,
        "split_half_prose": result.to_dict(),
        "prose_tool_mean_cosine": prose_tool_mean,
        "interpretation": interpretation,
        "status": "baseline_control",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    # Annotate existing prose↔tool results with the baseline
    if args.prose_tool_results.exists():
        prev = json.loads(args.prose_tool_results.read_text())
        prev["split_half_prose_baseline"] = result.to_dict()
        prev["interpretation"] = interpretation
        prev["status"] = (
            "confounded_until_scale_check"
            if result.band_mean_cosine < 0.70
            else "prose_stable_tool_unstable"
        )
        args.prose_tool_results.write_text(json.dumps(prev, indent=2))
        print(f"annotated {args.prose_tool_results} status={prev['status']}")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
