#!/usr/bin/env python3
"""Cache residual-stream means for tagged prose/tool windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcripts", type=Path, default=ROOT / "data" / "transcripts")
    from activation_pipeline.device import LOCAL_MODEL_KEY

    ap.add_argument("--model", default=LOCAL_MODEL_KEY, help="Mac default: qwen3-0.6b (MPS)")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "activations" / "smoke_coding.json",
    )
    ap.add_argument("--domain", action="append", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument(
        "--device",
        default=None,
        help="cpu|mps|cuda (default: mps on MacBook)",
    )
    ap.add_argument(
        "--allow-single-mode",
        action="store_true",
        help="Cache turns with only prose or only tool (needed for agentic tool-first turns)",
    )
    args = ap.parse_args()

    import torch

    from activation_pipeline.caching import cache_transcript_windows, load_transcripts
    from activation_pipeline.device import assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    domains = args.domain or ["coding"]
    assert_model_fits_machine(args.model)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        args.model,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    transcripts = load_transcripts(args.transcripts)
    cache = cache_transcript_windows(
        loaded,
        transcripts,
        layer_indices=args.layers,
        domains=domains,
        require_both_modes=not args.allow_single_mode,
        cast_dtype=getattr(torch, args.dtype),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)

    n_prose = sum(1 for r in cache.records if r.window_kind == "prose")
    n_tool = sum(1 for r in cache.records if r.window_kind == "tool_call")
    print(
        f"saved {args.out} (+ .pt)  records={len(cache.records)} "
        f"prose={n_prose} tool={n_tool} layers={cache.layer_indices} device={device_map}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
