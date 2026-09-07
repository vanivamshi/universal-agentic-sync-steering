#!/usr/bin/env python3
"""Phase-0 gap close-out on Qwen3-0.6B with progress logging."""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOG = ROOT / "logs" / "gap_closeout_pipeline.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str) -> int:
    log(f"START {label}: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    log(f"END {label} exit={r.returncode} elapsed={time.time()-t0:.1f}s")
    return r.returncode


def cache_gap() -> None:
    import torch
    from activation_pipeline.cache_io import ActivationCache
    from activation_pipeline.caching import cache_transcript_windows, load_transcripts
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    log("CACHE GAP — loading model")
    t0 = time.time()
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    log(f"model loaded in {time.time()-t0:.1f}s")
    transcripts = load_transcripts(ROOT / "data/transcripts/real/gap_agentic.jsonl")
    layers = [4, 22]
    out_records = []
    part = None
    for i, tr in enumerate(transcripts):
        log(f"cache {i+1}/{len(transcripts)} {tr.get('transcript_id')}")
        part = cache_transcript_windows(
            loaded,
            [tr],
            layer_indices=layers,
            domains=["coding", "therapy", "writing"],
            require_both_modes=False,
            cast_dtype=torch.float32,
        )
        out_records.extend(part.records)
        log(f"  +{len(part.records)} (total {len(out_records)})")
    assert part is not None
    cache = ActivationCache(
        format_version=part.format_version,
        model_key=part.model_key,
        hf_id=part.hf_id,
        layer_indices=part.layer_indices,
        meta={**part.meta, "n_transcripts": len(transcripts)},
        records=out_records,
    )
    out = ROOT / "data/activations/real_gap.json"
    cache.save(out)
    n_prose = sum(1 for r in cache.records if r.window_kind == "prose")
    n_tool = sum(1 for r in cache.records if r.window_kind == "tool_call")
    log(f"saved {out} records={len(cache.records)} prose={n_prose} tool={n_tool}")
    del loaded
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> int:
    LOG.write_text("")
    log("=== gap close-out pipeline start ===")
    py = sys.executable
    cache_gap()

    codes = []
    codes.append(
        run(
            [
                py,
                "scripts/run_subspace_compare.py",
                "--cache",
                "data/activations/real_gap.json",
                "--evaluation-dataset",
                "mind_the_gap",
                "--layers",
                "4",
                "22",
                "--by-domain",
                "--out",
                "data/results/subspace_gap.json",
            ],
            "subspace",
        )
    )
    codes.append(
        run(
            [
                py,
                "scripts/run_control_holdout_recheck.py",
                "--n-boot",
                "200",
                "--out",
                "data/results/control_holdout_recheck.json",
            ],
            "holdout",
        )
    )
    codes.append(
        run(
            [
                py,
                "scripts/run_logit_lens_layer_sweep.py",
                "--transcripts",
                "data/transcripts/real/gap_agentic.jsonl",
                "--source-filter",
                "mind_the_gap_tool_replay,mind_the_gap_scenarios",
                "--evaluation-dataset",
                "mind_the_gap",
                "--pair-mode",
                "transcript",
                "--alpha",
                "0.05",
                "--out",
                "data/results/logit_lens_layer_sweep_gap.json",
            ],
            "logit_gap",
        )
    )
    codes.append(
        run(
            [
                py,
                "scripts/run_logit_lens_layer_sweep.py",
                "--transcripts",
                "data/transcripts/real/tau_bench_short.jsonl",
                "--source-filter",
                "tau_bench_historical",
                "--evaluation-dataset",
                "tau_bench",
                "--pair-mode",
                "transcript",
                "--alpha",
                "0.05",
                "--out",
                "data/results/logit_lens_layer_sweep_tau.json",
            ],
            "logit_tau",
        )
    )
    codes.append(
        run(
            [
                py,
                "scripts/run_refusal_stability_bootstrap.py",
                "--n-boot",
                "400",
                "--layers",
                "4",
                "14",
                "22",
                "--out",
                "data/results/refusal_stability_layer_matched.json",
            ],
            "refusal",
        )
    )
    bad = sum(1 for c in codes if c != 0)
    log(f"=== done failures={bad}/{len(codes)} ===")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
