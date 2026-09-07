#!/usr/bin/env python3
"""Smoke-test window tagging, cache I/O, sensitivity + subspace tooling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smoke_toy_analysis() -> None:
    import torch

    from activation_pipeline.analysis import (
        LayerPerturbHooks,
        compare_prose_tool_subspaces,
        directional_sensitivity,
        plateau_depth,
        top_k_sensitive_directions,
    )
    from activation_pipeline.toy_model import ToyCausalLM

    model = ToyCausalLM(n_layers=8, hidden=32)
    model.eval()
    ids = torch.randint(1, 100, (1, 20))
    attn = torch.ones_like(ids)
    token_idx = list(range(5, 12))
    direction = torch.randn(32)

    hooks = LayerPerturbHooks(
        model,
        layer_k=1,
        direction=direction,
        token_indices=token_idx,
        read_layers=[1, 6],
    )

    def forward_clean():
        return hooks.run(ids, attn, epsilon=0.0)

    def forward_pert(eps: float):
        return hooks.run(ids, attn, epsilon=eps)

    def pool(t: torch.Tensor) -> torch.Tensor:
        return t[0, token_idx, :].float().mean(dim=0)

    sens = directional_sensitivity(
        forward_clean,
        forward_pert,
        layer_k=1,
        layer_L=6,
        threshold=0.1,
        direction_id="rand",
        pool=pool,
        eps_hi=20.0,
    )
    print(
        f"toy_sensitivity: eps*={sens.epsilon_star:.4f} "
        f"blowup={sens.blowup:.4f} converged={sens.converged}"
    )

    def blow_fn(eps: float) -> float:
        clean = pool(forward_clean()[6])
        pert = pool(forward_pert(eps)[6])
        return float(torch.linalg.norm(pert - clean) / (torch.linalg.norm(clean) + 1e-8))

    depth = plateau_depth(blow_fn, threshold=0.1)
    print(f"toy_plateau_depth: {depth}")

    ranked = top_k_sensitive_directions(
        [("a", 0.2), ("b", 1.5), ("c", 0.05)], k=2
    )
    assert ranked[0][0] == "c"
    print(f"toy_topk: {ranked}")

    prose = [torch.randn(32) + torch.tensor([1.0] + [0.0] * 31) for _ in range(5)]
    tool = [torch.randn(32) + torch.tensor([0.0, 1.0] + [0.0] * 30) for _ in range(5)]
    sub = compare_prose_tool_subspaces(prose, tool)
    print(
        f"toy_subspace: cos={sub.mean_cosine:.3f} disparity={sub.procrustes_disparity:.3f}"
    )


def smoke_windows_and_cache(local_files_only: bool) -> None:
    import torch
    from transformers import AutoTokenizer

    from activation_pipeline.analysis import compare_prose_tool_subspaces
    from activation_pipeline.cache_io import ActivationCache
    from activation_pipeline.caching import cache_transcript_windows, load_transcripts
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.registry import get_model_spec
    from activation_pipeline.windows import (
        embed_windows_in_chat,
        sanity_check_tagged,
        tag_transcript_assistant_turns,
    )

    model_key = "qwen3-0.6b"
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map

    model_key = LOCAL_MODEL_KEY
    spec = get_model_spec(model_key)
    tok = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=True, local_files_only=local_files_only
    )
    transcripts = load_transcripts(ROOT / "data" / "transcripts" / "coding.jsonl")
    tr = transcripts[0]
    tagged = tag_transcript_assistant_turns(tr, tok)[0]
    embedded = embed_windows_in_chat(tok, tr["messages"], tagged.message_index, tagged)
    warns = sanity_check_tagged(embedded)
    print(
        f"tag: {tr['transcript_id']} prose={len(embedded.prose_windows())} "
        f"tool={len(embedded.tool_windows())} warns={warns}"
    )
    assert embedded.prose_windows() and embedded.tool_windows()
    assert not warns

    device_map = resolve_device_map(None)
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype="float32",
        local_files_only=local_files_only,
    )
    k, L = spec.default_perturb_layer, spec.default_measure_layer
    cache = cache_transcript_windows(
        loaded,
        transcripts[:2],
        layer_indices=[k, L],
        domains=["coding"],
        cast_dtype=torch.float32,
    )
    out = ROOT / "data" / "activations" / "smoke_coding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.save(out)
    reloaded = ActivationCache.load(out)
    assert len(reloaded.records) == len(cache.records)
    print(f"cache: records={len(reloaded.records)} layers={reloaded.layer_indices}")

    prose = [
        r.mean_tensor(k)
        for r in reloaded.records
        if r.window_kind == "prose" and str(k) in r.layer_means
    ]
    tool = [
        r.mean_tensor(k)
        for r in reloaded.records
        if r.window_kind == "tool_call" and str(k) in r.layer_means
    ]
    if prose and tool:
        sub = compare_prose_tool_subspaces(prose, tool)
        print(
            f"cache_subspace(layer{k}): cos={sub.mean_cosine:.3f} "
            f"disparity={sub.procrustes_disparity:.3f} "
            f"n_prose={sub.n_prose} n_tool={sub.n_tool}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    args = ap.parse_args()

    smoke_toy_analysis()
    if not args.skip_hf:
        smoke_windows_and_cache(local_files_only=args.local_files_only)
    print("smoke_0.4_0.6: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
