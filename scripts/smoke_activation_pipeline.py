#!/usr/bin/env python3
"""Smoke-test residual-stream hooks (toy model always; optional HF smoke model)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: repo root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_toy() -> None:
    import torch

    from activation_pipeline.collect import collect_residual_stream
    from activation_pipeline.hooks import ResidualStreamHooks, resolve_decoder_layers
    from activation_pipeline.loader import LoadedModel
    from activation_pipeline.registry import get_model_spec
    from activation_pipeline.toy_model import ToyCausalLM

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=64):
            del truncation, max_length
            ids = []
            for t in texts:
                # deterministic fake tokenization
                row = [((ord(c) % 120) + 1) for c in t[:48]] or [1]
                ids.append(row)
            maxlen = max(len(r) for r in ids)
            input_ids = torch.zeros(len(ids), maxlen, dtype=torch.long)
            attn = torch.zeros(len(ids), maxlen, dtype=torch.long)
            for i, row in enumerate(ids):
                input_ids[i, : len(row)] = torch.tensor(row)
                attn[i, : len(row)] = 1
            return {"input_ids": input_ids, "attention_mask": attn}

    model = ToyCausalLM(n_layers=8, hidden=32)
    model.eval()
    layers = resolve_decoder_layers(model)
    assert len(layers) == 8

    hooks = ResidualStreamHooks(model, [1, 6], cast_dtype=torch.float32, store_cpu=True)
    x = torch.randint(1, 100, (2, 16))
    with hooks.capture():
        _ = model(input_ids=x)
    assert set(hooks.activations) == {1, 6}
    assert hooks.activations[1].shape == (2, 16, 32)
    print("toy_hooks: OK", {k: tuple(v.shape) for k, v in hooks.activations.items()})

    # collect_residual_stream path with a stub LoadedModel
    spec = get_model_spec("qwen2.5-0.5b-smoke")
    # Override layer defaults to fit toy (8 layers)
    from dataclasses import replace

    toy_spec = replace(spec, key="toy", n_layers=8, default_perturb_layer=1, default_measure_layer=6)
    loaded = LoadedModel(
        spec=toy_spec,
        model=model,
        tokenizer=_Tok(),  # type: ignore[arg-type]
        device_map="cpu",
        dtype=torch.float32,
    )

    batches = collect_residual_stream(
        loaded,
        ["hello tool call", "another string"],
        layer_indices=[1, 6],
        batch_size=2,
        cast_dtype=torch.float32,
    )
    assert len(batches) == 1
    assert batches[0].activations[1].shape[0] == 2
    print("toy_collect: OK", batches[0].activations[1].shape)


def run_hf_smoke(local_files_only: bool, model_key: str | None = None) -> None:
    from activation_pipeline.collect import collect_residual_stream
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.hooks import resolve_decoder_layers
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.memory import format_plan_report

    model_key = model_key or LOCAL_MODEL_KEY
    device_map = resolve_device_map(None)
    print(format_plan_report(model_key))
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype="float32",
        local_files_only=local_files_only,
    )
    n = len(resolve_decoder_layers(loaded.model))
    print(f"hf_smoke: loaded {loaded.spec.hf_id} layers={n}")
    batches = collect_residual_stream(
        loaded,
        ["I'll call a tool.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.py\"}}\n</tool_call>"],
        layer_indices=[0, n // 2, n - 1],
        batch_size=1,
        max_length=128,
        cast_dtype=None,
    )
    b = batches[0]
    for layer, act in b.activations.items():
        print(f"  layer {layer}: shape={tuple(act.shape)} dtype={act.dtype}")
    print("hf_smoke: OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hf",
        action="store_true",
        help="Also load smoke HF model (default qwen3-0.6b)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Registry key for HF smoke (default: qwen3-0.6b on MacBook MPS)",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download; use HF cache only",
    )
    ap.add_argument(
        "--print-plans",
        action="store_true",
        help="Print memory plans and exit",
    )
    args = ap.parse_args()

    if args.print_plans:
        from activation_pipeline.memory import format_plan_report

        print(format_plan_report())
        return 0

    run_toy()
    if args.hf:
        run_hf_smoke(local_files_only=args.local_files_only, model_key=args.model)
    else:
        print("skip hf smoke (pass --hf to run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
