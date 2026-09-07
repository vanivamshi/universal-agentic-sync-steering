# Activation pipeline (0.3)

Residual-stream hooks for Qwen3-32B (primary) and Llama-3.1-70B (replication), plus a small smoke model.

## Layout

| Path | Role |
|---|---|
| `activation_pipeline/registry.py` | Model keys, layer counts, default `(k, L)` |
| `activation_pipeline/memory.py` | VRAM / device_map / batch plans |
| `activation_pipeline/hooks.py` | `model.model.layers[i]` forward hooks |
| `activation_pipeline/loader.py` | HF load with plan defaults; refuses 32B on CPU/no-CUDA; MPS via load-then-`.to("mps")` |
| `activation_pipeline/collect.py` | Batched tokenize → forward → activations |
| `activation_pipeline/toy_model.py` | Tiny stand-in for CI / no-GPU smoke |

## Memory plan (this MacBook)

**Apple Silicon (MPS), no NVIDIA CUDA.** Local default model is **`qwen3-0.6b`**
(`activation_pipeline.device.LOCAL_MODEL_KEY`). Scripts refuse `qwen3-32b` /
`llama-3.1-70b` without CUDA.

| Model | Role | Plan |
|---|---|---|
| **Qwen3-0.6B** | **MacBook default** | MPS (`device_map=mps` via load→`.to("mps")`), float32, `k=4`, `L=22` |
| Qwen3-32B | Primary (CUDA host only) | ≥2×80 GB; not on this machine |
| Llama-3.1-70B | Replication (CUDA host only) | ≥2–4×80 GB; not on this machine |

Print plans:

```bash
PYTHONPATH=. python3 scripts/smoke_activation_pipeline.py --print-plans
```

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Smoke tests

Toy hooks (no download):

```bash
PYTHONPATH=. python3 scripts/smoke_activation_pipeline.py
```

Optional HF smoke (uses `Qwen/Qwen3-0.6B`; smaller stand-in for 32B):

```bash
PYTHONPATH=. python3 scripts/smoke_activation_pipeline.py --hf --local-files-only --model qwen3-0.6b
```

## Collect on a GPU box (Qwen3-32B)

```python
from activation_pipeline import load_model_and_tokenizer, collect_residual_stream

loaded = load_model_and_tokenizer("qwen3-32b")  # needs CUDA + enough VRAM
batches = collect_residual_stream(
    loaded,
    texts=["...assistant turn with <tool_call>...</tool_call>..."],
    # defaults: layers {8, 48} from registry
    batch_size=1,
    max_length=4096,
)
acts_k = batches[0].activations[8]   # (1, seq, hidden)
acts_L = batches[0].activations[48]
```

## Layer path

Hooks attach to **`model.model.layers[i]`** (HF Llama / Qwen2 / Qwen3). Captured tensor = post-block residual stream `(batch, seq, hidden)`.

## Next (0.4+)

Window tagging on transcript token spans → cache only prose / tool-call token slices from these activations.
