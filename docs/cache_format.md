# Cache format (0.5)

Activation caches store **mean residual-stream vectors** for each tagged prose / tool-call window.

## Files

Given `--out data/activations/smoke_coding.json`:

| File | Contents |
|---|---|
| `smoke_coding.json` | Human-readable metadata + mean vectors |
| `smoke_coding.pt` | Same records as tensors (preferred for reload) |

`ActivationCache.load(path)` accepts either suffix.

## Schema version

`format_version`: `"1.0"`

### Top-level

| Field | Meaning |
|---|---|
| `model_key` / `hf_id` | Registry + HF id used for collection |
| `layer_indices` | Layers hooked (default `{k, L}`) |
| `records` | One entry per window |
| `meta` | Provenance (domains, flags, live layer count) |

### Record

| Field | Meaning |
|---|---|
| `transcript_id`, `domain`, `message_index` | Source turn |
| `window_kind` | `prose` \| `tool_call` |
| `window_name` | Tool name when applicable |
| `token_start`, `token_end`, `n_tokens` | Indices in the chat-templated sequence |
| `layer_means` | Map `layer -> float list` (mean over window tokens) |

## Collect (smoke model on this machine)

```bash
# MacBook: defaults to qwen3-0.6b on MPS
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cache_activations.py \
  --domain coding \
  --out data/activations/smoke_coding.json
```

## Collect (Qwen3-32B — CUDA host only)

Not available on MacBook. On a CUDA box:

```bash
PYTHONPATH=. python3 scripts/cache_activations.py \
  --model qwen3-32b \
  --domain coding --domain therapy --domain writing \
  --out data/activations/qwen3_32b_all.json
```

(Loader refuses `qwen3-32b` without CUDA.)
