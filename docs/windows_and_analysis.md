# Windows + analysis (0.4–0.6)

## Roles

| Step | Role | Novelty |
|---|---|---|
| 0.4 Token windows | Infra / experimental design | New cuts for tool vs prose |
| 0.5 Activation cache | Infra | Collect once, reuse |
| 0.6 Sensitivity / plateau / Procrustes | **Method reuse** | Plateau lit + your subspace exploratory |

Scientific claims start at §1 (logit-lens gate) and §5 (RQ1).

## 0.4 Tagging

```bash
PYTHONPATH=. python3 scripts/tag_windows.py --out data/windows/token_windows.jsonl
```

Tool window = tokens overlapping `[body_start, body_end)` (excludes `<tool_call>` delimiters).  
Prose = other assistant content tokens. Chat-template embedding remaps indices into the full prompt.

## 0.6 Metrics (reimplemented)

- **Directional sensitivity** (primary): binary-search ε at layer `k` until relative L2 blowup at `L` exceeds τ (default 0.5).
- **Plateau depth**: largest grid ε with blowup &lt; τ (robustness only).
- **Top-k sensitive rank**: sort by ascending ε\*.
- **Subspace**: mean cosine (centroids) + orthogonal Procrustes on window-mean activation sets; principal angles between PCA subspaces (exploratory).

```bash
PYTHONPATH=. python3 scripts/smoke_analysis.py
.venv/bin/python scripts/run_subspace_compare.py \
  --cache data/activations/real_gap.json \
  --evaluation-dataset mind_the_gap --by-domain \
  --out data/results/subspace_gap.json
```

See `docs/subspace_analysis.md` for §2 pilot results.