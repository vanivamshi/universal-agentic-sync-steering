# §2 Orthogonal subspace analysis (exploratory)

Preregistered exploratory: **Procrustes alignment and mean cosine** computed on
the sets of residual-stream **window-mean activations** from prose and tool-call
windows (one vector per tagged window). Principal angles are an **extra
exploratory descriptor** (not a hypothesis test). Motivating evidence only —
does **not** accept/reject RQ1.

**How each quantity is computed**

| Quantity | Input | Procedure |
|---|---|---|
| Mean cosine | Window-mean vectors | Cosine between the prose-set centroid and the tool-set centroid |
| Procrustes disparity | Same centered point sets | Orthogonal Procrustes; report \(\|AR - B\|_F^2 / \|B\|_F^2\) |
| Principal angles | Same point sets | PCA subspaces fit independently to prose and tool sets (top-\(r\) right singular vectors of the centered matrices, \(r \le 4\)); angles between those subspaces |

**Sole model:** Qwen3-0.6B. No 32B track.

## Status (2026-08-06 refresh)

Expanded GAP cache: **n_prose=48, n_tool=24** (was 16/8). Tool group still
short of the ≥~30/group target; prose meets it. Descriptors remain exploratory
— better powered, still not a confirmatory subspace claim.

Artifact: `data/results/subspace_gap.json`

| Layer | mean cosine | Procrustes disparity | mean principal ∠ | n_prose | n_tool |
|---|---:|---:|---:|---:|---:|
| 4 | 0.888 | 1.091 | 76.6° | 48 | 24 |
| 22 | 0.871 | 0.723 | 67.1° | 48 | 24 |

## What the metrics mean

| Metric | Supports | Does **not** support |
|---|---|---|
| Mean cosine ≈ 0.8–0.9 | Centroids substantially aligned | Identity of the two modes |
| Procrustes disparity > 0 | Not perfectly alignable by an orthogonal map | Cross-dataset “closer/farther” rankings |
| Principal angles | Extra descriptor of estimated subspace orientation | A preregistered claim; small-*n* unstable |

**Interpretation (use this wording):**

> Across both datasets, prose and tool activations remain substantially aligned
> (cosine ≈ 0.8–0.9), but cannot be perfectly aligned by an orthogonal
> transformation, indicating measurable geometric differences without evidence
> of complete representational separation.

Do **not** compare raw disparity values across GAP vs τ (or L4 vs L22 as a
“closeness” ranking) without matched sample size / scaling / preprocessing.

Remaining to strengthen: grow tool windows to ≥30 (more GAP trajs or
multi-tool turns) before treating geometry as load-bearing for Phase 1.


## Observation to test on Qwen3-32B

Within each dataset, mean cosine is slightly lower at the later layer
(GAP 0.88→0.86; τ 0.85→0.79). Label only as a pilot observation for the
primary model — not a claim.

## Results (0.6B pilot)

**GAP** (`data/results/subspace_gap.json`, 16 prose / 8 tool):

| Layer | mean cosine | Procrustes disparity | mean principal ∠ |
|---|---|---|---|
| 4 | ≈ 0.877 | ≈ 1.53 | ≈ 77.5° |
| 22 | ≈ 0.865 | ≈ 0.82 | ≈ 66.8° |

**τ short** (`data/results/subspace_tau.json`, 7 prose / 6 tool):

| Layer | mean cosine | Procrustes disparity | mean principal ∠ |
|---|---|---|---|
| 4 | ≈ 0.855 | ≈ 1.22 | ≈ 82.3° |
| 22 | ≈ 0.794 | ≈ 1.45 | ≈ 83.2° |

Large principal angles together with high mean cosine suggest that the central
tendency of the activation sets is more similar than their estimated subspace
orientation. Given the small pilot sample (**6–16 points** in a high-dimensional
residual stream), PCA subspaces are underdetermined — treat angles and the
early→late cosine dip as descriptive hypotheses only, not signals to chase until
≈30+ windows per group.

## Separation from RQ1

| Section | Question |
|---|---|
| §2 | Are prose/tool activation sets geometrically related? |
| §5 | Do safety directions preserve privilege under perturbation? |

## Commands

```bash
.venv/bin/python scripts/run_subspace_compare.py \
  --cache data/activations/real_gap.json \
  --evaluation-dataset mind_the_gap --by-domain \
  --out data/results/subspace_gap.json

.venv/bin/python scripts/run_subspace_compare.py \
  --cache data/activations/tau_short.json \
  --evaluation-dataset tau_bench \
  --out data/results/subspace_tau.json
```
