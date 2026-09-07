# Tan-style directional agreement (extract quality)

L4. surface_gap pos=8 neg=22. Cached means only.

## Decision: `EXTRACT_UNRELIABLE_SURFACE`
surface_gap mean-diff deltas disagree — Tan predicts unreliable steer; do not invest in more α grids on this extract. Prefer RQ1 privilege or rebuild contrasts with higher agreement.

### surface_gap mean-diff
- δ pairwise cos=0.240
- LOO pairwise cos=0.985 (vs full=0.993)
- mean cos(δ, d)=0.576; anti-aligned frac=0.000
- tag=`LOW_AGREEMENT`

### Track A persona PCs vs same δ cloud
| PC | mean cos(δ,PC) | anti frac | tag | Track A tier |
|---:|---:|---:|---|---|
| 5 | 0.000 | 0.625 | `LOW_AGREEMENT` | NULL_OTHER |
| 12 | 0.005 | 0.375 | `LOW_AGREEMENT` | NULL_OTHER |
| 4 | 0.076 | 0.000 | `LOW_AGREEMENT` | NULL_OTHER |
| 7 | 0.046 | 0.250 | `LOW_AGREEMENT` | NULL_OTHER |
| 24 | 0.002 | 0.500 | `LOW_AGREEMENT` | NULL_OTHER |

### Reference: performance mean-diff (prior)
- E2 LOO pairwise=0.93840533043399 → `HIGH_AGREEMENT`
- P1=OK (in-sample only; holdout not licensed)

Bars: HIGH if δ-pairwise≥0.50 and LOO/align≥0.50 and anti-frac<0.30.
Anti-steerable *steer* rate: not available from aggregate Track A JSON.
Artifact: `/Users/vamshisunku.mohan/multidim-steering-agentic-alignment/data/results/tan_directional_agreement.json`

**Not a causal claim.**
