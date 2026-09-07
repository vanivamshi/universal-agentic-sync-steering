# §3 Safety direction validity (pilot)

Prereg §5.2 (frozen): split-half baseline, **layer-matched** reporting,
**BCa + joint** gap CIs, fixed mid/late stability set, §3-own correction family
for exploratory depth-grid tests.

## CI method (frozen)

| Quantity | Method |
|---|---|
| prose↔tool cosine | BCa bootstrap (not percentile) |
| gap \(=\cos_{pp}-\cos_{pt}\) | **Joint** bootstrap: each replicate resamples prose **and** tool, recomputes both cosines, then gap |

Pilot also logs percentile CIs to show small-*n* pathology (e.g. skewed intervals
that poorly cover the point estimate). Prefer BCa in 32B writeups; treat 0.6B
intervals as diagnostic only.

## Layer-matched joint BCa (0.6B, n_boot=400)

Canonical: `data/results/refusal_stability_layer_matched.json`

| Layer | pt point | pt BCa 95% | gap point | gap BCa 95% | excl. 0? |
|---|---|---|---|---|---|
| 4 | 0.36 | [0.40, 0.43]* | 0.18 | [−0.23, 0.46] | no |
| 14 | 0.31 | [0.25, 0.38] | 0.29 | [−0.04, 0.35] | **no** |
| 22 | 0.16 | [0.16, 0.20] | 0.51 | [0.42, 0.56] | **yes** |

\*L4 BCa interval does not contain the point estimate — known small-*n* BCa
instability; do not interpret as a precise CI. Under **joint** BCa, the L14 gap
no longer excludes 0 (unlike the earlier conditional percentile CI). Only L22
still shows a gap CI excluding 0. Status remains suggestive + scale-confounded.

## 32B design (frozen)

- Stability gate: majority (≥3/4) of fixed \(F_{mid}=\{0.50,0.60,0.70,0.80\}\)
  with \(\cos\ge 0.70\) (prefer BCa LO ≥ 0.70). No post-hoc layer picking.
- Depth grid: early + dense mid→late + final (trend exploratory).
- Correction: gate = no multiplicity; exploratory gap/trend = BH-FDR across
  depth-grid layers **within §3** (not folded into §1 GAP⊥τ).

## Commands

```bash
.venv/bin/python scripts/run_refusal_split_half.py --n-splits 20 --layers 4 14 22
.venv/bin/python scripts/run_refusal_stability_bootstrap.py --n-boot 400 --layers 4 14 22
```
