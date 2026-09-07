# Tan directional agreement diagnostic

Paper: Tan et al., *Understanding (Un)Reliability of Steering Vectors*.

**Goal:** test extract quality *before* more steering — whether per-example
activation differences that form a mean-diff (or that a persona PC is asked to
capture) actually point the same way.

**Script:** `scripts/run_tan_directional_agreement.py`  
**Artifact:** `data/results/tan_directional_agreement.{json,md}`

## Locked metrics (cached GAP L4 means; no new gen)

| Metric | What |
|---|---|
| δ pairwise cos | mean pairwise cosine among `h_pos_i − μ_neg` |
| LOO pairwise cos | leave-one-out mean-diff directions (performance E2 cousin) |
| mean cos(δ, d) | alignment of each δ to the averaged direction / PC |
| anti-aligned frac | fraction of δ with opposite sign to mean alignment |

**Bars:** `HIGH_AGREEMENT` if δ-pairwise ≥ 0.50 **and** LOO/align ≥ 0.50 **and**
anti-frac < 0.30. Else `LOW` / `MIXED` / `HIGH_BUT_ANTI_FRAC`.

**Important:** LOO-stable mean-diff can still be `LOW_AGREEMENT` when δ pairwise
is low (average washes out disagreement). Prefer δ pairwise as the Tan signal.

## Targets

1. `surface_gap` mean-diff (built from eliciting labels)
2. Track A PCs `{5,12,4,7,24}` vs the same δ cloud
3. `performance_meandiff` stored E2 as HIGH reference

Per-trial anti-steerable *steer* rates need trial-level steer logs (not in
aggregate Track A JSON) — deferred.
