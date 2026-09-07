# Collapse-avoidance test (existing Phase 1b, no new gens)

Collapse = `control_correct == 0`. Gain question stays `NOISE_STOP`.

## Decision: `AVOID_UNRESOLVED`
existing n cannot reject equal collapse rates. Nested selector cells overstate n=4. Do not expand. Do not claim a safety margin.

| arm | collapses | rate | 95% CI (Clopper–Pearson) |
|---|---:|---:|---|
| selector (4 locked bottoms) | 0/4 | 0.000 | [0.000, 0.602] |
| random (10×16 + 10×64) | 3/20 | 0.150 | [0.032, 0.379] |

Fisher exact one-sided (H1: p_sel < p_rand): **p = 0.563** (two-sided p = 1.000).
P(0 collapses | Binomial(n=4, p=0.15)) = 0.522.

Collapsed random draws:
- `random_64_seed_20260813` correct=0 uniq=0.479 held_ppl=8.22
- `random_64_seed_20260817` correct=0 uniq=0.630 held_ppl=8.19
- `random_16_seed_20260820` correct=0 uniq=0.430 held_ppl=8.00

All four selector bottoms: no collapse.
- nll_bottom_16 correct=0.667 uniq=0.939
- nll_bottom_64 correct=0.708 uniq=0.931
- gxa_bottom_16 correct=0.583 uniq=0.931
- gxa_bottom_64 correct=0.458 uniq=0.928

## Sensitivity (dependence)
- 2 methods (either-k): 0/2 vs 3/20, one-sided Fisher p = 0.740
- k=64 only: 0/2 vs 2/10, one-sided Fisher p = 0.682

Phase 1a n=12 random (not pooled): 3/20 collapse.

Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/collapse_avoidance_test.json`
