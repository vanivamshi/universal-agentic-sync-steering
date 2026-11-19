# Phase 9T — Closed-loop retention on one persistent trajectory

> Acquisition = unchanged 9R hybrid Γ'+beam. Retention CONTINUES the arrival conversation (no fresh episodes). Margins M_c are measured unsteered at each channel's decision site; selective arm steers only channels whose signed margin falls below `enter` (released above `exit`) or whose bit drifted last round. Frozen 8K α; no new v.

reps=2, rounds=2, enter=1.0, exit=2.5, seed=20261119. Trials: 10 (skipped S0==m*: 0, acquired: 10).

Direction sign check (M(+αv) − M(−αv) at canonical site): C: +8.42 → +1, H: +31.20 → +1, O: +3.66 → +1

## ret@2 | acquired, per $m^*$

| $m^*$ | trials | acq | none | continuous | selective |
|---|---:|---:|---:|---:|---:|
| `010` | 2 | 2 | 2/2 [0.34,1.00] | 2/2 [0.34,1.00] | 1/2 [0.09,0.91] |
| `011` | 2 | 2 | 2/2 [0.34,1.00] | 2/2 [0.34,1.00] | 1/2 [0.09,0.91] |
| `100` | 2 | 2 | 2/2 [0.34,1.00] | 2/2 [0.34,1.00] | 2/2 [0.34,1.00] |
| `101` | 2 | 2 | 1/2 [0.09,0.91] | 0/2 [0.00,0.66] | 1/2 [0.09,0.91] |
| `111` | 2 | 2 | 1/2 [0.09,0.91] | 2/2 [0.34,1.00] | 1/2 [0.09,0.91] |

## Pooled (acquired trials)

| metric | none | continuous | selective |
|---|---:|---:|---:|
| ret@1 | 9/10 [0.60,0.98] | 8/10 [0.49,0.94] | 7/10 [0.40,0.89] |
| ret@2 | 8/10 [0.49,0.94] | 8/10 [0.49,0.94] | 6/10 [0.31,0.83] |
| round hits | 17/20 | 16/20 | 13/20 |
| channels steered / round | 0.00 | 3.00 | 1.45 |

## Paired (ret@2)

| comparison | pairs | first-only / second-only | both | exact sign p |
|---|---:|---:|---:|---:|
| selective_vs_none | 10 | 0 / 2 | 6 | 0.500 |
| selective_vs_continuous | 10 | 1 / 3 | 5 | 0.625 |
| continuous_vs_none | 10 | 1 / 1 | 7 | 1.000 |

## Margin read-out vs observed bit

| channel | all arms | unsteered (none) |
|---|---:|---:|
| C | 25/60 | 9/20 |
| H | 46/60 | 15/20 |
| O | 32/60 | 12/20 |

## Reading guide

| result | suggests |
|---|---|
| none high | the persistent context alone retains m* (9S failure was the fresh-episode reset) |
| continuous > none | active steering on the same trajectory retains m* |
| selective ≈ continuous, fewer channels steered | feedback correction is sufficient and cheaper |
| selective < continuous | margin read-out / thresholds miss drift — check read-out accuracy, tune enter/exit |
| read-out accuracy low on a channel | the margin is not a reliable state decoder there; rely on observed-bit forcing |

Small n: report counts and CIs; a nonzero rate is not reliable retention.
