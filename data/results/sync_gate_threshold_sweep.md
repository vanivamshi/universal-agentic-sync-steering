# Gate threshold sweep + combined probe + targeted regen

## Offline τ selection (Exp A, freeze optimistic sim)

| probe | τ tuned | false-fire budget |
|---|---:|---:|
| report | -2.7805 | 0.1 |
| tool | -2.8336 | 0.1 |
| combined (LR proba) | 0.30 | 0.1 |
| report (old median) | -2.6811 | — |

## Live regen-only val (36 ep × 3 rep)

| arm | hidden | spurious | disclose\|private | fire rate |
|---|---:|---:|---:|---:|
| baseline | 0.306 | 0.000 | 0.694 | 0.000 |
| report_median_generic | 0.083 | 0.000 | 0.917 | 0.333 |
| report_tuned_generic | 0.139 | 0.000 | 0.861 | 0.250 |
| report_tuned_targeted | 0.139 | 0.000 | 0.861 | 0.250 |
| combined_tuned_generic | 0.139 | 0.000 | 0.861 | 0.250 |
| tool_tuned_generic | 0.083 | 0.000 | 0.917 | 0.500 |

- **Best arm:** `report_median_generic`  hidden 0.306 → 0.083 (Δ +0.222)
- Prior single-threshold gate: hidden 0.361→0.139 @ fire 0.389
