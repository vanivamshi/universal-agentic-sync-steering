# Adaptive retention controller — same-conversation retention

> One conversation per trial: round 0 acquisition (`full8k`, shared/forked), then 2 follow-up rounds in the same message history. Paired seeds across arms. Frozen 8K actuators; target-sign; no new $v$/gains.

reps=10, rounds=2, seed=20261120, targets=1.

Primary: $P_{hold} = P(S_1 = S_2 = m^*)$.

## Overall

| arm | $P_{hold}$ | $P_{hold}\mid S_0{=}m^*$ | recovery | collateral (steered) | collateral (unsteered) | steered rounds / trial | $m^*$ with $P_{hold}>0$ |
|-----|---:|---:|---:|---:|---:|---:|---:|
| `none` | **0.00** (0/10) | 0.00 (n=2) | 0.00 (n=18) | — | 0.27 | 0.00 | 0/1 |
| `continuous` | **0.10** (1/10) | 0.50 (n=2) | 0.00 (n=17) | 0.03 | — | 2.00 | 1/1 |
| `event_triggered` | **0.00** (0/10) | 0.00 (n=2) | 0.00 (n=18) | 0.03 | 0.33 | 1.80 | 0/1 |
| `adaptive` | **0.00** (0/10) | 0.00 (n=2) | 0.00 (n=18) | 0.03 | 0.33 | 1.80 | 0/1 |

## $P_{hold}$ per target

| $m^*$ | $P(S_0{=}m^*)$ | `none` | `continuous` | `event_triggered` | `adaptive` |
|---|---:|---:|---:|---:|---:|
| `111` | 0.20 | 0.00 | 0.10 | 0.00 | 0.00 |

## Paired $\Delta P_{hold}$

| ref vs other | n | mean Δ | ref-only | other-only |
|---|---:|---:|---:|---:|
| `adaptive` vs `none` | 10 | 0.00 | 0 | 0 |
| `adaptive` vs `continuous` | 10 | -0.10 | 0 | 1 |
| `adaptive` vs `event_triggered` | 10 | 0.00 | 0 | 0 |

## Adaptive decisions

- actions: `{'FULL': 15, 'CH': 2, 'H': 1}`
- reasons: `{'rule_default': 3, 'fallback_full': 13, 'explore_alternative': 2}`

Final success tables (target · drift → action: recovered/n, collateral/n):

- `111` · drift ['C']: CH 0/1 (coll 0), FULL 0/9 (coll 1)
- `111` · drift ['C', 'O']: FULL 0/6 (coll 0)
- `111` · drift ['H']: H 0/1 (coll 0), CH 0/1 (coll 0)

Exploratory; do not overclaim at low reps.
