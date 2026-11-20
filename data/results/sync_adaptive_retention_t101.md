# Adaptive retention controller — same-conversation retention

> One conversation per trial: round 0 acquisition (`full8k`, shared/forked), then 2 follow-up rounds in the same message history. Paired seeds across arms. Frozen 8K actuators; target-sign; no new $v$/gains.

reps=10, rounds=2, seed=20261120, targets=1.

Primary: $P_{hold} = P(S_1 = S_2 = m^*)$.

## Overall

| arm | $P_{hold}$ | $P_{hold}\mid S_0{=}m^*$ | recovery | collateral (steered) | collateral (unsteered) | steered rounds / trial | $m^*$ with $P_{hold}>0$ |
|-----|---:|---:|---:|---:|---:|---:|---:|
| `none` | **0.30** (3/10) | 0.50 (n=2) | 0.13 (n=15) | — | 0.02 | 0.00 | 1/1 |
| `continuous` | **0.20** (2/10) | 1.00 (n=2) | 0.00 (n=16) | 0.00 | — | 2.00 | 1/1 |
| `event_triggered` | **0.10** (1/10) | 0.50 (n=2) | 0.00 (n=17) | 0.00 | 0.11 | 1.70 | 1/1 |
| `adaptive` | **0.10** (1/10) | 0.50 (n=2) | 0.00 (n=17) | 0.00 | 0.11 | 1.70 | 1/1 |

## $P_{hold}$ per target

| $m^*$ | $P(S_0{=}m^*)$ | `none` | `continuous` | `event_triggered` | `adaptive` |
|---|---:|---:|---:|---:|---:|
| `101` | 0.20 | 0.30 | 0.20 | 0.10 | 0.10 |

## Paired $\Delta P_{hold}$

| ref vs other | n | mean Δ | ref-only | other-only |
|---|---:|---:|---:|---:|
| `adaptive` vs `none` | 10 | -0.20 | 0 | 2 |
| `adaptive` vs `continuous` | 10 | -0.10 | 0 | 1 |
| `adaptive` vs `event_triggered` | 10 | 0.00 | 0 | 0 |

## Adaptive decisions

- actions: `{'O': 1, 'FULL': 15, 'CH': 1}`
- reasons: `{'rule_default': 4, 'explore_alternative': 2, 'fallback_full': 11}`

Final success tables (target · drift → action: recovered/n, collateral/n):

- `101` · drift ['C', 'H']: FULL 0/1 (coll 0), CH 0/1 (coll 0)
- `101` · drift ['C', 'H', 'O']: FULL 0/1 (coll 0)
- `101` · drift ['H', 'O']: FULL 0/1 (coll 0)
- `101` · drift ['O']: O 0/1 (coll 0), FULL 0/12 (coll 0)

Exploratory; do not overclaim at low reps.
