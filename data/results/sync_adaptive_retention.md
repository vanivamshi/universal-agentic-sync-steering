# Adaptive retention controller — same-conversation retention

> One conversation per trial: round 0 acquisition (`full8k`, shared/forked), then 2 follow-up rounds in the same message history. Paired seeds across arms. Frozen 8K actuators; target-sign; no new $v$/gains.

reps=10, rounds=2, seed=20261120, targets=6.

Primary: $P_{hold} = P(S_1 = S_2 = m^*)$.

## Overall

| arm | $P_{hold}$ | $P_{hold}\mid S_0{=}m^*$ | recovery | collateral (steered) | collateral (unsteered) | steered rounds / trial | $m^*$ with $P_{hold}>0$ |
|-----|---:|---:|---:|---:|---:|---:|---:|
| `none` | **0.17** (10/58) | 0.57 (n=14) | 0.02 (n=92) | — | 0.17 | 0.00 | 2/6 |
| `continuous` | **0.22** (13/58) | 0.93 (n=14) | 0.00 (n=89) | 0.02 | — | 2.00 | 4/6 |
| `event_triggered` | **0.14** (8/58) | 0.57 (n=14) | 0.00 (n=94) | 0.04 | 0.09 | 1.62 | 2/6 |
| `adaptive` | **0.14** (8/58) | 0.57 (n=14) | 0.01 (n=93) | 0.04 | 0.12 | 1.60 | 2/6 |

## $P_{hold}$ per target

| $m^*$ | $P(S_0{=}m^*)$ | `none` | `continuous` | `event_triggered` | `adaptive` |
|---|---:|---:|---:|---:|---:|
| `001` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `011` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `100` | 0.89 | 0.78 | 0.89 | 0.78 | 0.78 |
| `101` | 0.20 | 0.30 | 0.20 | 0.10 | 0.10 |
| `110` | 0.20 | 0.00 | 0.20 | 0.00 | 0.00 |
| `111` | 0.20 | 0.00 | 0.10 | 0.00 | 0.00 |

## Paired $\Delta P_{hold}$

| ref vs other | n | mean Δ | ref-only | other-only |
|---|---:|---:|---:|---:|
| `adaptive` vs `none` | 58 | -0.03 | 0 | 2 |
| `adaptive` vs `continuous` | 58 | -0.09 | 0 | 5 |
| `adaptive` vs `event_triggered` | 58 | 0.00 | 0 | 0 |

## Adaptive decisions

- actions: `{'FULL': 75, 'H': 4, 'CH': 9, 'O': 4, 'HO': 1}`
- reasons: `{'rule_default': 23, 'fallback_full': 58, 'explore_alternative': 12}`

Final success tables (target · drift → action: recovered/n, collateral/n):

- `001` · drift ['C']: CH 0/1 (coll 1), FULL 0/4 (coll 0)
- `001` · drift ['C', 'H']: FULL 0/1 (coll 0), CH 0/1 (coll 0)
- `001` · drift ['C', 'O']: FULL 0/8 (coll 0)
- `001` · drift ['H']: H 0/1 (coll 0), CH 0/1 (coll 0)
- `001` · drift ['O']: O 0/1 (coll 1)
- `011` · drift ['C']: CH 0/1 (coll 0), FULL 0/6 (coll 0)
- `011` · drift ['C', 'H', 'O']: FULL 0/1 (coll 0)
- `011` · drift ['C', 'O']: FULL 0/3 (coll 0)
- `011` · drift ['H']: H 1/1 (coll 0)
- `011` · drift ['H', 'O']: FULL 0/2 (coll 1), HO 0/1 (coll 0)
- `011` · drift ['O']: O 0/1 (coll 1), FULL 0/3 (coll 0)
- `100` · drift ['C', 'H', 'O']: FULL 0/2 (coll 0)
- `100` · drift ['O']: O 0/1 (coll 0)
- `101` · drift ['C', 'H']: FULL 0/1 (coll 0), CH 0/1 (coll 0)
- `101` · drift ['C', 'H', 'O']: FULL 0/1 (coll 0)
- `101` · drift ['H', 'O']: FULL 0/1 (coll 0)
- `101` · drift ['O']: O 0/1 (coll 0), FULL 0/12 (coll 0)
- `110` · drift ['C']: CH 0/1 (coll 0), FULL 0/1 (coll 0)
- `110` · drift ['C', 'O']: FULL 0/14 (coll 0)
- `110` · drift ['H']: H 0/1 (coll 0), CH 0/1 (coll 0)
- `111` · drift ['C']: CH 0/1 (coll 0), FULL 0/9 (coll 1)
- `111` · drift ['C', 'O']: FULL 0/6 (coll 0)
- `111` · drift ['H']: H 0/1 (coll 0), CH 0/1 (coll 0)

Exploratory; do not overclaim at low reps.
