# Adaptive retention controller — same-conversation retention

> One conversation per trial: round 0 acquisition (`full8k`, shared/forked), then 2 follow-up rounds in the same message history. Paired seeds across arms. Frozen 8K actuators; target-sign; no new $v$/gains.

reps=10, rounds=2, seed=20261120, targets=1.

Primary: $P_{hold} = P(S_1 = S_2 = m^*)$.

## Overall

| arm | $P_{hold}$ | $P_{hold}\mid S_0{=}m^*$ | recovery | collateral (steered) | collateral (unsteered) | steered rounds / trial | $m^*$ with $P_{hold}>0$ |
|-----|---:|---:|---:|---:|---:|---:|---:|
| `none` | **0.00** (0/10) | 0.00 (n=1) | 0.00 (n=19) | — | 0.52 | 0.00 | 0/1 |
| `continuous` | **0.20** (2/10) | 1.00 (n=1) | 0.06 (n=17) | 0.04 | — | 2.00 | 1/1 |
| `event_triggered` | **0.00** (0/10) | 0.00 (n=1) | 0.11 (n=18) | 0.05 | 0.67 | 1.80 | 0/1 |
| `adaptive` | **0.00** (0/10) | 0.00 (n=1) | 0.11 (n=18) | 0.16 | 0.67 | 1.80 | 0/1 |

## $P_{hold}$ per target

| $m^*$ | $P(S_0{=}m^*)$ | `none` | `continuous` | `event_triggered` | `adaptive` |
|---|---:|---:|---:|---:|---:|
| `010` | 0.10 | 0.00 | 0.20 | 0.00 | 0.00 |

## Paired $\Delta P_{hold}$

| ref vs other | n | mean Δ | ref-only | other-only |
|---|---:|---:|---:|---:|
| `adaptive` vs `none` | 10 | 0.00 | 0 | 0 |
| `adaptive` vs `continuous` | 10 | -0.20 | 0 | 2 |
| `adaptive` vs `event_triggered` | 10 | 0.00 | 0 | 0 |

## Adaptive decisions

- actions: `{'FULL': 15, 'O': 1, 'H': 1, 'HO': 1}`
- reasons: `{'rule_default': 6, 'fallback_full': 11, 'explore_alternative': 1}`

Final success tables (target · drift → action: recovered/n, collateral/n):

- `010` · drift ['C', 'H']: FULL 1/1 (coll 0)
- `010` · drift ['C', 'H', 'O']: FULL 0/1 (coll 0)
- `010` · drift ['C', 'O']: FULL 0/12 (coll 0)
- `010` · drift ['H']: H 1/1 (coll 0)
- `010` · drift ['H', 'O']: FULL 0/1 (coll 0), HO 0/1 (coll 1)
- `010` · drift ['O']: O 0/1 (coll 1)

Exploratory; do not overclaim at low reps.
