# Phase 9V — held-out paired validation of frozen 9U retention gains

> New seeds (disjoint from 9U). Gains frozen from 9U, no tuning. Same conversation; A and B forked from the same acquired round-0 conversation with identical round seeds. Target-sign, 8K sites, forced FINAL:.

rounds=2, seed=31000000, frozen config `data/configs/phase9v_frozen_gains.json`.

## Primary — $P(S_1{=}S_2{=}m^*\mid S_0{=}m^*)$

| $m^*$ | B gains | A hold | A 95% CI | B hold | B 95% CI | B-only | A-only | McNemar p | gain A→B |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| `100` | O=0.75 | 40/40 | [0.91, 1.00] | 40/40 | [0.91, 1.00] | 0 | 0 | 1.000 | 8→0.75 |
| `101` | O=0.75 | 40/40 | [0.91, 1.00] | 40/40 | [0.91, 1.00] | 0 | 0 | 1.000 | 8→0.75 |
| `111` | H=1.5, O=1.5 | 15/24 | [0.43, 0.79] | 18/24 | [0.55, 0.88] | 8 | 5 | 0.581 | 8→3 |

**Pooled:** A 95/104 [0.84, 0.95] · B 98/104 [0.88, 0.97] · discordant B-only 8 / A-only 5 (McNemar p=0.581).

## Acquisition and end-to-end

| $m^*$ | acq | $P(S_0{=}m^*)$ | 95% CI | e2e A | e2e B | complete |
|---|---:|---:|---|---:|---:|---|
| `100` | 40/66 | 0.61 | [0.49, 0.71] | 0.61 | 0.61 | yes |
| `101` | 40/151 | 0.26 | [0.20, 0.34] | 0.26 | 0.26 | yes |
| `111` | 24/96 | 0.25 | [0.17, 0.35] | 0.16 | 0.19 | yes |

Gains were selected in 9U on n=2–6 calibration conversations; 9V is the first test on fresh seeds. Equal hold rates with overlapping CIs mean *not distinguishable at this n*, not proven equivalence.
