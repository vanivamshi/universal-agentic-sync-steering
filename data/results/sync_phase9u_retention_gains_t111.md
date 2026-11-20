# Phase 9U — retention: target-specific continuous gains (Arm B) vs continuous (Arm A)

> Same conversation; retention forked from acquired round-0 conversations (S_0 = m*). Calibration and evaluation use disjoint seed pools. Target-sign, 8K sites, forced FINAL:. α_O never raised.

rounds=2, n_cal=6, n_eval=12, seed=20261121.

## Held-out evaluation — $P(S_1{=}S_2{=}m^*\mid S_0{=}m^*)$

| $m^*$ | acq rate (eval) | chosen gains (C,H,O) | A continuous | B calibrated | B-only | A-only | total gain A→B |
|---|---:|---|---:|---:|---:|---:|---:|
| `111` | 12/46 | C=0, H=1.5, O=1.5 | 11/12 | 10/12 | 0 | 1 | 8→3 |

**Pooled:** A 11/12 (0.92) · B 10/12 (0.83).

## Calibration (cal pool)

### `111` — cal acq 6/27

| scale (C,H,O) | hold | total gain | first-flip channels |
|---|---:|---:|---|
| (0.0, 1.0, 1.0) **←** | 6/6 | 3 | — |
| (0.0, 0.5, 0.0) | 5/6 | 0.75 | {'H': 1} |
| (0.5, 1.0, 1.0) | 5/6 | 5.5 | {'C': 1} |
| (1.0, 1.0, 0.0) | 5/6 | 6.5 | {'C': 1} |
| (1.0, 0.5, 1.0) | 5/6 | 7.25 | {'C': 1} |
| (1.0, 1.0, 0.5) | 5/6 | 7.25 | {'C': 1} |
| (1.0, 1.0, 1.0) | 5/6 | 8 | {'C': 1} |
| (1.0, 1.5, 1.0) | 5/6 | 8.75 | {'C': 1} |
| (1.5, 1.0, 1.0) | 1/6 | 10.5 | {'C': 5} |
| (1.0, 0.0, 1.0) | 0/6 | 6.5 | {'H': 6, 'C': 1} |

Selection on small cal pools is noisy (winner's curse); only the held-out A vs B comparison counts.
