# Phase 9U — retention: target-specific continuous gains (Arm B) vs continuous (Arm A)

> Same conversation; retention forked from acquired round-0 conversations (S_0 = m*). Calibration and evaluation use disjoint seed pools. Target-sign, 8K sites, forced FINAL:. α_O never raised.

rounds=2, n_cal=6, n_eval=12, seed=20261121.

## Held-out evaluation — $P(S_1{=}S_2{=}m^*\mid S_0{=}m^*)$

| $m^*$ | acq rate (eval) | chosen gains (C,H,O) | A continuous | B calibrated | B-only | A-only | total gain A→B |
|---|---:|---|---:|---:|---:|---:|---:|
| `100` | 12/18 | C=0, H=0, O=0.75 | 12/12 | 12/12 | 0 | 0 | 8→0.75 |

**Pooled:** A 12/12 (1.00) · B 12/12 (1.00).

## Calibration (cal pool)

### `100` — cal acq 6/8

| scale (C,H,O) | hold | total gain | first-flip channels |
|---|---:|---:|---|
| (0.0, 0.0, 0.5) **←** | 6/6 | 0.75 | — |
| (0.0, 1.0, 1.0) | 6/6 | 3 | — |
| (0.5, 1.0, 1.0) | 6/6 | 5.5 | — |
| (1.0, 0.0, 1.0) | 6/6 | 6.5 | — |
| (1.0, 0.5, 1.0) | 6/6 | 7.25 | — |
| (1.0, 1.0, 0.5) | 6/6 | 7.25 | — |
| (1.0, 1.0, 1.0) | 6/6 | 8 | — |
| (1.0, 1.5, 1.0) | 6/6 | 8.75 | — |
| (1.5, 1.0, 1.0) | 6/6 | 10.5 | — |
| (1.0, 1.0, 0.0) | 3/6 | 6.5 | {'O': 3} |

Selection on small cal pools is noisy (winner's curse); only the held-out A vs B comparison counts.
