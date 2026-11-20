# Phase 9U — retention: target-specific continuous gains (Arm B) vs continuous (Arm A)

> Same conversation; retention forked from acquired round-0 conversations (S_0 = m*). Calibration and evaluation use disjoint seed pools. Target-sign, 8K sites, forced FINAL:. α_O never raised.

rounds=2, n_cal=6, n_eval=12, seed=20261121.

## Held-out evaluation — $P(S_1{=}S_2{=}m^*\mid S_0{=}m^*)$

| $m^*$ | acq rate (eval) | chosen gains (C,H,O) | A continuous | B calibrated | B-only | A-only | total gain A→B |
|---|---:|---|---:|---:|---:|---:|---:|
| `110` | 2/60 | C=0, H=0.75, O=0.75 | 1/2 | 2/2 | 1 | 0 | 8→1.5 |

**Pooled:** A 1/2 (0.50) · B 2/2 (1.00).

## Calibration (cal pool)

### `110` — cal acq 2/60

| scale (C,H,O) | hold | total gain | first-flip channels |
|---|---:|---:|---|
| (0.0, 0.5, 0.5) **←** | 2/2 | 1.5 | — |
| (0.0, 1.0, 1.0) | 2/2 | 3 | — |
| (1.0, 0.5, 1.0) | 2/2 | 7.25 | — |
| (1.0, 1.0, 0.5) | 2/2 | 7.25 | — |
| (1.0, 1.0, 1.0) | 2/2 | 8 | — |
| (1.0, 1.5, 1.0) | 2/2 | 8.75 | — |
| (0.5, 1.0, 1.0) | 1/2 | 5.5 | {'C': 1} |
| (1.0, 0.0, 1.0) | 0/2 | 6.5 | {'C': 2, 'H': 2} |
| (1.0, 1.0, 0.0) | 0/2 | 6.5 | {'O': 2} |
| (1.5, 1.0, 1.0) | 0/2 | 10.5 | {'C': 2} |

Selection on small cal pools is noisy (winner's curse); only the held-out A vs B comparison counts.
