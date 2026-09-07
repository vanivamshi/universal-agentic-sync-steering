# Selective steering v2 — eps*_beh + E/F/M

- Decision: **`SELECTIVE_V2_WEAK`**
- tau_beh=0.1607 (25% of C0/C2 readout spread on train)

## Predictor → effectiveness (Spearman)

| predictor | rho vs E | rho vs collateral |
|---|---:|---:|
| eps_global | 1.000 | 0.905 |
| **eps_beh** | **0.881** | — |
| geom_align | -0.714 | — |

## Per-direction (test)

| dir | eps_g | eps_b | align | E | F | M | EF | bidir | d+(extra) | d-(extra) |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| `orth_u2` | 7.517 | 2.0000 | 0.048 | 0.125 | 0.00 | 0.50 | 0.000 | False | -0.12 | +0.00 |
| `u1` | 7.517 | 1.4108 | 0.114 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |
| `u2` | 7.517 | 0.8177 | 0.196 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |
| `neg_u1` | 7.517 | 1.4108 | 0.114 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |
| `neg_u2` | 7.517 | 0.8177 | 0.196 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |
| `rand_sub` | 7.517 | 1.9734 | 0.081 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |
| `rand_full` | 7.517 | 2.0000 | 0.046 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |
| `rand_full2` | 7.517 | 2.0000 | 0.070 | 0.000 | 0.00 | 0.00 | 0.000 | False | +0.00 | +0.00 |

- Best SVD E=0.000 vs random E=0.125
- Best SVD E×F=0.000 vs random E×F=0.000
