# Channel controllability matrix

- Mode: `correlational_not_causal`
- approx_diagonal: **True**
- diag_abs_mean: 0.4657
- offdiag_l2: 0.7255

> Correlations of candidate projections with observed S. Do not treat as controllability. Run test_channel_controllability.py --mode causal.

Rows = affected channel (C, H, O); cols = steered direction (v_C, v_H, v_O).

| | v_C | v_H | v_O |
|---|-----|-----|-----|
| ΔC | 0.627 | 0.000 | 0.000 |
| ΔH | 0.616 | 0.302 | -0.196 |
| ΔO | -0.228 | 0.237 | 0.468 |

