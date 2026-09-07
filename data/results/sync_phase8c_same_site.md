# Phase 8C — Same-site H counterfactual

> Counterfactual activation test ≠ full-episode intervention. No change to $v_c$, α, target-sign, or H boundary.

α=1.5, n=16, seed=20260914, stage=both.

## Geometry check (frozen $h_0$)

| $\Delta B(+\alpha)$ | $\Delta B(-\alpha)$ | Phase 7 / 8B ref |
|---------------------:|--------------------:|-----------------:|
| +0.412 | -0.412 | ±0.412 |

## Stage A — same-site decision (acceptance table)

| condition | $E[G_H]$ (analytical) | $P(W\to C)$ | n_wrong |
|-----------|----------------------:|------------:|--------:|
| no_steer | +0.000 | 0.375 | 16 |
| wrong_sign | -0.412 | 0.188 | 16 |
| target_sign | +0.412 | 0.688 | 16 |

- Stage A pass: **True**
- ΔB matches Phase 7: **True**
- TS improves W→C vs wrong-sign: **True**
- TS improves W→C vs no-steer: **True**

Hooked residual $G$ (diagnostic, not acceptance):

| condition | $E[G_{hook}]$ |
|-----------|--------------:|
| wrong_sign | -0.412 |
| target_sign | +0.412 |

## Stage B — decision-token-only generation

| condition | $E[G_H]$ | $P(W\to C)$ | n_wrong |
|-----------|---------:|------------:|--------:|
| no_steer | +0.000 | 0.250 | 16 |
| wrong_sign | -0.412 | 0.000 | 16 |
| target_sign | +0.412 | 0.562 | 16 |

- Stage B pass: **True**


If Stage A passes: measurement path is fixed; proceed to decision-token-only (Stage B). If Stage A fails: H boundary / v_c geometry broken (should not happen). If A passes but B fails: token timing within H generation.

8-way still paused.
