# Phase 8H — O gain integration ($\alpha_O\in\{5,8\}$)

> $\alpha_C=\alpha_H=1.5$; early-FINAL O; H decision-token; $d_k=(2m_k^*-1)v_c^k$. No new $v$.

m* n=8, reps=2, eight_way=True.

## Aggregate by $\alpha_O$

| $\alpha_O$ | P(hit) | ΔE | P(C) | P(H) | P(O) | P(W→C)$_O$ | E0→E3 |
|-----------:|--------|---:|------|------|------|------------|-------|
| 1.5 | 0.31 | +0.38 | 0.56 | 0.69 | 0.75 | 0.57 | 1.38→1.31→1.31→1.00 |
| 5 | 0.31 | +0.38 | 0.56 | 0.69 | 0.75 | 0.57 | 1.38→1.31→1.31→1.00 |
| 8 | 0.31 | +0.38 | 0.56 | 0.69 | 0.75 | 0.57 | 1.38→1.31→1.31→1.00 |

| 8D ref (α_O=1.5) | 0.25 | +0.50 | — | 0.62 | 0.88 | — | — |

## Gate

- O gain helps vs α_O=1.5: **False**
- α_O=5 beats 1.5 on P(O): **False**
- α_O=5 beats 1.5 on P(W→C)_O: **False**
- α_O=8 beats 1.5 on P(O): **False**
- α_O=8 beats 1.5 on P(W→C)_O: **False**
- α_O=8 − 5 on P(O): **0.0**

If α_O=5/8 lifts P(O) and P(W→C)_O vs 1.5: gain calibration works. If O↑ but hit/ΔE flat: composition/order bottleneck. No new v.
