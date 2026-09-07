# Phase 8J — C actual decision control

> Close $B_C^{\mathrm{live}}\!\to\!S_C^{\mathrm{proxy}}\!\to\!S_C^{\mathrm{generation}}$. No C-skip for 8-way. No new $v$. Same-seed Stage B (8C protocol).

n=12, seed=20260916, α_B=(1.5, 3.0, 5.0, 8.0), max_new=64.

## Stage A — soft/proxy

| $\Delta B(+\alpha)$ | $\Delta B(-\alpha)$ | Phase7 ref |
|---------------------:|--------------------:|-----------:|
| +0.082 | -0.082 | ±0.082 |

| condition | $E[G]$ | $P(W\to C)$ | n_wrong |
|-----------|-------:|------------:|--------:|
| no_steer | +0.000 | 0.667 | 12 |
| wrong_sign | -0.082 | 0.250 | 12 |
| target_sign | +0.082 | 0.667 | 12 |

- Stage A pass: **False**
- 8F ref TS P(W→C)=0.917

## Stage B — decision-token generation (same seed)

| $\alpha_C$ | condition | $E[G]$ | $P(W\to C)_{gen}$ | n_wrong | pass |
|-----------:|-----------|-------:|------------------:|--------:|:----:|
| 1.5 | no_steer | +0.000 | 0.250 | 12 |  |
| 1.5 | wrong_sign | -0.082 | 0.250 | 12 |  |
| 1.5 | target_sign | +0.082 | 0.250 | 12 | False |
| 3 | no_steer | +0.000 | 0.250 | 12 |  |
| 3 | wrong_sign | -0.164 | 0.250 | 12 |  |
| 3 | target_sign | +0.164 | 0.333 | 12 | True |
| 5 | no_steer | +0.000 | 0.250 | 12 |  |
| 5 | wrong_sign | -0.273 | 0.250 | 12 |  |
| 5 | target_sign | +0.273 | 0.417 | 12 | True |
| 8 | no_steer | +0.000 | 0.250 | 12 |  |
| 8 | wrong_sign | -0.436 | 0.167 | 12 |  |
| 8 | target_sign | +0.436 | 0.167 | 12 | False |

| 8F Stage B (buggy seed) | target_sign | +0.082 | 0.500 | — | False |

## Gate

- Stage A pass: **False** (TS beats WS, not none; soft site fragile this seed)
- Stage B pass (any α): **True** (α=3,5 clear +0.05 bar only)
- Best α_C: **5.0** (TS P(W→C)=0.42)
- C generation fixed: **False** (absolute still weak; dose-responsive only)

$$
\boxed{C\text{ gen under-actuated: dose helps to }\alpha_C{=}5\text{; not yet H-grade}}
$$

If Stage B had passed strongly: wire C decision-token into H→C→O. Here: **partial** — next wire $\alpha_C{=}5$ stem-prefill into frozen $H\to C\to O$ and test $\Delta E_C$, still without skipping C. Conditional C-skip = diagnostic only, never 8-way success.

C remains required for universal 8-way.
