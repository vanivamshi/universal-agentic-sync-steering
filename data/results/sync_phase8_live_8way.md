# Phase 8A — Live-boundary 8-way

> $\mathcal{L}_{\mathrm{live}}=\sum_k\mathrm{softplus}(-\beta s_k B_k)$ with Phase-7 $B_k=w_k^\top h+b_k$.

mode=**fast**, α=1.5, β=1.0, reps=2/m*, frozen $v_c$.

## Aggregate

| Arm | P(hit) | ΔE | E[−ΔL_live] | frac L↓ | corr(−ΔL,ΔE) | G_C | G_H | G_O |
|-----|--------|-----|-------------|---------|--------------|-----|-----|-----|
| none | 0.062 | -0.250 | +0.293 | 0.56 | +0.099 | +0.00 | -0.39 | +0.05 |
| predictive | 0.188 | +0.000 | -0.529 | 0.56 | +0.094 | +0.00 | -0.24 | -0.24 |
| converted | 0.125 | -0.312 | -0.176 | 0.50 | +0.525 | +0.00 | -0.50 | +0.11 |
| random | 0.125 | -0.188 | -0.120 | 0.44 | +0.056 | +0.00 | +0.11 | +0.02 |

## P(W→C) at live trajectory

| Arm | C | H | O |
|-----|---|---|---|
| none | 0.33 | 0.00 | 0.38 |
| predictive | 0.56 | 0.25 | 0.14 |
| converted | 0.57 | 0.11 | 0.60 |
| random | 0.14 | 0.00 | 0.38 |

## Gate

- $v_c$ beats random on E[−ΔL_live]: **False**
- $v_c$ beats random on ΔE: **False**
- $v_c$ beats random on hit: **False**
- corr(−ΔL_live, ΔE) strong & positive: **True** (r=+0.53)
- Diagnostic closed: **False** (signal recovered; controller does not win)

**Interpretation:** Replacing template $M$ with live $B$ restores $\mathrm{corr}(-\Delta\mathcal L,\Delta E)$. But mean $E[-\Delta\mathcal L_{\mathrm{live}}]_{v_c}$ and $G_H$ do not beat controls — joint closed-loop control still fails, especially on H ($P(W\to C)=0.11$, $G_H=-0.50$). Note $G_C\equiv0$: C live site is the fixed task prompt (not steered-trajectory-conditioned).

No new $v$. No policy.
