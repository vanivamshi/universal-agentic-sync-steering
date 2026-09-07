# 8-way surrogate — $\mathcal{L}_{8\mathrm{way}}$ diagnostic

> $\mathcal{L}=\sum_k\log(1+e^{-\beta(2m_k^*-1)M_k})$. Primary: $\mathbb{E}[-\Delta\mathcal{L}]$.

mode=**fast**, β=1.0, α=1.5, reps=2/m*, arms=['none', 'predictive', 'converted', 'random'].

## Aggregate

| Arm | n | E[−ΔL] | frac L↓ | ΔE | P(hit) | corr(−ΔL,ΔE) | Δ~M_C | Δ~M_H | Δ~M_O |
|-----|---|--------|---------|-----|--------|--------------|-------|-------|-------|
| none | 16 | +0.000 | 0.00 | +0.000 | 0.188 | +nan | +0.00 | +0.00 | +0.00 |
| predictive | 16 | +0.061 | 0.38 | +0.125 | 0.188 | +0.039 | -0.10 | +0.02 | +0.03 |
| converted | 16 | +2.075 | 0.88 | +0.000 | 0.125 | -0.011 | +1.80 | +6.32 | +0.97 |
| random | 16 | -0.154 | 0.00 | +0.188 | 0.125 | -0.309 | -0.10 | -0.18 | -0.01 |

Pooled corr(−ΔL, ΔE) steered arms: **-0.075**
Pooled corr all arms: **-0.060**

## Gate

- $v_c$ beats random on E[−ΔL]: **True**
- $v_c$ beats none on E[−ΔL]: **True**
- $v_c$ beats predictive on E[−ΔL]: **True**
- $v_c$ beats random on hit: **False**
- Surrogate advantage stronger than hit: **True**
- corr strong (|r|≥0.3 steered): **False**

If vc >> random on E[-ΔL] but not on hit: discrete eval understates causal joint control. If both weak: problem is downstream of converted representation.

Directions frozen. No product objective. No policy.
