# 8-way validation — frozen $v_c$ vs controls

> **Question A:** Does converted improve sync vs baseline/random?
> **Question B:** Reliable arbitrary $m^*$? (hard; not required)

α=1.5, reps=8/m*, order=C→H→O, task=api

## Aggregate

| Arm | P(hit) | P(hit₀) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) | E₀→E₃ |
|-----|--------|---------|-----|------|------|------|------|------|------|-------|
| none | 0.09 | 0.11 | +0.09 | +0.09 | +0.12 | -0.12 | 0.45 | 0.50 | 0.50 | 1.64→1.55→1.42→1.55 |
| predictive | 0.12 | 0.16 | -0.11 | -0.17 | +0.14 | -0.08 | 0.41 | 0.55 | 0.50 | 1.44→1.61→1.47→1.55 |
| converted | 0.11 | 0.11 | +0.06 | -0.02 | -0.02 | +0.09 | 0.47 | 0.56 | 0.50 | 1.53→1.55→1.56→1.47 |
| random | 0.19 | 0.16 | +0.00 | -0.22 | +0.25 | -0.03 | 0.52 | 0.52 | 0.56 | 1.41→1.62→1.38→1.41 |

## Per $m^*$ (converted vs none)

| $m^*$ | none P(hit) | converted P(hit) | none ΔE | converted ΔE |
|-------|-------------:|-----------------:|--------:|-------------:|
| 000 | 0.00 | 0.00 | +0.50 | +0.12 |
| 001 | 0.00 | 0.00 | +0.00 | -0.25 |
| 010 | 0.25 | 0.25 | +0.00 | +0.12 |
| 011 | 0.50 | 0.50 | -0.25 | -0.12 |
| 100 | 0.00 | 0.12 | +0.12 | +0.50 |
| 101 | 0.00 | 0.00 | +0.38 | +0.62 |
| 110 | 0.00 | 0.00 | +0.00 | -0.25 |
| 111 | 0.00 | 0.00 | +0.00 | -0.25 |

## Per $m^*$ (all arms P(hit))

| $m^*$ | none | predictive | converted | random |
|-------|-----:|-----:|-----:|-----:|
| 000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 001 | 0.00 | 0.00 | 0.00 | 0.00 |
| 010 | 0.25 | 0.25 | 0.25 | 0.38 |
| 011 | 0.50 | 0.75 | 0.50 | 0.88 |
| 100 | 0.00 | 0.00 | 0.12 | 0.00 |
| 101 | 0.00 | 0.00 | 0.00 | 0.00 |
| 110 | 0.00 | 0.00 | 0.00 | 0.00 |
| 111 | 0.00 | 0.00 | 0.00 | 0.25 |

## Gate

- Question A (converted useful): **False**
- Question B (reliable arbitrary): **False**
- converted ΔE > 0: **True**
- converted beats none (ΔE): **False**
- converted beats random (ΔE): **True**
- converted beats predictive (ΔE): **True**
- ranking $v_c ≥ v_p ≥$ rand/none: **False**

Directions frozen. No new $v$.
