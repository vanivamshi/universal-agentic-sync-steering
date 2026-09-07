# 8-way validation — frozen $v_c$ vs controls

> **Question A:** Does converted improve sync vs baseline/random?
> **Question B:** Reliable arbitrary $m^*$? (hard; not required)

α=1.5, reps=2/m*, order=C→H→O, task=api

## Aggregate

| Arm | P(hit) | P(hit₀) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) | E₀→E₃ |
|-----|--------|---------|-----|------|------|------|------|------|------|-------|
| none | 0.12 | 0.19 | +0.06 | +0.12 | -0.12 | +0.06 | 0.56 | 0.50 | 0.44 | 1.56→1.44→1.56→1.50 |
| predictive | 0.06 | 0.19 | -0.12 | -0.06 | +0.25 | -0.31 | 0.44 | 0.38 | 0.56 | 1.50→1.56→1.31→1.62 |
| converted | 0.12 | 0.00 | +0.25 | +0.19 | +0.25 | -0.19 | 0.56 | 0.44 | 0.56 | 1.69→1.50→1.25→1.44 |
| random | 0.31 | 0.25 | +0.19 | -0.06 | +0.06 | +0.19 | 0.62 | 0.62 | 0.56 | 1.38→1.44→1.38→1.19 |

## Per $m^*$ (converted vs none)

| $m^*$ | none P(hit) | converted P(hit) | none ΔE | converted ΔE |
|-------|-------------:|-----------------:|--------:|-------------:|
| 000 | 0.00 | 0.00 | +0.00 | +0.50 |
| 001 | 0.00 | 0.00 | +0.50 | -0.50 |
| 010 | 0.00 | 0.00 | +0.50 | +0.50 |
| 011 | 0.50 | 1.00 | +1.00 | +2.00 |
| 100 | 0.00 | 0.00 | -1.50 | -0.50 |
| 101 | 0.50 | 0.00 | +1.00 | -0.50 |
| 110 | 0.00 | 0.00 | +0.00 | +0.50 |
| 111 | 0.00 | 0.00 | -1.00 | +0.00 |

## Per $m^*$ (all arms P(hit))

| $m^*$ | none | predictive | converted | random |
|-------|-----:|-----:|-----:|-----:|
| 000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 001 | 0.00 | 0.00 | 0.00 | 0.00 |
| 010 | 0.00 | 0.00 | 0.00 | 1.00 |
| 011 | 0.50 | 0.00 | 1.00 | 1.00 |
| 100 | 0.00 | 0.00 | 0.00 | 0.00 |
| 101 | 0.50 | 0.50 | 0.00 | 0.50 |
| 110 | 0.00 | 0.00 | 0.00 | 0.00 |
| 111 | 0.00 | 0.00 | 0.00 | 0.00 |

## Gate

- Question A (converted useful): **Partial** — best ΔE (+0.25); hit tied with none, below random
- Question B (reliable arbitrary): **False**
- converted ΔE > 0: **True**
- converted beats none (ΔE): **True**
- converted beats random (ΔE): **True** (hit: **False** — random 0.31 > 0.12)
- converted beats predictive (ΔE / hit): **True** / **True**
- ranking $v_c ≥ v_p ≥$ rand/none: **False** (random hit dominates)

Directions frozen. No new $v$. n=2/cell — hit ranking noisy.
