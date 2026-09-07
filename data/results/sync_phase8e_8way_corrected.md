# Phase 8E — 8-way with corrected (8D) controller

> Before/after vs legacy converted: $P(\mathrm{hit})=0.11$, $\Delta E=+0.06$.

Controller: $d_k=(2m_k^*-1)v_k$, H = decision-token-only, same seed, skip-noop.

α=1.5, reps=2/m*, order=C→H→O, task=api. No new $v$.

## Aggregate

| Arm | P(hit) | P(hit₀) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) | E₀→E₃ |
|-----|--------|---------|-----|------|------|------|------|------|------|-------|
| none | 0.06 | 0.06 | +0.00 | +0.00 | +0.00 | +0.00 | 0.56 | 0.50 | 0.75 | 1.19→1.19→1.19→1.19 |
| predictive | 0.19 | 0.12 | +0.31 | +0.00 | +0.19 | +0.12 | 0.56 | 0.69 | 0.56 | 1.50→1.50→1.31→1.19 |
| converted | 0.06 | 0.00 | +0.31 | +0.12 | +0.19 | +0.00 | 0.56 | 0.62 | 0.44 | 1.69→1.56→1.38→1.38 |
| random | 0.38 | 0.25 | +0.31 | +0.19 | +0.19 | -0.06 | 0.69 | 0.69 | 0.62 | 1.31→1.12→0.94→1.00 |

| **legacy converted** | **0.11** | — | **+0.06** | — | — | — | — | — | — | — |

## Per $m^*$ $P(\mathrm{hit}\mid m^*)$

| $m^*$ | none | predictive | converted | random |
|-------|-----:|-----:|-----:|-----:|
| 000 | 0.00 | 0.50 | 0.00 | 0.00 |
| 001 | 0.00 | 0.00 | 0.00 | 0.00 |
| 010 | 0.00 | 0.00 | 0.50 | 0.50 |
| 011 | 0.50 | 1.00 | 0.00 | 1.00 |
| 100 | 0.00 | 0.00 | 0.00 | 0.50 |
| 101 | 0.00 | 0.00 | 0.00 | 0.00 |
| 110 | 0.00 | 0.00 | 0.00 | 0.00 |
| 111 | 0.00 | 0.00 | 0.00 | 1.00 |

## Per $m^*$ ΔE (converted vs none)

| $m^*$ | none ΔE | converted ΔE | none P(hit) | converted P(hit) |
|-------|--------:|-------------:|------------:|-----------------:|
| 000 | +0.00 | +1.00 | 0.00 | 0.00 |
| 001 | +0.00 | +0.00 | 0.00 | 0.00 |
| 010 | +0.00 | +1.00 | 0.00 | 0.50 |
| 011 | +0.00 | +1.00 | 0.50 | 0.00 |
| 100 | +0.00 | -1.00 | 0.00 | 0.00 |
| 101 | +0.00 | +1.00 | 0.00 | 0.00 |
| 110 | +0.00 | +0.00 | 0.00 | 0.00 |
| 111 | +0.00 | -0.50 | 0.00 | 0.00 |

## Gate

- Converted P(hit): **0.0625** (legacy 0.11)
- Converted ΔE: **0.3125** (legacy +0.06)
- Beats legacy hit: **False**
- Beats legacy ΔE: **True**
- Beats none (ΔE): **True**
- Beats random (ΔE): **False**
- Beats predictive (ΔE): **False**
- Question A (corrected 8-way useful): **True**

Before/after: legacy converted P(hit)=0.11 ΔE=+0.06. Corrected 8-way uses 8D controller only — no new v.

Claim scope: reps=2 — directional before/after, not a stable rate claim.
