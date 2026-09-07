# Phase 8K — High-rep stratified 8-way validation

> **Frozen controller:** $H\to C\to O$; $C$ stem $\alpha{=}5$, $H$ decision-token $\alpha{=}1.5$, $O$ early-FINAL $\alpha{=}1.5$. No skip. No new $v$.

reps=8/m*, n=64, seed=20260920.

## Aggregate

| Arm | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |
|-----|--------|---:|-----:|-----:|-----:|------|------|------|
| **8K validated** | 0.20 | +0.36 | +0.19 | +0.17 | +0.00 | 0.64 | 0.58 | 0.64 |
| 8K exploratory (reps=2) | 0.25 | +0.44 | +0.25 | — | — | 0.69 | 0.69 | 0.56 |
| legacy converted | 0.11 | +0.06 | — | — | — | 0.47 | 0.56 | 0.50 |
| 8E corrected | 0.06 | +0.31 | — | — | — | 0.56 | 0.62 | 0.44 |

## Per $m^*$

| $m^*$ | P(hit) | n_hit/n | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) | legacy P(hit) |
|------:|-------:|--------:|---:|-----:|-----:|-----:|------|------|------|-------------:|
| 000 | 0.00 | 0/8 | -0.25 | +0.25 | -0.50 | +0.00 | 0.00 | 0.50 | 0.75 | 0.00 |
| 001 | 0.00 | 0/8 | -0.25 | +0.12 | -0.50 | +0.12 | 0.25 | 0.62 | 0.50 | 0.00 |
| 010 | 0.25 | 2/8 | +0.12 | +0.12 | +0.12 | -0.12 | 0.75 | 0.88 | 0.38 | 0.25 |
| 011 | 0.50 | 4/8 | +0.12 | +0.00 | +0.12 | +0.00 | 0.75 | 1.00 | 0.75 | 0.50 |
| 100 | 0.62 | 5/8 | +1.38 | +0.25 | +1.25 | -0.12 | 0.75 | 0.75 | 0.75 | 0.12 |
| 101 | 0.25 | 2/8 | +1.38 | +0.50 | +0.88 | +0.00 | 1.00 | 0.50 | 0.75 | 0.00 |
| 110 | 0.00 | 0/8 | +0.50 | +0.38 | +0.00 | +0.12 | 0.75 | 0.25 | 0.88 | 0.00 |
| 111 | 0.00 | 0/8 | -0.12 | -0.12 | +0.00 | +0.00 | 0.88 | 0.12 | 0.38 | 0.00 |

## Gate

- Beats legacy P(hit): **True**
- Beats 8E P(hit): **True**
- $\Delta E_C$ non-negative: **True**
- $m^*$ reachable ($P>0$): **4/8**
- $m^*$ meaningful ($P\ge0.25$): **4/8**
- Improved-controller claim: **True**
- Universal 8-way claim: **False** (supported only if 8/8 meaningful: False)

Supported wording: *frozen three-channel causal controller improves 8-way synchronization and removes C-stage regression.*

Report per-m* P(hit|m*). Do not call aggregate P(hit) universal. Universal only if every m* is meaningfully reachable.
