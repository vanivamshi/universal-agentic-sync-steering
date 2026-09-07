# Phase 4 — Pairwise composition matrix

> **Hypothesis:** If individual predictive→causal conversions are valid,
> their converted components should compose with limited cross-channel interference.

Controls: **converted** vs **predictive** vs **random**. Sites frozen.

## Jacobian diagonal dominance

$$\mathrm{DD}_k=\frac{|J_{kk}|}{\sum_{j\neq k}|J_{kj}|}$$

| Arm | class | DD_C | DD_H | DD_O |
|-----|-------|------|------|------|
| predictive | coupled_composition | 1.75 | 0.04 | 0.56 |
| converted | independent_composition | 5.41 | 3.65 | 5.68 |
| random | coupled_composition | 0.76 | 0.48 | 1.01 |

### Converted J (dM/α)

| affected\\steer | C | H | O |
|------------------|---|---|---|
| **C** | +3.00 | +0.17 | -0.38 |
| **H** | +0.98 | +11.24 | -2.10 |
| **O** | -0.11 | -0.14 | +1.44 |

## Pairwise intended |ΔM| (mean over e-patterns)

| Pair | question | converted | predictive | random | conv>pred |
|------|----------|-----------|------------|--------|-----------|
| C+H | Can earlier plan-state intervention compose with… | 8.08 | 0.28 | 0.57 | True |
| H+O | Does controlling execution preserve/control subs… | 7.30 | 0.69 | 0.35 | True |
| C+O | Can non-adjacent channels compose without H medi… | 2.07 | 0.10 | 0.08 | True |

## Collateral |ΔM| on the held-out channel

| Pair | converted | predictive | random |
|------|-----------|------------|--------|
| C+H | 0.18 | 0.05 | 0.08 |
| H+O | 0.56 | 0.08 | 0.05 |
| C+O | 2.04 | 1.35 | 1.08 |

## Gate

- Converted beats predictive on intended |ΔM|: **3/3** pairs
- Converted Jacobian class: **independent_composition**
- Ready for sequential C→H→O: **True**
- Ready for 8-way: **False**

If coupled: characterize coupling next — do not jump to 8-state controller.
