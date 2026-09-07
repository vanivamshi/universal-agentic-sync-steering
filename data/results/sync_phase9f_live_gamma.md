# Phase 9F — Live $\Gamma$-policy vs fixed $H\to C\to O$ (scoped)

> Frozen 8K actuators. Only action selection changes. $a^*=\arg\max_{a\in\mathrm{relevant}(s,m^*)}\Gamma(a\mid s,m^*)$ (empirical, no tuning).

Targets: `['010', '011', '100', '101', '110']`. reps=4, max_steps=4, seed=20260925. Paired free $S_0$.

## Aggregate

| policy | n | $P_{hit}$ | mean $\Delta E$ | mean $E_f$ | mean $P(E\downarrow)$ | mean step $\Delta E$ |
|--------|--:|----------:|-----------------:|-----------:|----------------------:|----------------------:|
| **$\Gamma$** | 20 | **0.65** | **1.45** | 0.70 | 0.62 | -1.12 |
| fixed | 20 | 0.40 | 0.75 | 1.40 | 0.45 | -0.39 |

$\Delta P_{hit}(\Gamma-\mathrm{fixed})$ = **0.25**

## Per $m^*$

| $m^*$ | $P_{hit}\Gamma$ | $P_{hit}$ fixed | $\Delta$ | mean $\Delta E$ $\Gamma$ | mean $\Delta E$ fixed |
|-------|----------------:|----------------:|---------:|------------------------:|-----------------------:|
| `010` | 0.25 | 0.50 | **-0.25** | 0.50 | 0.75 |
| `011` | 1.00 | 0.25 | **0.75** | 2.00 | 0.50 |
| `100` | 1.00 | 0.50 | **0.50** | 2.75 | 1.50 |
| `101` | 0.75 | 0.50 | **0.25** | 2.00 | 1.00 |
| `110` | 0.25 | 0.25 | **0.00** | 0.00 | 0.00 |

## Example trajectories (first rep each $m^*$)

- `010` $S_0$=`100`: $\Gamma$ `100→100→100→100→100` acts=['C', 'C', 'C', 'C'] hit=0; fixed `100→100→111→100→010` acts=['H', 'H', 'C', 'H'] hit=1
- `011` $S_0$=`100`: $\Gamma$ `100→011` acts=['O'] hit=1; fixed `100→100→100→100→100` acts=['H', 'H', 'H', 'H'] hit=0
- `100` $S_0$=`010`: $\Gamma$ `010→101→011→010→100` acts=['C', 'O', 'C', 'C'] hit=1; fixed `010→111→101→011→011` acts=['H', 'H', 'O', 'H'] hit=0
- `101` $S_0$=`011`: $\Gamma$ `011→011→100→011→100` acts=['C', 'C', 'O', 'C'] hit=0; fixed `011→111→111→100→011` acts=['H', 'H', 'H', 'O'] hit=0
- `110` $S_0$=`011`: $\Gamma$ `011→100→010→011→001` acts=['C', 'H', 'C', 'C'] hit=0; fixed `011→100→010→011→001` acts=['C', 'H', 'C', 'C'] hit=0

## Gate

- $P_{hit}$ $\Gamma$ / fixed: **0.65** / 0.40
- mean $\Delta E$ $\Gamma$ / fixed: **1.45** / 0.75
- mean $P(E\downarrow)$ $\Gamma$ / fixed: 0.62 / 0.45
- pairs $\Gamma$-only / fixed-only / both hit: 7 / 2 / 6

Scoped live test only. Controllers share free S0; differ only in single-channel action selection. Expand kernel before 000/001/111/full 8-way.

$$\boxed{\text{live }\Gamma\text{-selection vs fixed — scoped targets, frozen actuators}}$$
