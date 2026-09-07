# Phase 9E — Offline $\Gamma$-planner simulation

> Existing 9B/9C transitions only. No new $v$. $a^*=\arg\max_a\Gamma(a\mid s,m^*)$ vs fixed $H\to C\to O$.

Kernel: **96** transitions across **17** $(s,a)$ cells. Rollouts: 64/pair, max_steps=6.

Coverage gap as $s_0$: `['000', '001']`

## Striking $\Gamma$ sign flip (same $s$, different $m^*$)

| $s$ | $m^*$ | $\Gamma_C$ | $\Gamma_H$ | $\Gamma_O$ | $a^*$ |
|-----|-------|-----------:|-----------:|-----------:|------|
| `010` | `000` | -0.56 (n=9) | -0.78 (n=9) | -1.00 (n=1) | **C** |
| `010` | `111` | 0.56 (n=9) | 0.78 (n=9) | 1.00 (n=1) | **O** |
| `100` | `000` | -0.57 (n=14) | -0.46 (n=13) | -0.80 (n=10) | **H** |
| `100` | `111` | 0.57 (n=14) | 0.46 (n=13) | 0.80 (n=10) | **O** |

## Offline reachability by $m^*$ (best $s_0$ in kernel)

| $m^*$ | best $P_{hit}$ $\Gamma$ | from $s_0$ | best $P_{hit}$ fixed | from $s_0$ | mean $\Delta P_{hit}$ |
|-------|-------------------------:|-----------:|---------------------:|-----------:|----------------------:|
| `000` | 0.00 | `010` | 0.00 | `010` | 0.00 |
| `001` | 0.00 | `010` | 0.00 | `010` | 0.00 |
| `010` | 1.00 | `110` | 0.16 | `100` | 0.26 |
| `011` | 1.00 | `010` | 1.00 | `010` | 0.51 |
| `100` | 1.00 | `101` | 0.97 | `011` | 0.31 |
| `101` | 0.86 | `010` | 0.38 | `010` | 0.68 |
| `110` | 0.33 | `011` | 0.00 | `010` | 0.18 |
| `111` | 0.52 | `101` | 0.95 | `100` | -0.65 |

Targets with any offline hit: $\Gamma$ **6/8**, fixed **5/8**.

## Policy comparison (all simulated pairs)

- mean $\Delta P_{\mathrm{hit}}(\Gamma-\mathrm{fixed})$ = **0.153**
- pairs $\Gamma$ wins / fixed wins / ties: **20** / **7** / **15** (of 42)

### Top $\Gamma$ improvements

| $s_0$ | $m^*$ | $P_{hit}\Gamma$ | $P_{hit}$ fixed | $\Delta$ |
|-------|-------|----------------:|----------------:|---------:|
| `110` | `010` | 1.00 | 0.00 | **1.00** |
| `110` | `011` | 1.00 | 0.00 | **1.00** |
| `111` | `100` | 0.89 | 0.00 | **0.89** |
| `111` | `101` | 0.81 | 0.00 | **0.81** |
| `100` | `101` | 0.77 | 0.00 | **0.77** |
| `011` | `101` | 0.69 | 0.00 | **0.69** |
| `110` | `101` | 0.78 | 0.14 | **0.64** |
| `101` | `011` | 1.00 | 0.42 | **0.58** |

### Top fixed wins (Γ worse)

| $s_0$ | $m^*$ | $P_{hit}\Gamma$ | $P_{hit}$ fixed | $\Delta$ |
|-------|-------|----------------:|----------------:|---------:|
| `100` | `111` | 0.00 | 0.95 | -0.95 |
| `010` | `111` | 0.00 | 0.88 | -0.88 |
| `011` | `111` | 0.00 | 0.69 | -0.69 |
| `101` | `111` | 0.52 | 0.95 | -0.44 |
| `110` | `111` | 0.50 | 0.81 | -0.31 |

## Retrospective: would $\Gamma$ have chosen differently?

- comparisons: 672
- $P(\mathrm{agree})$: 0.34
- $P(\Gamma$ better mean $\Delta E)$: **0.61**
- $P(\Gamma$ worse mean $\Delta E)$: 0.02

## Gate

- mean $\Delta P_{hit}$: **0.153**
- $m^*$ reachable offline ($\Gamma$ / fixed): **6/8** / **5/8**
- coverage gaps as $s_0$: `['000', '001']`

Offline only. Positive mean ΔP_hit favors deploying Γ-choice in-loop next; coverage gaps (e.g. 000/001 as s0) limit claims of universal 8-way.

$$\boxed{a^*=\arg\max_a\Gamma(a\mid s,m^*)\text{ — planner change, not a new actuator}}$$
