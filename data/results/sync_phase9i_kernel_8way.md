# Phase 9I — Kernel expand + full 8-way $\Gamma'$

> Canonical planner $\Gamma'=\Gamma+$no-op memory. Frozen 8K. Primary criterion: $P(\exists t:S_t=m^*)>0$ for all $m^*$.

expand_reps=3, 8way_reps=2, max_steps=4, seed=20260928.

## Hard-path kernel coverage $(s,a)$

| cell | n |
|------|--:|
| `101|C` | 8 |
| `101|H` | 7 |
| `101|O` | 10 |
| `110|H` | 2 |
| `110|O` | 1 |
| `111|C` | 3 |
| `111|H` | 2 |
| `111|O` | 9 |

### New 9I densify

| cell | n | $P_{shift}$ | hist |
|------|--:|------------:|------|
| `010|H` | 3 | 1.00 | 111:2 100:1 |
| `011|H` | 3 | 1.00 | 100:1 111:1 101:1 |
| `100|C` | 2 | 0.50 | 100:1 011:1 |
| `101|C` | 6 | 0.67 | 100:3 101:2 111:1 |
| `101|H` | 5 | 0.80 | 111:3 101:1 010:1 |

## Full 8-way aggregate

| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\Delta E$ | $P(E\downarrow)$ | $P(\mathrm{repeat}|\mathrm{noop})$ |
|--------|--:|----------:|-----------:|-----------------:|------------------:|-------------------------------:|
| **$\Gamma'$** | 16 | 0.44 | **0.44** | 0.94 | 0.49 | 0.62 |
| $\Gamma$ | 16 | 0.44 | **0.44** | 0.94 | 0.49 | 1.00 |
| fixed | 16 | 0.38 | **0.38** | 0.81 | 0.34 | 1.00 |

## Per $m^*$: $P_{ever}$ / $P_{hit}$ ($\Gamma'$)

| $m^*$ | $P_{ever}\Gamma'$ | $P_{hit}\Gamma'$ | $P_{ever}\Gamma$ | $P_{ever}$ fixed |
|-------|-------------------:|-----------------:|-----------------:|-----------------:|
| `000` | 0.00 | 0.00 | 0.00 | 0.00 |
| `001` | 0.00 | 0.00 | 0.00 | 0.00 |
| `010` | 0.50 | 0.50 | 0.50 | 0.50 |
| `011` | 1.00 | 1.00 | 1.00 | 0.50 |
| `100` | 0.50 | 0.50 | 0.50 | 1.00 |
| `101` | 0.50 | 0.50 | 0.50 | 0.00 |
| `110` | 0.00 | 0.00 | 0.00 | 0.00 |
| `111` | 1.00 | 1.00 | 1.00 | 1.00 |

$m^*$ with $P_{ever}>0$: $\Gamma'$ **5/8**, $\Gamma$ 5/8, fixed 4/8.

## Gate

- Universal ever-reach under $\Gamma'$: **False**
- Hard ever-reach $000/001/111$: `{'000': False, '001': False, '111': True}`
- $\Delta P_{hit}(\Gamma'-\Gamma)$: 0.00
- $\Delta P_{ever}(\Gamma'-\Gamma)$: 0.00

Primary: ∀m* P(ever reach)>0 under Γ'. Secondary: final P(hit) and Γ' vs Γ vs fixed. No extra memory.

$$\boxed{\text{causal actuators}+\Gamma+\text{one-step memory}\rightarrow\text{8-way multi-step reachability}}$$
