# Phase 9O — Finite-horizon target-hitting planner

> $V_k(s)=\max_a\mathbb E[V_{k-1}(s')]$; $V_0=1[s=m^*]$. Hard $m^*\in\{000,001,110\}$. Temporary $E\uparrow$ allowed. No new $v$.

T=3, reps=4, seed=20261003, kernel_cells=17.

## Offline recommended paths (T=3)

### $m^*=000$
- from `101`: `101 → 111 → 011 → 100` acts=['H', 'H', 'H'] $V$=0.00

### $m^*=001$
- from `011`: `011 → 001` acts=['C'] $V$=0.16
- producers: `011|C`→0.10

### $m^*=110$
- from `101`: `101 → 111 → 110` acts=['H', 'C'] $V$=0.15
- producers: `111|C`→0.18, `011|H`→0.04

## Live: $\Gamma'$ vs hitting

| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\Delta E$ | mean # $E\uparrow$ |
|--------|--:|----------:|-----------:|-----------------:|--------------------:|
| **hitting** | 12 | 0.08 | **0.08** | -0.17 | 0.92 |
| $\Gamma'$ | 12 | 0.17 | **0.17** | 0.42 | 0.67 |

| $m^*$ | $P_{ever}$ hit | $P_{ever}$ $\Gamma'$ | $P_{hit}$ hit |
|-------|---------------:|---------------------:|-------------:|
| `000` | **0.00** | 0.00 | 0.00 |
| `001` | **0.00** | 0.00 | 0.00 |
| `110` | **0.25** | 0.50 | 0.25 |

## Gate

- Hard ever under hitting: **1/3** (Γ' 1/3)
- Hard gate (all 3): **False**
- $\Delta P_{\mathrm{ever}}$(hit$-\Gamma'$): -0.08

Primary: ∀ hard m* P(ever)>0 under hitting. Secondary: temporary E↑ usage; vs Γ'.

$$\boxed{a^*=\arg\max_a\mathbb E[V_{T-1}(S')]\quad\text{(hitting, not }\Delta E\text{)}}$$
