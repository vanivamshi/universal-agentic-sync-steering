# Phase 9H — Live $\Gamma'$ (anti-stagnation) vs $\Gamma$ vs fixed

> $\Gamma'(a)=-\infty$ if $a=a_{t-1}\land S_t=S_{t-1}$; else $\Gamma(a\mid S_t,m^*)$. Frozen 8K. 9G kernel. No other memory.

Targets `['010', '011', '100', '101', '110']`, reps=4, max_steps=4, seed=20260927, 9G_extra=64.

## Aggregate

| policy | n | $P_{hit}$ | mean $\Delta E$ | $P(E\downarrow)$ | $P(\mathrm{repeat}|\mathrm{noop})$ | $P(\mathrm{rescue}|\mathrm{noop})$ |
|--------|--:|----------:|-----------------:|------------------:|-------------------------------:|-------------------------------:|
| **$\Gamma'$** | 20 | **0.60** | 1.20 | 0.60 | 0.20 | 0.80 |
| $\Gamma$ | 20 | **0.55** | 1.00 | 0.57 | 1.00 | 0.00 |
| fixed | 20 | **0.55** | 1.00 | 0.57 | 1.00 | 0.00 |

$\Delta P_{hit}(\Gamma'-\Gamma)$ = **0.05**
$\Delta P_{hit}(\Gamma'-\mathrm{fixed})$ = **0.05**

## Per $m^*$ $P_{hit}$

| $m^*$ | $\Gamma'$ | $\Gamma$ | fixed | $\Gamma'-\Gamma$ |
|-------|----------:|---------:|------:|-----------------:|
| `010` | 0.25 | 0.25 | 0.75 | **0.00** |
| `011` | 1.00 | 1.00 | 0.75 | **0.00** |
| `100` | 1.00 | 1.00 | 1.00 | **0.00** |
| `101` | 0.50 | 0.25 | 0.00 | **0.25** |
| `110` | 0.25 | 0.25 | 0.25 | **0.00** |

## Example trajectories (first rep each $m^*$)

- `010` Γ': `011→010` acts=['O'] hit=1 rep_noop=0/0
- `010` Γ: `011→010` acts=['O'] hit=1 rep_noop=0/0
- `010` f: `011→010` acts=['O'] hit=1 rep_noop=0/0
- `011` Γ': `010→011` acts=['O'] hit=1 rep_noop=0/0
- `011` Γ: `010→011` acts=['O'] hit=1 rep_noop=0/0
- `011` f: `010→011` acts=['O'] hit=1 rep_noop=0/0
- `100` Γ': `011→010→100` acts=['C', 'C'] hit=1 rep_noop=0/0
- `100` Γ: `011→010→100` acts=['C', 'C'] hit=1 rep_noop=0/0
- `100` f: `011→100` acts=['H'] hit=1 rep_noop=0/0
- `101` Γ': `100→011→100→011→011` acts=['O', 'C', 'O', 'C'] hit=0 rep_noop=0/0
- `101` Γ: `100→011→100→011→011` acts=['O', 'C', 'O', 'C'] hit=0 rep_noop=0/0
- `101` f: `100→011→111→011→011` acts=['O', 'H', 'H', 'H'] hit=0 rep_noop=0/0
- `110` Γ': `011→011→010→011→101` acts=['C', 'O', 'C', 'C'] hit=0 rep_noop=0/1
- `110` Γ: `011→011→001→111→011` acts=['C', 'C', 'H', 'O'] hit=0 rep_noop=1/1
- `110` f: `011→011→001→111→011` acts=['C', 'C', 'H', 'O'] hit=0 rep_noop=1/1

## Gate (critical signature)

- $P(\mathrm{repeat}|\mathrm{noop})$ $\Gamma'$ / $\Gamma$: **0.20** / 1.00 (less? **True**)
- $P_{hit}$ $\Gamma'$ / $\Gamma$: **0.60** / 0.55 (up? **True**)
- $P(\mathrm{rescue}|\mathrm{noop})$ $\Gamma'$ / $\Gamma$: 0.80 / 0.00

Critical: P(repeat|noop)_Γ' < P(repeat|noop)_Γ and P(hit)_Γ' > P(hit)_Γ. Then expand kernel for remaining failures → full 8 targets.

$$\boxed{\Gamma'=\Gamma+\text{one-step no-op memory}}$$
