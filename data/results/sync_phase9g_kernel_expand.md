# Phase 9G — Kernel expansion + anti-stagnation

> No new $v$. Offline $\Gamma'$ safeguard, then densify sticky $(s,a)$ around $100/010/011$.

$$\Gamma'(a)=-\infty \text{ if } a=a_{t-1}\land S_t=S_{t-1};\quad \Gamma(a)\text{ else.}$$

## Offline anti-stagnation (on 9F gamma trajectories)

| kernel | sticky raw repeats | rescues | better mean $\Delta E$ | $P(\mathrm{disagree})$ | verdict |
|--------|-------------------:|--------:|------------------------:|-------------------------:|---------|
| pre-expand | 7 | 5 | 0 | 0.09 | anti-stagnation breaks sticky repeats (live-relevant) |
| post-expand | 7 | 5 | 0 | 0.09 | anti-stagnation breaks sticky repeats (live-relevant) |

### Rescue examples (raw wanted to repeat after no-op)

- $m^*$=`010` $s$=`100`: raw=`C` → anti=`H` (mean $\Delta E$ -0.29 → 0.00)
- $m^*$=`010` $s$=`100`: raw=`C` → anti=`H` (mean $\Delta E$ -0.29 → 0.00)
- $m^*$=`010` $s$=`100`: raw=`C` → anti=`H` (mean $\Delta E$ -0.29 → 0.00)
- $m^*$=`100` $s$=`011`: raw=`C` → anti=`H` (mean $\Delta E$ -1.75 → -1.50)
- $m^*$=`101` $s$=`011`: raw=`C` → anti=`H` (mean $\Delta E$ -1.25 → -0.50)

## Focus: $\Gamma(a\mid 100, m^*=010)$

| kernel | $\Gamma_C$ | $\Gamma_H$ | $\Gamma_O$ | $a^*$ |
|--------|-----------:|-----------:|-----------:|------|
| pre | 0.29 (n=14) | -0.08 (n=13) | 0.80 (n=10) | **O** |
| post | 0.27 (n=22) | 0.05 (n=21) | 0.78 (n=18) | **O** |

## Densified cells

| cell | n | $P_{shift}$ | after hist |
|------|--:|------------:|------------|
| `010|C` | 5 | 0.80 | 100:2 101:2 010:1 |
| `010|H` | 5 | 0.60 | 010:2 111:2 011:1 |
| `010|O` | 6 | 0.67 | 010:2 011:4 |
| `011|C` | 8 | 1.00 | 001:1 101:3 100:3 010:1 |
| `011|H` | 8 | 0.88 | 100:4 111:2 011:1 110:1 |
| `011|O` | 8 | 0.38 | 100:3 011:5 |
| `100|C` | 8 | 0.38 | 100:5 111:1 011:2 |
| `100|H` | 8 | 0.25 | 100:6 011:2 |
| `100|O` | 8 | 0.88 | 010:1 011:5 100:1 111:1 |

## Diagnosis: wrong $\Gamma$ vs missing memory

`100|C`: $P_{\mathrm{shift}}=0.38$ ⇒ **$P(\mathrm{stay})=0.62$**. When it moves ($100\to011$), $E$ to $010$ drops, so $\Gamma_C$ stays positive — the planner is not simply “wrong,” but **blind to no-ops**. That is exactly what $\Gamma'$ (anti-stagnation) fixes: after $S_{t+1}=S_t$, reject repeating $C$ and take $H$ (the action fixed used to rescue).

## Gate

- Anti-stag promising for live: **True** (5 sticky rescues $C\to H$ on 9F paths)
- Extra transitions: **64**
- Do **not** full 8-way yet — next is live $\Gamma'$ on scoped targets

Anti-stag switches C→H after no-op at 100 (matches fixed's successful first action). Densify sticky cells; then live Γ' before full 8-way.

$$\boxed{\text{stagnation is real }(P_{\mathrm{stay}}\approx0.62\text{ on }100|C);\text{ use }\Gamma'\text{ next}}$$
