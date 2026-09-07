# Phase 9J — Sink-seeking for hard targets

> $a^*=\arg\max_a\big[P(S'=m^*\mid s,a)+\lambda\Gamma'(a)\big]$. First test $\lambda=0.0$. Frozen 8K. No new $v$.

expand_reps=8, hard_reps=4, max_steps=7, seed=20260929.

## Producer landing estimates

| cell | target | n | $P(\mathrm{land})$ | hist |
|------|--------|--:|--------------------:|------|
| `011|C` | `001` | 20 | **0.10** | 001:2 100:8 111:1 101:6 010:1 |
| `011|H` | `000` | 23 | **0.00** | 100:11 011:3 111:7 110:1 101:1 |
| `100|O` | `110` | 26 | **0.00** | 011:17 100:5 010:2 111:2 |
| `111|C` | `110` | 11 | **0.18** | 011:1 100:4 101:2 110:2 111:1 |
| `111|H` | `110` | 10 | **0.00** | 111:3 010:2 011:5 |

## Hard-target live: $\Gamma'$ vs sink-seek

| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\Delta E$ | $P(\mathrm{stay}\mid\mathrm{land})$ |
|--------|--:|----------:|-----------:|-----------------:|---------------------------------------:|
| **sink-seek** | 12 | 0.00 | **0.08** | -0.08 | 0.00 |
| $\Gamma'$ | 12 | 0.08 | **0.08** | -0.25 | — |

### Per hard $m^*$

| $m^*$ | $P_{ever}$ SS | $P_{hit}$ SS | $P_{ever}$ $\Gamma'$ | $P_{hit}$ $\Gamma'$ | stay|land SS |
|-------|---------------:|-------------:|---------------------:|---------------------:|-------------------:|
| `000` | **0.25** | 0.00 | 0.00 | 0.00 | 0.00 |
| `001` | **0.00** | 0.00 | 0.00 | 0.00 | — |
| `110` | **0.00** | 0.00 | 0.25 | 0.25 | — |

### Retention exits (sink-seek landings)

- `000`: H→`010` stay=0
- `001`: no landings logged
- `110`: no landings logged

## Hard gate

- $P_{\mathrm{ever}}>0$ for all three under sink-seek: **False** (1/3; $\Gamma'$ 1/3)
- $\Delta$ aggregate $P_{\mathrm{ever}}$(SS$-\Gamma'$): 0.00

## Full 8-way

_Skipped — hard gate failed (use `--force-8way` to override)._

Primary hard gate: ∀m*∈{000,001,110} P(ever)>0 under sink-seek. Retention P(stay|land) is secondary. Full 8-way only if hard gate holds.

$$\boxed{a^*=\arg\max_a P(S'=m^*\mid s,a)\quad(\lambda=0)\text{ for hard }m^*}$$
