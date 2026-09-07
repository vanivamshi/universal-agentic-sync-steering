# Phase 9M — Target hold vs planner

> After acquire: **hold** (no steer) vs **planner** (stay-seek). Frozen hybrid acquisition. No new $v$.

reps=4, acq_steps=7, n_hold=2, seed=20261001.

$$\pi(s,m^*)=\mathrm{acquire}\ [s\neq m^*];\quad \mathrm{hold}\ [s=m^*]$$

## Diagnostic: $P_{stay}\mid acq$

| $m^*$ | $P_{acq}$ | $P_{stay}$ hold | $P_{stay}$ planner | $\Delta$ (hold$-$plan) | $P_{final}$ hold | $P_{final}$ plan |
|-------|----------:|-----------------:|-------------------:|-----------------------:|-----------------:|-----------------:|
| `000` | 0.50 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| `001` | 0.00 | **—** | — | — | 0.00 | 0.00 |
| `010` | 1.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| `011` | 1.00 | **0.25** | 0.50 | -0.25 | 0.25 | 0.50 |
| `100` | 1.00 | **0.25** | 0.75 | -0.50 | 0.25 | 0.75 |
| `101` | 1.00 | **0.00** | 0.25 | -0.25 | 0.00 | 0.25 |
| `110` | 0.00 | **—** | — | — | 0.00 | 0.00 |
| `111` | 0.50 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 |

mean $P_{stay}$ hold / planner: **0.08** / 0.25
Verdict: **mixed / inconclusive**

## Gate

- hold ≫ planner: **False**
- ran acquire+hold 8-way: **False**

If hold≫planner: stop acting at target. If both≈0: need retention mechanism beyond no-steer.

$$\boxed{s=m^*\Rightarrow\mathrm{hold};\quad s\neq m^*\Rightarrow\mathrm{acquire}}$$
