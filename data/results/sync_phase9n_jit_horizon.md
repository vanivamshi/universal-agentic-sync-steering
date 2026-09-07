# Phase 9N — Just-in-time / receding-horizon control

> Same frozen $v_c$/α. No hold. Intervene at decision sites ($C\to H\to O$, O last).

reps=4, rh_max_steps=3, seed=20261002.

## Aggregate

| arm | n | $P_{hit}$ | $P_{ever}$ | mean $\Delta E$ | $P_C$ | $P_H$ | $P_O$ | ever $m^*$ |
|-----|--:|----------:|-----------:|-----------------:|------:|------:|------:|-----------:|
| **JIT-RH** | 32 | **0.38** | 0.38 | 0.72 | 0.81 | 0.66 | 0.59 | **5/8** |
| JIT-oneshot | 32 | **0.09** | 0.09 | 0.41 | 0.62 | 0.62 | 0.50 | **2/8** |
| compose 8K | 32 | **0.19** | 0.19 | 0.53 | 0.75 | 0.69 | 0.44 | **4/8** |

## Per $m^*$ $P_{hit}$

| $m^*$ | JIT-RH | oneshot | compose |
|-------|-------:|--------:|--------:|
| `000` | 0.00 | 0.00 | 0.00 |
| `001` | 0.00 | 0.00 | 0.00 |
| `010` | 0.50 | 0.25 | 0.00 |
| `011` | 0.75 | 0.50 | 0.25 |
| `100` | 0.75 | 0.00 | 0.75 |
| `101` | 0.50 | 0.00 | 0.25 |
| `110` | 0.00 | 0.00 | 0.25 |
| `111` | 0.50 | 0.00 | 0.00 |

## Gate

- Best arm: **jit_rh** ($P_{hit}$=0.38)
- JIT beats compose: **True**
- Universal hit ($P_{hit}>0$ all $m^*$): **False** (5/8)

Primary: P_hit by arm. If JIT/RH ≥ compose on hard m*, early persistent-state objective was wrong.

$$\boxed{\text{intervene at decision sites; }P(S_{\mathrm{final}}=m^*)\text{ not retain}}$$
