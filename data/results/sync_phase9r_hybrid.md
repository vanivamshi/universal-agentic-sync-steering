# Phase 9R — Hybrid $\Gamma'$ + beam fallback (8-way acq×ret)

> Paired $S_0$. No new $v$. Beam over $\pm C,\pm H,\pm O$ when $\max\Gamma'<\tau$ ($\tau=0.0$), predicted no-op, or observed no-op.

reps=2, max_steps_acq=7, n_retain=2, beam K=3 T=3, seed=20261006.

## $\Gamma'$ alone

| $m^*$ | $P_{acq}$ | $P_{ret}$ | $P_{final}$ |
|-------|----------:|----------:|------------:|
| `000` | 0.00 | — | **0.00** |
| `001` | 0.50 | 0.00 | **0.00** |
| `010` | 0.50 | 0.00 | **0.00** |
| `011` | 1.00 | 0.00 | **0.00** |
| `100` | 1.00 | 0.50 | **0.50** |
| `101` | 1.00 | 0.00 | **0.00** |
| `110` | 0.00 | — | **0.00** |
| `111` | 1.00 | 0.00 | **0.00** |

States with $P_{acq}>0$: **6/8**; $P_{final}>0$: **1/8**.
Successful transitions target-sign rate: 1.00.

## $\Gamma'$ + beam fallback

| $m^*$ | $P_{acq}$ | $P_{ret}$ | $P_{final}$ | mean beam calls |
|-------|----------:|----------:|------------:|-----------------:|
| `000` | 0.50 | 0.00 | **0.00** | 2.00 |
| `001` | 1.00 | 0.00 | **0.00** | 1.00 |
| `010` | 1.00 | 0.00 | **0.00** | 1.00 |
| `011` | 1.00 | 0.00 | **0.00** | 1.00 |
| `100` | 1.00 | 0.50 | **0.50** | 1.00 |
| `101` | 1.00 | 0.00 | **0.00** | 1.00 |
| `110` | 1.00 | 0.00 | **0.00** | 1.00 |
| `111` | 1.00 | 0.00 | **0.00** | 1.00 |

States with $P_{acq}>0$: **8/8**; $P_{final}>0$: **1/8**.

### Sign vs target-sign (successful transitions)

- $P(\text{target sign})$ = **0.44**
- $P(\text{opposite sign})$ = **0.56**
- Among beam-sourced successes, $P(\text{opposite})$ = **0.68** (n=22)

Acq gate (hybrid 8/8): **True**. Final gate (hybrid 8/8): **False**.

Primary: hybrid P_acq>0 on all 8. Secondary: P_final>0 on all 8. Contrast Γ' alone. Report opposite-sign rate on successful transitions.

$$\boxed{\pi=\Gamma'\ \text{or}\ \mathrm{beam}(\pm v_C,\pm v_H,\pm v_O)}$$
