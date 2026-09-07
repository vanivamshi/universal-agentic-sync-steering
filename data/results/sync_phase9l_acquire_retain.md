# Phase 9L — $P_{\mathrm{acquire}}\times P_{\mathrm{retain}}$

> Hybrid: $\Gamma'$ soft, all-channel sink-seek ($\lambda=0$) hard. Retain = 2-step stay-seek after first landing. Frozen 8K.

reps=4, acq_steps=7, n_retain=2, seed=20260931.

## Per $m^*$

| $m^*$ | pol | n | $P_{acq}$ | $P_{ret}\mid acq$ | $P_{final}$ (prod) | $P_{final}$ (emp) |
|-------|-----|--:|----------:|-------------------:|-------------------:|-------------------:|
| `000` | SS | 4 | **0.25** | 0.00 | 0.00 | **0.00** |
| `001` | SS | 4 | **0.00** | — | 0.00 | **0.00** |
| `010` | Γ' | 4 | **0.75** | 0.00 | 0.00 | **0.00** |
| `011` | Γ' | 4 | **1.00** | 0.25 | 0.25 | **0.25** |
| `100` | Γ' | 4 | **0.75** | 0.00 | 0.00 | **0.00** |
| `101` | Γ' | 4 | **0.50** | 0.00 | 0.00 | **0.00** |
| `110` | SS | 4 | **0.50** | 0.00 | 0.00 | **0.00** |
| `111` | Γ' | 4 | **0.25** | 0.00 | 0.00 | **0.00** |

$m^*$ with $P_{acq}>0$: **7/8**. $m^*$ with $P_{\mathrm{final}}>0$: **1/8**.

### Hard sinks

- $P_{acq}$: `{'000': 0.25, '001': 0.0, '110': 0.5}`
- $P_{ret}\mid acq$: `{'000': 0.0, '001': nan, '110': 0.0}`
- $P_{\mathrm{final}}$: `{'000': 0.0, '001': 0.0, '110': 0.0}`

## Gate

- Universal acquire: **False**
- Universal final (retain): **False**
- mean $P_{acq}$ / mean $P_{\mathrm{final}}$: 0.50 / 0.03

Primary: per-m* P_acquire, P_retain|acq, P_final. Reliable 8-way needs ∀m* P_final>0 (ideally ≫0).

$$\boxed{P_{\mathrm{final}}=P_{\mathrm{acquire}}\cdot P_{\mathrm{retain}}}$$
