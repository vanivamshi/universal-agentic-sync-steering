# Exp 2b — localize / sign / small-α / random control

- Overall: `DOSE2B_HIT`
- Best config: `steer|-u2|last`
- Baseline mean extra=0.67 calls=0.83

| config | ρ(α,extra) | Δextra(best+) | Δcalls(αhi) | proj@αhi | collapse | decision |
|---|---:|---:|---:|---:|:---:|---|
| `amp|U|all` | -0.821 | -0.17 | -0.67 | 0.877 | Y | `LOCAL_NULL` |
| `amp|U|last` | 0.071 | +0.00 | +0.00 | 0.398 | n | `LOCAL_NULL` |
| `amp|rand|last` | 0.893 | +0.00 | -0.17 | 0.245 | n | `LOCAL_WEAK` |
| `amp|u1|last` | 0.250 | +0.00 | -0.33 | 0.258 | n | `LOCAL_NULL` |
| `amp|u2|last` | -0.571 | +0.00 | -0.50 | 0.267 | Y | `LOCAL_NULL` |
| `steer|+u1|last` | -0.200 | +0.00 | -0.33 | 0.5 | n | `LOCAL_NULL` |
| `steer|+u2|last` | 0.000 | +0.17 | +0.00 | 0.5 | n | `LOCAL_WEAK` |
| `steer|-u1|last` | -0.400 | +0.17 | -0.50 | 0.5 | Y | `LOCAL_NULL` |
| `steer|-u2|last` | 0.800 | +0.17 | +0.00 | 0.5 | n | `LOCAL_HIT` |

Primary = n_extra_paths with calls preserved. AmpHook U≡−U; oriented tests are steer ±u.
Advance to Exp 3 only on DOSE2B_HIT (evidence > random).
