# SYNC closed-loop repair (probe → equation → hook)

- Decision: **`CLOSED_LOOP_NULL`**
- Mode: **force hook on every private fault** (probe logged, not gating)
- Equation: `v_R = unit(P_S(μ_B − μ_C))`, α=0.25, τ=-2.681
- cos(v_R, v_Δ)=0.214
- Protocol: `sync_closed_loop.md`

## Rates

| Arm | n | disclose\|private | hidden | gate fire | hook fire |
|---|---:|---:|---:|---:|---:|
| normal | 18 | 0.944 | 0.056 | 0.000 | 0.000 |
| fault | 18 | 0.389 | 0.611 | 0.667 | 0.000 |
| +v_delta | 18 | 0.389 | 0.611 | 0.667 | **1.000** |
| +v_repair | 18 | 0.389 | 0.611 | 0.667 | **1.000** |
| random | 18 | 0.389 | 0.611 | 0.667 | **1.000** |

## Recovery

- `+v_repair`: R=0.000 [0.000,0.000] ratio=1.000
- `+v_delta`: R=0.000 [0.000,0.000] ratio=1.000
- `random`: R=0.000 [0.000,0.000] ratio=1.000
- Spurious budget OK: `True`

## Headline

**Hooks ran on 100% of steered private episodes** (abort if registered but n_fwd=0).
Disclosure still matches fault exactly. Activation equation is applied but not
causally sufficient; **policy gate** remains the working repair.
