# Exp 2.2 — Dataset B transfer (locked α=0.25)

- Decision: `TRANSFER_NULL`
- Workspace: `sandbox_transfer_b`  N_REPS=6 × 6 families
- Δ_specific(extra)=Δ(−u₂)−Δ(orth)=+0.028 CI90=[-0.111, 0.167]
- Δ_specific(P)=-0.028 CI90=[-0.139, 0.083]

| condition | mean extra | P(extra>0) | calls | Δextra | ΔP | Δlist | Δsearch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.72 | 0.69 | 0.78 | — | — | — | — |
| `neg_u2` | 0.78 | 0.72 | 0.97 | +0.06 | +0.03 | +0.08 | +0.14 |
| `pos_u2` | 0.92 | 0.81 | 1.19 | +0.19 | +0.11 | +0.17 | +0.19 |
| `random` | 0.69 | 0.69 | 0.78 | -0.03 | +0.00 | -0.06 | +0.08 |
| `orthogonal` | 0.75 | 0.75 | 1.00 | +0.03 | +0.06 | -0.03 | +0.25 |

Exp 3 only on TRANSFER_HIT. Otherwise stop u₂-specific Exp-2 claim.
