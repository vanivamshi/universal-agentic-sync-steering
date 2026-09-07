# H tool-decision boundary experiment

> Controller frozen: `d=g_H(e_H)v_H`. C observe-only.
> Population: m_H*=0 only. Primary: P(H'=0 | m_H*=0).
> M_H = logit(<tool_call>) - logit(FINAL).

## Diagnosis: **B**

Margin moves with α but does not yield reliable H=0 — weakly aligned direction; relearn H at tool-selection site rather than increasing α indefinitely.

- dM_H/dα = 0.6515066146850587
- crosses zero M_H: True
- max P(H=0) on v_H: 0.2
- random mean ΔM_H: 0.9808467864990235

## Dose curves (v_H)

| α | n | ΔM_H | M_H | P(M_H<0) | Δq_H | P(H=0) | mean O | E |
|---|---|------|-----|----------|------|--------|--------|---|
| 0.0 | 20 | +0.000 | +1.44 | 0.00 | +0.000 | 0.05 | 0.55 | 2.05 |
| -1.0 | 20 | -1.010 | +0.43 | 0.00 | -0.244 | 0.20 | 0.75 | 1.95 |
| -1.5 | 20 | -1.429 | +0.01 | 0.00 | -0.349 | 0.10 | 0.85 | 2.25 |
| -2.0 | 20 | -1.785 | -0.35 | 1.00 | -0.431 | 0.20 | 0.75 | 2.15 |
| -2.5 | 20 | -2.078 | -0.64 | 1.00 | -0.490 | 0.15 | 0.75 | 2.20 |
| -3.0 | 20 | -2.315 | -0.88 | 1.00 | -0.530 | 0.15 | 0.60 | 2.05 |

## Random control

| α | n | ΔM_H | P(H=0) |
|---|---|------|--------|
| -1.0 | 20 | +0.551 | 0.10 |
| -1.5 | 20 | +0.796 | 0.05 |
| -2.0 | 20 | +1.010 | 0.00 |
| -2.5 | 20 | +1.194 | 0.05 |
| -3.0 | 20 | +1.353 | 0.00 |

## Opposite +v_H

| α | n | ΔM_H | P(H=0) |
|---|---|------|--------|
| 2.0 | 20 | +2.576 | 0.00 |

## Primary endpoint summary

- P(H=0|α=0) = 0.050
- P(H=0|v_H) = 0.160
- P(H=0|random) = 0.040
- P(H=0|opp) = 0.000

## Next (gated)

- A → use min effective α; then H→O on same trajectories
- B → relearn H direction at tool-selection site (do not keep raising α)
- C → move intervention/measurement site
- 8-way still blocked

