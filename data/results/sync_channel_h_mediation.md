# Stage B.2 — Dense \(v_H\) mediation

> Hierarchical controller **hypothesis** (not yet a proven causal graph).
> Do **not** freeze factorized V; do **not** treat \(v_O\) as an actuator.

- Mediation reading: **H is a plausible causal intermediary (ΔH→ΔO)**
- H flips: 4 / 32 steered trials
- P(ΔO|ΔH)=0.75  P(ΔO|¬ΔH)=0.21428571428571427
- Hierarchy gate: **True**

## Dose curves (\(v_H\))

| α | n | P(H=1) | P(O=1) | q_H |
|---|---|--------|--------|-----|
| -1.0 | 4 | 0.75 | 0.50 | 0.249 |
| -0.75 | 4 | 0.75 | 0.75 | 0.278 |
| -0.5 | 4 | 1.00 | 1.00 | 0.361 |
| -0.25 | 4 | 0.75 | 0.50 | 0.400 |
| 0.0 | 4 | 1.00 | 1.00 | 0.599 |
| 0.25 | 4 | 1.00 | 0.75 | 0.514 |
| 0.5 | 4 | 1.00 | 0.75 | 0.544 |
| 0.75 | 4 | 0.75 | 1.00 | 0.674 |
| 1.0 | 4 | 1.00 | 0.50 | 0.721 |

## Slopes

- \(v_H\): {'dP_H/dα': 0.08333333333333337, 'dP_O/dα': 0.03333333333333343, 'dq_H/dα': 0.23716233470225714}
- rand: {'dP_H/dα': 0.04999999999999986, 'dP_O/dα': -0.10000000000000012, 'dq_H/dα': -0.002339549696734069}

## Matched random (subset)

| α | n | P(H=1) | P(O=1) | q_H |
|---|---|--------|--------|-----|
| -1.0 | 4 | 1.00 | 1.00 | 0.526 |
| -0.5 | 4 | 0.75 | 0.50 | 0.448 |
| 0.0 | 4 | 0.75 | 0.75 | 0.599 |
| 0.5 | 4 | 1.00 | 0.50 | 0.508 |
| 1.0 | 4 | 1.00 | 0.75 | 0.490 |

## Decision gate

- Dense \(v_H\) mediation → establish H as causal bottleneck
- Then hierarchical controller \(d=g_C(e_C)v_C+g_H(e_H)v_H\) (no \(v_O\))
- Only afterward consider 8-way
- Do **not** spend runs making O independently controllable

