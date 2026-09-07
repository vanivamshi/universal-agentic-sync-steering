# Phase 5C — Transition compatibility

> Failure ≠ wrong local direction (5A/5B).
> **Question:** what state transition does each intervention induce?

Frozen \(v_c\). α=1.5. Path C→H→O (suppress e=−1).

## Transition Jacobian \(T_{ij}\approx\partial M_i/\partial\alpha_j\)

| affected\\steer | C | H | O |
|------------------|---|---|---|
| **C** | -1.48 | +0.02 | +0.34 |
| **H** | -0.98 | -7.23 | +0.97 |
| **O** | +0.25 | +0.05 | -0.91 |

| Channel | diag dominance \|dM_j\|/Σ\_{i≠j}\|dM_i\| |
|---------|---------------------------------------------|
| C | 1.20 |
| H | 98.24 |
| O | 0.69 |

**T class:** `A_diagonal_transition` (mean |diag|=3.20, mean |off|=0.44) — driven by huge H diagonal.

**Nuance:** C column is **not** clean: \(T_{HC}\approx-0.98\) (C intervention moves H strongly). O DD only 0.69.

## Special: C|h0 vs C|h_H

| | \|ΔM_C\| | mean cross \|ΔM\| |
|--|---------|-------------------|
| C from h0 | 3.36 | 0.87 |
| C from h_H | 2.79 | 1.16 |
| retention / cross ratio | 0.83 | 1.33 |

ΔM vector C|h0: C=+1.14, H=+0.00, O=+0.11

ΔM vector C|h_H: C=+0.91, H=+0.94, O=+0.02

## Special: H|h0 vs H|h_C

| | \|ΔM_H\| | mean cross \|ΔM\| |
|--|---------|-------------------|
| H from h0 | 15.60 | 0.27 |
| H from h_C | 12.57 | 0.17 |
| retention / cross ratio | 0.81 | 0.66 |

ΔM vector H|h0: C=-0.02, H=+1.27, O=+0.04

ΔM vector H|h_C: C=-0.07, H=+1.73, O=-0.06

## Gate

- Cross worsens for C after H: **True** (ratio 1.33)
- Cross worsens for H after C: **False**
- Implication: local actuators OK; **C’s cross-channel transition coupling** is the sequential risk (esp. C→H). Behavioral threshold/path dependence still possible where margins look fine but bits regress.
- 8-way: **CLOSED**
