# Phase 3 validation — natural PLAN + held-out margin_grad

> Eliminates teacher-forced PLAN confound. Primary: P(H=0). Mechanistic: ΔM_H.

Activation directions that predict agent behavior need not provide causal control. Effective control depends on decision-boundary alignment, intervention at the computational site where the decision is formed, and temporally restricted intervention that avoids perturbing upstream generation.

- Reliable H-control gate: **False**

## Arms (held-out natural PLAN)

| Arm | n | P(H=0) | ΔM_H | ΔP_tool | mean O | task✓ | policy✓ | plan✓ |
|-----|---|--------|------|---------|--------|-------|---------|-------|
| none | 12 | 0.25 | +0.00 | +0.000 | 0.50 | 1.00 | 0.75 | 0.75 |
| margin_grad | 12 | 0.50 | -4.06 | -0.168 | 0.25 | 1.00 | 0.75 | 0.92 |
| random | 12 | 0.33 | -0.15 | -0.011 | 0.50 | 1.00 | 0.83 | 0.75 |

## Gate

```text
direction × site × timing  (Phase 3)
        ↓
validate without teacher forcing   ← this run
        ↓
held-out trajectories              ← this run
        ↓
reliable H control?
        ↓ yes
H→O → closed-loop sync → 8-way
```

Next: **keep Phase 3; do not open 8-way**

