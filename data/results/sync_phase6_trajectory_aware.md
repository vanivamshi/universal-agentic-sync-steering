# Phase 6 — Trajectory-aware gains (frozen \(v_c\))

> Can sequential sync be restored by choosing α to penalize transition collateral?

α0=1.5, λ=1.0, grid=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], reps=4 × 4 \(m^*\)

## C→H contrast replication

| | retention | cross ratio | \|ΔM_H\| collat h₀/h_H |
|--|-----------|-------------|------------------------|
| 5C | 0.83 | **1.33** | (cross 0.87→1.16) |
| **this run** | 0.94 | **0.27** | 1.47 / **0.17** |

Cross-worsening **did not replicate** (opposite direction). Retention remains high. Treat \(T_{HC}\) as unstable diagnostic, not a fixed gain to invert.

## Sequential fixed vs aware

| Arm | P(hit) | ΔE tot | ΔE_C | ΔE_H | ΔE_O | E0→E3 | ᾱ_C | C intended | C collat | C net |
|-----|--------|--------|------|------|------|-------|------|------------|----------|-------|
| fixed | 0.25 | −0.25 | −0.06 | −0.06 | −0.12 | 1.38→1.44→1.50→1.62 | 1.50 | +4.17 | 1.67 | +2.51 |
| aware | 0.25 | −0.12 | −0.06 | +0.12 | −0.19 | 1.38→1.44→1.31→1.50 | 1.36 | +4.08 | 1.49 | +2.59 |

## Gate

- Aware ΔE > fixed: **True** (both still **worsen** mean E)
- Aware hit ≥ fixed: **True** (tied at 0.25)
- Aware reduces C collateral: **True** (small)
- Aware improves C net: **True** (small)
- Restores sequential sync: **False**
- 8-way: **CLOSED**

**Verdict:** trajectory-aware discrete α gives a weak H-stage / collateral edge but does **not** restore sync with frozen \(v_c\). Directions frozen. No new \(v\).
