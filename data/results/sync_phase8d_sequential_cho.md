# Phase 8D — Sequential C→H→O (corrected H)

> $d_k=(2m_k^*-1)v_c^k$. H = decision-token-only (Phase 8C). Same seed per trial.

No new $v$. No policy. **8-way CLOSED.** Skip noop stages.

m* set: [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1)] · reps=2 · α=1.5 · fast=False

## Primary order C→H→O

| Arm | P(hit) | P(C) | P(H) | P(O) | P(W→C)_H | ΔE tot | E0→E3 |
|-----|--------|------|------|------|----------|--------|-------|
| corrected | 0.25 | 0.50 | 0.62 | 0.88 | 0.25 | +0.50 | 1.50→1.62→1.38→1.00 |
| Phase4 converted (ref) | 0.19 | 0.50 | 0.62 | 0.50 | — | +0.00 | 1.38→1.62→1.12→1.38 |

### Per-stage ΔE

| Arm | ΔE_C | ΔE_H | ΔE_O |
|-----|------|------|------|
| corrected | -0.12 | +0.25 | +0.38 |
| Phase4 converted | −0.25 | **+0.50** | −0.25 |

## Gate

- Beats Phase4 P(hit): **True**
- Beats Phase4 ΔE tot: **True**
- H stage reduces E: **True**
- ΔP(H) vs Phase4: **+0.01**
- P(W→C)_H: **0.25**
- Sequential composition improved: **True**

Tests whether Phase-8C H chain survives C→H→O sequential composition. No new v. 8-way still paused.
