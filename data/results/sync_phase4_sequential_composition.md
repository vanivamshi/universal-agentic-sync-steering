# Phase 4C — Sequential C→H→O composition (FROZEN)

> **Phase 4 conclusion:** Predictive-to-causal conversion produces directions that
> compose with substantially greater diagonal dominance than predictive counterparts.
> Pairwise compositionality does **not** guarantee sequential behavioral synchronization:
> order changes performance, and only **H** consistently reduces sync error.

$$
\boxed{\text{pairwise compositionality} \neq \text{sequential behavioral synchronization}}
$$

Closed-loop: \(e_t=m^*-S_t\) every stage. **8-way CLOSED.**

m* set: [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1)] · reps=4 · α=1.5

## Primary order C→H→O

| Arm | P(hit) | P(C) | P(H) | P(O) | ΔE tot | E0→E3 | task✓ | policy✓ |
|-----|--------|------|------|------|--------|-------|-------|---------|
| converted | 0.19 | 0.50 | 0.62 | 0.50 | +0.00 | 1.38→1.62→1.12→1.38 | 1.00 | 0.62 |
| predictive | 0.06 | 0.44 | 0.56 | 0.56 | −0.06 | 1.38→1.56→1.38→1.44 | 1.00 | 0.62 |
| random | 0.19 | 0.56 | 0.44 | 0.56 | −0.06 | 1.38→1.69→1.19→1.44 | 1.00 | 0.88 |

### Per-stage ΔE (primary)

| Arm | ΔE_C (0→1) | ΔE_H (1→2) | ΔE_O (2→3) |
|-----|------------|------------|------------|
| converted | −0.25 | **+0.50** | −0.25 |
| predictive | −0.19 | +0.19 | −0.06 |
| random | −0.31 | +0.50 | −0.25 |

**H is the only demonstrated behavioral actuator sequentially.**

## Order dependence (converted) — do not bury

| Order | P(hit) | ΔE tot |
|-------|--------|--------|
| C→H→O (primary) | 0.19 | 0.00 |
| **H→C→O** | **0.38** | **+0.12** |
| O→H→C | 0.12 | −0.25 |

$$
\boxed{\text{geometric compatibility vs temporal/trajectory compatibility}}
$$

## Gate

- 8-way: **CLOSED**
- Next bottleneck: do \(g_k(h_t)\) / \(v_c^{(k)}\) remain valid after \(h_0\to h_1\to h_2\)?
- Candidate: state-adaptive \(v_c^{(k)}(h_t)\propto\operatorname{Proj}_{g_k(h_t)}(v_p^{(k)})\)

Not a Phase 4 failure — identifies the sequential validity bottleneck.
