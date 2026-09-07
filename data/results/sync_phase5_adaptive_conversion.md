# Phase 5A — Static vs adaptive \(g_k(h_t)\) (LOCKED NEGATIVE)

> **Result:** trajectory validity \(\not\approx\) recompute local \(g_k(h_t)\).

| Arm | P(hit) | ΔE tot | ΔE_C | ΔE_H | ΔE_O |
|-----|--------|--------|------|------|------|
| static | 0.44 | +0.50 | −0.19 | +0.31 | +0.38 |
| adaptive | 0.44 | +0.44 | −0.12 | +0.19 | +0.38 |
| predictive | 0.25 | 0.00 | −0.19 | +0.38 | −0.19 |
| random | 0.25 | +0.31 | 0.00 | −0.19 | +0.50 |

- Hit tied; adaptive does not beat static.
- C regression persists.
- O (+0.38) shared by both converted arms — not from adaptation.

$$
\boxed{\text{local gradient recomputation}\ \not\Rightarrow\ \text{trajectory-valid control}}
$$

$$
\boxed{
\text{causal control}
=
\text{boundary alignment}
+
\text{state-dependent sensitivity}
+
\text{trajectory/state preservation}
}
$$

Next (5B): static vs adaptive vs \(v_{\mathrm{opt}}(h_t)\) — actuator/Jacobian drift.
**8-way CLOSED.**
