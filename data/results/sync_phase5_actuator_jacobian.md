# Phase 5B — Actuator / Jacobian adaptation

> 5A locked: gradient recomputation \(\not\Rightarrow\) trajectory-valid control.
> Does the **actuation map** drift, requiring local \(v_{\mathrm{opt}}\) re-ID?

## C actuator: before vs after H (mechanistic)

| Arm | |ΔM_C|(h0) | |ΔM_C|(after H) | retention |
|-----|---------------|---------------------|-----------|
| static | 2.220 | 1.890 | 0.85 |
| adaptive | 2.170 | 1.920 | 0.88 |
| v_opt | 2.220 | 1.920 | 0.86 |

- Actuator drift (retention < 0.7 for static/adaptive): **False**
- $v_{\mathrm{opt}}$ beats adaptive after H (≥1.25×): **False**

## Sequential C→H→O

| Arm | P(hit) | ΔE tot | ΔE_C | ΔE_H | ΔE_O | E0→E3 |
|-----|--------|--------|------|------|------|-------|
| static | 0.50 | +0.88 | +0.25 | -0.12 | +0.75 | 1.88→1.62→1.75→1.00 |
| adaptive | 0.50 | +0.88 | +0.38 | -0.25 | +0.75 | 1.88→1.50→1.75→1.00 |
| v_opt | 0.50 | +0.88 | +0.25 | -0.12 | +0.75 | 1.88→1.62→1.75→1.00 |

## Gate

- Opt beats adaptive after H (mechanistic): **False**
- Opt beats adaptive on sequential ΔE: **False**
- C regression fixed by v_opt: **True**
- 8-way: **CLOSED**

If opt ≫ adaptive after H → trajectory validity needs **local actuator re-identification**.

**Read:** all three arms tie on sequential sync. Local \(v_{\mathrm{opt}}\) does not unlock trajectory validity beyond static/adaptive conversion. The C sequential problem is not explained by an obsolete actuator that re-ID would recover.
