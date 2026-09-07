# Phase 3 — Predictive → causal conversion

> **Central hypothesis:** Predictive information becomes causally controllable when it
> has a component aligned with the local behavioral decision gradient \(g=\nabla_h M_H\).
>
> $$v_c=\operatorname{sign}(v_p^\top g)\,\operatorname{Proj}_g(v_p)/\|\operatorname{Proj}_g(v_p)\|$$

Decision site: post-PLAN tool/FINAL. α=1.5, local α=0.25.

## Hierarchy

$$
\text{Predictivity}
\rightarrow
\text{Boundary alignment}
\rightarrow
\text{Decision sensitivity}
\rightarrow
\text{Behavioral control}.
$$

| Predictor → outcome | Spearman ρ |
|---------------------|------------|
| \|corr\| → \|ΔM\| | −0.267 |
| **\|cos(v,g)\| → \|ΔM\|** | **+0.800** |
| product → \|ΔM\| | +0.883 |
| \|corr\| → P(H=0) | −0.400 |
| \|cos(v,g)\| → P(H=0) | +0.217 |

Predictivity alone does **not** track control; boundary alignment does.
The product improves further. Predictivity still matters as the source of
behavioral information to convert.

## Strong result: Proj / Orth decomposition

$$
v_p=\underbrace{\operatorname{Proj}_g(v_p)}_{\text{decision-relevant}}
+\underbrace{\operatorname{Orth}_g(v_p)}_{\text{decision-orthogonal}}
$$

### `mean_diff`

| Component | \|ΔM\| | P(H=0) |
|-----------|--------|--------|
| full | 0.437 | 0.00 |
| **Proj_g** | **15.658** | 0.00* |
| Orth_g | 0.036 | 0.12 |

### `v_H_old`

| Component | \|ΔM\| | P(H=0) |
|-----------|--------|--------|
| full | 0.109 | 0.12 |
| **Proj_g** | **15.658** | **1.00** |
| Orth_g | 0.174 | 0.00 |

\*Sign: \(\cos(\texttt{mean\_diff},g)<0\), so suppress-along-\(-v\) after naive Proj
points the wrong way — hence the operational \(\operatorname{sign}(v_p^\top g)\).
\|ΔM\| recovers `margin_grad` either way; directional behavior needs the sign fix.

## Per-direction metrics

| Direction | \|corr\| | AUC | cos(v,g) | \|ΔM\|_local | \|ΔM\| | bidir | P(H=0) | mean O |
|-----------|---------|-----|----------|-------------|--------|-------|--------|--------|
| margin_grad | 0.611 | 0.094 | +1.000 | 3.658 | 15.658 | True | 1.00 | 0.00 |
| mean_diff_proj_g | 0.611 | 0.906 | −1.000 | 3.658 | 15.658 | False | 0.00 | 1.00 |
| v_H_proj_g | 0.611 | 0.094 | +1.000 | 3.658 | 15.658 | True | 1.00 | 0.00 |
| fisher | 0.914 | 1.000 | +0.026 | 0.172 | 1.000 | True | 0.00 | 1.00 |
| mean_diff | 0.864 | 1.000 | −0.019 | 0.072 | 0.437 | False | 0.00 | 1.00 |
| v_H_orth_g | 0.660 | 0.083 | −0.000 | 0.028 | 0.174 | False | 0.00 | 1.00 |
| random | 0.246 | 0.786 | −0.023 | 0.020 | 0.138 | False | 0.12 | 0.88 |
| v_H_old | 0.661 | 0.083 | +0.009 | 0.004 | 0.109 | False | 0.12 | 0.88 |
| mean_diff_orth_g | 0.864 | 1.000 | +0.000 | 0.004 | 0.036 | False | 0.12 | 0.88 |

## Two conversion levels

| Level | Mapping | Status |
|-------|---------|--------|
| **Mechanistic** | \(v_p\rightarrow\operatorname{Proj}_g(v_p)\) moves \(M_H\) | **Supported** (\|ΔM\|≈15.7 vs Orth≈0.04) |
| **Behavioral** | \(\operatorname{Proj}_g\rightarrow H\) under natural PLAN | **Partial** (\(P(H=0)=0.50\); see `sync_h_phase3_validation`) |

Phase 3 science is established at the margin level by the conversion principle.
8-way / closed-loop are **not** part of that result — only later agentic validation
if natural-PLAN reliability passes.
