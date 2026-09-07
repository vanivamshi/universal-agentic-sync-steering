# Phase 3 — Controllability at the real decision boundary

> Phase 1: predict? **yes**. Phase 2: control? **mostly no**.
> Phase 3: **why** prediction ≠ control (site / boundary / sensitivity).

Predictive directions need not be causal actuators; site and boundary alignment dominate controllability.

Best Exp1 direction for Exp2/3: **`margin_grad`** (α=1.5)

## Exp 1 — Direction construction (post-PLAN)

| Direction | ΔM(+) | ΔM(−) | |ΔM| | bidir | P(H=0) | task✓ | policy✓ |
|-----------|-------|-------|------|-------|--------|-------|---------|
| margin_grad | +16.754 | -14.563 | 15.658 | True | 0.38 | 0.62 | 0.62 |
| fisher | +1.160 | -0.841 | 1.000 | True | 0.38 | 0.62 | 0.75 |
| v_diff_MH_prior | +0.531 | -0.613 | 0.572 | True | 0.25 | 0.75 | 0.75 |
| mean_diff | -0.401 | +0.474 | 0.437 | False | 0.25 | 0.75 | 0.75 |
| optimized | +2.294 | -2.218 | 2.256 | True | 0.12 | 0.88 | 0.62 |
| v_H_old | -0.103 | -0.114 | 0.109 | False | 0.12 | 0.88 | 0.88 |

## Exp 2 — Site sensitivity

| Site | |ΔM| | P(H=0) | ΔP_tool(−) |
|------|------|--------|------------|
| prompt_end | 3.098 | 0.12 | -0.000 |
| mid_plan | 0.090 | 0.12 | -0.000 |
| post_plan | 15.658 | 0.00 | -0.857 |
| tool_decision | 15.658 | 0.50 | -0.857 |

## Exp 3 — Temporal control

| Mode | P(H=0) | |ΔM| | task✓ | policy✓ |
|------|--------|------|-------|---------|
| constant | 0.38 | 15.658 | 0.62 | 0.50 |
| post_plan_only | 1.00 | 15.658 | 1.00 | 1.00 |
| all_tokens | 0.00 | 15.658 | 1.00 | 0.75 |

## Interpretation gate

- If **site** dominates direction choice → intervention locus is the main lever.
- If **margin_grad / optimized** >> mean-diff → boundary alignment beats H-label geometry.
- If **post_plan_only** >> constant → temporal localization matters.
- 8-way stays gated until P(H=0) reaches ~0.6–0.8 with low collateral.

