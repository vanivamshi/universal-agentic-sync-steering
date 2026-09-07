# Phase 4 — 3-channel conversion matrix

> Does \(v_p^{(k)}\rightarrow\operatorname{Proj}_{g_k}(v_p^{(k)})\) work for \(k\in\{C,H,O\}\) before composition?

Phase 3 freeze: H conversion supported. This matrix tests C and O.

## Per-channel arms

| Ch | Arm | site | cos(v,g) | |ΔM| | bidir | M0 |
|----|-----|------|----------|-------|-------|----|
| C | predictive | mid_PLAN_intent | -0.040 | 0.186 | False | +7.28 |
| C | mean_diff_site | mid_PLAN_intent | +0.122 | 0.701 | True | +7.28 |
| C | converted | mid_PLAN_intent | +1.000 | 3.361 | True | +7.28 |
| C | orth | mid_PLAN_intent | -0.000 | 0.049 | False | +7.28 |
| C | gradient | mid_PLAN_intent | +1.000 | 3.361 | True | +7.28 |
| H | predictive | post_PLAN_tool | +0.006 | 0.109 | False | +3.07 |
| H | mean_diff_site | post_PLAN_tool | -0.025 | 0.581 | False | +3.07 |
| H | converted | post_PLAN_tool | +1.000 | 15.599 | True | +3.07 |
| H | orth | post_PLAN_tool | +0.000 | 0.130 | False | +3.07 |
| H | gradient | post_PLAN_tool | +1.000 | 15.599 | True | +3.07 |
| O | predictive | post_tool_FINAL | +0.028 | 0.051 | False | +3.29 |
| O | mean_diff_site | post_tool_FINAL | -0.027 | 0.055 | False | +3.29 |
| O | converted | post_tool_FINAL | +1.000 | 1.832 | True | +3.29 |
| O | orth | post_tool_FINAL | +0.000 | 0.010 | False | +3.29 |
| O | gradient | post_tool_FINAL | +1.000 | 1.832 | True | +3.29 |

## Gate: Proj ≫ Orth

| Ch | \|ΔM\|_pred | \|ΔM\|_conv | \|ΔM\|_orth | proj≫orth | conv>pred | \|cos\| |
|----|-------------|-------------|-------------|---------|----------|--------|
| C | 0.186 | 3.361 | 0.049 | True | True | 0.040 |
| H | 0.109 | 15.599 | 0.130 | True | True | 0.006 |
| O | 0.051 | 1.832 | 0.010 | True | True | 0.028 |

## Cross-channel Jacobian \(J_{ij}\approx\partial M_i/\partial\alpha\) along \(v_c^j\)

| affected\\steer | C | H | O |
|------------------|---|---|---|
| **C** | +3.00 | +0.17 | -0.38 |
| **H** | +0.98 | +11.24 | -2.10 |
| **O** | -0.11 | -0.14 | +1.44 |

Channels with Proj≫Orth: **3/3**
Ready for pairwise composition: **True**

8-way stays closed until pairwise + sequential closed-loop pass.

If C or O fail conversion, that is informative: conversion is **channel- and boundary-specific**, not universal.
