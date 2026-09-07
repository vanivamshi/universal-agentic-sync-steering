# Hierarchical asymmetric controller eval (selected m*)

> Architecture frozen: **C → H → O(observe)**. Not a proven causal graph.
> Not 8-way. Not factorized V freeze.

Dense intervention experiments support H as a causal bottleneck and are consistent with a hierarchical C–H–O control structure, motivating an asymmetric controller with H as the primary actuator and O as an observed downstream outcome.

- mean ΔE (E_before − E_after): **+0.125** (positive = improved)
- α=0.75, scale_C=0.25, scale_H=1.0

## Per target

| m* | n | E_before | E_after | ΔE | Δq_H | ΔH | ΔO |
|----|---|----------|---------|----|------|----|----|
| [1, 1, 1] | 2 | 1.50 | 1.50 | +0.00 | -0.075 | +0.00 | +0.00 |
| [0, 0, 0] | 2 | 2.00 | 1.50 | +0.50 | -0.222 | +0.50 | -0.50 |
| [1, 0, 0] | 2 | 2.50 | 2.50 | +0.00 | -0.241 | +0.00 | +0.50 |
| [0, 1, 1] | 2 | 0.00 | 0.00 | +0.00 | -0.018 | +0.00 | +0.00 |

## Metrics legend

1. **Continuous:** Δq_C, Δq_H — preference shift before discrete flips
2. **Execution:** ΔH / tools — bottleneck outcome
3. **Sync:** ΔO / E_l1 — observed downstream, not steered

Architecture file: `data/directions/sync_hierarchical_architecture.json`

