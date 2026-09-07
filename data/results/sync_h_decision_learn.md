# H tool-decision direction learning

> Target: post-PLAN \(M_H\) / \(P_{\mathrm{tool}}\), not binary H labels.
> Pipeline: discover → causal ± screen → collateral → candidate file.

Best candidate: **`v_diff_MH`**

| name | dM(+) | dM(−) | bidir | ΔP_tool(−) | P(H=0) | task✓ | policy✓ | score |
|------|-------|-------|-------|------------|--------|-------|---------|-------|
| v_diff_MH | +0.897 | -0.871 | True | -0.119 | 0.38 | 0.62 | 0.88 | 882.18 |
| v_cov_MH | +0.820 | -0.815 | True | -0.110 | 0.25 | 0.75 | 0.62 | 810.96 |
| v_reg_MH | +0.257 | -0.470 | True | -0.078 | 0.00 | 1.00 | 1.00 | 359.73 |
| v_H_old | -0.039 | -0.111 | False | -0.016 | 0.25 | 0.88 | 0.38 | 74.97 |
| v_random | -0.504 | +0.241 | False | +0.021 | 0.00 | 1.00 | 0.75 | 370.22 |

Wrote `data/directions/sync_v_H_decision_L4.json` (candidate — not final freeze).

If best shows signed ΔM_H and improved P(H=0) vs old_v_H, run H→O replication then held-out closed-loop. Else iterate candidates.

