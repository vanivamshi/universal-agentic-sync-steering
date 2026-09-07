# H decision proxy diagnostic

## Finding: **PROXY_MARGIN_NOT_TRUE_ACTION_BOUNDARY**

At prompt end, P(PLAN)≈1 and P(<tool_call>)≈0. Old boundary M_H can cross zero without controlling the sampled action. The real local decision is post-PLAN: <tool_call> vs FINAL.

- prompt-end α=0: argmax=`PLAN`, P_PLAN=0.9983, P_tool=0.0000, M_H(tool−FINAL)=+1.44
- old v_H @ α=−2 mean ΔP_tool at post-PLAN site: **-0.0279** (tiny — weakly aligned at the *true* site)

## Implication

Do not keep raising α on old v_H. Relearn against post-PLAN M_H / P_tool, then causal-screen for sampled H.

Next: `learn_h_tool_decision_direction.py` targeting post-PLAN M_H / P_tool.

