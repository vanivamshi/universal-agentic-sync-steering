# Phase 7 — Episode-conditioned boundary

> Does frozen $v_c$ move the *live* decision boundary $h^{\mathrm{live}}\to S$?

n=32, causal-n=12, α=1.5, layer=4.

## Template vs live probe variance

| Channel | $M$ template | std $M$ live | std $h$ across episodes |
|---------|--------------|--------------|-------------------------|
| C | +7.285 | 4.547 | 0.0118 |
| H | +3.068 | 13.480 | 0.0693 |
| O | +3.287 | 3.855 | 0.0337 |

## A — Predictive: $h^{\mathrm{live}}\to S$ vs template

| Channel | n0/n1 | AUC live $h$ | AUC live $M$ | AUC template $M$ | live ≫ template |
|---------|-------|--------------|--------------|------------------|-----------------|
| C | 11/21 | 0.875 | 0.154 | 0.5 | **True** |
| H | 18/14 | 1.000 | 0.996 | 0.5 | **True** |
| O | 15/17 | 0.667 | 0.498 | 0.5 | **True** |

## B — Causal on activations: $\Delta B(h\pm\alpha v_c)$

| Channel | mean ΔB(+α) | mean ΔB(−α) | frac +↑B | mono dose |
|---------|-------------|-------------|----------|-----------|
| C | +0.082 | -0.082 | 1.00 | **True** |
| H | +0.412 | -0.412 | 1.00 | **True** |
| O | +0.156 | -0.156 | 1.00 | **True** |

## C — Local dose-response at live site ($S_{\mathrm{proxy}}$ from site logits)

| Channel | n | P(flip $S^-$↔$S^+$) | P(M mono) | P(B mono) | mean $M^-/M^0/M^+$ |
|---------|---|---------------------|-----------|-----------|---------------------|
| C | 12 | 0.42 | 1.00 | 1.00 | +1.56/+3.10/+6.27 |
| H | 12 | 0.33 | 1.00 | 1.00 | +1.14/+6.43/+12.14 |
| O | 12 | 0.17 | 1.00 | 1.00 | +1.38/+2.78/+4.10 |

## Gate

- Any recoverable live boundary (AUC≥0.65): **True**
- Any $v_c$ moves live $B$ monotonically: **True**
- Chain recoverable on ≥1 channel: **True**

If AUC_live ≫ template and v_c moves B monotonically with some S_proxy flips: missing map is recoverable. If AUC_live good but v_c does not move B: deeper problem.

Directions frozen. No new $v$. No policy.
