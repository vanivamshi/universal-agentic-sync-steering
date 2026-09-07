# Selective steering v4 — decision/logit controllability

- Decision: **`SELECTIVE_V4_WEAK`**
- Protocol: `docs/selective_steering_protocol.md`
- m_T = logit(`<tool_call>`) − max(logit(no-tool))
- u* = `theta_31` (argmax train S); S_spread=0.6728
- Spearman(train S → held |ΔP(tool)|) = **-0.700**

## Train decision scores (selected)

| direction | S | D_T | D_C |
|---|---:|---:|---:|
| `theta_31` ← u* | 2.1245 | -0.2516 | 0.0654 |
| `u1` | 2.0808 | -0.2422 | 0.0608 |
| `neg_u2` | 1.2438 | -0.1438 | 0.0674 |
| `rand_full` | 1.2006 | -0.1271 | 0.0563 |
| `orth_u2` | 1.1961 | -0.1265 | 0.0562 |
| `u2` | 0.4480 | -0.0368 | 0.0674 |
| `neg_u1` | 0.2745 | -0.0053 | 0.0608 |

## Held-out ±α (frozen)

| arm | train S | D_T (test) | |ΔP(tool)| | ΔP(+) | ΔP(−) | bidir |
|---|---:|---:|---:|---:|---:|:---:|
| `u_star` | 2.1245 | -0.2208 | 0.000 | +0.000 | +0.000 | False |
| `u1` | 2.0808 | -0.2092 | 0.000 | +0.000 | +0.000 | False |
| `u2` | 0.4480 | -0.0475 | 0.000 | +0.000 | +0.000 | False |
| `rand_full` | 1.2006 | -0.1451 | 0.000 | +0.000 | +0.000 | False |
| `orth_u2` | 1.1961 | -0.1446 | 0.000 | +0.000 | +0.000 | False |

v3 failed because discrete `n_extra_paths` has no local gradient. v4 differentiates the **decision logits** that produce the action.
