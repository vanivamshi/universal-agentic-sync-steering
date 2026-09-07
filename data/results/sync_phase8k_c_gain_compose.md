# Phase 8K — C gain compose ($H\to C\to O$)

> $\alpha_H=1.5$ decision-token; $\alpha_C=5$ stem-prefill; $\alpha_O=1.5$ early-FINAL. Target-sign. No skip. No new $v$.

m* n=4, reps=2, eight_way=True.

## Selected / primary

| | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |
|--|--------|---:|-----:|-----:|-----:|------|------|------|
| **8K** | 0.38 | +0.62 | **+0.00** | +0.62 | +0.00 | 0.62 | 0.62 | 0.88 |
| 8I H→C | 0.38 | +0.75 | -0.25 | +0.62 | +0.38 | 0.62 | 0.75 | 0.88 |

## Full 8-way

| | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |
|--|--------|---:|-----:|-----:|-----:|------|------|------|
| **8K 8-way** | 0.25 | +0.44 | **+0.25** | +0.12 | +0.06 | 0.69 | 0.69 | 0.56 |
| 8H ref | 0.31 | +0.38 | — | — | — | 0.56 | 0.69 | 0.75 |
| 8E conv | 0.06 | +0.31 | — | — | — | 0.56 | 0.62 | 0.44 |

## Gate

- $\Delta E_C \ge 0$: **True** (selected +0.00; **8-way +0.25**)
- $\Delta E_C$ improves vs 8I (−0.25): **True**
- H preserved (selected): **True** (+0.62)
- Compose pass: **True**
- Note: selected-4 C stages were all noops; clearest C evidence is full 8-way

$$
\boxed{\Delta E_C:\ -0.25\ \rightarrow\ 0\ /\ +0.25}
$$

C no longer hurts under the composed controller. No skip. No new $v$.
