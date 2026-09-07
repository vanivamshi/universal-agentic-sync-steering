# Phase 8G — Frozen $v_c^O$ dose sweep

> Ground truth: `score_output_disclose`. Proxy: During/docs (auxiliary). No new $v$.

n=12, seed=20260915, α∈[1.5, 3.0, 5.0, 8.0].

## Dose table

| α | $E[\|\Delta B\|]$ | $E[G_O]$ | P(proxy flip) | P($S_O$ flip) | P(W→C)$_{TS}$ | P(W→C)$_{WS}$ |
|--:|------------------:|---------:|--------------:|--------------:|--------------:|--------------:|
| 1.5 | 0.156 | +0.156 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.313 | +0.313 | 0.25 | 0.00 | 0.00 | 0.00 |
| 5 | 0.522 | +0.522 | 0.42 | 0.08 | 0.08 | 0.00 |
| 8 | 0.835 | +0.835 | 0.50 | 0.17 | 0.25 | 0.08 |

## Gate

- $E[G_O]>0$ all α: **True**
- $|\Delta B|$ increases with α: **True**
- Proxy flip increases: **True**
- **$S_O$ flip increases (disclose GT):** **True**
- Under-actuation (gain → behavior): **True**
- Proxy/behavior mismatch: **False**
- Gain helps W→C or $S_O$: **True**

If |ΔB|↑ and P(S_O flip)↑ with E[G]>0: under-actuation → use higher α_O. If |ΔB|↑ but S_O flat: proxy≠behavior → align scoring / early-FINAL, not more gain. Only then reopen O conversion.

Next: if under-actuation → raise α_O in controller; if mismatch → disclose-aligned early-FINAL; only then re-convert O.
