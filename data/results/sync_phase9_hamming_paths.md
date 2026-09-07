# Phase 9 — Hamming-path reachability

> Frozen 8K actuators. Compare direct $000\!\to\!111$ vs shortest Hamming paths. Final success $=S_{\mathrm{final}}{=}111$ only.

reps=4, require_source=True, max_free=10, seed=20260921.

## Aggregate ($P(S_{final}=111)$)

| arm | path | n (matched) | P(hit final) | P(all wp) | ΔE | mean free attempts |
|-----|------|------------:|-------------:|----------:|---:|-------------------:|
| direct | `000→111` | 4 | **0.00** | 0.00 | -1.00 | 10.0 |
| OHC | `000→001→011→111` | 4 | **0.00** | 0.00 | -0.50 | 10.0 |
| HOC | `000→010→011→111` | 4 | **0.00** | 0.00 | -0.50 | 10.0 |
| COH | `000→100→101→111` | 4 | **0.75** | 0.00 | +0.75 | 10.0 |

| 8K val one-shot (111) | — | 8 | **0.00** | — | — | — |

## Per-edge $P(s_{i+1}\mid s_i)$ (matched starts)

### OHC

| edge | P(hit waypoint) |
|------|----------------:|
| `000→001` | 0.00 |
| `001→011` | 0.00 |
| `011→111` | 0.00 |

### HOC

| edge | P(hit waypoint) |
|------|----------------:|
| `000→010` | 0.00 |
| `010→011` | 0.75 |
| `011→111` | 0.00 |

### COH

| edge | P(hit waypoint) |
|------|----------------:|
| `000→100` | 0.50 |
| `100→101` | 0.00 |
| `101→111` | 0.75 |

## Gate

- Path beats direct: **True**
- Best path: **COH** (P=0.75)
- Direct P(hit): **0.0**
- Any path reaches 111: **True**

Final success = S_final==111 only. Intermediates are control path, not credit. No new v.

If pass: *direct controllability ≠ global reachability; local controllability + intermediate states ⇒ global reachability.*
