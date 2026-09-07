# hurt_16 vs original top_16 overlap

Computed from existing artifacts (no new generation).

| comparison | intersection | union | Jaccard |
|---|---:|---:|---:|
| hurt_16 vs ΔNLL top_16 | 15 | 17 | **0.882** |
| hurt_16 vs \|Δ\| top_16 (npy) | 15 | 17 | **0.882** |
| hurt_16 vs g×a top_16 | 7 | 25 | 0.280 |

**Writeup note:** sign-flip `hurt_16` is **mostly redundant** with the original
magnitude landmine set (ΔNLL / \|Δ\| top_16). The polarity check mostly
re-derives channels already known to be catastrophic under \|Δ\| ranking —
not fully independent evidence that *signed-positive* is a new landmine class.
g×a top_16 overlap is only moderate (0.28).

help_16 ∩ ΔNLL top_16 Jaccard ≈ 0.03 (essentially disjoint) — the help set
is not recycling old tops.
