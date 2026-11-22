# Phase 9Y — order-swapped re-execution of the 9X conversations

> Diagnostic, not a replication: the same 100 9X conversations (same seeds, regenerated round 0, same round seeds, same frozen gains, same environment), with A/B execution order reversed for every pair. Analysed separately from 9X; never pooled.

Environment: torch 2.8.0+cu129, transformers 5.24.0, CUDA 12.9, NVIDIA RTX A6000, float32. Round 0 re-reached 111 in 100/100.

## 1. Does B keep its advantage after the swap?

| run | order AB / BA | A hold | B hold | B-only | A-only | McNemar p | decision rule |
|---|---|---:|---:|---:|---:|---:|---|
| 9X | 63 / 37 | 74/100 | 85/100 | 19 | 8 | 0.052 | B confirmed as candidate; not shown better |
| 9Y (swapped) | 37 / 63 | 74/100 | 85/100 | 19 | 8 | 0.052 | B confirmed as candidate; not shown better |

## 2. Within each original 9X order stratum

| 9X order | n | run | A hold | B hold | B-only | A-only | McNemar p |
|---|---:|---|---:|---:|---:|---:|---:|
| AB | 63 | 9X | 43 | 53 | 15 | 5 | 0.041 |
| AB | 63 | 9Y | 43 | 53 | 15 | 5 | 0.041 |
| BA | 37 | 9X | 31 | 32 | 4 | 3 | 1.000 |
| BA | 37 | 9Y | 31 | 32 | 4 | 3 | 1.000 |

## 3. Failure anatomy (9Y)

| arm | fails | C alone | C with H | H-first | final state | 9X: C alone / C+H |
|---|---:|---:|---:|---:|---|---|
| A | 26 | 13 | 7 | 6 | `011`×12, `001`×7, `101`×7 | 13 / 7 |
| B | 15 | 7 | 3 | 5 | `011`×7, `101`×5, `001`×3 | 8 / 3 |

## 4. Does execution position matter?

Per arm, each conversation was run once first and once second (9X vs 9Y):

| arm | n | hold when first | hold when second | first-only | second-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| A | 100 | 74 | 74 | 0 | 0 | 1.000 |
| B | 100 | 85 | 85 | 0 | 0 | 1.000 |

Within 9Y regardless of arm: first-executed fork held 84/100, second-executed 75/100 (first-only 18, second-only 9, McNemar p=0.122).

## 5. Reproducibility of each fork vs 9X

| arm | n | identical text + states | same states | same hold outcome |
|---|---:|---:|---:|---:|
| A | 100 | 99 | 99 | 100 |
| B | 100 | 99 | 99 | 100 |

Identical = every round's state, plan, final text, tool list and sensitive paths match 9X exactly. If forks are order-independent and generation is deterministic, every row is n/n.
