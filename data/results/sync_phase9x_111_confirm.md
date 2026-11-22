# Phase 9X — preregistered paired confirmation for 111 (A vs B)

> New seeds (disjoint from 9U, 9V and 9W). Arms frozen from 9W; no retuning. Each conversation acquired at round 0 under full 8K is forked to A and B in a random order drawn after acquisition, with identical round seeds. Target-sign, 8K sites, forced FINAL:. A flip is a failure.

rounds=2, seed=51000000, frozen config `data/configs/phase9x_frozen_111.json`. Environment: torch 2.8.0+cu129, transformers 5.24.0, CUDA 12.9, NVIDIA RTX A6000, float32.

## Primary: paired B vs A on $P(S_1{=}S_2{=}111\mid S_0{=}111)$

| arm | gains (C, H, O) | hold | 95% CI |
|---|---|---:|---|
| A (baseline (9V frozen)) | C=0, H=1.5, O=1.5 | 74/100 | [0.65, 0.82] |
| B (candidate (9W arm B)) | C=0.5, H=1.5, O=1.5 | 85/100 | [0.77, 0.91] |

Paired: both hold 66, B-only **19**, A-only **8**, neither 7. Exact McNemar p = **0.052**.

**Decision (preregistered rule):** B confirmed as candidate; not shown better.

## Secondary: failure anatomy

| arm | fails | C alone | C with H | C involved | first flip | final state | first-fail round |
|---|---:|---:|---:|---:|---|---|---|
| A | 26 | 13 | 7 | 20 | C×13, C+H×7, H×6 | `011`×12, `001`×7, `101`×7 | R2×13, R1×13 |
| B | 15 | 8 | 3 | 11 | C×8, H×4, C+H×3 | `011`×8, `101`×4, `001`×3 | R2×9, R1×6 |

## Acquisition and end-to-end

Acquired 100/323 (P=0.31 [0.26, 0.36]); arm order AB×63, BA×37; complete.

| arm | e2e $P(S_0{=}S_1{=}S_2{=}111)$ | 95% CI |
|---|---:|---|
| A | 0.23 | [0.19, 0.28] |
| B | 0.26 | [0.22, 0.31] |

Rule fixed before the run: B is confirmed as the 111 candidate if hold(B) ≥ hold(A) and B-only > A-only; B is called better than A only if McNemar p < 0.05. Do not pool with 9V (MPS); 9W ran in the same CUDA environment but on different seeds and is not pooled here either.
