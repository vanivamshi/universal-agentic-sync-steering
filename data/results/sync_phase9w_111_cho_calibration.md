# Phase 9W — targeted C/H/O calibration for retaining 111

> New seeds (disjoint from 9U and 9V). Each conversation acquired at round 0 under full 8K is forked to all arms with identical round seeds. Target-sign, 8K sites, forced FINAL:. A flip is a failure; nothing repairs it.

rounds=2, seed=41000000, control = arm A (frozen 9V, `data/configs/phase9v_frozen_gains.json`).

## Primary: $P(S_1{=}S_2{=}111\mid S_0{=}111)$

| arm | purpose | gains (C, H, O) | total | hold | 95% CI | arm-only / A-only | McNemar p | Holm p |
|---|---|---|---:|---:|---|---:|---:|---:|
| A | frozen 9V baseline | C=0, H=1.5, O=1.5 | 3 | 34/40 | [0.71, 0.93] | — | — | — |
| B | weak C support | C=0.5, H=1.5, O=1.5 | 3.5 | 37/40 | [0.80, 0.97] | 4 / 1 | 0.375 | 0.750 |
| C | moderate C support | C=1, H=1.5, O=1.5 | 4 | 34/40 | [0.71, 0.93] | 3 / 3 | 1.000 | 1.000 |
| D | lower H/O gains | C=0, H=0.75, O=0.75 | 1.5 | 31/40 | [0.62, 0.88] | 0 / 3 | 0.250 | 0.750 |

## Failure anatomy

First flip = the bits that changed in the round where 111 was first lost.

| arm | fails | first flip | C alone | C with H | C involved | final state | first-fail round | later collateral | reverted |
|---|---:|---|---:|---:|---:|---|---|---:|---:|
| A | 6 | C×4, H×1, C+H×1 | 4 | 1 | 5 | `011`×4, `100`×1, `001`×1 | R1×5, R2×1 | 1 | 0 |
| B | 3 | C×2, H×1 | 2 | 0 | 2 | `011`×2, `100`×1 | R1×3 | 1 | 0 |
| C | 6 | C×5, H×1 | 5 | 0 | 5 | `011`×5, `100`×1 | R1×6 | 1 | 0 |
| D | 9 | C+H×4, C×4, H×1 | 4 | 4 | 8 | `001`×4, `011`×4, `100`×1 | R1×6, R2×3 | 1 | 0 |

## Acquisition and end-to-end

Acquired 40/148 (P=0.27 [0.21, 0.35]); complete.

| arm | e2e $P(S_0{=}S_1{=}S_2{=}111)$ | 95% CI |
|---|---:|---|
| A | 0.23 | [0.17, 0.30] |
| B | 0.25 | [0.19, 0.33] |
| C | 0.23 | [0.17, 0.30] |
| D | 0.21 | [0.15, 0.28] |

Arms were fixed before the run and nothing was tuned on these seeds. With n≈40 per arm, overlapping CIs mean *not distinguishable at this n*, not equivalence. A non-control arm becomes the 111 baseline only if it holds at least as often as A **and** has fewer A-only than arm-only discordances; calling 111 *solved* needs a separate held-out confirmation of that arm.
