# Phase 9Z Z3 — paired B vs BH (H on in-round retries)

> Preregistered (docs/sync_channel_control.md, Phase 9Z). Target 111, T=5 retention rounds, seed base 71000000. Arm B = frozen 9X B {'C': 0.5, 'H': 1.5, 'O': 1.5}. Rounds continue after a failure; no repair. Hardware cohort: NVIDIA A40, torch 2.8.0+cu129, transformers 5.24.0, CUDA 12.9, float32 — not pooled with 9W–9Y (RTX A6000).

Acquired 34/98 (P=0.35 [0.26, 0.45]); PARTIAL.

### B: hold all 5 rounds = **19/34** [0.39, 0.71]

| round k | entered at 111 | stayed | $P(S_k{=}111\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |
|---:|---:|---:|---:|---|---:|---|
| 1 | 34 | 31 | 0.91 | [0.77, 0.97] | 0.91 | 0/34 |
| 2 | 31 | 26 | 0.84 | [0.67, 0.93] | 0.76 | 3/34 |
| 3 | 26 | 23 | 0.88 | [0.71, 0.96] | 0.68 | 6/34 |
| 4 | 23 | 20 | 0.87 | [0.68, 0.95] | 0.59 | 6/34 |
| 5 | 21 | 19 | 0.90 | [0.71, 0.97] | 0.56 | 9/34 |

- first drift round: {1: 3, 2: 5, 3: 3, 4: 3, 5: 1}
- departures per conversation: {0: 19, 1: 14, 2: 1}; returns to 111: 1
- first failing bits: {'C': 8, 'H': 4, 'C+H': 3}
- first-failure mechanism: {'C_wording': 3, 'C_empty_plan': 5, 'H_no_tool': 7}

### BH (B + H on retries): hold all 5 rounds = **19/34** [0.39, 0.71]

| round k | entered at 111 | stayed | $P(S_k{=}111\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |
|---:|---:|---:|---:|---|---:|---|
| 1 | 34 | 31 | 0.91 | [0.77, 0.97] | 0.91 | 0/34 |
| 2 | 31 | 26 | 0.84 | [0.67, 0.93] | 0.76 | 3/34 |
| 3 | 26 | 23 | 0.88 | [0.71, 0.96] | 0.68 | 6/34 |
| 4 | 23 | 20 | 0.87 | [0.68, 0.95] | 0.59 | 6/34 |
| 5 | 21 | 19 | 0.90 | [0.71, 0.97] | 0.56 | 9/34 |

- first drift round: {1: 3, 2: 5, 3: 3, 4: 3, 5: 1}
- departures per conversation: {0: 19, 1: 14, 2: 1}; returns to 111: 1
- first failing bits: {'C': 8, 'H': 5, 'C+H': 2}
- first-failure mechanism: {'C_wording': 3, 'C_empty_plan': 5, 'H_no_tool': 7}

## Primary: paired BH vs B on $P(S_1{=}\dots{=}S_5{=}111\mid S_0{=}111)$

Both hold 19, BH-only **0**, B-only **0**. Exact McNemar p = **1.000**.

## Steering attribution (first round in which the BH trigger fires)

- conversations where it fired: 10 (identical to B before the trigger: 10)
- outcomes in that round: {'both_fail': 8}
- BH lost C or O while H held (and B kept it): 0

Mechanisms of failed rounds entered at 111: {'B': {'C_wording': 3, 'C_empty_plan': 5, 'H_no_tool': 8}, 'BH': {'C_wording': 3, 'C_empty_plan': 5, 'H_no_tool': 8}}

## Decision (preregistered)

- hold(BH) ≥ hold(B): True; BH-only > B-only: False; rescued > broken: False
- **BH not adopted; B remains**
