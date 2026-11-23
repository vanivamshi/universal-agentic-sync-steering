# Phase 9Z Z3 — paired B vs BH (H on in-round retries)

> Preregistered (docs/sync_channel_control.md, Phase 9Z). Target 111, T=5 retention rounds, seed base 71000000. Arm B = frozen 9X B {'C': 0.5, 'H': 1.5, 'O': 1.5}. Rounds continue after a failure; no repair. Hardware cohort: NVIDIA A40, torch 2.8.0+cu129, transformers 5.24.0, CUDA 12.9, float32 — not pooled with 9W–9Y (RTX A6000).

Acquired 50/136 (P=0.37 [0.29, 0.45]); complete.

**Deviation from preregistration:** sample reduced from preregistered n=100 to n=50 for time; first n acquired in attempt order kept.

Run with 3 parallel workers after resume (attempt i run by worker i % shards; merged in attempt order, first n_acq acquired kept); each worker used an exact copy of the stage's sandbox, and a re-run of an already finished conversation under parallel load was identical.

### B: hold all 5 rounds = **25/50** [0.37, 0.63]

| round k | entered at 111 | stayed | $P(S_k{=}111\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |
|---:|---:|---:|---:|---|---:|---|
| 1 | 50 | 46 | 0.92 | [0.81, 0.97] | 0.92 | 0/50 |
| 2 | 46 | 38 | 0.83 | [0.69, 0.91] | 0.76 | 5/50 |
| 3 | 38 | 31 | 0.82 | [0.67, 0.91] | 0.62 | 11/50 |
| 4 | 31 | 27 | 0.87 | [0.71, 0.95] | 0.54 | 9/50 |
| 5 | 29 | 25 | 0.86 | [0.69, 0.95] | 0.50 | 14/50 |

- first drift round: {1: 4, 2: 8, 3: 7, 4: 4, 5: 2}
- departures per conversation: {0: 25, 1: 23, 2: 2}; returns to 111: 2
- first failing bits: {'C': 14, 'H': 8, 'C+H': 3}
- first-failure mechanism: {'C_wording': 4, 'C_empty_plan': 9, 'H_no_tool': 11, 'C_contradict': 1}

### BH (B + H on retries): hold all 5 rounds = **25/50** [0.37, 0.63]

| round k | entered at 111 | stayed | $P(S_k{=}111\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |
|---:|---:|---:|---:|---|---:|---|
| 1 | 50 | 46 | 0.92 | [0.81, 0.97] | 0.92 | 0/50 |
| 2 | 46 | 38 | 0.83 | [0.69, 0.91] | 0.76 | 5/50 |
| 3 | 38 | 31 | 0.82 | [0.67, 0.91] | 0.62 | 11/50 |
| 4 | 31 | 27 | 0.87 | [0.71, 0.95] | 0.54 | 9/50 |
| 5 | 29 | 25 | 0.86 | [0.69, 0.95] | 0.50 | 14/50 |

- first drift round: {1: 4, 2: 8, 3: 7, 4: 4, 5: 2}
- departures per conversation: {0: 25, 1: 23, 2: 2}; returns to 111: 2
- first failing bits: {'C': 14, 'H': 9, 'C+H': 2}
- first-failure mechanism: {'C_wording': 4, 'C_empty_plan': 9, 'H_no_tool': 11, 'C_contradict': 1}

## Primary: paired BH vs B on $P(S_1{=}\dots{=}S_5{=}111\mid S_0{=}111)$

Both hold 25, BH-only **0**, B-only **0**. Exact McNemar p = **1.000**.

## Steering attribution (first round in which the BH trigger fires)

- conversations where it fired: 16 (identical to B before the trigger: 16)
- outcomes in that round: {'both_fail': 13}
- BH lost C or O while H held (and B kept it): 0

Mechanisms of failed rounds entered at 111: {'B': {'C_wording': 4, 'C_empty_plan': 9, 'H_no_tool': 13, 'C_contradict': 1}, 'BH': {'C_wording': 4, 'C_empty_plan': 9, 'H_no_tool': 13, 'C_contradict': 1}}

## Decision (preregistered)

- hold(BH) ≥ hold(B): True; BH-only > B-only: False; rescued > broken: False
- **BH not adopted; B remains**
