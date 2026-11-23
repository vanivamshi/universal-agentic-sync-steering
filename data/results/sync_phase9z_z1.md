# Phase 9Z Z1 — recording-only B baseline (calibration)

> Preregistered (docs/sync_channel_control.md, Phase 9Z). Target 111, T=5 retention rounds, seed base 61000000. Arm B = frozen 9X B {'C': 0.5, 'H': 1.5, 'O': 1.5}. Rounds continue after a failure; no repair. Hardware cohort: NVIDIA A40, torch 2.8.0+cu129, transformers 5.24.0, CUDA 12.9, float32 — not pooled with 9W–9Y (RTX A6000).

Acquired 54/132 (P=0.41 [0.33, 0.49]); PARTIAL.

Determinism check (probes off vs on): 5/5 identical.

### B (A40 baseline): hold all 5 rounds = **23/54** [0.30, 0.56]

| round k | entered at 111 | stayed | $P(S_k{=}111\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |
|---:|---:|---:|---:|---|---:|---|
| 1 | 54 | 51 | 0.94 | [0.85, 0.98] | 0.94 | 0/54 |
| 2 | 51 | 41 | 0.80 | [0.68, 0.89] | 0.76 | 7/54 |
| 3 | 41 | 36 | 0.88 | [0.74, 0.95] | 0.67 | 15/54 |
| 4 | 36 | 31 | 0.86 | [0.71, 0.94] | 0.57 | 14/54 |
| 5 | 33 | 25 | 0.76 | [0.59, 0.87] | 0.43 | 15/54 |

- first drift round: {1: 3, 2: 10, 3: 5, 4: 5, 5: 8}
- departures per conversation: {0: 23, 1: 31}; returns to 111: 3
- first failing bits: {'C': 16, 'C+H': 5, 'H': 10}
- first-failure mechanism: {'C_empty_plan': 11, 'C_wording': 5, 'H_no_tool': 15}

## Z2 — signal validation (calibration only)

Failed-round mechanisms (rounds entered at 111): {'C_empty_plan': 11, 'C_wording': 5, 'H_no_tool': 15}

**C gate** (positive = `C_contradict`, negative = held; risk = −M_C as run):

- n_pos = 0, n_neg = 184, missing margin = 0
- AUROC = —, bootstrap 95% lower = —
- **gate: FAIL** (underpowered (n_pos=0 < 10))
- transparency only: AUROC vs any scorer-C failure = 0.65 (n_pos=16); cannot pass the gate

**H diagnostics** (no gate; the BH trigger is the observed missing tool call):

- turn-0 tool-call rate by round: {1: '54/54', 2: '45/51', 3: '37/41', 4: '32/36', 5: '31/33'}
- P(H fails | no tool call on turn 0) = 15/16; P(no tool call on turn 0 | H fails) = 15/15
- AUROC of −M_H (turn 0, as run) for no tool call on turn 0 = 0.99; for H failure = 0.99
