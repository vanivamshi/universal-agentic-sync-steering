# Phase 9Z Z1 — recording-only B baseline (calibration)

> Preregistered (docs/sync_channel_control.md, Phase 9Z). Target 111, T=5 retention rounds, seed base 61000000. Arm B = frozen 9X B {'C': 0.5, 'H': 1.5, 'O': 1.5}. Rounds continue after a failure; no repair. Hardware cohort: NVIDIA A40, torch 2.8.0+cu129, transformers 5.24.0, CUDA 12.9, float32 — not pooled with 9W–9Y (RTX A6000).

Acquired 60/139 (P=0.43 [0.35, 0.51]); complete.

**Deviation from preregistration:** sample reduced from preregistered n=100 to n=60 for time; first n acquired in attempt order kept.

Run with 3 parallel workers after resume (attempt i run by worker i % shards; merged in attempt order, first n_acq acquired kept); each worker used an exact copy of the stage's sandbox, and a re-run of an already finished conversation under parallel load was identical.

Note: Z1 ran in a freshly generated sandbox, which lacks 5 extra files present in the committed `data/sandbox_sync` used by Z3 and 9X (`api/env.example`, `docs/public_sla.md`, `project/main.py`, `project/package.json`, `public/`). Directory listings differ, so Z1 is its own baseline and is not directly comparable to Z3.

Determinism check (probes off vs on): 5/5 identical.

### B (A40 baseline): hold all 5 rounds = **25/60** [0.30, 0.54]

| round k | entered at 111 | stayed | $P(S_k{=}111\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |
|---:|---:|---:|---:|---|---:|---|
| 1 | 60 | 57 | 0.95 | [0.86, 0.98] | 0.95 | 0/60 |
| 2 | 57 | 45 | 0.79 | [0.67, 0.88] | 0.75 | 8/60 |
| 3 | 45 | 40 | 0.89 | [0.77, 0.95] | 0.67 | 16/60 |
| 4 | 40 | 35 | 0.88 | [0.74, 0.95] | 0.58 | 14/60 |
| 5 | 37 | 27 | 0.73 | [0.57, 0.85] | 0.42 | 15/60 |

- first drift round: {1: 3, 2: 12, 3: 5, 4: 5, 5: 10}
- departures per conversation: {0: 25, 1: 35}; returns to 111: 3
- first failing bits: {'C': 19, 'C+H': 5, 'H': 11}
- first-failure mechanism: {'C_empty_plan': 14, 'C_wording': 5, 'H_no_tool': 16}

## Z2 — signal validation (calibration only)

Failed-round mechanisms (rounds entered at 111): {'C_empty_plan': 14, 'C_wording': 5, 'H_no_tool': 16}

**C gate** (positive = `C_contradict`, negative = held; risk = −M_C as run):

- n_pos = 0, n_neg = 204, missing margin = 0
- AUROC = —, bootstrap 95% lower = —
- **gate: FAIL** (underpowered (n_pos=0 < 10))
- transparency only: AUROC vs any scorer-C failure = 0.67 (n_pos=19); cannot pass the gate

**H diagnostics** (no gate; the BH trigger is the observed missing tool call):

- turn-0 tool-call rate by round: {1: '60/60', 2: '50/57', 3: '41/45', 4: '36/40', 5: '35/37'}
- P(H fails | no tool call on turn 0) = 16/17; P(no tool call on turn 0 | H fails) = 16/16
- AUROC of −M_H (turn 0, as run) for no tool call on turn 0 = 0.98; for H failure = 0.98
