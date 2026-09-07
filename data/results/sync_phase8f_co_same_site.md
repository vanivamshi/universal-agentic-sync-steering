# Phase 8F — C/O same-site causal control

> 8C-equivalent for O + episode-conditioned C. H already validated. No new $v$.

α=1.5, n=12, seed=20260915, stage=both, channels=('O', 'C').

## Geometry (frozen $h_0$)

| Channel | $\Delta B(+\alpha)$ | $\Delta B(-\alpha)$ | Phase7 ref |
|---------|---------------------:|--------------------:|-----------:|
| O | +0.156 | -0.156 | ±0.156 |
| C | +0.082 | -0.082 | ±0.082 |

## Stage A — same-site decision

| Channel | condition | $E[G]$ | $P(W\to C)$ | n_wrong |
|---------|-----------|-------:|------------:|--------:|
| O | no_steer | +0.000 | 0.583 | 12 |
| O | wrong_sign | -0.156 | 0.583 | 12 |
| O | target_sign | +0.156 | 0.583 | 12 |
| C | no_steer | +0.000 | 0.833 | 12 |
| C | wrong_sign | -0.082 | 0.500 | 12 |
| C | target_sign | +0.082 | 0.917 | 12 |

### Stage A gates

- **O** pass=True: TS P(W→C)=0.583 vs WS 0.583 / none 0.583 (Phase7 flip ref 0.17)
- **C** pass=True: TS P(W→C)=0.917 vs WS 0.500 / none 0.833 (Phase7 flip ref 0.42)

## Stage B — decision-token-only

| Channel | condition | $E[G]$ | $P(W\to C)$ | n_wrong |
|---------|-----------|-------:|------------:|--------:|
| O | no_steer | +0.000 | 0.000 | 12 |
| O | wrong_sign | -0.156 | 0.000 | 12 |
| O | target_sign | +0.156 | 0.000 | 12 |
| C | no_steer | +0.000 | 0.500 | 12 |
| C | wrong_sign | -0.082 | 0.500 | 12 |
| C | target_sign | +0.082 | 0.500 | 12 |

### Stage B gates

- **O** pass=False: TS P(W→C)=0.000 vs WS 0.000 / none 0.000
- **C** pass=False: TS P(W→C)=0.500 vs WS 0.500 / none 0.500

H was already ✓ in Phase 8C. Next 8-way only after C and O pass local causal control.

No new $v$. No policy. No aggregate 8-way in this phase.
