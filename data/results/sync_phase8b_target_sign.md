# Phase 8B — Target-signed live-boundary control

> $d_k(m_k^*)=s_k v_c^k$, $s_k=2m_k^*-1$. No new $v$.

α=1.5, n=16 free runs. H primary.

## Activation-space $G$ (local $h\pm s\alpha v$)

| Channel | TS mean $G$ | TS frac $G{>}0$ | WS mean $G$ |
|---------|-------------|----------------|-------------|
| H | +0.412 | 1.00 | -0.412 |

## Behavioral single-channel (episode re-run)

| Channel | mode | mean $G$ | frac $G{>}0$ | P(W→C) | n_wrong |
|---------|------|----------|--------------|--------|---------|
| H | target_sign | -0.839 | 0.38 | 0.31 | 16 |
| H | wrong_sign | +0.685 | 0.47 | 0.19 | 16 |

### H by $m^*$ (target-sign)

| $H^*$ | mean $G$ | P(W→C) | n_wrong |
|-------|----------|--------|---------|
| 0 | -1.882 | 0.40 | 5 |
| 1 | +0.204 | 0.27 | 11 |

## Gate (H primary)

- Act TS mean $G_H>0$: **True**
- Beh TS mean $G_H>0$: **False**
- Beh TS beats wrong-sign on $G$: **False**
- P(W→C)_H target-sign: **0.3125** (Phase 8A was 0.11)
- P(W→C)_H wrong-sign: **0.1875**
- P(W→C) improved vs 0.11: **True**
- Sign matters for flips: **True**
- Controller bug is sign: **False**

If target-sign lifts P(W→C)_H ≫ 0.11 and G_H>0 vs wrong-sign: sign was the bug. If not: investigate H timing/capture, not representation.

No new $v$. No full 8-way yet.
