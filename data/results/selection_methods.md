# Three selection methods (your list)

Same site: L4 residual, n=12 GAP control, n_pos=8 n_neg=4, in-sample only.
Primary confirmatory metric: `control_correct`. Holdout not licensed.

| # | What you asked for | What ran | File | Pattern |
|---|---|---|---|---|
| 1 | Per-dimension causal ablation: zero each channel, **rank by Δperformance** | Zero each of 1024 channels, **rank by Δ teacher-forced NLL**, then confirmatory generate on top/bottom/random k | `channel_attribution_screen.md` § per-channel ΔNLL | `SELECTOR_WORKS` |
| 2 | Gradient×activation vs task-success, one backward pass | g×a vs NLL of the greedy continuation (control_correct is not differentiable), then same confirmatory | `channel_attribution_screen.md` § g×a | `SELECTOR_WORKS` |
| 3 | SAE features (different basis; train if no pretrained) | Throwaway ReLU SAE 4096 feats on GAP windows (2007 tokens), rank by contrastive latent | `sae_feature_screen.md` | `NULL` |

## Confirmatory `control_correct` (identity = 0.667)

| condition | #1 ΔNLL-rank | #2 g×a | #3 SAE |
|---|---:|---:|---:|
| bottom_16 | 0.833 | 0.667 | 0.667 |
| top_16 | **0.000** | **0.000** | 0.833 |
| random_16 | 0.750 | 0.750 | 0.750 |
| bottom_64 | 0.917 | 0.667 | 0.833 |
| top_64 | **0.000** | **0.000** | 0.500 |
| random_64 | 0.833 | 0.833 | 0.750 |

#1 was **not** a 1024×generate `control_correct` sweep. Ranking used ΔNLL; Δperformance was measured only on the ranked k-sets. g×a top-16 uniqueness collapsed (0.90→0.28).
