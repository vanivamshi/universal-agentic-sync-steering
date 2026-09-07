# L4 SAE feature selection (locked before run)

Third selector. g×a and per-channel ΔNLL rank residual *channels*. This ranks
**SAE features** at the same site, then confirmatory-ablates decoder columns.

**Not a published SAE for this model.** Train a throwaway ReLU SAE on GAP
window residuals. In-sample only. Same 12 control / 8/4 floor as mean-diff.

---

## Train (T)

- Site: L4 post-block residual.
- Corpus: tokens inside `data/windows/real_token_windows.jsonl` windows
  (prose + tool_call), cap 8192 tokens, seed **20260811**.
- Architecture: ReLU SAE, `n_features = 4 × hidden` (4096), decoder rows
  unit-normalized after each step, L1 on latents.
- Fail closed: if dead features (`mean f=0`) > 90% after train → `SAE_DEAD`.

This SAE is a **basis for ranking**, not a claim that features are
interpretable or that the dict is complete.

---

## Score (S)

Same 12 `family=control` greedy identity rollouts as the other selectors.

Last-prompt-token L4 residual → encode `f = ReLU((h − b_dec) W_enc + b_enc)`.

`s_j = mean_neg(f_j) − mean_pos(f_j)`. Rank by `|s_j|`.

**S1.** n_pos&lt;4 or n_neg&lt;4 → SHELVE.

---

## Confirmatory (C)

During generation, subtract selected features from the residual at every token:

`h ← h − Σ_{j∈S} f_j(h) d_j`

| condition | set |
|---|---|
| identity | no ablation |
| bottom-k | k lowest `|s|` |
| top-k | k highest `|s|` |
| random-k | seed 20260811 |

k ∈ {16, 64}. Primary: `control_correct`. Same C1 pattern names as
`docs/channel_attribution_protocol.md` (`SELECTOR_WORKS` / `PROXY_FAIL` /
`INVERTED` / `BOTH_HURT` / `NULL`).

Reporting: in-sample only. No holdout.
