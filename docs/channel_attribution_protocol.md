# L4 channel selection via gradient×activation (locked before run)

PCA ranks by variance (wrong). Mean-diff *constructs* one direction per
hand-defined outcome (not a general selector). This protocol is a **selector**:
score all 1024 L4 residual channels, then confirmatory-ablate ranked sets.

**Method chosen:** gradient×activation (cheap). Full per-channel causal sweep
(1024 forward gens) is deferred unless this proxy fails the confirmatory test.

**Not differentiable:** `control_correct` is a discrete rollout label.
**Surrogate (locked):** teacher-forced NLL of the unsteered greedy continuation,
contrastive across labels.

---

## Attribution (A)

Site: L4 residual, **last prompt token** (same site as mean-diff).

For each `family=control` prompt (n=12):
1. Greedy unsteered rollout → label `control_correct` + assistant text.
2. Teacher-force that continuation; NLL on assistant tokens only.
3. `I^{(t)} = h ⊙ ∂NLL/∂h` at the last prompt token.

**Contrastive score** (target = control_correct):
`s_i = mean_{neg} I_i − mean_{pos} I_i`

Rank by `|s_i|`. High = predicted load-bearing for the success/failure gap.
Low = predicted inert for that gap.

**A1.** If n_pos&lt;4 or n_neg&lt;4 → SHELVE (same floor as mean-diff).

This is a **local linear proxy**, not a causal effect. Ranking is not a claim
until confirmatory ablation.

---

## Confirmatory ablation (C) — in-sample, same 12

Zero selected channels at L4 (all tokens) during generation.

| condition | set |
|---|---|
| identity | no zeroing (reuse A rollouts) |
| bottom-k | k lowest `|s_i|` |
| top-k | k highest `|s_i|` |
| random-k | uniform sample, seed **20260811** (independent of top/bottom) |

k ∈ **{16, 64}**. Primary: `control_correct`. Log the same cheap side metrics
as the ablation/mean-diff screens. No holdout. No eliciting/baseline pooling.

**C1 pattern (locked, Δ vs identity, threshold 1/12):**

| pattern | rule | meaning |
|---|---|---|
| `SELECTOR_WORKS` | bottom-16 Δ_correct ≥ −1/12 **and** top-16 Δ_correct ≤ −1/12 | inert vs load-bearing split is causal *on this sample* |
| `PROXY_FAIL` | top-16 does not drop (Δ > −1/12) | g×a did not find load-bearing channels |
| `INVERTED` | bottom drops and top does not | ranking sign/magnitude wrong |
| `BOTH_HURT` | bottom-16 and top-16 both Δ ≤ −1/12 | not selective; zeroing k channels is just damage |
| `NULL` | neither k=16 set moves ≥1/12 | no signal at this k |

k=64 is supporting (expect more damage). Random-k is the non-ranked control.

**Reporting:** in-sample only. `SELECTOR_WORKS` ≠ validated selector for new
tasks. Same ceiling as mean-diff until a fresh benign set exists.

---

## What this does not do

- Does not replace mean-diff as a *steering vector* (different job).
- Does not train an SAE.
- Does not zero 1024 channels one-at-a-time.
