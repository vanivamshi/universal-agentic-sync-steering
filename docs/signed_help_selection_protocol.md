# Signed-help channel selection (locked before run)

`|ΔNLL|` ranking is **closed** (`NOISE_STOP`). It selects *inert* channels
(small |effect|), not channels whose removal is predicted to **help**.

This selector uses the **sign** of the same frozen per-channel ΔNLL
(`channel_nll_delta.npy`, original-12 teacher-forced continuations).
It does not re-fit the ranking on n=24. It does not reuse `|Δ|` bottom-k.

---

## Score (frozen)

`ΔNLL_i` = mean NLL(identity continuation | channel i zeroed) − mean NLL(identity).

- `ΔNLL_i > 0`: zeroing *hurts* the observed continuation (landmine / load-bearing).
- `ΔNLL_i < 0`: zeroing *helps* that continuation (candidate to remove).

**Help-k** = the k channels with the **most negative** ΔNLL.
Not smallest |ΔNLL|. Not g×a `|s|`.

k ∈ {16, 64} (same sizes as the closed random baseline).

---

## Confirm (n=24, same collect as Phase 1b)

Zero help-k at L4, greedy generate, same 24 controls.
Primary: `control_correct`. Log viol, refuse, uniq, gen_ppl, heldout_ppl.
Identity = Phase 1b identity (T=0, already measured). Do not re-draw random.

**Random bar (locked, already computed):**

| k | Phase 1b random p90 Δcorrect |
|---|---:|
| 16 | +0.092 |
| 64 | +0.133 |

**Decision (locked):**
- `HELP_SIGNAL` iff help-64 Δcorrect ≥ random_64 p90 **and** not COLLAPSE
  (`control_correct == 0`).
- help-16 is supporting (same bar vs random_16 p90).
- else `HELP_NOISE` — signed-help is also not better than chance. Stop.
  Do not retune k. Do not fall back to `|Δ|` bottom.

Collapse on help-k is `HELP_LANDMINE` (the signed tail was still load-bearing).

---

## What this does not do

- Does not reopen `|ΔNLL|` / g×a bottom-k.
- Does not 1024×generate `control_correct`.
- Does not claim holdout. n=24 is the same measurement set as Phase 1b.
