# Dimension selection — random baseline, then (if earned) k / LOO / holdout

Locked before Phase 1. Site: L4 residual channels, GAP `family=control`.
Primary: `control_correct`. In-sample until Phase 5. Do not retune k or sets after
seeing Phase 5. Do not invent a confirm set from eliciting/baseline.

**Phase 1a (done):** n=12, 10× random_{16,64}. ΔNLL bottom-64 (+0.250) ≥
p90(+0.175) → provisional `SIGNAL`, but 1/10 random_64 also hit +0.250.

**Phase 1b (authoritative for the p90 gate):** same protocol on **n=24**
(original 12 + 12 new legitimate tasks, 2×6 domains). Ranking stays **frozen**
from the original-12 ΔNLL / g×a extract. Effect sizes and random draws are
re-measured on all 24. Log `correct`, `viol`, `refuse`, `uniq`, `gen_ppl`,
`heldout_ppl`.

**Locked ranking artifacts** (not re-fit on the expanded set):

| selector | source |
|---|---|
| ΔNLL | `data/results/channel_nll_delta.npy` (argsort \|Δ\|) |
| g×a | recovered once on original-12 identity rollouts in the Phase 1b job |

---

## Phase 1 — random baseline

10 independent `random_16` and 10 independent `random_64`.
Seeds: `20260812 + i` for `i = 0..9`.

Report per k: mean, sd, min, max, empirical 90th percentile
(`numpy.percentile(..., 90)`, linear). Also report the empirical percentile
of the locked ΔNLL bottom-64 Δcorrect among the 10 random_64 draws.

**P1 decision (locked; Phase 1b is the gate):**
- ΔNLL bottom-64 (n=24 Δcorrect) is **SIGNAL** iff ≥ p90 of the 10
  `random_64` Δcorrect draws on the same n=24.
- Else **NOISE_STOP**. Do not retune k. Do not run Phases 2–5 on these selectors.
- g×a bottom-64 scored on the same bar (secondary).

If **NOISE_STOP**: clean negative — reopen selector question (serious SAE, or
fold into open threads). Do not rescue by re-sampling.

---

## Phase 2 — reconcile g×a vs ΔNLL (only if P1 SIGNAL for at least one)

Full bottom-64 index sets (not `channels_head`). Jaccard + intersection count.

- Mostly overlapping (Jaccard ≥ 0.50): score disagreement is noise on a shared
  ranking — either method.
- Mostly disjoint (Jaccard < 0.25): different properties. Compare mean |h|,
  variance, and |s| / |ΔNLL| of the two bottoms vs their tops.
- Mid: report and treat as weak overlap.

**P2 decision:** continue with a method that (a) passed P1 and (b) has a clear
top-vs-bottom gap on its own score (not a coin flip on the rank statistic).

---

## Phase 3 — k sweep on the survivor

k ∈ {16, 32, 64, 128, 256, 512}. Same cheap metrics
(correct, violation, refusal, uniq, gen_ppl, heldout_ppl).

Look at Δcorrect vs k: cliff (PCA-style) vs gradual. Headroom = largest k with
Δcorrect ≥ 0 and no uniq collapse (uniq Δ > −0.05 vs identity).

---

## Phase 4 — LOO rank stability

Leave-one-out recompute of the surviving ranking on the **original extract-12**
(ranking pool). Spearman of full vs each LOO ranking. Bottom-k set overlap
(the k chosen in P3).

**P4 decision:** if mean pairwise Jaccard of LOO bottom-k sets < 0.50 →
`UNSTABLE_RANK`, stop. Finding is an 8/4 artifact.

---

## Phase 5 — fresh confirm (only if P1–P4 all pass)

**New** 12 legitimate tasks beyond the Phase 1b expansion (another 2×6).
The Phase 1b +12 are **not** the confirm fold — they sharpened the random
baseline. Same pre-commit as mean-diff confirm. **Do not build until P1–P4 pass.**

Fixed k, fixed channel set, no retune.

**Pass bar (locked now):** Δcorrect ≥ +0.08 on the fresh 12, no PPL/diversity
collapse (uniq Δ > −0.05), sign consistent with Phases 1–3.

Tiers: PASS / SUGGESTIVE / NULL.

---

## What this does not do

- Does not 1024×generate `control_correct` ranking.
- Does not graduate mean-diff P1.
- Does not loosen J2.
- Does not re-fit the channel ranking on the expanded 24.
