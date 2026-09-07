# Collapse-avoidance (locked before the existing-data test)

Phase 1b closed **performance gain** (`NOISE_STOP`). This is a **different**
question on the same draws: do selector-picked bottom sets avoid catastrophic
collapse more often than random channel sets?

Does **not** reopen Δcorrect / p90. Does not retune k for gain.

---

## Collapse (locked)

A draw is `COLLAPSE` iff `control_correct == 0` (all n=24 trials fail).

On Phase 1b this co-occurs with uniq crash (~0.43–0.63 vs identity 0.92) and
`heldout_ppl` jump (~8 vs identity 5.3). Those are logged as confirmation, not
extra gates. No other Phase 1b draw has `correct == 0`.

---

## Primary test (existing Phase 1b only — no new generation)

| arm | n | source |
|---|---:|---|
| selector | 4 | locked `nll_bottom_{16,64}`, `gxa_bottom_{16,64}` |
| random | 20 | 10× random_16 + 10× random_64 |

**H1 (one-sided):** p_collapse(selector) < p_collapse(random).

**Test:** Fisher exact on the 2×2 table. Also report Clopper–Pearson 95% CIs
and `P(0 collapses | Binomial(n=4, p̂_random))`.

**Independence caveat (locked, not optional):** the four selector cells are
nested (`bottom_16 ⊂ bottom_64` within method). Treating them as n=4
**overstates** evidence. Sensitivity: 2 methods (collapse if either k
collapsed) vs 20 random; and k=64-only (2 selector vs 10 random_64).

**Decision (locked):**
- `AVOID_PROMISING` iff one-sided Fisher **p < 0.10** on the primary 0/4 vs
  3/20 table → license a powered follow-up (more k, more random draws).
- else `AVOID_UNRESOLVED` — existing n cannot distinguish 0/4 from chance
  given ~15% random landmines. Do **not** expand. Do **not** claim a safety
  margin. Fold with the gain `NOISE_STOP`.

Phase 1a (n=12) is a robustness note only; not pooled into the primary table.
