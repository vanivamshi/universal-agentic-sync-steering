# Firm random baseline for signed-help (locked before expand)

Provisional `HELP_SIGNAL` tied the **max** of only 10 random_64 draws
(+0.208). Empirical p90 from n=10 is noisy. Sign-flip polarity (`SIGN_OK`)
stands; this gate only firms the **random bar**.

---

## Expand (R)

- Keep Phase 1b random_64 seeds `20260812..20260821` (n=10, already measured).
- Add **30** new seeds: `20260822..20260851` → **N=40** random_64 total.
- Same n=24 controls, L4 ChannelZero, T=0. Primary metric: `control_correct`.
- Identity = Phase 1b locked identity (correct=0.583). Do not re-fit help
  ranking. Do not re-measure help_64 (locked Δ=+0.208).
- `heldout_ppl` may be skipped on expand draws (landmine seeds were multi×
  slower; gate does not use held_ppl). Phase 1b reused draws keep their held_ppl.

No new random_16 in this job (headline is help_64).

---

## Percentile + bootstrap CI (locked)

On the 40 Δcorrect values:
- Point p90 = `numpy.percentile(xs, 90)` (linear).
- Bootstrap: B=5000 resamples with replacement, seed **20260813**.
  Each resample → p90. Report 2.5th / 97.5th percentiles of those p90s
  as 95% CI on the p90 estimate.
- Also: empirical percentile of help_64 among the 40 draws
  (`100 * mean(random < help) + 50 * mean(random == help)`).

---

## Decision (locked; help_64 Δ = +0.208 fixed)

| outcome | rule |
|---|---|
| `HELP_CLEAR` | help_64 Δcorrect **>** upper 95% bootstrap CI of p90 |
| `HELP_BORDERLINE` | help_64 ≥ point p90 **and** help_64 ≤ CI_upper |
| `HELP_NOISE` | help_64 < point p90 |

`HELP_CLEAR` only licenses LOO (step 3) then fresh confirm.
`HELP_BORDERLINE` / `HELP_NOISE` → do not spend confirm-set budget; demote
provisional HELP_SIGNAL.

---

## Overlap note (step 2, no generation)

Report Jaccard(hurt_16, original ΔNLL top_16) and vs g×a top_16 before or
with this job. If Jaccard ≥ 0.5 with |Δ| top_16, write explicitly that
sign-flip hurt is **mostly redundant** with magnitude landmines.
