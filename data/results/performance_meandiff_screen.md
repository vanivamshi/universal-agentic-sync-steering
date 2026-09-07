# Performance mean-diff screen (in-sample only — not a causal claim)

Extract: control only. n_pos=8 n_neg=4 **FLOOR_OF_POWER**. E2=OK (LOO pairwise cos=0.938).

**P1 = `OK`** — in-sample screen hit, not a validated causal claim — do not write 'we found a performance direction' without this qualifier

| α | correct | Δ | viol_c | refuse_c | uniq | chars | gen_ppl | held_ppl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -4 | 0.833 | +0.167 | 0.167 | 0.000 | 0.881 | 298 | 1.14 | 5.363 |
| -2 | 0.750 | +0.083 | 0.167 | 0.083 | 0.906 | 291 | 1.17 | 5.328 |
| -1 | 0.750 | +0.083 | 0.167 | 0.083 | 0.902 | 285 | 1.20 | 5.314 |
| +0 | 0.667 | +0.000 | 0.167 | 0.250 | 0.889 | 298 | 1.19 | 5.302 |
| +1 | 0.667 | +0.000 | 0.083 | 0.250 | 0.892 | 312 | 1.20 | 5.296 |
| +2 | 0.583 | -0.083 | 0.083 | 0.333 | 0.849 | 349 | 1.21 | 5.294 |
| +4 | 0.750 | +0.083 | 0.083 | 0.167 | 0.876 | 285 | 1.16 | 5.301 |

Holdout **not licensed**. Fresh confirm set = 2 new legitimate tasks × 6 GAP domains (not paraphrases, not baseline/eliciting).
Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/performance_meandiff_screen.json`
