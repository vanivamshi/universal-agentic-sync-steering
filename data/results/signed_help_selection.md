# Signed-help channel selection (n=24)

Identity correct=0.583. Rank = most negative frozen ΔNLL (360/1024 channels ΔNLL<0).

## Decision: `HELP_SIGNAL`
help-64 Δcorrect ≥ Phase 1b random_64 p90. In-sample only.

| set | correct | Δ | vs p90 | viol | refuse | uniq | gen_ppl | held_ppl | Jaccard vs |Δ|bottom |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| help_16 | 0.750 | +0.167 | ≥ p90 (+0.092) | 0.250 | 0.000 | 0.904 | 1.19 | 5.354 | 0.000 |
| help_64 | 0.792 | +0.208 | ≥ p90 (+0.133) | 0.208 | 0.000 | 0.847 | 1.19 | 5.404 | 0.000 |

help_16 channels: 12, 191, 151, 181, 146, 46, 184, 187, 24, 320, 840, 719, 112, 253, 270, 794

Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/signed_help_selection.json`
