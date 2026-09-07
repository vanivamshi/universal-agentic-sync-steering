# L4 SAE feature selection (in-sample only)

n_pos=8 n_neg=4. dead=0.082. |s| nonzero=1187/4096.

## `NULL`
neither k=16 set moves ≥1/12.

| condition | correct | Δ | viol | refuse | uniq |
|---|---:|---:|---:|---:|---:|
| identity | 0.667 | +0.000 | 0.167 | 0.250 | 0.895 |
| bottom_16 | 0.667 | +0.000 | 0.167 | 0.083 | 0.847 |
| top_16 | 0.833 | +0.167 | 0.083 | 0.083 | 0.847 |
| random_16 | 0.750 | +0.083 | 0.167 | 0.083 | 0.829 |
| bottom_64 | 0.833 | +0.167 | 0.083 | 0.000 | 0.787 |
| top_64 | 0.500 | -0.167 | 0.083 | 0.000 | 0.890 |
| random_64 | 0.750 | +0.083 | 0.167 | 0.083 | 0.889 |

Holdout not licensed.
Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/sae_feature_screen.json`
