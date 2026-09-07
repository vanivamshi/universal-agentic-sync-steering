# L4 channel selection (in-sample only)

n_pos=8 n_neg=4. g×a nonzero |s|=1024/1024.

## g×a — `SELECTOR_WORKS`
bottom-16 inert and top-16 drops — ranking is causal on this sample (in-sample only).

| condition | correct | Δ | viol | refuse | uniq |
|---|---:|---:|---:|---:|---:|
| identity | 0.667 | +0.000 | 0.167 | 0.250 | 0.895 |
| bottom_16 | 0.667 | +0.000 | 0.167 | 0.250 | 0.914 |
| top_16 | 0.000 | -0.667 | 0.000 | 0.000 | 0.275 |
| random_16 | 0.750 | +0.083 | 0.167 | 0.167 | 0.868 |
| bottom_64 | 0.667 | +0.000 | 0.083 | 0.333 | 0.898 |
| top_64 | 0.000 | -0.667 | 0.000 | 0.000 | 0.212 |
| random_64 | 0.833 | +0.167 | 0.167 | 0.000 | 0.898 |

## per-channel ΔNLL — `SELECTOR_WORKS`
bottom-16 inert and top-16 drops — ranking is causal on this sample (in-sample only).

| condition | correct | Δ | viol | refuse | uniq |
|---|---:|---:|---:|---:|---:|
| identity | 0.667 | +0.000 | 0.167 | 0.250 | 0.895 |
| bottom_16 | 0.833 | +0.167 | 0.083 | 0.167 | 0.918 |
| top_16 | 0.000 | -0.667 | 0.000 | 0.000 | 1.000 |
| random_16 | 0.750 | +0.083 | 0.167 | 0.167 | 0.868 |
| bottom_64 | 0.917 | +0.250 | 0.083 | 0.000 | 0.908 |
| top_64 | 0.000 | -0.667 | 0.000 | 0.000 | 0.338 |
| random_64 | 0.833 | +0.167 | 0.167 | 0.000 | 0.898 |

Holdout not licensed.
Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/channel_attribution_screen.json`
