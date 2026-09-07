# Sign-flip validation of signed-help channels

Identity correct=0.583. hurt_k = most-positive frozen ΔNLL.

## Decision: `SIGN_OK`
hurt_16 Δcorrect ≤ 0 and help−hurt ≥ 2/24. Signed ranking has directional meaning on this sample (not holdout).

| set | correct | Δ | mean ΔNLL | uniq | held_ppl |
|---|---:|---:|---:|---:|---:|
| help_16 (locked) | 0.750 | +0.167 | -0.001556 | 0.904 | 5.354 |
| hurt_16 | 0.000 | -0.583 | 0.1797 | 0.962 | 12.494 |
| help_64 (locked) | 0.792 | +0.208 | -0.001003 | 0.847 | 5.404 |
| hurt_64 | 0.000 | -0.583 | 0.04674 | 0.176 | 12.621 |

help_16 − hurt_16 = +0.750 (bar for SIGN_OK: ≥ +0.083 and hurt≤0).

hurt_16 channels: 35, 1, 13, 3, 277, 96, 30, 16, 28, 41, 158, 9, 134, 29, 126, 236

Artifact: `/Users/vamshisunku.mohan/bluedot_alignment/data/results/signed_help_signflip.json`
