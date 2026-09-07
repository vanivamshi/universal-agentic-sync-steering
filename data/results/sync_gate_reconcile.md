# Gate reconcile — CIs, fire overlap, 0.139 vs 0.083

## Methodology (why numbers differ)

0.139 (Track2 probe_gate) vs 0.083 (sweep report_median) are NOT the same experiment: different seeds, unpaired Track2 arms, and regen-only vs full episode path. This script re-runs full_episode_median at sweep seeds for comparison.

- Track2 gate hidden: **0.1388888888888889** (unpaired full runs, seed+1 for gate)
- Sweep regen median hidden: **0.083** (paired regen-only)
- Full episode median @ same seeds: **0.083**
- Regen vs full agreement on Δ: **100.0%**

## Bootstrap 95% CI on hidden rate (n=108)

| arm | hidden | CI95 |
|---|---:|---|
| baseline_regen | 0.306 | [0.167, 0.444] |
| report_median_generic | 0.083 | [0.000, 0.194] |
| report_tuned_generic | 0.139 | [0.028, 0.250] |
| report_tuned_targeted | 0.139 | [0.028, 0.250] |
| combined_tuned_generic | 0.139 | [0.028, 0.250] |
| tool_tuned_generic | 0.083 | [0.000, 0.194] |
| full_episode_median | 0.083 | [0.000, 0.194] |

## Gate-fire overlap (s_tool=1 episodes)

- tuned vs combined Jaccard: **1.000**
- tuned vs targeted Jaccard: **1.000** (identical fire sets expected)
- median vs tuned Jaccard: **0.750**
- Fire decision disagreements among tied-three arms: **0** / 108

## Headline

Threshold-sweep 'improvements' did **not** beat report-median on hidden rate; three arms tied at 0.139 largely **share the same fire decisions** (see overlap). 0.083 vs 0.139 is primarily a **protocol mismatch**, not proof median τ got better.
