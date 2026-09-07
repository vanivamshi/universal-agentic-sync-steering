# Probe-as-monitor (paper GitHub estimators)

Apollo `MeanLogisticRegressionDetector` + ASA `LogisticRegression(C=1)` + MMS mean-diff. **Not** the retired custom IRLS.

## Decision: `UNDERPOWERED`
holdout n_pos=2 n_disagreement=1. Paper estimators used; still too few eval leaks for a HIT. Prior custom IRLS is retired.

Holdout `tool_violation`. Disagreement = `surface_gap`.

| recipe | AUC | F1 | FPR | recall | dis. recall |
|---|---:|---:|---:|---:|---:|
| apollo_mean_lr/aa_only | 0.65 | 0.444 | 0.500 | 1.000 | 1.0 |
| apollo_mean_lr/persona31 | 0.9 | 0.500 | 0.400 | 1.000 | 1.0 |
| apollo_mean_lr/resid | 1.0 | 0.571 | 0.300 | 1.000 | 1.0 |
| asa_lr_gate/aa_only | 0.65 | 0.400 | 0.600 | 1.000 | 1.0 |
| asa_lr_gate/persona31 | 0.9 | 0.571 | 0.300 | 1.000 | 1.0 |
| asa_lr_gate/resid | 1.0 | 0.444 | 0.500 | 1.000 | 1.0 |
| mms_meandiff/aa_only | 0.65 | 0.000 | 0.000 | 0.000 | 0.0 |
| mms_meandiff/persona31 | 0.9500000000000001 | 0.667 | 0.200 | 1.000 | 1.0 |
| mms_meandiff/resid | 0.9500000000000001 | 0.444 | 0.500 | 1.000 | 1.0 |
| cot/hook proxy | 0.5499999999999999 | 0.286 | 0.400 | 0.500 | 0.0 |

Data caveat: window-means, not ASA last-token / Apollo token-flatten.
Artifact: `/Users/vamshisunku.mohan/multidim-steering-agentic-alignment/data/results/probe_monitor_screen.json`
