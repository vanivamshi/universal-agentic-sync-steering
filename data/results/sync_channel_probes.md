# Channel representation probes (predictive)

- Source: `/Users/vamshisunku.mohan/multidim-steering-agentic-alignment/data/results/sync_channel_free_runs_cache.npz`
- n: 32
- C label: `observed_PLAN`

> Predictive AUC ≠ causal control. Next: single-channel ±v tests → J.

| Channel | n0 | n1 | AUC | ok_counts | phase |
|---------|----|----|-----|-----------|-------|
| C | 12 | 20 | 0.95 | True | h_report |
| H | 16 | 16 | 1.0 | True | h_tool |
| O | 7 | 9 | 0.8888888888888888 | True | h_report |

## Factor / cell occupancy (observed only — not a requirement to fill all 8)

```text
{
  "010": 6,
  "011": 6,
  "100": 11,
  "101": 5,
  "110": 1,
  "111": 3
}
```

## Factor balance (A)

- C: n0=12 n1=20
- H: n0=16 n1=16
- O: n0=7 n1=9

## Identification gate (A → B)

- learn_v_C: **True**
- learn_v_H: **True**
- learn_v_O: **True**
- ready_for_causal_J: **True**
- ready_for_8way: **False** (requires causal J freeze)

- C: AUC=0.950 ok
- H: AUC=1.000 ok
- O: AUC=0.889 ok
