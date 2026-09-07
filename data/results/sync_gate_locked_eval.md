# SYNC gate — locked paired eval (CANONICAL)

- Decision: **`GATE_LOCKED_HIT`**
- Protocol: `docs/sync_gate_locked_eval.md`
- τ (Disc median, private): **-2.681124**
- n episodes: **36** (3 reps × 12 conditions)

## Bootstrap 95% CI

| arm | hidden | CI95 | spurious | CI95 |
|---|---:|---|---:|---|
| baseline | 0.361 | [0.194, 0.528] | 0.000 | [0.000, 0.000] |
| probe gate | 0.083 | [0.000, 0.194] | 0.000 | [0.000, 0.000] |

- **Paired hidden drop** (baseline − gate): 0.278 [0.139, 0.444] excludes_zero=True
- Disclose|private (gate): 0.917
- Gate fire rate: 0.361

Supersedes unpaired Track 2 headline (0.139) and sweep tuning (`SWEEP_FAIL`).
