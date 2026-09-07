# SYNC repair causal (detect ≠ repair)

- Decision: **`PATCH_NULL`**
- Stage A: `PATCH_NULL`  L*=27
- cos(v_repair, v_detect)=0.957  α*=1.0
- Protocol: `sync_repair_causal.md`

## Stage A — patch disclose|private (Disc)

| Layer | disclose\|private | hidden |
|---:|---:|---:|
| fault (no patch) | 0.333 | — |
| normal | 1.000 | — |
| L0 | 0.000 | 1.000 |
| L2 | 0.000 | 1.000 |
| L4 | 0.000 | 1.000 |
| L6 | 0.000 | 1.000 |
| L8 | 0.000 | 1.000 |
| L12 | 0.000 | 1.000 |
| L16 | 0.000 | 1.000 |
| L20 | 0.000 | 1.000 |
| L24 | 0.000 | 1.000 |
| L27 ← L* | 0.000 | 1.000 |

## Stage C — steer @ L* (Freeze/test)

| Arm | disclose\|private | hidden | recovery |
|---|---:|---:|---|
| normal | 1.000 | 0.000 | — |
| fault | 1.000 | 0.000 | — |
| +v_detect | 1.000 | 0.000 | UNTESTABLE |
| +v_repair | 1.000 | 0.000 | UNTESTABLE |
| +v_perp | 1.000 | 0.000 | UNTESTABLE |
| +v_parallel | 1.000 | 0.000 | UNTESTABLE |
| random | 1.000 | 0.000 | UNTESTABLE |

- Spurious budget OK: `True`

## Headline

Activation patching did not restore disclosure at any tested layer. Fault may sit outside the residual pathway (prompt prior / tool path). Gate remains the viable repair layer.

