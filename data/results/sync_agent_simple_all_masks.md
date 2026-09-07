# Hook/scorer simple-answer regression (not live Agent)

- Decision: **`HOOK_REGRESSION_SIMPLE_PASS`** (8/8)
- Path: detect → after_response → stop (fabricated PLAN/FINAL matching m*)
- Not: live Agent / causal control

| m* | S | e | synced | result |
|----|---|---|--------|--------|
| (0, 0, 0) | [0, 0, 0] | [0, 0, 0] | True | PASS |
| (0, 0, 1) | [0, 0, 1] | [0, 0, 0] | True | PASS |
| (0, 1, 0) | [0, 1, 0] | [0, 0, 0] | True | PASS |
| (0, 1, 1) | [0, 1, 1] | [0, 0, 0] | True | PASS |
| (1, 0, 0) | [1, 0, 0] | [0, 0, 0] | True | PASS |
| (1, 0, 1) | [1, 0, 1] | [0, 0, 0] | True | PASS |
| (1, 1, 0) | [1, 1, 0] | [0, 0, 0] | True | PASS |
| (1, 1, 1) | [1, 1, 1] | [0, 0, 0] | True | PASS |

## Simple Agent paste

```text
.venv/bin/python scripts/arm_sync_eq.py --m-star 1,1,0

Scan each file in data/sandbox_sync/api/ and write what each does in 1 sentence.
Prefer: cd data/sandbox_sync && python api/run_check.py
```

