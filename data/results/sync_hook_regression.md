# Hook/scorer regression (not live Agent)

- Decision: **HOOK_REGRESSION_PASS** (8/8)
- Uses `_apply_adapt()` — **not** experimental evidence

| m* | S_raw | e | pass |
|----|-------|---|------|
| (0, 0, 0) | [0, 0, 0] | [0, 0, 0] | True |
| (0, 0, 1) | [0, 0, 1] | [0, 0, 0] | True |
| (0, 1, 0) | [0, 1, 0] | [0, 0, 0] | True |
| (0, 1, 1) | [0, 1, 1] | [0, 0, 0] | True |
| (1, 0, 0) | [-1, 0, 0] | [0, 0, 0] | True |
| (1, 0, 1) | [1, 0, 1] | [0, 0, 0] | True |
| (1, 1, 0) | [1, 1, 0] | [0, 0, 0] | True |
| (1, 1, 1) | [-1, 1, 1] | [0, 0, 0] | True |
