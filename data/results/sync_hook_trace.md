# SYNC hook-trace (Normal vs Fault)

- Decision: **`CASE_B_CONFIRM`**
- Intervene before: **report generation**
- Protocol: `sync_hook_trace.md`

## Rates

| Class | n | tool_hook | private | disclose | mean probe_report |
|---|---:|---:|---:|---:|---:|
| B (normal) | 6 | 1.000 | 1.000 | 1.000 | -2.341 |
| C (fault) | 6 | 1.000 | 1.000 | 0.167 | -2.877 |

- Case-B pairs (tool both fire, disclose B∧¬C): **5** / 6
- Case-A pairs (tool B∧¬C): **0** / 6

## Headline

Tool/execution hooks fire under Fault; asynchrony is at **report**. Closed-loop activation repair must register **before report generation**, not re-trigger tool execution.

## Closed-loop wiring (next)

```text
detect (probe) → equation (direction) → register activation hook → generate → verify
```

Equation alone does not fire tool or activation hooks.

