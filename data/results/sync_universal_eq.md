# Universal sync equation

- Equation: `e = m* - S`
- Protocol: `docs/sync_universal_equation.md`

| name | m* | S | e | decision |
|------|----|---|---|----------|
| case_b_hidden | (1, 1, 1) | (1, 1, 0) | (0, 0, 1) | EQ_ADAPT |
| false_cot_but_ok_hook_out | (0, 1, 1) | (0, 1, 1) | (0, 0, 0) | EQ_SYNC |
| false_cot_target_but_true_plan | (0, 1, 1) | (1, 1, 1) | (-1, 0, 0) | EQ_ADAPT |
| all_synced_111 | (1, 1, 1) | (1, 1, 1) | (0, 0, 0) | EQ_SYNC |

