# Channel controllability (Stage 2)

- Mode: `causal`
- S_base: `[0.0, 1.0, 0.5]`
- v_C_status: `candidate_from_observed_plan`
- approx_diagonal: **False**

| | v_C | v_H | v_O |
|---|-----|-----|-----|
| ΔC | 0.000 | -0.250 | 0.000 |
| ΔH | 0.000 | 0.250 | 0.000 |
| ΔO | 0.000 | 0.250 | 0.000 |

## Arms

- `+v_C`: ΔS=[0.0, 0.0, 0.0] target=0.000 collateral=[0.0, 0.0]
- `-v_C`: ΔS=[0.0, 0.0, 0.0] target=0.000 collateral=[0.0, 0.0]
- `+v_H`: ΔS=[0.0, 0.0, 0.0] target=0.000 collateral=[0.0, 0.0]
- `-v_H`: ΔS=[0.5, -0.5, -0.5] target=-0.500 collateral=[0.5, -0.5]
- `+v_O`: ΔS=[0.0, 0.0, 0.0] target=0.000 collateral=[0.0, 0.0]
- `-v_O`: ΔS=[0.0, 0.0, 0.0] target=0.000 collateral=[0.0, 0.0]
