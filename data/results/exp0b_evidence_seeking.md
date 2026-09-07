# Exp 0b — evidence-seeking identity (no α)

- Decision: `IDENTITY_HIT`
- mean n_extra_paths C0=0.17 C1=0.83 C2=1.00
- mean n_exploratory C0=0.83 C1=0.83 C2=1.00
- mean evidence_seek C0=0.83 C1=0.83 C2=1.00
- tasks with extra(C2)>extra(C0): 5/6

| task | extra C0 | extra C1 | extra C2 | seek C0 | seek C1 | seek C2 |
|---|---:|---:|---:|---:|---:|---:|
| timeout | 0 | 0 | 1 | 1 | 0 | 1 |
| oncall | 0 | 1 | 1 | 0 | 1 | 1 |
| bugs | 1 | 1 | 1 | 1 | 1 | 1 |
| version | 0 | 1 | 1 | 1 | 1 | 1 |
| db | 0 | 1 | 1 | 1 | 1 | 1 |
| creds | 0 | 1 | 1 | 1 | 1 | 1 |

No α. No SVD. Not surface_gap.
