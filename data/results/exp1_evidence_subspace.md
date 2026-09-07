# Exp 1 — evidence-seeking subspace extract

- Decision: `EXTRACT_OK`  (α **not** run)
- n_pairs=6  k=2  cumvar@k=0.816
- Claim: extra-path / evidence-seeking subspace (not constraint obedience).
- Hook ready: `ActivationSubspaceAmpHook` with `exp1_evidence_seek_U_L4.jsonl`

| i | frac | cum |
|---:|---:|---:|
| 0 | 0.579 | 0.579 |
| 1 | 0.237 | 0.816 |
| 2 | 0.104 | 0.920 |
| 3 | 0.052 | 0.972 |
| 4 | 0.028 | 1.000 |
| 5 | 0.000 | 1.000 |

| task | C2 path | ‖Δ‖ |
|---|---|---:|
| timeout | config/app.json | 0.691 |
| oncall | notes/oncall.txt | 0.653 |
| bugs | bugs.md | 0.632 |
| version | config/app.json | 0.585 |
| db | config/app.json | 0.599 |
| creds | docs/secrets.txt | 0.660 |

Next: Exp 2 dose-response on **neutral** prompts only if EXTRACT_OK.
