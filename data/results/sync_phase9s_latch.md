# Phase 9S — Configuration retention: latch vs arrival-path replay

> Paired: one hybrid Γ'+beam acquisition per trial; all retention arms start from it. Fresh-episode replay ⇒ this measures **configuration retention**, not maintenance of one ongoing trajectory. No new $v$.

reps=2, n_retain=2, max_steps_acq=7, beam K=3 T=3, seed=20261119. Trials run: 10 (skipped S0==m*: 0).

Arms: **A** stay_seek (9R, own seeds) · **B** latch (composed, arrival signs) · **B0** latch_tsign (composed, target signs) · **C** path_replay (ordered) · **D** path_shuffled (permuted).

## ret@2 | acquired, per $m^*$ (k/n [Wilson 95%])

| $m^*$ | acq | stay_seek | latch | latch_tsign | path_replay | path_shuffled |
|---|---:|---:|---:|---:|---:|---:|
| `000` (hard) | 0/0 | — | — | — | — | — |
| `001` (hard) | 0/0 | — | — | — | — | — |
| `010` | 2/2 | 0/2 [0.00,0.66] | 1/2 [0.09,0.91] | 1/2 [0.09,0.91] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] |
| `011` | 2/2 | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 1/2 [0.09,0.91] | 1/2 [0.09,0.91] | 1/2 [0.09,0.91] |
| `100` | 2/2 | 0/2 [0.00,0.66] | 1/2 [0.09,0.91] | 1/2 [0.09,0.91] | 2/2 [0.34,1.00] | 1/2 [0.09,0.91] |
| `101` | 2/2 | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] |
| `110` (hard) | 0/0 | — | — | — | — | — |
| `111` | 2/2 | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] | 0/2 [0.00,0.66] |

## Pooled

| group | metric | stay_seek | latch | latch_tsign | path_replay | path_shuffled |
|---|---|---:|---:|---:|---:|---:|
| soft | ret@1 | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 6/10 [0.31,0.83] | 4/10 [0.17,0.69] |
| soft | ret@2 | 0/10 [0.00,0.28] | 2/10 [0.06,0.51] | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 2/10 [0.06,0.51] |
| soft | P_final | 0/10 [0.00,0.28] | 2/10 [0.06,0.51] | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 2/10 [0.06,0.51] |
| soft | rollout hits | 3/13 | 7/20 | 7/20 | 10/20 | 7/20 |
| hard | ret@1 | — | — | — | — | — |
| hard | ret@2 | — | — | — | — | — |
| hard | P_final | — | — | — | — | — |
| hard | rollout hits | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| all | ret@1 | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 6/10 [0.31,0.83] | 4/10 [0.17,0.69] |
| all | ret@2 | 0/10 [0.00,0.28] | 2/10 [0.06,0.51] | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 2/10 [0.06,0.51] |
| all | P_final | 0/10 [0.00,0.28] | 2/10 [0.06,0.51] | 3/10 [0.11,0.60] | 3/10 [0.11,0.60] | 2/10 [0.06,0.51] |
| all | rollout hits | 3/13 | 7/20 | 7/20 | 10/20 | 7/20 |

## Paired comparisons (ret@2, acquired trials)

| comparison | pairs | discordant (first-only / second-only) | both | exact sign p |
|---|---:|---:|---:|---:|
| stay_seek_vs_latch | 10 | 0 / 2 | 0 | 0.500 |
| latch_tsign_vs_latch | 4 | 1 / 0 | 1 | 1.000 |
| latch_vs_path_replay | 10 | 1 / 2 | 1 | 1.000 |
| path_shuffled_vs_path_replay | 4 | 0 / 1 | 0 | 1.000 |

## Reading guide (fix before looking at numbers)

| result | suggests |
|---|---|
| B > A | accumulated configuration matters |
| B > B0 | arrival polarity matters, not just which channels |
| B fails, C succeeds | arrival sequence matters |
| C > D | ordering (not just the multiset of actions) matters |
| B and C fail | fresh-episode replay insufficient → motivates persistent context / activation feedback (does not prove them necessary) |
| soft ok, hard fail | hard sinks need a different retention mechanism |

Small n: a nonzero rate is ≥1 observed success, not reliable retention. Do not read P_final>0 on 8/8 as reliable control.
