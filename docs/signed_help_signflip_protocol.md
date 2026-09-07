# Sign-flip validation of signed-help channels (locked before run)

help_16 / help_64 passed an in-sample random p90 bar (`HELP_SIGNAL`).
One random_64 draw also hit +0.208. This experiment asks whether the
**sign of the ranking** is doing work, not whether “zero 16 channels”
is lucky.

---

## What is locked (do not change after numbers)

- Ranking: same frozen `channel_nll_delta.npy` (original-12 TF NLL).
- **help_16** already measured: Δcorrect = +0.167 on n=24.
- **hurt_16** = the 16 channels with the **most positive** ΔNLL
  (sign flip of the help rule). Not a new fit. Not `|Δ|` bottom.
- Same n=24 GAP controls, L4 ChannelZero, T=0, same metrics as Phase 1b.
- Identity = Phase 1b identity (correct = 0.583). Do not re-measure identity
  unless the job must recompute; Δ vs locked identity.

**Primary condition:** hurt_16 only.
**Supporting (optional in same job):** hurt_64 (most positive 64).

---

## Pass / fail (locked)

Compare hurt_16 to the already-recorded help_16 on the same n=24.

| outcome | rule |
|---|---|
| `SIGN_OK` | hurt_16 Δcorrect ≤ 0 **and** help_16 Δcorrect − hurt_16 Δcorrect ≥ 2/24 |
| `SIGN_WEAK` | hurt_16 Δcorrect < help_16 Δcorrect but hurt_16 Δcorrect > 0 |
| `SIGN_FAIL` | hurt_16 Δcorrect ≥ help_16 Δcorrect |

Collapse on hurt (`correct=0`) is the strongest form of Δcorrect ≤ 0 → counts toward
`SIGN_OK` if the gap bar also holds. It is **not** a fail.


No k retune. No re-ranking. No fresh confirm set in this job.

---

## Reporting

In-sample only. `SIGN_OK` ≠ validated for new tasks.
Artifact: `data/results/signed_help_signflip.{json,md}`.
