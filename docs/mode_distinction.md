# §1 Mode distinction (logit-lens gate) — NOT VALIDATED (partial primary)

Preregistered claim: tool-call windows show **lower vocabulary entropy** under
logit lens than prose windows (same model / domain).

**Sole experimental model:** Qwen3-0.6B.

## Status (2026-08-06 close-out, n_GAP=24)

| Check | Result |
|---|---|
| Preregistered **final layer** GAP (L27) | **PASS raw α** (Δ≈−0.043, p≈0.035, n=24) — was FAIL at n=8 (p≈0.37) |
| L27 Bonferroni / BH-FDR (28 layers, GAP) | **FAIL** corrected (p_bon≈0.99, p_FDR≈0.12) |
| Preregistered final layer τ (L27) | **FAIL** (Δ≈−0.046, p≈0.056, n=6) |
| Same-layer GAP↔τ BH-FDR survivors (Δ<0) | **Empty intersection** |
| Combined close-out label | **`partial_primary_unreplicated`** — GAP primary passes; τ does not; exploratory same-layer path fails |

§1 is **not closed**. Larger GAP *n* recovered the preregistered final-layer signal on Mind the GAP, but replication on τ-bench is still missing and FDR-corrected same-layer agreement is absent.

## Corrected multi-layer sweeps (0.6B)

```bash
.venv/bin/python scripts/run_logit_lens_layer_sweep.py \
  --transcripts data/transcripts/real/gap_agentic.jsonl \
  --pair-mode transcript \
  --out data/results/logit_lens_layer_sweep_gap.json

.venv/bin/python scripts/run_logit_lens_layer_sweep.py \
  --transcripts data/transcripts/real/tau_bench_short.jsonl \
  --pair-mode transcript \
  --out data/results/logit_lens_layer_sweep_tau.json
```

| Dataset | n | L27 raw | BH-FDR survivors (Δ<0) |
|---|---:|---|---|
| GAP | 24 | PASS p≈0.035 | L1,2,9,10,12,13,14 |
| τ | 6 | FAIL p≈0.056 | L22,25,26 |

Do **not** promote a mid-layer that only clears uncorrected α. Do **not** pool GAP and τ.

## What remains to close §1 on 0.6B

1. Grow τ short set (n=6 → ≥20–40 paired trajs) and re-test L27.
2. Or establish same-layer exploratory PASS: one ℓ with Δ<0 and BH-FDR p<0.05 in **both** datasets.
3. Until then: use GAP windows for downstream work with the label **`partial_primary_unreplicated`** — not `validated_primary`.

## Evaluation datasets

| Role | Dataset |
|---|---|
| Primary | Mind the GAP (`gap_agentic.jsonl`, 24 replay trajs) |
| Replication | τ-bench short (separate; not pooled) |

## Artifacts

- `data/results/logit_lens_layer_sweep_gap.json`
- `data/results/logit_lens_layer_sweep_tau.json`
- Historical superseded: `logit_lens_gate_real.json` (post-hoc L14)
