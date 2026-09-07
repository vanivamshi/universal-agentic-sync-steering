# Phase 1 — Persona PCA basis (Qwen3-0.6B)

Closes G4: multi-dimensional persona basis (Lu et al. method, Mac-scale subset).

**Active use:** sync geometry coordinate basis @ **L4 only** —
`data/directions/persona_pca_prose_L4.jsonl`. Extra layers / tool / structured
variants and PCA shortlist experiments were removed (did not work).

See `docs/archived_pca_shortlist.md`.

## Protocol

- **Roles:** 32 of Lu’s role instructions (seeded sample)
- **Questions:** 3 extraction questions per role
- **Modes:** prose (role system prompt) vs tool (same + Hermes tool-call instruction)
- **Readout:** mean residual over generated response tokens @ L4
- **Axis:** `assistant_axis = mean(default) − mean(roles)`
- **PCA:** SVD on the role matrix (n_roles × hidden), per mode

## Artifacts (kept)

| Path | Contents |
|---|---|
| `data/results/phase1_persona_pca.json` | Variance, angles, AA cosine, summary |
| `data/directions/persona_pca_prose_L4.jsonl` | L4 prose 31-PC basis (sync path) |

Regenerate L4 basis only:

```bash
.venv/bin/python scripts/run_phase1_persona_pca.py --layers 4 --skip-tool-mode
```

## Results (2026-08-06)

L4 prose↔tool **NEAR_90_DIVERGENCE** — geometric fact, not a steering win.
PCA shortlist / privilege / ablation tracks = closed negative.
