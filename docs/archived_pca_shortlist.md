# Archived — failed PCA shortlist / steering tracks (removed)

These lines did **not** give usable results. Artifacts and scripts were removed
so the working sync path stays clean.

## Removed (NULL / FAIL / SATURATE / DEGRADE)

| Track | Outcome |
|---|---|
| n60 dim selection + Heim PCA injection | SATURATE / NULL |
| n60 real-other select | NULL / empty |
| multidim exp2–5b persona-PC clusters | CLUSTER_WEAK / NULL |
| deception PC screens / all31 / jailbreak surface | NULL_OTHER / DEGRADE |
| pc ablation / privilege pilot / format-matched variants | DEGRADE / no hit |

## Kept (required for working path)

| Path | Why |
|---|---|
| `data/directions/persona_pca_prose_L4.jsonl` | Sync geometry coordinate basis @ L4 |
| `activation_pipeline/analysis/persona_pca.py` | Regenerate L4 basis |
| `scripts/run_phase1_persona_pca.py` | L4 export only (default `--layers 4`) |
| `data/results/phase1_persona_pca.json` | Historical variance stats |
| `docs/figures/step_5_heim_classic_graphs.png` | Heim layer graph (no PCA shortlist) |

## Working repair (not PCA steering)

- Probe + policy gate (`GATE_LOCKED_HIT`)
- Cursor Agent hooks (`AGENT_PATH_LIVE_HIT`)
- Activation steer on equation = **NULL** (by design)

Docs: `docs/sync_geometry_control.md`, `docs/sync_agent_demo_prompts.md`
