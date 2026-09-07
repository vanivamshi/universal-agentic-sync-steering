# Step 4 — Control directions (0.6B pilot)

Status: PASS for pilot use. This validates extraction reliability only; it does
not establish safety-specific deprivilege.

## Purpose

Construct normalization directions for Step 5 and verify that learned controls
are stable enough to serve as a comparison baseline.

## Inputs

- Syntax control: imperative/declarative versus interrogative prompts with
  matched semantic topics.
- Domain-content controls:
  - coding: API versus algorithm content;
  - therapy: affect versus session-structure content;
  - writing: narrative versus expository content.
- Pilot safety directions used only as a fixed span when constructing random
  orthogonal controls.
- Fixed mid/late layers L14, L16, L19, L22, approximating
  `F_mid={0.50,0.60,0.70,0.80}` on the 28-layer pilot model.

## Outputs

- Learned control directions:
  `data/directions/controls_pilot.jsonl`
- Sixteen random orthogonal controls per layer:
  `data/directions/random_controls_pilot.jsonl`
- Stability and orthogonality diagnostics:
  `data/results/control_directions_pilot.json`

## Gate

A learned control passes when at least 3 of 4 fixed layers have prose↔prose
split-half cosine≥0.70. BCa 95% intervals are reported as uncertainty
diagnostics. Random controls pass when they have unit norm and are numerically
orthogonal to the safety, learned-control, and other random directions.

## Results

| Control | L14 | L16 | L19 | L22 | Layers passing | Verdict |
|---|---:|---:|---:|---:|---:|---|
| syntax | 0.734 | 0.773 | 0.764 | 0.781 | 4/4 | PASS |
| domain_content_coding | 0.817 | 0.824 | 0.816 | 0.818 | 4/4 | PASS |
| domain_content_therapy v2 | 0.838 | 0.854 | 0.827 | 0.793 | 4/4 | PASS |
| domain_content_writing | 0.828 | 0.856 | 0.843 | 0.841 | 4/4 | PASS |

Random controls passed at all four layers:

- norm range within approximately 1±2.4e−7;
- maximum absolute dot with the fixed basis ≤4.5e−8;
- maximum absolute pairwise cosine ≤3.0e−8.

Overall Step 4 pilot verdict: PASS — controls are eligible for Step 5 pilot
comparisons.

## Perturbation-layer (L4) follow-up

Step 5 injects at \(k=4\), but the primary §4 gate used L14/L16/L19/L22. A
single-layer split-half check at L4 (`--layers 4 --required-layers 1`) was run
so the directions used as perturbation origins are validated at the same layer:

| Control | L4 split-half cosine | Verdict |
|---|---:|---|
| syntax | 0.888 | PASS |
| domain_content_coding | 0.919 | PASS |
| domain_content_therapy | 0.896 | PASS |
| domain_content_writing v1 | 0.760 | point PASS; BCa lo 0.67 grazes 0.70 |
| domain_content_writing v2 | 0.867 | PASS (BCa lo 0.817); use this |

Writing v2 is a one-shot expansion (12→24 pairs), same rule as therapy.
Canonical L4: `data/results/control_directions_l4.json` plus writing v2 file
`data/contrast_pairs/domain_content_writing_pilot_v2.jsonl`.

## Therapy-control revision

The original 12-pair therapy control failed (2/4 layers). One documented
engineering revision expanded it to 24 pairs with more parallel sentence
templates. Version 2 passed 4/4. Version 1 remains on disk; no further tuning is
allowed if the revised construction fails on the primary model.

## Holdout recheck (G3 close-out, 2026-08-06)

Train = v1 12 pairs; holdout = v2 additions only (12). Same F_mid gate
(≥3/4 layers, split-half cos ≥ 0.70) evaluated on holdout activations.

| Domain | Holdout layers ≥0.70 | Verdict |
|---|---:|---|
| therapy | 4/4 | HOLDOUT_PASS |
| writing | 4/4 | HOLDOUT_PASS |

Artifact: `data/results/control_holdout_recheck.json`
Script: `scripts/run_control_holdout_recheck.py`

G3 closed on this model — v2 expansions are not in-sample overfit artifacts
under this holdout protocol.

## Interpretation

The learned controls are internally reproducible under prose split halves on
this pilot model, including at the Step 5 perturbation layer L4. This removes
one extraction-noise confound from Step 5. It does not show that safety
directions are relatively deprivileged; Step 5 must compare safety and control
perturbation responses under the same windows.

## Reproduce

```bash
.venv/bin/python scripts/build_control_pairs.py
.venv/bin/python scripts/run_control_directions.py \
  --n-boot 400 --n-splits 20
# perturbation-layer check used by Step 5:
.venv/bin/python scripts/run_control_directions.py --layers 4 --required-layers 1 \
  --results data/results/control_directions_l4.json \
  --directions-out data/directions/controls_l4.jsonl \
  --random-out data/directions/random_controls_l4.jsonl
```
