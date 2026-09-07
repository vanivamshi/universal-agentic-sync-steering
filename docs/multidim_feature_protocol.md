# Multi-dimensional feature program (Engels et al. adapted)

> **ARCHIVED** — exp2–5b scripts/results removed (`docs/archived_pca_shortlist.md`).

Paper: Engels, Michaud, Liao, Gurnee, Tegmark — *Not All Language Model Features
Are One-Dimensionally Linear* (arXiv:2405.14860 / ICLR 2025).

**Goal here:** test whether deceptive / jailbreak-compliant *agentic* structure
(or the confirmed lookup→emission→rank-collapse chain) is an **irreducible
joint** representation rather than a single linear direction.

Single-dimension negatives already on record: Track A (persona PCs), \|ΔNLL\| /
g×a bottoms, signed-help `HELP_BORDERLINE`. Cumulative PCA ablation collapse
remains consistent with a multi-dim unit.

**Order (cost):** Exp 4 → Exp 2 → Exp 1 (SAE) → Exp 3 (shape-agnostic joint
intervene; ran) → **Exp 5 (Engels S/M + shape)** → Exp 6 (shape-matched steer,
only if Exp5 licenses). Do not train a broad SAE until Exp 2 finds a cluster
worth confirming. Do not lower J1–J4 for joint steering.

---

## Experiment 4 — Is lookup-nudge rank collapse a joint-structure signature?

**Data (frozen):** `data/results/nudge_causal_cache.npz` + meta  
Conditions: `format_only`, `emit_nudge`, `lookup_nudge` at L4 (primary; L14/L22
supporting). Split `lookup_nudge` by `has_tool_call` (D3 mechanism).

**Basis:** project trial residuals onto the locked **31-dim prose persona PCA**
(`data/directions/persona_pca_prose_L4.jsonl`). Also report ambient residual
PCA (no persona basis) as a robustness check.

**Geometry tests (locked, descriptive):**

| Test | What |
|---|---|
| G1 effective dim | dims@90% variance of the condition’s cloud (centered) |
| G2 plane mass | fraction of variance in top-2 cloud PCs |
| G3 circularity | in top-2 plane: least-squares circle fit; report relative RMS residual = RMS(‖r−R‖) / R. Low ⇒ ring-like |
| G4 angular structure | Rayleigh / resultant length of angles in top-2 plane (vs uniform). High ⇒ preferred angles, not a filled disk |

**Primary contrast:** `lookup_emitted` vs `format_only` at L4.

**Decision (locked):**

| Tag | Rule |
|---|---|
| `JOINT_CANDIDATE` | G1(emitted) ≤ G1(format) − 5 **and** G2(emitted) ≥ 0.50 **and** (G3 residual ≤ 0.35 **or** G4 resultant ≥ 0.40) |
| `RANK_ONLY` | G1 drops as known, but G2 < 0.50 **or** (G3 > 0.35 **and** G4 < 0.40) — compression without coherent 2D manifold |
| `NULL_GEOMETRY` | no clear G1 drop on this recompute |

`JOINT_CANDIDATE` motivates Exp 2/1 targeting this emission subspace.  
`RANK_ONLY` = collapse is still real (D1–D4) but **not** an Engels-style circle/plane unit on this check.  
Not a causal claim. Not deception-outcome evidence.

---

## Experiment 2 — Co-activation clustering on 31 PCs (after Exp 4)

**Data:** `data/activations/gap_deception.json` L4 window means, joined to
`data/labels/gap_deception_eliciting.jsonl` by `transcript_id`. One vector per
transcript = mean over that transcript’s L4 windows, then project onto prose
31-PC basis.

**Clustering (locked):** Pearson ρ among the 31 PC scores across transcripts.
Distance = 1 − |ρ|. Hierarchical **average** linkage. Clusters = connected
components after cutting dendrogram at distance **0.60** (within-cluster mean
|ρ| ≥ 0.40). Keep clusters with size ≥ 3.

**Weak member (locked):** for both `surface_gap` and `intent_contradiction` in
`deception_pc_screen_dual.json` L4, the PC’s CI includes 0 **or** |r| < 0.25.

**Joint score:** first principal component of the cluster’s PC-score matrix
across transcripts (signed so corr with `tool_violation` ≥ 0).

**Decision (locked):**

| Tag | Rule |
|---|---|
| `CLUSTER_HIT` | ≥1 cluster size≥3, all members weak, joint \|r\| to `surface_gap` or `intent_contradiction` ≥ 0.35 **and** bootstrap CI95 excludes 0 (n_boot=1000, seed 20260813) |
| `CLUSTER_WEAK` | cluster exists with joint \|r\| ≥ max member \|r\| + 0.10 but misses HIT bar |
| `CLUSTER_NULL` | no size≥3 cluster, or no joint gain over members |

Historical gate: Exp 1 only if `CLUSTER_HIT`. **Superseded** by Exp 1 override
below (user request). Exp 3 may still target Exp4 emission plane without Exp1/2 HIT.

---

## Experiment 1 — Broad GAP L4 SAE + irreducibility (user override)

Originally gated on Exp2 `CLUSTER_HIT`. **Override:** run anyway after Exp4
`JOINT_CANDIDATE` + Exp2 `CLUSTER_WEAK` (user request 2026-08-13).

**Train (locked):**
- Site: L4 residual. Corpus: `real_token_windows.jsonl` tokens + L4 window means
  from `real_gap.pt`, `real_all.pt`, `gap_deception.pt` (broad agentic, not
  persona-role prompts). Cap **16384** rows, seed **20260813**.
- ReLU SAE, `n_features = 4 × hidden`, L1=1e-3, 4000 steps, batch 256.
- Fail closed: dead features > 90% → `SAE_DEAD`.

**Irreducibility / differential (locked):**
- Encode transcript-mean L4 from gap_deception eliciting + control.
- Cluster live features by average-linkage on distance `1 − |ρ|`, cut **0.50**,
  size ∈ [2, 12].
- Cluster is **irreducible-candidate** if mean pairwise co-occurrence
  (both active above median) ≥ 0.35 among members.
- **Differential:** Cohen’s d of mean cluster activation
  (`surface_gap=1` vs `family=control`). HIT if |d| ≥ 0.50 **and** irreducible.

Decision: `SAE_HIT` / `SAE_WEAK` / `SAE_NULL` / `SAE_DEAD`. Saves decoder
subspaces for Exp 3. Not a causal claim.

## Experiment 3 — Joint subspace J1–J4 (same bars as Track A)

**Targets (locked order):**
1. Exp4 emission plane: top-2 ambient PCs of `lookup_emitted` trial residuals
   (from `nudge_causal_cache.npz`).
2. Exp2 strongest joint cluster PCs (persona basis) if requested.
3. Exp1 SAE differential clusters if `SAE_HIT`/`SAE_WEAK`.

**Intervention:** `h ← h − α QQᵀ h` at L4 (`ActivationAblateHook` with scaled
α via `ActivationSubspaceScaleHook`). α grid: `{0.25, 0.5, 0.75, 1.0}` (positive
= project-out fraction). Reverse = −α (amplify subspace).

**Gates:** identical J1–J4 / `PASS_CAUSAL` as `run_jailbreak_surface_steering.py`.
No bar lowering. Primary outcome `surface_gap`.

**Status note:** Exp3 α-ablation is **shape-agnostic**. NULL / TOO_WIDE /
WEAK_SIDE_EFFECT there does **not** refute Engels multi-dim structure — it only
fails that intervention class. Shape-matched steer is Exp 6 (gated on Exp 5).

---

## Experiment 5 — Engels irreducibility + shape (before any new steer)

**Goal:** recover the paper’s missing steps A/B on Exp1 SAE hit clusters so a
later steer can be **shape-matched** (performance aim), not another αQQᵀ blob.

**Data (frozen):** Exp1 SAE `data/directions/multidim_exp1_sae.pt` + clusters in
`multidim_exp1_sae.json`. Activation cloud = pooled L4 window means from
`gap_deception.json`, `real_gap.pt`, `real_all.pt` (no retrain; no model gen).

**Per cluster (Alg. 1 style):**
1. Encode cloud; keep only cluster latents; decode → \(\hat{x}_C\).
2. Keep points with any cluster feature active.
3. PCA; on planes 1–2, 2–3, … compute Engels **S** (min MI over rotations) and
   **M₀.₁** (ε-mixture). Score = \(\bar{S}(1-\bar{M})\).
4. Geometry (Exp4 helpers): dims@90, plane mass, circle rel RMS, angular
   resultant; silhouette on best circle plane.
5. Null: same-size random live feature sets (n_null=12). Jaccard reported as
   diagnostic only — **not** the irreducibility claim.

**Decision (locked):**

| Tag | Rule |
|---|---|
| `IRRED_CANDIDATE` | score ≥ null p90 **or** (S̄ ≥ pool S q75 ∧ M̄ ≤ pool M q25); n_active≥80 |
| `REDUCIBLE` | fails both gates |
| `UNDERPOWERED` | n_active < 80 |
| `CIRCLE_LIKE` | best-plane circle_rms≤0.35 ∧ plane_mass≥0.50 |
| `DISCRETE_RING` | CIRCLE_LIKE ∧ silhouette≥0.40 at k≥3 |
| `LINE_LIKE` | var_pc1≥0.85 |
| `BLOB` | none of the above |
| `SHAPE_STEER_CANDIDATE` | ≥1 target with IRRED_CANDIDATE ∧ shape ∈ {CIRCLE, DISCRETE_RING, LINE} |

**Steer license:** only `SHAPE_MATCHED_STEER_OK` → Exp 6 (rotate / mode / line).
No new α project-out. Not causal.

Script: `scripts/run_multidim_exp5_engels_geometry.py`.

---

## Experiment 5b — Engels A/B on Exp4 emission plane + control dims

**Why:** Exp5 SAE C0/C1 → `REDUCIBLE`. Remaining paper target = Exp4
`JOINT_CANDIDATE` emission plane. Must beat **locked control dimensions**.

**Data (frozen):** `nudge_causal_cache.npz` L4 trials; `lookup_emitted` vs
`format_only`. Persona-31 + ambient.

**Target plane:** top-2 PCs of `lookup_emitted` residuals (ambient primary;
persona-31 supporting).

**Controls (locked files):**
- `data/directions/controls_l4.jsonl` — `syntax`, `domain_content_{coding,therapy,writing}`
  each expanded to a 2D plane (dir + random ortho).
- `data/directions/random_controls_l4.jsonl` + extra QR random planes (n=24) as null.

**Tests:** Engels S / M₀.₁ + circle/line/blob shape on each plane (same bars as Exp5;
`min_active=30` because emitted n≈37 — underpowered vs paper token clouds).

**Decision (locked):**

| Tag | Rule |
|---|---|
| `IRRED_CANDIDATE` | target score ≥ random-plane p90 |
| `SHAPE_STEER_CANDIDATE` | IRRED + shape ∈ {CIRCLE, DISCRETE_RING, LINE} **and** score > max learned control |
| `NOT_SPECIFIC_VS_CONTROLS` | IRRED+shape but ≤ a learned control |
| `REDUCIBLE` / `IRRED_BLOB` / `UNDERPOWERED` | as named |

Exp6 only if `SHAPE_STEER_CANDIDATE`. Control arms required in any later steer.

Script: `scripts/run_multidim_exp5b_emission_engels.py`.
