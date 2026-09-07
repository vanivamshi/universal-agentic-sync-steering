# Preregistration / Scope Lock

**Project:** Do Safety-Relevant Activation Directions Survive Agentic Execution?  
**Status:** Locked for data collection (0.1)  
**Date locked:** 2026-07-21  
**Primary model for collection:** Qwen3-32B (Hermes-style `<tool_call>` format)

This document freezes metrics, gates, windows, models, domains, and direction inventory before transcript collection and activation caching. Changes after lock require an explicit amendment note below.

---

## 1. Primary metric

**Directional sensitivity** (sole primary hypothesis metric):

> Perturbation magnitude \(\varepsilon\) along a candidate unit direction \(d\) at early layer \(k\), such that the residual-stream L2 change at late layer \(L\) exceeds a fixed blowup threshold \(\tau\).

Formally: smallest \(\varepsilon > 0\) with
\[
\| h_L(x + \varepsilon d) - h_L(x) \|_2 > \tau
\]
where \(h_\ell\) is the residual stream at layer \(\ell\).

**Lower sensitivity ⇒ more geometric privilege** (harder to blow up → stronger error correction along \(d\)).

### Robustness checks (not primary)

| Check | Role |
|---|---|
| Plateau depth | Secondary; do not drive accept/reject of RQ1 |
| Top-\(k\) sensitive-direction rank of safety dirs | Secondary |

### Default layer pair (Qwen3-32B)

| Symbol | Default | Notes |
|---|---|---|
| \(k\) (perturb) | layer 8 | Early; amend if smoke tests show saturation |
| \(L\) (measure) | layer 48 | Late; full sweep deferred to experiment §9 |
| \(\tau\) | \(0.5 \times \|h_L(x)\|_2\) relative | Relative blowup; absolute \(\tau\) logged as sensitivity analysis |

Layer defaults may be adjusted after §0.6 smoke tests; record the final pair here before RQ1.

---

## 2. Primary hypothesis and falsification (RQ1)

**H1:** Safety-relevant directions are **relatively deprivileged** in tool-call windows vs prose windows in the same turn: their privilege drops more than control directions under the same mode contrast.

**Arithmetic (locked 2026-08-06):** Δ = ε\*_tool − ε\*_prose. Larger ε\* ⇒ harder to blow up ⇒ more privilege. So privilege loss ⇒ **negative** Δ. Interaction I = Δ_safety − Δ_learned; H1 predicts **I < 0**. (An intermediate pilot write-up that equated “larger Δ” with privilege loss was backwards and is amended here; see `docs/rq1_privilege.md`.)

**Primary contrast:** learned-control interaction I under absolute and norm_matched blowup (relative excluded after 0.6B pilot), not absolute sensitivity alone and not random-pooled contrast.

**Falsify (null):** Safety and control directions drop privilege **equally** in tool-call mode (global brittleness). Then RQ1 is rejected; do not claim safety-specific deprivilege.

**Confirm:** Safety dirs lose privilege **more** than learned controls (I < 0 with the §6 confirmatory gate in `to do.txt`).

---

## 3. Control-normalized comparison

Every window measures, in parallel:

1. **Safety directions** (see §7)
2. **Random controls:** unit vectors sampled uniformly from residual-stream subspace, then orthogonalized against the safety set (Gram–Schmidt); \(n_{\text{rand}} = 16\) per window unless amended
3. **Task / non-safety controls:** domain-matched directions (see §7.2)

**Reporting rule:** Always report safety vs controls within mode. Absolute drops without controls are exploratory only.

---

## 4. Window definitions (token-level)

### Tool-call window

- **Start:** first token *after* the opening boundary token (`<tool_call>`)
- **End:** last token *before* the closing boundary token (`</tool_call>`)
- **Exclude:** opening and closing delimiter tokens themselves
- **Rationale:** avoid prose-about-whether-to-call and delimiter-specialized tokens

If multiple `<tool_call>` blocks exist in one assistant turn, each is a separate tool-call window.

### Prose window

- Assistant-turn tokens **outside** any structured tool-call span (and outside `<tool_response>` / tool-result user wrappers when those appear as context)
- Prefer same-turn prose that immediately precedes a tool call when available (planning → execution)

### Chat / delimiter convention (collection lock)

Primary collection uses **Qwen3 Hermes-style** tool calls:

```text
<tool_call>
{"name": "<function-name>", "arguments": { ... }}
</tool_call>
```

Tool results in follow-up turns:

```text
<tool_response>
...
</tool_response>
```

Llama-3.1 replication (§10) may use native Llama tool format; window rules still exclude boundary tokens analogously.

---

## 5. Validation gates (preregistered)

### 5.1 Logit-lens mode gate (blocks experiments past §1)

**Claim:** Tool-call windows show **lower vocabulary entropy** under logit lens than prose windows (same model, matched domains).

**Test:** Paired comparison of mean token entropy (logit-lens projected vocab distribution) tool vs prose; require tool mean entropy **significantly lower** (one-sided paired test, \(\alpha = 0.05\)) and effect in the expected direction.

**On fail:** Revise window tagging / delimiter handling; re-cache; do **not** proceed to subspace or RQ1.

#### Multiplicity / correction unit (frozen for 32B close-out)

| Role | What is tested | Correction family |
|---|---|---|
| **Primary confirmatory gate** | Preregistered **final residual layer**, one test **per dataset** (GAP; τ) | None (single preregistered test per dataset, \(\alpha=0.05\)) |
| **Exploratory layer sweep** | All probed layers within a dataset | Bonferroni **and** BH-FDR applied **within that dataset** across layers. GAP and τ are **separate** families — do **not** pool layer×dataset into one joint correction. |
| **Cross-dataset replication** | See **Replication PASS criterion** below | Consistency check, not an additional FDR family |

Rationale: the scientific claim is “mode distinction holds on dataset D,” then separately “it replicates at the same layer on dataset D′.” Pooling GAP×τ into one FDR family would let a strong effect on one dataset purchase significance on the other. Per-dataset families keep those claims independent; same-layer replication is then an explicit consistency check.

Pilot (0.6B) sweeps follow the same correction unit for documentation; they do **not** close §1.

#### Replication PASS criterion (frozen — decide before seeing 32B numbers)

**Primary-path close-out:** §1 is closed as validated iff the preregistered final-layer test **passes on both GAP and τ** (each at \(\alpha=0.05\), expected direction). No layer search required.

**Exploratory-path close-out** (only if primary fails and a corrected sweep is reported): a layer \(\ell\) counts as a **replication PASS** iff **all** of the following hold:

1. On GAP: \(\Delta < 0\) (tool entropy lower) **and** BH-FDR–adjusted \(p < 0.05\) within GAP’s layer family.
2. On τ: \(\Delta < 0\) **and** BH-FDR–adjusted \(p < 0.05\) within τ’s layer family.
3. The **same** layer index \(\ell\) (or the same relative depth bin if architectures differ — freeze the depth-fraction mapping before the run) satisfies (1) and (2).

**Not sufficient for replication PASS:** same sign of \(\Delta\) alone; overlapping confidence intervals alone; uncorrected \(p < 0.05\) on either dataset; a layer that passes on one dataset and is merely non-significant on the other.

**Reporting rule:** If no layer meets the criterion, report “no same-layer corrected replication” — do not promote the strongest single-dataset layer post-hoc.

#### Outcome labels (frozen — no post-hoc renaming)

| Primary (final layer, both datasets) | Exploratory same-layer replication | Label | Downstream |
|---|---|---|---|
| PASS on GAP **and** τ | (ignored for close-out) | `validated_primary` | §1 closed; exploratory sweep is supplementary only |
| FAIL on GAP and/or τ | PASS at some \(\ell\) | `validated_exploratory_only` | §1 closed **with caveat**: final layer failed; window analyses must report \(\ell\); do not silently treat final-layer as validated |
| PASS on exactly one of {GAP, τ} | any | `partial_primary_unreplicated` | **Not closed** — do not proceed as if §1 validated |
| FAIL on both | FAIL (no \(\ell\)) | `not_validated` | **Not closed** — revise windows / accept null for mode gate |

**Middle-case rule:** `validated_primary` and `validated_exploratory_only` are the only close-out labels. Partial primary success never upgrades via exploratory. Exploratory success never upgrades a one-dataset primary pass.

### 5.2 Refusal stability gate (blocks refusal arm of RQ1)

**Procedure:** Extract refusal via mean difference on contrastive pairs (Arditi-style) separately on prose and on tool-call hidden states.

**Required same-mode baseline:** Before interpreting prose↔tool cosine, report prose↔prose split-half cosine **at the same layers**. Band means alone are not sufficient.

**Stability:** Layer-wise cosine similarity between prose-extracted and tool-extracted refusal directions. Report point estimates **and** bootstrap 95% CIs.

**CI method (frozen for 32B; pilot may show percentile pathology):**
- Use **BCa** (bias-corrected and accelerated), not percentile, for prose↔tool cosine and for the gap.
- Gap CI must be **joint**: each bootstrap replicate resamples prose pairs **and** tool rows, recomputes prose↔prose split-half cosine **and** prose↔tool cosine on that replicate, then \(\mathrm{gap}^* = \cos^*_{\mathrm{pp}} - \cos^*_{\mathrm{pt}}\). Do **not** subtract a fixed split-half point from a bootstrapped tool cosine (conditional CI understates uncertainty and can yield percentile intervals that straddle the point estimate oddly at small \(n\)).

**Threshold:** \(\cos \ge 0.70\) on the fixed mid/late set below.

**On fail (\(\cos < 0.70\)):** Do **not** run cross-mode privilege comparison for refusal. Replace with analysis of why refusal lacks a stable linear representation in tool-call mode (finding in its own right). Assistant Axis / harmlessness arms proceed if they pass their own stability checks (same threshold unless amended).

#### 32B depth sampling for refusal stability (frozen)

Pilot (0.6B, 28 layers) suggested a **monotonic increase** in the layer-matched gap
\((\cos_{\text{prose↔prose}} - \cos_{\text{prose↔tool}})\) from early → mid → late
(L4≈0, L14≈0.39, L22≈0.57). This is three checkpoints on one small model — not a claim.

On **Qwen3-32B** (\(n_{\text{layers}}=64\)), sample a **depth grid**
\(f = \ell / (n_{\text{layers}}-1)\):

| \(f\) | Role |
|---|---|
| 0.15, 0.25 | Early / floor (diagnostic only) |
| 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 | Mid→late trend grid |
| 1.00 (final) | Terminal |

**Stability gate set (pre-specified, not chosen post-hoc):**
\(F_{\mathrm{mid}} = \{f \in \{0.50, 0.60, 0.70, 0.80\}\}\).
PASS iff a **majority** (\(\ge 3/4\)) of layers in \(F_{\mathrm{mid}}\) have prose↔tool \(\cos \ge 0.70\) (prefer also BCa CI lower bound \(\ge 0.70\)). Early/final layers are never used to rescue a failed mid/late majority.

#### Correction-family scope for the depth grid (frozen — separate from §1)

| Analysis | Estimand | Correction family |
|---|---|---|
| §1 exploratory entropy sweep | entropy \(\Delta\) by layer | **Within-dataset** layers; GAP ⊥ τ (prereg §5.1) |
| §3 **stability gate** | majority rule on fixed \(F_{\mathrm{mid}}\) | **None** (pre-specified set + majority; no layer picking) |
| §3 **exploratory** gap-at-layer / depth trend | \(\mathrm{gap}_\ell = \cos_{\mathrm{pp},\ell} - \cos_{\mathrm{pt},\ell}\) (or trend summary) | **Own family:** BH-FDR across depth-grid layers **within §3**, per model. **Not** folded into §1’s GAP⊥τ entropy family (different estimand). Do not promote a post-hoc single layer from this family into the stability gate. |

Monotonicity of gap with depth is **exploratory** (report trend + corrected per-layer gap CIs/tests if used); it does **not** redefine \(F_{\mathrm{mid}}\).

### 5.3 Control-direction stability gate (blocks control-normalized RQ1)

The task-relevant controls (`syntax`, `domain_content`) are extracted from
mean-difference contrast sets and may be noisy for the same reasons as the
safety directions. They cannot serve as a normalization baseline in §5 until
their own same-mode reliability is established.

**Required baseline:** For every learned control direction, extract the
direction independently on two random halves of its prose contrast set and
report layer-matched split-half cosine with joint BCa 95% CIs.

**PASS criterion:** A learned control is eligible for §5 only if at least 3 of
the 4 preregistered layers in
\(F_{\mathrm{mid}}=\{0.50,0.60,0.70,0.80\}\) have prose↔prose split-half
\(\cos \ge 0.70\). Apply this criterion separately to:

- `syntax`;
- each domain-specific `domain_content` direction (coding, therapy, writing).

**On fail:** Do not use the unstable learned control in the safety-vs-control
interaction. Expand or revise its contrast set before re-extraction, document
the revision as an amendment, or report that the relevant control-normalized
arm is unavailable. Do not select the most stable layer post-hoc.

**Random orthogonal controls:** These are sampled rather than learned and
therefore do not receive a split-half extraction gate. Their random seed,
unit norm, pairwise cosine distribution, and orthogonality to the safety and
learned-control set must be recorded.

**Interpretation rule:** §5 can claim safety-specific relative deprivilege only
when the compared learned controls pass this gate. If safety and control
directions are both unstable, the result is extraction noise / null, not
safety-specific deprivilege.

---

## 6. Models and domains

| Role | Model | When |
|---|---|---|
| Primary | Qwen3-32B | §0 collection onward |
| Replication | Llama-3.1-70B | After Qwen RQ1 story (§10) |

| Domain | Drift role | Collection priority |
|---|---|---|
| Coding | Low-drift baseline | First |
| Therapy | High-drift | Second |
| Open-ended writing | Intermediate | Third |

Domain effects are expected mainly on **prose** geometry; tool-call privilege shifts may be more uniform — tested in experiment §8.

---

## 7. Direction inventory

### 7.1 Safety-relevant

| ID | Direction | Source method |
|---|---|---|
| `refusal` | Refusal | Arditi et al. (2024) mean-diff on contrastive pairs |
| `assistant_axis` | Assistant / persona axis | Lu et al. (2026) |
| `harmlessness` | Harmlessness / RepE | Zou et al. (2023) RepE-style contrast |

### 7.2 Controls

| ID | Type | Construction |
|---|---|---|
| `rand_*` | Random orthogonal | Uniform on residual subspace; orthogonalize vs safety set |
| `syntax` | Task / non-safety | Contrastive pairs differing in syntactic form, same domain content |
| `domain_content` | Task / non-safety | Contrastive pairs differing in domain topic, matched syntax |

Domain-specific content directions: coding (e.g. API vs algorithm topic), therapy (session-structure vs affect topic), writing (genre/style topic). Exact pair lists live under `data/contrast_pairs/` when built (post-0.2).

---

## 8. Exploratory (preregistered, not primary)

**Orthogonal subspace analysis:** Procrustes alignment and mean cosine on the sets of residual-stream window-mean activations from prose vs tool-call windows (centroid cosine + Procrustes on the point sets). Principal angles between PCA subspaces fit independently to each set may be reported as an extra exploratory descriptor (not a hypothesis metric). Motivating evidence only; does not accept/reject RQ1.

---

## 9. Depth / non-locality caveat (interpretation rule)

Null or weak sensitivity results may mean (a) no geometric privilege, or (b) error correction outside probed \((k, L)\). Distinguish via systematic layer sweeps (experiment §9) before strong claims.

---

## 10. Outcomes → monitoring implications (locked framing)

| Outcome | Implication |
|---|---|
| **A** Directions generalize | Conversational monitors OK in agentic execution; text/tool safety gap needs another mechanism |
| **B** Directions degrade (relative) | Prefer monitors on prose planning windows, not during tool execution |
| **C** New tool-specific privileged dirs | Extract / train monitors from tool-call activations directly |

---

## 11. What to log during collection (so §0.5 matches this lock)

Per transcript turn used for caching:

- `transcript_id`, `domain`, `message_index`
- Full tokenized assistant text
- Character and token spans for each tool-call and prose window
- Model id + tokenizer revision
- Layer list cached (at minimum \(\{k, L\}\) plus intermediates needed for §9)

---

## Amendments

| Date | Change | Reason |
|---|---|---|
| 2026-07-21 | Added smoke model `Qwen/Qwen2.5-0.5B-Instruct` for pipeline bring-up only; RQ1 remains Qwen3-32B | 0.3 activation pipeline validated on machine without GPU |
| 2026-07-21 | Confirmed default Qwen3-32B layer pair (k=8, L=48) in `activation_pipeline/registry.py` | Align code with §1 defaults |
| 2026-07-28 | §2 pilot: cosine+Procrustes (+ PCA principal angles exploratory); no cross-dataset disparity ranking | Reviewer clarity |
| 2026-07-28 | **§1 correction:** Post-hoc L14 “PASS” retracted. Preregistered final-layer FAIL on 0.6B. Multi-layer sweep + Bonferroni/BH-FDR shows L14 does not survive correction. Mode distinction **not validated** until 32B + larger *n* + same-layer GAP↔τ. Canonical: `logit_lens_layer_sweep_gap.json`. | Multiple-comparisons / preregistration integrity |
| 2026-07-28 | **§1 outcome labels frozen:** `validated_primary` \| `validated_exploratory_only` \| `partial_primary_unreplicated` \| `not_validated`. Partial primary never upgrades via exploratory. | Close middle-case ambiguity |
| 2026-07-28 | **§3 CI method frozen:** BCa (not percentile) + **joint** gap bootstrap (resample prose+tool; recompute pp and pt each replicate). | Fix skewed low-n percentile / conditional gap CI |
| 2026-07-28 | **§3 depth-grid correction scope frozen:** Stability gate = majority on fixed \(F_{mid}\) (no multiplicity). Exploratory gap/trend = BH-FDR across depth-grid layers **within §3**, not folded into §1 GAP⊥τ. | Avoid recreating post-hoc layer problem |
| 2026-08-06 | **§5/§6 sign convention locked:** Δ=ε*_tool−ε*_prose; privilege loss ⇒ Δ<0; H1 ⇒ I=Δ_safety−Δ_learned<0. Retracts intermediate “Positive Δ ⇒ privilege loss” wording. §6 confirmatory gate frozen in `to do.txt`. | Prevent CONFIRMATORY/REVERSED flip; match ε* physics + AA pilot reading |
| 2026-07-30 | **§4 therapy-control revision:** Initial 12-pair affect-vs-session-structure set failed (2/4 layers). Expanded once to 24 pairs with parallel templates; v2 passed 4/4. Preserve v1 and do not iterate again if the primary-model version fails. | Predeclared gate required rebuilding or excluding an unstable learned control |
