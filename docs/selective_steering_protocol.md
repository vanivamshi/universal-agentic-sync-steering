# Selective steering — geometry predicts controllability

**Status:** ACTIVE (post u₂ / sync steering nulls).

## Question

> Can ε*(u) on a **train split** predict which directions give **selective**
> behavioral control on a **held-out test split**?

Target behavior: **proximal extra-path tool use** (`n_extra_paths`).

## Frozen protocol

| Lock | Value |
|---|---|
| Layer / site | L4, last token (`ActivationSteerHook`, `pos_mode=last`) |
| Basis candidates | Exp1 SVD `u1`, `u2`, `±u_i`, random-in-subspace, random-full, orth-to-u₂ |
| ε* | Min ε s.t. relative L2 blowup @ L4 ≥ 0.5; median over **train** tasks |
| Train tasks | Dataset A: `timeout`, `oncall`, `bugs`, `version` |
| Test tasks | Dataset A held-out: `db`, `creds` |
| α | 0.25 (matched unit-norm directions) |
| Arms | baseline, `+αu`, `−αu` per candidate |
| Target effect | \|Δ mean `n_extra_paths`\| vs paired baseline |
| Collateral | \|Δ calls\| + \|Δ list\| + \|Δ search\| |
| Selectivity | S(u) = target / (collateral + 0.01) |
| Controllability C(u) | max(\|Δ(+α)\|, \|Δ(−α)\|) on target |
| Prediction test | Spearman(ε*, C) on candidates; ε* ↛ collateral |

## Decisions

| Tag | Rule |
|---|---|
| `SELECTIVE_HIT` | Best S among SVD dirs beats best random S × 1.5 **and** Spearman(ε*, C) ≤ −0.5 on test |
| `SELECTIVE_WEAK` | Target effect > 0 but geometry does not predict |
| `SELECTIVE_NULL` | No direction beats random on selectivity |

Script: `scripts/run_selective_steering.py`  
Artifacts: `data/results/selective_steering.{json,md}`

### Result — `SELECTIVE_WEAK`

Train ε* (4 tasks) → test paired ±α steer (2 held-out tasks, N=4).

| direction | ε* train | target \|Δextra\| | collateral | S |
|---|---:|---:|---:|---:|
| rand_sub / rand_full | 7.52 | 0.125 | 0.125 | **0.93** |
| u1 | 7.52 | 0.125 | 0.250 | 0.48 |
| u2, neg_u2, orth | 7.52 | 0.000 | 0.000 | 0.00 |

- Spearman(ε*, controllability) = **+0.45** (wrong sign; need ≤ −0.5)
- Spearman(ε*, collateral) = +0.21
- Best SVD S / best random S = **0.52** (random wins)
- All train ε* collapsed to ~7.52 (no spread → geometry ranking inert)

**Interpretation:** small nonspecific target perturbation on test (`u1`/random only);
ε* does **not** predict selective controllability. Does not establish geometry-guided
selective steering. Aligns with u₂ `REPLICATE_WEAK` / `TRANSFER_NULL`.

---

## ACTIVE — v2 (fix measurement, same dataset)

**Do not** switch to a controlled paired dataset or C/T/O channel experiments yet.
First fix whether **any** geometric score predicts controllability.

| Change | v1 | v2 |
|---|---|---|
| Global ε* | L2 blowup @ L4 | kept (compare) |
| **ε*_beh** | — | min ε s.t. \|Δ r(h)\| ≥ τ; r = dot(h, v_beh), v_beh = mean(C2−C0) on train |
| **geom_align** | — | \|u · v_beh\| (subspace geometry, no ε search) |
| Quality | S = target/collateral | **E** effectiveness, **F** specificity, **M** = E/α |
| Bidirectional | reported | required for HIT |
| Predictors tested | ε* global only | global vs beh vs align → E on held-out |

Script: `scripts/run_selective_steering_v2.py`  
Artifacts: `data/results/selective_steering_v2.{json,md}`

| Decision | Rule |
|---|---|
| `SELECTIVE_V2_HIT` | \|ρ(eps_beh, E)\| > \|ρ(eps_global, E)\|, ρ_beh ≤ −0.3, SVD E×F beats random |
| `SELECTIVE_V2_GEOM_HIT` | ρ(geom_align, E) ≥ 0.5 and SVD beats random |
| `SELECTIVE_V2_NEG` | No predictor correlates; random ≥ SVD |
| `SELECTIVE_V2_WEAK` | Behavioral effect without geometry prediction |

**Useful negative:** if ε*_beh also saturates / fails to predict E, conclude:
*31D privilege metric may detect sync (separate program) but does not characterize
steerability* — not “try a bigger dataset.”

### Result — v2 `SELECTIVE_V2_WEAK`

Script: `run_selective_steering_v2.py`. Same train/test split as v1.

**Measurement fix (train):** ε*_beh **does discriminate** where ε*_global saturates:

| direction | ε*_global | **ε*_beh** | geom_align |
|---|---:|---:|---:|
| u2 / neg_u2 | 7.52 | **0.82** | 0.196 |
| u1 / neg_u1 | 7.52 | 1.41 | 0.114 |
| rand_full / orth | 7.52 | 2.00 (cap) | ≤0.08 |

**Controllability (test, E/F/M):** SVD directions **null** (E=0); only `orth_u2` weak
(E=0.125, F=0, not bidirectional). Random ≥ SVD on test.

| predictor | ρ vs E | note |
|---|---:|---|
| ε*_global | +1.00 | saturated; artifact with sparse E |
| ε*_beh | +0.88 | **wrong sign** — low ε*_beh (u2) did not → high E |
| geom_align | −0.71 | high-align SVD null on test |

**Conclusion:** behavior-specific readout fixes **direction ranking on train** but
**does not yet predict selective causal control on held-out test**. Next step is
NOT a bigger dataset — refine readout/threshold or accept negative:
*geometry detects representation structure; steerability requires a different metric
or intervention site.*

Artifacts: `data/results/selective_steering_v2.{json,md}`

---

## ACTIVE — v4 (decision/logit-level controllability)

**Do not** differentiate discrete sampled actions again (v3 Jacobian = 0).
Move one level upstream:

\[
m_T(h)=\mathrm{logit}(\texttt{<tool\_call>})-\max_i\mathrm{logit}(\texttt{no-tool}_i)
\]

\[
D_T(u)=\frac{m_T(h+\alpha u)-m_T(h-\alpha u)}{2\alpha},\qquad
D_C(u)=\mathrm{RMS}(\partial_\alpha\ell_{\setminus\{T,\neg T\}})
\]

\[
\mathcal{S}(u)=\frac{|D_T(u)|}{\lambda+|D_C(u)|},\qquad
u^*=\arg\max_{|u|=1}\mathcal{S}(u)
\]

(Exp1 2D circle sweep + SVD ±uᵢ + random + orth; freeze before held-out.)

Held-out test: ±α → |ΔP(tool)|; prediction Spearman(train S → held |ΔP|).

Script: `scripts/run_selective_steering_v4.py`  
Artifacts: `data/results/selective_steering_v4.{json,md}`

| Decision | Rule |
|---|---|
| `SELECTIVE_V4_HIT` | S discriminates; ρ(S,|ΔP|)≥0.5; u* beats random; bidirectional signal |
| `SELECTIVE_V4_WEAK` | Continuous D_T alive but held-out selective control incomplete |
| `SELECTIVE_V4_NULL` | Decision margins also flat / no prediction |

**Key claim vs v3:** differentiate the **decision that produces the action**, not the sampled action.

### Result — v4 `SELECTIVE_V4_WEAK`

Script: `run_selective_steering_v4.py`. Same train/test split.

**Continuous signal is alive (v3 fix worked):**

| direction | S | D_T | D_C |
|---|---:|---:|---:|
| **u* = theta_31** | **2.12** | **−0.252** | 0.065 |
| u1 | 2.08 | −0.242 | 0.061 |
| rand_full | 1.20 | −0.127 | 0.056 |
| u2 | 0.45 | −0.037 | 0.067 |
| neg_u1 | 0.27 | −0.005 | 0.061 |

S_spread = **0.67** (not saturated). Test-set D_T for u* = −0.22 (decision margin moves).

**Held-out sampling gap:** |ΔP(tool)| = **0** for all arms at α=0.25 — logit margin moves, but first-turn tool emission does not flip. Spearman(S→|ΔP|) is not interpretable under a flat |ΔP| vector.

**Interpretation:** the equation now measures a **real continuous decision sensitivity** (unlike v3). Selective *sampling* control is not yet established — need either larger α, a softer behavioral readout (P(tool) via temperature / first-token sample), or accept that margin≠flip on this saturated tool-use baseline.

Artifacts: `data/results/selective_steering_v4.{json,md}`
