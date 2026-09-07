# Sync channel control — Phase record

**Milestone (9J+9K):** all 8 states are **reachable** under frozen actuators
(\(\exists\) empirically observed control path to each \(m^*\)).

**Still open:** universal 8-way **reliable target-conditioned attainment**
(\(P_{\mathrm{acquire}}\cdot P_{\mathrm{retain}}\) for hard sinks; retention not yet reliable).

Directions frozen (\(v_c\)). **No new \(v\)** — problem is retention (after 9R), not actuators/planning.
**9Q/9R:** beam over \(\{\pm C,\pm H,\pm O\}\) → hybrid \(\Gamma'+\)beam acquires **all 8** (\(P_{\mathrm{acq}}>0\)); \(P_{\mathrm{final}}\) still sparse.

---

# 8-way replication (reps=8) — complete

Protocol unchanged (only \(n\) increased):

```bash
.venv/bin/python scripts/run_sync_8way_validation.py --reps 8
```

Primary: \(\Delta E=E_0-E_3\). Secondary: \(P(S_3=m^*)\).

## Aggregate

| Arm | n | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O |
|-----|---|--------|----|------|------|------|
| none | 64 | 0.094 | +0.094 | +0.094 | +0.125 | -0.125 |
| predictive | 64 | 0.125 | -0.109 | -0.172 | +0.141 | -0.078 |
| converted | 64 | 0.109 | +0.062 | -0.016 | -0.016 | +0.094 |
| random | 64 | 0.188 | +0.000 | -0.219 | +0.250 | -0.031 |

## Decision

- Converted still beats predictive on ΔE, but **does not beat none** on ΔE.
- Random still has higher hit than converted.
- **Call:** random confound → diagnose before policy.

---

# Target-state / hard-strata highlights

From `sync_8way_validation_enriched.md`:

- Per-\(m^*\): converted does not show broad hit improvements; gains remain selective.
- Hard strata:
  - need_C_or_O: none +0.25, converted +0.22, random +0.38
  - need_C_and_O: none +0.65, converted +0.82, random +0.67
- Converted aggregate \(P(C),P(H),P(O)\) ≈ (0.469, 0.563, 0.500).

Interpretation: converted helps some difficult C/O regions on **error reduction**, but not enough to dominate globally.

---

# 5C status

Matched replicate keeps trajectory-coupling signal:

- C|h₀→C|h_H cross ratio: **1.33**
- H|h₀→H|h_C cross ratio: 0.66

Still treat \(T_{HC}\) magnitude as diagnostic, not precise gain.

---

# 8-way soft-margin surrogate — **done (fast)**

$$
\mathcal L_{8\mathrm{way}}=\sum_k\log\bigl(1+e^{-\beta(2m_k^*-1)M_k}\bigr)
\qquad
\text{primary: }\mathbb E[-\Delta\mathcal L]
$$

| Arm | E[−ΔL] | ΔE | P(hit) |
|-----|--------|-----|--------|
| none | +0.00 | +0.00 | 0.19 |
| predictive | +0.06 | +0.12 | 0.19 |
| **converted** | **+2.08** | +0.00 | 0.12 |
| random | −0.15 | +0.19 | 0.12 |

- \(v_c \gg\) all controls on continuous joint-target signal (88% trials lower \(L\)).
- Hit/ΔE still weak; \(\operatorname{corr}(-\Delta L,\Delta E)\approx -0.08\) (does **not** predict discrete sync).
- **Read:** converted moves the joint margin potential; discrete 8-way failure is **downstream** of that continuous signal.

`data/results/sync_8way_surrogate.{json,md}`

---

# Boundary-crossing + calibration — **done**

$$
\text{margin improvement}\;\not\Rightarrow\;\text{boundary crossing}
$$

| Finding | Result |
|---------|--------|
| P(W→C) converted | ≈ **0.43** (C/H/O) |
| Δ~M on wrong vs correct bits | **wrong** (H +14 vs +0.3) — not deepening-correct |
| L_wrong / D_wrong ↔ ΔE | still **uncorrelated** |
| Unsteered $M_k$ | **deterministic** (std=0 on fixed templates) |
| $P(S{=}1\mid M{>}0)$ free-run | C **0.21**, H 0.92, O 0.67 |

$$
\boxed{v_c\text{ controls site probes; probes }\neq\text{ episode decision boundary}}
$$

Next (still no new $v$): episode-conditioned / boundary-localized $M\!\to\!S$ map.

`data/results/sync_boundary_crossing.{json,md}`

---

# Phase 7 — Episode-conditioned boundary — **done**

$$
\boxed{h_k^{\mathrm{live}}\to S_k}
\qquad\text{then}\qquad
\boxed{v_c\to\Delta B_k\to S_{\mathrm{proxy}}}
$$

| Channel | AUC live $h$ | AUC tmpl $M$ | ΔB(+α) mono | P(flip $S^\pm$) |
|---------|--------------|--------------|-------------|-----------------|
| C | **0.875** | 0.5 | ✓ | 0.42 |
| H | **1.000** | 0.5 | ✓ | 0.33 |
| O | **0.667** | 0.5 | ✓ | 0.17 |

Live $M$ varies (H std **13.5** vs template 0). Frozen $v_c$ moves live $B$ on **all** channels.
**Chain recoverable** — missing map identified; ready for boundary-based 8-way objective (still no new $v$).

`data/results/sync_phase7_live_boundary.{json,md}`

---

# Phase 8A — Live-boundary 8-way — **done (fast)**

$$
\mathcal L_{\mathrm{live}}=\sum_k\mathrm{softplus}(-\beta s_k B_k),\quad
G_k=s_k(B_{\mathrm{after}}-B_{\mathrm{before}})
$$

| Arm | P(hit) | ΔE | E[−ΔL_live] | corr(−ΔL,ΔE) | G_H | P(W→C) H |
|-----|--------|-----|-------------|--------------|-----|----------|
| none | 0.06 | −0.25 | **+0.29** | +0.10 | −0.39 | 0.00 |
| predictive | 0.19 | +0.00 | −0.53 | +0.09 | −0.24 | 0.25 |
| converted | 0.12 | −0.31 | −0.18 | **+0.53** | −0.50 | 0.11 |
| random | 0.12 | −0.19 | −0.12 | +0.06 | +0.11 | 0.00 |

**Wins:** live $\mathcal L$ finally **tracks** discrete ΔE for $v_c$ ($r{=}0.53$; template was $\approx0$).
**Fails:** $v_c$ does **not** beat controls on mean −ΔL / ΔE / hit; $G_H{<}0$; H flip still weak.
$G_C{\equiv}0$ under this capture (C site = fixed task prompt, not intervention-conditioned).

$$
\boxed{\mathcal L_{\mathrm{live}}\leftrightarrow\Delta E\text{ recovered;\ joint controller still does not win}}
$$

Next bottleneck: post-boundary / order / C-site capture — **not** a new $v$.

`data/results/sync_phase8_live_8way.{json,md}`

---

# Phase 8B — Target-signed control — **done (H primary)**

$$
d_k(m_k^*)=s_k v_c^k,\quad s_k=2m_k^*-1
$$

α=1.5, n=16 free × both $H^*$ × {target-sign, wrong-sign}. No new $v$.

| Probe | Target-sign | Wrong-sign |
|-------|-------------|------------|
| Act-space $E[G_H]$ | **+0.41** (frac>0 = 1.00) | −0.41 |
| Beh $E[G_H]$ | −0.84 | **+0.68** |
| Beh $P(W\to C)_H$ | **0.31** | 0.19 |

vs Phase 8A converted $P(W\to C)_H{=}0.11$, $G_H{=}{-}0.50$.

**Wins:** $P(W\to C)_H$ jumps **0.11 → 0.31** under $s_H v_c^H$; wrong-sign only 0.19 — sign conditioning moves flips.
Act-space $G$ is exactly as Phase 7 predicts ($s\Delta B>0$ by construction).

**Fails:** behavioral re-run $E[G_H]_{\mathrm{TS}}{<}0$ and **worse** than wrong-sign. Local $h\pm s\alpha v$ and post-episode live $B$ disagree. Gate `controller_bug_is_sign=False`.

$$
\boxed{
P(W\to C)_H\uparrow\text{ under target-sign;\ live }G_H\text{ still broken}
}
$$

**Read:** sign is a real lever on bit flips, but not a clean actuator bug fix. Next bottleneck is **H timing / post-steer $B$ capture** (and same-seed pairing), not representation. Still no new $v$. C/O target-sign and signed 8-way wait on that.

`data/results/sync_phase8b_target_sign.{json,md}`

---

# Phase 8C — Same-site counterfactual — **done (A+B pass)**

$$
\text{counterfactual activation test}\neq\text{full-episode intervention}
$$

No change to $v_c$, α, $s_k$, or H boundary. n=16, same seed as 8B.

### Geometry (frozen $h_0$)

$\Delta B(+\alpha)=+0.412$, $\Delta B(-\alpha)=-0.412$ — exact Phase 7 / 8B act-space match.

### Stage A — same-site decision

| condition | $E[G_H]$ | $P(W\to C)$ |
|-----------|---------:|------------:|
| no_steer | 0 | 0.38 |
| wrong_sign | **−0.41** | 0.19 |
| target_sign | **+0.41** | **0.69** |

Hooked residual $G$ also ±0.41. **Stage A pass.**

### Stage B — PLAN teacher-forced + decision-token-only hook

| condition | $E[G_H]$ | $P(W\to C)$ |
|-----------|---------:|------------:|
| no_steer | 0 | 0.25 |
| wrong_sign | −0.41 | **0.00** |
| target_sign | +0.41 | **0.56** |

**Stage B pass.** Chain recovered:

$$
\boxed{v_c^H\rightarrow h_H^{\mathrm{live}}\rightarrow\text{H decision}\rightarrow S_H}
$$

vs Phase 8B full-tool-phase re-run ($G$ inverted, $P(W\to C){=}0.31$): the bug was **measurement / whole-phase hook**, not $v_c$ or target-sign.

**Next:** rebuild C→H→O with corrected H semantics (target-sign + decision-token-only). 8-way still paused until that controller path is wired. C live-site still separate.

`data/results/sync_phase8c_same_site.{json,md}`

---

# Phase 8D — Sequential C→H→O corrected H — **done (fast reps=2)**

$$
d_k=(2m_k^*-1)v_c^k
$$

H = decision-token-only + carried PLAN prefill. Same seed. Skip noop stages. No new $v$.

| Arm | P(hit) | P(H) | P(W→C)_H | ΔE tot | E0→E3 |
|-----|--------|------|----------|--------|-------|
| corrected (8D) | **0.25** | 0.62 | 0.25 | **+0.50** | 1.50→1.62→1.38→1.00 |
| Phase4 converted | 0.19 | 0.62 | — | 0.00 | 1.38→1.62→1.12→1.38 |

Per-stage ΔE: C −0.12, H **+0.25**, O **+0.38** (Phase4 H was +0.50 but O was −0.25).

$$
\boxed{\text{sequential composition improved vs Phase 4; 8-way still paused}}
$$

`data/results/sync_phase8d_sequential_cho.{json,md}`

Fast path: `--reps 2` (~2 min). Full: `--reps 4`.

---

# Phase 8E — Full 8-way with 8D controller — **done (reps=2)**

Before/after vs legacy converted ($P(\mathrm{hit})=0.11$, $\Delta E=+0.06$).

| Arm | P(hit) | ΔE | P(C) | P(H) | P(O) |
|-----|--------|-----|------|------|------|
| none | 0.06 | 0.00 | 0.56 | 0.50 | 0.75 |
| predictive | 0.19 | +0.31 | 0.56 | 0.69 | 0.56 |
| **converted** | **0.06** | **+0.31** | 0.56 | 0.62 | 0.44 |
| random | 0.38 | +0.31 | 0.69 | 0.69 | 0.62 |
| legacy converted | 0.11 | +0.06 | — | — | — |

**Wins:** converted $\Delta E$ **+0.31 vs legacy +0.06**; H-stage $\Delta E_H=+0.19$.
**Fails:** converted $P(\mathrm{hit})$ **not** above legacy (0.06 vs 0.11); random/predictive match converted on $\Delta E$; random leads on hit.

$$
\boxed{\Delta E\text{ improves vs legacy controller; full 8-state hit not yet}}
$$

Note: `none` $\Delta E\equiv0$ under skip-noop (no resample). reps=2 — directional, not stable rates.

`data/results/sync_phase8e_8way_corrected.{json,md}`

---

# Phase 8F — C/O same-site causal — **done**

H ✓ (8C). Episode-conditioned C stem (`…I will `) + O post-tool `FINAL:`. Vectors frozen. n=12.

### Stage A (same-site soft bit)

| Channel | $E[G]_{\mathrm{TS}}$ | $P(W\to C)$ TS / WS / none | Stage A G |
|---------|---------------------:|----------------------------:|:---------:|
| O | +0.156 | 0.58 / 0.58 / 0.58 | ✓ geometry |
| C | +0.082 | **0.92** / 0.50 / 0.83 | ✓ + sign helps |

$\Delta B$ matches Phase 7 exactly (±0.156 O, ±0.082 C).

### Stage B (decision-token generation)

| Channel | $P(W\to C)$ TS / WS / none | Stage B |
|---------|----------------------------:|:-------:|
| O | 0.00 / 0.00 / 0.00 | **fail** |
| C | 0.50 / 0.50 / 0.50 | **fail** |

$$
\boxed{
\text{O: }B\text{ moves, decision does not;\ }
\text{C: soft-bit ok, generation not yet}
}
$$

**Read:** O is the binding constraint for 8-way (Phase7 flip 0.17 confirmed). C live stem recovers soft control ($G_C\not\equiv0$), but token generation does not yet track $s_C$. Do **not** re-run aggregate 8-way until O (and C generation) pass local causal control. No new $v$.

`data/results/sync_phase8f_co_same_site.{json,md}`

---

# Phase 8G — Frozen $v_c^O$ dose — **done (under-actuation)**

Disclose = GT; During/docs = auxiliary. No new $v$.

| α | $\|\Delta B\|$ | $E[G_O]$ | P(proxy flip) | P($S_O$ flip) | P(W→C)$_{TS}$ |
|--:|-------------:|---------:|--------------:|--------------:|--------------:|
| 1.5 | 0.16 | +0.16 | 0.00 | 0.00 | 0.00 |
| 3 | 0.31 | +0.31 | 0.25 | 0.00 | 0.00 |
| 5 | 0.52 | +0.52 | 0.42 | 0.08 | 0.08 |
| 8 | 0.84 | +0.84 | 0.50 | **0.17** | **0.25** |

$$
\boxed{\text{O under-actuation: raise }\alpha_O\text{ (not new }v\text{)}}
$$

$E[G_O]>0$ at all doses. Proxy leads behavior (flips from α=3; disclose from α=5). Next: wire higher $\alpha_O$ (e.g. 5–8) into controller + early-FINAL; then C gen; then 8-way. Re-convert O only if that fails.

`data/results/sync_phase8g_o_dose.{json,md}`

---

# Phase 8H — O gain integration — **done**

$\alpha_C=\alpha_H=1.5$; early-FINAL O; H decision-token. Full 8 $m^*$, reps=2. No new $v$.

| $\alpha_O$ | P(hit) | ΔE | P(C) | P(H) | P(O) | P(W→C)$_O$ |
|-----------:|--------|---:|------|------|------|------------|
| 1.5 | **0.31** | +0.38 | 0.56 | 0.69 | 0.75 | 0.57 |
| 5 | 0.31 | +0.38 | 0.56 | 0.69 | 0.75 | 0.57 |
| 8 | 0.31 | +0.38 | 0.56 | 0.69 | 0.75 | 0.57 |
| 8E converted (ref) | 0.06 | +0.31 | 0.56 | 0.62 | 0.44 | — |

$$
\boxed{\text{early-FINAL lifts O/hit vs 8E; }\alpha_O{>}1.5\text{ adds no net gain in-loop}}
$$

**Read:** Isolated 8G under-actuation is real, but with early-FINAL in the sequential controller $\alpha_O{=}1.5$ already saturates O ($P(O){=}0.75$, $P(W\to C)_O{=}0.57$). $\alpha_O{=}5$ identical to 1.5; $\alpha_O{=}8$ flips 2/16 episodes with **zero net** aggregate change. Remaining 8-way gap is **composition (C/H)**, not O gain. Do not re-convert O yet.

`data/results/sync_phase8h_o_gain.{json,md}`

---

# Phase 8I — C/H composition (O frozen) — **done**

O locked: $\alpha_O=1.5$, early-FINAL. Selected 4 $m^*$, reps=2. No new $v$; no 8-way reopen.

| order | P(hit) | ΔE | ΔE_C | ΔE_H | P(C)\|CH | P(H)\|CH | P(both CH) |
|-------|--------|---:|-----:|-----:|--------:|--------:|-----------:|
| C→H→O | 0.38 | +0.38 | −0.12 | +0.50 | 0.38 | 0.75 | 0.38 |
| **H→C→O** | 0.38 | **+0.75** | −0.25 | +0.62 | 0.50 | 0.75 | 0.38 |

Live-site hooked $\Delta B$ (Phase-7 $B$):

| | $R$ (self) | cross_ratio |
|--|------------:|------------:|
| $R_{C|H}$ | **1.68** | **3.19** |
| $R_{H|C}$ | **0.73** | 0.79 |

$$
\boxed{H\to C\text{ preferred; }C\to H\text{ coupling confirmed (cross }3.19\text{)}}
$$

**Read:** Same discrete hit, but H-first doubles ΔE. C stage still **hurts** ($\Delta E_C<0$ both orders); after C→H, C bit *regresses* (0.50→0.38) while H recovers. Corrected live sites amplify the 5C asymmetry ($R_{C|H}{>}1$, $R_{H|C}{<}1$).

$$
\boxed{C\text{ remains the limiting actuator under sequential composition}}
$$

**Do not skip C for 8-way** — that abandons one required factor of $m^*\in\{0,1\}^3$. Conditional C-skip is diagnostic-only (collateral), never universal-controller success. Lock order **H→C→O**; next = fix **C generation** (8F soft✓ / gen✗ gap), same methodology that fixed H in 8C.

`data/results/sync_phase8i_ch_composition.{json,md}`

---

# Phase 8J — C actual decision control — **done (partial)**

Same-seed Stage B (8C protocol). Primary gen = first-token run/only. n=12. No new $v$. **C not skipped.**

| $\alpha_C$ | $P(W\to C)_{gen}$ TS | vs WS/none | pass |
|-----------:|---------------------:|:----------:|:---:|
| 1.5 | 0.25 | flat | ✗ |
| 3 | 0.33 | ↑ | ✓ (weak) |
| **5** | **0.42** | ↑ | ✓ best |
| 8 | 0.17 | collapse | ✗ |

Stage A: $\Delta B=\pm0.082$ (Phase7 match); TS beats WS (0.67 vs 0.25) but not none — soft site directional, fragile. 8F's 0.92 not replicated this seed.

$$
\boxed{C\text{ gen under-actuated: dose helps to }\alpha_C{=}5\text{; not yet H-grade}}
$$

**Read:** Same pattern as O before early-FINAL calibration — generation is **dose-responsive** ($0.25\to0.42$) then overshoots at 8. Absolute $P(W\to C){=}0.42$ is far from H's 8C (~0.56–0.69).

`data/results/sync_phase8j_c_decision.{json,md}`

---

# Phase 8K — Compose $\alpha_C=5$ stem into $H\to C\to O$ — **done**

$$
\alpha_H=1.5\ \text{(decision-token)},\quad
\alpha_C=5\ \text{(stem-prefill)},\quad
\alpha_O=1.5\ \text{(early-FINAL)}
$$

Selected 4 $m^*$ (paired vs 8I) + auto full 8-way. No skip. No new $v$.

| | P(hit) | ΔE | ΔE_C | ΔE_H | ΔE_O | P(C) | P(H) | P(O) |
|--|--------|---:|-----:|-----:|-----:|------|------|------|
| **8K selected** | 0.38 | +0.62 | **+0.00** | +0.62 | +0.00 | 0.62 | 0.62 | 0.88 |
| 8I H→C | 0.38 | +0.75 | −0.25 | +0.62 | +0.38 | 0.62 | 0.75 | 0.88 |
| **8K 8-way** | 0.25 | +0.44 | **+0.25** | +0.12 | +0.06 | 0.69 | 0.69 | 0.56 |
| 8H / 8E | 0.31 / 0.06 | +0.38 / +0.31 | — | — | — | — | — | — |

$$
\boxed{\Delta E_C:\ -0.25\ \rightarrow\ 0\ \text{(sel)}\ /\ +0.25\ \text{(8-way)}}
$$

**Read:** C no longer hurts. Selected-4 C stages were mostly noops ($\Delta E_C{=}0$) with H held (+0.62). Full 8-way: **$\Delta E_C{=}{+}0.25$**. Hit 0.25 beats 8E (0.06), below 8H (0.31). Locked three-factor controller; no more mechanistic exploration unless C regresses.

`data/results/sync_phase8k_c_gain_compose.{json,md}`

---

# Phase 8K validation — high-rep stratified 8-way — **done**

Frozen stack only. reps=8/$m^*$, n=64, seed=20260920. **No controller changes.**

### Aggregate

| Arm | P(hit) | ΔE | ΔE_C | P(C) | P(H) | P(O) |
|-----|--------|---:|-----:|------|------|------|
| **8K validated** | **0.20** | **+0.36** | **+0.19** | 0.64 | 0.58 | 0.64 |
| 8K exploratory | 0.25 | +0.44 | +0.25 | 0.69 | 0.69 | 0.56 |
| legacy converted | 0.11 | +0.06 | — | 0.47 | 0.56 | 0.50 |
| 8E | 0.06 | +0.31 | — | 0.56 | 0.62 | 0.44 |

### Per $m^*$ $P(\mathrm{hit}\mid m^*)$

| $m^*$ | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|------:|----:|----:|----:|----:|----:|----:|----:|----:|
| 8K val | 0 | 0 | 0.25 | **0.50** | **0.62** | 0.25 | 0 | 0 |
| legacy | 0 | 0 | 0.25 | 0.50 | 0.12 | 0 | 0 | 0 |

$$
\boxed{
\text{improved 8-way controller (not universal): }
4/8\ m^*\text{ reachable; }\Delta E_C{>}0
}
$$

**Supported claim:** a frozen three-channel causal controller **improves** 8-way synchronization and **removes C-stage regression**.

**Not supported:** universal 8-way sync ($P(\mathrm{hit}){=}0.20$; only 4/8 $m^*$ with $P{>}0$). Hard cells remain $000,001,110,111$. Largest win: $100$ ($0.12\to0.62$).

`data/results/sync_phase8k_8way_validation.{json,md}`

---

# Phase 9 — Hamming-path reachability — **done (partial)**

Frozen 8K. Protocol: free $\to$ home to $000$ $\to$ direct or one-bit path to $111$. Final credit $=S{=}111$ only. reps=4.

| arm | path | P(homed) | P(hit 111) | P(all wp) |
|-----|------|---------:|-----------:|----------:|
| direct | $000\to111$ | 0.00 | 0.50 | — |
| OHC | $000\to001\to011\to111$ | 0.00 | 0.00 | 0.00 |
| HOC | $000\to010\to011\to111$ | 0.00 | 0.00 | 0.00 |
| **COH** | $000\to100\to101\to111$ | 0.00 | **0.75** | 0.00 |

$$
\boxed{P_{\mathrm{COH}}(111)=0.75 > P_{\mathrm{direct}}(111)=0.50}
$$

**Read (careful):** Path **order** matters — COH beats direct; OHC/HOC never reach 111. But **homing to $000$ never succeeded** ($P_{\mathrm{homed}}{=}0$), and **no path hit all waypoints** ($P(\mathrm{all\ wp}){=}0$). So this is **not** yet proof of intermediate-state reachability from $000$. Likely mechanism: last edge of COH is $H$ ($101\to111$, $P{=}0.75$), while OHC/HOC end on hard $C$ ($011\to111$, $P{=}0$). Next: either start paths from a reachable soft state (e.g. $100$), or raise reps and restrict analysis to successful homes — still no new $v$.

`data/results/sync_phase9_hamming_paths.{json,md}`

---

# Phase 9B — 8-state one-bit transition graph — **done (skeleton)**

All 24 directed Hamming edges, reps=2, frozen 8K. free→home→one-bit; all-pairs max-product paths.

**Best $000\to111$ path (recovered):**

$$
000\rightarrow100\rightarrow101\rightarrow111
\quad (P_{\mathrm{path}}\approx0.06)
$$

= COH order from Phase 9.

| metric | value |
|--------|------:|
| pairs with path $P\ge0.05$ | **47/56** (84%) |
| universal via paths | **False** (9 weak pairs) |
| edges with home data | 9/24 |
| strongest edge | $101\to111$ (H), $P\|{\mathrm{homed}}=0.75$ |
| easy O band | $010\leftrightarrow011$ (2/2 uncond) |

$$
\boxed{
\text{transition-graph skeleton exists; not yet universal multi-step reachability}
}
$$

**Read:** Supports the abstraction (controller = path planner over the cube). Does **not** yet support “all eight states mutually reachable with reliable edges” — most edges are $0/2$ hits, homes rare, Laplace fills gaps. Hard sink: routes into $000$/$110$ stay weak. Densify high-value edges before asserting universal path control — still no new $v$.

`data/results/sync_phase9_transition_graph.{json,md}`

---

# Phase 9C — Densify sink edges (Beta $p_{05}$) — **done (negative)**

Frozen 8K. Explicit source construct → conditional one-bit. Planner uses $\hat p=k/n$ and $p_{\mathrm{plan}}=\mathrm{Beta}(k{+}1,n{-}k{+}1)$ 5th pct — **no Laplace**.

| edge | $n$ | $k$ | $p_{05}$ |
|------|----:|----:|---------:|
| $010\to000$, $100\to000$ | 8 | 0 | 0.006 |
| $010\to110$, $100\to110$, $111\to110$ | 8 | 0 | 0.006 |
| $001\to000$ | 0 | — | source not reproducible |
| $101\to100$ | 8 | 1 | 0.041 |

Connectivity at $\tau\in\{0.5,0.7\}$: **SCC = False** (0 edges kept — even best 9B $2/2$ only reaches $p_{05}\approx0.37$).

$$
\boxed{\text{sinks }000/110\text{ have no reliable in-edge under frozen 8K}}
$$

**Read:** Soft sources construct; **transitions into sinks fail**. Universal path control is blocked at the sink cut, not by missing planner abstraction. Do not use the 9B Laplace graph as a planner.

`data/results/sync_phase9c_densify_edges.{json,md}`

---

# Phase 9D — Target-aware shift / $\Gamma$ — **done (offline)**

Reanalysis of 9C (n=48) + 9B trajectories. No new $v$.

$$
E=\|m^*-S\|_1,\quad
P_\downarrow=P(E_{\mathrm{after}}<E_{\mathrm{before}}),\quad
P_\uparrow=P(E_{\mathrm{after}}>E_{\mathrm{before}}),\quad
\Gamma=P_\downarrow-P_\uparrow.
$$

**vs local $m^*=s'$ (intended sink/edge):** every densified sink-in edge has $\Gamma\le 0$. Worst: $010\to000$, $\Gamma=-0.88$ with $P_{\mathrm{shift}}=1$ — the system **moves**, but **away** from the sink.

**vs global $m^*=111$:** several of those same interventions are helpful — e.g. $010\to000$ attempt has $\Gamma=+0.88$ toward 111. Channel aggregate on 9C neighborhood: H $\Gamma=+0.69$, C $+0.56$, O $-0.75$.

$$
\boxed{\text{edge }P(s\to s')\text{ hides anti-sink / pro-111 drift; use }\Gamma}
$$

`data/results/sync_phase9d_shift.{json,md}`

---

# Phase 9E — Offline $\Gamma$-planner — **done**

Empirical kernel from 9B+9C (96 transitions, 17 $(s,a)$ cells). No new $v$.

$$
a^*=\arg\max_{a\in\{C,H,O\}}\Gamma(a\mid s,m^*)
\quad\text{vs fixed }H\to C\to O.
$$

| result | value |
|--------|------:|
| mean $\Delta P_{\mathrm{hit}}(\Gamma-\mathrm{fixed})$ | **+0.153** |
| pairs $\Gamma$ / fixed / tie | 20 / 7 / 15 |
| $m^*$ with any offline hit ($\Gamma$ / fixed) | **6/8** / 5/8 |
| retro $P(\Gamma$ better mean $\Delta E)$ | **0.61** |
| coverage gap as $s_0$ | $000$, $001$ |

Sign flip confirmed: at $010$, $a^*$ is $C$ for $m^*=000$ but $O$ for $m^*=111$.

**Caveat:** for $m^*=111$, fixed often still wins (1-step $\Gamma$ can be myopic — e.g. $O$ from soft states). Sinks $000$/$001$ remain unreachable offline. Positive average $\Delta P_{\mathrm{hit}}$ supports a **live $\Gamma$-choice** experiment next — not a new actuator.

`data/results/sync_phase9e_gamma_planner.{json,md}`

---

# Phase 9F — Live $\Gamma$-policy (scoped) — **done**

Frozen 8K. Empirical $a^*=\arg\max_{a\in\mathrm{relevant}}\Gamma(a\mid s,m^*)$ vs fixed $H\to C\to O$. Targets $\{010,011,100,101,110\}$, reps=4, paired $S_0$, $E_0>0$.

| policy | $P_{\mathrm{hit}}$ | mean $\Delta E$ | mean $P(E\downarrow)$ |
|--------|-------------------:|----------------:|----------------------:|
| **$\Gamma$** | **0.65** | **1.45** | **0.62** |
| fixed | 0.40 | 0.75 | 0.45 |

$$
\Delta P_{\mathrm{hit}}(\Gamma-\mathrm{fixed})=+0.25
$$

Per-$m^*$: big gains on $011$ ($+0.75$), $100$ ($+0.50$), $101$ ($+0.25$); $010$ favors fixed ($-0.25$); $110$ tied at $0.25$. Pairs: $\Gamma$-only 7 / fixed-only 2 / both 6.

$$
\boxed{\text{live confirmation: action selection }>\text{ new actuator (scoped)}}
$$

Next: expand kernel (esp. $000$/$001$/$111$), then full 8-way — still no new $v$.

`data/results/sync_phase9f_live_gamma.{json,md}`

---

# Phase 9G — Kernel expand + anti-stagnation — **done**

Offline $\Gamma'$ on 9F paths + live densify of sticky cells around $100/010/011$ (+64 transitions).

**Anti-stagnation:** on sticky no-ops, $\Gamma'$ switches $C\to H$ (5 rescues) — same first action fixed used to beat gamma on $010$.

**`100|C` densified:** $P_{\mathrm{shift}}=0.38$ ($P_{\mathrm{stay}}=0.62$). $\Gamma_C$ stays $>0$ because moves that do occur help $m^*=010$ — so this is **insufficient memory**, not a fully wrong $\Gamma$.

$$
\boxed{P_{\mathrm{stay}}(100|C)\approx0.62;\quad\text{next = live }\Gamma'\text{ (not full 8-way yet)}}
$$

`data/results/sync_phase9g_kernel_expand.{json,md}` (+ `sync_phase9g_kernel.json`)

---

# Phase 9H — Live $\Gamma'$ vs $\Gamma$ vs fixed — **done**

Scoped 5 targets, reps=4, 9G kernel, frozen 8K. Critical signature **holds**:

| policy | $P_{\mathrm{hit}}$ | $\Delta E$ | $P(\mathrm{repeat}\mid\mathrm{noop})$ | $P(\mathrm{rescue}\mid\mathrm{noop})$ |
|--------|-------------------:|-----------:|--------------------------------------:|--------------------------------------:|
| **$\Gamma'$** | **0.60** | **1.20** | **0.20** | **0.80** |
| $\Gamma$ | 0.55 | 1.00 | 1.00 | 0.00 |
| fixed | 0.55 | 1.00 | 1.00 | 0.00 |

$$
\boxed{P(\mathrm{repeat}\mid\mathrm{noop}):1.0\to0.2;\quad P(\mathrm{hit}):0.55\to0.60}
$$

Gain is specifically from one-step memory, not from retuning $\Gamma$. Next: expand kernel on remaining failures → full 8 targets.

`data/results/sync_phase9h_live_antistag.{json,md}`

---

# Phase 9I — Kernel expand + full 8-way — **done (partial reach)**

Free-batch soft densify (+9G) then 8-way reps=2, max_steps=4, frozen 8K + $\Gamma'$.

| policy | $P_{\mathrm{hit}}$ | $P_{\mathrm{ever}}$ | ever $m^*$ |
|--------|-------------------:|--------------------:|-----------:|
| $\Gamma'$ / $\Gamma$ | 0.44 | 0.44 | **5/8** |
| fixed | 0.38 | 0.38 | 4/8 |

Hard: $111$ reachable ($P_{\mathrm{ever}}=1$); **$000$, $001$, $110$ still $0$**. No $\Gamma'$ vs $\Gamma$ gap at this budget (anti-stag still cuts repeats $1.0\to0.62$).

$$
\boxed{\text{not universal yet; sinks }000/001/110\text{ remain the gap}}
$$

`data/results/sync_phase9i_kernel_8way.{json,md}`

---

# Phase 9J — Sink-seeking hard targets — **done (hard gate pass; 8-way not universal)**

Densify producers `011|H,C`, `100|O`, `111|C,H`. Live hard $m^*\in\{000,001,110\}$: $\Gamma'$ vs sink-seek $\lambda=0$.

**Critical fix:** pool over **all channels**, not Hamming-relevant only (else `111|C→110` / `011|C→001` are unreachable by construction).

| policy (hard) | $P_{\mathrm{ever}}$ | ever $m^*$ |
|---------------|--------------------:|-----------:|
| **sink-seek (all-ch)** | **0.25** | **3/3** |
| $\Gamma'$ | 0.08 | 1/3 |

Per hard: SS $P_{\mathrm{ever}}=0.25$ each for $000/001/110$. Retention: all exits leave (`stay=0`).

Full 8-way hybrid (reps=2): still **5/8** ever (same soft set as 9I); hard not reproduced at that $S_0$ budget.

$$
\boxed{\text{acquisition objective mismatch confirmed; retention + denser 8-way still open}}
$$

`data/results/sync_phase9j_sink_seeking.{json,md}`

---

# Phase 9K — $000\to(010\mid 011)\to 111$ escape — **done (path complete)**

Acquire $000$ (sink-seek), forced escape $O$/$H$, then $\Gamma'$ to $111$. reps=8; $2/8$ acquired $000$.

| arm | forced | $P_{\mathrm{ever}}111\mid\mathrm{acq}$ |
|-----|--------|----------------------------------------:|
| via_011 | $O$ | **0.50** |
| via_010 | $H$ | **1.00** |

Given acquisition, **any-arm $P(\mathrm{ever}111)=1.0$**. Observed escapes include $000\to111$ (direct $H$), $000\to\cdots\to011\to\cdots\to111$, and $000\xrightarrow{O}000\xrightarrow{H}111$.

$$
\boxed{000\xrightarrow{O/H}\text{soft / direct}\xrightarrow{\Gamma'}111\text{ — sink}\to 111\text{ path closed}}
$$

`data/results/sync_phase9k_sink_to_111.{json,md}`

---

# Milestone — all 8 states reachable (not yet reliable 8-way)

Across 9I (soft+$111$), 9J (hard sink-seek), 9K (escape from $000$):

$$
\boxed{\forall m^*\in\{0,1\}^3,\ \exists\text{ an empirically observed control path to }m^*}
$$

| Claim | Status |
|-------|--------|
| all 8 states reachable (frozen actuators) | **supported** |
| universal 8-way reliable target-conditioned attainment | **not yet** |

Hard $000/001/110$: demonstrated in dedicated sink-seek (low $P_{\mathrm{acquire}}$); retention $P_{\mathrm{retain}}\approx 0$ in 9J probes. Soft+$111$: Γ′ multi-step in full 8-way.

**Next (no new $v$):** for each $m^*$, especially hard sinks, measure

$$
P_{\mathrm{final}}(m^*)=P_{\mathrm{acquire}}(m^*)\cdot P_{\mathrm{retain}}(m^*\mid\mathrm{acquire}).
$$

Progression: causal dirs → boundary → timing → composition → target-conditioned planning → state-memory → **all 8 reachable** → **reliable 8-way attainment**.

---

# Phase 9L — $P_{\mathrm{acquire}}\times P_{\mathrm{retain}}$ — **done (attainment gap)**

Hybrid planner (Γ' soft / sink-seek hard), reps=4, 2-step stay-seek after landing.

| $m^*$ | $P_{acq}$ | $P_{ret}\mid acq$ | $P_{\mathrm{final}}$ |
|-------|----------:|------------------:|---------------------:|
| 000 | 0.25 | 0.00 | 0.00 |
| 001 | 0.00 | — | 0.00 |
| 010 | 0.75 | 0.00 | 0.00 |
| **011** | **1.00** | **0.25** | **0.25** |
| 100 | 0.75 | 0.00 | 0.00 |
| 101 | 0.50 | 0.00 | 0.00 |
| 110 | 0.50 | 0.00 | 0.00 |
| 111 | 0.25 | 0.00 | 0.00 |

$P_{acq}>0$: **7/8**. $P_{\mathrm{final}}>0$: **1/8** (only `011`). Retention is the bottleneck — including soft targets under forced stay-seek.

$$
\boxed{P_{\mathrm{final}}=P_{acq}\cdot P_{ret};\quad\text{reliable 8-way not yet; retain}\approx 0}
$$

`data/results/sync_phase9l_acquire_retain.{json,md}`

---

# Phase 9M — Target hold vs planner — **done (mixed)**

After hybrid acquire: **hold** (no steer) vs **planner** (stay-seek), $n_{hold}=2$, reps=4.

| | mean $P_{stay}\mid acq$ | $m^*$ with $P_{final}>0$ |
|--|------------------------:|-------------------------:|
| hold | 0.15 | 3/8 |
| planner | **0.20** | **4/8** |

hold ≫ planner: **False**. Soft states sometimes retain; hard $000/001/110$ stay $\approx 0$ under both. Loss is not mainly “planner keeps poking” — need a retention mechanism beyond no-steer. Full acquire+hold 8-way not auto-run.

$$
\boxed{\text{mixed: not controller-only destruction; intrinsic instability on hard sinks}}
$$

`data/results/sync_phase9m_target_hold.{json,md}`

---

# Phase 9N — Just-in-time / receding-horizon — **done (JIT-RH wins; not universal)**

Same frozen $v_c$/α. No hold. Arms: JIT-RH ($C\to H\to O$ one remaining bit/step), JIT-oneshot (all mismatches in one episode), compose 8K (cumulative $H\to C\to O$).

| arm | $P_{hit}$ | $m^*$ with hit$>0$ |
|-----|----------:|-------------------:|
| **JIT-RH** | **0.38** | **5/8** |
| compose 8K | 0.19 | 4/8 |
| JIT-oneshot | 0.09 | 2/8 |

Per hard: $000/001$ still 0 under all arms; $110$ only compose (0.25). Soft+$111$ improve under JIT-RH.

$$
\boxed{\text{JIT-RH }>\text{ compose; decision-site timing helps; hard sinks still open}}
$$

`data/results/sync_phase9n_jit_horizon.{json,md}`

---

# Phase 9O — Target-hitting lookahead — **done (kernel-limited; gate fail)**

Finite-horizon $V_k=\max_a\mathbb E[V_{k-1}]$ for hard $\{000,001,110\}$, T=3, vs $\Gamma'$.

**Offline:** $001$: $011\xrightarrow{C}001$ ($V\approx0.16$); $110$: $101\to111\to110$ ($V\approx0.15$); **$000$: $V=0$** — no landings in merged kernel.

**Live** (reps=4):

| policy | $P_{\mathrm{ever}}$ | hard ever |
|--------|--------------------:|----------:|
| hitting | 0.08 | 1/3 ($110$ only) |
| $\Gamma'$ | **0.17** | 1/3 ($110$) |

Hitting did **not** beat $\Gamma'$. $001$ path exists offline but failed live; $000$ unplannable until kernel has sink-in mass.

$$
\boxed{\text{lookahead OK in principle; sparse/zero }P(\to 000)\text{ blocks planning}}
$$

`data/results/sync_phase9o_hitting_planner.{json,md}`

---

# Phase 9P — Active producer densification — **done (partial mass restored)**

Target-sign densify of known producers (not bit-flip). Beta CI. Then live max-$P(\mathrm{land})$.

**Densify (key rows):**

| target | best producer | $n$ | $k$ | $\hat p$ | $p_{05}$ |
|--------|---------------|----:|----:|---------:|---------:|
| 000 | `111\|H` | 11 | 1 | 0.09 | 0.03 |
| 000 | `011\|C` | 12 | 1 | 0.08 | 0.03 |
| 000 | `011\|H` | 12 | 0 | 0.00 | ~0 |
| 001 | `011\|C` | 12 | 0 | 0.00 | ~0 |
| 110 | `111\|H` | 3 | 1 | 0.33 | 0.10 |

**Live $P_{\mathrm{ever}}$:** max-$P(\mathrm{land})$ — `110`:**1.00**, `000/001`:**0**. Gate **1/3**.

$$
\boxed{000\text{ mass restored (rare); }011|H\to000\text{ falsified under target-sign; }110\text{ acq OK}}
$$

`data/results/sync_phase9p_producer_densify.{json,md}` (+ `sync_phase9p_kernel.json`)

---

# Phase 9Q — Bidirectional beam search — **done (hard gate pass)**

Hypothesis: target-sign is too restrictive; hard reachability needs short-horizon
search over \(\mathcal A=\{\pm C,\pm H,\pm O\}\) with **actual rollouts** (not kernel).

Protocol: beam \(K{=}3\), horizon \(T{=}3\), reps=2, free \(S_0\neq m^*\), score = exact hit then \(-E\).
No new \(v\). No densify.

| $m^*$ | $P_{hit}$ | example |
|-------|----------:|---------|
| `000` | 0.50 | `010 →(+O)→ 010 →(-C)→ 000` |
| `001` | **1.00** | `011 →(+C)→ 001` (anti–target-sign on C) |
| `110` | **1.00** | `111 →(+C)→ 100 →(+H)→ 110` |

Hard gate ($P_{hit}>0$ all 3): **3/3**.

Key contrast with 9P: under $m^*=001$ target-sign, $P(011\xrightarrow{C}001)=0/12$;
beam finds $011\xrightarrow{+C}001$ in one step — polarity opposite the target bit.

$$
\boxed{\text{hard sinks via planning over }\pm v_C,\pm v_H,\pm v_O\text{ — not stronger actuators / denser kernel}}
$$

`data/results/sync_phase9q_bidir_beam.{json,md}` · `scripts/run_phase9q_bidir_beam.py`

**Next (optional):** cache winning paths; hybrid controller (Γ' easy / beam hard); then re-measure full 8-way $P_{\mathrm{acq}}\cdot P_{\mathrm{ret}}$.

---

# Phase 9R — Hybrid $\Gamma'$ + beam fallback — **done (acq 8/8; final open)**

Paired $S_0$. Arms: $\Gamma'$ alone vs $\Gamma'$+beam when $\max\Gamma'<\tau$ / predicted no-op / observed no-op.
$\tau{=}0$, $K{=}3$, $T{=}3$, reps=2, $n_{\mathrm{retain}}{=}2$. No new $v$.

| arm | states $P_{acq}>0$ | states $P_{final}>0$ |
|-----|-------------------:|---------------------:|
| $\Gamma'$ | **6/8** (miss `000`,`110`) | **1/8** (`100`) |
| $\Gamma'$+beam | **8/8** | **1/8** (`100`) |

Hybrid $P_{acq}$: `000` 0.50; all others **1.00**. Retention collapses almost everywhere ($P_{ret}{=}0$ except `100`).

**Sign finding (successful transitions, hybrid):**

| | rate |
|--|-----:|
| target-sign | 0.44 |
| opposite-sign | **0.56** |
| opposite among beam-sourced | **0.68** (n=22) |

$$
\boxed{\text{acq 8/8 via }\Gamma'+\mathrm{beam};\ \text{opposite polarity often required};\ P_{\mathrm{final}}\text{ still retention-limited}}
$$

`data/results/sync_phase9r_hybrid.{json,md}` · `scripts/run_phase9r_hybrid.py`

**Next:** retention / hold after planned acquisition (not new $v$); optional path cache from beam hits.

---

# Artifacts

- `data/results/sync_8way_validation.json` (+ `_enriched`, `_analysis`, `_r2`)
- `data/results/sync_8way_surrogate.{json,md}`
- `data/results/sync_boundary_crossing.{json,md}`
- `data/results/sync_phase7_live_boundary.{json,md,cache.npz}`
- `data/results/sync_phase8_live_8way.{json,md}`
- `data/results/sync_phase8b_target_sign.{json,md}`
- `data/results/sync_phase8c_same_site.{json,md}`
- `data/results/sync_phase8d_sequential_cho.{json,md}`
- `data/results/sync_phase8e_8way_corrected.{json,md}`
- `data/results/sync_phase8f_co_same_site.{json,md}`
- `data/results/sync_phase8g_o_dose.{json,md}`
- `data/results/sync_phase8h_o_gain.{json,md}`
- `data/results/sync_phase8i_ch_composition.{json,md}`
- `data/results/sync_phase8j_c_decision.{json,md}`
- `data/results/sync_phase8k_c_gain_compose.{json,md}`
- `data/results/sync_phase8k_8way_validation.{json,md}`
- `data/results/sync_phase9_hamming_paths.{json,md}`
- `data/results/sync_phase9_transition_graph.{json,md}`
- `data/results/sync_phase9c_densify_edges.{json,md}`
- `data/results/sync_phase9d_shift.{json,md}`
- `data/results/sync_phase9e_gamma_planner.{json,md}`
- `data/results/sync_phase9f_live_gamma.{json,md}`
- `data/results/sync_phase9g_kernel_expand.{json,md}`
- `data/results/sync_phase9g_kernel.json`
- `data/results/sync_phase9h_live_antistag.{json,md}`
- `data/results/sync_phase9i_kernel_8way.{json,md}`
- `data/results/sync_phase9i_kernel.json`
- `data/results/sync_phase9j_sink_seeking.{json,md}`
- `data/results/sync_phase9j_kernel.json`
- `data/results/sync_phase9k_sink_to_111.{json,md}`
- `data/results/sync_phase9l_acquire_retain.{json,md}`
- `data/results/sync_phase9m_target_hold.{json,md}`
- `data/results/sync_phase9n_jit_horizon.{json,md}`
- `data/results/sync_phase9o_hitting_planner.{json,md}`
- `data/results/sync_phase9p_producer_densify.{json,md}`
- `data/results/sync_phase9p_kernel.json`
- `data/results/sync_phase9q_bidir_beam.{json,md}`
- `data/results/sync_phase9r_hybrid.{json,md}`

---

# Do not

- No new vectors / no controller retuning without a new failure mode
- Do not raise $\alpha_O$ / re-convert O / re-convert C
- **Do not skip C** as success for universal 8-way
- **Do not call $P(\mathrm{hit}){=}0.20$ universal 8-way synchronization**
- Do not treat Phase 9 COH as proven intermediate-state reachability from $000$ (homes failed; waypoints missed)
- Do not claim universal multi-step reachability from the reps=2 skeleton graph (Laplace + rare homes)
- **Do not use Laplace-smoothed $P_{\mathrm{path}}$ as a planner** — Phase 9C: sinks lack reliable in-edges
- Prefer $\Gamma$ / $P_\downarrow$ over brittle exact-edge $P(s\to s')$ when judging actuator usefulness
- Do not claim universal 8-way from offline $\Gamma$ alone (gaps at $000$/$001$; $m^*=111$ still favors fixed)
- Do not overclaim Phase 9F as full 8-way (scoped 5 targets; $010$/$110$ still weak)
- Do not claim Phase 9I alone as universal ($000$/$001$/$110$ ever-reach still 0 under $\Gamma'$ in that run)
- **Do not equate “all 8 reachable” with reliable 8-way attainment** — hard sinks need $P_{\mathrm{acquire}}\cdot P_{\mathrm{retain}}$
- Do not search for another actuator for the hard-sink gap — problem is planning/retention
- Do not restrict hard sink-seek to Hamming-relevant bits (blocks side-effect producers)
- Do not equate hard-gate acquisition with retention or with full 8-way at low reps
- Do not equate target-bit sign with globally correct action (9P/9Q: anti-sign can be required)
- Do not densify the empirical kernel hoping it alone unlocks hard sinks (9O/9P → 9Q beam)
- Do not tune $\Gamma$ weights / complex history policies yet — binary anti-stag first
- Conditional C-skip = diagnostic only
- Do not treat $\mathcal L_{8\mathrm{way}}$ on fixed templates as discrete sync proof
- Do not overclaim from reps=2 exploratory runs
- Locked actuators: $\alpha_C{=}5$ stem, $\alpha_H{=}1.5$ decision-token, $\alpha_O{=}1.5$ early-FINAL; **selection** may be $\Gamma'$ / bidirectional beam fallback (9R)
- Do not treat acq 8/8 as universal attainment — $P_{\mathrm{final}}$ still retention-bound (9R)
