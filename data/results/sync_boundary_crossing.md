# Boundary-crossing diagnostic

> Target-signed margin movement ⇏ behavioral boundary crossing.

Source: `sync_8way_surrogate.json` (frozen $v_c$) + free-run calibration.

## Headline

$$
\boxed{
v_c\text{ moves probe margins on wrong bits a lot, but }P(W\to C)\approx 0.43
}
$$

and critically:

$$
\boxed{
\text{unsteered }M_k\text{ is deterministic (fixed site templates) — not episode-conditioned}
}
$$

| Channel | free-run $M$ | std | $P(S{=}1\mid M{>}0)$ |
|---------|--------------|-----|----------------------|
| C | +7.28 | **0** | 0.20–0.21 |
| H | +3.07 | **0** | 0.91–0.92 |
| O | +3.29 | **0** | 0.67–0.78 |

So $M_k$ is a **mechanistic probe**, not a calibrated discrete decision variable for the live episode. That alone can explain large $\mathbb{E}[-\Delta\mathcal L]$ with $\mathrm{corr}(-\Delta L,\Delta E)\approx 0$.

---

## A — Per-bit transitions (converted)

| Channel | P(W→C) | P(C→W) | W2C | W2W | C2C | C2W |
|---------|--------|--------|-----|-----|-----|-----|
| C | 0.43 | 0.56 | 3 | 4 | 4 | 5 |
| H | 0.43 | 0.22 | 3 | 4 | 7 | 2 |
| O | 0.44 | 0.43 | 4 | 5 | 4 | 3 |

### All arms P(W→C)

| Arm | C | H | O |
|-----|---|---|---|
| none | 0.00 | 0.00 | 0.00 |
| predictive | 0.22 | 0.11 | 0.62 |
| converted | 0.43 | 0.43 | 0.44 |
| random | 0.50 | 0.14 | 0.78 |

---

## Δ~M on wrong vs already-correct bits (converted)

| Channel | Δ~M \| wrong₀ | Δ~M \| correct₀ |
|---------|----------------|------------------|
| C | **+4.02** (n=7) | +0.07 (n=9) |
| H | **+14.02** (n=7) | +0.33 (n=9) |
| O | **+1.65** (n=9) | +0.09 (n=7) |

**Not** “deepening already-correct bits.” Movement is concentrated on **wrong** bits — yet flips stay ~43%.

---

## C — $\mathcal{L}_{8\mathrm{way}}$ vs $\mathcal{L}_{\mathrm{wrong}}$ / $D_{\mathrm{wrong}}$

| Arm | E[−ΔL] | E[−ΔL_wrong] | E[−ΔD_wrong] | corr(−ΔL,ΔE) | corr(−ΔL_w,ΔE) | corr(−ΔD_w,ΔE) |
|-----|--------|---------------|---------------|--------------|----------------|----------------|
| none | +0.000 | +0.000 | +0.000 | — | — | — |
| predictive | +0.061 | +0.113 | +0.100 | +0.039 | −0.185 | +0.017 |
| converted | +2.075 | +2.111 | +8.823 | −0.011 | −0.100 | −0.100 |
| random | −0.154 | −0.173 | −0.259 | −0.309 | −0.267 | −0.538 |

Restricting to wrong bits **does not** restore correlation with ΔE.

---

## B — P(W→C | $\tilde M_0$) (converted)

Probe $\tilde M_0$ has almost no diversity pre-steer (deterministic $M_0$). Bins are mostly uninformative; H wrong trials all sit at $\tilde M_0=-3.07$.

---

## D — Calibration

### Free-run (n=24): $M$ never crosses 0

| Channel | P(S=1\|M<0) | P(S=1\|M>0) | sign accuracy |
|---------|-------------|-------------|---------------|
| C | — | 0.21 | 0.21 |
| H | — | 0.92 | 0.92 |
| O | — | 0.67 | 0.67 |

### After converted steer ($M_1$ vs $S_{\mathrm{final}}$)

| Channel | mean $M_1$ | P(S=1\|M<0) | P(S=1\|M>0) | sign acc |
|---------|------------|-------------|-------------|----------|
| C | +8.65 | — | 0.44 | 0.44 |
| H | −2.70 | **0.57** | 0.89 | 0.69 |
| O | +2.98 | — | 0.75 | 0.75 |

Only H under steer shows a partial $M\leftrightarrow S$ link; even then $P(S{=}1\mid M{<}0)=0.57$ (far from a sharp threshold at 0).

---

## Gate

- mean P(W→C) converted: **0.43**
- Δ~M concentrated on wrong bits: **True**
- Δ~M deepening already-correct: **False**
- L_wrong predicts ΔE better: **False**
- M calibrated to S (free-run): **False**

$$
\boxed{
v_c\text{ provides strong continuous causal movement on site probes,}
}
$$

$$
\boxed{
\text{but those probes are not the episode’s discrete decision boundary.}
}
$$

Missing object (later): a **boundary-localized, episode-conditioned** map from activation control → bit flip — not a new global $v$.

No new $v$. No policy yet.
