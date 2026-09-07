# Phase 8I — C/H composition (O frozen)

> O locked: $\alpha_O=1.5$, early-FINAL. Compare $C\to H$ vs $H\to C$. Target-sign; H decision-token. No new $v$; no 8-way reopen.

m* n=4, reps=2.

## Behavioral order

| order | P(hit) | ΔE | ΔE_C | ΔE_H | P(C)|_CH | P(H)|_CH | P(both CH) |
|-------|--------|---:|-----:|-----:|----------:|----------:|-----------:|
| C→H→O | 0.38 | +0.38 | -0.12 | +0.50 | 0.38 | 0.75 | 0.38 |
| H→C→O | 0.38 | +0.75 | -0.25 | +0.62 | 0.50 | 0.75 | 0.38 |

| 8H ref (C→H→O) | 0.31 | +0.38 | — | — | 0.56 | 0.69 | — |

### After each stage (bit correct)

| order | P(C) after C | P(H) after C | P(C) after H | P(H) after H |
|-------|-------------:|-------------:|-------------:|-------------:|
| C→H→O | 0.50 | 0.38 | 0.38 | 0.75 |
| H→C→O | 0.50 | 0.75 | 0.50 | 0.75 |

## Live-site cross ratios (Phase-7 $B$, hooked $\Delta B$)

$R_{C|H}=|\Delta B_C(h_H)|/|\Delta B_C(h_0)|$, $R_{H|C}=|\Delta B_H(h_C)|/|\Delta B_H(h_0)|$; cross_ratio = cross-after-prior / cross-alone.

| | $R$ (self) | cross_ratio | $|\Delta B|$ $h_0$ | $|\Delta B|$ prior |
|--|------------:|------------:|-------------------:|-------------------:|
| $C|H$ | **1.68** | **3.19** | 0.082 | 0.137 |
| $H|C$ | **0.73** | **0.79** | 0.412 | 0.300 |

5C ref: C cross_ratio=1.33, H=0.66.

## Gate

- Preferred order: **H_then_C**
- H→C beats C→H: **True**
- Orders similar: **False**
- $R_{C|H}$: **1.6793246319855546**
- $R_{H|C}$: **0.7274426232401952**
- C cross_ratio (5C-style): **3.185008208410052**
- C cross worsens (>1.25): **True**

If H→C outperforms C→H: prefer H first. If orders similar/weak: next is conditional C skip (small policy), not re-convert. O stays frozen. No 8-way reopen.
