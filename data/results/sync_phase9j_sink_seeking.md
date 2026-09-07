# Phase 9J — Sink-seeking for hard targets

> $a^*=\arg\max_a\big[P(S\'=m^*\mid s,a)+\lambda\Gamma\'(a)\big]$ over **all channels** (not Hamming-relevant only).
First test $\lambda=0$. Frozen 8K. No new $v$.

expand_reps=8, hard_reps=4, 8way_reps=2, max_steps=7, seed=20260929.

## Producer landing estimates

| cell | target | n | $P(\mathrm{land})$ | hist |
|------|--------|--:|--------------------:|------|
| `011|C` | `001` | 20 | **0.10** | 001:2 100:8 111:1 101:6 010:1 |
| `011|H` | `000` | 23 | **0.00** | 100:11 011:3 111:7 110:1 101:1 |
| `100|O` | `110` | 26 | **0.00** | 011:17 100:5 010:2 111:2 |
| `111|C` | `110` | 11 | **0.18** | 011:1 100:4 101:2 110:2 111:1 |
| `111|H` | `110` | 10 | **0.00** | 111:3 010:2 011:5 |

## Hard-target live: $\Gamma'$ vs sink-seek (all-channels)

| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\Delta E$ | $P(\mathrm{stay}\mid\mathrm{land})$ |
|--------|--:|----------:|-----------:|-----------------:|---------------------------------------:|
| **sink-seek** | 12 | 0.00 | **0.25** | 0.00 | 0.00 |
| $\Gamma'$ | 12 | 0.08 | **0.08** | -0.25 | — |

### Per hard $m^*$

| $m^*$ | $P_{ever}$ SS | $P_{hit}$ SS | $P_{ever}$ $\Gamma'$ | $P_{hit}$ $\Gamma'$ | stay|land SS |
|-------|---------------:|-------------:|---------------------:|---------------------:|-------------------:|
| `000` | **0.25** | 0.00 | 0.00 | 0.00 | 0.00 |
| `001` | **0.25** | 0.00 | 0.00 | 0.00 | 0.00 |
| `110` | **0.25** | 0.00 | 0.25 | 0.25 | 0.00 |

### Retention exits (sink-seek)

- `000`: H→`010` stay=0
- `001`: C→`111` stay=0
- `110`: H→`111` stay=0

## Hard gate

- $P_{\mathrm{ever}}>0$ for all three under sink-seek: **True** (3/3; $\Gamma'$ 1/3)

Round-1 (relevant-only pool) failed the gate (1/3). All-channels pool recovers **3/3**.

## Full 8-way: hybrid ($\Gamma'$ soft + sink-seek hard) vs $\Gamma'$

| policy | n | $P_{hit}$ | $P_{ever}$ | mean $\Delta E$ | ever $m^*$ |
|--------|--:|----------:|-----------:|-----------------:|-----------:|
| **hybrid** | 16 | 0.25 | **0.44** | 0.44 | **5/8** |
| $\Gamma'$ | 16 | 0.25 | **0.44** | 0.62 | **5/8** |

| $m^*$ | $P_{ever}$ hyb | $P_{hit}$ hyb | $P_{ever}$ $\Gamma'$ |
|-------|---------------:|--------------:|---------------------:|
| `000` | 0.00 | 0.00 | 0.00 |
| `001` | 0.00 | 0.00 | 0.00 |
| `010` | 0.50 | 0.00 | 0.50 |
| `011` | 1.00 | 0.50 | 1.00 |
| `100` | 1.00 | 0.50 | 1.00 |
| `101` | 0.50 | 0.50 | 0.50 |
| `110` | 0.00 | 0.00 | 0.00 |
| `111` | 0.50 | 0.50 | 0.50 |

Universal ever-reach (hybrid): **False**

8-way reps=2 did not reproduce hard ever-reach (different free $S_0$ budget). Hard gate stands on the dedicated hard-target test.

$$\boxed{\text{sink-seek all-channels: hard }P_{\mathrm{ever}}>0\text{ for }000/001/110;\text{ retention + 8-way coverage still open}}$$
