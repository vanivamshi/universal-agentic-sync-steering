# Phase 9B — 8-state one-bit transition graph

> Frozen 8K. For each directed Hamming edge: free $\to$ home to $s$ $\to$ one-bit to $s'$. Then all-pairs max-product paths. No new $v$.

reps=2/edge, n_edges=24, seed=20260922, Laplace α=0.5, min_edge_p=0.05.

## Edge matrix $P(s\to s')$ (smoothed; prefer $|$homed when $n_{homed}{>}0$)

| from\to | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|------|------:|------:|------:|------:|------:|------:|------:|------:|
| **000** | — | 0.17* | 0.17* | · | 0.50* | · | · | · |
| **001** | 0.17* | — | · | 0.50* | · | 0.50* | · | · |
| **010** | 0.17* | · | — | 0.83* | · | · | 0.17* | · |
| **011** | · | 0.17 | 0.83* | — | · | · | · | 0.25 |
| **100** | 0.17 | · | · | · | — | 0.17 | 0.25 | · |
| **101** | · | 0.17* | · | · | 0.25 | — | · | 0.75 |
| **110** | · | · | 0.17* | · | 0.17* | · | — | 0.17* |
| **111** | · | · | · | 0.25 | · | 0.25 | 0.17* | — |

\* = no successful homes; unconditional P after home attempt.

## Strongest / weakest edges

| edge | ch | P_homed | n_homed | P_uncond | home_rate |
|------|----|--------:|--------:|---------:|----------:|
| `010→011` | O | nan | 0 | 0.83 | 0.00 |
| `011→010` | O | nan | 0 | 0.83 | 0.00 |
| `101→111` | H | 0.75 | 1 | 0.50 | 0.50 |
| `000→100` | C | nan | 0 | 0.50 | 0.00 |
| `001→101` | C | nan | 0 | 0.50 | 0.00 |
| `001→011` | H | nan | 0 | 0.50 | 0.00 |
| `011→111` | C | 0.25 | 1 | 0.17 | 0.50 |
| `100→110` | H | 0.25 | 1 | 0.17 | 0.50 |
| … | | | | | |
| `010→000` | H | nan | 0 | 0.17 | 0.00 |
| `101→001` | C | nan | 0 | 0.17 | 0.00 |
| `110→010` | C | nan | 0 | 0.17 | 0.00 |
| `110→100` | H | nan | 0 | 0.17 | 0.00 |
| `110→111` | O | nan | 0 | 0.17 | 0.00 |
| `111→110` | O | nan | 0 | 0.17 | 0.00 |

## All-pairs reachability

- Reachable pairs (edge P≥0.05): **47/56** (84%)
- Strong paths (product P≥0.05): **47/56** (84%)
- $000\to111$ best path: `['000', '100', '101', '111']` P≈0.0625

### Selected path examples

| source | target | path | P_path | hops |
|-------:|-------:|------|-------:|-----:|
| 000 | 111 | `000→100→101→111` | 0.062 | 3 |
| 000 | 110 | `000→100→110` | 0.125 | 2 |
| 111 | 000 | `111→011→010→000` | 0.035 | 3 |
| 100 | 111 | `100→101→111` | 0.125 | 2 |
| 011 | 100 | `011→010→000→100` | 0.069 | 3 |
| 001 | 110 | `001→011→010→110` | 0.069 | 3 |

## Gate

- Mutual reachability (weak): **47/56**
- Universal via paths (strong): **False** (9 weak pairs, mostly into $000$/$001$/$110$)
- $000\to111$ path: **`000→100→101→111`** (COH), $P\approx0.06$
- Edges with successful home: **9/24** (conditional estimates sparse)

$$
\boxed{
\text{transition-graph skeleton exists; not yet universal multi-step reachability}
}
$$

Laplace + reps=2 inflate many zero-hit edges to $P\approx0.17$. Treat as a **planner prior**, not a reliability claim. Densify before asserting universal path control.
