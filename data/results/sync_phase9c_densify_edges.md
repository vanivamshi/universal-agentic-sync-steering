# Phase 9C — Densify weak edges (Beta $p_{05}$ planner)

> Explicit source construction + conditional one-bit trials. No Laplace. $p_{\mathrm{plan}}=\mathrm{Beta}(k{+}1,n{-}k{+}1)$ 5th pct.

Frozen 8K unchanged. Budget spent on **sink neighborhood** (000 / 110), not already-strong O-band edges.

## Densified conditional edges

| edge | ch | n | k | $\hat p$ | $p_{plan,05}$ | construct_rate | note |
|------|----|--:|--:|--------:|--------------:|---------------:|------|
| `001→000` | O | 0 | 0 | — | 0.000 | 0.00 | source 001 not reproducible |
| `010→000` | H | 8 | 0 | 0.00 | 0.006 | 0.62 | soft→000 dead |
| `100→000` | C | 8 | 0 | 0.00 | 0.006 | 0.80 | soft→000 dead |
| `010→110` | C | 8 | 0 | 0.00 | 0.006 | 0.62 | soft→110 dead |
| `100→110` | H | 8 | 0 | 0.00 | 0.006 | 0.89 | soft→110 dead |
| `111→110` | O | 8 | 0 | 0.00 | 0.006 | 0.89 | into 110 dead |
| `101→100` | O | 8 | 1 | 0.12 | 0.041 | 0.50 | soft reverse weak |

## Strong connectivity (planner graph)

Even best 9B edges are only $n{=}2$ (e.g. `010→011` $2/2$ → $p_{05}\approx0.37$), so **no edge** clears $\tau\in\{0.5,0.7\}$ until strong edges are densified too.

| $\tau$ | edges kept | reachable pairs | strongly connected |
|--------:|-----------:|----------------:|:------------------:|
| 0.5 | 0 | 0/56 | **False** |
| 0.7 | 0 | 0/56 | **False** |

## Sink verdict

- **No reliable incoming edge** into `000` or `110` among tested soft/hard sources ($0/8$ on every completed sink-in edge).
- Soft sources (`010`,`100`) construct fine; the **transition itself** fails.
- `101→100` is the only densified non-zero: $1/8$, $p_{05}=0.041$ — still far below $\tau$.

$$
\boxed{\text{not strongly connected; sinks 000/110 lack reliable in-edges under frozen 8K}}
$$

Universal multi-step reachability is **not** established. Next would require either densifying *working* edges for a softer $\tau$ band, or new actuation for sink transitions — not more Laplace planning.
