# Phase 9Q — Bidirectional beam search ($\pm C,\pm H,\pm O$)

> Actual rollouts. Hard $m^*\in\{000,001,110\}$. No new $v$. No kernel. Score = exact hit, then $-E$.

beam=3, horizon=3, reps=2, seed=20261005.

## Results

| $m^*$ | n | $P_{hit}$ | mean $E_f$ | example path |
|-------|--:|----------:|-----------:|--------------|
| `000` | 2 | **0.50** | 0.50 | `010→010→000` acts=['+O', '-C'] |
| `001` | 2 | **1.00** | 0.00 | `011→001` acts=['+C'] |
| `110` | 2 | **1.00** | 0.00 | `111→100→110` acts=['+C', '+H'] |

Hard gate ($P_{hit}>0$ all 3): **True** (3/3)

Primary: ∀m*∈{000,001,110} P_hit>0 under bidir beam. If yes: planning over signed actuators is the missing mechanism.

$$\boxed{\text{beam over }\pm v_C,\pm v_H,\pm v_O\text{ — planning, not new actuators}}$$
