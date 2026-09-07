# Phase 9D — Target-aware shift / $\Gamma$ (reanalysis)

> Offline on Phase 9B/9C trajectories. No new $v$, no controller change.

$E=\|m^*-S\|_1$,\quad $P_\downarrow=P(E_{\mathrm{after}}<E_{\mathrm{before}})$,\quad $P_\uparrow=P(E_{\mathrm{after}}>E_{\mathrm{before}})$,\quad $\Gamma=P_\downarrow-P_\uparrow$.

9C conditional trials: **48**. 9B (all home→edge): 48; 9B at-source: 12.

## 9C edges vs local $m^*=s'$ (intended one-bit target)

| edge | ch | n | $P_{hit}$ | $P_{shift}$ | $P_\downarrow$ | $P_\uparrow$ | $P_0$ | $\Gamma$ | mean $\Delta E$ | after hist |
|------|----|--:|----------:|------------:|---------------:|-------------:|------:|----------:|-----------------:------------|
| `100→110` | H | 8 | 0.00 | 0.50 | 0.00 | 0.00 | 1.00 | **0.00** | 0.00 | 111:4 100:4 |
| `100→000` | C | 8 | 0.00 | 0.50 | 0.00 | 0.50 | 0.50 | **-0.50** | 0.75 | 100:4 111:2 110:2 |
| `010→110` | C | 8 | 0.00 | 1.00 | 0.00 | 0.62 | 0.38 | **-0.62** | 0.62 | 101:4 100:3 011:1 |
| `111→110` | O | 8 | 0.00 | 1.00 | 0.00 | 0.62 | 0.38 | **-0.62** | 0.62 | 011:5 010:3 |
| `101→100` | O | 8 | 0.12 | 1.00 | 0.12 | 0.88 | 0.00 | **-0.75** | 1.25 | 011:4 010:3 100:1 |
| `010→000` | H | 8 | 0.00 | 1.00 | 0.00 | 0.88 | 0.12 | **-0.88** | 1.12 | 011:3 101:2 111:2 100:1 |

## Same 9C trials vs global $m^*=111$

| edge | n | $P_\downarrow$ | $P_\uparrow$ | $\Gamma$ | $P_r$ (r=0..3) |
|------|--:|---------------:|-------------:|----------:|----------------|
| `100→110` | 8 | 0.50 | 0.00 | **0.50** | 0.50,0.00,0.50,0.00 |
| `100→000` | 8 | 0.50 | 0.00 | **0.50** | 0.25,0.25,0.50,0.00 |
| `010→110` | 8 | 0.62 | 0.00 | **0.62** | 0.00,0.62,0.38,0.00 |
| `111→110` | 8 | 0.00 | 1.00 | **-1.00** | 0.00,0.62,0.38,0.00 |
| `101→100` | 8 | 0.00 | 0.50 | **-0.50** | 0.00,0.50,0.50,0.00 |
| `010→000` | 8 | 0.88 | 0.00 | **0.88** | 0.25,0.62,0.12,0.00 |

## Channel aggregate vs $m^*=111$

### 9C sink-neighborhood interventions

| ch | n | $P_{shift}$ | $P_\downarrow$ | $P_\uparrow$ | $\Gamma$ |
|----|--:|------------:|---------------:|-------------:|----------:|
| C | 16 | 0.75 | 0.56 | 0.00 | **0.56** |
| H | 16 | 0.75 | 0.69 | 0.00 | **0.69** |
| O | 16 | 1.00 | 0.00 | 0.75 | **-0.75** |

### 9B all edges (S_before = S_home)

| ch | n | $P_{shift}$ | $P_\downarrow$ | $P_\uparrow$ | $\Gamma$ |
|----|--:|------------:|---------------:|-------------:|----------:|
| C | 16 | 0.81 | 0.31 | 0.38 | **-0.06** |
| H | 16 | 0.56 | 0.25 | 0.25 | **0.00** |
| O | 16 | 0.81 | 0.56 | 0.12 | **0.44** |

## Sink read

- `010→000`: $P_{hit}=0.00$ but $P_{shift}=1.00$, $\Gamma=-0.88$ (mean $\Delta E=1.12$)
- `100→000`: $P_{hit}=0.00$ but $P_{shift}=0.50$, $\Gamma=-0.50$ (mean $\Delta E=0.75$)
- `010→110`: $P_{hit}=0.00$ but $P_{shift}=1.00$, $\Gamma=-0.62$ (mean $\Delta E=0.62$)
- `100→110`: $P_{hit}=0.00$ but $P_{shift}=0.50$, $\Gamma=0.00$ (mean $\Delta E=0.00$)
- `111→110`: $P_{hit}=0.00$ but $P_{shift}=1.00$, $\Gamma=-0.62$ (mean $\Delta E=0.62$)

## Gate

- Any $\Gamma>0$ (local) on 9C densified edges: **False**
- Best local $\Gamma$: `('100→110', 0.0, 0.0, 0.0, 0.5, 0.0, 8)`
- Worst local $\Gamma$: `('010→000', -0.875, 0.0, 0.875, 1.0, 0.0, 8)`

If Γ≪0 on sink-in interventions, actuators push away from the intended sink even while 'failing' the exact edge — binary P(s→s') hides this.

$$\boxed{\Gamma\text{ measures control tendency; edge }P(s\to s')\text{ alone is too brittle}}$$
