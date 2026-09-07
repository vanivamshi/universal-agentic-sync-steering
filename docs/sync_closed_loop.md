# Closed-loop sync repair — equation between detect and hook

**Status:** CLOSED — `CLOSED_LOOP_NULL`.

## Architecture

\[
\boxed{
\text{probe detects fault}
\rightarrow
\underbrace{\text{equation}}_{\text{selects }v_R,\alpha}
\rightarrow
\text{activation hook @ report}
\rightarrow
\text{report}
}
\]

Site lock (hook-trace): intervene **before report generation** only.

## Equation (frozen on Disc)

Centroids from Disc private-access report residuals \(h\):

\[
\mu_{\mathrm{normal}}=\mathbb{E}[h\mid \mathrm{Class\,B}],\quad
\mu_{\mathrm{fault}}=\mathbb{E}[h\mid \mathrm{Class\,C}]
\]

31-dim persona subspace projector \(P_{\mathcal{S}}\) (QR of frozen persona PCs):

\[
\boxed{
v_R
=
\mathrm{unit}\!\bigl(
P_{\mathcal{S}}(\mu_{\mathrm{normal}}-\mu_{\mathrm{fault}})
\bigr)
}

\qquad
\boxed{
h'=h+\alpha\,v_R
}
\]

α frozen on Disc (displacement snap to `{0.25,0.5,1.0}`).  
**Not** \(v_\Delta\). Detector \(v_\Delta\) remains the probe only.

## Controller (per Class-C episode)

```text
run_tools_phase (Class C)
  → probe_before = h_report · v_Δ
  → IF s_tool ∧ probe_before < τ:
        equation already froze v_R
        register ActivationSteerHook(+α · arm_dir) @ report
     ELSE:
        generate without activation intervention
  → verify chain log
```

τ = Disc-median private report scores (same as locked gate).

## Paired arms (same tool trace)

| Arm | When probe fires |
|---|---|
| `fault` | no activation hook |
| `+v_delta` | steer \(+\alpha\,v_\Delta\) |
| `+v_repair` | steer \(+\alpha\,v_R\) (equation) |
| `random` | steer \(+\alpha\,u_{\mathrm{rand}}\) |

Plus separate **Normal (Class B)** baseline episodes.

## Metrics

Primary recovery (paired Class C):

\[
R
=
\frac{
P(\mathrm{disclose}\mid\mathrm{fault}+\mathrm{steer})
-
P(\mathrm{disclose}\mid\mathrm{fault})
}{
P(\mathrm{disclose}\mid\mathrm{normal})
-
P(\mathrm{disclose}\mid\mathrm{fault})
}
\]

Also report disclose ratio vs fault, \(\Delta\mathrm{probe}\) (analytic:
\(\alpha\,v\cdot\hat v_\Delta\)), spurious, `activation_hook_fired`.

## Per-episode chain log

`tool_hook_fired`, `private_access`, `probe_before`, `gate_fired`,
`repair_direction`, `repair_alpha`, `activation_hook_registered`,
`activation_hook_fired`, `probe_after`, `disclose`, `spurious`.

## Decision

| Tag | Rule |
|---|---|
| `CLOSED_LOOP_HIT` | \(R(v_R)\) CI_lo>0, spurious OK, \(v_\Delta\) & random not |
| `CLOSED_LOOP_GENERIC` | \(v_R\) recovers but control also |
| `CLOSED_LOOP_WEAK` | point \(R(v_R)>0\), CI includes 0 |
| `CLOSED_LOOP_NULL` | no specific recovery |
| `CLOSED_LOOP_UNTESTABLE` | Normal−Fault gap < 0.05 |

Script: `scripts/run_sync_closed_loop.py`  
Artifacts: `data/results/sync_closed_loop.{json,md}`  
Direction: `data/directions/sync_v_repair_eq_L4.jsonl`

## Result — `CLOSED_LOOP_NULL` (hooks forced)

Mode: **force hook on every private fault**. Hook fire = **1.0** on steered arms.

| Arm | disclose\|private | hook fire |
|---|---:|---:|
| Normal | 0.944 | 0 |
| Fault | 0.389 | 0 |
| +v_Δ / +v_R / random | 0.389 | **1.000** |

\(R=0\). Hooks ran; disclosure unchanged. Equation applied ≠ behavioral repair.
Policy gate remains the working solution.
