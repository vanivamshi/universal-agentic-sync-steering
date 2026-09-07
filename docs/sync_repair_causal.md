# Sync repair causal — detect ≠ repair

**Status:** CLOSED — `PATCH_NULL`.

## Claim under test

\[
\boxed{v_\Delta\ \text{(detect)}\ \neq\ v_{\mathrm{repair}}\ \text{(restore)}}
\]

Do **not** retune α on \(v_\Delta\). Do **not** iterate selective-steering equations.

## Result (locked)

Stage A activation patching at report last-token **failed at every tested layer**
(Disc disclose 0.333 → 0.000). Stage C on Freeze was **untestable** (fault already
disclosed). See `data/results/sync_repair_causal.{json,md}`.

Interpretation: Class-C omission prior is not repaired by transplanting Normal
report residuals; geometry remains a **monitor**, gate remains the **repair**.

## Stage A — Locate (activation patching)

Paired Normal (Class B) / Fault (Class C) on matched tasks. Cache multi-layer
last-token residuals at **report** (and tool-decision for Δ diagnostic).

For each layer \(L\) in a pre-registered grid, on Fault finalize:

\[
h_L^{\mathrm{fault}} \leftarrow h_L^{\mathrm{normal}}
\]

(full last-token residual replace via `ActivationResidualPatchHook`).

| Result | Tag implication |
|---|---|
| Some \(L\) restores disclose\|private toward Normal | causal site exists → Stage B at \(L^*\) |
| No \(L\) restores | `PATCH_NULL` — fault may be outside residual pathway / prompt-prior only |

\(L^*\) = argmax paired disclose recovery on **Disc** tasks; confirm on Freeze.

Also report \(\|\Delta_L\|_2 = \|h_L^B - h_L^C\|_2\) and earliest layer with
separability AUC ≥ 0.65 (diagnostic, not the causal lock).

## Stage B — Construct (Disc only)

At \(L^*\):

\[
\hat v_{\mathrm{repair}}
=
\mathrm{unit}\!\left(
\mathbb{E}[h_{L^*}^{\mathrm{normal}} - h_{L^*}^{\mathrm{fault}}]
\right)
\]

\[
\hat v_{\mathrm{detect}}
=
\mathrm{unit}\!\left(
\mathbb{E}[h_{L^*}^{\mathrm{disclosed}} - h_{L^*}^{\mathrm{hidden}}]
\right)
\]

(private-access Disc episodes; detect uses realized \(S_{\mathrm{output}}\)).

Orthogonalize:

\[
v_\parallel = (v_R\cdot v_D)\,v_D,\quad
v_\perp = v_R - v_\parallel
\]

Freeze directions before Freeze/val eval. Save under `data/directions/`.

## Stage C — Steer (Freeze/val, paired Class-C traces)

Arms (report-point @ \(L^*\), last-token): `fault`, `+v_detect`, `+v_repair`,
`+v_perp`, `+v_parallel`, `random`.

Primary α: displacement-matched on Disc
(\(\alpha \approx 0.25 \cdot \mathrm{median}\|h^B-h^C\|\) snapped to
`{0.25,0.5,1.0}`). Secondary: α grid on Freeze labeled exploratory.

Primary metric: recovery \(R\) (same as `docs/sync_fault_recovery.md`).

| Tag | Rule |
|---|---|
| `REPAIR_STEER_HIT` | \(R(v_R)\) CI_lo > 0, spurious OK, \(R(v_D)\) and random not |
| `REPAIR_STEER_GENERIC` | \(v_R\) recovers but random/`v_D` also |
| `REPAIR_STEER_WEAK` | point \(R(v_R)>0\), CI includes 0 |
| `REPAIR_STEER_NULL` | no specific repair |
| `PATCH_NULL` | Stage A found no restoring layer |

## Explicitly forbidden

- Another \(v_\Delta\) α sweep claiming repair
- Fitting \(v_{\mathrm{repair}}\) on Freeze/val
- Selective-steering v5 before this closed
- Re-running Stage A report last-token patch grid without a **new** site
  (e.g. tool-phase multi-token / attention) or mechanism

Script: `scripts/run_sync_repair_causal.py`  
Artifacts: `data/results/sync_repair_causal.{json,md}`
