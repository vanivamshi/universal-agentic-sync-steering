# Live decision-state representation discovery

**Status:** SUPERSEDED as primary by
[`docs/sync_geometry_control.md`](sync_geometry_control.md)
(tool↔report sync). Keep as background methodology for live decision capture.

**Prerequisite closed ladder** (do not reopen):

```text
FLOOR_CORPUS → proximal target → Exp 0 NULL → Exp 0b IDENTITY_HIT
→ Exp 1 EXTRACT_OK → Exp 2 DOSE_NULL → Exp 2b local −u₂ hit
→ Exp 2.1 REPLICATE_WEAK (nonspecific) → Exp 2.2 TRANSFER_NULL
→ Exp 2.3 PERTURB_GENERIC (−u₂ inside random Δ)
→ u₂ / old representation CLOSED; Exp 3 blocked
```

Artifacts frozen under `docs/proximal_tool_steer.md` and
`data/results/exp2{,b,1,2,3}_*`. Do **not** retune α/layer/u₂ or resample
Dataset B for that vector.

**Question (new):** can activations at a *live* tool-decision point predict and
(later) causally shift proximal exploratory tool expansion?

`surface_gap` remains **terminal** — out of scope until a robust proximal
representation and causal effect exist.

---

## What this is / is not

| Do | Do not |
|---|---|
| Collect **unforced** agent trajectories | Teacher-force tool JSON for primary discovery |
| Label at the **pre-action** residual | Use post-hoc full-turn pooling as the primary unit |
| Require **predictive** held-out probe first | Jump to SVD / mean-diff → steer |
| Separate discovery / freeze / causal sets | Discover and claim causality on the same trajectories |
| Match activation-displacement norms in causal tests | Ignore generic perturbation controls (Exp 2.1 lesson) |

Reuse existing: Hermes tools, agent loop, `ActivationSteerHook`, proximal
`extra` / exploratory metrics, Qwen3-0.6B, L4 as *default readout layer*
(layer may be re-selected only under the probe gate below — not by steering
search).

---

## Unit of analysis: matched live decision state

### Decision point `t`

For each assistant generation that emits a tool call (or ends the turn with an
answer / stop), the primary activation is:

> **last residual at layer L immediately before the first new token of that
> generation step** (prefill last token of the prompt that conditions the
> next action).

Record alongside `h_t`:

- `task_id`, question text
- `files_seen` (paths returned by prior tool results)
- `tools_called` so far (ordered)
- `turn_index`
- `next_action` label (below)
- sandbox / dataset split id

Do **not** teacher-force `next_action` when building the primary corpus.

### Labels (locked)

| Label | Definition |
|---|---|
| `minimal` | Next action is `read_file(allowed.txt)` **or** answer/stop with **no** new tool call, when `allowed.txt` was already readable / previously opened this episode |
| `expand` | Next action is `list_dir`, `search_code`, or `read_file(path)` with `path ∉ {allowed.txt, "."}` |
| `other` | Everything else (e.g. first-ever `read_file(allowed.txt)` when nothing read yet; malformed calls). **Excluded** from contrast pairs |

Primary construct: **propensity to expand the tool search** given the current
information state — not “constraint obedience,” not `surface_gap`.

### Matched pairs (required for mean-diff / subspace)

A pair `(minimal, expand)` is **matched** only if all hold:

1. Same `task_id` and same sandbox version
2. Same `files_seen` set (order ignored)
3. Same `turn_index` band: both in `{0}`, or both in `{1,2}`, or both in `{≥3}`
4. Same prior tool-name multiset (optional soft match; hard-require for v1)
5. Labels are `minimal` vs `expand` as above

Unmatched singles may enter the **probe** as iid examples but must **not**
enter the mean-diff / SVD discovery matrix.

---

## Datasets (strict separation)

| Split | Role | Rule |
|---|---|---|
| **Disc** | Collect live states; train probe; form candidate `d` / `U` | Task ids disjoint from Freeze/Causal |
| **Freeze** | Select layer / k / sign / α *if* any; freeze artifact | Task ids disjoint; no causal claim |
| **Causal** | Steer + controls only | Never used for discovery or α search |

Suggested sizes (Qwen3-0.6B / Mac): ≥60 Disc trajectories, ≥30 Freeze, ≥30
Causal; ≥20 matched Disc pairs before mean-diff. If matched pairs < 12 after
collection, stop with `MATCH_FLOOR` — do not invent pairs by teacher-forcing.

Workspaces: prefer new sandboxes (not `sandbox_exp0` alone). May reuse
`sandbox_perturb_c` / `sandbox_transfer_b` **only** as Causal *evaluation*
after Disc/Freeze used different task families.

---

## Phase A — Predictive validation (gate)

**Before any steering.**

1. Collect Disc (+ optional Freeze) live states at candidate layers
   `{2, 4, 6}` (default primary L4).
2. Binary probe: `h_t → P(expand)` vs `minimal` (logistic / linear).
3. **Held-out by `task_id`** (never by random timesteps of the same task).
4. Report AUROC / accuracy + 90% bootstrap CI.

| Gate | Rule |
|---|---|
| `PROBE_HIT` | held-out AUROC ≥ 0.65 **and** CI lower > 0.55 |
| `PROBE_WEAK` | AUROC ≥ 0.58 but CI touches ≤ 0.55 |
| `PROBE_NULL` | else |

**Stop if `PROBE_NULL`.** No direction, no steer. Document and redesign
matching/labels before retrying.

Layer choice: pick the layer with best Freeze AUROC under `PROBE_HIT` only;
default L4 if ties.

---

## Phase B — Representation freeze

Only if Phase A ≥ `PROBE_WEAK`.

On **Disc** matched pairs only:

```text
d₀ = mean(h_expand) − mean(h_minimal)   # unit-normalized
```

Optional: SVD on centered matched Δ rows → `U_k` with cumvar ≥ 0.80, k ≤ 4 —
but **primary causal candidate is `d₀`** unless Freeze probe shows subspace
coords beat `d₀` by ≥ 0.05 AUROC.

Save: `data/directions/live_expand_d_L{L}.jsonl` (+ optional `U`).

On **Freeze**: confirm probe still ≥ `PROBE_WEAK` using frozen `d₀` projection
(or subspace coords). If Freeze fails → `FREEZE_FAIL`, do not Causal-steer.

Sign: define `+d` as the expand-positive direction (higher probe score toward
expand). Causal tests use `±α d` with α chosen **only on Freeze** from a tiny
grid `{0.1, 0.25, 0.5}` maximizing Freeze probe margin — **not** Causal metrics.

---

## Phase C — Causal test (only after freeze)

Held-out **Causal** tasks. Neutral affordance. Same gen locks as Exp 2
(T=0.2, max_turns=4, max_new_tokens=192) unless preregistered otherwise.

Hook: `ActivationSteerHook`, `pos_mode=last`, frozen layer, **unit-norm**
directions, **matched ‖α d‖**.

| Condition | Purpose |
|---|---|
| baseline | reference |
| `+d` @ α* | candidate expand |
| `−d` @ α* | sign control |
| orthogonal-to-d | matched non-signal axis |
| random × 5 | generic perturbation distribution |

Primary: paired `Δextra` and `P(extra>0)` vs baseline; **outlier test** of `+d`
vs random Δ distribution (same rule as Exp 2.3 `PERTURB_OUTLIER`).

| Decision | Rule |
|---|---|
| `LIVE_STEER_HIT` | `+d` lifts extra/P with CI90 excluding ≤0 **and** is random-outlier **and** `−d` does not match the lift |
| `LIVE_STEER_GENERIC` | lift exists but inside random / orth band |
| `LIVE_STEER_NULL` | no reliable lift |

`LIVE_STEER_HIT` licenses later specificity / dose work. Generic or null →
stop this representation; do not α-sweep on Causal.

---

## Explicitly forbidden (post-u₂)

- Teacher-forced C0=`allowed.txt` vs C2=task path as the **primary** discovery
  contrast
- Steering before `PROBE_HIT` / `PROBE_WEAK`
- Choosing α or layer using Causal outcomes
- Claiming success from Dataset-A-only local ρ without Freeze + Causal
- Reintroducing `surface_gap` as primary endpoint
- Rescuing closed `u₂` / Exp 1 `U` as the live candidate

---

## Deliverables (when implementing)

| Artifact | Path |
|---|---|
| This spec | `docs/representation_discovery_live.md` |
| Collection script | `scripts/run_live_decision_collect.py` (not yet) |
| Probe / freeze | `scripts/run_live_decision_probe.py` (not yet) |
| Causal | `scripts/run_live_decision_steer.py` (not yet) |
| Directions | `data/directions/live_expand_d_L*.jsonl` |
| Results | `data/results/live_decision_*.{json,md}` |

**Implementation order:** collect → probe gate → freeze → causal. No code for
later phases until earlier gates pass.

---

## One-sentence summary

Discover whether live pre-action residuals encode expand-vs-minimal tool
choice under matched information states; only if a held-out probe works,
freeze a direction on Disc/Freeze and test causality on Causal with
random/orthogonal controls that close the generic-perturbation loophole.
