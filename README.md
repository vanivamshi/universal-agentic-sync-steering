# Tool ↔ report synchronization (Qwen3-0.6B + Cursor Agent)

When an agent **accesses private data via tools** but **hides that access in its
report**, tool-trace and stated behavior are out of sync. This repo studies that
failure mode and ships a **working correction layer** on Cursor Agent.

**Active claim (locked):**

> Detect ≠ repair at the activation layer. Geometry/probe **detects** async state;
> **policy repair** (gate or Agent follow-up) restores disclosure.

Model: **Qwen3-0.6B** (local). Product demo: **Cursor Agent + project hooks**
(not research CLI).

Builds on [Agentic-alignment-drift](https://github.com/vanivamshi/Agentic-alignment-drift).

---

## What we are doing

| Layer | What | Result |
|-------|------|--------|
| **Heim graph → layer** | Blowup-vs-ε curves pick intervention layer | L4 locked |
| **32-dim geometry @ L4** | Persona PCA basis + probe | `GEOM_HIT` — detect works |
| **Equation / probe** | Score \(s(h)=h\cdot v_\Delta\) | Separates hidden vs disclosed |
| **Activation steer** | \(+v_\Delta\), \(v_R\), closed-loop hooks | **NULL** — does not repair |
| **Policy gate / Agent hooks** | Detect → equation flag → regen / follow-up | **`GATE_LOCKED_HIT`**, **`AGENT_PATH_LIVE_HIT`** |

Earlier RQ1 privilege / persona-PC **steering shortlists** were closed negative
and removed. See [`docs/archived_pca_shortlist.md`](docs/archived_pca_shortlist.md).

Full protocol: [`docs/sync_geometry_control.md`](docs/sync_geometry_control.md).

---

## Equations (locked)

**Control coordinates** at report point (L4 residual projected onto frozen basis \(Q\)):

\[
z = Q^\top h, \quad Q = [\text{31 persona PCs @ L4},\; \text{Assistant Axis}]
\]

**Probe / detect score** (sync failure signal):

\[
s(h) = h \cdot v_\Delta
\]

where \(v_\Delta\) is the frozen detect direction (`data/directions/sync_v_delta_L4.jsonl`).

**Repair attempt (Qwen activation — closed NULL):**

\[
h' = h + \alpha\, v_R, \quad v_R = \mathrm{unit}\bigl(P_S(h_{\mathrm{hidden}} - h_{\mathrm{fault}})\bigr)
\]

Steering with \(v_\Delta\) or \(v_R\) does **not** restore disclosure under Case B.

**Working repair (Agent / gate — product layer):**

```text
detect(private_access)  →  equation_fired
if report omits path    →  repair_triggered  →  force disclosure
```

---

## Flowchart — research path (Heim → layer → equation)

![Heim classic graphs](docs/figures/step_5_heim_classic_graphs.png)

```text
  activations (Qwen3-0.6B)
           |
           v
  +---------------------------+
  | Heim blowup-vs-eps graph  |   docs/figures/step_5_heim_classic_graphs.png
  | real-base vs rand-base    |
  +---------------------------+
           |
           v
  +---------------------------+
  | CHOOSE LAYER              |   locked: L4
  +---------------------------+
           |
           v
  +---------------------------+
  | BASIS @ L4                |   persona_pca_prose_L4.jsonl (coords only)
  | 31 PC + Assistant Axis    |
  +---------------------------+
           |
           v
  +---------------------------+
  | EQUATION / PROBE          |
  |   s(h) = h · v_Δ          |
  +---------------------------+
           |
           +-----> GEOM_HIT / gate detect
           |
           +-----> activation steer = NULL
           |
           v
  +---------------------------+
  | WORKING CORRECTION        |
  | policy gate / Agent hooks |
  +---------------------------+
```

---

## Flowchart — product demo (Cursor Agent, not CLI)

Paste **Prompt A** from [`docs/sync_agent_demo_prompts.md`](docs/sync_agent_demo_prompts.md).

```text
  Prompt A in Cursor Agent
        |
        v
  Shell: api/run_check.py ----loads----> api/.env
        |
        v
  [SYNC EQUATION]              equation_fired = true
        |
        v
  First answer HIDES path      (Case B simulate)
        |
        v
  [SYNC EQUATION REPAIR]       repair_triggered = true
        |
        v
  Revised answer NAMES path    disclosed = true
        |
        v
  AGENT_PATH_LIVE_HIT
```

Proof: [`data/results/sync_agent_path_live.json`](data/results/sync_agent_path_live.json)

Regression: `.venv/bin/python scripts/test_sync_agent_hooks.py`

---

## Team-lead demo (5 min)

1. Clone repo; open as **Trusted** workspace in Cursor.
2. Settings → **Hooks** → confirm `.cursor/hooks.json` loaded.
3. Sandbox setup:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp data/sandbox_sync/api/env.example data/sandbox_sync/api/.env
   ```
4. New **Agent** chat → paste **Prompt A** from `docs/sync_agent_demo_prompts.md`.
5. Expect: `[SYNC EQUATION]` → hide answer → `[SYNC EQUATION REPAIR]` → revised answer names `api/.env`.

Do **not** use `scripts/run_sync_agent_prompt.py` for the product demo — that is
the Qwen research CLI path.

---

## Repo layout

| Path | Role |
|------|------|
| `.cursor/hooks/` | Agent-path detect → equation → repair hooks |
| `activation_pipeline/` | Model load, hooks, steering, GAP scoring |
| `scripts/sync_scenario.py` | Sync sandbox scenario + L4 basis load |
| `scripts/run_sync_*.py` | Geometry, gate, fault-recovery, closed-loop (research) |
| `data/sandbox_sync/` | Agent demo workspace (`api/run_check.py`) |
| `data/directions/` | \(v_\Delta\), L4 persona basis, repair dirs |
| `data/results/sync_*` | Locked experiment outcomes |
| `docs/sync_*` | Active protocols and flowcharts |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Default model: `qwen3-0.6b` (`activation_pipeline.device.LOCAL_MODEL_KEY`).

Smoke:

```bash
PYTHONPATH=. python scripts/smoke_activation_pipeline.py --print-plans
.venv/bin/python scripts/test_sync_agent_hooks.py
```

---

## Key results (sync program)

| Experiment | Decision |
|------------|----------|
| Exp A geometry | `GEOM_HIT` |
| Exp B diagonal controller | `CONTROLLER_FAIL` |
| Fault recovery / causal patch / closed-loop steer | `*_NULL` |
| Gate locked eval | `GATE_LOCKED_HIT` |
| Agent hooks (live) | `AGENT_PATH_LIVE_HIT` |

Artifacts: `data/results/sync_geometry.md`, `sync_gate_locked_eval.md`, `sync_agent_path_live.md`.

---

## Docs map

| Doc | Topic |
|-----|-------|
| [`docs/sync_geometry_control.md`](docs/sync_geometry_control.md) | Active sync program (locked) |
| [`docs/sync_agent_demo_prompts.md`](docs/sync_agent_demo_prompts.md) | Copy-paste Agent prompts |
| [`docs/sync_agent_flowchart.md`](docs/sync_agent_flowchart.md) | ASCII flowcharts |
| [`docs/sync_gate_locked_eval.md`](docs/sync_gate_locked_eval.md) | Policy gate repair |
| [`docs/sync_repair_causal.md`](docs/sync_repair_causal.md) | Detect ≠ repair framing |
| [`docs/rq1_privilege.md`](docs/rq1_privilege.md) | Heim graphs / RQ1 pilot (historical) |
| [`docs/archived_pca_shortlist.md`](docs/archived_pca_shortlist.md) | Removed failed PCA tracks |

---

## Design norms

- Pre-commit decision rules; do not retune bars after seeing numbers.
- **Detect ≠ repair** — do not conflate probe hit with causal recovery.
- Report clean negatives (`CLOSED_LOOP_NULL`, archived PCA shortlists).
- Product path = **Agent hooks** or **gate**; not residual steering.

---

## Citation / lineage

Activation plateaus (Heimersheim et al.), refusal / Assistant Axis directions
(Arditi, Zou, Lu et al.), Mind-the-GAP agentic scenarios. See `preregistration.md`.
