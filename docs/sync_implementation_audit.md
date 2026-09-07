# Sync implementation audit

Honest status under the **universal alignment** framing  
(`docs/sync_universal_equation.md`).

---

## Research claims

| Claim | Question | Valid evidence |
|-------|----------|----------------|
| **1. Detection** | Can hooks identify private-data access? | Real tool evidence → `sync_secret_detect.py` |
| **2. Measurement** | Can free-run PLAN / hook / FINAL be scored vs a policy encoding `m*`? | Raw `S` from live Agent / Qwen; never fill_unknown |
| **3. Text / policy repair** | Does an explicit repair instruction reduce `E_sync`? | Live free revise after stop followup — **weak** (instruction-following) |
| **4. Universal alignment** | Does an intervention make the **model** implement the policy on free re-runs, and generalize? | Track 1 activation / trained policy; held-out tasks; **no** answer substitution |

A reviewer must not be able to say we rewrote answers until `e=0`.

---

## What is fixed vs free

| Fixed (OK) | Free (required) |
|------------|-----------------|
| Policy / `m*` as **scoring encoding** | Tool choice, PLAN, FINAL |
| Hook / scorer definitions | Whether access happens under authorization |
| Task + authorization wording | Post-intervention behavior on re-run |

| Fixed (NOT OK for alignment evidence) | Where |
|---------------------------------------|--------|
| Canned `AGENT_*` / `NATURAL` answers | `agent_session_write_live_results.py`, `record_sync_agent_live_results.py` |
| `_repair()` / `_apply_adapt()` closing `e` | same / `run_sync_hook_regression.py` |
| Teaching `api⇒disclose`, `docs⇒hide` | folder-as-label shortcuts |

---

## Deprecated as alignment evidence

| Artifact | Why |
|----------|-----|
| `sync_agent_all_masks_run.*` | Replay + `_apply_adapt` |
| `sync_agent_live_all_masks.*` (current packaged run) | Canned answers + scripted repair to hit `e=0` |
| `test_sync_agent_simple_all.py` | Fabricates PLAN/FINAL matching `m*` |

**Still valid for plumbing:** `run_sync_hook_regression.py` → `HOOK_REGRESSION_PASS` (scorer/hooks only).

---

## Target experiment (implemented entry points)

```bash
.venv/bin/python scripts/run_sync_policy_free_run.py --init
.venv/bin/python scripts/run_sync_policy_free_run.py --summarize
.venv/bin/python scripts/test_sync_policy.py
.venv/bin/python scripts/run_sync_intervention_grid.py --mode qwen
```

| Module | Role |
|--------|------|
| `scripts/sync_policy.py` | Policy, situations, violations, `desired_m_star` |
| `scripts/run_sync_policy_free_run.py` | Free-run init/arm/summarize (`measure_only`) |
| Stop hook | Skips text repair when `policy_mode=measure_only` |

Cursor Agent path: Layer A measurement (+ optional weak Layer C if measure_only off).  
Causal / universal claim: Track 1 Qwen activation.
---

## Scorer notes (Layer A)

- `S_hook` from per-turn `tool_hook_this_turn`
- Store raw `S` only
- `plan` = reported PLAN / CoT **proxy**, not internal CoT
- Zero/few-shot prompts may elicit PLAN format only — must not encode `m*` or hide/disclose targets

---

## File map

| Role | Path |
|------|------|
| Protocol (this framing) | `docs/sync_universal_equation.md` |
| Scorer | `scripts/sync_eq.py` |
| Arm scoring cell | `scripts/arm_sync_eq.py` |
| Detect / score / stop | `.cursor/hooks/sync_secret_*.py` |
| Hook regression only | `scripts/run_sync_hook_regression.py` |
| Live log | `scripts/sync_live_log.py` / `run_sync_agent_live_record.py` |
| Track 1 intervention | `scripts/run_sync_intervention_grid.py`, `run_sync_product_demo.py` |
