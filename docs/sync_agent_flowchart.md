# Full pipeline flowchart (ASCII)

Two tracks — do not conflate. Three layers — see `docs/sync_universal_equation.md`.

---

## Layer A — Scorer (8-cell benchmark)

```text
  neutral task prompt  (no channel instructions)
           |
           v
  observe S = (S_plan, S_hook, S_out)
           |
           v
  m* = benchmark cell (armed for scoring ONLY)
           |
           v
  e = m* − S     E_sync = ||e||_1
```

`m*` is **target configuration**, not “tell the model what to say.”

---

## Layer B — Intervention (research question)

```text
  run 1 (no intervention)  →  S_before, e_before
           |
           v
  apply repair from e_before  (activation @ L4 and/or policy gate)
           |
           v
  run 2 (regen)            →  S_after, e_after
           |
           v
  ΔE_sync = ||e_before||_1 − ||e_after||_1
```

**Primary metric:** does intervention **reduce** sync error?  
Not: arm m* → force e=0 via adapt text.

---

## Track 1 — Qwen + L4 activation

```text
  activations (Qwen3-0.6B)
           |
           v
  Heim graph → layer L4 → basis + v_Δ probe
           |
           +---- detect / geometry (GEOM_HIT)
           |
           +---- activation steer as REPAIR = NULL (so far)
           |
           v
  policy gate @ report  =  repair that works (disclose path)
```

Scripts: `run_sync_product_demo.py`, `run_sync_fault_recovery.py`

---

## Track 2 — Cursor Agent (the demo)

```text
  New Agent chat — paste arm + neutral task
        |
        v
  Agent Shell: run_check.py ----loads----> api/.env
        |
        v
  [SYNC EQUATION]          (visible in chat)
        |
        v
  Agent PLAN + FINAL
        |
        v
  e ≠ 0  →  [SYNC EQUATION REPAIR] or Adapt …  (visible in chat)
        |
        v
  Agent revises  →  synced for benchmark cell m*
```

Prompts: `docs/sync_agent_demo_prompts.md`  
Proof = Agent messages, not CLI scripts.

---

## One-line lock

```text
Goal                     =  universal policy (auth ↔ execute ↔ report), not S=m* scripting
m*                       =  policy encoding for scoring (not a prompt)
Free model behavior      =  required; we do not write answers for the model
8-cell visit-all         =  not required; cells are diagnostic encodings
E_sync / violation rate  =  measure deviation from policy
Text repair until e=0    =  weak demo only
Activation / trained     =  path to alignment claim (Track 1)
Execution ground truth   =  hooks/tools
plan channel             =  reported PLAN (CoT proxy), not internal CoT
```

Protocol: `docs/sync_universal_equation.md`  
Audit: `docs/sync_implementation_audit.md`
