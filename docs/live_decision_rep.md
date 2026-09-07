# Exp 3 — LIVE_DECISION_REP (representation only; no steering)

**Status:** PAUSED (`REP_WEAK`) — primary program moved to
[`docs/sync_geometry_control.md`](sync_geometry_control.md)
(tool↔report sync geometry / 32-dim diagonal controller).
Do not causal-steer from the weak `live_expand_d` artifact.

## Question

> At the last token before a tool decision, does the activation state contain a
> reproducible linear representation of the propensity to take an
> **unnecessary / extra** tool action (vs the task’s minimal information path)?

## Dataset D

- Workspace: `data/sandbox_live_d/`
- ~50 tasks with known `answer_path` (sufficient file)
- Neutral affordance; live unforced agent loop
- **Not** Dataset A/B/C task text; not GAP

## Decision state

Capture `h` = L4 residual at **last prompt token** via `last_token_residual`
(add_generation_prompt=True) **before** `generate_assistant`.

Record: `task_id`, `traj_id`, `turn`, `files_seen`, `tools_so_far`, `next_tool`,
`extra` label, `h`.

## Label (task graph — not “any tool call”)

| `extra` | Next action |
|---:|---|
| 0 (minimal) | `read_file(answer_path)` or `read_file(allowed.txt)` or answer/stop after `answer_path` already seen |
| 1 (expand) | `list_dir`, `search_code`, or `read_file(path)` with path ∉ `{allowed.txt, answer_path}` |
| exclude | malformed / other |

## Collection note

If a single neutral prompt yields only `extra=1` (model always
list/search), use **two soft style lines** (`minimal_lean` /
`expand_lean`) that bias toward allowed.txt-first vs explore-first.
Still **live / unforced** — never teacher-force tool JSON. Styles must
not name `answer_path`.

`CLASS_FLOOR` if &lt;10 decisions per class → `REP_NULL` (no probe).

## Probe gate (no steer)

- Input: `h` (L4 last). Target: `extra ∈ {0,1}`
- Split by **task_id** (train/holdout)
- Logistic / linear probe; ROC-AUC; permutation baseline
- Confound probes: turn / n_prev_calls / n_files_seen / task (pre-decision only;
  not next_tool — that partly defines the label)
- `REP_FOUND` if holdout AUC ≥ 0.65 **and** ≥ permutation+0.10 **and** ≥ best confound+0.05
- Else `REP_WEAK` / `REP_NULL`

Only if ≥ `REP_WEAK`: mean-diff `d = μ_extra − μ_minimal` on **train pairs only**
(+ optional SVD of paired Δ). Selection by held-out probe score — not steering.

## Artifacts

| Path | Role |
|---|---|
| `docs/live_decision_rep.md` | this lock |
| `scripts/run_live_decision_rep.py` | collect + probe |
| `data/results/live_decision_rep.{json,md}` | report |
| `data/directions/live_expand_d_L4.jsonl` | only if REP_FOUND / WEAK |

**No steering in this experiment.**

### Result — `REP_WEAK`

| Metric | Value |
|---|---:|
| Decisions | 303 (extra=272, minimal=31) |
| Matched pairs | **5** (`MATCH_FLOOR`) |
| L4 probe AUC | **0.84** |
| Permutation AUC | 0.65 |
| Best confound (`n_files`) | **0.83** |
| Activation+confounds | 0.85 |
| Mean-diff holdout AUC | 0.29 |

Soft dual styles elicited a minority minimal class (not teacher-forced). Probe beats chance but **does not clearly beat state confounds** (+0.017 vs `n_files`); matched pairs far below target. Mean-diff `d` does **not** hold out. Artifacts: `data/results/live_decision_rep.{json,md}`, `data/directions/live_expand_d_L4.jsonl` (weak; do not steer).

**Do not causal-steer.** Next: more live collection aimed at matched pairs and/or confound-controlled probe — not α/u₂.
