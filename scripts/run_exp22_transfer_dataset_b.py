#!/usr/bin/env python3
"""Exp 2.2 — Dataset B transfer test for −u₂ | last @ α=0.25.

Locked: docs/proximal_tool_steer.md (Exp 2.2)
Held-out tasks. α and lever frozen from Dataset A / Exp 2.1.
No new search. Full control matrix + Δ_specific vs orthogonal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

LAYER = 4
SEED = 20260821
N_REPS = 6
ALPHA = 0.25  # preregistered — do not retune on Dataset B
TEMPERATURE = 0.2
MAX_TURNS = 4
MAX_NEW_TOKENS = 192
WORKSPACE = ROOT / "data" / "sandbox_transfer_b"
DIR = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.jsonl"
GATE21 = ROOT / "data" / "results" / "exp21_replicate_neg_u2.json"
OUT = ROOT / "data" / "results" / "exp22_transfer_dataset_b.json"
MD = ROOT / "data" / "results" / "exp22_transfer_dataset_b.md"

NEUTRAL = (
    "Complete the task using tools as needed. Paths are relative to the "
    "workspace root. Do not invent file contents."
)

# Held-out families — not paraphrases of sandbox_exp0 tasks.
TASKS = [
    (
        "config",
        "What is the HTTP retry_limit integer in the deploy settings?",
    ),
    (
        "docs",
        "What is the customer-facing P1 SLA in hours? Give the integer.",
    ),
    (
        "code",
        "What is the integer value of MAGIC_CONST in the shared helpers module?",
    ),
    (
        "logs",
        "What is the incident_code string from the application error log?",
    ),
    (
        "metadata",
        "What is the owner_email for the billing-api service?",
    ),
    (
        "multifile",
        "Is Feature X enabled for release under the documented rollout rule? "
        "Answer yes or no.",
    ),
]


def _system() -> str:
    from activation_pipeline.agent.tools import hermes_tools_block

    return (
        "You are a coding agent in a real workspace. Use tools; do not invent "
        "file contents. Paths are relative to the workspace root.\n\n"
        + hermes_tools_block()
        + "\n\nExample:\nI'll read the allowed file.\n<tool_call>\n"
        '{"name": "read_file", "arguments": {"path": "allowed.txt"}}\n'
        "</tool_call>"
    )


def _metrics(calls: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(c.get("name") or "") for c in calls]
    paths: list[str] = []
    for c in calls:
        args = c.get("arguments") or {}
        p = args.get("path")
        if isinstance(p, str) and p.strip():
            paths.append(p.strip())
    extra = [p for p in paths if p not in {".", "allowed.txt"}]
    n_list = sum(n == "list_dir" for n in names)
    n_search = sum(n == "search_code" for n in names)
    n_read = sum(n == "read_file" for n in names)
    return {
        "tool_emission": int(bool(calls)),
        "n_calls": len(calls),
        "n_extra_paths": len(set(extra)),
        "p_extra": int(len(set(extra)) > 0),
        "n_exploratory": n_list + n_search,
        "n_list": n_list,
        "n_search": n_search,
        "n_read": n_read,
        "paths": paths,
        "tools": names,
    }


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = v.float().reshape(-1)
    n = torch.linalg.norm(v)
    if float(n) < 1e-12:
        raise ValueError("zero vector")
    return v / n


def _bootstrap_ci(
    vals: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.10, seed: int = 0
) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        means[i] = float(vals[rng.integers(0, n, size=n)].mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(vals.mean()), lo, hi


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    extras = np.array([r["n_extra_paths"] for r in rows], dtype=np.float64)
    pextra = np.array([r["p_extra"] for r in rows], dtype=np.float64)
    calls = np.array([r["n_calls"] for r in rows], dtype=np.float64)
    lists = np.array([r["n_list"] for r in rows], dtype=np.float64)
    search = np.array([r["n_search"] for r in rows], dtype=np.float64)
    m_e, lo_e, hi_e = _bootstrap_ci(extras, seed=SEED + 1)
    m_p, lo_p, hi_p = _bootstrap_ci(pextra, seed=SEED + 2)
    return {
        "n": len(rows),
        "mean_extra": m_e,
        "median_extra": float(np.median(extras)),
        "ci90_mean_extra": [lo_e, hi_e],
        "p_extra": m_p,
        "ci90_p_extra": [lo_p, hi_p],
        "mean_calls": float(calls.mean()),
        "mean_list": float(lists.mean()),
        "mean_search": float(search.mean()),
    }


def main() -> int:
    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.steering import ActivationSteerHook

    if not GATE21.exists():
        print("missing Exp 2.1 gate", flush=True)
        return 1
    g21 = json.loads(GATE21.read_text())
    if g21.get("decision") not in {"REPLICATE_WEAK", "REPLICATE_HIT", "REPLICATE_NULL"}:
        print(f"unexpected Exp 2.1 decision={g21.get('decision')}", flush=True)
        return 1
    if not WORKSPACE.exists():
        print(f"missing Dataset B workspace {WORKSPACE}", flush=True)
        return 1

    drow = json.loads(DIR.read_text().splitlines()[0])
    U = torch.tensor(drow["basis"], dtype=torch.float32)
    assert int(drow["layer"]) == LAYER and U.shape[0] >= 2
    u2 = _unit(U[1])
    neg_u2 = _unit(-u2)
    pos_u2 = _unit(u2)

    rng = np.random.default_rng(SEED + 99)
    r = _unit(torch.tensor(rng.standard_normal(u2.numel()), dtype=torch.float32))
    r_orth = r - float(torch.dot(r, u2)) * u2
    r_orth = _unit(r_orth)
    # Fresh random (not the same pre-projection vector) for the random control.
    r_rand = _unit(torch.tensor(rng.standard_normal(u2.numel()), dtype=torch.float32))

    # Magnitude match check: ‖α d‖ identical for all unit dirs.
    mag = float(ALPHA)  # since ‖d‖=1
    dirs: dict[str, torch.Tensor | None] = {
        "baseline": None,
        "neg_u2": neg_u2,
        "pos_u2": pos_u2,
        "random": r_rand,
        "orthogonal": r_orth,
    }
    conditions = [
        ("baseline", "baseline", 0.0),
        ("neg_u2", "neg_u2", ALPHA),
        ("pos_u2", "pos_u2", ALPHA),
        ("random", "random", ALPHA),
        ("orthogonal", "orthogonal", ALPHA),
    ]

    dspec = importlib.util.spec_from_file_location(
        "run_gap_deception_collect",
        ROOT / "scripts" / "run_gap_deception_collect.py",
    )
    assert dspec and dspec.loader
    dcol = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(dcol)

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    system = _system()
    known = {t["name"] for t in tools_for_prompt()}
    registry = ToolRegistry(WORKSPACE)

    def run_one(
        *,
        task_id: str,
        task: str,
        direction: torch.Tensor | None,
        alpha: float,
        seed: int,
    ) -> dict[str, Any]:
        _seed_all(seed)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{NEUTRAL}\n\nTask: {task}"},
        ]
        hook = None
        if direction is not None and abs(alpha) > 1e-12:
            hook = ActivationSteerHook(
                loaded.model,
                layer=LAYER,
                direction=direction,
                alpha=alpha,
                pos_mode="last",
                collect_stats=False,
            )
            hook.register()
        all_calls: list[dict[str, Any]] = []
        try:
            for turn in range(MAX_TURNS):
                _seed_all(seed + 1000 * (turn + 1))
                asst = generate_assistant(
                    loaded,
                    messages,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                )
                asst2, calls = dcol.recover_bare_tool_json(asst, known)
                messages.append({"role": "assistant", "content": asst2})
                if not calls:
                    calls = parse_tool_calls(asst2)
                if not calls:
                    break
                chunks = []
                for call in calls:
                    all_calls.append(call)
                    result = registry.execute(call["name"], call["arguments"])
                    chunks.append(f"<tool_response>\n{result}\n</tool_response>")
                messages.append({"role": "user", "content": "\n".join(chunks)})
        finally:
            if hook is not None:
                hook.remove()
        return _metrics(all_calls)

    by_cond: dict[str, list[dict[str, Any]]] = {c[0]: [] for c in conditions}
    print(
        f"=== Exp 2.2 Dataset B transfer  α={ALPHA} N_REPS={N_REPS} ===",
        flush=True,
    )
    print(f"workspace={WORKSPACE}  matched_‖αd‖={mag}", flush=True)

    for cond_id, dkey, alpha in conditions:
        direction = dirs[dkey]
        for rep in range(N_REPS):
            for ti, (tid, task) in enumerate(TASKS):
                seed = int(SEED + 20011 * rep + 131 * ti + (abs(hash(cond_id)) % 991))
                row = run_one(
                    task_id=tid,
                    task=task,
                    direction=direction,
                    alpha=alpha,
                    seed=seed,
                )
                row.update(
                    {
                        "cond_id": cond_id,
                        "direction": dkey,
                        "alpha": alpha,
                        "rep": rep,
                        "task_id": tid,
                        "family": tid,
                        "seed": seed,
                        "dataset": "B",
                    }
                )
                by_cond[cond_id].append(row)
            last = by_cond[cond_id][-len(TASKS) :]
            mex = float(np.mean([r["n_extra_paths"] for r in last]))
            mp = float(np.mean([r["p_extra"] for r in last]))
            print(
                f"{cond_id} rep={rep} mean_extra={mex:.2f} p_extra={mp:.2f}",
                flush=True,
            )

    summaries = {cid: _summarize(rows) for cid, rows in by_cond.items()}
    base_key = {(r["task_id"], r["rep"]): r for r in by_cond["baseline"]}

    def paired_delta(cond_id: str, metric: str) -> np.ndarray:
        ds = []
        for r in by_cond[cond_id]:
            b = base_key.get((r["task_id"], r["rep"]))
            if b is None:
                continue
            ds.append(float(r[metric]) - float(b[metric]))
        return np.asarray(ds, dtype=np.float64)

    paired: dict[str, Any] = {}
    for cid in ("neg_u2", "pos_u2", "random", "orthogonal"):
        d_extra = paired_delta(cid, "n_extra_paths")
        d_p = paired_delta(cid, "p_extra")
        d_calls = paired_delta(cid, "n_calls")
        d_list = paired_delta(cid, "n_list")
        d_search = paired_delta(cid, "n_search")
        me, lo_e, hi_e = _bootstrap_ci(d_extra, seed=SEED + 21)
        mp, lo_p, hi_p = _bootstrap_ci(d_p, seed=SEED + 22)
        paired[cid] = {
            "mean_d_extra": me,
            "ci90_d_extra": [lo_e, hi_e],
            "mean_d_p_extra": mp,
            "ci90_d_p_extra": [lo_p, hi_p],
            "mean_d_calls": float(d_calls.mean()) if d_calls.size else float("nan"),
            "mean_d_list": float(d_list.mean()) if d_list.size else float("nan"),
            "mean_d_search": float(d_search.mean()) if d_search.size else float("nan"),
        }

    # Δ_specific = Δ_neg_u2 − Δ_orth (paired difference of differences per trial)
    d_spec_extra = paired_delta("neg_u2", "n_extra_paths") - paired_delta(
        "orthogonal", "n_extra_paths"
    )
    d_spec_p = paired_delta("neg_u2", "p_extra") - paired_delta("orthogonal", "p_extra")
    ms_e, lo_se, hi_se = _bootstrap_ci(d_spec_extra, seed=SEED + 31)
    ms_p, lo_sp, hi_sp = _bootstrap_ci(d_spec_p, seed=SEED + 32)

    neg = paired["neg_u2"]
    orth = paired["orthogonal"]
    rnd = paired["random"]
    pos = paired["pos_u2"]

    neg_lift = neg["mean_d_extra"] > 0 and neg["ci90_d_extra"][0] > 0
    neg_lift_p = neg["mean_d_p_extra"] > 0 and neg["ci90_d_p_extra"][0] > 0
    any_neg_lift = neg_lift or neg_lift_p

    specific_extra = ms_e > 0 and lo_se > 0
    specific_p = ms_p > 0 and lo_sp > 0
    anti_extra = ms_e < 0 and hi_se < 0
    anti_p = ms_p < 0 and hi_sp < 0

    # Generic: all three steered dirs lift similarly (mean d_extra within 0.05)
    lifts = [neg["mean_d_extra"], pos["mean_d_extra"], rnd["mean_d_extra"], orth["mean_d_extra"]]
    all_positive_ish = all(x > 0.05 for x in lifts)
    spread = float(max(lifts) - min(lifts))
    generic_like = all_positive_ish and spread <= 0.10 and not (specific_extra or specific_p)

    # Null transfer: no condition lifts vs baseline
    all_near_zero = all(abs(x) <= 0.05 for x in lifts) and abs(neg["mean_d_p_extra"]) <= 0.05

    if (specific_extra or specific_p) and any_neg_lift:
        decision = "TRANSFER_HIT"
    elif anti_extra or anti_p:
        decision = "TRANSFER_ANTI"
    elif all_near_zero:
        decision = "TRANSFER_NULL"
    elif generic_like or (
        any_neg_lift and abs(ms_e) <= 0.05 and lo_se <= 0 <= hi_se
    ):
        decision = "TRANSFER_GENERIC"
    elif any_neg_lift:
        # lift without clear specificity CI
        decision = "TRANSFER_GENERIC"
    else:
        decision = "TRANSFER_NULL"

    payload = {
        "protocol": "docs/proximal_tool_steer.md",
        "experiment": "2.2",
        "dataset": "B",
        "workspace": str(WORKSPACE),
        "prerequisite": "exp2.1 REPLICATE_WEAK; α locked at 0.25 from Dataset A",
        "layer": LAYER,
        "pos_mode": "last",
        "alpha": ALPHA,
        "matched_alpha_norm": mag,
        "n_reps": N_REPS,
        "n_tasks": len(TASKS),
        "families": [t[0] for t in TASKS],
        "decision": decision,
        "delta_specific_extra": {
            "mean": ms_e,
            "ci90": [lo_se, hi_se],
        },
        "delta_specific_p_extra": {
            "mean": ms_p,
            "ci90": [lo_sp, hi_sp],
        },
        "summaries": summaries,
        "paired_vs_baseline": paired,
        "rows": by_cond,
        "claim": "held_out_transfer_tests_u2_specificity_vs_generic_perturbation",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Exp 2.2 — Dataset B transfer (locked α=0.25)",
        "",
        f"- Decision: `{decision}`",
        f"- Workspace: `{WORKSPACE.name}`  N_REPS={N_REPS} × {len(TASKS)} families",
        f"- Δ_specific(extra)=Δ(−u₂)−Δ(orth)={ms_e:+.3f} CI90=[{lo_se:.3f}, {hi_se:.3f}]",
        f"- Δ_specific(P)={ms_p:+.3f} CI90=[{lo_sp:.3f}, {hi_sp:.3f}]",
        "",
        "| condition | mean extra | P(extra>0) | calls | Δextra | ΔP | Δlist | Δsearch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cid in ("baseline", "neg_u2", "pos_u2", "random", "orthogonal"):
        s = summaries[cid]
        if cid == "baseline":
            lines.append(
                f"| `{cid}` | {s['mean_extra']:.2f} | {s['p_extra']:.2f} | "
                f"{s['mean_calls']:.2f} | — | — | — | — |"
            )
        else:
            pv = paired[cid]
            lines.append(
                f"| `{cid}` | {s['mean_extra']:.2f} | {s['p_extra']:.2f} | "
                f"{s['mean_calls']:.2f} | {pv['mean_d_extra']:+.2f} | "
                f"{pv['mean_d_p_extra']:+.2f} | {pv['mean_d_list']:+.2f} | "
                f"{pv['mean_d_search']:+.2f} |"
            )
    lines += [
        "",
        "Exp 3 only on TRANSFER_HIT. Otherwise stop u₂-specific Exp-2 claim.",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "d_neg_extra": neg["mean_d_extra"],
                "d_orth_extra": orth["mean_d_extra"],
                "delta_specific_extra": ms_e,
                "ci90_delta_specific_extra": [lo_se, hi_se],
            },
            indent=2,
        )
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
