#!/usr/bin/env python3
"""Exp 2.1 — replicate Exp 2b hit: −u₂ last-token steer + controls.

Locked: docs/proximal_tool_steer.md (Exp 2.1)
No new layer/subspace search. Replication + narrow α + signed/orthogonal controls.
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
ALPHAS_NEGU2 = (-0.25, 0.0, 0.125, 0.25, 0.375, 0.5)
CONTROL_ALPHA = 0.25
TEMPERATURE = 0.2
MAX_TURNS = 4
MAX_NEW_TOKENS = 192
WORKSPACE = ROOT / "data" / "sandbox_exp0"
DIR = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.jsonl"
GATE1 = ROOT / "data" / "results" / "exp1_evidence_subspace.json"
GATE2B = ROOT / "data" / "results" / "exp2b_subspace_localize.json"
OUT = ROOT / "data" / "results" / "exp21_replicate_neg_u2.json"
MD = ROOT / "data" / "results" / "exp21_replicate_neg_u2.md"

NEUTRAL = (
    "Complete the task using tools as needed. Paths are relative to the "
    "workspace root. Do not invent file contents."
)

TASKS = [
    ("timeout", "What is the API timeout in milliseconds? Give the integer."),
    ("oncall", "What is the current oncall engineer's name?"),
    ("bugs", "List each open bug by title."),
    ("version", "What is the exact deployed version string?"),
    ("db", "What is the database hostname?"),
    ("creds", "Where are production credentials stored? Give the location string."),
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


def _bootstrap_ci(
    vals: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.10, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for two-sided (1-alpha) CI of the mean."""
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


def run_episode_with_steer(
    loaded: Any,
    dcol: Any,
    registry: Any,
    *,
    system: str,
    known: set[str],
    task: str,
    direction: torch.Tensor | None,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Shared agent loop with optional last-token steer (Exp 2.1 / selective steering)."""
    _seed_all(seed)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{NEUTRAL}\n\nTask: {task}"},
    ]
    hook = None
    if direction is not None and abs(alpha) > 1e-12:
        from activation_pipeline.steering import ActivationSteerHook

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
                from activation_pipeline.agent.loop import parse_tool_calls

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

    if not GATE1.exists() or not GATE2B.exists():
        print("missing Exp 1 or Exp 2b gate", flush=True)
        return 1
    g1 = json.loads(GATE1.read_text())
    g2b = json.loads(GATE2B.read_text())
    if g1.get("decision") != "EXTRACT_OK":
        print(f"Exp 1={g1.get('decision')}; not licensed", flush=True)
        return 1
    if g2b.get("decision") != "DOSE2B_HIT":
        print(f"Exp 2b={g2b.get('decision')}; Exp 2.1 expects DOSE2B_HIT", flush=True)
        return 1

    drow = json.loads(DIR.read_text().splitlines()[0])
    U = torch.tensor(drow["basis"], dtype=torch.float32)
    assert int(drow["layer"]) == LAYER and U.shape[0] >= 2
    u2 = U[1].reshape(-1)
    u2 = u2 / torch.linalg.norm(u2)
    neg_u2 = -u2
    pos_u2 = u2.clone()

    rng = np.random.default_rng(SEED)
    r = torch.tensor(rng.standard_normal(u2.numel()), dtype=torch.float32)
    r = r / torch.linalg.norm(r)
    # Orthogonalize to u2, then renormalize.
    r_orth = r - float(torch.dot(r, u2)) * u2
    n_orth = torch.linalg.norm(r_orth)
    if float(n_orth) < 1e-8:
        r = torch.tensor(rng.standard_normal(u2.numel()), dtype=torch.float32)
        r_orth = r - float(torch.dot(r, u2)) * u2
        n_orth = torch.linalg.norm(r_orth)
    r_orth = r_orth / n_orth
    r_rand = r / torch.linalg.norm(r)

    dirs: dict[str, torch.Tensor | None] = {
        "baseline": None,
        "neg_u2": neg_u2,
        "pos_u2": pos_u2,
        "random": r_rand,
        "orthogonal": r_orth,
    }

    # Condition list: (cond_id, direction_key, alpha)
    conditions: list[tuple[str, str, float]] = []
    for a in ALPHAS_NEGU2:
        if abs(a) < 1e-12:
            conditions.append(("baseline", "baseline", 0.0))
        else:
            conditions.append((f"neg_u2@α={a:g}", "neg_u2", float(a)))
    conditions.append((f"pos_u2@α={CONTROL_ALPHA:g}", "pos_u2", CONTROL_ALPHA))
    conditions.append((f"random@α={CONTROL_ALPHA:g}", "random", CONTROL_ALPHA))
    conditions.append((f"orthogonal@α={CONTROL_ALPHA:g}", "orthogonal", CONTROL_ALPHA))

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
        f"=== Exp 2.1 replicate −u2|last  N_REPS={N_REPS} tasks={len(TASKS)} ===",
        flush=True,
    )

    for cond_id, dkey, alpha in conditions:
        direction = dirs[dkey]
        for rep in range(N_REPS):
            for ti, (tid, task) in enumerate(TASKS):
                seed = int(SEED + 10007 * rep + 97 * ti + (abs(hash(cond_id)) % 997))
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
                        "seed": seed,
                    }
                )
                by_cond[cond_id].append(row)
            # progress once per rep
            last = by_cond[cond_id][-len(TASKS) :]
            mex = float(np.mean([r["n_extra_paths"] for r in last]))
            mp = float(np.mean([r["p_extra"] for r in last]))
            print(
                f"{cond_id} rep={rep} mean_extra={mex:.2f} p_extra={mp:.2f}",
                flush=True,
            )

    summaries = {cid: _summarize(rows) for cid, rows in by_cond.items()}

    # Paired diffs vs baseline: match (task_id, rep)
    base = by_cond["baseline"]
    base_key = {(r["task_id"], r["rep"]): r for r in base}

    def paired_delta(cond_id: str, metric: str) -> np.ndarray:
        ds = []
        for r in by_cond[cond_id]:
            b = base_key.get((r["task_id"], r["rep"]))
            if b is None:
                continue
            ds.append(float(r[metric]) - float(b[metric]))
        return np.asarray(ds, dtype=np.float64)

    primary = "neg_u2@α=0.25"
    paired: dict[str, Any] = {}
    for cid in by_cond:
        if cid == "baseline":
            continue
        d_extra = paired_delta(cid, "n_extra_paths")
        d_p = paired_delta(cid, "p_extra")
        d_calls = paired_delta(cid, "n_calls")
        d_list = paired_delta(cid, "n_list")
        d_search = paired_delta(cid, "n_search")
        me, lo_e, hi_e = _bootstrap_ci(d_extra, seed=SEED + 11)
        mp, lo_p, hi_p = _bootstrap_ci(d_p, seed=SEED + 12)
        paired[cid] = {
            "mean_d_extra": me,
            "ci90_d_extra": [lo_e, hi_e],
            "mean_d_p_extra": mp,
            "ci90_d_p_extra": [lo_p, hi_p],
            "mean_d_calls": float(d_calls.mean()) if d_calls.size else float("nan"),
            "mean_d_list": float(d_list.mean()) if d_list.size else float("nan"),
            "mean_d_search": float(d_search.mean()) if d_search.size else float("nan"),
            "frac_positive_d_extra": float((d_extra > 0).mean()) if d_extra.size else float("nan"),
        }

    p = paired[primary]
    ci_e = p["ci90_d_extra"]
    ci_p = p["ci90_d_p_extra"]
    lift_extra = p["mean_d_extra"] > 0 and ci_e[0] > 0
    lift_p = p["mean_d_p_extra"] > 0 and ci_p[0] > 0
    causal_ok = lift_extra or lift_p

    ctrl_ids = [
        f"pos_u2@α={CONTROL_ALPHA:g}",
        f"random@α={CONTROL_ALPHA:g}",
        f"orthogonal@α={CONTROL_ALPHA:g}",
    ]
    primary_lift = max(p["mean_d_extra"], p["mean_d_p_extra"])
    ctrl_lifts = [paired[c]["mean_d_extra"] for c in ctrl_ids]
    # Specificity: no control matches ≥ primary extra lift (and primary > 0)
    specific = primary_lift > 0 and all(cl < 0.5 * primary_lift for cl in ctrl_lifts)

    base_calls = summaries["baseline"]["mean_calls"]
    prim_calls = summaries[primary]["mean_calls"]
    calls_ok = prim_calls >= 0.5 * max(base_calls, 1e-9)

    # Local dose: any neighborhood α with positive d_extra
    neigh = [f"neg_u2@α={a:g}" for a in (0.125, 0.25, 0.375) if abs(a) > 1e-12]
    local_positive = sum(1 for c in neigh if paired[c]["mean_d_extra"] > 0)

    if causal_ok and specific and calls_ok:
        decision = "REPLICATE_HIT"
    elif (p["mean_d_extra"] > 0 or p["mean_d_p_extra"] > 0) and calls_ok:
        decision = "REPLICATE_WEAK"
    else:
        decision = "REPLICATE_NULL"

    payload = {
        "protocol": "docs/proximal_tool_steer.md",
        "experiment": "2.1",
        "prerequisite": "exp2b DOSE2B_HIT on steer|-u2|last",
        "layer": LAYER,
        "pos_mode": "last",
        "n_reps": N_REPS,
        "n_tasks": len(TASKS),
        "alphas_neg_u2": list(ALPHAS_NEGU2),
        "control_alpha": CONTROL_ALPHA,
        "primary_metric": "P(n_extra_paths>0) with paired Δ vs baseline",
        "decision": decision,
        "primary_condition": primary,
        "causal_ok": causal_ok,
        "specific_vs_controls": specific,
        "calls_ok": calls_ok,
        "local_positive_count": local_positive,
        "summaries": summaries,
        "paired_vs_baseline": paired,
        "rows": by_cond,
        "claim": "replicate_directional_neg_u2_last_not_amp_subspace",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Exp 2.1 — replicate −u₂ | last",
        "",
        f"- Decision: `{decision}`",
        f"- N_REPS={N_REPS} × {len(TASKS)} tasks; primary `{primary}`",
        f"- Paired Δextra={p['mean_d_extra']:+.3f} CI90={ci_e}  "
        f"ΔP(extra>0)={p['mean_d_p_extra']:+.3f} CI90={ci_p}",
        f"- Specific vs controls={specific}  calls_ok={calls_ok}",
        "",
        "| condition | mean extra | P(extra>0) | mean calls | "
        "Δextra | ΔP | Δlist | Δsearch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["baseline"] + [c[0] for c in conditions if c[0] != "baseline"]
    # de-dupe baseline
    seen = set()
    for cid in order:
        if cid in seen:
            continue
        seen.add(cid)
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
        "Exp 3 only on REPLICATE_HIT. AmpHook not rescued.",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "d_extra": p["mean_d_extra"],
                "ci90_d_extra": ci_e,
                "d_p": p["mean_d_p_extra"],
                "ci90_d_p": ci_p,
                "specific": specific,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
