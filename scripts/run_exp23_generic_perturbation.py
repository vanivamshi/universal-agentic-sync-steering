#!/usr/bin/env python3
"""Exp 2.3 — generic last-token perturbation diagnostic (not a u₂ rescue).

Locked: docs/proximal_tool_steer.md (Exp 2.3)
Dataset C medium-headroom. α=0.25. Compare −u₂ Δ vs 5 random directions.
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
N_RANDOM = 5
ALPHA = 0.25
TEMPERATURE = 0.2
MAX_TURNS = 4
MAX_NEW_TOKENS = 192
WORKSPACE = ROOT / "data" / "sandbox_perturb_c"
DIR = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.jsonl"
GATE22 = ROOT / "data" / "results" / "exp22_transfer_dataset_b.json"
OUT = ROOT / "data" / "results" / "exp23_generic_perturbation.json"
MD = ROOT / "data" / "results" / "exp23_generic_perturbation.md"

NEUTRAL = (
    "Complete the task using tools as needed. Paths are relative to the "
    "workspace root. Do not invent file contents."
)

TASKS = [
    ("port", "What integer port does the metrics scraper listen on?"),
    ("tz", "What is the default_timezone string?"),
    ("hash", "Which password-hash algorithm is documented? One token."),
    ("queue", "What is the primary message queue name?"),
    ("build", "What is the default build target name?"),
    ("ttl", "What is the cache ttl_seconds integer?"),
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
    return {
        "n_calls": len(calls),
        "n_extra_paths": len(set(extra)),
        "p_extra": int(len(set(extra)) > 0),
        "n_list": n_list,
        "n_search": n_search,
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
    return (
        float(vals.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    extras = np.array([r["n_extra_paths"] for r in rows], dtype=np.float64)
    pextra = np.array([r["p_extra"] for r in rows], dtype=np.float64)
    calls = np.array([r["n_calls"] for r in rows], dtype=np.float64)
    m_e, lo_e, hi_e = _bootstrap_ci(extras, seed=SEED + 1)
    m_p, lo_p, hi_p = _bootstrap_ci(pextra, seed=SEED + 2)
    return {
        "n": len(rows),
        "mean_extra": m_e,
        "ci90_mean_extra": [lo_e, hi_e],
        "p_extra": m_p,
        "ci90_p_extra": [lo_p, hi_p],
        "mean_calls": float(calls.mean()),
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

    if not GATE22.exists():
        print("missing Exp 2.2 gate", flush=True)
        return 1
    g22 = json.loads(GATE22.read_text())
    if g22.get("decision") == "TRANSFER_HIT":
        print("Exp 2.2 was TRANSFER_HIT; Exp 2.3 diagnostic unexpected", flush=True)
        return 1
    if not WORKSPACE.exists():
        print(f"missing Dataset C {WORKSPACE}", flush=True)
        return 1

    drow = json.loads(DIR.read_text().splitlines()[0])
    U = torch.tensor(drow["basis"], dtype=torch.float32)
    assert int(drow["layer"]) == LAYER and U.shape[0] >= 2
    u2 = _unit(U[1])
    neg_u2 = _unit(-u2)
    pos_u2 = _unit(u2)

    rng = np.random.default_rng(SEED + 404)
    r0 = _unit(torch.tensor(rng.standard_normal(u2.numel()), dtype=torch.float32))
    r_orth = _unit(r0 - float(torch.dot(r0, u2)) * u2)
    randoms: list[torch.Tensor] = []
    for i in range(N_RANDOM):
        randoms.append(
            _unit(torch.tensor(rng.standard_normal(u2.numel()), dtype=torch.float32))
        )

    dirs: dict[str, torch.Tensor | None] = {
        "baseline": None,
        "neg_u2": neg_u2,
        "pos_u2": pos_u2,
        "orthogonal": r_orth,
    }
    for i, rv in enumerate(randoms, start=1):
        dirs[f"random{i}"] = rv

    conditions = [("baseline", "baseline", 0.0)]
    for key in ("neg_u2", "pos_u2", "orthogonal"):
        conditions.append((key, key, ALPHA))
    for i in range(1, N_RANDOM + 1):
        conditions.append((f"random{i}", f"random{i}", ALPHA))

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
        f"=== Exp 2.3 generic perturbation  α={ALPHA} N_RANDOM={N_RANDOM} ===",
        flush=True,
    )

    for cond_id, dkey, alpha in conditions:
        direction = dirs[dkey]
        for rep in range(N_REPS):
            for ti, (tid, task) in enumerate(TASKS):
                seed = int(SEED + 30011 * rep + 149 * ti + (abs(hash(cond_id)) % 983))
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
                        "dataset": "C",
                    }
                )
                by_cond[cond_id].append(row)
            last = by_cond[cond_id][-len(TASKS) :]
            mex = float(np.mean([r["n_extra_paths"] for r in last]))
            print(f"{cond_id} rep={rep} mean_extra={mex:.2f}", flush=True)

    summaries = {cid: _summarize(rows) for cid, rows in by_cond.items()}
    base_key = {(r["task_id"], r["rep"]): r for r in by_cond["baseline"]}

    def paired_delta(cond_id: str, metric: str) -> np.ndarray:
        ds = []
        for r in by_cond[cond_id]:
            b = base_key[(r["task_id"], r["rep"])]
            ds.append(float(r[metric]) - float(b[metric]))
        return np.asarray(ds, dtype=np.float64)

    paired: dict[str, Any] = {}
    for cid in by_cond:
        if cid == "baseline":
            continue
        d_extra = paired_delta(cid, "n_extra_paths")
        d_p = paired_delta(cid, "p_extra")
        me, lo_e, hi_e = _bootstrap_ci(d_extra, seed=SEED + 41)
        mp, lo_p, hi_p = _bootstrap_ci(d_p, seed=SEED + 42)
        paired[cid] = {
            "mean_d_extra": me,
            "ci90_d_extra": [lo_e, hi_e],
            "mean_d_p_extra": mp,
            "ci90_d_p_extra": [lo_p, hi_p],
            "mean_d_calls": float(paired_delta(cid, "n_calls").mean()),
        }

    base_extra = summaries["baseline"]["mean_extra"]
    if base_extra < 0.15 or base_extra > 0.85:
        decision = "PERTURB_FLOOR"
    else:
        rand_deltas = np.array(
            [paired[f"random{i}"]["mean_d_extra"] for i in range(1, N_RANDOM + 1)],
            dtype=np.float64,
        )
        neg_d = paired["neg_u2"]["mean_d_extra"]
        r_mean = float(rand_deltas.mean())
        r_std = float(rand_deltas.std(ddof=1)) if N_RANDOM > 1 else 0.0
        r_min = float(rand_deltas.min())
        r_max = float(rand_deltas.max())
        outlier = neg_d > r_max and (r_std < 1e-12 or neg_d > r_mean + 1.5 * r_std)
        if outlier:
            decision = "PERTURB_OUTLIER"
        else:
            decision = "PERTURB_GENERIC"

    rand_deltas = [
        paired[f"random{i}"]["mean_d_extra"] for i in range(1, N_RANDOM + 1)
    ]

    payload = {
        "protocol": "docs/proximal_tool_steer.md",
        "experiment": "2.3",
        "dataset": "C",
        "workspace": str(WORKSPACE),
        "note": "diagnostic_only_not_u2_rescue",
        "layer": LAYER,
        "pos_mode": "last",
        "alpha": ALPHA,
        "n_reps": N_REPS,
        "n_random": N_RANDOM,
        "decision": decision,
        "baseline_mean_extra": base_extra,
        "neg_u2_d_extra": paired["neg_u2"]["mean_d_extra"],
        "random_d_extra": rand_deltas,
        "random_d_extra_mean": float(np.mean(rand_deltas)),
        "random_d_extra_std": float(np.std(rand_deltas, ddof=1)),
        "random_d_extra_min": float(np.min(rand_deltas)),
        "random_d_extra_max": float(np.max(rand_deltas)),
        "summaries": summaries,
        "paired_vs_baseline": paired,
        "rows": by_cond,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Exp 2.3 — generic last-token perturbation (Dataset C)",
        "",
        f"- Decision: `{decision}`",
        f"- Baseline mean extra={base_extra:.2f}  −u₂ Δextra={paired['neg_u2']['mean_d_extra']:+.3f}",
        f"- Random Δextra mean={payload['random_d_extra_mean']:+.3f} "
        f"std={payload['random_d_extra_std']:.3f} "
        f"range=[{payload['random_d_extra_min']:+.3f}, {payload['random_d_extra_max']:+.3f}]",
        "",
        "| condition | mean extra | Δextra | ΔP | Δcalls |",
        "|---|---:|---:|---:|---:|",
    ]
    order = ["baseline", "neg_u2", "pos_u2", "orthogonal"] + [
        f"random{i}" for i in range(1, N_RANDOM + 1)
    ]
    for cid in order:
        s = summaries[cid]
        if cid == "baseline":
            lines.append(
                f"| `{cid}` | {s['mean_extra']:.2f} | — | — | — |"
            )
        else:
            p = paired[cid]
            lines.append(
                f"| `{cid}` | {s['mean_extra']:.2f} | {p['mean_d_extra']:+.3f} | "
                f"{p['mean_d_p_extra']:+.3f} | {p['mean_d_calls']:+.3f} |"
            )
    lines += [
        "",
        "Diagnostic only. Does not reopen u₂ Exp-2 or license Exp 3.",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "baseline_extra": base_extra,
                "neg_u2_d": paired["neg_u2"]["mean_d_extra"],
                "random_mean": payload["random_d_extra_mean"],
                "random_range": [
                    payload["random_d_extra_min"],
                    payload["random_d_extra_max"],
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
