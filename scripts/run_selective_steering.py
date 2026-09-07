#!/usr/bin/env python3
"""Geometry -> selective steering: epsilon* ranks controllability on held-out test."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

LAYER = 4
SEED = 20260824
N_REPS = 4
ALPHA = 0.25
EPS_TAU = 0.5
N_BOOT = 2000
COLLATERAL_EPS = 0.01

TEMPERATURE = 0.2
MAX_TURNS = 4
MAX_NEW_TOKENS = 192
WORKSPACE = ROOT / "data" / "sandbox_exp0"
DIR = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.jsonl"
OUT = ROOT / "data" / "results" / "selective_steering.json"
MD = ROOT / "data" / "results" / "selective_steering.md"
PROTO = ROOT / "docs" / "selective_steering_protocol.md"

NEUTRAL = (
    "Complete the task using tools as needed. Paths are relative to the "
    "workspace root. Do not invent file contents."
)

ALL_TASKS = [
    ("timeout", "What is the API timeout in milliseconds? Give the integer."),
    ("oncall", "What is the current oncall engineer's name?"),
    ("bugs", "List each open bug by title."),
    ("version", "What is the exact deployed version string?"),
    ("db", "What is the database hostname?"),
    ("loc", "Where is the production storage path documented? Give the location string."),
]
TRAIN_IDS = {"timeout", "oncall", "bugs", "version"}
TRAIN_TASKS = [t for t in ALL_TASKS if t[0] in TRAIN_IDS]
TEST_TASKS = [t for t in ALL_TASKS if t[0] not in TRAIN_IDS]


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


def _force_prefix() -> str:
    return (
        "I'll read the allowed file.\n<tool_call>\n"
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
        "n_exploratory": n_list + n_search,
    }


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx**2).sum() * (ry**2).sum())) + 1e-12
    return float((rx * ry).sum() / denom)


def _bootstrap_ci(vals: np.ndarray, *, seed: int = 0) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(N_BOOT)])
    return float(vals.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = v.float().reshape(-1)
    n = torch.linalg.norm(v)
    if float(n) < 1e-12:
        raise ValueError("zero vector")
    return v / n


def _build_candidates(U: torch.Tensor, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    u1 = _unit(U[0])
    u2 = _unit(U[1])
    r_full = _unit(torch.tensor(rng.standard_normal(u1.numel()), dtype=torch.float32))
    r_full2 = _unit(torch.tensor(rng.standard_normal(u1.numel()), dtype=torch.float32))
    coef = torch.tensor(rng.standard_normal(U.shape[0]), dtype=torch.float32)
    r_sub = _unit(coef @ U)
    orth = r_full - float(torch.dot(r_full, u2)) * u2
    orth = _unit(orth)
    return {
        "u1": u1,
        "u2": u2,
        "neg_u1": -u1,
        "neg_u2": -u2,
        "rand_sub": r_sub,
        "rand_full": r_full,
        "rand_full2": r_full2,
        "orth_u2": orth,
    }


def _epsilon_star_one(
    loaded: Any,
    *,
    direction: torch.Tensor,
    task: str,
    device: torch.device,
) -> float:
    _spec = importlib.util.spec_from_file_location(
        "run_06b_watchlist", ROOT / "scripts" / "run_06b_watchlist.py"
    )
    assert _spec and _spec.loader
    w06 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = w06
    _spec.loader.exec_module(w06)
    from activation_pipeline.analysis.sensitivity import LayerPerturbHooks

    messages = [
        {"role": "system", "content": _system()},
        {"role": "user", "content": f"{NEUTRAL}\n\nTask: {task}"},
        {"role": "assistant", "content": _force_prefix()},
    ]
    tok = loaded.tokenizer
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tok(rendered, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    idxs = [ids.shape[1] - 1]

    hooks = LayerPerturbHooks(
        loaded.model,
        layer_k=LAYER,
        direction=direction,
        token_indices=idxs,
        read_layers=[LAYER],
    )
    clean = hooks.run(ids, attn, 0.0)
    clean_L = clean[LAYER][0, idxs[0], :].float()

    def pool(t):
        return t[0, idxs[0], :].float()

    def forward_pert(eps: float):
        return hooks.run(ids, attn, eps)

    eps_star, _, _ = w06.epsilon_star(
        clean_L=clean_L,
        forward_pert=forward_pert,
        layer_L=LAYER,
        pool=pool,
        threshold=EPS_TAU,
        max_iter=24,
        blowup_mode="relative",
        ref_norm=float(torch.linalg.norm(clean_L)),
    )
    return float(eps_star)


def main() -> int:
    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.steering import ActivationSteerHook

    dspec = importlib.util.spec_from_file_location(
        "run_gap_deception_collect", ROOT / "scripts" / "run_gap_deception_collect.py"
    )
    assert dspec and dspec.loader
    dcol = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(dcol)

    drow = json.loads(DIR.read_text().splitlines()[0])
    U = torch.tensor(drow["basis"], dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    candidates = _build_candidates(U, rng)

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    device = next(loaded.model.parameters()).device
    system = _system()
    known = {t["name"] for t in tools_for_prompt()}
    registry = ToolRegistry(WORKSPACE)

    print("=== Phase 1: epsilon* on train ===", flush=True)
    eps_train: dict[str, float] = {}
    for did, vec in candidates.items():
        vals = []
        for tid, task in TRAIN_TASKS:
            v = _epsilon_star_one(loaded, direction=vec, task=task, device=device)
            vals.append(v)
            print(f"  {did}/{tid} eps*={v:.4f}", flush=True)
        eps_train[did] = float(np.median(vals))

    svd_ids = ["u1", "u2", "neg_u1", "neg_u2"]
    rand_ids = ["rand_sub", "rand_full", "rand_full2"]
    ranked = sorted(eps_train.items(), key=lambda x: x[1])
    print(f"Ranked eps*: {[(k, round(v, 4)) for k, v in ranked]}", flush=True)

    _e21 = importlib.util.spec_from_file_location(
        "run_exp21_replicate_neg_u2", ROOT / "scripts" / "run_exp21_replicate_neg_u2.py"
    )
    assert _e21 and _e21.loader
    e21 = importlib.util.module_from_spec(_e21)
    sys.modules[_e21.name] = e21
    _e21.loader.exec_module(e21)
    e21.generate_assistant = generate_assistant
    run_episode = e21.run_episode_with_steer

    print("=== Phase 2: selective steering on test ===", flush=True)
    by_dir: dict[str, dict[str, list[dict[str, Any]]]] = {
        did: {"baseline": [], "plus": [], "minus": []} for did in candidates
    }

    for did, vec in candidates.items():
        for rep in range(N_REPS):
            for ti, (tid, task) in enumerate(TEST_TASKS):
                seed = int(SEED + 30007 * rep + 97 * ti + (abs(hash(did)) % 997))
                common = dict(
                    loaded=loaded,
                    dcol=dcol,
                    registry=registry,
                    system=system,
                    known=known,
                    task=task,
                    seed=seed,
                )
                row_b = run_episode(**common, direction=None, alpha=0.0)
                row_p = run_episode(**common, direction=vec, alpha=ALPHA)
                row_m = run_episode(**common, direction=vec, alpha=-ALPHA)
                for arm, row in (("baseline", row_b), ("plus", row_p), ("minus", row_m)):
                    row.update({"direction_id": did, "arm": arm, "rep": rep, "task_id": tid, "seed": seed})
                    by_dir[did][arm].append(row)
        last = by_dir[did]["plus"][-len(TEST_TASKS) :]
        mex = float(np.mean([r["n_extra_paths"] for r in last]))
        print(f"  {did} last-rep mean_extra(+α)={mex:.2f}", flush=True)

    def _paired_deltas(
        steer: list[dict[str, Any]], base: list[dict[str, Any]], metric: str
    ) -> np.ndarray:
        base_key = {(r["task_id"], r["rep"]): r for r in base}
        ds: list[float] = []
        for r in steer:
            b = base_key.get((r["task_id"], r["rep"]))
            if b is None:
                continue
            ds.append(float(r[metric]) - float(b[metric]))
        return np.asarray(ds, dtype=np.float64)

    def _dir_metrics(did: str) -> dict[str, Any]:
        base = by_dir[did]["baseline"]
        plus = by_dir[did]["plus"]
        minus = by_dir[did]["minus"]
        d_extra_p = _paired_deltas(plus, base, "n_extra_paths")
        d_extra_m = _paired_deltas(minus, base, "n_extra_paths")
        mean_dp = float(d_extra_p.mean()) if d_extra_p.size else 0.0
        mean_dm = float(d_extra_m.mean()) if d_extra_m.size else 0.0
        target_effect = max(abs(mean_dp), abs(mean_dm))

        def _collateral(steer: list[dict[str, Any]]) -> float:
            d_calls = _paired_deltas(steer, base, "n_calls")
            d_list = _paired_deltas(steer, base, "n_list")
            d_search = _paired_deltas(steer, base, "n_search")
            parts = [d_calls, d_list, d_search]
            return float(
                sum(abs(d.mean()) if d.size else 0.0 for d in parts)
            )

        coll_p = _collateral(plus)
        coll_m = _collateral(minus)
        collateral = 0.5 * (coll_p + coll_m)
        controllability = target_effect
        selectivity = target_effect / (collateral + COLLATERAL_EPS)
        return {
            "direction_id": did,
            "eps_star_train": eps_train[did],
            "target_effect": target_effect,
            "collateral": collateral,
            "selectivity": selectivity,
            "controllability": controllability,
            "mean_d_extra_plus": mean_dp,
            "mean_d_extra_minus": mean_dm,
            "collateral_plus": coll_p,
            "collateral_minus": coll_m,
        }

    per_dir = [_dir_metrics(did) for did in candidates]
    eps_arr = np.array([d["eps_star_train"] for d in per_dir], dtype=np.float64)
    c_arr = np.array([d["controllability"] for d in per_dir], dtype=np.float64)
    coll_arr = np.array([d["collateral"] for d in per_dir], dtype=np.float64)
    rho_eps_c = _spearman(eps_arr, c_arr)
    rho_eps_coll = _spearman(eps_arr, coll_arr)

    best_svd_s = max(d["selectivity"] for d in per_dir if d["direction_id"] in svd_ids)
    rand_pool = rand_ids + ["orth_u2"]
    best_rand_s = max(d["selectivity"] for d in per_dir if d["direction_id"] in rand_pool)
    best_target = max(d["target_effect"] for d in per_dir)

    if best_svd_s > 1.5 * best_rand_s and rho_eps_c <= -0.5:
        decision = "SELECTIVE_HIT"
    elif best_target > 0.05:
        decision = "SELECTIVE_WEAK"
    else:
        decision = "SELECTIVE_NULL"

    payload: dict[str, Any] = {
        "protocol": str(PROTO.relative_to(ROOT)),
        "layer": LAYER,
        "seed": SEED,
        "n_reps": N_REPS,
        "alpha": ALPHA,
        "train_tasks": [t[0] for t in TRAIN_TASKS],
        "test_tasks": [t[0] for t in TEST_TASKS],
        "eps_star_train": eps_train,
        "eps_ranked": ranked,
        "decision": decision,
        "best_svd_selectivity": best_svd_s,
        "best_random_selectivity": best_rand_s,
        "best_target_effect": best_target,
        "spearman_eps_controllability": rho_eps_c,
        "spearman_eps_collateral": rho_eps_coll,
        "per_direction": {d["direction_id"]: d for d in per_dir},
        "rows": by_dir,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Selective steering — geometry vs controllability",
        "",
        f"- Decision: `{decision}`",
        f"- Train ε* ranked: {[(k, round(v, 4)) for k, v in ranked]}",
        f"- Spearman(ε*, C)={rho_eps_c:.3f}  Spearman(ε*, collateral)={rho_eps_coll:.3f}",
        f"- Best SVD S={best_svd_s:.3f}  best random S={best_rand_s:.3f}  "
        f"ratio={best_svd_s / (best_rand_s + 1e-9):.2f}",
        "",
        "| direction | ε* train | target | collateral | S | C | Δextra(+) | Δextra(−) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in sorted(per_dir, key=lambda x: x["selectivity"], reverse=True):
        lines.append(
            f"| `{d['direction_id']}` | {d['eps_star_train']:.4f} | "
            f"{d['target_effect']:.3f} | {d['collateral']:.3f} | "
            f"{d['selectivity']:.3f} | {d['controllability']:.3f} | "
            f"{d['mean_d_extra_plus']:+.3f} | {d['mean_d_extra_minus']:+.3f} |"
        )
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "rho_eps_c": rho_eps_c,
                "best_svd_s": best_svd_s,
                "best_rand_s": best_rand_s,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
