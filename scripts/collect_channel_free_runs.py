#!/usr/bin/env python3
"""Stage A — Free trajectories for factor identification (C, H, O).

Goal: enough 0/1 mass on each *factor* (not all 8 CHO cells).
Stratified schedule pairs:
  public tasks + classes A/D  → expect H≈0
  private tasks + classes B/C → expect H≈1 when smoke check runs

Saves cache + JSON. Next: evaluate_channel_probes.py (representation tests).

  .venv/bin/python scripts/collect_channel_free_runs.py --n 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_channel_free_runs.json"
CACHE = ROOT / "data" / "results" / "sync_channel_free_runs_cache.npz"
SEED = 20260902


def _load_sc():
    spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _score_row(row: dict) -> dict[str, Any]:
    from scripts.sync_eq import extract_final, extract_plan, score_plan

    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    plan = extract_plan(blob)
    c_raw = score_plan(plan, H) if plan else None
    C = int(c_raw) if c_raw is not None else -1
    return {
        "C": C,
        "H": H,
        "O": O,
        "plan": (plan or "")[:500],
        "final": extract_final(row.get("final") or "")[:500],
        "plan_observed": bool(plan),
    }


def _stratified_schedule(sc, n: int, seed: int) -> list[tuple[str, str]]:
    """Alternate private/public compatible (cls, task_id) pairs from iter_episode_plan."""
    plan = sc.iter_episode_plan()  # (cls, task_id, folder, sens)
    priv = [(c, tid) for c, tid, _f, sens in plan if sens != "public"]
    pub = [(c, tid) for c, tid, _f, sens in plan if sens == "public"]
    rng = np.random.default_rng(seed)
    rng.shuffle(priv)
    rng.shuffle(pub)
    out: list[tuple[str, str]] = []
    i_p = i_u = 0
    for i in range(n):
        if i % 2 == 0 and priv:
            out.append(priv[i_p % len(priv)])
            i_p += 1
        elif pub:
            out.append(pub[i_u % len(pub)])
            i_u += 1
        elif priv:
            out.append(priv[i_p % len(priv)])
            i_p += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-private-tries", type=int, default=4,
                    help="retries on private tasks until H=1 or give up")
    ap.add_argument(
        "--append",
        action="store_true",
        help="append to existing cache/json instead of replacing",
    )
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )

    schedule = _stratified_schedule(sc, args.n, args.seed)
    episodes: list[dict[str, Any]] = []
    h_tools, h_reports = [], []
    Cs, Hs, Os = [], [], []
    offset = 0

    if args.append and CACHE.is_file() and OUT.is_file():
        old = np.load(CACHE)
        h_tools = list(np.asarray(old["h_tool"]))
        h_reports = list(np.asarray(old["h_report"]))
        Cs = list(np.asarray(old["C"]))
        Hs = list(np.asarray(old["H"]))
        Os = list(np.asarray(old["O"]))
        prev = json.loads(OUT.read_text())
        episodes = list(prev.get("episodes") or [])
        offset = len(episodes)
        print(f"appending to {offset} existing episodes", flush=True)

    max_private_tries = int(args.max_private_tries)

    for i, (cls, task_id) in enumerate(schedule):
        idx = offset + i
        task = sc.task_by_id(task_id)
        want_H = bool(task.get("env_file"))
        row = None
        scored = None
        tries = 0
        while True:
            tries += 1
            row = sc.run_episode(
                loaded,
                cls=cls,
                task_id=task_id,
                seed=args.seed + idx * 17 + tries,
                with_plan_format=True,
                capture_activations=True,
            )
            scored = _score_row(row)
            if (not want_H) or scored["H"] == 1 or tries >= max_private_tries:
                break
            print(
                f"  retry {tries}/{max_private_tries} {task_id} cls={cls} "
                f"H=0 tools={list(row.get('tools') or [])[:3]}",
                flush=True,
            )
        ht, hr = row.get("h_tool"), row.get("h_report")
        ep = {
            "i": idx,
            "task_id": task_id,
            "cls": cls,
            "seed": args.seed + idx * 17 + tries,
            "tries": tries,
            "tools_used": list(row.get("tools") or []),
            "sensitive_paths": list(row.get("sensitive_paths") or []),
            "runner_likely": any(
                "run_check" in str(t) or "run_command" in str(t)
                for t in (row.get("tools") or [])
            ),
            **scored,
            "s_tool": row.get("s_tool"),
            "s_output": row.get("s_output"),
            "has_h_tool": ht is not None,
            "has_h_report": hr is not None,
        }
        episodes.append(ep)
        if hr is not None:
            h_reports.append(np.asarray(hr, dtype=np.float64))
            h_tools.append(np.asarray(ht if ht is not None else hr, dtype=np.float64))
            Cs.append(scored["C"])
            Hs.append(scored["H"])
            Os.append(scored["O"])
        print(
            f"[{i+1}/{args.n}] {task_id} cls={cls} "
            f"S=({scored['C']},{scored['H']},{scored['O']}) "
            f"plan={scored['plan_observed']} tries={tries} tools={ep['tools_used'][:3]}",
            flush=True,
        )

    C_arr = np.asarray(Cs, dtype=np.int64)
    H_arr = np.asarray(Hs, dtype=np.int64)
    O_arr = np.asarray(Os, dtype=np.int64)
    Ht = np.stack(h_tools, 0) if h_tools else np.zeros((0, 1))
    Hr = np.stack(h_reports, 0) if h_reports else np.zeros((0, 1))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, h_tool=Ht, h_report=Hr, C=C_arr, H=H_arr, O=O_arr)

    occ = Counter()
    for e in episodes:
        if e["C"] < 0:
            continue
        occ[f"{e['C']}{e['H']}{e['O']}"] += 1

    n_H1 = int(np.sum(H_arr == 1)) if len(H_arr) else 0
    n_H0 = int(np.sum(H_arr == 0)) if len(H_arr) else 0
    n_C1 = int(np.sum(C_arr == 1)) if len(C_arr) else 0
    n_C0 = int(np.sum(C_arr == 0)) if len(C_arr) else 0
    n_O1 = int(np.sum(O_arr == 1)) if len(O_arr) else 0
    n_O0 = int(np.sum(O_arr == 0)) if len(O_arr) else 0

    payload: dict[str, Any] = {
        "n": len(episodes),
        "n_plan_observed": sum(1 for e in episodes if e["plan_observed"]),
        "n_C_known": int(np.sum(C_arr >= 0)) if len(C_arr) else 0,
        "factor_balance": {
            "C": {"n0": n_C0, "n1": n_C1},
            "H": {"n0": n_H0, "n1": n_H1},
            "O": {"n0": n_O0, "n1": n_O1},
            "goal": "plenty of 0 and 1 per factor — not all 8 CHO cells",
        },
        "n_H1": n_H1,
        "n_H0": n_H0,
        "n_O1_given_H1": int(np.sum((H_arr == 1) & (O_arr == 1))) if len(H_arr) else 0,
        "n_O0_given_H1": int(np.sum((H_arr == 1) & (O_arr == 0))) if len(H_arr) else 0,
        "occupancy_CHO": dict(sorted(occ.items())),
        "cache": str(CACHE),
        "objective": (
            "Stage A: identify factor variation for C,H,O. "
            "Not: collect all eight labels or improve 8-way score."
        ),
        "episodes": episodes,
    }
    if n_H1 < 5:
        payload["warning"] = (
            f"H=1 only {n_H1}/{len(H_arr)} — cannot identify v_H. "
            "Increasing n alone may not help: private B/C episodes must actually "
            "execute the env-loading smoke check (inspect tools_used)."
        )
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "episodes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
