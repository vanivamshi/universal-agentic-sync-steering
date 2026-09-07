#!/usr/bin/env python3
"""Exp 2b — diagnose Exp 2 DOSE_NULL: sign × component × position × small α.

Locked: docs/proximal_tool_steer.md (Exp 2b)

AmpHook is QQᵀ-symmetric (U ≡ −U). Oriented ±u tests use ActivationSteerHook.
Random k=2 AmpHook is the specificity control.
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
SEED = 20260820
# Small local regime — Exp 2 collapsed already at +0.5.
ALPHAS_AMP = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)
# Sign lives in the direction for steer; keep α ≥ 0.
ALPHAS_STEER = (0.0, 0.1, 0.25, 0.5)
TEMPERATURE = 0.2
MAX_TURNS = 4
MAX_NEW_TOKENS = 192
WORKSPACE = ROOT / "data" / "sandbox_exp0"
DIR = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.jsonl"
GATE1 = ROOT / "data" / "results" / "exp1_evidence_subspace.json"
GATE2 = ROOT / "data" / "results" / "exp2_subspace_dose.json"
OUT = ROOT / "data" / "results" / "exp2b_subspace_localize.json"
MD = ROOT / "data" / "results" / "exp2b_subspace_localize.md"

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
    n_other = len(names) - n_list - n_search - n_read
    return {
        "tool_emission": int(bool(calls)),
        "n_calls": len(calls),
        "n_unique_tools": len(set(names)),
        "n_unique_paths": len(set(paths)),
        "n_extra_paths": len(set(extra)),
        "n_exploratory": n_list + n_search,
        "n_list": n_list,
        "n_search": n_search,
        "n_read": n_read,
        "n_other": n_other,
        "evidence_seek": int(len(extra) > 0 or n_list + n_search > 0),
        "paths": paths,
        "tools": names,
    }


def _mean(rows: list[dict], k: str) -> float:
    return float(np.mean([r[k] for r in rows])) if rows else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64)))
    if float(rx.std()) < 1e-12 or float(ry.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _cfg_key(kind: str, basis_id: str, pos: str) -> str:
    return f"{kind}|{basis_id}|{pos}"


def main() -> int:
    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.steering import (
        ActivationSteerHook,
        ActivationSubspaceAmpHook,
    )

    if not GATE1.exists():
        print("missing Exp 1 result", flush=True)
        return 1
    g1 = json.loads(GATE1.read_text())
    if g1.get("decision") != "EXTRACT_OK":
        print(f"Exp 1 decision={g1.get('decision')}; Exp 2b not licensed", flush=True)
        return 1
    if not GATE2.exists():
        print("missing Exp 2 result (run Exp 2 first)", flush=True)
        return 1

    drow = json.loads(DIR.read_text().splitlines()[0])
    U = torch.tensor(drow["basis"], dtype=torch.float32)
    assert int(drow["layer"]) == LAYER
    k = int(drow["k"])
    assert U.ndim == 2 and U.shape[0] == k
    u1 = U[0]
    u2 = U[1]

    rng = np.random.default_rng(SEED)
    R = torch.tensor(rng.standard_normal((k, U.shape[1])), dtype=torch.float32)
    Qt, _ = torch.linalg.qr(R.T, mode="reduced")
    U_rand = Qt[:, :k].T.contiguous()

    bases: dict[str, torch.Tensor] = {
        "U": U,
        "u1": u1.reshape(1, -1),
        "u2": u2.reshape(1, -1),
        "rand": U_rand,
    }
    steer_dirs: dict[str, torch.Tensor] = {
        "+u1": u1,
        "-u1": -u1,
        "+u2": u2,
        "-u2": -u2,
    }

    # Factorial (compact): amp localization + components; steer signs on last;
    # random control on last (extract-matched).
    amp_cfgs = [
        ("U", "all"),
        ("U", "last"),
        ("u1", "last"),
        ("u2", "last"),
        ("rand", "last"),
    ]
    steer_cfgs = [
        ("+u1", "last"),
        ("-u1", "last"),
        ("+u2", "last"),
        ("-u2", "last"),
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
        kind: str,
        basis_id: str,
        pos: str,
        alpha: float,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{NEUTRAL}\n\nTask: {task}"},
        ]
        hook: Any = None
        if abs(alpha) > 1e-12:
            if kind == "amp":
                hook = ActivationSubspaceAmpHook(
                    loaded.model,
                    layer=LAYER,
                    basis=bases[basis_id],
                    alpha=alpha,
                    pos_mode=pos,
                    collect_stats=True,
                )
            else:
                hook = ActivationSteerHook(
                    loaded.model,
                    layer=LAYER,
                    direction=steer_dirs[basis_id],
                    alpha=alpha,
                    pos_mode=pos,
                    collect_stats=True,
                )
            hook.register()
        all_calls: list[dict[str, Any]] = []
        try:
            for _ in range(MAX_TURNS):
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
            stats = hook.mean_stats() if hook is not None else {}
            if hook is not None:
                hook.remove()
        m = _metrics(all_calls)
        m.update(
            {
                "task_id": task_id,
                "kind": kind,
                "basis_id": basis_id,
                "pos": pos,
                "alpha": alpha,
                "hook_stats": stats,
            }
        )
        return m

    results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    print("=== Exp 2b localize / sign / dose ===", flush=True)

    # Shared α=0 baseline (no hook).
    base_rows: list[dict[str, Any]] = []
    for tid, task in TASKS:
        row = run_one(
            task_id=tid, task=task, kind="amp", basis_id="U", pos="all", alpha=0.0
        )
        base_rows.append(row)
        print(
            f"α=0 {tid} extra={row['n_extra_paths']} calls={row['n_calls']} "
            f"list={row['n_list']} search={row['n_search']}",
            flush=True,
        )
    results[_cfg_key("baseline", "none", "none")] = {"0.0": base_rows}

    def sweep_amp(basis_id: str, pos: str) -> None:
        key = _cfg_key("amp", basis_id, pos)
        results[key] = {}
        for a in ALPHAS_AMP:
            if abs(a) < 1e-12:
                results[key][str(a)] = base_rows
                continue
            rows = []
            for tid, task in TASKS:
                row = run_one(
                    task_id=tid,
                    task=task,
                    kind="amp",
                    basis_id=basis_id,
                    pos=pos,
                    alpha=a,
                )
                rows.append(row)
                print(
                    f"amp {basis_id} pos={pos} α={a:+g} {tid} "
                    f"extra={row['n_extra_paths']} calls={row['n_calls']} "
                    f"Δn={row['hook_stats'].get('mean_delta_norm', float('nan')):.3g}",
                    flush=True,
                )
            results[key][str(a)] = rows

    def sweep_steer(basis_id: str, pos: str) -> None:
        key = _cfg_key("steer", basis_id, pos)
        results[key] = {}
        for a in ALPHAS_STEER:
            if abs(a) < 1e-12:
                results[key][str(a)] = base_rows
                continue
            rows = []
            for tid, task in TASKS:
                row = run_one(
                    task_id=tid,
                    task=task,
                    kind="steer",
                    basis_id=basis_id,
                    pos=pos,
                    alpha=a,
                )
                rows.append(row)
                print(
                    f"steer {basis_id} pos={pos} α={a:+g} {tid} "
                    f"extra={row['n_extra_paths']} calls={row['n_calls']} "
                    f"Δn={row['hook_stats'].get('mean_delta_norm', float('nan')):.3g}",
                    flush=True,
                )
            results[key][str(a)] = rows

    for bid, pos in amp_cfgs:
        sweep_amp(bid, pos)
    for bid, pos in steer_cfgs:
        sweep_steer(bid, pos)

    summaries: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    for key, by_a in results.items():
        if key.startswith("baseline"):
            continue
        kind, basis_id, pos = key.split("|")
        alphas = (
            list(ALPHAS_AMP) if kind == "amp" else list(ALPHAS_STEER)
        )
        mean_extra = [_mean(by_a[str(a)], "n_extra_paths") for a in alphas]
        mean_calls = [_mean(by_a[str(a)], "n_calls") for a in alphas]
        mean_list = [_mean(by_a[str(a)], "n_list") for a in alphas]
        mean_search = [_mean(by_a[str(a)], "n_search") for a in alphas]
        mean_read = [_mean(by_a[str(a)], "n_read") for a in alphas]
        rho = _spearman(alphas, mean_extra)
        e0 = mean_extra[alphas.index(0.0)]
        # Local Δ: max positive α vs 0
        a_hi = max(alphas)
        e_hi = mean_extra[alphas.index(a_hi)]
        c0 = mean_calls[alphas.index(0.0)]
        c_hi = mean_calls[alphas.index(a_hi)]
        # Also check best positive-α mean extra among grid
        pos_alphas = [a for a in alphas if a > 0]
        best_pos_a = max(pos_alphas, key=lambda a: mean_extra[alphas.index(a)])
        e_best = mean_extra[alphas.index(best_pos_a)]
        # Mean proj/delta at |α|=a_hi (first task avg)
        hi_rows = by_a[str(a_hi)]
        proj_keys = []
        for r in hi_rows:
            hs = r.get("hook_stats") or {}
            if "mean_proj_norm" in hs:
                proj_keys.append(hs["mean_proj_norm"])
            elif "mean_abs_coord" in hs:
                proj_keys.append(hs["mean_abs_coord"])
        delta_keys = [
            float((r.get("hook_stats") or {}).get("mean_delta_norm", float("nan")))
            for r in hi_rows
        ]
        mean_proj = float(np.nanmean(proj_keys)) if proj_keys else float("nan")
        mean_delta = float(np.nanmean(delta_keys)) if delta_keys else float("nan")

        # Local HIT: Spearman ≥ 0.60 on small grid AND some +α raises extra
        # without collapsing calls below half of baseline.
        calls_ok = c_hi >= 0.5 * c0 if c0 > 0 else c_hi > 0
        local_hit = (
            rho == rho
            and rho >= 0.60
            and e_best > e0
            and calls_ok
        )
        local_weak = (
            ((rho == rho and rho >= 0.60) or e_best > e0)
            and calls_ok
            and not local_hit
        )
        decision = (
            "LOCAL_HIT" if local_hit else "LOCAL_WEAK" if local_weak else "LOCAL_NULL"
        )
        # Collapse flag: +α kills tools while moving activation
        collapse = e_hi < e0 - 0.25 and c_hi < 0.5 * max(c0, 1e-9) and mean_delta > 1e-3

        summ = {
            "kind": kind,
            "basis_id": basis_id,
            "pos": pos,
            "alphas": alphas,
            "mean_n_extra_paths": {str(a): mean_extra[i] for i, a in enumerate(alphas)},
            "mean_n_calls": {str(a): mean_calls[i] for i, a in enumerate(alphas)},
            "mean_n_list": {str(a): mean_list[i] for i, a in enumerate(alphas)},
            "mean_n_search": {str(a): mean_search[i] for i, a in enumerate(alphas)},
            "mean_n_read": {str(a): mean_read[i] for i, a in enumerate(alphas)},
            "spearman_alpha_extra": rho,
            "delta_extra_best_pos_vs_0": e_best - e0,
            "best_pos_alpha": best_pos_a,
            "delta_extra_ahi_vs_0": e_hi - e0,
            "delta_calls_ahi_vs_0": c_hi - c0,
            "mean_proj_or_coord_at_ahi": mean_proj,
            "mean_delta_norm_at_ahi": mean_delta,
            "tool_collapse_at_ahi": collapse,
            "decision": decision,
        }
        summaries[key] = summ
        candidates.append({"key": key, **summ})

    hits = [c for c in candidates if c["decision"] == "LOCAL_HIT"]
    weaks = [c for c in candidates if c["decision"] == "LOCAL_WEAK"]
    # Prefer evidence U over random for overall decision
    evidence_hits = [c for c in hits if c["basis_id"] != "rand"]
    rand_hits = [c for c in hits if c["basis_id"] == "rand"]

    if evidence_hits and not rand_hits:
        overall = "DOSE2B_HIT"
        best = max(
            evidence_hits,
            key=lambda c: (c["spearman_alpha_extra"], c["delta_extra_best_pos_vs_0"]),
        )
    elif evidence_hits and rand_hits:
        overall = "DOSE2B_NONSPECIFIC"
        best = max(
            evidence_hits,
            key=lambda c: (c["spearman_alpha_extra"], c["delta_extra_best_pos_vs_0"]),
        )
    elif weaks and any(c["basis_id"] != "rand" for c in weaks):
        overall = "DOSE2B_WEAK"
        best = max(
            [c for c in weaks if c["basis_id"] != "rand"],
            key=lambda c: (c.get("spearman_alpha_extra") or -1, c["delta_extra_best_pos_vs_0"]),
        )
    else:
        overall = "DOSE2B_NULL"
        best = max(
            candidates,
            key=lambda c: (c.get("spearman_alpha_extra") or -99, c["delta_extra_best_pos_vs_0"]),
        )

    payload = {
        "protocol": "docs/proximal_tool_steer.md",
        "experiment": "2b",
        "prerequisite": "exp1 EXTRACT_OK; exp2 DOSE_NULL",
        "layer": LAYER,
        "k": k,
        "alphas_amp": list(ALPHAS_AMP),
        "alphas_steer": list(ALPHAS_STEER),
        "affordance": "neutral",
        "note_amp_sign": "AmpHook QQᵀ is sign-invariant; ±u via SteerHook",
        "decision": overall,
        "best_config": best["key"] if best else None,
        "summaries": summaries,
        "baseline_mean_extra": _mean(base_rows, "n_extra_paths"),
        "baseline_mean_calls": _mean(base_rows, "n_calls"),
        "rows": results,
        "claim": "extra_path_evidence_seeking_not_constraint_obedience",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Exp 2b — localize / sign / small-α / random control",
        "",
        f"- Overall: `{overall}`",
        f"- Best config: `{best['key'] if best else None}`",
        f"- Baseline mean extra={payload['baseline_mean_extra']:.2f} "
        f"calls={payload['baseline_mean_calls']:.2f}",
        "",
        "| config | ρ(α,extra) | Δextra(best+) | Δcalls(αhi) | "
        "proj@αhi | collapse | decision |",
        "|---|---:|---:|---:|---:|:---:|---|",
    ]
    for c in sorted(candidates, key=lambda x: x["key"]):
        lines.append(
            f"| `{c['key']}` | {c['spearman_alpha_extra']:.3f} | "
            f"{c['delta_extra_best_pos_vs_0']:+.2f} | "
            f"{c['delta_calls_ahi_vs_0']:+.2f} | "
            f"{c['mean_delta_norm_at_ahi']:.3g} | "
            f"{'Y' if c['tool_collapse_at_ahi'] else 'n'} | "
            f"`{c['decision']}` |"
        )
    lines += [
        "",
        "Primary = n_extra_paths with calls preserved. AmpHook U≡−U; "
        "oriented tests are steer ±u.",
        "Advance to Exp 3 only on DOSE2B_HIT (evidence > random).",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": overall,
                "best": best["key"] if best else None,
                "best_rho": best.get("spearman_alpha_extra") if best else None,
                "best_d_extra": best.get("delta_extra_best_pos_vs_0") if best else None,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
