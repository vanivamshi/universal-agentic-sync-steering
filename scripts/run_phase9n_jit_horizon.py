#!/usr/bin/env python3
"""Phase 9N — Just-in-time / receding-horizon control (no hold).

Hypothesis from 9L/9M: persistent S=m* is the wrong objective. Control each
bit at its decision site instead of acquire-then-stabilize.

Arms (same free S0, frozen 8K v_c / α):
  jit_rh     — receding horizon: among remaining errors, intervene next in
               decision order C→H→O (O last); one channel per episode; stop at hit
  jit_oneshot— one episode with all currently mismatched bits active at sites
  compose    — Phase 8K cumulative H→C→O (layered active set)

No new v. No hold/retain. Primary: P(S_final=m*) for all 8.

  .venv/bin/python -u scripts/run_phase9n_jit_horizon.py --reps 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts.sync_eq import sync_error_norm  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase9n_jit_horizon.json"
MD = ROOT / "data" / "results" / "sync_phase9n_jit_horizon.md"
SEED = 20261002
# Chronological decision sites in generation (O terminal)
DECISION_ORDER = ("C", "H", "O")
IMPL_ORDER = ("H", "C", "O")  # 8K compose order
CH_IDX = {"C": 0, "H": 1, "O": 2}
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
ARMS = ("jit_rh", "jit_oneshot", "compose")


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _e(mstar, S) -> list[int]:
    return [int(a) != int(b) for a, b in zip(mstar, S)]


def _E(mstar, S) -> int:
    return sum(_e(mstar, S))


def remaining(mstar, S) -> list[str]:
    return [a for a in DECISION_ORDER if int(S[CH_IDX[a]]) != int(mstar[CH_IDX[a]])]


def run_jit_rh(p8k, sc, loaded, *, dirs, mstar, seed, S0, plan0, max_steps: int):
    """Observe → correct one remaining bit (C→H→O) → observe; no hold."""
    S = list(S0)
    plan = plan0 or ""
    traj = [list(S)]
    steps = []
    for t in range(max_steps):
        if S == list(mstar):
            break
        rem = remaining(mstar, S)
        if not rem:
            break
        a = rem[0]  # next in decision order
        active = {a: (p8k._s_star(mstar[CH_IDX[a]]), dirs[a])}
        h_pf = plan if a == "H" else None
        out = p8k._run_episode(
            sc, loaded, active=active, seed=seed + 1 + t, h_plan_prefill=h_pf
        )
        Sa = list(out["S"])
        if out.get("plan_prefill"):
            plan = out["plan_prefill"]
        steps.append(
            {
                "t": t,
                "a": a,
                "remaining_before": rem,
                "S_before": list(S),
                "S_after": Sa,
                "dE": _E(mstar, Sa) - _E(mstar, S),
            }
        )
        S = Sa
        traj.append(list(S))
    return _pack("jit_rh", mstar, S0, traj, steps)


def run_jit_oneshot(p8k, sc, loaded, *, dirs, mstar, seed, S0, plan0):
    """One episode: all mismatched bits steered at their decision sites."""
    rem = remaining(mstar, S0)
    if not rem:
        return _pack("jit_oneshot", mstar, S0, [list(S0)], [])
    active = {a: (p8k._s_star(mstar[CH_IDX[a]]), dirs[a]) for a in rem}
    h_pf = plan0 if ("H" in active and "C" not in active) else None
    out = p8k._run_episode(
        sc, loaded, active=active, seed=seed + 1, h_plan_prefill=h_pf
    )
    Sa = list(out["S"])
    steps = [
        {
            "t": 0,
            "a": "+".join(rem),
            "remaining_before": rem,
            "S_before": list(S0),
            "S_after": Sa,
            "dE": _E(mstar, Sa) - _E(mstar, S0),
        }
    ]
    return _pack("jit_oneshot", mstar, S0, [list(S0), Sa], steps)


def run_compose(p8k, sc, loaded, *, dirs, mstar, seed, S0, plan0):
    """8K-style cumulative active set in IMPL_ORDER H→C→O."""
    S = list(S0)
    plan = plan0 or ""
    traj = [list(S)]
    steps = []
    active: dict = {}
    t = 0
    for k in IMPL_ORDER:
        if int(S[CH_IDX[k]]) == int(mstar[CH_IDX[k]]):
            continue
        if k in active:
            continue
        active[k] = (p8k._s_star(mstar[CH_IDX[k]]), dirs[k])
        h_pf = None
        if "H" in active and "C" not in active:
            h_pf = plan
        out = p8k._run_episode(
            sc, loaded, active=dict(active), seed=seed, h_plan_prefill=h_pf
        )
        Sa = list(out["S"])
        if out.get("plan_prefill"):
            plan = out["plan_prefill"]
        steps.append(
            {
                "t": t,
                "a": "+".join(active.keys()),
                "added": k,
                "S_before": list(S),
                "S_after": Sa,
                "dE": _E(mstar, Sa) - _E(mstar, S),
            }
        )
        S = Sa
        traj.append(list(S))
        t += 1
    return _pack("compose", mstar, S0, traj, steps)


def _pack(arm, mstar, S0, traj, steps):
    Sf = traj[-1]
    return {
        "arm": arm,
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "S0": list(S0),
        "S_final": list(Sf),
        "traj_S": traj,
        "path": [_key(s) for s in traj],
        "steps": steps,
        "actions": [s.get("a") for s in steps],
        "hit": int(Sf == list(mstar)),
        "ever": int(any(s == list(mstar) for s in traj)),
        "E0": _E(mstar, S0),
        "E_final": _E(mstar, Sf),
        "delta_E": _E(mstar, S0) - _E(mstar, Sf),
        "bit_C": int(Sf[0] == mstar[0]),
        "bit_H": int(Sf[1] == mstar[1]),
        "bit_O": int(Sf[2] == mstar[2]),
        "n_steps": len(steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-steps-rh", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    dirs = p8k._load_vc()

    by_arm: dict[str, list] = defaultdict(list)
    by_m: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    paired = []

    n_tot = 8 * args.reps
    idx = 0
    print(
        f"=== Phase 9N JIT/RH vs oneshot vs compose "
        f"reps={args.reps} rh_steps={args.max_steps_rh} ===",
        flush=True,
    )

    for mi, mstar in enumerate(ALL_MSTAR):
        for r in range(args.reps):
            seed0 = args.seed + 200 * mi + 19 * r
            # free S0
            seed = seed0
            S0 = None
            plan0 = ""
            for attempt in range(10):
                seed = seed0 + 29 * attempt
                torch.manual_seed(seed)
                free = p8k._run_episode(sc, loaded, active={}, seed=seed)
                S0 = list(free["S"])
                plan0 = free.get("plan_prefill") or ""
                if S0 != list(mstar):
                    break
            idx += 1
            print(
                f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                f"E0={_E(mstar, S0)} rem={remaining(mstar, S0)}",
                flush=True,
            )
            if S0 == list(mstar):
                print("    skip S0==m*", flush=True)
                continue

            trials = {}
            trials["jit_rh"] = run_jit_rh(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                mstar=mstar,
                seed=seed,
                S0=S0,
                plan0=plan0,
                max_steps=args.max_steps_rh,
            )
            trials["jit_oneshot"] = run_jit_oneshot(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                mstar=mstar,
                seed=seed,
                S0=S0,
                plan0=plan0,
            )
            trials["compose"] = run_compose(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                mstar=mstar,
                seed=seed,
                S0=S0,
                plan0=plan0,
            )
            for arm in ARMS:
                t = trials[arm]
                by_arm[arm].append(t)
                by_m[_key(mstar)][arm].append(t)
                print(
                    f"    {arm}: hit={t['hit']} E={t['E0']}→{t['E_final']} "
                    f"path={t['path']} acts={t['actions']}",
                    flush=True,
                )
            paired.append(
                {
                    "mstar": list(mstar),
                    "mstar_key": _key(mstar),
                    "seed": seed,
                    "S0": S0,
                    **trials,
                }
            )

    def agg(trials: list[dict]) -> dict[str, Any]:
        if not trials:
            return {"n": 0}

        def m(k):
            return float(np.mean([t[k] for t in trials]))

        return {
            "n": len(trials),
            "P_hit": m("hit"),
            "P_ever": m("ever"),
            "mean_delta_E": m("delta_E"),
            "P_C": m("bit_C"),
            "P_H": m("bit_H"),
            "P_O": m("bit_O"),
            "mean_n_steps": m("n_steps"),
        }

    arm_agg = {a: agg(by_arm[a]) for a in ARMS}
    per_m = {
        _key(m): {a: agg(by_m[_key(m)][a]) for a in ARMS} for m in ALL_MSTAR
    }
    n_hit = {
        a: sum(1 for m in ALL_MSTAR if (per_m[_key(m)][a].get("P_hit") or 0) > 0)
        for a in ARMS
    }

    best_arm = max(ARMS, key=lambda a: arm_agg[a].get("P_hit") or 0)
    gate = {
        "hypothesis": (
            "Just-in-time decision-site control beats early acquire/hold; "
            "P(S_final=m*) is the objective, not retention"
        ),
        "decision_order": list(DECISION_ORDER),
        "P_hit": {a: arm_agg[a].get("P_hit") for a in ARMS},
        "n_mstar_hit_gt0": n_hit,
        "best_arm": best_arm,
        "jit_beats_compose": (arm_agg["jit_rh"].get("P_hit") or 0)
        > (arm_agg["compose"].get("P_hit") or 0)
        or (arm_agg["jit_oneshot"].get("P_hit") or 0)
        > (arm_agg["compose"].get("P_hit") or 0),
        "universal_hit_best": n_hit[best_arm] == 8,
        "read": (
            "Primary: P_hit by arm. If JIT/RH ≥ compose on hard m*, "
            "early persistent-state objective was wrong."
        ),
    }

    payload = {
        "protocol": "Phase 9N JIT / receding-horizon vs compose",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps_rh": args.max_steps_rh,
        "agg": arm_agg,
        "per_mstar": per_m,
        "paired": paired,
        "gate": gate,
        "sync_error_norm_ref": sync_error_norm,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9N — Just-in-time / receding-horizon control",
        "",
        "> Same frozen $v_c$/α. No hold. Intervene at decision sites "
        r"($C\to H\to O$, O last).",
        "",
        f"reps={args.reps}, rh_max_steps={args.max_steps_rh}, seed={args.seed}.",
        "",
        "## Aggregate",
        "",
        "| arm | n | $P_{hit}$ | $P_{ever}$ | mean $\\Delta E$ | "
        "$P_C$ | $P_H$ | $P_O$ | ever $m^*$ |",
        "|-----|--:|----------:|-----------:|-----------------:|"
        "------:|------:|------:|-----------:|",
    ]
    labels = {
        "jit_rh": "**JIT-RH**",
        "jit_oneshot": "JIT-oneshot",
        "compose": "compose 8K",
    }
    for a in ARMS:
        st = arm_agg[a]
        lines.append(
            f"| {labels[a]} | {st['n']} | **{fmt(st.get('P_hit'))}** | "
            f"{fmt(st.get('P_ever'))} | {fmt(st.get('mean_delta_E'))} | "
            f"{fmt(st.get('P_C'))} | {fmt(st.get('P_H'))} | {fmt(st.get('P_O'))} | "
            f"**{n_hit[a]}/8** |"
        )
    lines += [
        "",
        "## Per $m^*$ $P_{hit}$",
        "",
        "| $m^*$ | JIT-RH | oneshot | compose |",
        "|-------|-------:|--------:|--------:|",
    ]
    for m in ALL_MSTAR:
        mk = _key(m)
        st = per_m[mk]
        lines.append(
            f"| `{mk}` | {fmt(st['jit_rh'].get('P_hit'))} | "
            f"{fmt(st['jit_oneshot'].get('P_hit'))} | "
            f"{fmt(st['compose'].get('P_hit'))} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- Best arm: **{best_arm}** ($P_{{hit}}$={fmt(arm_agg[best_arm].get('P_hit'))})",
        f"- JIT beats compose: **{gate['jit_beats_compose']}**",
        f"- Universal hit ($P_{{hit}}>0$ all $m^*$): **{gate['universal_hit_best']}** "
        f"({n_hit[best_arm]}/8)",
        "",
        gate["read"],
        "",
        r"$$\boxed{\text{intervene at decision sites; }P(S_{\mathrm{final}}=m^*)\text{ not retain}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
