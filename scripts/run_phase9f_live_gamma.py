#!/usr/bin/env python3
"""Phase 9F — Live target-conditioned Γ action selection (scoped).

Frozen 8K actuators (sites/gains/v_c). Only the action-selection rule changes:

  a* = argmax_{a in relevant(s,m*)} Γ(a | s, m*)

where relevant = channels whose bit still differs from m*, and Γ is the
exact empirical estimate from the Phase 9E kernel (no weight tuning).

Compare vs fixed H→C→O (first differing bit), same single-channel step
mechanics, same seeds.

Scoped targets (kernel coverage; exclude 000/001/111 initially):
  010, 011, 100, 101, 110

  .venv/bin/python scripts/run_phase9f_live_gamma.py --reps 4
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

OUT = ROOT / "data" / "results" / "sync_phase9f_live_gamma.json"
MD = ROOT / "data" / "results" / "sync_phase9f_live_gamma.md"

SEED = 20260925
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
CHANNELS = ORDER

# Offline-supported targets only (exclude 000/001 and myopic-hard 111)
SCOPED_MSTAR = (
    (0, 1, 0),  # 010
    (0, 1, 1),  # 011
    (1, 0, 0),  # 100
    (1, 0, 1),  # 101
    (1, 1, 0),  # 110
)

MAX_STEPS = 4
MIN_N = 1


def _load_p8k():
    path = ROOT / "scripts" / "run_phase8k_c_gain_compose.py"
    spec = importlib.util.spec_from_file_location("phase8k", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_p9e():
    path = ROOT / "scripts" / "run_phase9e_gamma_planner.py"
    spec = importlib.util.spec_from_file_location("phase9e", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s: list[int] | tuple[int, ...] | str) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _E(mstar: tuple[int, int, int], S: list[int] | tuple[int, ...]) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _relevant(s: list[int] | tuple[int, ...], mstar: tuple[int, int, int]) -> list[str]:
    return [a for a in ORDER if int(s[CH_IDX[a]]) != int(mstar[CH_IDX[a]])]


def _fixed_action(s: list[int] | tuple[int, ...], mstar: tuple[int, int, int]) -> str | None:
    rel = _relevant(s, mstar)
    return rel[0] if rel else None


def build_gamma_lookup(p9e) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(s_key, a, mstar_key) -> {Gamma, n, mean_dE, ...}."""
    rows = p9e._load_transitions()
    ker = p9e._build_kernel(rows)
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for s in p9e.STATES:
        sk = _key(s)
        for a in CHANNELS:
            outs = ker.get((sk, a), [])
            for mstar in p9e.STATES:
                st = p9e._gamma_from_outcomes(s, outs, mstar)
                lookup[(sk, a, _key(mstar))] = st
    return lookup


def choose_gamma_action(
    s: list[int] | tuple[int, ...],
    mstar: tuple[int, int, int],
    lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """argmax Γ among relevant bits; fallback to fixed if no empirical support."""
    rel = _relevant(s, mstar)
    meta: dict[str, Any] = {"relevant": rel, "by_a": {}, "fallback": False}
    if not rel:
        return None, meta
    cands = []
    for a in rel:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "Gamma": None}
        meta["by_a"][a] = {"Gamma": st.get("Gamma"), "n": st.get("n"), "mean_dE": st.get("mean_dE")}
        if int(st.get("n") or 0) >= MIN_N and st.get("Gamma") is not None:
            cands.append((a, float(st["Gamma"]), int(st["n"])))
    if not cands:
        a = _fixed_action(s, mstar)
        meta["fallback"] = True
        meta["a_star"] = a
        return a, meta
    cands.sort(key=lambda t: (t[1], t[2], -ORDER.index(t[0])), reverse=True)
    a_star = cands[0][0]
    meta["a_star"] = a_star
    meta["Gamma_star"] = cands[0][1]
    return a_star, meta


def apply_channel(
    p8k,
    sc,
    loaded,
    *,
    dirs: dict,
    mstar: tuple[int, int, int],
    channel: str,
    seed: int,
    plan_prefill: str,
) -> dict[str, Any]:
    """Apply a single frozen 8K channel toward m* (non-cumulative)."""
    active = {channel: (p8k._s_star(mstar[CH_IDX[channel]]), dirs[channel])}
    h_pf = plan_prefill if channel == "H" else None
    out = p8k._run_episode(sc, loaded, active=active, seed=seed, h_plan_prefill=h_pf)
    return out


def run_policy_trial(
    p8k,
    sc,
    loaded,
    *,
    dirs: dict,
    lookup: dict,
    mstar: tuple[int, int, int],
    seed: int,
    policy: str,
    max_steps: int = MAX_STEPS,
    S0: list[int] | None = None,
    plan0: str | None = None,
) -> dict[str, Any]:
    """One live trial under fixed or gamma selection (shared optional free state)."""
    if S0 is None:
        torch.manual_seed(seed)
        free = p8k._run_episode(sc, loaded, active={}, seed=seed)
        S = list(free["S"])
        plan = free.get("plan_prefill") or ""
        n_ep = 1
    else:
        S = list(S0)
        plan = plan0 or ""
        n_ep = 0

    traj_S = [list(S)]
    traj_E = [_E(mstar, S)]
    steps = []
    for t in range(max_steps):
        if S == list(mstar):
            break
        if policy == "fixed":
            a = _fixed_action(S, mstar)
            meta = {"relevant": _relevant(S, mstar), "a_star": a, "fallback": False, "by_a": {}}
        elif policy == "gamma":
            a, meta = choose_gamma_action(S, mstar, lookup)
        else:
            raise ValueError(policy)
        if a is None:
            break
        Eb = _E(mstar, S)
        out = apply_channel(
            p8k, sc, loaded, dirs=dirs, mstar=mstar, channel=a, seed=seed + 1 + t, plan_prefill=plan
        )
        n_ep += 1
        Sa = list(out["S"])
        if out.get("plan_prefill"):
            plan = out["plan_prefill"]
        Ea = _E(mstar, Sa)
        step = {
            "t": t,
            "a": a,
            "policy": policy,
            "S_before": list(S),
            "S_after": Sa,
            "E_before": Eb,
            "E_after": Ea,
            "dE": Ea - Eb,
            "dS": int(Sa != S),
            "down": int(Ea < Eb),
            "up": int(Ea > Eb),
            "neutral": int(Ea == Eb),
            "hit_after": int(Sa == list(mstar)),
            "meta": meta,
            # did Γ-argmax (unrestricted among relevant) agree with chosen a?
            "a_gamma_unconstrained": choose_gamma_action(S, mstar, lookup)[0],
        }
        steps.append(step)
        S = Sa
        traj_S.append(list(S))
        traj_E.append(Ea)

    E0, Ef = traj_E[0], traj_E[-1]
    n_steps = len(steps)
    return {
        "policy": policy,
        "mstar": list(mstar),
        "mstar_key": _key(mstar),
        "seed": seed,
        "S0": traj_S[0],
        "S_final": traj_S[-1],
        "traj_S": traj_S,
        "traj_E": traj_E,
        "steps": steps,
        "hit": int(traj_S[-1] == list(mstar)),
        "delta_E": int(E0 - Ef),  # positive = progress (Hamming)
        "delta_E_norm": float(sync_error_norm(p8k._e(mstar, traj_S[0])) - sync_error_norm(p8k._e(mstar, traj_S[-1]))),
        "E0": E0,
        "E_final": Ef,
        "n_steps": n_steps,
        "n_episodes": n_ep + (0 if S0 is not None else 0),
        "P_down_steps": float(np.mean([s["down"] for s in steps])) if steps else float("nan"),
        "P_up_steps": float(np.mean([s["up"] for s in steps])) if steps else float("nan"),
        "P_shift_steps": float(np.mean([s["dS"] for s in steps])) if steps else float("nan"),
        "mean_step_dE": float(np.mean([s["dE"] for s in steps])) if steps else float("nan"),
        "actions": [s["a"] for s in steps],
        "n_fallback": sum(1 for s in steps if s.get("meta", {}).get("fallback")),
    }


def _agg(trials: list[dict]) -> dict[str, Any]:
    if not trials:
        return {"n": 0}
    def m(k):
        xs = [t[k] for t in trials if t.get(k) == t.get(k)]
        return float(np.mean(xs)) if xs else float("nan")
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "mean_delta_E": m("delta_E"),
        "mean_E_final": m("E_final"),
        "mean_E0": m("E0"),
        "mean_P_down_steps": m("P_down_steps"),
        "mean_P_up_steps": m("P_up_steps"),
        "mean_P_shift_steps": m("P_shift_steps"),
        "mean_step_dE": m("mean_step_dE"),
        "mean_n_steps": m("n_steps"),
        "mean_n_fallback": m("n_fallback"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_p8k()
    p9e = _load_p9e()
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()
    lookup = build_gamma_lookup(p9e)
    n_cells = sum(1 for v in lookup.values() if int(v.get("n") or 0) > 0)
    print(
        f"=== Phase 9F live Γ vs fixed | targets={[ _key(m) for m in SCOPED_MSTAR ]} "
        f"reps={args.reps} max_steps={args.max_steps} γ-cells={n_cells} ===",
        flush=True,
    )

    paired: list[dict[str, Any]] = []
    by_policy: dict[str, list] = {"gamma": [], "fixed": []}
    by_mstar: dict[str, dict[str, list]] = defaultdict(lambda: {"gamma": [], "fixed": []})

    idx = 0
    for mi, mstar in enumerate(SCOPED_MSTAR):
        for r in range(args.reps):
            seed0 = args.seed + 100 * mi + 17 * r
            # shared free state; re-roll if already at m* (vacuous planner test)
            S0 = None
            plan0 = ""
            seed = seed0
            for attempt in range(12):
                seed = seed0 + 31 * attempt
                torch.manual_seed(seed)
                free = p8k._run_episode(sc, loaded, active={}, seed=seed)
                S0 = list(free["S"])
                plan0 = free.get("plan_prefill") or ""
                if S0 != list(mstar):
                    break
            print(
                f"  [{idx+1}/{len(SCOPED_MSTAR)*args.reps}] m*={_key(mstar)} r={r} "
                f"S0={_key(S0)} E0={_E(mstar, S0)} seed={seed}",
                flush=True,
            )
            if S0 == list(mstar):
                print("    skip: could not obtain S0≠m* after re-rolls", flush=True)
                idx += 1
                continue
            tg = run_policy_trial(
                p8k, sc, loaded, dirs=dirs, lookup=lookup, mstar=mstar, seed=seed,
                policy="gamma", max_steps=args.max_steps, S0=S0, plan0=plan0,
            )
            tf = run_policy_trial(
                p8k, sc, loaded, dirs=dirs, lookup=lookup, mstar=mstar, seed=seed,
                policy="fixed", max_steps=args.max_steps, S0=S0, plan0=plan0,
            )
            tg["n_episodes"] = 1 + tg["n_steps"]
            tf["n_episodes"] = 1 + tf["n_steps"]
            print(
                f"    γ: hit={tg['hit']} E={tg['E0']}→{tg['E_final']} acts={tg['actions']} "
                f"path={[ _key(s) for s in tg['traj_S'] ]}",
                flush=True,
            )
            print(
                f"    f: hit={tf['hit']} E={tf['E0']}→{tf['E_final']} acts={tf['actions']} "
                f"path={[ _key(s) for s in tf['traj_S'] ]}",
                flush=True,
            )
            pair = {
                "mstar": list(mstar),
                "mstar_key": _key(mstar),
                "seed": seed,
                "S0": S0,
                "gamma": tg,
                "fixed": tf,
                "delta_hit": int(tg["hit"]) - int(tf["hit"]),
                "delta_delta_E": int(tg["delta_E"]) - int(tf["delta_E"]),
                "delta_E_final": int(tf["E_final"]) - int(tg["E_final"]),
            }
            paired.append(pair)
            by_policy["gamma"].append(tg)
            by_policy["fixed"].append(tf)
            by_mstar[_key(mstar)]["gamma"].append(tg)
            by_mstar[_key(mstar)]["fixed"].append(tf)
            idx += 1

    agg_g = _agg(by_policy["gamma"])
    agg_f = _agg(by_policy["fixed"])
    per_m = {
        mk: {"gamma": _agg(v["gamma"]), "fixed": _agg(v["fixed"]),
             "delta_P_hit": _agg(v["gamma"]).get("P_hit", 0) - _agg(v["fixed"]).get("P_hit", 0)}
        for mk, v in by_mstar.items()
    }

    gate = {
        "hypothesis": (
            "Live empirical Γ-argmax among remaining-error bits beats fixed H→C→O "
            "on scoped targets without new actuators"
        ),
        "scoped_targets": [_key(m) for m in SCOPED_MSTAR],
        "reps": args.reps,
        "P_hit_gamma": agg_g.get("P_hit"),
        "P_hit_fixed": agg_f.get("P_hit"),
        "delta_P_hit": (agg_g.get("P_hit") or 0) - (agg_f.get("P_hit") or 0),
        "mean_delta_E_gamma": agg_g.get("mean_delta_E"),
        "mean_delta_E_fixed": agg_f.get("mean_delta_E"),
        "mean_P_down_gamma": agg_g.get("mean_P_down_steps"),
        "mean_P_down_fixed": agg_f.get("mean_P_down_steps"),
        "n_pairs_gamma_hit_only": sum(1 for p in paired if p["gamma"]["hit"] and not p["fixed"]["hit"]),
        "n_pairs_fixed_hit_only": sum(1 for p in paired if p["fixed"]["hit"] and not p["gamma"]["hit"]),
        "n_pairs_both_hit": sum(1 for p in paired if p["gamma"]["hit"] and p["fixed"]["hit"]),
        "gamma_wins_delta_E": sum(1 for p in paired if p["delta_delta_E"] > 0),
        "fixed_wins_delta_E": sum(1 for p in paired if p["delta_delta_E"] < 0),
        "read": (
            "Scoped live test only. Controllers share free S0; differ only in "
            "single-channel action selection. Expand kernel before 000/001/111/full 8-way."
        ),
    }

    payload = {
        "protocol": "Phase 9F live Γ vs fixed (scoped)",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps": args.max_steps,
        "controller": {
            "frozen": "8K",
            "order_fixed": list(ORDER),
            "alpha": {"C": 5.0, "H": 1.5, "O": 1.5},
            "gamma": "empirical argmax among remaining-error bits; fallback=fixed",
            "no_weight_tuning": True,
        },
        "scoped_mstar": [_key(m) for m in SCOPED_MSTAR],
        "agg": {"gamma": agg_g, "fixed": agg_f},
        "per_mstar": per_m,
        "paired": paired,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9F — Live $\\Gamma$-policy vs fixed $H\\to C\\to O$ (scoped)",
        "",
        "> Frozen 8K actuators. Only action selection changes. "
        r"$a^*=\arg\max_{a\in\mathrm{relevant}(s,m^*)}\Gamma(a\mid s,m^*)$ (empirical, no tuning).",
        "",
        f"Targets: `{[_key(m) for m in SCOPED_MSTAR]}`. reps={args.reps}, "
        f"max_steps={args.max_steps}, seed={args.seed}. Paired free $S_0$.",
        "",
        "## Aggregate",
        "",
        "| policy | n | $P_{hit}$ | mean $\\Delta E$ | mean $E_f$ | "
        r"mean $P(E\downarrow)$ | mean step $\Delta E$ |",
        "|--------|--:|----------:|-----------------:|-----------:|----------------------:|----------------------:|",
        f"| **$\\Gamma$** | {agg_g['n']} | **{fmt(agg_g.get('P_hit'))}** | "
        f"**{fmt(agg_g.get('mean_delta_E'))}** | {fmt(agg_g.get('mean_E_final'))} | "
        f"{fmt(agg_g.get('mean_P_down_steps'))} | {fmt(agg_g.get('mean_step_dE'))} |",
        f"| fixed | {agg_f['n']} | {fmt(agg_f.get('P_hit'))} | "
        f"{fmt(agg_f.get('mean_delta_E'))} | {fmt(agg_f.get('mean_E_final'))} | "
        f"{fmt(agg_f.get('mean_P_down_steps'))} | {fmt(agg_f.get('mean_step_dE'))} |",
        "",
        f"$\\Delta P_{{hit}}(\\Gamma-\\mathrm{{fixed}})$ = **{fmt(gate['delta_P_hit'])}**",
        "",
        "## Per $m^*$",
        "",
        "| $m^*$ | $P_{hit}\\Gamma$ | $P_{hit}$ fixed | $\\Delta$ | mean $\\Delta E$ $\\Gamma$ | mean $\\Delta E$ fixed |",
        "|-------|----------------:|----------------:|---------:|------------------------:|-----------------------:|",
    ]
    for m in SCOPED_MSTAR:
        mk = _key(m)
        st = per_m[mk]
        lines.append(
            f"| `{mk}` | {fmt(st['gamma'].get('P_hit'))} | {fmt(st['fixed'].get('P_hit'))} | "
            f"**{fmt(st['delta_P_hit'])}** | {fmt(st['gamma'].get('mean_delta_E'))} | "
            f"{fmt(st['fixed'].get('mean_delta_E'))} |"
        )

    lines += [
        "",
        "## Example trajectories (first rep each $m^*$)",
        "",
    ]
    seen = set()
    for p in paired:
        mk = p["mstar_key"]
        if mk in seen:
            continue
        seen.add(mk)
        g, f = p["gamma"], p["fixed"]
        lines.append(
            f"- `{mk}` $S_0$=`{_key(p['S0'])}`: "
            f"$\\Gamma$ `{'→'.join(_key(s) for s in g['traj_S'])}` acts={g['actions']} "
            f"hit={g['hit']}; "
            f"fixed `{'→'.join(_key(s) for s in f['traj_S'])}` acts={f['actions']} "
            f"hit={f['hit']}"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- $P_{{hit}}$ $\\Gamma$ / fixed: **{fmt(gate['P_hit_gamma'])}** / {fmt(gate['P_hit_fixed'])}",
        f"- mean $\\Delta E$ $\\Gamma$ / fixed: **{fmt(gate['mean_delta_E_gamma'])}** / {fmt(gate['mean_delta_E_fixed'])}",
        f"- mean $P(E\\downarrow)$ $\\Gamma$ / fixed: {fmt(gate['mean_P_down_gamma'])} / {fmt(gate['mean_P_down_fixed'])}",
        f"- pairs $\\Gamma$-only / fixed-only / both hit: "
        f"{gate['n_pairs_gamma_hit_only']} / {gate['n_pairs_fixed_hit_only']} / {gate['n_pairs_both_hit']}",
        "",
        gate["read"],
        "",
        r"$$\boxed{\text{live }\Gamma\text{-selection vs fixed — scoped targets, frozen actuators}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
