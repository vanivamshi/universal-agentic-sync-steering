#!/usr/bin/env python3
"""Phase 9G — Targeted kernel expansion + anti-stagnation diagnostic.

1) Offline: on Phase 9F trajectories, test the safeguard

     Γ'(a) = -∞  if a == a_{t-1} and S_t == S_{t-1}
             Γ(a) otherwise

   among remaining-error bits. No coefficient tuning.

2) Live: densify failure (s,a) cells around 100/010/011, esp. 100|C stagnation.

3) Rebuild empirical kernel; recompute Γ; re-run offline anti-stag with expanded kernel.

No new v. No full 8-way yet.

  .venv/bin/python scripts/run_phase9g_kernel_expand.py --offline-only
  .venv/bin/python scripts/run_phase9g_kernel_expand.py --reps 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9g_kernel_expand.json"
MD = ROOT / "data" / "results" / "sync_phase9g_kernel_expand.md"
P9F = ROOT / "data" / "results" / "sync_phase9f_live_gamma.json"
KERNEL_OUT = ROOT / "data" / "results" / "sync_phase9g_kernel.json"

SEED = 20260926
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

# Failure neighborhood from 9F sticky analysis
EXPAND_STATES = ((1, 0, 0), (0, 1, 0), (0, 1, 1))  # 100, 010, 011
EXPAND_ACTIONS = ORDER


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s: list[int] | tuple[int, ...] | str) -> str:
    if isinstance(s, str):
        return s
    return "".join(str(int(x)) for x in s)


def _parse(s: str) -> tuple[int, int, int]:
    return (int(s[0]), int(s[1]), int(s[2]))


def _E(mstar, S) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _relevant(s, mstar) -> list[str]:
    return [a for a in ORDER if int(s[CH_IDX[a]]) != int(mstar[CH_IDX[a]])]


def build_lookup(p9e, extra_rows: list[dict] | None = None):
    rows = p9e._load_transitions()
    if extra_rows:
        rows = rows + extra_rows
    ker = p9e._build_kernel(rows)
    lookup = {}
    for s in p9e.STATES:
        sk = _key(s)
        for a in ORDER:
            outs = ker.get((sk, a), [])
            for mstar in p9e.STATES:
                lookup[(sk, a, _key(mstar))] = p9e._gamma_from_outcomes(s, outs, mstar)
    return lookup, ker, rows


def choose_action(
    s,
    mstar,
    lookup,
    *,
    a_prev: str | None,
    stagnated: bool,
    anti_stag: bool,
) -> tuple[str | None, dict[str, Any]]:
    rel = _relevant(s, mstar)
    meta: dict[str, Any] = {"relevant": rel, "by_a": {}, "blocked": None}
    if not rel:
        return None, meta
    cands = []
    for a in rel:
        st = lookup.get((_key(s), a, _key(mstar))) or {"n": 0, "Gamma": None}
        g = st.get("Gamma")
        meta["by_a"][a] = {"Gamma": g, "n": st.get("n"), "mean_dE": st.get("mean_dE")}
        if anti_stag and stagnated and a_prev is not None and a == a_prev:
            meta["blocked"] = a
            continue
        if int(st.get("n") or 0) >= 1 and g is not None:
            cands.append((a, float(g), int(st["n"])))
    if not cands:
        # fallback: first relevant not blocked
        for a in rel:
            if anti_stag and stagnated and a == a_prev:
                continue
            return a, {**meta, "fallback": True, "a_star": a}
        return rel[0], {**meta, "fallback": True, "a_star": rel[0], "forced_blocked": True}
    cands.sort(key=lambda t: (t[1], t[2], -ORDER.index(t[0])), reverse=True)
    a_star = cands[0][0]
    meta["a_star"] = a_star
    meta["Gamma_star"] = cands[0][1]
    return a_star, meta


def offline_antistag(p9e, lookup, paired: list[dict]) -> dict[str, Any]:
    """Replay 9F gamma trials; compare raw Γ vs anti-stagnation Γ'."""
    n_steps = 0
    n_disagree = 0
    n_blocked = 0
    n_sticky_raw = 0
    rescue = []  # cases where anti would break a no-op repeat
    mean_dE_raw = []
    mean_dE_anti = []
    path_rescues = []

    for p in paired:
        mstar = tuple(p["mstar"])
        g = p["gamma"]
        S = list(g["S0"])
        a_prev = None
        stagnated = False
        raw_acts = []
        anti_acts = []
        would_rescue = False
        for step in g["steps"]:
            n_steps += 1
            # what raw / anti would pick *at this state* (recompute)
            a_raw, meta_raw = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=False
            )
            a_anti, meta_anti = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=True
            )
            raw_acts.append(a_raw)
            anti_acts.append(a_anti)
            if a_raw != a_anti:
                n_disagree += 1
            if meta_anti.get("blocked"):
                n_blocked += 1
            # sticky: previous step was no-op and raw wants to repeat
            if stagnated and a_raw == a_prev:
                n_sticky_raw += 1
                if a_anti != a_raw:
                    would_rescue = True
                    # expected ΔE under kernel for both
                    def md(a):
                        st = lookup.get((_key(S), a, _key(mstar))) or {}
                        return st.get("mean_dE")

                    rescue.append(
                        {
                            "mstar": _key(mstar),
                            "s": _key(S),
                            "a_raw": a_raw,
                            "a_anti": a_anti,
                            "mean_dE_raw": md(a_raw),
                            "mean_dE_anti": md(a_anti),
                            "realized_a": step["a"],
                            "realized_dS": step["dS"],
                        }
                    )
            # advance using *realized* 9F transition (diagnostic, not re-sim)
            Sa = list(step["S_after"])
            stagnated = Sa == S
            a_prev = step["a"]  # history from realized policy
            # also track counterfactual expected dE of choices at this step
            for a_ch, bucket in ((a_raw, mean_dE_raw), (a_anti, mean_dE_anti)):
                st = lookup.get((_key(S), a_ch, _key(mstar))) if a_ch else None
                if st and st.get("mean_dE") is not None:
                    bucket.append(float(st["mean_dE"]))
            S = Sa

        if would_rescue and not g["hit"]:
            path_rescues.append(
                {
                    "mstar": _key(mstar),
                    "S0": _key(g["S0"]),
                    "realized_acts": g["actions"],
                    "raw_recomputed": raw_acts,
                    "anti_acts": anti_acts,
                    "hit_realized": g["hit"],
                    "path": [_key(s) for s in g["traj_S"]],
                }
            )

    # How many sticky CCCC-style failures get an alternative with better mean_dE?
    n_better = sum(
        1
        for r in rescue
        if r["mean_dE_anti"] is not None
        and r["mean_dE_raw"] is not None
        and r["mean_dE_anti"] < r["mean_dE_raw"]
    )

    # Also: when gamma missed & sticky, did anti pick the action fixed used successfully?
    n_align_fixed = 0
    n_sticky_miss = 0
    for p in paired:
        if p["gamma"]["hit"] or not p["fixed"]["hit"]:
            continue
        # gamma miss, fixed hit — check if sticky was involved
        g = p["gamma"]
        stagnated = False
        a_prev = None
        S = list(g["S0"])
        mstar = tuple(p["mstar"])
        for step in g["steps"]:
            a_raw, _ = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=False
            )
            a_anti, _ = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=True
            )
            if stagnated and a_raw == a_prev:
                n_sticky_miss += 1
                # fixed's first action from same S0 (paired)
                f_acts = p["fixed"]["actions"]
                if a_anti and f_acts and a_anti == f_acts[0]:
                    n_align_fixed += 1
                elif a_anti and a_anti != a_raw:
                    # anti switched away from failing repeat
                    n_align_fixed += 0  # counted via rescue
            Sa = list(step["S_after"])
            stagnated = Sa == S
            a_prev = step["a"]
            S = Sa

    return {
        "n_gamma_trials": len(paired),
        "n_steps": n_steps,
        "n_disagree_raw_vs_anti": n_disagree,
        "P_disagree": n_disagree / n_steps if n_steps else None,
        "n_blocked_applications": n_blocked,
        "n_sticky_raw_repeats": n_sticky_raw,
        "n_sticky_rescues": len(rescue),
        "n_rescue_better_mean_dE": n_better,
        "P_rescue_better_given_sticky": (n_better / len(rescue)) if rescue else None,
        "n_gamma_miss_fixed_hit": sum(
            1 for p in paired if (not p["gamma"]["hit"]) and p["fixed"]["hit"]
        ),
        "n_sticky_in_gamma_miss": n_sticky_miss,
        "mean_expected_dE_raw": float(np.mean(mean_dE_raw)) if mean_dE_raw else None,
        "mean_expected_dE_anti": float(np.mean(mean_dE_anti)) if mean_dE_anti else None,
        "rescue_examples": rescue[:20],
        "path_rescue_examples": path_rescues[:10],
        "verdict": (
            "anti-stagnation breaks sticky repeats (live-relevant)"
            if len(rescue) > 0
            else "no sticky rescues"
        ),
    }


def densify_cell(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    s_target: tuple[int, int, int],
    action: str,
    mstar_for_sign: tuple[int, int, int],
    reps: int,
    seed0: int,
) -> list[dict[str, Any]]:
    """Collect (s,a)->S_after with S constructed ≈ s_target via free+home."""
    p9 = _load_mod("phase9", ROOT / "scripts" / "run_phase9_hamming_paths.py")
    rows = []
    n_at = 0
    attempts = 0
    max_att = max(reps * 4, 16)
    while n_at < reps and attempts < max_att:
        seed = seed0 + 19 * attempts
        attempts += 1
        home = p9._home_to_source(
            p8k, sc, loaded, dirs=dirs, source=s_target, seed=seed
        )
        if not home["homed"]:
            # soft bridge for hard-ish states
            continue
        n_at += 1
        # apply single channel toward mstar_for_sign on that bit
        # For kernel expansion we want action `action` with target-sign for
        # flipping the action's bit toward a probe target that needs that flip.
        # Use mstar that differs from s only on `action` bit when possible.
        mstar = list(s_target)
        mstar[CH_IDX[action]] = 1 - int(s_target[CH_IDX[action]])
        mstar_t = tuple(mstar)
        active = {action: (p8k._s_star(mstar_t[CH_IDX[action]]), dirs[action])}
        h_pf = home["plan_prefill"] if action == "H" else None
        out = p8k._run_episode(
            sc, loaded, active=active, seed=seed + 7, h_plan_prefill=h_pf
        )
        Sa = tuple(out["S"])
        Sb = tuple(s_target)
        rows.append(
            {
                "src": "9G",
                "edge": f"{_key(Sb)}|{action}",
                "S_before": Sb,
                "S_after": Sa,
                "a": action,
                "to_intended": mstar_t,
                "dS": int(Sa != Sb),
                "seed": seed,
            }
        )
        print(
            f"    {_key(Sb)}|{action} n={n_at}/{reps} →{_key(Sa)} dS={int(Sa!=Sb)}",
            flush=True,
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=8, help="densify trials per (s,a)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--offline-only", action="store_true")
    args = ap.parse_args()

    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    paired = []
    if P9F.exists():
        paired = json.loads(P9F.read_text()).get("paired") or []

    lookup0, ker0, rows0 = build_lookup(p9e)
    print("=== Phase 9G offline anti-stagnation (pre-expand kernel) ===", flush=True)
    anti0 = offline_antistag(p9e, lookup0, paired)
    print(
        f"  sticky_raw={anti0['n_sticky_raw_repeats']} rescues={anti0['n_sticky_rescues']} "
        f"better_dE={anti0['n_rescue_better_mean_dE']} "
        f"P_disagree={anti0['P_disagree']}",
        flush=True,
    )

    extra_rows: list[dict] = []
    expand_stats: dict[str, Any] = {}

    if not args.offline_only:
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

        cells = [(s, a) for s in EXPAND_STATES for a in EXPAND_ACTIONS]
        print(
            f"=== Phase 9G densify {len(cells)} cells × reps={args.reps} ===",
            flush=True,
        )
        for i, (s, a) in enumerate(cells):
            print(f"  [{i+1}/{len(cells)}] densify {_key(s)}|{a}", flush=True)
            cell_rows = densify_cell(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                s_target=s,
                action=a,
                mstar_for_sign=s,
                reps=args.reps,
                seed0=args.seed + 1000 * i,
            )
            extra_rows.extend(cell_rows)
            n = len(cell_rows)
            n_shift = sum(r["dS"] for r in cell_rows)
            expand_stats[f"{_key(s)}|{a}"] = {
                "n": n,
                "P_shift": n_shift / n if n else None,
                "after_hist": dict(Counter(_key(r["S_after"]) for r in cell_rows)),
            }
            # checkpoint
            KERNEL_OUT.write_text(
                json.dumps(
                    {
                        "extra_rows": [
                            {**r, "S_before": list(r["S_before"]), "S_after": list(r["S_after"]),
                             "to_intended": list(r["to_intended"])}
                            for r in extra_rows
                        ],
                        "expand_stats": expand_stats,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # normalize extra for kernel merge
    extra_norm = []
    for r in extra_rows:
        extra_norm.append(
            {
                "src": "9G",
                "edge": r["edge"],
                "S_before": tuple(r["S_before"]),
                "S_after": tuple(r["S_after"]),
                "a": r["a"],
                "to_intended": tuple(r["to_intended"]),
            }
        )

    lookup1, ker1, rows1 = build_lookup(p9e, extra_norm if extra_norm else None)
    anti1 = offline_antistag(p9e, lookup1, paired) if paired else anti0

    # Focus cell: 100|C under m*=010
    focus = {}
    for label, lk in (("pre", lookup0), ("post", lookup1)):
        focus[label] = {
            a: lk.get(("100", a, "010"))
            for a in ORDER
        }

    gate = {
        "hypothesis": (
            "Sticky 9F failures are anti-stagnation / sparse-kernel issues, "
            "not missing actuators"
        ),
        "anti_stag_pre": {
            "n_sticky": anti0["n_sticky_raw_repeats"],
            "n_rescues": anti0["n_sticky_rescues"],
            "n_better": anti0["n_rescue_better_mean_dE"],
            "verdict": anti0["verdict"],
        },
        "anti_stag_post": {
            "n_sticky": anti1["n_sticky_raw_repeats"],
            "n_rescues": anti1["n_sticky_rescues"],
            "n_better": anti1["n_rescue_better_mean_dE"],
            "verdict": anti1["verdict"],
        },
        "n_extra_transitions": len(extra_norm),
        "expand_stats": expand_stats,
        "promising_for_live_antistag": bool(anti0["n_sticky_rescues"] > 0),
        "note_mean_dE": (
            "Pre-expand kernel can rate sticky C as good mean_dE; densify 100|C "
            "to test wrong-Γ vs anti-stag-needed."
        ),
        "read": (
            "Anti-stag switches C→H after no-op at 100 (matches fixed's successful "
            "first action). Densify sticky cells; then live Γ' before full 8-way."
        ),
    }

    payload = {
        "protocol": "Phase 9G kernel expand + anti-stagnation diagnostic",
        "seed": args.seed,
        "reps": args.reps,
        "offline_only": args.offline_only,
        "anti_stagnation_pre": anti0,
        "anti_stagnation_post": anti1,
        "focus_100_vs_010": focus,
        "expand_stats": expand_stats,
        "kernel_coverage_post": {
            f"{sk}|{a}": len(outs) for (sk, a), outs in sorted(ker1.items())
        },
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9G — Kernel expansion + anti-stagnation",
        "",
        "> No new $v$. Offline $\\Gamma'$ safeguard, then densify sticky $(s,a)$ around $100/010/011$.",
        "",
        r"$$\Gamma'(a)=-\infty \text{ if } a=a_{t-1}\land S_t=S_{t-1};\quad \Gamma(a)\text{ else.}$$",
        "",
        "## Offline anti-stagnation (on 9F gamma trajectories)",
        "",
        "| kernel | sticky raw repeats | rescues | better mean $\\Delta E$ | $P(\\mathrm{disagree})$ | verdict |",
        "|--------|-------------------:|--------:|------------------------:|-------------------------:|---------|",
        f"| pre-expand | {anti0['n_sticky_raw_repeats']} | {anti0['n_sticky_rescues']} | "
        f"{anti0['n_rescue_better_mean_dE']} | {fmt(anti0['P_disagree'])} | {anti0['verdict']} |",
        f"| post-expand | {anti1['n_sticky_raw_repeats']} | {anti1['n_sticky_rescues']} | "
        f"{anti1['n_rescue_better_mean_dE']} | {fmt(anti1['P_disagree'])} | {anti1['verdict']} |",
        "",
        "### Rescue examples (raw wanted to repeat after no-op)",
        "",
    ]
    for r in (anti0.get("rescue_examples") or [])[:8]:
        lines.append(
            f"- $m^*$=`{r['mstar']}` $s$=`{r['s']}`: raw=`{r['a_raw']}` → anti=`{r['a_anti']}` "
            f"(mean $\\Delta E$ {fmt(r['mean_dE_raw'])} → {fmt(r['mean_dE_anti'])})"
        )

    lines += [
        "",
        "## Focus: $\\Gamma(a\\mid 100, m^*=010)$",
        "",
        "| kernel | $\\Gamma_C$ | $\\Gamma_H$ | $\\Gamma_O$ | $a^*$ |",
        "|--------|-----------:|-----------:|-----------:|------|",
    ]
    for label in ("pre", "post"):
        blk = focus[label]
        def g(a):
            st = blk.get(a) or {}
            return fmt(st.get("Gamma")), st.get("n") or 0
        # argmax
        cands = []
        for a in ORDER:
            st = blk.get(a) or {}
            if st.get("Gamma") is not None and (st.get("n") or 0) >= 1:
                cands.append((a, st["Gamma"], st["n"]))
        a_star = max(cands, key=lambda t: (t[1], t[2]))[0] if cands else "—"
        gc, nc = g("C")
        gh, nh = g("H")
        go, no = g("O")
        lines.append(
            f"| {label} | {gc} (n={nc}) | {gh} (n={nh}) | {go} (n={no}) | **{a_star}** |"
        )

    if expand_stats:
        lines += [
            "",
            "## Densified cells",
            "",
            "| cell | n | $P_{shift}$ | after hist |",
            "|------|--:|------------:|------------|",
        ]
        for ck, st in sorted(expand_stats.items()):
            hist = " ".join(f"{k}:{v}" for k, v in list((st.get("after_hist") or {}).items())[:4])
            lines.append(
                f"| `{ck}` | {st['n']} | {fmt(st.get('P_shift'))} | {hist} |"
            )

    lines += [
        "",
        "## Gate",
        "",
        f"- Anti-stag promising for live: **{gate['promising_for_live_antistag']}**",
        f"- Extra transitions: {gate['n_extra_transitions']}",
        "",
        gate["read"],
        "",
        r"$$\boxed{\text{expand sticky kernel + test }\Gamma'\text{ before full 8-way}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
