#!/usr/bin/env python3
"""Phase 9K — Complete 000→111 via soft intermediaries 010 / 011.

9J: landings on 000 exit to 010 (H) or 011 (O), never direct 000→111.
Hypothesis: planned escape closes sink→111:

  000 --O--> 011 --Γ'--> 111
  000 --H--> 010 --Γ'--> 111

Protocol (frozen 8K, no new v):
1) Acquire 000 with sink-seek (λ=0, all-channels), keeping plan continuity.
2) Forced first escape action (O or H), then Γ' toward 111.
3) Pair both arms per rep by replaying acquisition via episode cache.

Primary: P(ever 111 | acquired 000). Secondary: intermediary 010 vs 011.

  .venv/bin/python -u scripts/run_phase9k_sink_to_111.py --reps 6
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

OUT = ROOT / "data" / "results" / "sync_phase9k_sink_to_111.json"
MD = ROOT / "data" / "results" / "sync_phase9k_sink_to_111.md"
SEED = 20260930
M000 = (0, 0, 0)
M111 = (1, 1, 1)
ORDER = ("H", "C", "O")
HARD_SET = {M000, (0, 0, 1), (1, 1, 0)}
ARMS = (
    ("via_011", "O", "011"),
    ("via_010", "H", "010"),
)


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


def _E(mstar, S) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def run_acq_then_escape(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    seed: int,
    S0: list[int],
    plan0: str,
    forced_a: str,
    max_steps_acq: int,
    max_steps_escape: int,
    cache: dict,
) -> dict[str, Any]:
    """One continuous chain: sink-seek →000, then forced_a + Γ' →111."""
    S = list(S0)
    plan = plan0 or ""
    traj_S = [list(S)]
    steps = []
    a_prev = None
    stagnated = False
    phase = "acq"
    land_t = None
    ever_000 = int(S == list(M000))
    ever_111_after_000 = 0  # only counts 111 reached in escape phase
    via_first = None
    escape_path = []

    if S == list(M000):
        land_t = -1
        phase = "escape"
        escape_path = ["000"]

    t = 0
    acq_steps_used = 0
    esc_steps_used = 0

    while True:
        if phase == "escape" and S == list(M111) and ever_000:
            ever_111_after_000 = 1
            break
        if phase == "acq" and acq_steps_used >= max_steps_acq:
            break
        if phase == "escape" and esc_steps_used >= max_steps_escape:
            break

        if phase == "acq":
            a, meta = p9j.choose_sink_seek(
                S, M000, lookup, a_prev=a_prev, stagnated=stagnated, lam=0.0
            )
            if a is None:
                break
            mstar = M000
            acq_steps_used += 1
        else:
            if esc_steps_used == 0:
                a = forced_a
                meta = {"mode": "forced_escape", "a_star": a}
            else:
                a, meta = p9j.choose_gamma_prime(
                    S, M111, lookup, a_prev=a_prev, stagnated=stagnated
                )
                if a is None:
                    break
            mstar = M111
            esc_steps_used += 1

        Eb = _E(mstar, S)
        out, hit_cache = p9j.apply_channel(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            mstar=mstar,
            channel=a,
            seed=seed + 1 + t,
            plan_prefill=plan,
            cache=cache,
        )
        Sa = list(out["S"])
        if out.get("plan_prefill"):
            plan = out["plan_prefill"]
        Ea = _E(mstar, Sa)
        steps.append(
            {
                "t": t,
                "phase": phase,
                "a": a,
                "mstar": _key(mstar),
                "S_before": list(S),
                "S_after": Sa,
                "E_before": Eb,
                "E_after": Ea,
                "dS": int(Sa != S),
                "mode": meta.get("mode"),
                "cache_hit": hit_cache,
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj_S.append(list(S))
        t += 1

        if phase == "escape":
            escape_path.append(_key(S))
            if via_first is None and _key(S) in ("010", "011"):
                via_first = _key(S)
            if S == list(M111):
                ever_111_after_000 = 1
                break

        if phase == "acq" and S == list(M000):
            ever_000 = 1
            land_t = t - 1
            phase = "escape"
            escape_path = ["000"]
            a_prev = None
            stagnated = False

    path = [_key(s) for s in traj_S]
    esc_keys = escape_path
    return {
        "forced_a": forced_a,
        "S0": list(S0),
        "S_final": list(S),
        "traj_S": traj_S,
        "path": path,
        "escape_path": esc_keys,
        "steps": steps,
        "actions": [s["a"] for s in steps],
        "acq_actions": [s["a"] for s in steps if s["phase"] == "acq"],
        "escape_actions": [s["a"] for s in steps if s["phase"] == "escape"],
        "ever_000": ever_000,
        "ever_111": ever_111_after_000,
        "hit_111": int(ever_000 and S == list(M111)),
        "land_t": land_t,
        "intermediary_first": via_first,
        "saw_010": int("010" in esc_keys),
        "saw_011": int("011" in esc_keys),
        "n_cache_hits": sum(1 for s in steps if s.get("cache_hit")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--max-steps-acq", type=int, default=7)
    ap.add_argument("--max-steps-escape", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")

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
    lookup, _, _ = p9j.build_lookup(p9e, p9g, None)

    paired = []
    by_arm: dict[str, list] = defaultdict(list)

    print(
        f"=== Phase 9K 000→(010|011)→111 reps={args.reps} ===",
        flush=True,
    )

    for r in range(args.reps):
        seed0 = args.seed + 200 * r
        S0, plan0, seed = p9j.free_S0(p8k, sc, loaded, mstar=M000, seed0=seed0)
        print(
            f"\n--- rep {r+1}/{args.reps} S0={_key(S0)} E0={_E(M000, S0)} ---",
            flush=True,
        )
        # shared cache: first arm pays for acq; second arm replays acq
        cache: dict = {}
        arms_out = {}
        for arm_name, forced_a, intended_via in ARMS:
            out = run_acq_then_escape(
                p9j,
                p8k,
                sc,
                loaded,
                dirs=dirs,
                lookup=lookup,
                seed=seed,
                S0=S0,
                plan0=plan0,
                forced_a=forced_a,
                max_steps_acq=args.max_steps_acq,
                max_steps_escape=args.max_steps_escape,
                cache=cache,
            )
            out["arm"] = arm_name
            out["intended_via"] = intended_via
            out["intended_via_hit"] = int(
                out.get("intermediary_first") == intended_via
                or (intended_via == "011" and out["saw_011"])
                or (intended_via == "010" and out["saw_010"])
            )
            arms_out[arm_name] = out
            by_arm[arm_name].append(out)
            print(
                f"  {arm_name}: ever000={out['ever_000']} ever111={out['ever_111']} "
                f"via={out['intermediary_first']} esc={out.get('escape_path')} "
                f"cache_hits={out['n_cache_hits']} "
                f"path={out['path']} acts={out['actions']}",
                flush=True,
            )
        paired.append({"rep": r, "seed": seed, "S0": S0, **arms_out})

    def agg(trials: list[dict]) -> dict[str, Any]:
        acq = [t for t in trials if t.get("ever_000")]
        if not trials:
            return {"n": 0}

        def m(rows, k):
            xs = [t[k] for t in rows if t.get(k) == t.get(k)]
            return float(np.mean(xs)) if xs else float("nan")

        return {
            "n": len(trials),
            "n_acq": len(acq),
            "P_acq_000": len(acq) / len(trials),
            "P_ever_111": m(trials, "ever_111"),
            "P_ever_111_given_acq": m(acq, "ever_111"),
            "P_hit_111_given_acq": m(acq, "hit_111"),
            "P_saw_010_given_acq": m(acq, "saw_010"),
            "P_saw_011_given_acq": m(acq, "saw_011"),
            "P_intended_via_given_acq": m(acq, "intended_via_hit"),
            "path_examples": [
                {
                    "path": t["path"],
                    "acts": t["actions"],
                    "ever000": t["ever_000"],
                    "ever111": t["ever_111"],
                }
                for t in trials
            ],
        }

    aggs = {name: agg(by_arm[name]) for name, _, _ in ARMS}
    any_ever = []
    for p in paired:
        acq_ok = any((p.get(n) or {}).get("ever_000") for n, _, _ in ARMS)
        if not acq_ok:
            continue
        any_ever.append(
            int(any((p.get(n) or {}).get("ever_111") for n, _, _ in ARMS))
        )

    gate = {
        "hypothesis": "Forced 000—O/H→010/011 then Γ' reaches 111 under frozen 8K",
        "n_reps": args.reps,
        "P_ever_111_given_acq_any_arm": float(np.mean(any_ever)) if any_ever else None,
        "n_reps_with_acq": len(any_ever),
        "per_arm": {
            n: {
                "P_acq_000": aggs[n].get("P_acq_000"),
                "P_ever_111_given_acq": aggs[n].get("P_ever_111_given_acq"),
                "P_hit_111_given_acq": aggs[n].get("P_hit_111_given_acq"),
            }
            for n, _, _ in ARMS
        },
        "path_complete": bool(
            any_ever and max((aggs[n].get("P_ever_111_given_acq") or 0) for n, _, _ in ARMS) > 0
        ),
        "read": "Primary: P(ever 111|acq 000)>0 on either arm completes the sink→111 path.",
    }

    payload = {
        "protocol": "Phase 9K 000→(010|011)→111",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps_acq": args.max_steps_acq,
        "max_steps_escape": args.max_steps_escape,
        "agg": aggs,
        "paired": paired,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9K — $000\\to(010\\mid 011)\\to 111$",
        "",
        "> Sink-seek acquire $000$, forced escape $O$/$H$, then $\\Gamma'$ to $111$. Frozen 8K.",
        "",
        f"reps={args.reps}, acq_steps={args.max_steps_acq}, "
        f"escape_steps={args.max_steps_escape}, seed={args.seed}.",
        "",
        "## Arms",
        "",
        "| arm | forced | $P_{acq}000$ | $P_{ever}111\\mid acq$ | $P_{hit}111\\mid acq$ | "
        "$P$(via) | $P$(010) | $P$(011) |",
        "|-----|--------|-------------:|----------------------:|---------------------:|"
        "---------:|---------:|---------:|",
    ]
    for name, forced, via in ARMS:
        a = aggs[name]
        lines.append(
            f"| `{name}` | `{forced}`→`{via}` | {fmt(a.get('P_acq_000'))} | "
            f"**{fmt(a.get('P_ever_111_given_acq'))}** | {fmt(a.get('P_hit_111_given_acq'))} | "
            f"{fmt(a.get('P_intended_via_given_acq'))} | "
            f"{fmt(a.get('P_saw_010_given_acq'))} | {fmt(a.get('P_saw_011_given_acq'))} |"
        )
    lines += ["", "### Paths", ""]
    for name, _, _ in ARMS:
        lines.append(f"**{name}**")
        for ex in aggs[name].get("path_examples") or []:
            lines.append(
                f"- 000={ex['ever000']} 111={ex['ever111']} "
                f"`{'→'.join(ex['path'])}` acts={ex['acts']}"
            )
        lines.append("")
    lines += [
        "## Gate",
        "",
        f"- Path complete: **{gate['path_complete']}**",
        f"- $P(\\mathrm{{ever}}111\\mid\\mathrm{{acq}},\\mathrm{{any\\ arm}})$: "
        f"**{fmt(gate['P_ever_111_given_acq_any_arm'])}** "
        f"(n_acq_reps={gate['n_reps_with_acq']})",
        "",
        gate["read"],
        "",
        r"$$\boxed{000\xrightarrow{O/H}010/011\xrightarrow{\Gamma'}111}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
