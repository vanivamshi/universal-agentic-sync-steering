#!/usr/bin/env python3
"""Phase 9U (Arm B first) — continuous steering with target-specific gains.

Question: conditional on acquisition, does continuous steering with
per-target calibrated gains hold the target at least as well as the
unchanged 8K continuous configuration, with less intervention?

  A  continuous      C,H,O target-sign every round, locked 8K gains, forced FINAL:
  B  calibrated      same sites/signs/prefix, per-target gain multipliers

Protocol (per target, same conversation throughout retention):
  1. Acquire: round 0 under full 8K steering on many seeds; keep only
     conversations with S_0 = m*. Two DISJOINT seed pools: calibration, eval.
  2. Calibrate (cal pool only): one-factor-at-a-time gain multipliers, then
     the combination of per-channel winners. Every config is forked from the
     same acquired conversations with the same round seeds.
       C ∈ {0, 0.5, 1.5}   H ∈ {0, 0.5, 1.5}   O ∈ {0, 0.5}
     (O is never raised above α_O=1.5 — locked; scale 0 = channel not steered)
     Pick max P_hold; ties → lower total gain → base.
  3. Evaluate (held-out eval pool): A vs B paired on identical forks + seeds.

Primary: P_hold | acquired = P(S_1 = S_2 = m* | S_0 = m*). No recovery-after-
drift experiments: a flip is recorded as a failure only.

  .venv/bin/python -u scripts/run_phase9u_retention_gains.py --target 101
  .venv/bin/python -u scripts/run_phase9u_retention_gains.py --merge 100,101,110,111
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts import sync_adaptive_retention as sar  # noqa: E402

RESULTS = ROOT / "data" / "results"
SEED = 20261121
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
BASE = (1.0, 1.0, 1.0)
GRID = {"C": (0.0, 0.5, 1.5), "H": (0.0, 0.5, 1.5), "O": (0.0, 0.5)}


def out_paths(tag: str) -> tuple[Path, Path]:
    return (RESULTS / f"sync_phase9u_retention_gains{tag}.json",
            RESULTS / f"sync_phase9u_retention_gains{tag}.md")


def ofat_configs() -> list[tuple[float, float, float]]:
    cfgs = [BASE]
    for ch, vals in GRID.items():
        for v in vals:
            sc = list(BASE)
            sc[sar.CH_IDX[ch]] = v
            cfgs.append(tuple(sc))
    return cfgs


def total_gain(scale) -> float:
    return sum(sar.GAINS[ch] * scale[sar.CH_IDX[ch]] for ch in sar.CHANNELS)


def rank_key(st: dict[str, Any]):
    """Higher hold first; then less total gain; then base."""
    return (-st["P_hold"], st["total_gain"], 0 if tuple(st["scale"]) == BASE else 1)


def run_fixed(round_fn, msgs0, action: sar.Action, target, seeds) -> dict[str, Any]:
    """Same conversation, same action every round. A flip is just recorded."""
    msgs, S = msgs0, list(target)
    states, rounds = [list(S)], []
    for k, sd in enumerate(seeds, start=1):
        msgs, S_new, info = round_fn(msgs, action, list(target), sd)
        rounds.append({"k": k, "before": S, "after": S_new,
                       "collateral": list(sar.collateral(S, S_new, target)),
                       "action": action.name, "channels": list(action.channels), "info": info})
        S = list(S_new)
        states.append(S)
    return {"states": states, "path": [sar.key(s) for s in states], "rounds": rounds,
            "hold": all(s == list(target) for s in states[1:])}


def hold_stats(trials: list[dict], scale) -> dict[str, Any]:
    n = len(trials)
    k = sum(int(t["hold"]) for t in trials)
    first_flip = {}
    for t in trials:
        for r in t["rounds"]:
            if r["after"] != r["before"]:
                for ch in sar.identify_drift(r["after"], r["before"]):
                    first_flip[ch] = first_flip.get(ch, 0) + 1
                break
    return {"scale": list(scale), "n": n, "n_hold": k, "P_hold": (k / n) if n else 0.0,
            "total_gain": total_gain(scale), "first_flip_channels": first_flip,
            "paths": [t["path"] for t in trials]}


def acquire_pool(round_fn, target, *, seed0: int, need: int, max_tries: int, label: str):
    pool, tries = [], 0
    while len(pool) < need and tries < max_tries:
        sd = seed0 + 41 * tries
        tries += 1
        msgs0, S0, info = round_fn(None, sar.CONTINUOUS, list(target), sd)
        ok = S0 == list(target)
        print(f"  acquire[{label}] try={tries} seed={sd} S0={sar.key(S0)} {'KEEP' if ok else ''}", flush=True)
        if ok:
            pool.append({"seed": sd, "msgs0": msgs0})
    return pool, tries


def run_target(args) -> int:
    from scripts.run_adaptive_retention import _load_mod, make_round_fn
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    target = tuple(int(c) for c in args.target)
    ti = ALL_MSTAR.index(target)
    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    assert p8k._alphas() == sar.GAINS
    sc = p8k._load_sc()
    if args.sandbox:
        sc.WORKSPACE = Path(args.sandbox).resolve()
    sc.ensure_sandbox()
    loaded = load_model_and_tokenizer(LOCAL_MODEL_KEY, device_map=resolve_device_map(None),
                                      dtype="float32", local_files_only=True)
    round_fn = make_round_fn(sc, p8k, loaded, p8k._load_vc())
    out_json, out_md = out_paths(f"_t{args.target}")
    payload: dict[str, Any] = {"protocol": "9U Arm B: target-specific continuous gains vs 8K continuous",
                               "target": list(target), "seed": args.seed, "rounds": args.rounds,
                               "n_cal": args.n_cal, "n_eval": args.n_eval, "grid": GRID, "final": False}

    def save():
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def rseeds(s0):
        return [s0 + 7919 * k for k in range(1, args.rounds + 1)]

    base_seed = args.seed + 100000 * ti
    print(f"=== 9U target={args.target} n_cal={args.n_cal} n_eval={args.n_eval} ===", flush=True)
    cal, cal_tries = acquire_pool(round_fn, target, seed0=base_seed + 1, need=args.n_cal,
                                  max_tries=args.max_tries, label="cal")
    ev, ev_tries = acquire_pool(round_fn, target, seed0=base_seed + 50001, need=args.n_eval,
                                max_tries=args.max_tries, label="eval")
    payload["acquisition"] = {"cal": {"n_acq": len(cal), "tries": cal_tries},
                              "eval": {"n_acq": len(ev), "tries": ev_tries}}
    save()

    def evaluate(scale, pool):
        act = sar.scaled_continuous(scale)
        trials = [run_fixed(round_fn, c["msgs0"], act, target, rseeds(c["seed"])) | {"seed": c["seed"]}
                  for c in pool]
        st = hold_stats(trials, scale)
        print(f"    {act.name:26s} hold={st['n_hold']}/{st['n']} paths={st['paths']}", flush=True)
        return st, trials

    # --- calibration ---
    calib: dict[str, Any] = {}
    if cal:
        print("  -- calibrate (OFAT)", flush=True)
        for scale in ofat_configs():
            calib[str(scale)], _ = evaluate(scale, cal)
            payload["calibration"] = calib
            save()
        combo = list(BASE)
        for ch, vals in GRID.items():
            i = sar.CH_IDX[ch]
            cands = [calib[str(tuple(BASE))]] + [
                calib[str(tuple(BASE[:i]) + (v,) + tuple(BASE[i + 1:]))] for v in vals]
            combo[i] = sorted(cands, key=rank_key)[0]["scale"][i]
        combo = tuple(combo)
        if str(combo) not in calib:
            print("  -- calibrate (combined winners)", flush=True)
            calib[str(combo)], _ = evaluate(combo, cal)
        chosen = tuple(sorted(calib.values(), key=rank_key)[0]["scale"])
    else:
        chosen = BASE
    payload["calibration"] = calib
    payload["chosen_scale"] = list(chosen)
    payload["chosen_gains"] = sar.scaled_continuous(chosen).gains()
    save()
    print(f"  chosen scale={chosen} gains={payload['chosen_gains']}", flush=True)

    # --- held-out evaluation: A vs B on identical forks / seeds ---
    print("  -- evaluate (held-out)", flush=True)
    stA, trA = evaluate(BASE, ev)
    if chosen == BASE:
        stB, trB = stA, trA  # identical config ⇒ identical deterministic rollouts
    else:
        stB, trB = evaluate(chosen, ev)
    pairs = list(zip(trA, trB))
    payload["eval"] = {
        "A_continuous": stA, "B_calibrated": stB,
        "paired": {"n": len(pairs),
                   "B_only": sum(1 for a, b in pairs if b["hold"] and not a["hold"]),
                   "A_only": sum(1 for a, b in pairs if a["hold"] and not b["hold"])},
        "trials": {"A": trA, "B": trB},
    }
    payload["final"] = True
    save()
    write_md(out_md, [payload])
    print(json.dumps({"out": str(out_json), "chosen": chosen, "A": stA["P_hold"], "B": stB["P_hold"]}), flush=True)
    return 0


def fmt(x, nd=2):
    return "—" if x is None else f"{float(x):.{nd}f}"


def write_md(path: Path, parts: list[dict]) -> None:
    L = ["# Phase 9U — retention: target-specific continuous gains (Arm B) vs continuous (Arm A)", "",
         "> Same conversation; retention forked from acquired round-0 conversations (S_0 = m*). "
         "Calibration and evaluation use disjoint seed pools. Target-sign, 8K sites, forced FINAL:. "
         "α_O never raised.", "",
         f"rounds={parts[0]['rounds']}, n_cal={parts[0]['n_cal']}, n_eval={parts[0]['n_eval']}, seed={parts[0]['seed']}.", "",
         "## Held-out evaluation — $P(S_1{=}S_2{=}m^*\\mid S_0{=}m^*)$", "",
         "| $m^*$ | acq rate (eval) | chosen gains (C,H,O) | A continuous | B calibrated | B-only | A-only | total gain A→B |",
         "|---|---:|---|---:|---:|---:|---:|---:|"]
    tA = tB = nA = 0
    for p in parts:
        e = p.get("eval") or {}
        A, B = e.get("A_continuous"), e.get("B_calibrated")
        acq = p["acquisition"]["eval"]
        if not A:
            L.append(f"| `{sar.key(p['target'])}` | {acq['n_acq']}/{acq['tries']} | — | — | — | — | — | — |")
            continue
        g = p["chosen_gains"]
        gs = ", ".join(f"{ch}={g.get(ch, 0):g}" for ch in sar.CHANNELS)
        L.append(f"| `{sar.key(p['target'])}` | {acq['n_acq']}/{acq['tries']} | {gs} | "
                 f"{A['n_hold']}/{A['n']} | {B['n_hold']}/{B['n']} | {e['paired']['B_only']} | {e['paired']['A_only']} | "
                 f"{A['total_gain']:g}→{B['total_gain']:g} |")
        tA += A["n_hold"]; tB += B["n_hold"]; nA += A["n"]
    if nA:
        L += ["", f"**Pooled:** A {tA}/{nA} ({fmt(tA / nA)}) · B {tB}/{nA} ({fmt(tB / nA)})."]
    L += ["", "## Calibration (cal pool)", ""]
    for p in parts:
        if not p.get("calibration"):
            continue
        L += [f"### `{sar.key(p['target'])}` — cal acq {p['acquisition']['cal']['n_acq']}/{p['acquisition']['cal']['tries']}", "",
              "| scale (C,H,O) | hold | total gain | first-flip channels |", "|---|---:|---:|---|"]
        for st in sorted(p["calibration"].values(), key=rank_key):
            mark = " **←**" if st["scale"] == p["chosen_scale"] else ""
            L.append(f"| {tuple(st['scale'])}{mark} | {st['n_hold']}/{st['n']} | {st['total_gain']:g} | {st['first_flip_channels'] or '—'} |")
        L.append("")
    L += ["Selection on small cal pools is noisy (winner's curse); only the held-out A vs B comparison counts.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def merge(args) -> int:
    parts = [json.loads(out_paths(f"_t{t}")[0].read_text()) for t in args.merge.split(",")]
    out_json, out_md = out_paths("")
    out_json.write_text(json.dumps({"parts": parts}, indent=2, default=str), encoding="utf-8")
    write_md(out_md, parts)
    print(out_md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="e.g. 101")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-cal", type=int, default=6, help="acquired conversations for calibration")
    ap.add_argument("--n-eval", type=int, default=12, help="acquired conversations for held-out eval")
    ap.add_argument("--max-tries", type=int, default=60, help="round-0 attempts per pool")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--merge", default="", help="comma list of targets to merge")
    args = ap.parse_args()
    if args.merge:
        return merge(args)
    assert args.target and len(args.target) == 3, "--target required"
    return run_target(args)


if __name__ == "__main__":
    raise SystemExit(main())
