#!/usr/bin/env python3
"""Phase 9V — held-out paired validation of FROZEN 9U calibrated retention gains.

Question: on new conversations, does continuous steering with the per-target
gains selected in 9U (Arm B) retain an acquired target as reliably as the full
8K continuous configuration (Arm A), with less intervention?

No calibration happens here. Arm B's gains are read from
data/configs/phase9v_frozen_gains.json (written once by --freeze from the 9U
per-target JSONs; source SHA-256s are re-verified on every run).

Protocol (per target, same conversation throughout retention):
  1. Acquire: round 0 under full 8K steering on NEW seeds (disjoint from every
     9U seed range — asserted). Every attempt is recorded (acquisition rate).
  2. For each conversation with S_0 = m*: fork it to A and B; both arms use the
     identical round seeds. A = FULL_8K continuous, B = frozen CAL9U_<m*>.
  3. Primary:  P(S_1 = S_2 = m* | S_0 = m*) per arm, Wilson 95% CI; paired
     discordance (B-only / A-only) with exact McNemar.
     Also: acquisition rate P(S_0 = m*), end-to-end P(S_0=S_1=S_2=m*) per arm.
  A flip is recorded as a failure only (no recovery).

Defaults prioritise 100 and 101 (n_acq=40), then 110 (large try budget; 9U
acquired 2/60), then 111.

  python -u scripts/run_phase9v_heldout_retention.py --freeze
  python -u scripts/run_phase9v_heldout_retention.py --target 100     # resumable
  python -u scripts/run_phase9v_heldout_retention.py --merge 100,101,110,111
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
from scripts import sync_retention_controller as rc  # noqa: E402

RESULTS = ROOT / "data" / "results"
FROZEN = ROOT / "data" / "configs" / "phase9v_frozen_gains.json"
TARGETS = ("100", "101", "110", "111")
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]

# New seed base, far from 9U (20261121 + 100000*ti + {1, 50001} + 41*tries + 7919*k).
SEED = 31000000
SEED_STRIDE = 41
ROUND_STRIDE = 7919
TARGET_STRIDE = 100000
# Per-target defaults: (n_acq wanted, max round-0 attempts).
DEFAULTS = {"100": (40, 200), "101": (40, 400), "110": (20, 1200), "111": (24, 300)}

# Conservative envelope of every seed 9U could have used (8 targets, up to 2000 tries/pool).
_9U_LO = 20261121
_9U_HI = 20261121 + TARGET_STRIDE * 7 + 50001 + SEED_STRIDE * 2000 + ROUND_STRIDE * 10


def out_paths(tag: str) -> tuple[Path, Path]:
    return (RESULTS / f"sync_phase9v_heldout_retention{tag}.json",
            RESULTS / f"sync_phase9v_heldout_retention{tag}.md")


def acq_seed(seed: int, ti: int, i: int) -> int:
    return seed + TARGET_STRIDE * ti + SEED_STRIDE * i


def round_seeds(s0: int, rounds: int) -> list[int]:
    return [s0 + ROUND_STRIDE * k for k in range(1, rounds + 1)]


def assert_disjoint_from_9u(seed: int, ti: int, max_tries: int, rounds: int, cfg: dict) -> None:
    lo = acq_seed(seed, ti, 0)
    hi = round_seeds(acq_seed(seed, ti, max_tries - 1), rounds)[-1]
    assert hi < _9U_LO or lo > _9U_HI, f"9V seeds [{lo},{hi}] overlap 9U envelope [{_9U_LO},{_9U_HI}]"
    assert SEED_STRIDE * max_tries + ROUND_STRIDE * rounds < TARGET_STRIDE, "seed ranges of targets would overlap"
    used = {s for t in cfg["targets"].values() for s in t["eval_seeds_9u"]}
    mine = {acq_seed(seed, ti, i) for i in range(max_tries)}
    assert not (used & mine), "9V acquisition seed reused from 9U eval"


# --------------------------------------------------------------------------- freeze

def freeze(args) -> int:
    if FROZEN.exists() and not args.force_freeze:
        print(f"{FROZEN} already exists — gains are frozen. Refusing to overwrite (use --force-freeze).")
        return 1
    srcs = {t: RESULTS / f"sync_phase9u_retention_gains_t{t}.json" for t in TARGETS}
    cfg = rc.freeze_from_9u(srcs, ROOT)
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for t, v in cfg["targets"].items():
        print(f"  {t}: scale={v['scale']} gains={v['gains']} total={v['total_gain']:g}")
    print(FROZEN)
    return 0


# --------------------------------------------------------------------------- run

def run_target(args) -> int:
    from scripts.run_adaptive_retention import _load_mod, make_round_fn
    from scripts.run_phase9u_retention_gains import run_fixed
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    tkey = args.target
    target = tuple(int(c) for c in tkey)
    ti = ALL_MSTAR.index(target)
    cfg = rc.load_frozen(FROZEN, ROOT)
    actB = rc.retention_action(cfg, tkey)
    actA = sar.CONTINUOUS
    n_need, max_tries = DEFAULTS[tkey]
    n_need = args.n_acq or n_need
    max_tries = args.max_tries or max_tries
    assert_disjoint_from_9u(args.seed, ti, max_tries, args.rounds, cfg)

    out_json, out_md = out_paths(f"_t{tkey}")
    payload: dict[str, Any] = {
        "protocol": "9V held-out paired: A=FULL_8K continuous vs B=frozen 9U calibrated continuous",
        "target": list(target), "seed": args.seed, "rounds": args.rounds,
        "n_acq_wanted": n_need, "max_tries": max_tries,
        "frozen_config": str(FROZEN.relative_to(ROOT)),
        "frozen_source_sha256": cfg["targets"][tkey]["source_sha256"],
        "arms": {"A": {"name": actA.name, "gains": actA.gains()},
                 "B": {"name": actB.name, "gains": actB.gains(), "scale": cfg["targets"][tkey]["scale"]}},
        "attempts": [], "trials": [], "final": False,
    }
    if args.resume and out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        assert prev["arms"] == payload["arms"] and prev["seed"] == args.seed and prev["rounds"] == args.rounds, \
            "resume: config differs from the saved run"
        payload["attempts"], payload["trials"] = prev["attempts"], prev["trials"]
        print(f"resume: {len(payload['attempts'])} attempts, {len(payload['trials'])} paired trials", flush=True)

    def save():
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    assert p8k._alphas() == sar.GAINS
    sc = p8k._load_sc()
    if args.sandbox:
        sc.WORKSPACE = Path(args.sandbox).resolve()
    sc.ensure_sandbox()
    loaded = load_model_and_tokenizer(LOCAL_MODEL_KEY, device_map=resolve_device_map(args.device or None),
                                      dtype="float32", local_files_only=args.local_files_only)
    round_fn = make_round_fn(sc, p8k, loaded, p8k._load_vc())

    print(f"=== 9V target={tkey} A={actA.gains()} B={actB.gains()} need={n_need} max_tries={max_tries} ===", flush=True)
    i = len(payload["attempts"])
    while len(payload["trials"]) < n_need and i < max_tries:
        sd = acq_seed(args.seed, ti, i)
        i += 1
        msgs0, S0, _ = round_fn(None, sar.CONTINUOUS, list(target), sd)
        ok = S0 == list(target)
        payload["attempts"].append({"i": i - 1, "seed": sd, "S0": list(S0), "acquired": ok})
        print(f"  acquire try={i} seed={sd} S0={sar.key(S0)} {'KEEP' if ok else ''}", flush=True)
        if ok:
            rs = round_seeds(sd, args.rounds)
            a = run_fixed(round_fn, msgs0, actA, target, rs)
            b = run_fixed(round_fn, msgs0, actB, target, rs)
            payload["trials"].append({"seed": sd, "round_seeds": rs, "A": a, "B": b})
            print(f"    A {a['path']} hold={a['hold']} | B {b['path']} hold={b['hold']}", flush=True)
        save()

    payload["summary"] = summarize(payload)
    payload["final"] = len(payload["trials"]) >= n_need or i >= max_tries
    save()
    write_md(out_md, [payload])
    print(json.dumps({"out": str(out_json), **{k: payload["summary"][k] for k in ("n_acq", "tries", "A", "B")}},
                     default=str), flush=True)
    return 0


# --------------------------------------------------------------------------- analysis

def summarize(p: dict[str, Any]) -> dict[str, Any]:
    tries = len(p["attempts"])
    trials = p["trials"]
    n = len(trials)
    s: dict[str, Any] = {"tries": tries, "n_acq": n,
                         "P_acq": (n / tries) if tries else None, "P_acq_ci": rc.wilson(n, tries)}
    for arm in ("A", "B"):
        k = sum(int(t[arm]["hold"]) for t in trials)
        s[arm] = {"n_hold": k, "n": n, "P_hold": (k / n) if n else None, "ci": rc.wilson(k, n),
                  "P_e2e": (k / tries) if tries else None, "e2e_ci": rc.wilson(k, tries),
                  "total_gain": sum(p["arms"][arm]["gains"].values())}
    b_only = sum(1 for t in trials if t["B"]["hold"] and not t["A"]["hold"])
    a_only = sum(1 for t in trials if t["A"]["hold"] and not t["B"]["hold"])
    s["paired"] = {"n": n, "B_only": b_only, "A_only": a_only, "both": sum(1 for t in trials if t["A"]["hold"] and t["B"]["hold"]),
                   "diff_B_minus_A": ((b_only - a_only) / n) if n else None,
                   "mcnemar_p": rc.mcnemar_exact(b_only, a_only)}
    return s


def _ci(ci) -> str:
    lo, hi = ci
    return "—" if lo is None else f"[{lo:.2f}, {hi:.2f}]"


def _p(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def write_md(path: Path, parts: list[dict]) -> None:
    p0 = parts[0]
    L = ["# Phase 9V — held-out paired validation of frozen 9U retention gains", "",
         "> New seeds (disjoint from 9U). Gains frozen from 9U, no tuning. Same conversation; A and B forked from the "
         "same acquired round-0 conversation with identical round seeds. Target-sign, 8K sites, forced FINAL:.", "",
         f"rounds={p0['rounds']}, seed={p0['seed']}, frozen config `{p0['frozen_config']}`.", "",
         "## Primary — $P(S_1{=}S_2{=}m^*\\mid S_0{=}m^*)$", "",
         "| $m^*$ | B gains | A hold | A 95% CI | B hold | B 95% CI | B-only | A-only | McNemar p | gain A→B |",
         "|---|---|---:|---|---:|---|---:|---:|---:|---:|"]
    pool = {"A": 0, "B": 0, "n": 0, "bo": 0, "ao": 0}
    for p in parts:
        s = p.get("summary") or summarize(p)
        g = ", ".join(f"{k}={v:g}" for k, v in p["arms"]["B"]["gains"].items())
        A, B, pr = s["A"], s["B"], s["paired"]
        L.append(f"| `{sar.key(p['target'])}` | {g} | {A['n_hold']}/{A['n']} | {_ci(A['ci'])} | {B['n_hold']}/{B['n']} | "
                 f"{_ci(B['ci'])} | {pr['B_only']} | {pr['A_only']} | {pr['mcnemar_p']:.3f} | "
                 f"{A['total_gain']:g}→{B['total_gain']:g} |")
        pool["A"] += A["n_hold"]; pool["B"] += B["n_hold"]; pool["n"] += A["n"]
        pool["bo"] += pr["B_only"]; pool["ao"] += pr["A_only"]
    if len(parts) > 1 and pool["n"]:
        L += ["", f"**Pooled:** A {pool['A']}/{pool['n']} {_ci(rc.wilson(pool['A'], pool['n']))} · "
                  f"B {pool['B']}/{pool['n']} {_ci(rc.wilson(pool['B'], pool['n']))} · "
                  f"discordant B-only {pool['bo']} / A-only {pool['ao']} (McNemar p={rc.mcnemar_exact(pool['bo'], pool['ao']):.3f})."]
    L += ["", "## Acquisition and end-to-end", "",
          "| $m^*$ | acq | $P(S_0{=}m^*)$ | 95% CI | e2e A | e2e B | complete |", "|---|---:|---:|---|---:|---:|---|"]
    for p in parts:
        s = p.get("summary") or summarize(p)
        L.append(f"| `{sar.key(p['target'])}` | {s['n_acq']}/{s['tries']} | {_p(s['P_acq'])} | {_ci(s['P_acq_ci'])} | "
                 f"{_p(s['A']['P_e2e'])} | {_p(s['B']['P_e2e'])} | {'yes' if p.get('final') else 'partial'} |")
    L += ["", "Gains were selected in 9U on n=2–6 calibration conversations; 9V is the first test on fresh seeds. "
              "Equal hold rates with overlapping CIs mean *not distinguishable at this n*, not proven equivalence.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def merge(args) -> int:
    parts = []
    for t in args.merge.split(","):
        p = json.loads(out_paths(f"_t{t}")[0].read_text(encoding="utf-8"))
        p["summary"] = summarize(p)
        parts.append(p)
    out_json, out_md = out_paths("")
    out_json.write_text(json.dumps({"parts": parts}, indent=2, default=str), encoding="utf-8")
    write_md(out_md, parts)
    print(out_md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true", help="write the frozen-gains config from 9U (once)")
    ap.add_argument("--force-freeze", action="store_true")
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-acq", type=int, default=0, help="override per-target default")
    ap.add_argument("--max-tries", type=int, default=0, help="override per-target default")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--merge", default="")
    args = ap.parse_args()
    if args.freeze:
        return freeze(args)
    if args.merge:
        return merge(args)
    assert args.target, "--target, --freeze or --merge required"
    return run_target(args)


if __name__ == "__main__":
    raise SystemExit(main())
