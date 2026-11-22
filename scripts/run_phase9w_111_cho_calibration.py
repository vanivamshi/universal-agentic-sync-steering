#!/usr/bin/env python3
"""Phase 9W — targeted C/H/O calibration for retaining target 111.

9V left 111 unsolved: full 8K holds 15/24 and the frozen 9U configuration
(C=0, H=1.5, O=1.5) holds 18/24. The two fail differently:
  * full 8K (C=5): every failure flips C alone          (111 -> 011)
  * calibrated (C=0): 5/6 failures flip C and H together (111 -> 001)
So 9W asks whether a *weak* C gain, or lower H/O gains, stops C from flipping.
It does not raise C toward 5, because 9V suggests strong C is counterproductive.

Arms (all continuous, target-sign, forced FINAL:, same conversation):
  A  C=0    H=1.5   O=1.5    frozen 9V configuration for 111 (control)
  B  C=0.5  H=1.5   O=1.5    weak C support
  C  C=1.0  H=1.5   O=1.5    moderate C support
  D  C=0    H=0.75  O=0.75   lower H/O gains

Protocol:
  1. Acquire: round 0 under full 8K steering on NEW seeds (disjoint from 9U and 9V;
     asserted). Every attempt is recorded.
  2. Each conversation with S_0 = 111 is forked to A, B, C, D with identical round seeds.
  3. Primary: P(S_1 = S_2 = 111 | S_0 = 111) per arm, Wilson 95% CI.
     Paired vs A (B-only / A-only, exact McNemar, Holm across the 3 comparisons).
  4. Failure anatomy per arm: final state, round of first failure, which bits flipped
     in that round (C alone vs C with H, ...), and any bit that flipped back.
  A flip counts as a failure only; nothing repairs it.

  python -u scripts/run_phase9w_111_cho_calibration.py              # resumable with --resume
  python -u scripts/run_phase9w_111_cho_calibration.py --report     # re-render md from json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts import sync_adaptive_retention as sar  # noqa: E402
from scripts import sync_retention_controller as rc  # noqa: E402

RESULTS = ROOT / "data" / "results"
FROZEN_9V = ROOT / "data" / "configs" / "phase9v_frozen_gains.json"
OUT_JSON = RESULTS / "sync_phase9w_111_cho_calibration.json"
OUT_MD = RESULTS / "sync_phase9w_111_cho_calibration.md"

TARGET = (1, 1, 1)
TKEY = "111"
TI = 7  # index of 111 in the C,H,O enumeration (keeps 9V's per-target seed layout)

# Gain multipliers on the locked 8K gains (C=5, H=1.5, O=1.5).
ARM_SCALES: dict[str, tuple[float, float, float]] = {
    "A": (0.0, 1.0, 1.0),   # C=0    H=1.5  O=1.5
    "B": (0.1, 1.0, 1.0),   # C=0.5  H=1.5  O=1.5
    "C": (0.2, 1.0, 1.0),   # C=1.0  H=1.5  O=1.5
    "D": (0.0, 0.5, 0.5),   # C=0    H=0.75 O=0.75
}
ARM_PURPOSE = {
    "A": "frozen 9V baseline",
    "B": "weak C support",
    "C": "moderate C support",
    "D": "lower H/O gains",
}
ARMS = tuple(ARM_SCALES)
CONTROL = "A"

SEED = 41000000
SEED_STRIDE = 41
ROUND_STRIDE = 7919
TARGET_STRIDE = 100000
N_ACQ = 40
MAX_TRIES = 400

# Seed envelopes of earlier phases (conservative).
_9U_LO, _9U_HI = 20261121, 20261121 + TARGET_STRIDE * 7 + 50001 + SEED_STRIDE * 2000 + ROUND_STRIDE * 10
_9V_LO, _9V_HI = 31000000, 31000000 + TARGET_STRIDE * 7 + SEED_STRIDE * 1200 + ROUND_STRIDE * 10


def arm_action(arm: str) -> sar.Action:
    return sar.scaled_continuous(ARM_SCALES[arm], name=f"9W_{arm}")


def acq_seed(seed: int, i: int) -> int:
    return seed + TARGET_STRIDE * TI + SEED_STRIDE * i


def round_seeds(s0: int, rounds: int) -> list[int]:
    return [s0 + ROUND_STRIDE * k for k in range(1, rounds + 1)]


def assert_new_seeds(seed: int, max_tries: int, rounds: int) -> None:
    lo = acq_seed(seed, 0)
    hi = round_seeds(acq_seed(seed, max_tries - 1), rounds)[-1]
    for name, a, b in (("9U", _9U_LO, _9U_HI), ("9V", _9V_LO, _9V_HI)):
        assert hi < a or lo > b, f"9W seeds [{lo},{hi}] overlap {name} envelope [{a},{b}]"


def assert_control_is_frozen_9v() -> dict[str, Any]:
    """Arm A must be exactly the 9V-frozen 111 configuration (SHA-checked)."""
    cfg = rc.load_frozen(FROZEN_9V, ROOT)
    frozen = rc.retention_action(cfg, TKEY)
    a = arm_action(CONTROL)
    assert a.gains() == frozen.gains() and a.channels == frozen.channels, (a.gains(), frozen.gains())
    return cfg


# --------------------------------------------------------------------------- run

def run(args) -> int:
    from scripts.run_adaptive_retention import _load_mod, make_round_fn
    from scripts.run_phase9u_retention_gains import run_fixed
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    cfg = assert_control_is_frozen_9v()
    assert_new_seeds(args.seed, args.max_tries, args.rounds)
    actions = {a: arm_action(a) for a in ARMS}

    payload: dict[str, Any] = {
        "protocol": "9W paired C/H/O calibration for 111: A=frozen 9V vs B,C,D variants",
        "target": list(TARGET), "seed": args.seed, "rounds": args.rounds,
        "n_acq_wanted": args.n_acq, "max_tries": args.max_tries,
        "control": CONTROL, "frozen_9v_config": str(FROZEN_9V.relative_to(ROOT)),
        "frozen_9v_source_sha256": cfg["targets"][TKEY]["source_sha256"],
        "arms": {a: {"name": act.name, "scale": list(ARM_SCALES[a]), "gains": act.gains(),
                     "purpose": ARM_PURPOSE[a]} for a, act in actions.items()},
        "attempts": [], "trials": [], "final": False,
    }
    if args.resume and OUT_JSON.exists():
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        assert prev["arms"] == payload["arms"] and prev["seed"] == args.seed and prev["rounds"] == args.rounds, \
            "resume: config differs from the saved run"
        payload["attempts"], payload["trials"] = prev["attempts"], prev["trials"]
        print(f"resume: {len(payload['attempts'])} attempts, {len(payload['trials'])} trials", flush=True)

    def save():
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    assert p8k._alphas() == sar.GAINS
    sc = p8k._load_sc()
    if args.sandbox:
        sc.WORKSPACE = Path(args.sandbox).resolve()
    sc.ensure_sandbox()
    loaded = load_model_and_tokenizer(LOCAL_MODEL_KEY, device_map=resolve_device_map(args.device or None),
                                      dtype="float32", local_files_only=args.local_files_only)
    round_fn = make_round_fn(sc, p8k, loaded, p8k._load_vc())

    print(f"=== 9W target=111 arms={ {a: actions[a].gains() for a in ARMS} } "
          f"need={args.n_acq} max_tries={args.max_tries} ===", flush=True)
    i = len(payload["attempts"])
    while len(payload["trials"]) < args.n_acq and i < args.max_tries:
        sd = acq_seed(args.seed, i)
        i += 1
        msgs0, S0, _ = round_fn(None, sar.CONTINUOUS, list(TARGET), sd)
        ok = S0 == list(TARGET)
        payload["attempts"].append({"i": i - 1, "seed": sd, "S0": list(S0), "acquired": ok})
        print(f"  acquire try={i} seed={sd} S0={sar.key(S0)} {'KEEP' if ok else ''}", flush=True)
        if ok:
            rs = round_seeds(sd, args.rounds)
            trial: dict[str, Any] = {"seed": sd, "round_seeds": rs}
            for a in ARMS:
                trial[a] = run_fixed(round_fn, msgs0, actions[a], TARGET, rs)
            payload["trials"].append(trial)
            print("    " + " | ".join(f"{a} {'>'.join(trial[a]['path'])} {'hold' if trial[a]['hold'] else 'FAIL'}"
                                      for a in ARMS), flush=True)
        save()

    payload["summary"] = summarize(payload)
    payload["final"] = len(payload["trials"]) >= args.n_acq or i >= args.max_tries
    save()
    write_md(OUT_MD, payload)
    print(json.dumps({"out": str(OUT_JSON), "n_acq": payload["summary"]["n_acq"],
                      **{a: payload["summary"]["arms"][a]["n_hold"] for a in ARMS}}), flush=True)
    return 0


# --------------------------------------------------------------------------- analysis

def flip_label(before: list[int], after: list[int]) -> str:
    """Bits that changed between two states, e.g. 'C', 'C+H', or '' if none."""
    return "+".join(ch for ch in sar.CHANNELS
                    if int(before[sar.CH_IDX[ch]]) != int(after[sar.CH_IDX[ch]]))


def failure_anatomy(res: dict[str, Any]) -> dict[str, Any] | None:
    """Describe how one arm's trial failed; None if it held."""
    if res["hold"]:
        return None
    states = res["states"]
    first = next(k for k in range(1, len(states)) if states[k] != list(TARGET))
    lost_any = {ch for s in states[1:] for ch in sar.identify_drift(s, TARGET)}
    reverted = sorted(ch for ch in lost_any if ch not in sar.identify_drift(states[-1], TARGET))
    return {
        "final": sar.key(states[-1]),
        "first_fail_round": first,
        "first_flip": flip_label(states[first - 1], states[first]),
        # bits that were correct entering a later round and wrong after it
        "later_collateral": [ch for k in range(first + 1, len(states))
                             for ch in sar.collateral(states[k - 1], states[k], TARGET)],
        "reverted": reverted,
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for r, k in enumerate(order):
        running = max(running, min(1.0, (m - r) * pvals[k]))
        out[k] = running
    return out


def summarize(p: dict[str, Any]) -> dict[str, Any]:
    tries, trials = len(p["attempts"]), p["trials"]
    n = len(trials)
    s: dict[str, Any] = {"tries": tries, "n_acq": n, "P_acq": (n / tries) if tries else None,
                         "P_acq_ci": rc.wilson(n, tries), "arms": {}, "paired_vs_control": {}}
    for a in p["arms"]:
        k = sum(int(t[a]["hold"]) for t in trials)
        anat = [x for x in (failure_anatomy(t[a]) for t in trials) if x]
        first = Counter(x["first_flip"] for x in anat)
        s["arms"][a] = {
            "n_hold": k, "n": n, "P_hold": (k / n) if n else None, "ci": rc.wilson(k, n),
            "P_e2e": (k / tries) if tries else None, "e2e_ci": rc.wilson(k, tries),
            "total_gain": sum(p["arms"][a]["gains"].values()),
            "n_fail": len(anat),
            "final_states": dict(Counter(x["final"] for x in anat)),
            "first_fail_round": dict(Counter(str(x["first_fail_round"]) for x in anat)),
            "first_flip": dict(first),
            "C_alone": first.get("C", 0),
            "C_with_H": sum(v for f, v in first.items() if "C" in f.split("+") and "H" in f.split("+")),
            "C_involved": sum(v for f, v in first.items() if "C" in f.split("+")),
            "later_collateral": sum(len(x["later_collateral"]) for x in anat),
            "reverted": sum(1 for x in anat if x["reverted"]),
        }
    raw = {}
    for a in p["arms"]:
        if a == p["control"]:
            continue
        x_only = sum(1 for t in trials if t[a]["hold"] and not t[p["control"]]["hold"])
        c_only = sum(1 for t in trials if t[p["control"]]["hold"] and not t[a]["hold"])
        raw[a] = rc.mcnemar_exact(x_only, c_only)
        s["paired_vs_control"][a] = {"arm_only": x_only, "control_only": c_only,
                                     "diff": ((x_only - c_only) / n) if n else None, "mcnemar_p": raw[a]}
    for a, ph in holm(raw).items():
        s["paired_vs_control"][a]["holm_p"] = ph
    return s


def _ci(ci) -> str:
    lo, hi = ci
    return "—" if lo is None else f"[{lo:.2f}, {hi:.2f}]"


def _g(g: dict[str, float]) -> str:
    return ", ".join(f"{ch}={g.get(ch, 0.0):g}" for ch in sar.CHANNELS)


def _counts(d: dict[str, int], fmt=str) -> str:
    return ", ".join(f"{fmt(k)}×{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "—"


def write_md(path: Path, p: dict[str, Any]) -> None:
    s = p.get("summary") or summarize(p)
    c = p["control"]
    L = ["# Phase 9W — targeted C/H/O calibration for retaining 111", "",
         "> New seeds (disjoint from 9U and 9V). Each conversation acquired at round 0 under full 8K is forked to all arms "
         "with identical round seeds. Target-sign, 8K sites, forced FINAL:. A flip is a failure; nothing repairs it.", "",
         f"rounds={p['rounds']}, seed={p['seed']}, control = arm {c} (frozen 9V, `{p['frozen_9v_config']}`).", "",
         "## Primary: $P(S_1{=}S_2{=}111\\mid S_0{=}111)$", "",
         "| arm | purpose | gains (C, H, O) | total | hold | 95% CI | arm-only / A-only | McNemar p | Holm p |",
         "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for a, meta in p["arms"].items():
        A = s["arms"][a]
        pr = s["paired_vs_control"].get(a)
        pv = "—" if pr is None else f"{pr['arm_only']} / {pr['control_only']}"
        mp = "—" if pr is None else f"{pr['mcnemar_p']:.3f}"
        hp = "—" if pr is None else f"{pr['holm_p']:.3f}"
        L.append(f"| {a} | {meta['purpose']} | {_g(meta['gains'])} | {A['total_gain']:g} | {A['n_hold']}/{A['n']} | "
                 f"{_ci(A['ci'])} | {pv} | {mp} | {hp} |")
    L += ["", "## Failure anatomy", "",
          "First flip = the bits that changed in the round where 111 was first lost.", "",
          "| arm | fails | first flip | C alone | C with H | C involved | final state | first-fail round | later collateral | reverted |",
          "|---|---:|---|---:|---:|---:|---|---|---:|---:|"]
    for a in p["arms"]:
        A = s["arms"][a]
        L.append(f"| {a} | {A['n_fail']} | {_counts(A['first_flip'])} | {A['C_alone']} | {A['C_with_H']} | "
                 f"{A['C_involved']} | {_counts(A['final_states'], lambda k: f'`{k}`')} | "
                 f"{_counts(A['first_fail_round'], lambda k: f'R{k}')} | {A['later_collateral']} | {A['reverted']} |")
    L += ["", "## Acquisition and end-to-end", "",
          f"Acquired {s['n_acq']}/{s['tries']} (P={s['P_acq'] or 0:.2f} {_ci(s['P_acq_ci'])}); "
          f"{'complete' if p.get('final') else 'partial'}.", "",
          "| arm | e2e $P(S_0{=}S_1{=}S_2{=}111)$ | 95% CI |", "|---|---:|---|"]
    for a in p["arms"]:
        A = s["arms"][a]
        L.append(f"| {a} | {A['P_e2e'] or 0:.2f} | {_ci(A['e2e_ci'])} |")
    L += ["", "Arms were fixed before the run and nothing was tuned on these seeds. With n≈40 per arm, overlapping CIs mean "
              "*not distinguishable at this n*, not equivalence. A non-control arm becomes the 111 baseline only if it "
              "holds at least as often as A **and** has fewer A-only than arm-only discordances; calling 111 *solved* needs "
              "a separate held-out confirmation of that arm.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def report(_args) -> int:
    p = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    p["summary"] = summarize(p)
    OUT_JSON.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")
    write_md(OUT_MD, p)
    print(OUT_MD)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-acq", type=int, default=N_ACQ)
    ap.add_argument("--max-tries", type=int, default=MAX_TRIES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true", help="re-render summary/md from the saved json")
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--sandbox", default="")
    args = ap.parse_args()
    return report(args) if args.report else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
