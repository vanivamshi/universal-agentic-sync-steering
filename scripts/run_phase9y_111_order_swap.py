#!/usr/bin/env python3
"""Phase 9Y — order-swapped re-execution of the 9X conversations (order-effect diagnostic).

9X: B (C=0.5) held 85/100 vs A (C=0) 74/100 (19 B-only / 8 A-only, p=0.052),
but the random arm order came out AB x63 / BA x37, and B's advantage was
concentrated in AB-order conversations (15/5 vs 4/3). Each fork starts from the
same round-0 conversation with the same round seeds, so outcomes should not
depend on which arm runs first. 9Y tests that.

Protocol: for every 9X acquired conversation (same seed, same order of
conversations), regenerate round 0 under full 8K with the same seed, check it
still reaches 111, then run A and B in the REVERSE of their 9X order with the
same round seeds and the same frozen gains (9X frozen config, SHA-checked), in
the same pinned environment (asserted equal to 9X's).

This is NOT an independent replication: it reuses the 9X conversations, so it
is analysed separately and never pooled with 9X as 200 conversations.

Outputs:
  1. Paired B vs A in 9Y (exact McNemar), and the 9X decision rule re-applied.
  2. The same comparison within each original 9X order stratum (AB, BA).
  3. C-alone and C+H failure counts per arm.
  4. Position effects: per arm, outcome when run first vs second (9X vs 9Y,
     paired per conversation; exact McNemar), and the pooled first- vs
     second-executed hold rate in 9Y regardless of arm.
  5. Reproducibility: per arm, whether the 9Y fork reproduces the 9X fork
     exactly (states, and per-round plan / final text / tools / sensitive paths).

  python -u scripts/run_phase9y_111_order_swap.py --device cuda [--resume]
  python -u scripts/run_phase9y_111_order_swap.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts import run_phase9w_111_cho_calibration as p9w  # noqa: E402
from scripts import run_phase9x_111_confirm as p9x  # noqa: E402
from scripts import sync_adaptive_retention as sar  # noqa: E402
from scripts import sync_retention_controller as rc  # noqa: E402

RESULTS = ROOT / "data" / "results"
SRC_9X = p9x.OUT_JSON
OUT_JSON = RESULTS / "sync_phase9y_111_order_swap.json"
OUT_MD = RESULTS / "sync_phase9y_111_order_swap.md"
TARGET = p9x.TARGET
ARMS = p9x.ARMS
FP_KEYS = ("plan", "final", "tools", "sensitive_paths", "n_messages")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def swap(order: list[str]) -> list[str]:
    return list(reversed(order))


def fingerprint(res: dict[str, Any]) -> list[Any]:
    """What a fork produced: states plus each round's plan, final, tools, sensitive paths."""
    return [res["states"]] + [[r["info"].get(k) for k in FP_KEYS] for r in res["rounds"]]


def load_9x() -> dict[str, Any]:
    x = json.loads(SRC_9X.read_text(encoding="utf-8"))
    assert x["final"] is True and len(x["trials"]) == x["n_acq_wanted"], "9X run not complete"
    return x


# --------------------------------------------------------------------------- run

def run(args) -> int:
    from scripts.run_adaptive_retention import _load_mod, make_round_fn
    from scripts.run_phase9u_retention_gains import run_fixed
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    x = load_9x()
    cfg = p9x.load_frozen()
    assert _sha(p9x.FROZEN) == x["frozen_config_sha256"], "9X frozen config changed"
    env = p9x.environment()
    assert env == x["environment"], f"environment differs from 9X: {env} vs {x['environment']}"
    actions = {a: p9x.arm_action(cfg, a) for a in ARMS}
    assert {a: actions[a].gains() for a in ARMS} == {a: x["arms"][a]["gains"] for a in ARMS}

    payload: dict[str, Any] = {
        "protocol": "9Y order-swapped re-execution of 9X conversations (diagnostic, not a replication)",
        "target": list(TARGET), "rounds": x["rounds"], "control": "A",
        "source_9x": str(SRC_9X.relative_to(ROOT)), "source_9x_sha256": _sha(SRC_9X),
        "frozen_config": x["frozen_config"], "frozen_config_sha256": x["frozen_config_sha256"],
        "environment": env, "arms": x["arms"],
        "attempts": [], "trials": [], "final": False,
    }
    if args.resume and OUT_JSON.exists():
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for k in ("source_9x_sha256", "arms", "environment"):
            assert prev[k] == payload[k], f"resume refused: {k} differs"
        payload["attempts"], payload["trials"] = prev["attempts"], prev["trials"]
        print(f"resume: {len(payload['trials'])} trials", flush=True)

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

    print(f"=== 9Y order swap of {len(x['trials'])} 9X conversations env={env} ===", flush=True)
    for xt in x["trials"][len(payload["trials"]):]:
        sd, rs = xt["seed"], xt["round_seeds"]
        msgs0, S0, _ = round_fn(None, sar.CONTINUOUS, list(TARGET), sd)
        reacq = S0 == list(TARGET)
        order = swap(xt["order"])
        trial: dict[str, Any] = {"seed": sd, "round_seeds": rs, "order": order, "order_9x": xt["order"],
                                 "reacquired": reacq, "S0": list(S0)}
        if reacq:
            for a in order:
                trial[a] = run_fixed(round_fn, msgs0, actions[a], TARGET, rs)
        payload["trials"].append(trial)
        print(f"  seed={sd} S0={sar.key(S0)} order={''.join(order)} " + (" | ".join(
            f"{a} {'>'.join(trial[a]['path'])} {'hold' if trial[a]['hold'] else 'FAIL'}"
            f"{'' if fingerprint(trial[a]) == fingerprint(xt[a]) else ' (differs from 9X)'}" for a in ARMS)
            if reacq else "NOT REACQUIRED"), flush=True)
        save()

    payload["summary"] = summarize(payload, x)
    payload["final"] = len(payload["trials"]) == len(x["trials"])
    save()
    write_md(OUT_MD, payload)
    print(json.dumps({"out": str(OUT_JSON), **payload["summary"]["headline"]}), flush=True)
    return 0


# --------------------------------------------------------------------------- analysis

def _paired(trials: list[dict], arm: str, ctl: str) -> dict[str, Any]:
    n = len(trials)
    a = sum(int(t[arm]["hold"]) for t in trials)
    c = sum(int(t[ctl]["hold"]) for t in trials)
    ao = sum(1 for t in trials if t[arm]["hold"] and not t[ctl]["hold"])
    co = sum(1 for t in trials if t[ctl]["hold"] and not t[arm]["hold"])
    return {"n": n, f"{arm}_hold": a, f"{ctl}_hold": c, f"{arm}_only": ao, f"{ctl}_only": co,
            "mcnemar_p": rc.mcnemar_exact(ao, co)}


def summarize(p: dict[str, Any], x: dict[str, Any]) -> dict[str, Any]:
    xmap = {t["seed"]: t for t in x["trials"]}
    ok = [t for t in p["trials"] if t["reacquired"]]
    sub = {"control": "A", "arms": p["arms"], "attempts": [{}] * len(p["trials"]), "trials": ok}
    s = p9x.summarize(sub)  # paired B vs A, decision rule, failure anatomy
    s["n_trials"] = len(p["trials"])
    s["n_reacquired"] = len(ok)

    # 2. within each original 9X order stratum (9Y vs 9X side by side)
    s["by_9x_order"] = {}
    for o in (["A", "B"], ["B", "A"]):
        k = "".join(o)
        y_t = [t for t in ok if t["order_9x"] == o]
        x_t = [xmap[t["seed"]] for t in y_t]
        s["by_9x_order"][k] = {"9Y": _paired(y_t, "B", "A"), "9X": _paired(x_t, "B", "A")}

    # 4a. per arm: run first vs run second, paired per conversation across 9X and 9Y
    s["position_by_arm"] = {}
    for a in ARMS:
        first = second = f_only = s_only = 0
        for t in ok:
            xt = xmap[t["seed"]]
            h_x, h_y = xt[a]["hold"], t[a]["hold"]
            h_first, h_second = (h_x, h_y) if xt["order"][0] == a else (h_y, h_x)
            first += h_first
            second += h_second
            f_only += h_first and not h_second
            s_only += h_second and not h_first
        s["position_by_arm"][a] = {"n": len(ok), "first_hold": first, "second_hold": second,
                                   "first_only": f_only, "second_only": s_only,
                                   "mcnemar_p": rc.mcnemar_exact(f_only, s_only)}
    # 4b. in 9Y, first-executed vs second-executed fork, regardless of arm
    f = sum(int(t[t["order"][0]]["hold"]) for t in ok)
    sec = sum(int(t[t["order"][1]]["hold"]) for t in ok)
    fo = sum(1 for t in ok if t[t["order"][0]]["hold"] and not t[t["order"][1]]["hold"])
    so = sum(1 for t in ok if t[t["order"][1]]["hold"] and not t[t["order"][0]]["hold"])
    s["position_9y"] = {"n": len(ok), "first_hold": f, "second_hold": sec, "first_only": fo, "second_only": so,
                        "mcnemar_p": rc.mcnemar_exact(fo, so)}

    # 5. reproducibility vs 9X
    s["reproduce"] = {}
    for a in ARMS:
        same_fp = sum(1 for t in ok if fingerprint(t[a]) == fingerprint(xmap[t["seed"]][a]))
        same_states = sum(1 for t in ok if t[a]["states"] == xmap[t["seed"]][a]["states"])
        same_hold = sum(1 for t in ok if t[a]["hold"] == xmap[t["seed"]][a]["hold"])
        s["reproduce"][a] = {"n": len(ok), "identical": same_fp, "same_states": same_states, "same_hold": same_hold}

    pr = s["paired_vs_control"]["B"]
    s["headline"] = {"n": len(ok), "A": s["arms"]["A"]["n_hold"], "B": s["arms"]["B"]["n_hold"],
                     "B_only": pr["arm_only"], "A_only": pr["control_only"], "p": pr["mcnemar_p"],
                     "identical_A": s["reproduce"]["A"]["identical"], "identical_B": s["reproduce"]["B"]["identical"],
                     "verdict": s["decision"]["verdict"]}
    return s


def write_md(path: Path, p: dict[str, Any]) -> None:
    x = load_9x()
    s = p.get("summary") or summarize(p, x)
    xs = x["summary"]
    pr, xpr = s["paired_vs_control"]["B"], xs["paired_vs_control"]["B"]
    env = p["environment"]
    L = ["# Phase 9Y — order-swapped re-execution of the 9X conversations", "",
         "> Diagnostic, not a replication: the same 100 9X conversations (same seeds, regenerated round 0, same round "
         "seeds, same frozen gains, same environment), with A/B execution order reversed for every pair. "
         "Analysed separately from 9X; never pooled.", "",
         f"Environment: torch {env['torch']}, transformers {env['transformers']}, CUDA {env['cuda']}, "
         f"{env['device_name']}, {env['dtype']}. Round 0 re-reached 111 in {s['n_reacquired']}/{s['n_trials']}.", "",
         "## 1. Does B keep its advantage after the swap?", "",
         "| run | order AB / BA | A hold | B hold | B-only | A-only | McNemar p | decision rule |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    L.append(f"| 9X | {xs['order_counts']['AB']} / {xs['order_counts']['BA']} | {xs['arms']['A']['n_hold']}/{xs['n_acq']} | "
             f"{xs['arms']['B']['n_hold']}/{xs['n_acq']} | {xpr['arm_only']} | {xpr['control_only']} | "
             f"{xpr['mcnemar_p']:.3f} | {xs['decision']['verdict']} |")
    L.append(f"| 9Y (swapped) | {s['order_counts']['AB']} / {s['order_counts']['BA']} | {s['arms']['A']['n_hold']}/{s['n_acq']} | "
             f"{s['arms']['B']['n_hold']}/{s['n_acq']} | {pr['arm_only']} | {pr['control_only']} | "
             f"{pr['mcnemar_p']:.3f} | {s['decision']['verdict']} |")
    L += ["", "## 2. Within each original 9X order stratum", "",
          "| 9X order | n | run | A hold | B hold | B-only | A-only | McNemar p |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for k, v in s["by_9x_order"].items():
        for run in ("9X", "9Y"):
            q = v[run]
            L.append(f"| {k} | {q['n']} | {run} | {q['A_hold']} | {q['B_hold']} | {q['B_only']} | {q['A_only']} | "
                     f"{q['mcnemar_p']:.3f} |")
    L += ["", "## 3. Failure anatomy (9Y)", "",
          "| arm | fails | C alone | C with H | H-first | final state | 9X: C alone / C+H |", "|---|---:|---:|---:|---:|---|---|"]
    for a in ARMS:
        A, X = s["arms"][a], xs["arms"][a]
        L.append(f"| {a} | {A['n_fail']} | {A['C_alone']} | {A['C_with_H']} | {A['first_flip'].get('H', 0)} | "
                 f"{p9w._counts(A['final_states'], lambda k: f'`{k}`')} | {X['C_alone']} / {X['C_with_H']} |")
    L += ["", "## 4. Does execution position matter?", "",
          "Per arm, each conversation was run once first and once second (9X vs 9Y):", "",
          "| arm | n | hold when first | hold when second | first-only | second-only | McNemar p |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        q = s["position_by_arm"][a]
        L.append(f"| {a} | {q['n']} | {q['first_hold']} | {q['second_hold']} | {q['first_only']} | {q['second_only']} | "
                 f"{q['mcnemar_p']:.3f} |")
    q = s["position_9y"]
    L += ["", f"Within 9Y regardless of arm: first-executed fork held {q['first_hold']}/{q['n']}, second-executed "
              f"{q['second_hold']}/{q['n']} (first-only {q['first_only']}, second-only {q['second_only']}, "
              f"McNemar p={q['mcnemar_p']:.3f}).", "",
          "## 5. Reproducibility of each fork vs 9X", "",
          "| arm | n | identical text + states | same states | same hold outcome |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        r = s["reproduce"][a]
        L.append(f"| {a} | {r['n']} | {r['identical']} | {r['same_states']} | {r['same_hold']} |")
    L += ["", "Identical = every round's state, plan, final text, tool list and sensitive paths match 9X exactly. "
              "If forks are order-independent and generation is deterministic, every row is n/n.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def report(_args) -> int:
    p = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    p["summary"] = summarize(p, load_9x())
    OUT_JSON.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")
    write_md(OUT_MD, p)
    print(OUT_MD)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--sandbox", default="")
    args = ap.parse_args()
    return report(args) if args.report else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
