#!/usr/bin/env python3
"""Phase 9X — preregistered paired confirmation of the 9W weak-C candidate for 111.

9W (n=40) selected B = (C=0.5, H=1.5, O=1.5) over the 9V baseline A = (C=0, H=1.5,
O=1.5): 37/40 vs 34/40, discordant 4 B-only / 1 A-only, McNemar p=0.375. That
satisfies the selection rule but is not evidence that B is better. 9X checks
whether the result replicates. No gains are retuned.

Arms (both frozen in data/configs/phase9x_frozen_111.json; written once by
--freeze from the 9W result file and the 9V frozen config, SHA-256 re-verified
on every run):
  A  C=0    H=1.5  O=1.5   baseline (9V frozen)
  B  C=0.5  H=1.5  O=1.5   candidate (9W arm B)

Protocol:
  1. Acquire: round 0 under full 8K on NEW seeds (base 51000000, disjoint from
     9U, 9V and 9W; asserted). Every attempt is recorded.
  2. Only after S_0 = 111 is the arm order drawn: a coin seeded by the
     acquisition seed decides whether A or B runs first. Both forks start from
     the identical round-0 conversation and use identical round seeds.
  3. Primary outcome: S_1 = S_2 = 111 (both-round retention).
     Primary comparison: paired B vs A, exact two-sided McNemar.
     Secondary: first-failure flips that are C alone, and C+H together.
  4. Selection rule (fixed before the run): B is confirmed as the 111 candidate
     if hold(B) >= hold(A) AND B-only > A-only. "B is better than A" is claimed
     only if McNemar p < 0.05 in B's favour.
  5. Environment pinned: torch / transformers / CUDA / GPU are recorded and a
     resume in a different environment is refused.

  python -u scripts/run_phase9x_111_confirm.py --freeze
  python -u scripts/run_phase9x_111_confirm.py --device cuda [--resume]
  python -u scripts/run_phase9x_111_confirm.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts import run_phase9w_111_cho_calibration as p9w  # noqa: E402
from scripts import sync_adaptive_retention as sar  # noqa: E402
from scripts import sync_retention_controller as rc  # noqa: E402

RESULTS = ROOT / "data" / "results"
FROZEN = ROOT / "data" / "configs" / "phase9x_frozen_111.json"
SRC_9W = RESULTS / "sync_phase9w_111_cho_calibration.json"
OUT_JSON = RESULTS / "sync_phase9x_111_confirm.json"
OUT_MD = RESULTS / "sync_phase9x_111_confirm.md"
SCHEMA = "phase9x-frozen-111/v1"

TARGET, TKEY = p9w.TARGET, p9w.TKEY
ARMS = ("A", "B")
CONTROL = "A"

SEED = 51000000
SEED_STRIDE = 41
ROUND_STRIDE = 7919
N_ACQ = 100
MAX_TRIES = 700

_9W_LO, _9W_HI = 41000000, 41000000 + p9w.TARGET_STRIDE * 7 + SEED_STRIDE * 2000 + ROUND_STRIDE * 10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acq_seed(seed: int, i: int) -> int:
    return seed + SEED_STRIDE * i


def round_seeds(s0: int, rounds: int) -> list[int]:
    return [s0 + ROUND_STRIDE * k for k in range(1, rounds + 1)]


def arm_order(s0: int) -> list[str]:
    """Arm order for one acquired conversation, drawn after acquisition."""
    return ["A", "B"] if random.Random(s0 ^ 0x9A9A).random() < 0.5 else ["B", "A"]


def assert_new_seeds(seed: int, max_tries: int, rounds: int) -> None:
    lo = acq_seed(seed, 0)
    hi = round_seeds(acq_seed(seed, max_tries - 1), rounds)[-1]
    for name, a, b in (("9U", p9w._9U_LO, p9w._9U_HI), ("9V", p9w._9V_LO, p9w._9V_HI), ("9W", _9W_LO, _9W_HI)):
        assert hi < a or lo > b, f"9X seeds [{lo},{hi}] overlap {name} envelope [{a},{b}]"


# --------------------------------------------------------------------------- freeze

def build_frozen() -> dict[str, Any]:
    w = json.loads(SRC_9W.read_text(encoding="utf-8"))
    assert w["final"] is True and w["target"] == list(TARGET)
    assert w["summary"]["arms"]["B"]["n_hold"] == 37 and w["summary"]["arms"]["A"]["n_hold"] == 34, "unexpected 9W result"
    p9w.assert_control_is_frozen_9v()
    return {
        "schema": SCHEMA,
        "target": TKEY,
        "base_gains": dict(sar.GAINS),
        "arms": {
            "A": {"role": "baseline (9V frozen)", "scale": list(w["arms"]["A"]["scale"]), "gains": w["arms"]["A"]["gains"]},
            "B": {"role": "candidate (9W arm B)", "scale": list(w["arms"]["B"]["scale"]), "gains": w["arms"]["B"]["gains"]},
        },
        "source_9w": str(SRC_9W.relative_to(ROOT)),
        "source_9w_sha256": _sha(SRC_9W),
        "source_9v": str(p9w.FROZEN_9V.relative_to(ROOT)),
        "source_9v_sha256": _sha(p9w.FROZEN_9V),
    }


def load_frozen() -> dict[str, Any]:
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert cfg["schema"] == SCHEMA and cfg["target"] == TKEY
    assert cfg["base_gains"] == dict(sar.GAINS), "locked 8K gains changed since freeze"
    assert _sha(ROOT / cfg["source_9w"]) == cfg["source_9w_sha256"], "9W source changed since freeze"
    assert _sha(ROOT / cfg["source_9v"]) == cfg["source_9v_sha256"], "9V frozen config changed since freeze"
    for a, v in cfg["arms"].items():
        assert sar.scaled_continuous(v["scale"]).gains() == v["gains"], a
    assert cfg["arms"]["A"]["gains"] == {"H": 1.5, "O": 1.5}
    assert cfg["arms"]["B"]["gains"] == {"C": 0.5, "H": 1.5, "O": 1.5}
    return cfg


def arm_action(cfg: dict[str, Any], arm: str) -> sar.Action:
    return sar.scaled_continuous(cfg["arms"][arm]["scale"], name=f"9X_{arm}")


def freeze(args) -> int:
    if FROZEN.exists() and not args.force_freeze:
        print(f"{FROZEN} already exists; refusing to overwrite (use --force-freeze).")
        return 1
    FROZEN.write_text(json.dumps(build_frozen(), indent=2) + "\n", encoding="utf-8")
    load_frozen()
    print(FROZEN)
    return 0


# --------------------------------------------------------------------------- run

def environment() -> dict[str, Any]:
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "float32"}


def run(args) -> int:
    from scripts.run_adaptive_retention import _load_mod, make_round_fn
    from scripts.run_phase9u_retention_gains import run_fixed
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    cfg = load_frozen()
    assert_new_seeds(args.seed, args.max_tries, args.rounds)
    actions = {a: arm_action(cfg, a) for a in ARMS}
    env = environment()

    payload: dict[str, Any] = {
        "protocol": "9X preregistered paired confirmation for 111: A=9V baseline vs B=9W weak-C candidate",
        "target": list(TARGET), "seed": args.seed, "rounds": args.rounds,
        "n_acq_wanted": args.n_acq, "max_tries": args.max_tries, "control": CONTROL,
        "frozen_config": str(FROZEN.relative_to(ROOT)), "frozen_config_sha256": _sha(FROZEN),
        "environment": env,
        "arms": {a: {"name": actions[a].name, "scale": cfg["arms"][a]["scale"], "gains": actions[a].gains(),
                     "purpose": cfg["arms"][a]["role"]} for a in ARMS},
        "attempts": [], "trials": [], "final": False,
    }
    if args.resume and OUT_JSON.exists():
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for k in ("arms", "seed", "rounds", "frozen_config_sha256", "environment"):
            assert prev[k] == payload[k], f"resume refused: {k} differs from the saved run"
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

    print(f"=== 9X target=111 A={actions['A'].gains()} B={actions['B'].gains()} need={args.n_acq} "
          f"max_tries={args.max_tries} env={env} ===", flush=True)
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
            order = arm_order(sd)
            trial: dict[str, Any] = {"seed": sd, "round_seeds": rs, "order": order}
            for a in order:
                trial[a] = run_fixed(round_fn, msgs0, actions[a], TARGET, rs)
            payload["trials"].append(trial)
            print(f"    order={''.join(order)} " + " | ".join(
                f"{a} {'>'.join(trial[a]['path'])} {'hold' if trial[a]['hold'] else 'FAIL'}" for a in ARMS), flush=True)
        save()

    payload["summary"] = summarize(payload)
    payload["final"] = len(payload["trials"]) >= args.n_acq or i >= args.max_tries
    save()
    write_md(OUT_MD, payload)
    print(json.dumps({"out": str(OUT_JSON), "n_acq": payload["summary"]["n_acq"],
                      **{a: payload["summary"]["arms"][a]["n_hold"] for a in ARMS},
                      "decision": payload["summary"]["decision"]}), flush=True)
    return 0


# --------------------------------------------------------------------------- analysis

def summarize(p: dict[str, Any]) -> dict[str, Any]:
    s = p9w.summarize(p)
    A, B = s["arms"]["A"], s["arms"]["B"]
    pr = s["paired_vs_control"]["B"]
    selected = B["n_hold"] >= A["n_hold"] and pr["arm_only"] > pr["control_only"]
    better = selected and pr["mcnemar_p"] < 0.05
    s["decision"] = {
        "selection_rule_met": selected,
        "B_better_than_A": better,
        "verdict": ("B better than A (p<0.05)" if better else
                    "B confirmed as candidate; not shown better" if selected else
                    "B not confirmed"),
    }
    s["order_counts"] = {"AB": sum(1 for t in p["trials"] if t.get("order") == ["A", "B"]),
                         "BA": sum(1 for t in p["trials"] if t.get("order") == ["B", "A"])}
    return s


def write_md(path: Path, p: dict[str, Any]) -> None:
    s = p.get("summary") or summarize(p)
    pr = s["paired_vs_control"]["B"]
    env = p["environment"]
    L = ["# Phase 9X — preregistered paired confirmation for 111 (A vs B)", "",
         "> New seeds (disjoint from 9U, 9V and 9W). Arms frozen from 9W; no retuning. Each conversation acquired at "
         "round 0 under full 8K is forked to A and B in a random order drawn after acquisition, with identical round "
         "seeds. Target-sign, 8K sites, forced FINAL:. A flip is a failure.", "",
         f"rounds={p['rounds']}, seed={p['seed']}, frozen config `{p['frozen_config']}`. "
         f"Environment: torch {env['torch']}, transformers {env['transformers']}, CUDA {env['cuda']}, "
         f"{env['device_name']}, {env['dtype']}.", "",
         "## Primary: paired B vs A on $P(S_1{=}S_2{=}111\\mid S_0{=}111)$", "",
         "| arm | gains (C, H, O) | hold | 95% CI |", "|---|---|---:|---|"]
    for a in ARMS:
        A = s["arms"][a]
        L.append(f"| {a} ({p['arms'][a]['purpose']}) | {p9w._g(p['arms'][a]['gains'])} | {A['n_hold']}/{A['n']} | "
                 f"{p9w._ci(A['ci'])} |")
    L += ["", f"Paired: both hold {s['arms']['A']['n_hold'] - pr['control_only']}, B-only **{pr['arm_only']}**, "
              f"A-only **{pr['control_only']}**, neither {s['n_acq'] - s['arms']['A']['n_hold'] - pr['arm_only']}. "
              f"Exact McNemar p = **{pr['mcnemar_p']:.3f}**.", "",
          f"**Decision (preregistered rule):** {s['decision']['verdict']}.", "",
          "## Secondary: failure anatomy", "",
          "| arm | fails | C alone | C with H | C involved | first flip | final state | first-fail round |",
          "|---|---:|---:|---:|---:|---|---|---|"]
    for a in ARMS:
        A = s["arms"][a]
        L.append(f"| {a} | {A['n_fail']} | {A['C_alone']} | {A['C_with_H']} | {A['C_involved']} | "
                 f"{p9w._counts(A['first_flip'])} | {p9w._counts(A['final_states'], lambda k: f'`{k}`')} | "
                 f"{p9w._counts(A['first_fail_round'], lambda k: f'R{k}')} |")
    L += ["", "## Acquisition and end-to-end", "",
          f"Acquired {s['n_acq']}/{s['tries']} (P={s['P_acq'] or 0:.2f} {p9w._ci(s['P_acq_ci'])}); "
          f"arm order AB×{s['order_counts']['AB']}, BA×{s['order_counts']['BA']}; "
          f"{'complete' if p.get('final') else 'partial'}.", "",
          "| arm | e2e $P(S_0{=}S_1{=}S_2{=}111)$ | 95% CI |", "|---|---:|---|"]
    for a in ARMS:
        A = s["arms"][a]
        L.append(f"| {a} | {A['P_e2e'] or 0:.2f} | {p9w._ci(A['e2e_ci'])} |")
    L += ["", "Rule fixed before the run: B is confirmed as the 111 candidate if hold(B) ≥ hold(A) and B-only > A-only; "
              "B is called better than A only if McNemar p < 0.05. Do not pool with 9V (MPS); 9W ran in the same "
              "CUDA environment but on different seeds and is not pooled here either.", ""]
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
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--force-freeze", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-acq", type=int, default=N_ACQ)
    ap.add_argument("--max-tries", type=int, default=MAX_TRIES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--device", default="", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--sandbox", default="")
    args = ap.parse_args()
    if args.freeze:
        return freeze(args)
    if args.report:
        return report(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
