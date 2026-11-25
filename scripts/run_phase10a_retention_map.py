#!/usr/bin/env python3
"""Phase 10A — Eight-state retention map (one protocol for all m*).

Question: once S*=(C,H,O) is acquired, can the controller keep that exact
state across successive interactions?

Protocol (identical for every m*; fixed before held-out data is seen):

  1. calibrate  — target-independent sweep over frozen steering configs
                  g=(g_C,g_H,g_O), g_k ∈ {-2,-1,0,+1,+2} (multiplier on the
                  frozen 8K gains α_C=5, α_H=1.5, α_O=1.5; 0 = channel off;
                  sign = actuator polarity). 125 configs × N_cal seeds.
                  One sweep estimates P(S | g) for all eight S at once.
  2. select     — per m*: g* = argmax P̂_cal(m* | g) among configs with
                  task_ok ≥ 0.9; ties → fewer active channels → smaller Σ|g|.
                  Baseline = g=(0,0,0) (no steering).
  3. evaluate   — held-out seeds (disjoint from calibration). Per attempt:
                  acquire with the frozen 9R hybrid (Γ' + beam), verify
                  S_0 = m* on all three bits, then run BOTH arms for R=5
                  rounds from the same S_0 / plan with paired round seeds:
                    baseline   : no steering
                    calibrated : frozen g*(m*)
                  Every round is run even after a failure (full trajectory).
  4. report     — P_acq; P(S_1..S_5 = m* | S_0 = m*) per arm (Wilson CI);
                  per-round marginals; first failure round + bits flipped;
                  Markov check P(S_t=m*|S_{t-1}=m*) vs P(S_t=m*|S_{t-1}≠m*).

A "round" is one fresh episode of the neutral task (same as 9L/9M/9R
retain steps). The only carried context is the PLAN prefill used on the H
decision path (H active, C inactive).

  .venv/bin/python -u scripts/run_phase10a_retention_map.py calibrate --shard 0 --n-shards 8
  .venv/bin/python -u scripts/run_phase10a_retention_map.py select
  .venv/bin/python -u scripts/run_phase10a_retention_map.py evaluate --shard 0 --n-shards 16
  .venv/bin/python -u scripts/run_phase10a_retention_map.py report
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT_DIR = ROOT / "data" / "results" / "phase10a"
CALIB_GLOB = "calib_shard*.jsonl"
EVAL_GLOB = "eval_shard*.jsonl"
SELECTION = OUT_DIR / "selection.json"
REPORT_JSON = ROOT / "data" / "results" / "sync_phase10a_retention_map.json"
REPORT_MD = ROOT / "data" / "results" / "sync_phase10a_retention_map.md"

CHANNELS = ("C", "H", "O")  # state bit order
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
GAIN_LEVELS = (-2, -1, 0, 1, 2)
ALL_CONFIGS = list(itertools.product(GAIN_LEVELS, repeat=3))  # (gC, gH, gO)
BASELINE_CFG = (0, 0, 0)

N_ROUNDS = 5
CAL_SEED_BASE = 30_000_000
EVAL_SEED_BASE = 40_000_000
SEED_STRIDE = 50_021  # > internal episode offsets (≤ ~9_100)
ROUND_OFFSET = 600_000
TASK_OK_MIN = 0.9


def _key(s) -> str:
    return "".join(str(int(x)) for x in s)


def _cfg_key(g) -> str:
    return ",".join(f"{int(x):+d}" for x in g)


def _cfg_from_key(k: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in k.split(","))  # type: ignore[return-value]


def _n_active(g) -> int:
    return sum(1 for x in g if x != 0)


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- runtime ----


class Runtime:
    """Model + frozen actuators, loaded once per worker."""

    def __init__(self, need_planner: bool = False):
        from activation_pipeline.device import (
            LOCAL_MODEL_KEY,
            assert_model_fits_machine,
            resolve_device_map,
        )
        from activation_pipeline.loader import load_model_and_tokenizer

        self.p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
        self.sc = self.p8k._load_sc()
        self.sc.ensure_sandbox()
        assert_model_fits_machine(LOCAL_MODEL_KEY)
        self.loaded = load_model_and_tokenizer(
            LOCAL_MODEL_KEY,
            device_map=resolve_device_map(None),
            dtype="float32",
            local_files_only=True,
        )
        self.dirs = self.p8k._load_vc()
        if need_planner:
            p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
            p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
            self.p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
            self.p9q = _load_mod("phase9q", ROOT / "scripts" / "run_phase9q_bidir_beam.py")
            self.p9r = _load_mod("phase9r", ROOT / "scripts" / "run_phase9r_hybrid.py")
            self.lookup, _k, _r = self.p9j.build_lookup(p9e, p9g, None)

    def episode(self, g, *, seed: int, plan_prefill: str = "") -> dict[str, Any]:
        """One episode under frozen config g (multipliers on frozen 8K α).

        Mirrors phase8k._run_episode (same sites, prefill rules) but with a
        per-channel signed multiplier instead of a fixed ±1.
        """
        p8k, sc, loaded, dirs = self.p8k, self.sc, self.loaded, self.dirs
        from scripts.sync_h_decision import make_steer_hook

        base = {"C": p8k.ALPHA_C, "H": p8k.ALPHA_H, "O": p8k.ALPHA_O}
        gm = dict(zip(CHANNELS, g))
        h_hook = c_hook = o_hook = tool_c_hook = None
        c_stem = None
        if gm["H"]:
            h_hook = make_steer_hook(loaded, dirs["H"], base["H"] * gm["H"])
        if gm["C"]:
            c_hook = make_steer_hook(loaded, dirs["C"], base["C"] * gm["C"])
            tool_c_hook = make_steer_hook(loaded, dirs["C"], base["C"] * gm["C"])
            c_stem = p8k.C_STEM
        if gm["O"]:
            o_hook = make_steer_hook(loaded, dirs["O"], base["O"] * gm["O"])

        def hook_for_turn(phase: str, turn: int):
            if phase == "tool" and turn > 0:
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

        try:
            row = sc.run_episode(
                loaded,
                cls=p8k.NEUTRAL_CLS,
                task_id=p8k.NEUTRAL_TASK,
                seed=int(seed),
                hook_for_turn=hook_for_turn,
                with_plan_format=True,
                capture_activations=False,
                h_decision_hook=h_hook,
                h_plan_prefill=(plan_prefill or None) if (gm["H"] and c_stem is None) else None,
                c_decision_hook=c_hook,
                c_stem_prefill=c_stem,
                report_prefill="FINAL: " if gm["O"] else None,
            )
        finally:
            for h in (h_hook, c_hook, o_hook, tool_c_hook):
                if h is not None:
                    try:
                        h.remove()
                    except Exception:
                        pass
        S = p8k._S_from_row(row)
        from scripts.sync_eq import extract_plan

        plan = extract_plan((row.get("final") or "") + "\n" + (row.get("messages_text") or ""))
        prefill = (plan if plan.endswith("\n") else plan + "\n") if plan else ""
        return {
            "S": S,
            "plan_prefill": prefill,
            "task_ok": int(bool((row.get("final") or "").strip())),
        }


def _done_ids(paths, field: str) -> set:
    done = set()
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)[field])
    return done


# -------------------------------------------------------------- calibrate ----


def cmd_calibrate(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"calib_shard{args.shard}.jsonl"
    seeds = [CAL_SEED_BASE + SEED_STRIDE * i for i in range(args.n_cal)]
    # Unit of work = one seed (free run first, its plan feeds H-only configs).
    mine = [s for i, s in enumerate(seeds) if i % args.n_shards == args.shard]
    done = _done_ids([out], "id")
    rt = Runtime()
    with out.open("a", encoding="utf-8") as fh:
        for seed in mine:
            free = None
            for g in [BASELINE_CFG] + [c for c in ALL_CONFIGS if c != BASELINE_CFG]:
                rid = f"{seed}|{_cfg_key(g)}"
                if rid in done and g != BASELINE_CFG:
                    continue
                t0 = time.time()
                if g == BASELINE_CFG:
                    ep = rt.episode(g, seed=seed)
                    free = ep
                    if rid in done:
                        continue
                else:
                    ep = rt.episode(g, seed=seed, plan_prefill=free["plan_prefill"])
                rec = {
                    "id": rid,
                    "seed": seed,
                    "cfg": _cfg_key(g),
                    "S": _key(ep["S"]),
                    "task_ok": ep["task_ok"],
                    "sec": round(time.time() - t0, 2),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[cal {args.shard}] seed={seed} g={rec['cfg']} S={rec['S']} {rec['sec']}s", flush=True)
    return 0


# ----------------------------------------------------------------- select ----


def _load_jsonl(glob: str) -> list[dict]:
    rows = []
    for p in sorted(OUT_DIR.glob(glob)):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def cmd_select(args) -> int:
    rows = _load_jsonl(CALIB_GLOB)
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)
    n_seeds = len({r["seed"] for r in rows})
    stats = {}
    for ck, rs in by_cfg.items():
        n = len(rs)
        cnt = Counter(r["S"] for r in rs)
        stats[ck] = {
            "n": n,
            "task_ok": sum(r["task_ok"] for r in rs) / n,
            "P": {_key(m): cnt.get(_key(m), 0) / n for m in ALL_MSTAR},
            "k": {_key(m): cnt.get(_key(m), 0) for m in ALL_MSTAR},
        }
    eligible = [
        ck for ck, st in stats.items() if st["task_ok"] >= TASK_OK_MIN and st["n"] >= n_seeds
    ]

    def rank_key(ck, mk):
        g = _cfg_from_key(ck)
        return (-stats[ck]["P"][mk], _n_active(g), sum(abs(x) for x in g), ck)

    sel = {}
    for m in ALL_MSTAR:
        mk = _key(m)
        ranked = sorted(eligible, key=lambda ck: rank_key(ck, mk))
        best = ranked[0]
        p_best = stats[best]["P"][mk]
        # minimum-effective: fewest channels within one count of best
        tol = 1.0 / max(1, n_seeds)
        min_eff = sorted(
            [ck for ck in eligible if stats[ck]["P"][mk] >= p_best - tol - 1e-9],
            key=lambda ck: (_n_active(_cfg_from_key(ck)), sum(abs(x) for x in _cfg_from_key(ck)), -stats[ck]["P"][mk]),
        )[0]
        best_by_nact = {}
        for na in range(4):
            cands = [ck for ck in eligible if _n_active(_cfg_from_key(ck)) == na]
            if cands:
                b = sorted(cands, key=lambda ck: rank_key(ck, mk))[0]
                best_by_nact[na] = {"cfg": b, "P": stats[b]["P"][mk]}
        b0 = _cfg_key(BASELINE_CFG)
        sel[mk] = {
            "calibrated_cfg": best,
            "P_cal": p_best,
            "k_cal": stats[best]["k"][mk],
            "n_cal": stats[best]["n"],
            "P_cal_wilson": wilson(stats[best]["k"][mk], stats[best]["n"]),
            "baseline_P_cal": stats[b0]["P"][mk] if b0 in stats else None,
            "min_effective_cfg": min_eff,
            "min_effective_P_cal": stats[min_eff]["P"][mk],
            "best_by_n_active": best_by_nact,
            "top5": [{"cfg": ck, "P": stats[ck]["P"][mk]} for ck in ranked[:5]],
            "predicted_P5_indep": p_best**N_ROUNDS,
        }
    payload = {
        "rule": (
            "g* = argmax P_cal(m*|g) over g∈{-2..2}^3 × frozen 8K α, task_ok≥0.9; "
            "ties: fewer active channels, smaller Σ|g|."
        ),
        "n_cal_seeds": n_seeds,
        "n_configs": len(stats),
        "n_eligible": len(eligible),
        "selection": sel,
        "config_stats": stats,
    }
    SELECTION.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for mk, s in sel.items():
        print(
            f"m*={mk}  g*={s['calibrated_cfg']}  P_cal={s['P_cal']:.2f} "
            f"base={s['baseline_P_cal']:.2f}  min_eff={s['min_effective_cfg']} "
            f"(P={s['min_effective_P_cal']:.2f})"
        )
    print(f"eligible {len(eligible)}/{len(stats)} configs; seeds={n_seeds}")
    return 0


# --------------------------------------------------------------- evaluate ----


def _trial_list(n_attempts: int) -> list[tuple[int, tuple, int]]:
    out = []
    for mi, m in enumerate(ALL_MSTAR):
        for r in range(n_attempts):
            out.append((mi, m, r))
    return out


def _run_rounds(rt: Runtime, g, mstar, *, S0, plan0, round_seeds) -> list[dict]:
    S, plan = list(S0), plan0 or ""
    rounds = []
    for k, rs in enumerate(round_seeds, start=1):
        ep = rt.episode(g, seed=rs, plan_prefill=plan)
        Sa = list(ep["S"])
        flipped = [ch for i, ch in enumerate(CHANNELS) if Sa[i] != mstar[i]]
        rounds.append(
            {
                "k": k,
                "seed": rs,
                "S_before": _key(S),
                "S": _key(Sa),
                "on_target": int(Sa == list(mstar)),
                "wrong_bits": flipped,
                "task_ok": ep["task_ok"],
            }
        )
        S = Sa
        if ep.get("plan_prefill"):
            plan = ep["plan_prefill"]
    return rounds


def cmd_evaluate(args) -> int:
    sel = json.loads(SELECTION.read_text())["selection"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"eval_shard{args.shard}.jsonl"
    trials = _trial_list(args.n_attempts)
    # interleave states across shards so partial runs stay balanced
    mine = [t for i, t in enumerate(trials) if i % args.n_shards == args.shard]
    done = _done_ids(list(OUT_DIR.glob(EVAL_GLOB)), "id")
    rt = Runtime(need_planner=True)
    with out.open("a", encoding="utf-8") as fh:
        for mi, mstar, r in mine:
            tid = f"{_key(mstar)}|{r}"
            if tid in done:
                continue
            t0 = time.time()
            seed0 = EVAL_SEED_BASE + 1_000_003 * mi + SEED_STRIDE * r
            S0, plan0, seed = rt.p9j.free_S0(rt.p8k, rt.sc, rt.loaded, mstar=mstar, seed0=seed0)
            if S0 == list(mstar):
                acq = {
                    "acquire": 1, "S_final": list(S0), "plan_final": plan0,
                    "path": [_key(S0)], "actions": [], "n_beam_calls": 0, "land_t": 0,
                }
                free_hit = 1
            else:
                free_hit = 0
                acq = rt.p9r.run_trial(
                    rt.p9j, rt.p9q, rt.p8k, rt.sc, rt.loaded,
                    dirs=rt.dirs, lookup=rt.lookup, mstar=mstar, seed=seed,
                    S0=S0, plan0=plan0, arm="gamma_prime_beam",
                    tau=0.0, beam=3, horizon=3, max_steps_acq=args.max_steps_acq,
                    n_retain=0, cache={}, beam_cache={},
                )
            verified = int(bool(acq["acquire"]) and list(acq["S_final"]) == list(mstar))
            rec: dict[str, Any] = {
                "id": tid,
                "mstar": _key(mstar),
                "attempt": r,
                "seed": seed,
                "S0_free": _key(S0),
                "free_S0_is_target": free_hit,
                "acquired": verified,
                "acq_path": acq.get("path"),
                "acq_actions": acq.get("actions"),
                "acq_beam_calls": acq.get("n_beam_calls"),
                "arms": {},
            }
            if verified:
                round_seeds = [seed + ROUND_OFFSET + SEED_STRIDE * k for k in range(1, N_ROUNDS + 1)]
                plan_acq = acq.get("plan_final") or plan0 or ""
                arms = {
                    "baseline": BASELINE_CFG,
                    "calibrated": _cfg_from_key(sel[_key(mstar)]["calibrated_cfg"]),
                }
                for arm, g in arms.items():
                    rounds = _run_rounds(
                        rt, g, mstar, S0=mstar, plan0=plan_acq, round_seeds=round_seeds
                    )
                    rec["arms"][arm] = {"cfg": _cfg_key(g), "rounds": rounds}
            rec["sec"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            summ = {
                a: "".join(str(x["on_target"]) for x in v["rounds"]) for a, v in rec["arms"].items()
            }
            print(f"[eval {args.shard}] {tid} acq={verified} {summ} {rec['sec']}s", flush=True)
    return 0


# ----------------------------------------------------------------- report ----


def _arm_stats(trs: list[dict], arm: str) -> dict[str, Any]:
    rs = [t["arms"][arm] for t in trs if arm in t.get("arms", {})]
    n = len(rs)
    if n == 0:
        return {"n": 0}
    seqs = [[x["on_target"] for x in a["rounds"]] for a in rs]
    k_all = sum(1 for s in seqs if all(s))
    first_fail = Counter()
    mech = Counter()
    land = Counter()
    for a, s in zip(rs, seqs):
        if all(s):
            continue
        j = s.index(0)
        first_fail[j + 1] += 1
        rd = a["rounds"][j]
        mech["+".join(rd["wrong_bits"]) or "?"] += 1
        land[rd["S"]] += 1
    per_round = [sum(s[k] for s in seqs) / n for k in range(N_ROUNDS)]
    # Markov check over all transitions (S_0 = m* always)
    stay_n = stay_k = ret_n = ret_k = 0
    for s in seqs:
        prev = 1
        for x in s:
            if prev:
                stay_n += 1
                stay_k += x
            else:
                ret_n += 1
                ret_k += x
            prev = x
    surv = []
    for k in range(1, N_ROUNDS + 1):
        surv.append(sum(1 for s in seqs if all(s[:k])) / n)
    any_task_fail = sum(1 for a in rs for x in a["rounds"] if not x["task_ok"])
    return {
        "n": n,
        "cfg": Counter(a["cfg"] for a in rs).most_common(1)[0][0],
        "k_all5": k_all,
        "P_all5": k_all / n,
        "P_all5_wilson": wilson(k_all, n),
        "survival": surv,
        "per_round_on_target": per_round,
        "mean_rounds_on_target": float(np.mean([sum(s) for s in seqs])),
        "first_fail_round": dict(sorted(first_fail.items())),
        "first_fail_bits": dict(mech.most_common()),
        "first_fail_landing": dict(land.most_common()),
        "P_stay": stay_k / stay_n if stay_n else float("nan"),
        "n_stay": stay_n,
        "P_return": ret_k / ret_n if ret_n else float("nan"),
        "n_return": ret_n,
        "task_fail_rounds": any_task_fail,
    }


def _classify(base: dict, cal: dict) -> str:
    if not base.get("n"):
        return "not acquired"
    lo_b = base["P_all5_wilson"][0]
    lo_c, hi_c = cal["P_all5_wilson"] if cal.get("n") else (0, 0)
    if base["P_all5"] >= 0.8 and lo_b >= 0.5:
        return "naturally stable"
    if cal.get("n") and cal["P_all5"] >= 0.8 and lo_c >= 0.5:
        return "stable with steering"
    if cal.get("n") and lo_c > base["P_all5_wilson"][1]:
        return "steering helps, not stable"
    if cal.get("n") and cal["P_all5"] > 0:
        return "unstable (partial)"
    return "unstable"


def _paired_delta(trs: list[dict]) -> dict[str, Any]:
    pairs = [
        (all(x["on_target"] for x in t["arms"]["calibrated"]["rounds"]),
         all(x["on_target"] for x in t["arms"]["baseline"]["rounds"]))
        for t in trs
        if "calibrated" in t.get("arms", {}) and "baseline" in t.get("arms", {})
    ]
    b = sum(1 for c, z in pairs if c and not z)  # cal only
    c_ = sum(1 for c, z in pairs if z and not c)  # base only
    # exact two-sided McNemar
    n = b + c_
    if n == 0:
        p = 1.0
    else:
        from math import comb

        tail = sum(comb(n, i) for i in range(0, min(b, c_) + 1)) / 2**n
        p = min(1.0, 2 * tail)
    return {"n_pairs": len(pairs), "cal_only": b, "base_only": c_, "mcnemar_p": p}


def cmd_report(args) -> int:
    sel_payload = json.loads(SELECTION.read_text())
    sel = sel_payload["selection"]
    rows = _load_jsonl(EVAL_GLOB)
    # de-dup by id (keep last)
    by_id = {r["id"]: r for r in rows}
    rows = list(by_id.values())
    per = {}
    for m in ALL_MSTAR:
        mk = _key(m)
        trs = [r for r in rows if r["mstar"] == mk]
        n_att = len(trs)
        n_acq = sum(r["acquired"] for r in trs)
        acq_trs = [r for r in trs if r["acquired"]]
        base = _arm_stats(acq_trs, "baseline")
        cal = _arm_stats(acq_trs, "calibrated")
        per[mk] = {
            "n_attempts": n_att,
            "n_acq": n_acq,
            "P_acq": n_acq / n_att if n_att else float("nan"),
            "P_acq_wilson": wilson(n_acq, n_att),
            "n_free_S0_target": sum(r.get("free_S0_is_target", 0) for r in trs),
            "baseline": base,
            "calibrated": cal,
            "paired": _paired_delta(acq_trs),
            "selection": sel[mk],
            "class": _classify(base, cal),
            "P_final_calibrated": (n_acq / n_att) * cal["P_all5"] if (n_att and cal.get("n")) else 0.0,
        }
    payload = {
        "protocol": "Phase 10A eight-state retention map",
        "n_rounds": N_ROUNDS,
        "selection_rule": sel_payload["rule"],
        "n_cal_seeds": sel_payload["n_cal_seeds"],
        "per_mstar": per,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(_render_md(payload), encoding="utf-8")
    print(REPORT_MD.read_text())
    return 0


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{float(x):.{nd}f}"


def _ci(t):
    return f"[{_f(t[0])}, {_f(t[1])}]"


def _render_md(p: dict) -> str:
    per = p["per_mstar"]
    L = [
        "# Phase 10A — Eight-state retention map",
        "",
        f"> One protocol for all eight $m^*$: hybrid $\\Gamma'$+beam acquisition (9R, frozen), "
        f"verify $S_0=m^*$, then **{p['n_rounds']} rounds** under a frozen configuration. "
        "Baseline = no steering; calibrated = per-state config selected on a "
        f"disjoint calibration sweep ({p['n_cal_seeds']} seeds × 125 configs). "
        "Every round is run; nothing stops at first failure.",
        "",
        "Primary metric: $P(S_1=\\dots=S_5=m^*\\mid S_0=m^*)$ (Wilson 95% CI). "
        "Acquisition reported separately.",
        "",
        "## Retention map",
        "",
        "| $m^*$ | $P_{acq}$ (n) | $g^*$ $(g_C,g_H,g_O)$ | baseline $P_5$ | calibrated $P_5$ | 95% CI (cal) | McNemar p | class |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mk, s in per.items():
        b, c = s["baseline"], s["calibrated"]
        L.append(
            f"| `{mk}` | {_f(s['P_acq'])} ({s['n_acq']}/{s['n_attempts']}) | `{s['selection']['calibrated_cfg']}` | "
            f"{(str(b.get('k_all5')) + '/' + str(b['n'])) if b.get('n') else '—'} ({_f(b.get('P_all5'))}) | "
            f"**{(str(c.get('k_all5')) + '/' + str(c['n'])) if c.get('n') else '—'} ({_f(c.get('P_all5'))})** | "
            f"{_ci(c['P_all5_wilson']) if c.get('n') else '—'} | {_f(s['paired']['mcnemar_p'], 3)} | {s['class']} |"
        )
    L += [
        "",
        "$g$ = signed multiplier on the frozen 8K gains ($\\alpha_C=5,\\alpha_H=1.5,\\alpha_O=1.5$); 0 = channel off.",
        "",
        "## Survival curve $P(S_1..S_k=m^*)$",
        "",
        "| $m^*$ | arm | k=1 | k=2 | k=3 | k=4 | k=5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for mk, s in per.items():
        for arm in ("baseline", "calibrated"):
            a = s[arm]
            if not a.get("n"):
                continue
            L.append(f"| `{mk}` | {arm} | " + " | ".join(_f(x) for x in a["survival"]) + " |")
    L += [
        "",
        "## Failure analysis (first off-target round)",
        "",
        "| $m^*$ | arm | first-fail round counts | wrong bits at first fail | landed in |",
        "|---|---|---|---|---|",
    ]
    for mk, s in per.items():
        for arm in ("baseline", "calibrated"):
            a = s[arm]
            if not a.get("n") or a["k_all5"] == a["n"]:
                continue
            L.append(
                f"| `{mk}` | {arm} | {a['first_fail_round']} | {a['first_fail_bits']} | {a['first_fail_landing']} |"
            )
    L += [
        "",
        "## Are rounds independent? (Markov check, calibrated arm)",
        "",
        "If $P(\\text{on}\\mid\\text{prev on})\\approx P(\\text{on}\\mid\\text{prev off})$ and "
        "$P_5\\approx p^5$, retention is a per-round attainment rate, not a held state.",
        "",
        "| $m^*$ | $P_{cal}$ (calib.) | $p^5$ pred. | $P_5$ obs. | $P(\\text{on}\\mid\\text{prev on})$ (n) | $P(\\text{on}\\mid\\text{prev off})$ (n) |",
        "|---|---|---|---|---|---|",
    ]
    for mk, s in per.items():
        c = s["calibrated"]
        if not c.get("n"):
            continue
        L.append(
            f"| `{mk}` | {_f(s['selection']['P_cal'])} | {_f(s['selection']['predicted_P5_indep'])} | "
            f"{_f(c['P_all5'])} | {_f(c['P_stay'])} ({c['n_stay']}) | {_f(c['P_return'])} ({c['n_return']}) |"
        )
    L += [
        "",
        "## Minimum effective configuration (calibration sweep only)",
        "",
        "| $m^*$ | baseline $P_{cal}$ | best 1-ch | best 2-ch | best 3-ch | min-effective (≤1 count of best) |",
        "|---|---|---|---|---|---|",
    ]
    for mk, s in per.items():
        sl = s["selection"]
        bb = sl["best_by_n_active"]

        def cell(na):
            d = bb.get(str(na)) or bb.get(na)
            return f"`{d['cfg']}` {_f(d['P'])}" if d else "—"

        L.append(
            f"| `{mk}` | {_f(sl['baseline_P_cal'])} | {cell(1)} | {cell(2)} | {cell(3)} | "
            f"`{sl['min_effective_cfg']}` {_f(sl['min_effective_P_cal'])} |"
        )
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("calibrate")
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--n-shards", type=int, default=1)
    a.add_argument("--n-cal", type=int, default=16)
    sub.add_parser("select")
    e = sub.add_parser("evaluate")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--n-shards", type=int, default=1)
    e.add_argument("--n-attempts", type=int, default=60)
    e.add_argument("--max-steps-acq", type=int, default=7)
    sub.add_parser("report")
    args = ap.parse_args()
    return {
        "calibrate": cmd_calibrate,
        "select": cmd_select,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
