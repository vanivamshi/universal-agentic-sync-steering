#!/usr/bin/env python3
"""Phase 9P — Active producer densification for hard sinks {000,001,110}.

Kernel was censored: rare sink-ins underrepresented → planner has no mass.
Do NOT change v or planner objective. Actively measure known producers with
correct target-sign for the hard m* (not bit-flip densify).

Protocol:
1) Mine observed s--a--> hard landings from logs/kernels
2) Construct source (home soft / bridge 111), apply a with m*-target-sign, reps
3) Beta(k+1, n-k+1) CI; producer table
4) Live hard acq: a* = argmax_a P(S'=m*|s,a) from updated kernel (all channels)

  .venv/bin/python -u scripts/run_phase9p_producer_densify.py --reps 12
  .venv/bin/python -u scripts/run_phase9p_producer_densify.py --live-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9p_producer_densify.json"
MD = ROOT / "data" / "results" / "sync_phase9p_producer_densify.md"
KERNEL_OUT = ROOT / "data" / "results" / "sync_phase9p_kernel.json"
SEED = 20261004
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
SOFT = {(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)}
HARD = [(0, 0, 0), (0, 0, 1), (1, 1, 0)]
HARD_KEYS = {"000", "001", "110"}

# Priority producers to densify (from history + mining); (source, action, target)
PRIORITY: list[tuple[tuple[int, int, int], str, tuple[int, int, int]]] = [
    ((0, 1, 1), "H", (0, 0, 0)),  # 011|H → 000
    ((0, 1, 1), "C", (0, 0, 0)),  # 011|C → 000 (mined)
    ((1, 1, 1), "H", (0, 0, 0)),  # 111|H → 000
    ((0, 1, 1), "C", (0, 0, 1)),  # 011|C → 001
    ((1, 1, 1), "C", (1, 1, 0)),  # 111|C → 110
    ((1, 1, 1), "H", (1, 1, 0)),  # 111|H → 110
    ((0, 1, 1), "H", (1, 1, 0)),  # 011|H → 110
]


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


def _parse(sk: str) -> tuple[int, int, int]:
    return (int(sk[0]), int(sk[1]), int(sk[2]))


def _E(mstar, S) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _neighbors(s):
    out = []
    for i in range(3):
        t = list(s)
        t[i] = 1 - t[i]
        out.append(tuple(t))
    return out


def beta_stats(k: int, n: int) -> dict[str, float]:
    """Beta(k+1, n-k+1) mean and central 90% CI via quantiles (approx)."""
    if n <= 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None, "hat_p": None}
    # posterior
    a, b = k + 1, n - k + 1
    # sample for quantiles
    rng = np.random.default_rng(0)
    draws = rng.beta(a, b, size=20000)
    return {
        "hat_p": k / n,
        "mean": float(a / (a + b)),
        "p05": float(np.quantile(draws, 0.05)),
        "p50": float(np.quantile(draws, 0.50)),
        "p95": float(np.quantile(draws, 0.95)),
    }


def mine_producers() -> dict[str, Any]:
    """Collect s|a→hard counts from kernels + paired steps."""
    counts: Counter = Counter()
    sources: dict[tuple, set] = defaultdict(set)

    def add(sb, a, sa, srcfile):
        if a not in ORDER:
            return
        sk, tk = _key(sb), _key(sa)
        if tk not in HARD_KEYS:
            return
        key = (tk, sk, a)
        counts[key] += 1
        sources[key].add(srcfile)

    for path in (ROOT / "data" / "results").glob("sync_phase9*.json"):
        try:
            j = json.loads(path.read_text())
        except Exception:
            continue
        for r in j.get("extra_rows") or []:
            if "S_before" in r and "S_after" in r and "a" in r:
                add(r["S_before"], r["a"], r["S_after"], path.name)
        for key in ("paired", "live_paired", "hard_paired", "trials", "full_trials"):
            items = j.get(key) or []
            if not isinstance(items, list):
                continue
            for p in items:
                for pol, t in list(p.items()):
                    if not isinstance(t, dict) or "steps" not in t:
                        continue
                    for st in t["steps"]:
                        if st.get("a") and "S_before" in st and "S_after" in st:
                            add(st["S_before"], st["a"], st["S_after"], f"{path.name}:{pol}")

    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    for r in p9e._load_transitions():
        add(r["S_before"], r["a"], r["S_after"], "p9e")

    by_target: dict[str, list] = defaultdict(list)
    for (tk, sk, a), n in sorted(counts.items(), key=lambda x: -x[1]):
        by_target[tk].append(
            {
                "target": tk,
                "source": sk,
                "a": a,
                "n_mined": n,
                "files": sorted(sources[(tk, sk, a)])[:6],
            }
        )
    return {"counts": by_target, "n_edges": len(counts)}


def construct_source(p9, p8k, sc, loaded, *, dirs, source, seed):
    if tuple(source) in SOFT:
        home = p9._home_to_source(
            p8k, sc, loaded, dirs=dirs, source=source, seed=seed
        )
        return {
            "ok": int(home["homed"]),
            "S": list(home["S_home"]),
            "plan": home.get("plan_prefill") or "",
            "method": "home" if home["homed"] else "home_miss",
        }
    # hard e.g. 111: soft bridge
    bridges = [br for br in _neighbors(source) if br in SOFT]
    if not bridges:
        return {"ok": 0, "S": [0, 0, 0], "plan": "", "method": "no_bridge"}
    br = bridges[seed % len(bridges)]
    bh = p9._home_to_source(p8k, sc, loaded, dirs=dirs, source=br, seed=seed)
    if not bh["homed"]:
        return {
            "ok": 0,
            "S": list(bh["S_home"]),
            "plan": bh.get("plan_prefill") or "",
            "method": "bridge_home_miss",
        }
    edge = p9._run_one_bit_transition(
        p8k,
        sc,
        loaded,
        dirs=dirs,
        from_s=br,
        to_s=source,
        seed=seed + 9,
        plan_prefill=bh["plan_prefill"],
    )
    if edge["S"] == list(source):
        return {
            "ok": 1,
            "S": list(source),
            "plan": edge.get("plan_prefill") or bh["plan_prefill"] or "",
            "method": f"bridge_{_key(br)}",
        }
    return {
        "ok": 0,
        "S": list(edge["S"]),
        "plan": edge.get("plan_prefill") or bh.get("plan_prefill") or "",
        "method": "bridge_miss",
    }


def densify_producer(
    p9,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    source: tuple[int, int, int],
    action: str,
    mstar: tuple[int, int, int],
    reps: int,
    seed0: int,
    max_att: int,
) -> list[dict]:
    """Construct source, apply action with m*-target-sign, record landing."""
    rows = []
    n_ok = 0
    attempts = 0
    while n_ok < reps and attempts < max_att:
        seed = seed0 + 17 * attempts
        attempts += 1
        cst = construct_source(
            p9, p8k, sc, loaded, dirs=dirs, source=source, seed=seed
        )
        if not cst["ok"] or tuple(cst["S"]) != tuple(source):
            print(
                f"    miss construct {_key(source)}|{action} "
                f"att={attempts} method={cst['method']} S={_key(cst['S'])}",
                flush=True,
            )
            continue
        active = {action: (p8k._s_star(mstar[CH_IDX[action]]), dirs[action])}
        h_pf = cst["plan"] if action == "H" else None
        out = p8k._run_episode(
            sc, loaded, active=active, seed=seed + 7, h_plan_prefill=h_pf
        )
        Sa = tuple(out["S"])
        row = {
            "src": "9P",
            "edge": f"{_key(source)}|{action}",
            "S_before": list(source),
            "S_after": list(Sa),
            "a": action,
            "mstar": list(mstar),
            "to_intended": list(mstar),
            "land_hard": _key(Sa) if _key(Sa) in HARD_KEYS else None,
            "hit_target": int(Sa == tuple(mstar)),
            "dS": int(Sa != source),
            "method": cst["method"],
            "seed": seed,
        }
        rows.append(row)
        n_ok += 1
        print(
            f"    {_key(source)}|{action}→{_key(mstar)} n={n_ok}/{reps} "
            f"→{_key(Sa)} hit={row['hit_target']} via={cst['method']}",
            flush=True,
        )
    return rows


def build_producer_table(rows: list[dict], mined: dict) -> dict[str, list]:
    """Per target: (source,a) stats with Beta CI, merging 9P rows."""
    # group 9P by (target mstar, source, a)
    grp: dict[tuple, list] = defaultdict(list)
    for r in rows:
        mk = _key(r["mstar"])
        sk = _key(r["S_before"])
        grp[(mk, sk, r["a"])].append(r)

    table: dict[str, list] = {}
    for mstar in HARD:
        mk = _key(mstar)
        # candidates: priority + mined
        cands = set()
        for src, a, tgt in PRIORITY:
            if tuple(tgt) == tuple(mstar):
                cands.add((_key(src), a))
        for edge in mined.get("counts", {}).get(mk, []):
            cands.add((edge["source"], edge["a"]))

        entries = []
        for sk, a in sorted(cands):
            trials = grp.get((mk, sk, a), [])
            n = len(trials)
            k = sum(1 for t in trials if t.get("hit_target"))
            # also count landings on mk even if listed under other — hit_target is exact
            hist = Counter(_key(t["S_after"]) for t in trials)
            st = beta_stats(k, n) if n else beta_stats(0, 0)
            mined_n = 0
            for e in mined.get("counts", {}).get(mk, []):
                if e["source"] == sk and e["a"] == a:
                    mined_n = e["n_mined"]
            entries.append(
                {
                    "source": sk,
                    "a": a,
                    "n": n,
                    "k": k,
                    "hat_p": st.get("hat_p"),
                    "beta_mean": st.get("mean"),
                    "p05": st.get("p05"),
                    "p95": st.get("p95"),
                    "after_hist": dict(hist),
                    "n_mined_prior": mined_n,
                    "score": st.get("p05") if st.get("p05") is not None else -1.0,
                }
            )
        entries.sort(key=lambda e: (e["score"] if e["score"] is not None else -1, e["n"]), reverse=True)
        table[mk] = entries
    return table


def choose_max_pland(s, mstar, land_p, *, a_prev, stagnated):
    """a* = argmax P(S'=m*|s,a) from land_p dict (sk,a,mk)->stats."""
    sk = _key(s)
    mk = _key(mstar)
    meta = {"mode": "max_P_land", "by_a": {}, "blocked": None}
    if sk == mk:
        return None, meta
    cands = []
    for a in ORDER:
        if stagnated and a_prev == a:
            meta["blocked"] = a
            continue
        st = land_p.get((sk, a, mk))
        if not st or st.get("n", 0) < 1:
            meta["by_a"][a] = {"p": None, "n": 0}
            continue
        # use beta lower bound for conservative selection
        p = st.get("p05") if st.get("p05") is not None else st.get("hat_p")
        meta["by_a"][a] = {"p": p, "hat_p": st.get("hat_p"), "n": st.get("n")}
        if p is not None:
            cands.append((a, float(p), int(st["n"])))
    if not cands:
        # fallback any with hat_p
        for a in ORDER:
            st = land_p.get((sk, a, mk))
            if st and st.get("hat_p") is not None:
                cands.append((a, float(st["hat_p"]), int(st["n"])))
        meta["fallback"] = True
    if not cands:
        return None, meta
    cands.sort(key=lambda t: (t[1], t[2], -ORDER.index(t[0])), reverse=True)
    meta["a_star"] = cands[0][0]
    meta["p_star"] = cands[0][1]
    return cands[0][0], meta


def land_p_from_table(table: dict, rows: list[dict]) -> dict:
    """(sk,a,mk) -> beta stats from 9P densify rows for that m* sign."""
    grp: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (_key(r["S_before"]), r["a"], _key(r["mstar"]))
        grp[key].append(r)
    out = {}
    for key, trials in grp.items():
        n = len(trials)
        k = sum(1 for t in trials if t.get("hit_target"))
        st = beta_stats(k, n)
        out[key] = {"n": n, "k": k, **st}
    return out


def run_live_acq(
    p9j,
    p8k,
    sc,
    loaded,
    *,
    dirs,
    land_p,
    lookup,
    mstar,
    seed,
    S0,
    plan0,
    max_steps,
    cache,
    use_hitting: bool,
):
    S = list(S0)
    plan = plan0 or ""
    traj = [list(S)]
    steps = []
    a_prev = None
    stagnated = False
    ever = int(S == list(mstar))
    for t in range(max_steps):
        if S == list(mstar):
            ever = 1
            break
        if use_hitting:
            a, meta = choose_max_pland(
                S, mstar, land_p, a_prev=a_prev, stagnated=stagnated
            )
            if a is None:
                # fallback Γ'
                a, meta = p9j.choose_gamma_prime(
                    S, mstar, lookup, a_prev=a_prev, stagnated=stagnated
                )
                meta["fallback_to"] = "gamma_prime"
        else:
            a, meta = p9j.choose_gamma_prime(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated
            )
        if a is None:
            break
        out, hit = p9j.apply_channel(
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
        if Sa == list(mstar):
            ever = 1
        steps.append(
            {
                "t": t,
                "a": a,
                "S_before": list(S),
                "S_after": Sa,
                "mode": meta.get("mode"),
                "p_star": meta.get("p_star"),
                "cache_hit": hit,
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj.append(list(S))
        if ever:
            break
    return {
        "mstar_key": _key(mstar),
        "path": [_key(s) for s in traj],
        "actions": [s["a"] for s in steps],
        "hit": int(S == list(mstar)),
        "ever": ever,
        "E0": _E(mstar, S0),
        "E_final": _E(mstar, S),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=12, help="densify trials per producer")
    ap.add_argument("--max-att", type=int, default=0)
    ap.add_argument("--reps-live", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--densify-only", action="store_true")
    ap.add_argument("--live-only", action="store_true")
    args = ap.parse_args()
    max_att = args.max_att or max(24, args.reps * 4)

    mined = mine_producers()
    print(f"=== Mined producer edges: {mined['n_edges']} ===", flush=True)
    for mk in ("000", "001", "110"):
        top = (mined["counts"].get(mk) or [])[:5]
        print(f"  {mk}: {[(e['source'], e['a'], e['n_mined']) for e in top]}", flush=True)

    new_rows: list[dict] = []
    if KERNEL_OUT.exists() and args.live_only:
        new_rows = json.loads(KERNEL_OUT.read_text()).get("extra_rows") or []
        print(f"loaded {len(new_rows)} prior 9P rows", flush=True)

    if not args.live_only:
        from activation_pipeline.device import (
            LOCAL_MODEL_KEY,
            assert_model_fits_machine,
            resolve_device_map,
        )
        from activation_pipeline.loader import load_model_and_tokenizer

        p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
        p9 = _load_mod("phase9", ROOT / "scripts" / "run_phase9_hamming_paths.py")
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

        # resume prior
        prior = []
        if KERNEL_OUT.exists():
            prior = json.loads(KERNEL_OUT.read_text()).get("extra_rows") or []
            print(f"resumed {len(prior)} 9P rows", flush=True)

        # count prior per (source,a,mstar)
        prior_n = Counter(
            (_key(r["S_before"]), r["a"], _key(r["mstar"])) for r in prior
        )
        new_rows = list(prior)
        print(f"=== Densify {len(PRIORITY)} producers × reps={args.reps} ===", flush=True)
        for i, (source, action, mstar) in enumerate(PRIORITY):
            have = prior_n[(_key(source), action, _key(mstar))]
            need = max(0, args.reps - have)
            print(
                f"  [{i+1}/{len(PRIORITY)}] {_key(source)}|{action}→{_key(mstar)} "
                f"have={have} need={need}",
                flush=True,
            )
            if need <= 0:
                continue
            rows = densify_producer(
                p9,
                p8k,
                sc,
                loaded,
                dirs=dirs,
                source=source,
                action=action,
                mstar=mstar,
                reps=need,
                seed0=args.seed + 1000 * i,
                max_att=max_att,
            )
            new_rows.extend(rows)
            KERNEL_OUT.write_text(
                json.dumps({"extra_rows": new_rows}, indent=2), encoding="utf-8"
            )

    table = build_producer_table(new_rows, mined)
    land_p = land_p_from_table(table, new_rows)
    print("=== Producer table (9P densify, Beta p05) ===", flush=True)
    for mk in ("000", "001", "110"):
        print(f"  m*={mk}", flush=True)
        for e in table[mk][:6]:
            print(
                f"    {e['source']}|{e['a']}: n={e['n']} k={e['k']} "
                f"hat={e['hat_p']} p05={e['p05']} hist={e['after_hist']}",
                flush=True,
            )

    live = {"agg": {}, "per": {}, "paired": []}
    if not args.densify_only:
        from activation_pipeline.device import (
            LOCAL_MODEL_KEY,
            assert_model_fits_machine,
            resolve_device_map,
        )
        from activation_pipeline.loader import load_model_and_tokenizer

        p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
        p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
        p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
        p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
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
        # merge 9P into lookup extras for Γ' fallback kernel
        extras = []
        for r in new_rows:
            extras.append(
                {
                    "src": "9P",
                    "edge": r["edge"],
                    "S_before": tuple(r["S_before"]),
                    "S_after": tuple(r["S_after"]),
                    "a": r["a"],
                    "to_intended": tuple(r.get("mstar") or r["S_after"]),
                }
            )
        lookup, _, _ = p9g.build_lookup(p9e, extras if extras else None)
        # also rebuild land_p — already from 9P target-sign rows

        by_pol = {"max_pland": [], "gamma_prime": []}
        by_m = defaultdict(lambda: {"max_pland": [], "gamma_prime": []})
        n_tot = len(HARD) * args.reps_live
        idx = 0
        print(
            f"=== Live hard max-P(land) vs Γ' reps={args.reps_live} ===",
            flush=True,
        )
        for mi, mstar in enumerate(HARD):
            for r in range(args.reps_live):
                seed0 = args.seed + 400 * mi + 19 * r
                S0, plan0, seed = p9j.free_S0(
                    p8k, sc, loaded, mstar=mstar, seed0=seed0
                )
                idx += 1
                print(
                    f"  [{idx}/{n_tot}] m*={_key(mstar)} S0={_key(S0)}",
                    flush=True,
                )
                if S0 == list(mstar):
                    continue
                cache: dict = {}
                trials = {}
                for name, use_h in (("max_pland", True), ("gamma_prime", False)):
                    trials[name] = run_live_acq(
                        p9j,
                        p8k,
                        sc,
                        loaded,
                        dirs=dirs,
                        land_p=land_p,
                        lookup=lookup,
                        mstar=mstar,
                        seed=seed,
                        S0=S0,
                        plan0=plan0,
                        max_steps=args.max_steps,
                        cache=cache,
                        use_hitting=use_h,
                    )
                    t = trials[name]
                    print(
                        f"    {name}: ever={t['ever']} hit={t['hit']} "
                        f"path={t['path']} acts={t['actions']}",
                        flush=True,
                    )
                    by_pol[name].append(t)
                    by_m[_key(mstar)][name].append(t)
                live["paired"].append(
                    {"mstar": list(mstar), "S0": S0, "seed": seed, **trials}
                )

        def agg(xs):
            if not xs:
                return {"n": 0}
            return {
                "n": len(xs),
                "P_ever": float(np.mean([t["ever"] for t in xs])),
                "P_hit": float(np.mean([t["hit"] for t in xs])),
            }

        live["agg"] = {k: agg(v) for k, v in by_pol.items()}
        live["per"] = {
            _key(m): {k: agg(by_m[_key(m)][k]) for k in by_pol}
            for m in HARD
        }

    n_ever = sum(
        1
        for mk in ("000", "001", "110")
        if (live.get("per", {}).get(mk, {}).get("max_pland", {}).get("P_ever") or 0) > 0
    )

    # best producer per target by p05
    best = {}
    for mk in ("000", "001", "110"):
        ents = [e for e in table[mk] if e["n"] > 0]
        best[mk] = ents[0] if ents else None

    gate = {
        "hypothesis": (
            "Active target-sign densify of producer edges restores kernel mass "
            "into hard sinks; max P(land) recovers P_acq"
        ),
        "best_producers": best,
        "n_9p_rows": len(new_rows),
        "live_P_ever_max_pland": {
            mk: live.get("per", {}).get(mk, {}).get("max_pland", {}).get("P_ever")
            for mk in ("000", "001", "110")
        },
        "n_hard_ever_max_pland": n_ever if live.get("per") else None,
        "hard_acq_gate": n_ever == 3 if live.get("per") else None,
        "read": (
            "Primary: densified hat_p / p05 for producers. "
            "Secondary: live P_ever under max-P(land)."
        ),
    }

    payload = {
        "protocol": "Phase 9P active producer densification",
        "seed": args.seed,
        "reps": args.reps,
        "mined": mined,
        "producer_table": table,
        "n_rows": len(new_rows),
        "live": live,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if new_rows:
        KERNEL_OUT.write_text(
            json.dumps({"extra_rows": new_rows, "producer_table": table}, indent=2),
            encoding="utf-8",
        )

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9P — Active producer densification",
        "",
        "> Construct source → apply action with **hard $m^*$ target-sign** → "
        r"estimate $P(m^*\mid s,a)$ via Beta. No new $v$.",
        "",
        f"densify_reps={args.reps}, live_reps={args.reps_live}, seed={args.seed}.",
        "",
        "## Producer table (9P densify)",
        "",
        "| target | source | $a$ | $n$ | $k$ | $\\hat p$ | Beta mean | $p_{05}$ | $p_{95}$ | hist |",
        "|--------|--------|----:|----:|----:|----------:|----------:|---------:|---------:|------|",
    ]
    for mk in ("000", "001", "110"):
        for e in table[mk]:
            if e["n"] == 0 and e.get("n_mined_prior", 0) == 0:
                continue
            hist = " ".join(f"{k}:{v}" for k, v in list((e.get("after_hist") or {}).items())[:4])
            lines.append(
                f"| `{mk}` | `{e['source']}` | `{e['a']}` | {e['n']} | {e['k']} | "
                f"{fmt(e.get('hat_p'))} | {fmt(e.get('beta_mean'))} | "
                f"**{fmt(e.get('p05'))}** | {fmt(e.get('p95'))} | {hist} |"
            )
    lines += ["", "## Best by $p_{05}$", ""]
    for mk, e in best.items():
        if not e:
            lines.append(f"- `{mk}`: no densified mass")
        else:
            lines.append(
                f"- `{mk}`: `{e['source']}|{e['a']}` $\\hat p$={fmt(e.get('hat_p'))} "
                f"$p_{{05}}$={fmt(e.get('p05'))} (n={e['n']})"
            )
    if live.get("per"):
        lines += [
            "",
            "## Live hard $P_{acq}$ (ever)",
            "",
            "| $m^*$ | max $P(\\mathrm{land})$ | $\\Gamma'$ |",
            "|-------|------------------------:|----------:|",
        ]
        for mk in ("000", "001", "110"):
            st = live["per"][mk]
            lines.append(
                f"| `{mk}` | **{fmt(st['max_pland'].get('P_ever'))}** | "
                f"{fmt(st['gamma_prime'].get('P_ever'))} |"
            )
        lines += [
            "",
            f"Hard acq gate (ever>0 all 3): **{gate['hard_acq_gate']}** "
            f"({gate['n_hard_ever_max_pland']}/3)",
            "",
        ]
    lines += [
        gate["read"],
        "",
        r"$$\boxed{\text{active producer densify}\rightarrow\hat P(s\xrightarrow{a}m^*)}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
