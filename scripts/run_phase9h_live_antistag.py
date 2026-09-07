#!/usr/bin/env python3
"""Phase 9H — Live Γ' (anti-stagnation) vs Γ vs fixed.

Exactly one safeguard beyond Phase 9F Γ:

  Γ'(a) = -∞  if a == a_{t-1} and S_t == S_{t-1}
          Γ(a|S_t,m*) otherwise

Frozen 8K actuators. Expanded 9G kernel. Scoped targets {010,011,100,101,110}.
Paired free S0 (E0>0). Three-way: Γ' vs Γ vs fixed.

  .venv/bin/python scripts/run_phase9h_live_antistag.py --reps 4
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

OUT = ROOT / "data" / "results" / "sync_phase9h_live_antistag.json"
MD = ROOT / "data" / "results" / "sync_phase9h_live_antistag.md"
KERNEL_9G = ROOT / "data" / "results" / "sync_phase9g_kernel.json"

SEED = 20260927  # new seed family; same protocol as 9F
ORDER = ("H", "C", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}
CHANNELS = ORDER

SCOPED_MSTAR = (
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
)

MAX_STEPS = 4
MIN_N = 1
POLICIES = ("gamma", "gamma_prime", "fixed")  # gamma first so Γ' often cache-hits


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


def _E(mstar, S) -> int:
    return sum(int(a) != int(b) for a, b in zip(mstar, S))


def _relevant(s, mstar) -> list[str]:
    return [a for a in ORDER if int(s[CH_IDX[a]]) != int(mstar[CH_IDX[a]])]


def _fixed_action(s, mstar) -> str | None:
    rel = _relevant(s, mstar)
    return rel[0] if rel else None


def load_lookup_with_9g(p9e, p9g):
    extra = []
    if KERNEL_9G.exists():
        raw = json.loads(KERNEL_9G.read_text()).get("extra_rows") or []
        for r in raw:
            extra.append(
                {
                    "src": "9G",
                    "edge": r.get("edge"),
                    "S_before": tuple(r["S_before"]),
                    "S_after": tuple(r["S_after"]),
                    "a": r["a"],
                    "to_intended": tuple(r.get("to_intended") or r["S_after"]),
                }
            )
    lookup, ker, rows = p9g.build_lookup(p9e, extra if extra else None)
    return lookup, ker, len(extra)


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
    meta: dict[str, Any] = {
        "relevant": rel,
        "by_a": {},
        "blocked": None,
        "fallback": False,
        "anti_stag": anti_stag,
    }
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
        if int(st.get("n") or 0) >= MIN_N and g is not None:
            cands.append((a, float(g), int(st["n"])))
    if not cands:
        for a in rel:
            if anti_stag and stagnated and a == a_prev:
                continue
            meta["fallback"] = True
            meta["a_star"] = a
            return a, meta
        meta["fallback"] = True
        meta["a_star"] = rel[0]
        meta["forced_blocked"] = True
        return rel[0], meta
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
    dirs,
    mstar,
    channel,
    seed,
    plan_prefill,
    cache: dict | None = None,
):
    """Apply one channel; cache by (seed, channel, plan) to share across policies."""
    key = (int(seed), channel, plan_prefill or "")
    if cache is not None and key in cache:
        return cache[key]
    active = {channel: (p8k._s_star(mstar[CH_IDX[channel]]), dirs[channel])}
    h_pf = plan_prefill if channel == "H" else None
    out = p8k._run_episode(sc, loaded, active=active, seed=seed, h_plan_prefill=h_pf)
    if cache is not None:
        cache[key] = out
    return out


def run_policy_trial(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    lookup,
    mstar,
    seed,
    policy: str,
    max_steps: int,
    S0: list[int],
    plan0: str,
    cache: dict | None = None,
) -> dict[str, Any]:
    S = list(S0)
    plan = plan0 or ""
    traj_S = [list(S)]
    traj_E = [_E(mstar, S)]
    steps = []
    a_prev = None
    stagnated = False
    n_noop_contexts = 0
    n_repeat_after_noop = 0
    n_rescue_after_noop = 0
    n_cache_hits = 0

    for t in range(max_steps):
        if S == list(mstar):
            break
        prev_was_noop = bool(stagnated and a_prev is not None)
        if policy == "fixed":
            a = _fixed_action(S, mstar)
            meta = {"relevant": _relevant(S, mstar), "a_star": a, "blocked": None}
        elif policy == "gamma":
            a, meta = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=False
            )
        elif policy == "gamma_prime":
            a, meta = choose_action(
                S, mstar, lookup, a_prev=a_prev, stagnated=stagnated, anti_stag=True
            )
        else:
            raise ValueError(policy)
        if a is None:
            break

        if prev_was_noop:
            n_noop_contexts += 1
            if a == a_prev:
                n_repeat_after_noop += 1

        Eb = _E(mstar, S)
        ep_seed = seed + 1 + t
        cache_key = (int(ep_seed), a, plan or "")
        used_cache = cache is not None and cache_key in cache
        out = apply_channel(
            p8k,
            sc,
            loaded,
            dirs=dirs,
            mstar=mstar,
            channel=a,
            seed=ep_seed,
            plan_prefill=plan,
            cache=cache,
        )
        if used_cache:
            n_cache_hits += 1
        Sa = list(out["S"])
        if out.get("plan_prefill"):
            plan = out["plan_prefill"]
        Ea = _E(mstar, Sa)
        dS = int(Sa != S)
        down = int(Ea < Eb)
        rescued = int(prev_was_noop and a != a_prev and (dS == 1 or down == 1))
        if rescued:
            n_rescue_after_noop += 1

        steps.append(
            {
                "t": t,
                "a": a,
                "policy": policy,
                "S_before": list(S),
                "S_after": Sa,
                "E_before": Eb,
                "E_after": Ea,
                "dE": Ea - Eb,
                "dS": dS,
                "down": down,
                "up": int(Ea > Eb),
                "neutral": int(Ea == Eb),
                "prev_was_noop": int(prev_was_noop),
                "repeat_after_noop": int(prev_was_noop and a == a_prev),
                "rescue_after_noop": rescued,
                "blocked": meta.get("blocked"),
                "cached": int(used_cache),
                "meta": {k: v for k, v in meta.items() if k != "by_a"},
                "by_a": meta.get("by_a"),
            }
        )
        stagnated = Sa == S
        a_prev = a
        S = Sa
        traj_S.append(list(S))
        traj_E.append(Ea)

    E0, Ef = traj_E[0], traj_E[-1]
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
        "delta_E": int(E0 - Ef),
        "delta_E_norm": float(
            sync_error_norm(p8k._e(mstar, traj_S[0]))
            - sync_error_norm(p8k._e(mstar, traj_S[-1]))
        ),
        "E0": E0,
        "E_final": Ef,
        "n_steps": len(steps),
        "actions": [s["a"] for s in steps],
        "P_down_steps": float(np.mean([s["down"] for s in steps])) if steps else float("nan"),
        "P_up_steps": float(np.mean([s["up"] for s in steps])) if steps else float("nan"),
        "P_shift_steps": float(np.mean([s["dS"] for s in steps])) if steps else float("nan"),
        "mean_step_dE": float(np.mean([s["dE"] for s in steps])) if steps else float("nan"),
        "n_noop_contexts": n_noop_contexts,
        "n_repeat_after_noop": n_repeat_after_noop,
        "n_rescue_after_noop": n_rescue_after_noop,
        "P_repeat_after_noop": (
            n_repeat_after_noop / n_noop_contexts if n_noop_contexts else float("nan")
        ),
        "P_rescue_given_noop": (
            n_rescue_after_noop / n_noop_contexts if n_noop_contexts else float("nan")
        ),
        "n_cache_hits": n_cache_hits,
    }


def _agg(trials: list[dict]) -> dict[str, Any]:
    if not trials:
        return {"n": 0}

    def m(k):
        xs = [t[k] for t in trials if t.get(k) == t.get(k)]
        return float(np.mean(xs)) if xs else float("nan")

    n_noop = sum(t["n_noop_contexts"] for t in trials)
    n_rep = sum(t["n_repeat_after_noop"] for t in trials)
    n_res = sum(t["n_rescue_after_noop"] for t in trials)
    return {
        "n": len(trials),
        "P_hit": m("hit"),
        "mean_delta_E": m("delta_E"),
        "mean_E_final": m("E_final"),
        "mean_E0": m("E0"),
        "mean_P_down_steps": m("P_down_steps"),
        "mean_P_shift_steps": m("P_shift_steps"),
        "mean_step_dE": m("mean_step_dE"),
        "mean_n_steps": m("n_steps"),
        "n_noop_contexts": n_noop,
        "P_repeat_after_noop": n_rep / n_noop if n_noop else float("nan"),
        "P_rescue_given_noop": n_res / n_noop if n_noop else float("nan"),
        "n_repeat_after_noop": n_rep,
        "n_rescue_after_noop": n_res,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()
    lookup, ker, n_extra = load_lookup_with_9g(p9e, p9g)

    print(
        f"=== Phase 9H Γ' vs Γ vs fixed | targets={[ _key(m) for m in SCOPED_MSTAR ]} "
        f"reps={args.reps} 9G_extra={n_extra} ker_cells={len(ker)} ===",
        flush=True,
    )

    paired: list[dict] = []
    by_policy: dict[str, list] = {p: [] for p in POLICIES}
    by_mstar: dict[str, dict[str, list]] = defaultdict(lambda: {p: [] for p in POLICIES})

    idx = 0
    n_total = len(SCOPED_MSTAR) * args.reps
    for mi, mstar in enumerate(SCOPED_MSTAR):
        for r in range(args.reps):
            seed0 = args.seed + 100 * mi + 17 * r
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
            idx += 1
            print(
                f"  [{idx}/{n_total}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                f"E0={_E(mstar, S0)} seed={seed}",
                flush=True,
            )
            if S0 == list(mstar):
                print("    skip: S0==m*", flush=True)
                continue

            trials = {}
            ep_cache: dict = {}
            for pol in POLICIES:
                trials[pol] = run_policy_trial(
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    lookup=lookup,
                    mstar=mstar,
                    seed=seed,
                    policy=pol,
                    max_steps=args.max_steps,
                    S0=S0,
                    plan0=plan0,
                    cache=ep_cache,
                )
                tag = {"gamma_prime": "Γ'", "gamma": "Γ", "fixed": "f"}[pol]
                t = trials[pol]
                print(
                    f"    {tag}: hit={t['hit']} E={t['E0']}→{t['E_final']} "
                    f"acts={t['actions']} path={[ _key(s) for s in t['traj_S'] ]} "
                    f"rep_noop={t['n_repeat_after_noop']}/{t['n_noop_contexts']} "
                    f"rescue={t['n_rescue_after_noop']} cache={t.get('n_cache_hits', 0)}",
                    flush=True,
                )
                by_policy[pol].append(t)
                by_mstar[_key(mstar)][pol].append(t)
            print(f"    ep_cache size={len(ep_cache)}", flush=True)

            paired.append(
                {
                    "mstar": list(mstar),
                    "mstar_key": _key(mstar),
                    "seed": seed,
                    "S0": S0,
                    **trials,
                    "delta_hit_prime_minus_gamma": trials["gamma_prime"]["hit"] - trials["gamma"]["hit"],
                    "delta_hit_prime_minus_fixed": trials["gamma_prime"]["hit"] - trials["fixed"]["hit"],
                    "delta_hit_gamma_minus_fixed": trials["gamma"]["hit"] - trials["fixed"]["hit"],
                }
            )

    aggs = {p: _agg(by_policy[p]) for p in POLICIES}
    per_m = {
        mk: {p: _agg(v[p]) for p in POLICIES}
        | {
            "delta_P_hit_prime_gamma": _agg(v["gamma_prime"]).get("P_hit", 0)
            - _agg(v["gamma"]).get("P_hit", 0),
            "delta_P_hit_prime_fixed": _agg(v["gamma_prime"]).get("P_hit", 0)
            - _agg(v["fixed"]).get("P_hit", 0),
        }
        for mk, v in by_mstar.items()
    }

    aggp, aggg, aggf = aggs["gamma_prime"], aggs["gamma"], aggs["fixed"]
    gate = {
        "hypothesis": "one-step Γ + one-step memory (Γ') > one-step Γ on scoped targets",
        "scoped_targets": [_key(m) for m in SCOPED_MSTAR],
        "reps": args.reps,
        "P_hit": {p: aggs[p].get("P_hit") for p in POLICIES},
        "mean_delta_E": {p: aggs[p].get("mean_delta_E") for p in POLICIES},
        "mean_P_down": {p: aggs[p].get("mean_P_down_steps") for p in POLICIES},
        "P_repeat_after_noop": {p: aggs[p].get("P_repeat_after_noop") for p in POLICIES},
        "P_rescue_given_noop": {p: aggs[p].get("P_rescue_given_noop") for p in POLICIES},
        "delta_P_hit_prime_minus_gamma": (aggp.get("P_hit") or 0) - (aggg.get("P_hit") or 0),
        "delta_P_hit_prime_minus_fixed": (aggp.get("P_hit") or 0) - (aggf.get("P_hit") or 0),
        "signature_less_repeat": (
            (aggp.get("P_repeat_after_noop") or 1) < (aggg.get("P_repeat_after_noop") or 0)
            if aggp.get("n_noop_contexts") and aggg.get("n_noop_contexts")
            else None
        ),
        "signature_hit_up": (aggp.get("P_hit") or 0) > (aggg.get("P_hit") or 0),
        "n_pairs_prime_only_vs_gamma": sum(
            1 for p in paired if p["gamma_prime"]["hit"] and not p["gamma"]["hit"]
        ),
        "n_pairs_gamma_only_vs_prime": sum(
            1 for p in paired if p["gamma"]["hit"] and not p["gamma_prime"]["hit"]
        ),
        "read": (
            "Critical: P(repeat|noop)_Γ' < P(repeat|noop)_Γ and P(hit)_Γ' > P(hit)_Γ. "
            "Then expand kernel for remaining failures → full 8 targets."
        ),
    }

    payload = {
        "protocol": "Phase 9H live Γ' vs Γ vs fixed",
        "seed": args.seed,
        "reps": args.reps,
        "max_steps": args.max_steps,
        "n_9g_extra": n_extra,
        "controller": {
            "frozen": "8K",
            "gamma": "empirical argmax among remaining-error bits",
            "gamma_prime": "Γ + block repeat after no-op",
            "fixed": "H→C→O first differing bit",
            "no_weight_tuning": True,
        },
        "agg": aggs,
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
        "# Phase 9H — Live $\\Gamma'$ (anti-stagnation) vs $\\Gamma$ vs fixed",
        "",
        r"> $\Gamma'(a)=-\infty$ if $a=a_{t-1}\land S_t=S_{t-1}$; else $\Gamma(a\mid S_t,m^*)$. "
        "Frozen 8K. 9G kernel. No other memory.",
        "",
        f"Targets `{[_key(m) for m in SCOPED_MSTAR]}`, reps={args.reps}, "
        f"max_steps={args.max_steps}, seed={args.seed}, 9G_extra={n_extra}.",
        "",
        "## Aggregate",
        "",
        "| policy | n | $P_{hit}$ | mean $\\Delta E$ | $P(E\\downarrow)$ | "
        r"$P(\mathrm{repeat}|\mathrm{noop})$ | $P(\mathrm{rescue}|\mathrm{noop})$ |",
        "|--------|--:|----------:|-----------------:|------------------:|-------------------------------:|-------------------------------:|",
    ]
    for p, label in (
        ("gamma_prime", "**$\\Gamma'$**"),
        ("gamma", "$\\Gamma$"),
        ("fixed", "fixed"),
    ):
        a = aggs[p]
        lines.append(
            f"| {label} | {a['n']} | **{fmt(a.get('P_hit'))}** | {fmt(a.get('mean_delta_E'))} | "
            f"{fmt(a.get('mean_P_down_steps'))} | {fmt(a.get('P_repeat_after_noop'))} | "
            f"{fmt(a.get('P_rescue_given_noop'))} |"
        )

    lines += [
        "",
        f"$\\Delta P_{{hit}}(\\Gamma'-\\Gamma)$ = **{fmt(gate['delta_P_hit_prime_minus_gamma'])}**",
        f"$\\Delta P_{{hit}}(\\Gamma'-\\mathrm{{fixed}})$ = **{fmt(gate['delta_P_hit_prime_minus_fixed'])}**",
        "",
        "## Per $m^*$ $P_{hit}$",
        "",
        "| $m^*$ | $\\Gamma'$ | $\\Gamma$ | fixed | $\\Gamma'-\\Gamma$ |",
        "|-------|----------:|---------:|------:|-----------------:|",
    ]
    for m in SCOPED_MSTAR:
        mk = _key(m)
        st = per_m[mk]
        lines.append(
            f"| `{mk}` | {fmt(st['gamma_prime'].get('P_hit'))} | {fmt(st['gamma'].get('P_hit'))} | "
            f"{fmt(st['fixed'].get('P_hit'))} | **{fmt(st['delta_P_hit_prime_gamma'])}** |"
        )

    lines += ["", "## Example trajectories (first rep each $m^*$)", ""]
    seen = set()
    for p in paired:
        mk = p["mstar_key"]
        if mk in seen:
            continue
        seen.add(mk)
        for pol, tag in (("gamma_prime", "Γ'"), ("gamma", "Γ"), ("fixed", "f")):
            t = p[pol]
            lines.append(
                f"- `{mk}` {tag}: `{'→'.join(_key(s) for s in t['traj_S'])}` "
                f"acts={t['actions']} hit={t['hit']} "
                f"rep_noop={t['n_repeat_after_noop']}/{t['n_noop_contexts']}"
            )

    lines += [
        "",
        "## Gate (critical signature)",
        "",
        f"- $P(\\mathrm{{repeat}}|\\mathrm{{noop}})$ $\\Gamma'$ / $\\Gamma$: "
        f"**{fmt(aggp.get('P_repeat_after_noop'))}** / {fmt(aggg.get('P_repeat_after_noop'))} "
        f"(less? **{gate['signature_less_repeat']}**)",
        f"- $P_{{hit}}$ $\\Gamma'$ / $\\Gamma$: **{fmt(aggp.get('P_hit'))}** / {fmt(aggg.get('P_hit'))} "
        f"(up? **{gate['signature_hit_up']}**)",
        f"- $P(\\mathrm{{rescue}}|\\mathrm{{noop}})$ $\\Gamma'$ / $\\Gamma$: "
        f"{fmt(aggp.get('P_rescue_given_noop'))} / {fmt(aggg.get('P_rescue_given_noop'))}",
        "",
        gate["read"],
        "",
        r"$$\boxed{\Gamma'=\Gamma+\text{one-step no-op memory}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
