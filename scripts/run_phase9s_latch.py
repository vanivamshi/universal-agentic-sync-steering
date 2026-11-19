#!/usr/bin/env python3
"""Phase 9S — Configuration retention: latch vs arrival-path replay.

Diagnosis (after 9R): the controller records S at each step but does not keep
the intervention history that produced it. 9R "retention" re-runs a fresh
episode (new seed) with ONE channel at TARGET sign, so P_ret measured an
independent re-acquisition, not maintenance of an acquired configuration.

9S makes that history explicit (`activation_pipeline.sync_state.SyncState`)
and asks what, if anything, makes a re-run reproduce m*. Acquisition is
identical to the 9R hybrid arm and is run ONCE per trial; every retention arm
starts from the same acquired trial (paired).

Retention arms (all fresh-episode replay — this is CONFIGURATION retention,
not persistence of one ongoing model trajectory):

  A  stay_seek      unchanged 9R retain loop (one channel, target sign).
                    Own seeds (seed+1+t) — baseline replication, not seed-paired.
  B  latch          composed `active` from the arrival path, ARRIVAL signs,
                    all channels simultaneously in one episode.
  B0 latch_tsign    same channels as B, signs forced to TARGET polarity.
                    B vs B0 isolates arrival polarity.
  C  path_replay    ordered arrival path replayed step by step (single-channel
                    episodes, plan threaded as in acquisition) from plan0.
  D  path_shuffled  same signed actions as C, permuted order. C vs D isolates
                    arrival ordering. Degenerate when no distinct permutation.

B/B0/C/D share rollout seeds: rollout k uses seed_k, sub-episode j of a path
replay uses seed_k + j. Episodes are cached by (seed, config, plan) so arms
whose configs coincide reuse the same rollout.

Metrics per arm: ret@1, ret@2 (= stayed on every rollout up to k), raw
rollout hit rate, P_final = acquired AND ret@n_retain over all trials. Counts,
Wilson 95% CIs, soft vs hard pooling, paired discordance + exact sign test.
No new v; frozen 8K actuators and gains.

  .venv/bin/python -u scripts/run_phase9s_latch.py --reps 4
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9s_latch.json"
MD = ROOT / "data" / "results" / "sync_phase9s_latch.md"
SEED = 20261119
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
HARD = {(0, 0, 0), (0, 0, 1), (1, 1, 0)}
REPLAY_ARMS = ("latch", "latch_tsign", "path_replay", "path_shuffled")
ARMS = ("stay_seek",) + REPLAY_ARMS
ROLLOUT_SEED_STRIDE = 7919

# episode_fn(active: {ch: sign}, seed, plan) -> {"S": [c,h,o], "plan_prefill": str}
EpisodeFn = Callable[[dict[str, int], int, str], dict[str, Any]]


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_sync_state():
    # Load by path: importing the package __init__ pulls in torch.
    return _load_mod("sync_state", ROOT / "activation_pipeline" / "sync_state.py")


def _key(s) -> str:
    return "".join(str(int(x)) for x in s)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sign_test(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant pairs (McNemar exact)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# Retention arms (pure; episode_fn injected)
# --------------------------------------------------------------------------
def shuffled_path(path: list, rng: random.Random) -> tuple[list, bool]:
    """Permutation of `path` that differs from it; (path, True) if impossible."""
    distinct = {tuple(p) for p in itertools.permutations(path)} if len(path) <= 7 else None
    if distinct is not None and len(distinct) <= 1:
        return list(path), True
    for _ in range(200):
        cand = list(path)
        rng.shuffle(cand)
        if cand != list(path):
            return cand, False
    return list(path), True


def run_config_rollouts(
    episode_fn: EpisodeFn, active: dict[str, int], plan: str, seeds: list[int], mstar
) -> list[dict]:
    out = []
    for sk in seeds:
        ep = episode_fn(active, sk, plan)
        out.append({"seed": sk, "S": list(ep["S"]), "stay": int(list(ep["S"]) == list(mstar))})
    return out


def run_path_rollouts(
    episode_fn: EpisodeFn, path: list, plan0: str, seeds: list[int], mstar
) -> list[dict]:
    out = []
    for sk in seeds:
        plan = plan0 or ""
        sub = []
        S = None
        for j, (ch, sign) in enumerate(path):
            ep = episode_fn({ch: int(sign)}, sk + j, plan)
            S = list(ep["S"])
            plan = ep.get("plan_prefill") or plan
            sub.append(_key(S))
        out.append({"seed": sk, "S": S, "sub_traj": sub, "stay": int(S == list(mstar))})
    return out


def retention_from_rollouts(rolls: list[dict], n_retain: int) -> dict[str, Any]:
    stays = [r["stay"] for r in rolls]
    ret_at = {}
    for k in range(1, n_retain + 1):
        ret_at[k] = int(len(stays) >= k and all(stays[:k]))
    return {"stays": stays, "ret_at": ret_at, "hit_rate": sum(stays) / len(stays) if stays else None}


def run_replay_arms(
    episode_fn: EpisodeFn,
    ss,  # sync_state module
    *,
    state,
    mstar,
    plan0: str,
    plan_acq: str,
    seed: int,
    n_retain: int,
) -> dict[str, dict]:
    seeds = [seed + ROLLOUT_SEED_STRIDE * (k + 1) for k in range(n_retain)]
    rng = random.Random(seed)
    path = list(state.path)
    shuf, degenerate = shuffled_path(path, rng)
    active = state.active
    active_t = state.target_signed_active(mstar)

    arms: dict[str, dict] = {}
    for name, rolls, extra in (
        ("latch", run_config_rollouts(episode_fn, active, plan_acq, seeds, mstar),
         {"config": {c: s for c, s in active.items()}}),
        ("latch_tsign", run_config_rollouts(episode_fn, active_t, plan_acq, seeds, mstar),
         {"config": active_t, "same_as_latch": active_t == active}),
        ("path_replay", run_path_rollouts(episode_fn, path, plan0, seeds, mstar),
         {"path": [ss.act_name(c, s) for c, s in path]}),
        ("path_shuffled", run_path_rollouts(episode_fn, shuf, plan0, seeds, mstar),
         {"path": [ss.act_name(c, s) for c, s in shuf], "degenerate": degenerate}),
    ):
        arms[name] = {**retention_from_rollouts(rolls, n_retain), "rollouts": rolls, **extra}
    return arms


def stay_seek_from_trial(tr: dict, n_retain: int) -> dict[str, Any]:
    """9R baseline: per-rollout stays from the retain step log (breaks on leave)."""
    stays = [int(s["stayed"]) for s in tr["steps"] if s.get("phase") == "retain"]
    ret_at = {k: int(len(stays) >= k and all(stays[:k])) for k in range(1, n_retain + 1)}
    return {
        "stays": stays,
        "ret_at": ret_at,
        "hit_rate": sum(stays) / len(stays) if stays else None,
        "actions": [s.get("a") for s in tr["steps"] if s.get("phase") == "retain"],
    }


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def summarize(trials: list[dict], n_retain: int) -> dict[str, Any]:
    K = n_retain

    def agg(rows: list[dict], arm: str) -> dict[str, Any]:
        n = len(rows)
        acq = [t for t in rows if t["acquire"]]
        res: dict[str, Any] = {"n": n, "n_acq": len(acq)}
        for k in range(1, K + 1):
            kk = sum(t["arms"][arm]["ret_at"][k] for t in acq)
            res[f"ret@{k}"] = {"k": kk, "n": len(acq), "p": kk / len(acq) if acq else None,
                               "ci": wilson(kk, len(acq))}
        nf = sum(t["arms"][arm]["ret_at"][K] for t in acq)
        res["final"] = {"k": nf, "n": n, "p": nf / n if n else None, "ci": wilson(nf, n)}
        hs = [h for t in acq for h in t["arms"][arm]["stays"]]
        res["rollout_hits"] = {"k": sum(hs), "n": len(hs)}
        return res

    per_m = {}
    for m in ALL_MSTAR:
        rows = [t for t in trials if tuple(t["mstar"]) == m]
        per_m[_key(m)] = {arm: agg(rows, arm) for arm in ARMS}
    groups = {
        "soft": [t for t in trials if tuple(t["mstar"]) not in HARD],
        "hard": [t for t in trials if tuple(t["mstar"]) in HARD],
        "all": trials,
    }
    pooled = {g: {arm: agg(rows, arm) for arm in ARMS} for g, rows in groups.items()}

    def paired(a: str, b: str, filt=None) -> dict[str, Any]:
        rows = [t for t in trials if t["acquire"] and (filt is None or filt(t))]
        ya = [t["arms"][a]["ret_at"][K] for t in rows]
        yb = [t["arms"][b]["ret_at"][K] for t in rows]
        b_only = sum(1 for x, y in zip(ya, yb) if x == 0 and y == 1)
        a_only = sum(1 for x, y in zip(ya, yb) if x == 1 and y == 0)
        return {"n_pairs": len(rows), f"{a}_only": a_only, f"{b}_only": b_only,
                "both": sum(1 for x, y in zip(ya, yb) if x and y), "p_sign": sign_test(a_only, b_only)}

    comparisons = {
        # names list arms in (first, second) order, matching the discordance columns
        "stay_seek_vs_latch": paired("stay_seek", "latch"),
        "latch_tsign_vs_latch": paired(
            "latch_tsign", "latch", lambda t: not t["arms"]["latch_tsign"]["same_as_latch"]),
        "latch_vs_path_replay": paired("latch", "path_replay"),
        "path_shuffled_vs_path_replay": paired(
            "path_shuffled", "path_replay", lambda t: not t["arms"]["path_shuffled"]["degenerate"]),
    }
    return {"per_mstar": per_m, "pooled": pooled, "comparisons": comparisons}


def render_md(payload: dict) -> str:
    s = payload["summary"]
    K = payload["n_retain"]

    def frac(d):
        if not d or d["n"] == 0:
            return "—"
        ci = d.get("ci")
        ci_s = f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
        return f"{d['k']}/{d['n']}{ci_s}"

    L = [
        "# Phase 9S — Configuration retention: latch vs arrival-path replay",
        "",
        "> Paired: one hybrid Γ'+beam acquisition per trial; all retention arms start from it. "
        "Fresh-episode replay ⇒ this measures **configuration retention**, not maintenance of "
        "one ongoing trajectory. No new $v$.",
        "",
        f"reps={payload['reps']}, n_retain={K}, max_steps_acq={payload['max_steps_acq']}, "
        f"beam K={payload['beam']} T={payload['horizon']}, seed={payload['seed']}. "
        f"Trials run: {payload['n_trials']} (skipped S0==m*: {payload['n_skipped']}).",
        "",
        "Arms: **A** stay_seek (9R, own seeds) · **B** latch (composed, arrival signs) · "
        "**B0** latch_tsign (composed, target signs) · **C** path_replay (ordered) · "
        "**D** path_shuffled (permuted).",
        "",
        f"## ret@{K} | acquired, per $m^*$ (k/n [Wilson 95%])",
        "",
        "| $m^*$ | acq | " + " | ".join(ARMS) + " |",
        "|---|---:|" + "---:|" * len(ARMS),
    ]
    for m in ALL_MSTAR:
        mk = _key(m)
        row = s["per_mstar"][mk]
        acq = f"{row['latch']['n_acq']}/{row['latch']['n']}"
        tag = " (hard)" if m in HARD else ""
        L.append(f"| `{mk}`{tag} | {acq} | " + " | ".join(frac(row[a][f"ret@{K}"]) for a in ARMS) + " |")
    L += ["", "## Pooled", "", "| group | metric | " + " | ".join(ARMS) + " |",
          "|---|---|" + "---:|" * len(ARMS)]
    for g in ("soft", "hard", "all"):
        for k in range(1, K + 1):
            L.append(f"| {g} | ret@{k} | " + " | ".join(frac(s["pooled"][g][a][f"ret@{k}"]) for a in ARMS) + " |")
        L.append(f"| {g} | P_final | " + " | ".join(frac(s["pooled"][g][a]["final"]) for a in ARMS) + " |")
        L.append(f"| {g} | rollout hits | " + " | ".join(
            f"{s['pooled'][g][a]['rollout_hits']['k']}/{s['pooled'][g][a]['rollout_hits']['n']}" for a in ARMS) + " |")
    L += ["", f"## Paired comparisons (ret@{K}, acquired trials)", "",
          "| comparison | pairs | discordant (first-only / second-only) | both | exact sign p |",
          "|---|---:|---:|---:|---:|"]
    for name, c in s["comparisons"].items():
        ks = [k for k in c if k.endswith("_only")]
        L.append(f"| {name} | {c['n_pairs']} | {c[ks[0]]} / {c[ks[1]]} | {c['both']} | {c['p_sign']:.3f} |")
    L += [
        "",
        "## Reading guide (fix before looking at numbers)",
        "",
        "| result | suggests |",
        "|---|---|",
        "| B > A | accumulated configuration matters |",
        "| B > B0 | arrival polarity matters, not just which channels |",
        "| B fails, C succeeds | arrival sequence matters |",
        "| C > D | ordering (not just the multiset of actions) matters |",
        "| B and C fail | fresh-episode replay insufficient → motivates persistent context / activation feedback (does not prove them necessary) |",
        "| soft ok, hard fail | hard sinks need a different retention mechanism |",
        "",
        "Small n: a nonzero rate is ≥1 observed success, not reliable retention. Do not read "
        "P_final>0 on 8/8 as reliable control.",
        "",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-steps-acq", type=int, default=7)
    ap.add_argument("--n-retain", type=int, default=2)
    ap.add_argument("--tau-gamma", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mstar", nargs="*", default=None, help="subset, e.g. 000 110")
    args = ap.parse_args()

    import torch  # noqa: F401  (heavy deps only at run time)

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    ss = _load_sync_state()
    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
    p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
    p9q = _load_mod("phase9q", ROOT / "scripts" / "run_phase9q_bidir_beam.py")
    p9r = _load_mod("phase9r", ROOT / "scripts" / "run_phase9r_hybrid.py")

    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()
    lookup, _ker, _rows = p9j.build_lookup(p9e, p9g, None)

    ep_cache: dict = {}

    def episode_fn(active: dict[str, int], seed: int, plan: str) -> dict[str, Any]:
        # H plan prefill only when H active (8K drops it itself when C is also active).
        h_pf = plan if "H" in active else None
        key = (int(seed), tuple(sorted(active.items())), h_pf or "")
        if key not in ep_cache:
            act = {ch: (int(s), dirs[ch]) for ch, s in active.items()}
            ep_cache[key] = p8k._run_episode(sc, loaded, active=act, seed=seed, h_plan_prefill=h_pf)
        return ep_cache[key]

    targets = ALL_MSTAR if not args.mstar else [tuple(int(c) for c in m) for m in args.mstar]
    trials: list[dict] = []
    n_skipped = 0
    n_tot = len(targets) * args.reps
    idx = 0
    print(f"=== Phase 9S latch reps={args.reps} n_retain={args.n_retain} ===", flush=True)

    for m in targets:
        mi = ALL_MSTAR.index(m)
        mstar = list(m)
        for r in range(args.reps):
            idx += 1
            seed0 = args.seed + 200 * mi + 19 * r
            S0, plan0, seed = p9j.free_S0(p8k, sc, loaded, mstar=mstar, seed0=seed0)
            print(f"  [{idx}/{n_tot}] m*={_key(m)} r={r} S0={_key(S0)}", flush=True)
            if S0 == mstar:
                n_skipped += 1
                print("    skip S0==m*", flush=True)
                continue
            tr = p9r.run_trial(
                p9j, p9q, p8k, sc, loaded, dirs=dirs, lookup=lookup, mstar=mstar, seed=seed,
                S0=S0, plan0=plan0, arm="gamma_prime_beam", tau=args.tau_gamma, beam=args.beam,
                horizon=args.horizon, max_steps_acq=args.max_steps_acq, n_retain=args.n_retain,
                cache={}, beam_cache={},
            )
            state = ss.SyncState.from_trial_steps(S0, plan0, tr["steps"])
            rec: dict[str, Any] = {
                "mstar": mstar, "mstar_key": _key(m), "hard": m in HARD, "seed": seed, "S0": S0,
                "acquire": int(tr["acquire"]), "acq_path": tr["path"], "n_beam_calls": tr["n_beam_calls"],
                "state": state.to_dict(), "polarity": state.polarity_stats(mstar),
            }
            if tr["acquire"]:
                assert state.at(mstar), (state.S, mstar)
                arms = {"stay_seek": stay_seek_from_trial(tr, args.n_retain)}
                arms.update(run_replay_arms(
                    episode_fn, ss, state=state, mstar=mstar, plan0=tr["plan0"],
                    plan_acq=tr["plan_acq"], seed=seed, n_retain=args.n_retain,
                ))
                rec["arms"] = arms
                print("    acq path=" + " ".join(state.acts()) + "  ret@n: " + "  ".join(
                    f"{a}={arms[a]['ret_at'][args.n_retain]}" for a in ARMS), flush=True)
            else:
                print("    not acquired", flush=True)
            trials.append(rec)

    payload = {
        "protocol": "Phase 9S configuration retention: latch vs arrival-path replay (paired)",
        "seed": args.seed, "reps": args.reps, "n_retain": args.n_retain,
        "max_steps_acq": args.max_steps_acq, "tau_gamma": args.tau_gamma,
        "beam": args.beam, "horizon": args.horizon, "arms": list(ARMS),
        "targets": [_key(m) for m in targets], "n_trials": len(trials), "n_skipped": n_skipped,
        "summary": summarize(trials, args.n_retain),
        "path_cache": [  # (S0, m*) -> arrival recipe, reusable as an open-loop plan
            {"S0": _key(t["S0"]), "mstar": t["mstar_key"], "path": t["state"]["path"]}
            for t in trials if t["acquire"]
        ],
        "trials": trials,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    MD.write_text(render_md(payload), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} and {MD.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
