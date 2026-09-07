#!/usr/bin/env python3
"""Phase 9Q — Bidirectional beam search over ±C,±H,±O (actual rollouts).

9P showed target-sign can suppress useful transitions. Do not densify more.
Search over six signed actuators with live model rollouts:

  A = {+C,-C,+H,-H,+O,-O}

Hard targets only {000,001,110}. Beam width K, horizon T.
Score: exact hit first, then -E. No new v. No kernel planner.

  .venv/bin/python -u scripts/run_phase9q_bidir_beam.py --reps 2 --beam 3 --horizon 3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9q_bidir_beam.json"
MD = ROOT / "data" / "results" / "sync_phase9q_bidir_beam.md"
SEED = 20261005
ORDER = ("C", "H", "O")  # channel labels
CH_IDX = {"C": 0, "H": 1, "O": 2}
# six signed actions: (channel, sign) with sign ∈ {+1,-1}
ACTIONS: list[tuple[str, int]] = [
    ("C", +1),
    ("C", -1),
    ("H", +1),
    ("H", -1),
    ("O", +1),
    ("O", -1),
]
HARD = [(0, 0, 0), (0, 0, 1), (1, 1, 0)]


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


def _act_name(ch: str, sign: int) -> str:
    return f"{'+' if sign > 0 else '-'}{ch}"


@dataclass
class Node:
    S: list[int]
    plan: str
    acts: list[str] = field(default_factory=list)
    traj: list[list[int]] = field(default_factory=list)
    seed_base: int = 0
    depth: int = 0

    def score(self, mstar) -> tuple:
        hit = int(self.S == list(mstar))
        # primary hit, then lower E, then shorter path
        return (hit, -_E(mstar, self.S), -self.depth)


def apply_signed(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    channel: str,
    sign: int,
    seed: int,
    plan_prefill: str,
    cache: dict,
):
    key = (int(seed), channel, int(sign), plan_prefill or "")
    if key in cache:
        return cache[key], True
    active = {channel: (int(sign), dirs[channel])}
    h_pf = plan_prefill if channel == "H" else None
    out = p8k._run_episode(
        sc, loaded, active=active, seed=seed, h_plan_prefill=h_pf
    )
    cache[key] = out
    return out, False


def beam_search(
    p8k,
    sc,
    loaded,
    *,
    dirs,
    mstar,
    S0: list[int],
    plan0: str,
    seed0: int,
    beam: int,
    horizon: int,
    cache: dict,
) -> dict[str, Any]:
    """Expand K-best with all 6 signed actions for up to `horizon` steps."""
    root = Node(
        S=list(S0),
        plan=plan0 or "",
        acts=[],
        traj=[list(S0)],
        seed_base=seed0,
        depth=0,
    )
    if root.S == list(mstar):
        return {
            "found": True,
            "path": [_key(S0)],
            "acts": [],
            "S_final": list(S0),
            "n_expansions": 0,
            "n_cache_hits": 0,
            "beam_final": [],
        }

    beam_nodes = [root]
    n_exp = 0
    n_cache = 0
    best_hit: Node | None = None
    history = []

    for depth in range(horizon):
        candidates: list[Node] = []
        for node in beam_nodes:
            if node.S == list(mstar):
                candidates.append(node)
                continue
            for ch, sign in ACTIONS:
                n_exp += 1
                # unique deterministic seed per (path, action)
                path_tag = "|".join(node.acts) + f"|{_act_name(ch, sign)}|{depth}"
                path_h = int(
                    hashlib.md5(path_tag.encode(), usedforsecurity=False).hexdigest()[:8],
                    16,
                )
                seed = node.seed_base + 1009 * (depth + 1) + (path_h % 10007)
                out, hit = apply_signed(
                    p8k,
                    sc,
                    loaded,
                    dirs=dirs,
                    channel=ch,
                    sign=sign,
                    seed=seed,
                    plan_prefill=node.plan,
                    cache=cache,
                )
                if hit:
                    n_cache += 1
                Sa = list(out["S"])
                plan = out.get("plan_prefill") or node.plan
                child = Node(
                    S=Sa,
                    plan=plan,
                    acts=node.acts + [_act_name(ch, sign)],
                    traj=node.traj + [Sa],
                    seed_base=node.seed_base,
                    depth=depth + 1,
                )
                candidates.append(child)
                if Sa == list(mstar) and (
                    best_hit is None or child.depth < best_hit.depth
                ):
                    best_hit = child

        # keep top beam by score
        candidates.sort(key=lambda n: n.score(mstar), reverse=True)
        # dedupe by (S, acts) roughly — keep distinct S when possible
        kept = []
        seen_s = set()
        for n in candidates:
            sk = (_key(n.S), tuple(n.acts))
            if sk in seen_s:
                continue
            seen_s.add(sk)
            kept.append(n)
            if len(kept) >= beam:
                break
        # if all same, still fill beam
        if len(kept) < beam:
            for n in candidates:
                if n not in kept:
                    kept.append(n)
                if len(kept) >= beam:
                    break
        beam_nodes = kept
        history.append(
            {
                "depth": depth + 1,
                "beam": [
                    {
                        "S": _key(n.S),
                        "E": _E(mstar, n.S),
                        "acts": n.acts,
                        "hit": int(n.S == list(mstar)),
                    }
                    for n in beam_nodes
                ],
            }
        )
        if any(n.S == list(mstar) for n in beam_nodes):
            best_hit = next(n for n in beam_nodes if n.S == list(mstar))
            break

    if best_hit is None:
        # best among final beam
        beam_nodes.sort(key=lambda n: n.score(mstar), reverse=True)
        best_hit = beam_nodes[0]

    found = best_hit.S == list(mstar)
    return {
        "found": found,
        "path": [_key(s) for s in best_hit.traj],
        "traj": [list(s) for s in best_hit.traj],
        "acts": best_hit.acts,
        "S_final": list(best_hit.S),
        "plan_final": best_hit.plan,
        "E0": _E(mstar, S0),
        "E_final": _E(mstar, best_hit.S),
        "hit": int(found),
        "n_expansions": n_exp,
        "n_cache_hits": n_cache,
        "history": history,
        "beam_final": [
            {"S": _key(n.S), "E": _E(mstar, n.S), "acts": n.acts}
            for n in beam_nodes
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
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

    by_m: dict[str, list] = defaultdict(list)
    paired = []
    n_tot = len(HARD) * args.reps
    idx = 0

    print(
        f"=== Phase 9Q bidir beam K={args.beam} T={args.horizon} "
        f"reps={args.reps} actions={len(ACTIONS)} ===",
        flush=True,
    )

    for mi, mstar in enumerate(HARD):
        for r in range(args.reps):
            seed0 = args.seed + 300 * mi + 19 * r
            S0, plan0, seed = p9j.free_S0(
                p8k, sc, loaded, mstar=mstar, seed0=seed0
            )
            idx += 1
            print(
                f"  [{idx}/{n_tot}] m*={_key(mstar)} r={r} S0={_key(S0)} "
                f"E0={_E(mstar, S0)}",
                flush=True,
            )
            if S0 == list(mstar):
                print("    skip S0==m*", flush=True)
                continue
            cache: dict = {}
            result = beam_search(
                p8k,
                sc,
                loaded,
                dirs=dirs,
                mstar=mstar,
                S0=S0,
                plan0=plan0,
                seed0=seed,
                beam=args.beam,
                horizon=args.horizon,
                cache=cache,
            )
            result["mstar"] = list(mstar)
            result["mstar_key"] = _key(mstar)
            result["S0"] = list(S0)
            result["seed"] = seed
            by_m[_key(mstar)].append(result)
            paired.append(result)
            print(
                f"    found={result['found']} hit={result['hit']} "
                f"path={result['path']} acts={result['acts']} "
                f"exp={result['n_expansions']} cache={result['n_cache_hits']}",
                flush=True,
            )

    per = {}
    for m in HARD:
        mk = _key(m)
        rows = by_m[mk]
        n = len(rows)
        n_found = sum(1 for t in rows if t["found"])
        per[mk] = {
            "n": n,
            "P_hit": n_found / n if n else float("nan"),
            "P_found": n_found / n if n else float("nan"),
            "mean_E_final": float(np.mean([t["E_final"] for t in rows])) if rows else None,
            "paths": [
                {"found": t["found"], "path": t["path"], "acts": t["acts"]}
                for t in rows
            ],
        }

    n_reach = sum(1 for mk, st in per.items() if (st.get("P_hit") or 0) > 0)
    gate = {
        "hypothesis": (
            "Short-horizon beam search over ±C/±H/±O reaches hard sinks "
            "without target-sign restriction or kernel densify"
        ),
        "beam": args.beam,
        "horizon": args.horizon,
        "n_actions": len(ACTIONS),
        "P_hit": {mk: per[mk].get("P_hit") for mk in per},
        "n_hard_hit": n_reach,
        "hard_gate": n_reach == 3,
        "read": (
            "Primary: ∀m*∈{000,001,110} P_hit>0 under bidir beam. "
            "If yes: planning over signed actuators is the missing mechanism."
        ),
    }

    payload = {
        "protocol": "Phase 9Q bidirectional beam search ±C/H/O",
        "seed": args.seed,
        "reps": args.reps,
        "beam": args.beam,
        "horizon": args.horizon,
        "actions": [_act_name(c, s) for c, s in ACTIONS],
        "per_mstar": per,
        "paired": paired,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Phase 9Q — Bidirectional beam search ($\\pm C,\\pm H,\\pm O$)",
        "",
        "> Actual rollouts. Hard $m^*\\in\\{000,001,110\\}$. "
        "No new $v$. No kernel. Score = exact hit, then $-E$.",
        "",
        f"beam={args.beam}, horizon={args.horizon}, reps={args.reps}, seed={args.seed}.",
        "",
        "## Results",
        "",
        "| $m^*$ | n | $P_{hit}$ | mean $E_f$ | example path |",
        "|-------|--:|----------:|-----------:|--------------|",
    ]
    for m in HARD:
        mk = _key(m)
        st = per[mk]
        ex = ""
        for p in st.get("paths") or []:
            if p["found"]:
                ex = f"`{'→'.join(p['path'])}` acts={p['acts']}"
                break
        if not ex and st.get("paths"):
            p = st["paths"][0]
            ex = f"`{'→'.join(p['path'])}` acts={p['acts']}"
        lines.append(
            f"| `{mk}` | {st['n']} | **{fmt(st.get('P_hit'))}** | "
            f"{fmt(st.get('mean_E_final'))} | {ex} |"
        )
    lines += [
        "",
        f"Hard gate ($P_{{hit}}>0$ all 3): **{gate['hard_gate']}** ({n_reach}/3)",
        "",
        gate["read"],
        "",
        r"$$\boxed{\text{beam over }\pm v_C,\pm v_H,\pm v_O\text{ — planning, not new actuators}}$$",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
