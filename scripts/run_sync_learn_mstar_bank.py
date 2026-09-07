#!/usr/bin/env python3
"""Build natural observation bank for m*-conditioned steering.

No hand labels and no scripted S=m*. Episodes are free model runs; H/O come
from execution hooks + FINAL scoring (automatic measurement of natural behavior).

Buckets activations by *observed* (H, O). Plan is uncovered only because older
geometry runs did not record PLAN text — not because we require annotations.

Writes:
  data/directions/sync_mstar_bank_L4.json

Controller loads this bank and sets
  d(m*) = unit( μ[m*_H, m*_O] − μ[S_H, S_O] )
when both centroids exist; else falls back to ±v_repair on out only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "results" / "sync_geometry_cache.npz"
GEOM = ROOT / "data" / "results" / "sync_geometry.json"
OUT = ROOT / "data" / "directions" / "sync_mstar_bank_L4.json"
LAYER = 4
MIN_N = 5


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cache = np.load(CACHE)
    Hr = np.asarray(cache["Hr"], dtype=np.float64)
    s_tool = np.asarray(cache["s_tool"], dtype=np.int64)
    s_out = np.asarray(cache["s_out"], dtype=np.int64)
    geom = json.loads(GEOM.read_text())
    episodes = geom["episodes"]
    assert len(episodes) == len(Hr)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (h, o) in enumerate(zip(s_tool.tolist(), s_out.tolist())):
        buckets[(int(h), int(o))].append(i)

    centroids: dict[str, Any] = {}
    for (h, o), idxs in sorted(buckets.items()):
        key = f"{h},{o}"
        X = Hr[idxs]
        centroids[key] = {
            "h": h,
            "o": o,
            "n": len(idxs),
            "mu": X.mean(0).astype(float).tolist() if len(idxs) else None,
        }

    # Pairwise directions toward each available (h,o) target from each source
    directions: dict[str, Any] = {}
    usable = {
        (c["h"], c["o"]): np.asarray(c["mu"], dtype=np.float64)
        for c in centroids.values()
        if c["n"] >= args.min_n and c["mu"] is not None
    }
    for tgt, mu_t in usable.items():
        for src, mu_s in usable.items():
            if tgt == src:
                continue
            v = _unit(mu_t - mu_s)
            directions[f"from_{src[0]}{src[1]}_to_{tgt[0]}{tgt[1]}"] = {
                "src_ho": list(src),
                "tgt_ho": list(tgt),
                "vector": v.astype(float).tolist(),
            }

    # Also store absolute target directions from global mean (exploration)
    mu_all = Hr.mean(0)
    abs_dirs = {}
    for tgt, mu_t in usable.items():
        abs_dirs[f"to_{tgt[0]}{tgt[1]}"] = {
            "tgt_ho": list(tgt),
            "vector": _unit(mu_t - mu_all).astype(float).tolist(),
        }

    # Out-axis repair equivalent: (1,1) - (1,0) when both exist
    v_out = None
    if (1, 1) in usable and (1, 0) in usable:
        v_out = _unit(usable[(1, 1)] - usable[(1, 0)]).astype(float).tolist()

    payload = {
        "layer": LAYER,
        "source": str(CACHE),
        "min_n": args.min_n,
        "channels_covered": ["hook", "out"],
        "channels_missing": ["plan"],
        "centroid_counts": {k: centroids[k]["n"] for k in centroids},
        "centroids_ho": {
            k: {"h": v["h"], "o": v["o"], "n": v["n"], "mu": v["mu"]}
            for k, v in centroids.items()
            if v["n"] >= args.min_n
        },
        "pairwise_directions": directions,
        "absolute_directions": abs_dirs,
        "v_out_disclose_minus_hide": v_out,
        "note": (
            "Natural free-run bank: no hand labels. d(m*) uses observed (H,O) "
            "only. Plan bit appears once free runs also record PLAN (Agent "
            "protocol already does). Eval: run_sync_intervention_grid --mode qwen."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "centroid_counts": payload["centroid_counts"],
        "n_pairwise": len(directions),
        "has_v_out": v_out is not None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
