#!/usr/bin/env python3
"""Compute (abs−rel) residuals for expanded randoms; compare to rand_02's +2.1."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.controls import build_random_orthogonal_controls
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.directions import DirectionVector
    from activation_pipeline.loader import load_model_and_tokenizer
    from scripts.run_unexplained_diagnostics import load_jsonl, run_privilege_blowup_modes

    prev = json.loads((ROOT / "data/results/unexplained_diagnostics.json").read_text())
    abs_rows = {
        d["direction_id"]: d for d in prev["random_expand"]["per_direction_absolute"]
    }
    layer_k, layer_L = 4, 22
    seed = 20260809
    n_random = 32
    max_transcripts = 8
    # Norm ratio from prior magnitude check
    shrink = 0.0894  # 1 - N_tool/N_prose ≈ 0.0894

    print("===== LOAD MODEL =====", flush=True)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    transcripts = load_transcripts(
        ROOT / "data/transcripts/real/gap_agentic.jsonl"
    )
    persona_dirs = load_jsonl(
        ROOT / f"data/directions/persona_pca_prose_L{layer_k}.jsonl"
    )
    fixed = []
    for d in persona_dirs:
        if int(d["layer"]) != layer_k:
            continue
        if d.get("kind") == "assistant_axis" or (
            d.get("kind") == "persona_pc" and int(d.get("meta", {}).get("pc_index", 99)) < 5
        ):
            fixed.append(
                DirectionVector(
                    direction_id=d["direction_id"],
                    kind=d.get("kind", "fixed"),
                    mode="prose",
                    layer=layer_k,
                    vector=d["vector"],
                    n_pos=0,
                    n_neg=0,
                )
            )
    directions, diagnostics = build_random_orthogonal_controls(
        n_random=n_random,
        hidden_size=loaded.spec.hidden_size,
        layer=layer_k,
        seed=seed,
        fixed_basis=[torch.tensor(f.vector, dtype=torch.float32) for f in fixed],
    )
    normed = [
        {
            "direction_id": d.direction_id,
            "kind": "random_control",
            "vector": d.vector,
            "layer": layer_k,
        }
        for d in directions
    ]
    assert diagnostics.passed
    assert set(d["direction_id"] for d in normed) == set(abs_rows)

    print("===== RELATIVE (for residual) =====", flush=True)
    t0 = time.time()
    rel = run_privilege_blowup_modes(
        loaded,
        directions=normed,
        transcripts=transcripts,
        layer_k=layer_k,
        layer_L=layer_L,
        max_transcripts=max_transcripts,
        threshold_rel=0.5,
        max_iter=16,
        modes=["relative"],
    )
    rel_rows = {
        d["direction_id"]: d for d in rel["by_blowup_mode"]["relative"]["per_direction"]
    }

    residuals = []
    for did in sorted(abs_rows):
        a = abs_rows[did]
        r = rel_rows[did]
        obs = a["mean_delta"] - r["mean_delta"]
        pred = shrink * r["mean_eps_prose"]
        resid = obs - pred
        residuals.append(
            {
                "direction_id": did,
                "delta_rel": r["mean_delta"],
                "delta_abs": a["mean_delta"],
                "eps_prose_rel": r["mean_eps_prose"],
                "obs_shift_abs_minus_rel": obs,
                "predicted_shift": pred,
                "residual": resid,
            }
        )

    arr = np.array([x["residual"] for x in residuals], dtype=np.float64)
    abs_arr = np.array([x["delta_abs"] for x in residuals], dtype=np.float64)
    # Prior: rand_02 residual ≈ +2.10; absolute Δ ≈ +5.51
    n_resid_ge_2 = int((arr >= 2.0).sum())
    n_resid_abs_ge_2 = int((np.abs(arr) >= 2.0).sum())
    n_abs_ge_5_5 = int((abs_arr >= 5.5).sum())
    n_abs_le_m3_5 = int((abs_arr <= -3.5).sum())
    z = (arr - arr.mean()) / (arr.std(ddof=1) + 1e-12)

    if n_resid_ge_2 <= 2:
        decision = (
            f"SAMPLING_VARIANCE — only {n_resid_ge_2}/{len(arr)} draws have "
            "residual≥+2.0 (rand_02-class); consistent with n=4 luck"
        )
    elif n_resid_ge_2 >= int(0.25 * len(arr)):
        decision = (
            f"CLUSTER — {n_resid_ge_2}/{len(arr)} have residual≥+2.0; "
            "not pure sampling noise"
        )
    else:
        decision = (
            f"HEAVY_TAIL — {n_resid_ge_2}/{len(arr)} residual≥+2.0; "
            "tail thicker than Gaussian but not a dominant cluster"
        )

    out = {
        "model": "qwen3-0.6b",
        "n_random": len(residuals),
        "shrink_factor": shrink,
        "prior_rand_02_residual": 2.10,
        "prior_rand_02_abs_delta": 5.51,
        "summary": {
            "residual_mean": float(arr.mean()),
            "residual_std": float(arr.std(ddof=1)),
            "residual_min": float(arr.min()),
            "residual_max": float(arr.max()),
            "n_residual_ge_plus_2": n_resid_ge_2,
            "n_residual_abs_ge_2": n_resid_abs_ge_2,
            "frac_residual_ge_plus_2": n_resid_ge_2 / len(arr),
            "n_abs_delta_ge_5_5": n_abs_ge_5_5,
            "n_abs_delta_le_minus_3_5": n_abs_le_m3_5,
            "n_residual_z_ge_2": int((np.abs(z) >= 2).sum()),
            "percentiles_residual": {
                str(p): float(np.percentile(arr, p)) for p in (5, 25, 50, 75, 95)
            },
        },
        "decision": decision,
        "per_direction": residuals,
        "elapsed_s": time.time() - t0,
    }
    path = ROOT / "data/results/random_residual_expand.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"[resid] mean={arr.mean():+.3f} sd={arr.std(ddof=1):.3f} "
        f"n≥+2={n_resid_ge_2}/{len(arr)} n_|z|≥2={(np.abs(z)>=2).sum()}",
        flush=True,
    )
    print(f"decision: {decision}", flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
