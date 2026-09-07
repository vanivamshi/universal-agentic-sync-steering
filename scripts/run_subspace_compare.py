#!/usr/bin/env python3
"""§2 Orthogonal subspace compare: Procrustes + mean cosine (prose vs tool)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _by_layer(
    records: list[Any], layer: int
) -> tuple[list, list]:
    prose, tool = [], []
    key = str(layer)
    for r in records:
        if key not in r.layer_means:
            continue
        if r.window_kind == "prose":
            prose.append(r.mean_tensor(layer))
        elif r.window_kind == "tool_call":
            tool.append(r.mean_tensor(layer))
    return prose, tool


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "activations" / "real_gap.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "subspace_gap.json",
    )
    ap.add_argument(
        "--evaluation-dataset",
        default="",
        help="Label in results JSON (e.g. mind_the_gap, tau_bench)",
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers to compare (default: all layers present in cache)",
    )
    ap.add_argument(
        "--by-domain",
        action="store_true",
        help="Also report per-domain subspace metrics",
    )
    args = ap.parse_args()

    from activation_pipeline.analysis.subspace import compare_prose_tool_subspaces
    from activation_pipeline.cache_io import ActivationCache

    cache = ActivationCache.load(args.cache)
    layers = args.layers or list(cache.layer_indices)
    eval_name = args.evaluation_dataset or args.cache.stem

    overall: list[dict[str, Any]] = []
    for layer in layers:
        prose, tool = _by_layer(cache.records, layer)
        if not prose or not tool:
            overall.append(
                {
                    "layer": layer,
                    "error": f"need prose+tool; got n_prose={len(prose)} n_tool={len(tool)}",
                }
            )
            continue
        sub = compare_prose_tool_subspaces(prose, tool)
        row = {
            "layer": layer,
            "mean_cosine": sub.mean_cosine,
            "procrustes_disparity": sub.procrustes_disparity,
            "principal_angles_deg": sub.principal_angles_deg,
            "mean_principal_angle_deg": sub.mean_principal_angle_deg,
            "max_principal_angle_deg": sub.max_principal_angle_deg,
            "subspace_rank": sub.subspace_rank,
            "n_prose": sub.n_prose,
            "n_tool": sub.n_tool,
            "hidden": sub.hidden,
        }
        overall.append(row)
        ang = (
            f" mean∠={sub.mean_principal_angle_deg:.1f}° max∠={sub.max_principal_angle_deg:.1f}°"
            if sub.mean_principal_angle_deg is not None
            else ""
        )
        print(
            f"[{eval_name}] L{layer}: cos={sub.mean_cosine:.4f} "
            f"disparity={sub.procrustes_disparity:.4f}{ang} "
            f"n_prose={sub.n_prose} n_tool={sub.n_tool}"
        )

    by_domain: dict[str, list[dict[str, Any]]] = {}
    if args.by_domain:
        domains = sorted({r.domain for r in cache.records})
        for dom in domains:
            by_domain[dom] = []
            subset = [r for r in cache.records if r.domain == dom]
            for layer in layers:
                prose, tool = _by_layer(subset, layer)
                if not prose or not tool:
                    by_domain[dom].append(
                        {
                            "layer": layer,
                            "error": f"n_prose={len(prose)} n_tool={len(tool)}",
                        }
                    )
                    continue
                sub = compare_prose_tool_subspaces(prose, tool)
                by_domain[dom].append(
                    {
                        "layer": layer,
                        "mean_cosine": sub.mean_cosine,
                        "procrustes_disparity": sub.procrustes_disparity,
                        "principal_angles_deg": sub.principal_angles_deg,
                        "mean_principal_angle_deg": sub.mean_principal_angle_deg,
                        "max_principal_angle_deg": sub.max_principal_angle_deg,
                        "subspace_rank": sub.subspace_rank,
                        "n_prose": sub.n_prose,
                        "n_tool": sub.n_tool,
                        "hidden": sub.hidden,
                    }
                )
                ang = (
                    f" mean∠={sub.mean_principal_angle_deg:.1f}°"
                    if sub.mean_principal_angle_deg is not None
                    else ""
                )
                print(
                    f"  domain={dom} L{layer}: cos={sub.mean_cosine:.4f} "
                    f"disparity={sub.procrustes_disparity:.4f}{ang}"
                )

    payload = {
        "evaluation_dataset": eval_name,
        "cache": str(args.cache),
        "model_note": (
            "Qwen3-0.6B pilot; confirmatory subspace analysis planned on Qwen3-32B."
        ),
        "analysis": (
            "preregistered exploratory — mean cosine + Procrustes; "
            "principal angles added as extra exploratory descriptor (not a hypothesis test). "
            "Does not decide RQ1. Do not compare raw disparity across datasets without "
            "matched preprocessing."
        ),
        "interpretation": (
            "Across both datasets, prose and tool activations remain substantially aligned "
            "(cosine ≈0.8–0.9), but cannot be perfectly aligned by an orthogonal "
            "transformation, indicating measurable geometric differences without "
            "evidence of complete representational separation. A mild early→late cosine "
            "drop is an observation to test on Qwen3-32B."
        ),
        "method": (
            "Mean cosine between centroids of window-mean residual activations; "
            "Procrustes on the same centered point sets; principal angles between "
            "PCA subspaces fit independently to each set (top-r≤4)."
        ),
        "status": "frozen_pilot",
        "layers": overall,
        "by_domain": by_domain,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
