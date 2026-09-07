#!/usr/bin/env python3
"""Step 4: extract, validate, and save learned/random control directions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=[14, 16, 19, 22],
        help="0.6B pilot equivalents of F_mid={0.50,0.60,0.70,0.80}",
    )
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--required-layers", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument(
        "--directions-out",
        type=Path,
        default=ROOT / "data" / "directions" / "controls_pilot.jsonl",
    )
    ap.add_argument(
        "--random-out",
        type=Path,
        default=ROOT / "data" / "directions" / "random_controls_pilot.jsonl",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data" / "results" / "control_directions_pilot.json",
    )
    args = ap.parse_args()

    from activation_pipeline.controls import (
        build_random_orthogonal_controls,
        collect_pair_activations,
        full_directions_from_activations,
        learned_control_stability,
    )
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.directions import save_directions
    from activation_pipeline.loader import load_model_and_tokenizer

    model_key = args.model or LOCAL_MODEL_KEY
    assert_model_fits_machine(model_key)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    layers = [int(layer) for layer in args.layers]
    print(
        f"controls model={model_key} device={device_map} layers={layers} "
        f"boot={args.n_boot}"
    )

    pairs_dir = ROOT / "data" / "contrast_pairs"
    control_specs = [
        (
            "syntax",
            "syntax",
            None,
            pairs_dir / "syntax_control_pilot.jsonl",
        ),
        (
            "domain_content_coding",
            "domain_content",
            "coding",
            pairs_dir / "domain_content_coding_pilot.jsonl",
        ),
        (
            "domain_content_therapy",
            "domain_content",
            "therapy",
            pairs_dir / "domain_content_therapy_pilot_v2.jsonl",
        ),
        (
            "domain_content_writing",
            "domain_content",
            "writing",
            pairs_dir / "domain_content_writing_pilot.jsonl",
        ),
    ]

    learned_directions = []
    stability_results = []
    for index, (direction_id, kind, domain, path) in enumerate(control_specs):
        pairs = load_jsonl(path)
        positive, negative = collect_pair_activations(
            loaded,
            pairs,
            layers=layers,
        )
        directions = full_directions_from_activations(
            positive,
            negative,
            layers=layers,
            direction_id=direction_id,
            kind=kind,
            domain=domain,
        )
        stability = learned_control_stability(
            positive,
            negative,
            layers=layers,
            direction_id=direction_id,
            kind=kind,
            domain=domain,
            n_boot=args.n_boot,
            n_splits_per_stat=args.n_splits,
            seed=args.seed + index * 1_000,
            threshold=args.threshold,
            required_layers=args.required_layers,
        )
        learned_directions.extend(directions)
        stability_results.append(stability)
        layer_summary = ", ".join(
            f"L{row['layer']}={row['split_half_cosine']:.3f}"
            for row in stability.layers
        )
        print(
            f"{direction_id}: {layer_summary}; "
            f"passing={stability.n_layers_passing}/{len(layers)} "
            f"{'PASS' if stability.passed else 'FAIL'}"
        )

    # Extract the pilot safety basis at the same layers solely so random controls
    # can be orthogonal to both safety and learned-control spans.
    safety_specs = [
        ("refusal", "refusal", pairs_dir / "refusal_pilot.jsonl"),
        (
            "assistant_axis",
            "assistant_axis",
            pairs_dir / "assistant_axis_pilot.jsonl",
        ),
        (
            "harmlessness",
            "harmlessness",
            pairs_dir / "harmlessness_pilot.jsonl",
        ),
    ]
    safety_basis = []
    for direction_id, kind, path in safety_specs:
        positive, negative = collect_pair_activations(
            loaded,
            load_jsonl(path),
            layers=layers,
        )
        safety_basis.extend(
            full_directions_from_activations(
                positive,
                negative,
                layers=layers,
                direction_id=direction_id,
                kind=kind,
                domain=None,
            )
        )

    by_layer_fixed = {layer: [] for layer in layers}
    for direction in [*learned_directions, *safety_basis]:
        by_layer_fixed[direction.layer].append(direction.tensor())

    random_directions = []
    random_diagnostics = []
    for layer in layers:
        directions, diagnostics = build_random_orthogonal_controls(
            layer=layer,
            hidden_size=loaded.spec.hidden_size,
            fixed_basis=by_layer_fixed[layer],
            n_random=args.n_random,
            seed=args.seed,
        )
        random_directions.extend(directions)
        random_diagnostics.append(diagnostics)
        print(
            f"random L{layer}: fixed_dot={diagnostics.max_abs_dot_with_fixed_basis:.2e} "
            f"pairwise={diagnostics.max_abs_pairwise_cosine:.2e} "
            f"{'PASS' if diagnostics.passed else 'FAIL'}"
        )

    args.directions_out.parent.mkdir(parents=True, exist_ok=True)
    save_directions(args.directions_out, learned_directions)
    save_directions(args.random_out, random_directions)

    all_learned_pass = all(result.passed for result in stability_results)
    all_random_pass = all(result.passed for result in random_diagnostics)
    payload = {
        "model": model_key,
        "pilot": True,
        "layers": layers,
        "depth_fractions_approx": [
            layer / (loaded.spec.n_layers - 1) for layer in layers
        ],
        "gate": {
            "threshold": args.threshold,
            "required_layers": args.required_layers,
            "total_layers": len(layers),
            "criterion": (
                f"at least {args.required_layers}/{len(layers)} fixed mid/late "
                f"layers with split-half cosine >= {args.threshold}"
            ),
        },
        "learned_controls": [result.to_dict() for result in stability_results],
        "random_controls": [result.to_dict() for result in random_diagnostics],
        "all_learned_controls_passed": all_learned_pass,
        "all_random_controls_passed": all_random_pass,
        "step4_passed": bool(all_learned_pass and all_random_pass),
        "status": (
            "eligible_for_step5_pilot"
            if all_learned_pass and all_random_pass
            else "blocked_for_step5_rebuild_failed_controls"
        ),
        "artifacts": {
            "learned_directions": str(args.directions_out),
            "random_directions": str(args.random_out),
            "contrast_pairs": [str(spec[3]) for spec in control_specs],
        },
        "interpretation": (
            "Step 4 validates extraction reliability only. It does not show "
            "safety-specific deprivilege; that interaction is tested in Step 5."
        ),
        "revision_history": [
            {
                "control": "domain_content_therapy",
                "version": "v1",
                "n_pairs": 12,
                "result": "FAIL (2/4 layers >= 0.70)",
                "action": (
                    "One documented revision: expand to 24 pairs with more "
                    "parallel syntax. If v2 fails, exclude; do not tune again."
                ),
            }
        ],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.results}")
    print(
        f"STEP 4 {'PASS' if payload['step4_passed'] else 'FAIL'}: "
        f"{payload['status']}"
    )
    return 0 if payload["step4_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
