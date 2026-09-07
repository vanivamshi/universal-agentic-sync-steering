"""G3 holdout recheck for therapy/writing v2 domain-content controls.

Extract direction on train (v1 12 pairs), evaluate split-half stability on
holdout (v2 additions only). Passes if ≥3/4 F_mid layers have holdout
split-half cosine ≥ 0.70 — same gate as §4, but out-of-sample pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--device", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=[14, 16, 19, 22])
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--required-layers", type=int, default=3)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "control_holdout_recheck.json",
    )
    args = ap.parse_args()

    from activation_pipeline.controls import collect_pair_activations, learned_control_stability
    from activation_pipeline.device import resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    loaded = load_model_and_tokenizer(
        args.model,
        device_map=resolve_device_map(args.device),
        dtype="float32",
        local_files_only=True,
    )
    pairs_dir = ROOT / "data" / "contrast_pairs"
    specs = [
        (
            "therapy",
            pairs_dir / "domain_content_therapy_pilot.jsonl",
            pairs_dir / "domain_content_therapy_pilot_v2.jsonl",
        ),
        (
            "writing",
            pairs_dir / "domain_content_writing_pilot.jsonl",
            pairs_dir / "domain_content_writing_pilot_v2.jsonl",
        ),
    ]
    results: dict = {
        "model": args.model,
        "layers": args.layers,
        "threshold": args.threshold,
        "required_layers": args.required_layers,
        "protocol": (
            "train=v1 pairs; holdout=v2 additions only; "
            "split-half stability on holdout activations"
        ),
        "domains": {},
    }

    for domain, train_path, full_v2_path in specs:
        train = _load_pairs(train_path)
        full = _load_pairs(full_v2_path)
        holdout = full[len(train) :]
        if len(holdout) < 4:
            raise SystemExit(f"{domain}: need ≥4 holdout pairs, got {len(holdout)}")

        print(f"{domain}: train={len(train)} holdout={len(holdout)} collecting…")
        # Direction from train (sanity: ensure train itself still passes)
        pos_tr, neg_tr = collect_pair_activations(loaded, train, layers=args.layers)
        train_stab = learned_control_stability(
            pos_tr,
            neg_tr,
            layers=args.layers,
            direction_id=f"domain_content_{domain}_train",
            kind="domain_content",
            domain=domain,
            n_boot=args.n_boot,
            n_splits_per_stat=args.n_splits,
            threshold=args.threshold,
            required_layers=args.required_layers,
        )
        # Holdout-only stability (the overfitting check)
        pos_ho, neg_ho = collect_pair_activations(loaded, holdout, layers=args.layers)
        holdout_stab = learned_control_stability(
            pos_ho,
            neg_ho,
            layers=args.layers,
            direction_id=f"domain_content_{domain}_holdout",
            kind="domain_content",
            domain=domain,
            n_boot=args.n_boot,
            n_splits_per_stat=args.n_splits,
            threshold=args.threshold,
            required_layers=args.required_layers,
        )
        results["domains"][domain] = {
            "n_train": len(train),
            "n_holdout": len(holdout),
            "train": train_stab.to_dict(),
            "holdout": holdout_stab.to_dict(),
            "holdout_pass": holdout_stab.passed,
        }
        print(
            f"  train passed={train_stab.passed} "
            f"({train_stab.n_layers_passing}/{len(args.layers)}); "
            f"holdout passed={holdout_stab.passed} "
            f"({holdout_stab.n_layers_passing}/{len(args.layers)})"
        )
        for row in holdout_stab.layers:
            print(
                f"    holdout L{row['layer']}: cos={row['split_half_cosine']:.3f} "
                f"BCa={row['bca_ci95']}"
            )

    n_pass = sum(1 for d in results["domains"].values() if d["holdout_pass"])
    results["summary"] = {
        "n_domains": len(results["domains"]),
        "n_holdout_pass": n_pass,
        "all_holdout_pass": n_pass == len(results["domains"]),
        "verdict": (
            "HOLDOUT_PASS"
            if n_pass == len(results["domains"])
            else "HOLDOUT_FAIL_OVERFIT_RISK"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out} verdict={results['summary']['verdict']}")
    return 0 if results["summary"]["all_holdout_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
