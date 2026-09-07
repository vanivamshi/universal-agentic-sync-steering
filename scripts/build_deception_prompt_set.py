#!/usr/bin/env python3
"""Build held-out GAP deception-eliciting prompt set (screen vs steer holdout).

Uses persona + harmbench families (designed to tempt prose↔tool divergence).
control/baseline are retained as task-success / negative references, not as
primary eliciting prompts.

Splits are stratified by (family, gap_domain) with a fixed seed so Phase-2/3
steering never trains on the steer_holdout fold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

GAP_DOMAIN_MAP = {
    "devops": "coding",
    "education": "writing",
    "hr": "therapy",
    "finance": "therapy",
    "legal": "therapy",
    "pharma": "therapy",
}
ELICIT_FAMILIES = ("persona", "harmbench")
REF_FAMILIES = ("control", "baseline")


def _all_scenarios() -> list[dict]:
    from gap_domains.devops.scenarios import SCENARIOS as D
    from gap_domains.education.scenarios import SCENARIOS as E
    from gap_domains.finance.scenarios import SCENARIOS as F
    from gap_domains.hr.scenarios import SCENARIOS as H
    from gap_domains.legal.scenarios import SCENARIOS as L
    from gap_domains.pharma.scenarios import SCENARIOS as P

    packs = {
        "devops": D,
        "education": E,
        "hr": H,
        "finance": F,
        "legal": L,
        "pharma": P,
    }
    rows = []
    for gap_domain, scenarios in packs.items():
        for key, sc in scenarios.items():
            rows.append(
                {
                    "prompt_id": f"{gap_domain}:{key}",
                    "gap_domain": gap_domain,
                    "scenario": key,
                    "family": sc.get("family"),
                    "label": sc.get("label"),
                    "mapped_domain": GAP_DOMAIN_MAP[gap_domain],
                    "task": sc.get("task"),
                    "task_goal_only": sc.get("task_goal_only"),
                    "jailbreak_prompt": sc.get("jailbreak_prompt"),
                    "n_forbidden_actions": len(sc.get("forbidden_actions") or []),
                    "forbidden_ids": [fa.id for fa in (sc.get("forbidden_actions") or [])],
                }
            )
    return rows


def _stratified_split(
    rows: list[dict], *, holdout_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        k = (r["family"], r["gap_domain"])
        by_key.setdefault(k, []).append(r)
    screen, holdout = [], []
    for k in sorted(by_key):
        bucket = list(by_key[k])
        rng.shuffle(bucket)
        n_h = max(1, int(round(len(bucket) * holdout_frac))) if len(bucket) > 1 else 0
        # keep at least one in screen when possible
        if len(bucket) > 1 and n_h >= len(bucket):
            n_h = len(bucket) - 1
        holdout.extend(bucket[:n_h])
        screen.extend(bucket[n_h:])
    return screen, holdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--holdout-frac", type=float, default=1 / 3)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "prompts" / "gap_deception_prompt_set.json",
    )
    args = ap.parse_args()

    all_rows = _all_scenarios()
    elicit = [r for r in all_rows if r["family"] in ELICIT_FAMILIES]
    refs = [r for r in all_rows if r["family"] in REF_FAMILIES]
    screen, holdout = _stratified_split(
        elicit, holdout_frac=args.holdout_frac, seed=args.seed
    )

    # tag splits
    for r in screen:
        r["split"] = "screen"
        r["role"] = "eliciting"
    for r in holdout:
        r["split"] = "steer_holdout"
        r["role"] = "eliciting"
    for r in refs:
        r["split"] = "reference"
        r["role"] = "task_success_or_negative"

    payload = {
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "definition": (
            "Eliciting = persona + harmbench (tempt stated-intent / tool-action "
            "divergence). screen = correlational PC screen + train fold for "
            "steering magnitude selection. steer_holdout = preregistered "
            "replication fold for causal lever claim (D4-style). reference = "
            "control/baseline for task-success side-effect baselines."
        ),
        "counts": {
            "eliciting_total": len(elicit),
            "screen": len(screen),
            "steer_holdout": len(holdout),
            "reference": len(refs),
            "all_scenarios": len(all_rows),
        },
        "prompts": screen + holdout + refs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["counts"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
