#!/usr/bin/env python3
"""Run Stage A: collect free trajectories → representation probes → print gate.

Does NOT learn V, run causal J, or 8-way.

  .venv/bin/python scripts/run_channel_stage_a.py --n 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--skip-collect", action="store_true", help="only re-probe existing cache")
    args = ap.parse_args()

    if not args.skip_collect:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "collect_channel_free_runs.py"),
            "--n",
            str(args.n),
            "--seed",
            str(args.seed),
        ]
        if args.append:
            cmd.append("--append")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            return r.returncode

    r2 = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_channel_probes.py")],
        cwd=str(ROOT),
    )
    if r2.returncode != 0:
        return r2.returncode

    probes = json.loads((ROOT / "data" / "results" / "sync_channel_probes.json").read_text())
    gate = probes["identification_gate"]
    bal = probes.get("factor_balance") or {}
    print("\n=== Stage A summary ===")
    print(json.dumps({"factor_balance": bal, "gate": gate}, indent=2))
    if not gate.get("ready_for_causal_J"):
        print(
            "\nNext: fix factor balance / task distribution, then re-run Stage A.\n"
            "Do NOT run learn/causal/8-way until ready_for_causal_J is true.",
            file=sys.stderr,
        )
        return 0
    print(
        "\nStage A gate passed for representation. Next Stage B:\n"
        "  .venv/bin/python scripts/learn_channel_directions.py --from-free-runs\n"
        "  .venv/bin/python scripts/test_channel_controllability.py --mode causal --reps 2\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
