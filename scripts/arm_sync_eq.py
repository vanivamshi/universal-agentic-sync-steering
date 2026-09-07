#!/usr/bin/env python3
"""Arm m* for scoring + alignment controller target.

m* is the requested behavioral state for the activation controller (and scorer).
It is NOT pasted into the Agent PLAN/FINAL prompt as hide/disclose text.

  .venv/bin/python scripts/arm_sync_eq.py --m-star 1,1,0
  .venv/bin/python scripts/arm_sync_eq.py --fix          # (1,1,1)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cursor" / "hooks" / "state" / "sync_session.json"
sys.path.insert(0, str(ROOT))

from scripts.sync_eq import parse_m_star  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m-star", default=None, help="e.g. 1,1,0 or 011")
    ap.add_argument("--fix", action="store_true", help="m*=(1,1,1)")
    ap.add_argument(
        "--text-repair",
        action="store_true",
        help="DEPRECATED Layer-C: allow stop-hook text adapt followups",
    )
    args = ap.parse_args()
    if args.fix:
        m = [1, 1, 1]
    elif args.m_star:
        s = args.m_star.strip().replace(" ", "")
        if len(s) == 3 and set(s) <= {"0", "1"}:
            m = [int(s[0]), int(s[1]), int(s[2])]
        else:
            m = parse_m_star([int(x) for x in s.split(",")])
    else:
        ap.error("use --m-star or --fix")

    state: dict = {}
    if STATE.is_file():
        try:
            state = json.loads(STATE.read_text())
        except Exception:
            state = {}
    state["m_star"] = m
    state["S"] = None
    state["e"] = None
    state["followup_from_eq"] = None
    # Natural free-run by default — activation controller owns control, not text adapt.
    state["policy_mode"] = "text_repair" if args.text_repair else "measure_only"
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"m_star": m, "equation": "e = m* - S"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
