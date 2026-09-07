#!/usr/bin/env python3
"""Reset sync session state at Agent session start (clean team-lead demos)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".cursor" / "hooks" / "state" / "sync_session.json"

IDLE = {
    "phase": "idle",
    "private_access": False,
    "paths": [],
    "tool_hook_fired": False,
    "tool_hook_this_turn": False,
    "equation_fired": False,
    "repair_triggered": False,
    "disclosed": False,
    "events": [],
}


def main() -> int:
    sys.stdin.read()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    proof = None
    if STATE.is_file():
        try:
            prev = json.loads(STATE.read_text())
            proof = prev.get("proof")
        except Exception:
            proof = None
    out = dict(IDLE)
    if proof:
        out["proof"] = proof
    # Do not carry m* across sessions — arm each demo explicitly.
    STATE.write_text(json.dumps(out, indent=2) + "\n")
    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
