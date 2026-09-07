#!/usr/bin/env python3
"""DEPRECATED — hook/scorer regression moved to run_sync_hook_regression.py.

The old 8/8 AGENT_ALL_MASKS_PASS artifact used replay + _apply_adapt(); it is not
live Agent evidence or causal control proof. This stub forwards to the regression runner.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "run_sync_hook_regression.py"


def main() -> int:
    print(
        "DEPRECATED: run_sync_agent_all_masks.py → run_sync_hook_regression.py\n"
        "See docs/sync_implementation_audit.md",
        file=sys.stderr,
    )
    return subprocess.call([sys.executable, str(TARGET), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
