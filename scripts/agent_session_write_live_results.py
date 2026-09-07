#!/usr/bin/env python3
"""DEPRECATED — canned answers / scripted repair. Not universal-alignment evidence.

Use:
  .venv/bin/python scripts/run_sync_policy_free_run.py --init
See docs/sync_universal_equation.md
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "DEPRECATED: agent_session_write_live_results.py packages canned PLAN/FINAL.\n"
        "Use free-run measurement instead:\n"
        "  .venv/bin/python scripts/run_sync_policy_free_run.py --init\n"
        "Causal control: scripts/run_sync_intervention_grid.py --mode qwen\n"
        "See docs/sync_universal_equation.md",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
