#!/usr/bin/env python3
"""DEPRECATED — canned NATURAL + scripted _follow_policy_repair. Not free-run evidence.

Use:
  .venv/bin/python scripts/run_sync_policy_free_run.py --init
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "DEPRECATED: record_sync_agent_live_results.py uses canned answers.\n"
        "Use:\n"
        "  .venv/bin/python scripts/run_sync_policy_free_run.py --init\n"
        "See docs/sync_universal_equation.md",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
