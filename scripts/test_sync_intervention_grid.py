#!/usr/bin/env python3
"""Smoke test for Layer B intervention grid (hooks mode, no model)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "results" / "sync_intervention_grid.json"


def main() -> int:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sync_intervention_grid.py"), "--mode", "hooks"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return r.returncode
    report = json.loads(OUT.read_text())
    assert report["mode"] == "hooks"
    assert report["n_cells"] == 8
    assert "median_delta_E_sync" in report
    for row in report["rows"]:
        assert len(row["m_star"]) == 3
        assert "delta_E_sync" in row
        assert row["task_id"] == report.get("neutral_task", "api")
        assert row["intervention"] == "none_measure_only"
        assert row["delta_E_sync"] == 0
    assert len({row["task_id"] for row in report["rows"]}) == 1
    print(f"INTERVENTION_GRID_HOOKS_PASS n_improved={report['n_improved']}/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
