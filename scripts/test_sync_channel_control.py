#!/usr/bin/env python3
"""Smoke: Stage-1 learn + correlational J + 8-way refuse without freeze."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))


def main() -> int:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "learn_channel_directions.py"), "--allow-geometry-proxy"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        return r.returncode
    bank_path = ROOT / "data" / "directions" / "sync_channel_V_L4.json"
    assert bank_path.is_file(), bank_path
    blob = json.loads(bank_path.read_text())
    assert "v_C" in blob and "v_H" in blob and "v_O" in blob
    assert blob.get("frozen") is False

    from scripts.sync_channel_control import ChannelBank, compose_direction, error_e

    bank = ChannelBank.load(bank_path)
    e = error_e([0, 1, 0], [1, 1, 1])
    assert e == [-1, 0, -1]
    d = compose_direction(e, bank.V)
    assert d.shape == (bank.V.shape[1],)
    assert float(np.linalg.norm(d)) > 0

    r2 = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_channel_controllability.py"), "--mode", "correlational"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if r2.returncode != 0:
        print(r2.stderr or r2.stdout)
        return r2.returncode
    mat = json.loads((ROOT / "data" / "results" / "sync_channel_controllability_matrix.json").read_text())
    assert mat["mode"] == "correlational_not_causal"
    assert len(mat["J"]) == 3

    r_p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_channel_probes.py"), "--geometry-fallback"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if r_p.returncode != 0:
        print(r_p.stderr or r_p.stdout)
        return r_p.returncode
    probes = json.loads((ROOT / "data" / "results" / "sync_channel_probes.json").read_text())
    assert probes["identification_gate"]["ready_for_8way"] is False

    r3 = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sync_controller_8way.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r3.returncode == 2, r3.stdout + r3.stderr
    assert "REFUSED" in (r3.stderr or "")

    print("CHANNEL_CONTROL_STAGE12_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
