#!/usr/bin/env python3
"""Stage 2 helper — build / summarize the 3×3 controllability matrix J.

Modes:
  correlational  — NOT causal; projections of free-run H onto candidates vs ΔS
                   (diagnostic only; predictive ≠ control)
  from-json      — load deltas already measured by test_channel_controllability.py

Causal estimation lives in test_channel_controllability.py (--mode causal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_channel_control import (  # noqa: E402
    CHANNEL_V,
    ChannelBank,
    jacobian_from_deltas,
    summarize_jacobian,
    unit,
)

CACHE = ROOT / "data" / "results" / "sync_geometry_cache.npz"
OUT = ROOT / "data" / "results" / "sync_channel_controllability_matrix.json"
MD = ROOT / "data" / "results" / "sync_channel_controllability_matrix.md"


def _correlational_J(bank: ChannelBank) -> dict:
    cache = np.load(CACHE)
    Hr = np.asarray(cache["Hr"], dtype=np.float64)
    s_tool = np.asarray(cache["s_tool"], dtype=np.float64)
    s_out = np.asarray(cache["s_out"], dtype=np.float64)
    # C proxy from class if present in geometry
    import json as _json
    eps = _json.loads((ROOT / "data/results/sync_geometry.json").read_text())["episodes"]
    s_c = np.array([1.0 if (e.get("class_intended") in ("B", "C")) else 0.0 for e in eps])

    S = np.stack([s_c, s_tool, s_out], axis=1)  # (n, 3)
    # Effect proxy: corr(h·v_j, S_i) as soft stand-in for J_ij — NOT causal
    J = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        proj = Hr @ bank.V[j]
        for i in range(3):
            a, b = proj, S[:, i]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                J[i, j] = 0.0
            else:
                J[i, j] = float(np.corrcoef(a, b)[0, 1])
    summary = summarize_jacobian(J)
    summary["mode"] = "correlational_not_causal"
    summary["warning"] = (
        "Correlations of candidate projections with observed S. "
        "Do not treat as controllability. Run test_channel_controllability.py --mode causal."
    )
    return summary


def _write_md(report: dict) -> None:
    J = np.asarray(report["J"], dtype=np.float64)
    lines = [
        "# Channel controllability matrix",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- approx_diagonal: **{report.get('approx_diagonal')}**",
        f"- diag_abs_mean: {report.get('diag_abs_mean'):.4f}",
        f"- offdiag_l2: {report.get('offdiag_l2'):.4f}",
        "",
    ]
    if report.get("warning"):
        lines += [f"> {report['warning']}", ""]
    lines += [
        "Rows = affected channel (C, H, O); cols = steered direction (v_C, v_H, v_O).",
        "",
        "| | v_C | v_H | v_O |",
        "|---|-----|-----|-----|",
        f"| ΔC | {J[0,0]:.3f} | {J[0,1]:.3f} | {J[0,2]:.3f} |",
        f"| ΔH | {J[1,0]:.3f} | {J[1,1]:.3f} | {J[1,2]:.3f} |",
        f"| ΔO | {J[2,0]:.3f} | {J[2,1]:.3f} | {J[2,2]:.3f} |",
        "",
    ]
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("correlational", "from-json"), default="correlational")
    ap.add_argument("--deltas", type=Path, default=None, help="JSON from causal test")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.mode == "correlational":
        if not CHANNEL_V.is_file():
            raise SystemExit("run learn_channel_directions.py first")
        bank = ChannelBank.load(CHANNEL_V)
        report = _correlational_J(bank)
    else:
        if not args.deltas or not args.deltas.is_file():
            raise SystemExit("--deltas required for from-json")
        blob = json.loads(args.deltas.read_text())
        J = jacobian_from_deltas(blob["mean_deltas"])
        report = summarize_jacobian(J)
        report["mode"] = "causal_from_json"
        report["source"] = str(args.deltas)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    _write_md(report)
    print(json.dumps({"out": str(args.out), "md": str(MD), **{k: report[k] for k in ("mode", "approx_diagonal", "diag_abs_mean", "offdiag_l2")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
