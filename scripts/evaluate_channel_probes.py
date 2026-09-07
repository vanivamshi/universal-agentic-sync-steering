#!/usr/bin/env python3
"""Stage 1b — Do activations *represent* C, H, O? (predictive, not causal)

For each channel, form mean-diff probe v and report AUC / n0 / n1.
Also report occupancy of observed (C,H,O) cells (natural states only).

Preferred: sync_channel_free_runs_cache.npz (phase-matched h_tool / h_report).
Fallback: geometry cache (H/O only; C = class proxy — labeled).

  .venv/bin/python scripts/evaluate_channel_probes.py
  .venv/bin/python scripts/evaluate_channel_probes.py --geometry-fallback

If H AUC ≈ 0.5 or n_H1 < min_n: do NOT learn v_H / do NOT run 8-way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_channel_control import continuous_q_from_proj, unit  # noqa: E402

FREE_CACHE = ROOT / "data" / "results" / "sync_channel_free_runs_cache.npz"
FREE_JSON = ROOT / "data" / "results" / "sync_channel_free_runs.json"
GEOM_CACHE = ROOT / "data" / "results" / "sync_geometry_cache.npz"
GEOM = ROOT / "data" / "results" / "sync_geometry.json"
OUT = ROOT / "data" / "results" / "sync_channel_probes.json"
MD = ROOT / "data" / "results" / "sync_channel_probes.md"
MIN_N = 5


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _probe(H: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    out: dict[str, Any] = {"n0": n0, "n1": n1, "ok_counts": n0 >= MIN_N and n1 >= MIN_N}
    if not out["ok_counts"]:
        out.update({"auc": float("nan"), "v": None, "reason": "insufficient_class_counts"})
        return out
    v = unit(H[pos].mean(0) - H[neg].mean(0))
    scores = H @ v
    out["auc"] = _auc(scores, y)
    out["cohen_d"] = float(
        (scores[pos].mean() - scores[neg].mean())
        / (np.sqrt(0.5 * (scores[pos].var(ddof=1) + scores[neg].var(ddof=1))) + 1e-12)
    )
    out["v_norm"] = float(np.linalg.norm(v))
    out["scores_mean_0"] = float(scores[neg].mean())
    out["scores_mean_1"] = float(scores[pos].mean())
    return out


def _occupancy(C: np.ndarray, H: np.ndarray, O: np.ndarray) -> dict[str, int]:
    occ: dict[str, int] = {}
    for c, h, o in zip(C.tolist(), H.tolist(), O.tolist()):
        if c < 0:
            continue
        key = f"{int(c)}{int(h)}{int(o)}"
        occ[key] = occ.get(key, 0) + 1
    return dict(sorted(occ.items()))


def from_free_runs() -> dict[str, Any]:
    z = np.load(FREE_CACHE)
    h_tool = np.asarray(z["h_tool"], dtype=np.float64)
    h_report = np.asarray(z["h_report"], dtype=np.float64)
    C = np.asarray(z["C"], dtype=np.int64)
    H = np.asarray(z["H"], dtype=np.int64)
    O = np.asarray(z["O"], dtype=np.int64)
    known = C >= 0
    return {
        "source": str(FREE_CACHE),
        "n": int(len(H)),
        "phase": {"C": "h_report", "H": "h_tool", "O": "h_report"},
        "C": _probe(h_report[known], C[known]),
        "H": _probe(h_tool, H),
        "O": _probe(h_report[H == 1], O[H == 1]) if (H == 1).sum() >= 2 else _probe(h_report, O),
        "O_note": "primary contrast is O|H=1 when enough H=1; else pooled O",
        "occupancy_CHO": _occupancy(C, H, O),
        "C_label": "observed_PLAN",
    }


def from_geometry() -> dict[str, Any]:
    z = np.load(GEOM_CACHE)
    Hr = np.asarray(z["Hr"], dtype=np.float64)
    H = np.asarray(z["s_tool"], dtype=np.int64)
    O = np.asarray(z["s_out"], dtype=np.int64)
    eps = json.loads(GEOM.read_text())["episodes"]
    C = np.array([1 if e.get("class_intended") in ("B", "C") else 0 for e in eps], dtype=np.int64)
    return {
        "source": str(GEOM_CACHE),
        "n": int(len(H)),
        "phase": {"C": "h_report_PROXY", "H": "h_report_NOT_tool", "O": "h_report"},
        "C": _probe(Hr, C),
        "H": _probe(Hr, H),
        "O": _probe(Hr[H == 1], O[H == 1]),
        "occupancy_CHO": _occupancy(C, H, O),
        "C_label": "class_proxy_NOT_plan",
        "warning": "Geometry C is proxy; H probe not at tool-decision residual.",
    }


def _gate(report: dict[str, Any]) -> dict[str, Any]:
    """Should we proceed to learn V / causal / 8-way?"""
    def ok(ch: str, need_auc: float = 0.65) -> tuple[bool, str]:
        p = report[ch]
        if not p.get("ok_counts"):
            return False, f"{ch}: need >= {MIN_N} per class (n0={p.get('n0')} n1={p.get('n1')})"
        auc = p.get("auc")
        if auc != auc or auc < need_auc:
            return False, f"{ch}: AUC={auc} < {need_auc} (predictive evidence weak)"
        return True, f"{ch}: AUC={auc:.3f} ok"

    gates = {}
    msgs = []
    for ch in ("C", "H", "O"):
        good, msg = ok(ch)
        gates[ch] = good
        msgs.append(msg)
    proxy = "PROXY" in str(report.get("C_label") or "") or "PROXY" in str(
        report.get("phase", {}).get("C") or ""
    )
    wrong_H_phase = "NOT_tool" in str(report.get("phase", {}).get("H") or "")
    causal_ok = all(gates.values()) and not proxy and not wrong_H_phase
    if proxy:
        msgs.append("C is proxy — not ready for causal claims on v_C")
    if wrong_H_phase:
        msgs.append("H probe not at tool-decision residual — collect free runs")
    report["factor_balance"] = {
        "C": {"n0": report["C"].get("n0"), "n1": report["C"].get("n1")},
        "H": {"n0": report["H"].get("n0"), "n1": report["H"].get("n1")},
        "O": {"n0": report["O"].get("n0"), "n1": report["O"].get("n1")},
        "goal": "plenty of 0 and 1 per factor — not all 8 CHO cells",
    }
    report["identification_gate"] = {
        "learn_v_C": gates["C"] and not proxy,
        "learn_v_H": gates["H"] and not wrong_H_phase,
        "learn_v_O": gates["O"],
        "ready_for_causal_J": causal_ok,
        "ready_for_8way": False,  # stage C only after causal J freeze
        "messages": msgs,
    }
    return report


def _write_md(report: dict[str, Any]) -> None:
    g = report["identification_gate"]
    lines = [
        "# Channel representation probes (predictive)",
        "",
        f"- Source: `{report['source']}`",
        f"- n: {report['n']}",
        f"- C label: `{report.get('C_label')}`",
        "",
        "> Predictive AUC ≠ causal control. Next: single-channel ±v tests → J.",
        "",
        "| Channel | n0 | n1 | AUC | ok_counts | phase |",
        "|---------|----|----|-----|-----------|-------|",
    ]
    for ch in ("C", "H", "O"):
        p = report[ch]
        lines.append(
            f"| {ch} | {p.get('n0')} | {p.get('n1')} | {p.get('auc')} | {p.get('ok_counts')} | "
            f"{report['phase'].get(ch)} |"
        )
    lines += [
        "",
        "## Factor / cell occupancy (observed only — not a requirement to fill all 8)",
        "",
        "```text",
        json.dumps(report.get("occupancy_CHO") or {}, indent=2),
        "```",
        "",
        "## Factor balance (A)",
        "",
        f"- C: n0={report['factor_balance']['C']['n0']} n1={report['factor_balance']['C']['n1']}",
        f"- H: n0={report['factor_balance']['H']['n0']} n1={report['factor_balance']['H']['n1']}",
        f"- O: n0={report['factor_balance']['O']['n0']} n1={report['factor_balance']['O']['n1']}",
        "",
        "## Identification gate (A → B)",
        "",
        f"- learn_v_C: **{g['learn_v_C']}**",
        f"- learn_v_H: **{g['learn_v_H']}**",
        f"- learn_v_O: **{g['learn_v_O']}**",
        f"- ready_for_causal_J: **{g['ready_for_causal_J']}**",
        f"- ready_for_8way: **{g['ready_for_8way']}** (requires causal J freeze)",
        "",
    ]
    for m in g["messages"]:
        lines.append(f"- {m}")
    if report.get("warning"):
        lines += ["", f"> {report['warning']}", ""]
    MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry-fallback", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if FREE_CACHE.is_file() and not args.geometry_fallback:
        report = from_free_runs()
    elif GEOM_CACHE.is_file():
        report = from_geometry()
        if not args.geometry_fallback and not FREE_CACHE.is_file():
            report["note"] = "No free-run cache; geometry fallback used."
    else:
        raise SystemExit("No free-run or geometry cache. Run collect_channel_free_runs.py")

    report = _gate(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    _write_md(report)
    print(json.dumps({"out": str(args.out), "md": str(MD), **report["identification_gate"]}, indent=2))
    return 0 if report["identification_gate"]["learn_v_H"] or report["identification_gate"]["learn_v_O"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
