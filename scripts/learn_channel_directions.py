#!/usr/bin/env python3
"""Stage 1 — Learn candidate v_C, v_H, v_O from free-run activations.

Preferred corpus (Experiment 1):
  data/results/sync_channel_free_runs_cache.npz
  built by collect_channel_free_runs.py (PLAN + h_tool + h_report)

  v_H = unit(μ(h_tool | H=1) − μ(h_tool | H=0))     # tool-phase residual
  v_O = unit(μ(h_report | H=1,O=1) − μ(h_report | H=1,O=0))
  v_C = unit(μ(h_report | C=1) − μ(h_report | C=0))  # requires observed PLAN

Fallback (--allow-geometry-proxy): old geometry cache; v_C is class proxy
(NOT a real plan direction — marked in meta; Stage 2 should refuse freeze).

  .venv/bin/python scripts/learn_channel_directions.py --from-free-runs
  .venv/bin/python scripts/learn_channel_directions.py --allow-geometry-proxy
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

from scripts.sync_channel_control import (  # noqa: E402
    ALPHA_DEFAULT,
    CHANNEL_V,
    ChannelBank,
    LAYER_DEFAULT,
    orthogonalize_channels,
    project_out,
    save_bank,
    unit,
)

FREE_CACHE = ROOT / "data" / "results" / "sync_channel_free_runs_cache.npz"
FREE_JSON = ROOT / "data" / "results" / "sync_channel_free_runs.json"
GEOM_CACHE = ROOT / "data" / "results" / "sync_geometry_cache.npz"
GEOM = ROOT / "data" / "results" / "sync_geometry.json"
MIN_N = 5


def _mean_diff(H: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray | None, dict]:
    meta = {"n_pos": int(np.sum(pos)), "n_neg": int(np.sum(neg))}
    if meta["n_pos"] < MIN_N or meta["n_neg"] < MIN_N:
        return None, {**meta, "ok": False}
    return unit(H[pos].mean(0) - H[neg].mean(0)), {**meta, "ok": True}


def learn_from_free_runs() -> tuple[np.ndarray, dict[str, Any]]:
    if not FREE_CACHE.is_file():
        raise SystemExit(
            f"missing {FREE_CACHE}\n"
            "Run: .venv/bin/python scripts/collect_channel_free_runs.py --n 30"
        )
    z = np.load(FREE_CACHE)
    h_tool = np.asarray(z["h_tool"], dtype=np.float64)
    h_report = np.asarray(z["h_report"], dtype=np.float64)
    C = np.asarray(z["C"], dtype=np.int64)
    H = np.asarray(z["H"], dtype=np.int64)
    O = np.asarray(z["O"], dtype=np.int64)

    v_H, m_H = _mean_diff(h_tool, H == 1, H == 0)
    priv = H == 1
    v_O, m_O = _mean_diff(h_report, priv & (O == 1), priv & (O == 0))
    known = C >= 0
    v_C, m_C = _mean_diff(h_report, known & (C == 1), known & (C == 0))

    if v_H is None:
        raise SystemExit(f"insufficient H contrast: {m_H}")
    if v_O is None:
        raise SystemExit(f"insufficient O contrast given H=1: {m_O}")
    if v_C is None:
        raise SystemExit(
            f"insufficient observed PLAN C contrast: {m_C}\n"
            "Need more free runs with PLAN: ... text (collect_channel_free_runs.py)."
        )

    meta = {
        "stage": 1,
        "source": str(FREE_CACHE),
        "status": "candidates_from_free_runs_not_causally_validated",
        "v_C_status": "candidate_from_observed_plan",
        "v_H_status": "candidate_from_h_tool",
        "v_O_status": "candidate_from_h_report_given_H1",
        "contrast": {"C": m_C, "H": m_H, "O": m_O},
        "phase_match": {"v_C": "report", "v_H": "tool", "v_O": "report"},
        "cos_raw": {
            "C_H": float(np.dot(v_C, v_H)),
            "C_O": float(np.dot(v_C, v_O)),
            "H_O": float(np.dot(v_H, v_O)),
        },
        "note": "Do not freeze until Stage-2 causal J passes.",
    }
    return np.stack([v_C, v_H, v_O], 0), meta


def learn_geometry_proxy() -> tuple[np.ndarray, dict[str, Any]]:
    """DEPRECATED for claims — v_C is class proxy; v_H from report residual."""
    cache = np.load(GEOM_CACHE)
    Hr = np.asarray(cache["Hr"], dtype=np.float64)
    s_tool = np.asarray(cache["s_tool"], dtype=np.int64)
    s_out = np.asarray(cache["s_out"], dtype=np.int64)
    episodes = json.loads(GEOM.read_text())["episodes"]
    cls = np.array([str(e.get("class_intended") or "") for e in episodes])

    v_H, m_H = _mean_diff(Hr, s_tool == 1, s_tool == 0)
    priv = s_tool == 1
    v_O, m_O = _mean_diff(Hr, priv & (s_out == 1), priv & (s_out == 0))
    v_C, m_C = _mean_diff(Hr, np.isin(cls, ["B", "C"]), np.isin(cls, ["A", "D"]))
    if v_H is None or v_O is None or v_C is None:
        raise SystemExit(f"geometry proxy failed H={m_H} O={m_O} C={m_C}")
    meta = {
        "stage": 1,
        "source": str(GEOM_CACHE),
        "status": "GEOMETRY_PROXY_NOT_FOR_CLAIMS",
        "v_C_name": "v_C_proxy",
        "note_naming": "Do not call geometry direction v_C; it is v_C_proxy only.",
        "v_C_status": "v_C_proxy_class_BC_vs_AD_NOT_plan",
        "v_H_status": "WARNING_learned_from_h_report_not_h_tool",
        "v_O_status": "candidate_from_h_report_given_H1",
        "contrast": {"C": m_C, "H": m_H, "O": m_O},
        "cos_raw": {
            "C_H": float(np.dot(v_C, v_H)),
            "C_O": float(np.dot(v_C, v_O)),
            "H_O": float(np.dot(v_H, v_O)),
        },
        "note": "Proxy only. Collect free runs with PLAN before Stage-2 freeze.",
    }
    return np.stack([v_C, v_H, v_O], 0), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-free-runs", action="store_true", default=True)
    ap.add_argument("--allow-geometry-proxy", action="store_true")
    ap.add_argument("--orthogonalize", action="store_true")
    ap.add_argument("--force", action="store_true", help="learn even if Stage-A gate fails")
    ap.add_argument("--project-out-collateral", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=CHANNEL_V)
    args = ap.parse_args()

    if args.allow_geometry_proxy:
        V_raw, meta = learn_geometry_proxy()
    elif FREE_CACHE.is_file() or args.from_free_runs:
        if not FREE_CACHE.is_file():
            raise SystemExit(
                "No free-run cache. Either:\n"
                "  .venv/bin/python scripts/collect_channel_free_runs.py --n 30\n"
                "or exploratory only:\n"
                "  .venv/bin/python scripts/learn_channel_directions.py --allow-geometry-proxy"
            )
        # Stage A gate
        probe_path = ROOT / "data" / "results" / "sync_channel_probes.json"
        if probe_path.is_file() and not args.force:
            gate = json.loads(probe_path.read_text()).get("identification_gate") or {}
            need = ("learn_v_C", "learn_v_H", "learn_v_O")
            if not all(gate.get(k) for k in need):
                raise SystemExit(
                    "Stage A gate failed: " + str(gate.get("messages"))
                    + "\nRe-run: scripts/run_channel_stage_a.py\n"
                    + "Or pass --force to learn anyway (not recommended)."
                )
        V_raw, meta = learn_from_free_runs()
    else:
        V_raw, meta = learn_geometry_proxy()

    v_C, v_H, v_O = V_raw[0], V_raw[1], V_raw[2]
    if args.project_out_collateral:
        v_H = project_out(v_H, v_O, v_C)
        v_O = project_out(v_O, v_H, v_C)
        v_C = project_out(v_C, v_H, v_O)
    V = np.stack([v_C, v_H, v_O], 0)
    if args.orthogonalize:
        V = orthogonalize_channels(V)

    bank = ChannelBank(V=V, layer=LAYER_DEFAULT, alpha=ALPHA_DEFAULT, meta=meta, frozen=False)
    save_bank(bank, args.out)
    print(json.dumps({"out": str(args.out), "meta_status": meta["status"], "v_C_status": meta["v_C_status"], "contrast": meta["contrast"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
