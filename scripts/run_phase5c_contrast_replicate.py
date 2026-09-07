#!/usr/bin/env python3
"""Phase 5C contrast replication — matched protocol, more reps.

Only the C|h0 vs C|h_H and H|h0 vs H|h_C contrasts (no path Jacobian).
Same packing as run_phase5_transition_compatibility.py:
  active[k]=(±1, v); make_steer_hook(loaded, compose(±v), α).

If cross_ratio(C after H) > 1.25 → keep as trajectory-coupling finding.
Else → discard as unstable diagnostic. Do not block 8-way on this.

  .venv/bin/python scripts/run_phase5c_contrast_replicate.py --reps 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts.sync_channel_margins import SPECS, margin_at_site, messages_for_channel, unit  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase5c_contrast_replicate.json"
MD = ROOT / "data" / "results" / "sync_phase5c_contrast_replicate.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

SEED = 20260907
ALPHA = 1.5
CHANNELS = ("C", "H", "O")


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


def _compose(active: dict[str, tuple[float, np.ndarray]]) -> np.ndarray | None:
    d = None
    for k in CHANNELS:
        if k not in active:
            continue
        ek, vk = active[k]
        term = float(ek) * vk
        d = term if d is None else d + term
    return d


def measure_M(loaded, sc, *, active: dict | None = None, alpha: float = ALPHA) -> dict[str, float]:
    active = active or {}
    pd = _compose(active)
    out: dict[str, float] = {}
    for k in CHANNELS:
        msgs = messages_for_channel(sc, k)
        if pd is None:
            out[k] = float(margin_at_site(loaded, msgs, SPECS[k])["M"])
        else:
            hook = make_steer_hook(loaded, pd, alpha)
            out[k] = float(margin_at_site(loaded, msgs, SPECS[k], hook=hook)["M"])
    return out


def apply_and_delta(
    loaded, sc, *, j: str, e_j: float, v_j: np.ndarray, prior: dict, alpha: float
) -> dict[str, Any]:
    M0 = measure_M(loaded, sc, active=prior, alpha=alpha)
    after = dict(prior)
    after[j] = (float(e_j), v_j)
    M1 = measure_M(loaded, sc, active=after, alpha=alpha)
    dM = {k: float(M1[k] - M0[k]) for k in CHANNELS}
    return {
        "dM": dM,
        "cross_abs_mean": float(np.mean([abs(dM[i]) for i in CHANNELS if i != j])),
    }


def _summarize(rows: list, key_from: str, key_to: str, channel: str) -> dict[str, float]:
    a = [abs(r[key_from]["dM"][channel]) for r in rows]
    b = [abs(r[key_to]["dM"][channel]) for r in rows]
    ca = [r[key_from]["cross_abs_mean"] for r in rows]
    cb = [r[key_to]["cross_abs_mean"] for r in rows]
    return {
        f"abs_dM_{channel}_from": float(np.mean(a)),
        f"abs_dM_{channel}_to": float(np.mean(b)),
        "retention": float(np.mean(b) / max(np.mean(a), 1e-9)),
        "cross_from": float(np.mean(ca)),
        "cross_to": float(np.mean(cb)),
        "cross_ratio": float(np.mean(cb) / max(np.mean(ca), 1e-9)),
        "dM_H_from_mean": float(np.mean([r[key_from]["dM"]["H"] for r in rows])),
        "dM_H_to_mean": float(np.mean([r[key_to]["dM"]["H"] for r in rows])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    vc = _load_vc()
    signs = (+1.0, -1.0)

    print("=== C|h0 vs C|h_H (matched 5C) ===", flush=True)
    c_rows = []
    for r in range(args.reps):
        for ej in signs:
            c0 = apply_and_delta(
                loaded, sc, j="C", e_j=ej, v_j=vc["C"], prior={}, alpha=args.alpha
            )
            prior_H = {"H": (-1.0, vc["H"])}
            if r % 2 == 1:
                prior_H = {"H": (ej, vc["H"])}
            cH = apply_and_delta(
                loaded, sc, j="C", e_j=ej, v_j=vc["C"], prior=prior_H, alpha=args.alpha
            )
            c_rows.append({"rep": r, "e_C": ej, "prior_H": prior_H["H"][0], "C_h0": c0, "C_hH": cH})
            print(
                f"  r={r} eC={ej:+.0f}: |dM_C| {abs(c0['dM']['C']):.2f}→{abs(cH['dM']['C']):.2f} "
                f"cross {c0['cross_abs_mean']:.2f}→{cH['cross_abs_mean']:.2f}",
                flush=True,
            )

    print("=== H|h0 vs H|h_C (matched 5C) ===", flush=True)
    h_rows = []
    for r in range(args.reps):
        for ej in signs:
            h0 = apply_and_delta(
                loaded, sc, j="H", e_j=ej, v_j=vc["H"], prior={}, alpha=args.alpha
            )
            prior_C = {"C": (-1.0, vc["C"])}
            if r % 2 == 1:
                prior_C = {"C": (ej, vc["C"])}
            hC = apply_and_delta(
                loaded, sc, j="H", e_j=ej, v_j=vc["H"], prior=prior_C, alpha=args.alpha
            )
            h_rows.append({"rep": r, "e_H": ej, "prior_C": prior_C["C"][0], "H_h0": h0, "H_hC": hC})
            print(
                f"  r={r} eH={ej:+.0f}: |dM_H| {abs(h0['dM']['H']):.2f}→{abs(hC['dM']['H']):.2f} "
                f"cross {h0['cross_abs_mean']:.2f}→{hC['cross_abs_mean']:.2f}",
                flush=True,
            )

    c_sum = _summarize(c_rows, "C_h0", "C_hH", "C")
    h_sum = _summarize(h_rows, "H_h0", "H_hC", "H")
    keep_CH = bool(c_sum["cross_ratio"] > 1.25)
    gate = {
        "C_cross_worsens": keep_CH,
        "H_cross_worsens": bool(h_sum["cross_ratio"] > 1.25),
        "keep_as_trajectory_coupling": keep_CH,
        "discard_as_unstable": not keep_CH,
        "note": (
            "Keep C→H coupling finding iff cross_ratio>1.25; else discard. "
            "8-way validation proceeds regardless."
        ),
    }
    payload = {
        "protocol": "5C contrast replicate (matched)",
        "alpha": args.alpha,
        "reps": args.reps,
        "seed": args.seed,
        "C_contrast": c_sum,
        "H_contrast": h_sum,
        "gate": gate,
        "original_5C": {"C_retention": 0.83, "C_cross_ratio": 1.33, "H_retention": 0.81, "H_cross_ratio": 0.66},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 5C — Contrast replication (matched protocol)",
        "",
        f"α={args.alpha}, reps={args.reps} (matched packing: active=(±1,v), hook α).",
        "",
        "## C|h0 → C|h_H",
        "",
        f"| | retention | cross ratio | cross from→to |",
        f"|--|-----------|-------------|----------------|",
        f"| original 5C | 0.83 | **1.33** | 0.87→1.16 |",
        f"| **replicate** | {c_sum['retention']:.2f} | **{c_sum['cross_ratio']:.2f}** | "
        f"{c_sum['cross_from']:.2f}→{c_sum['cross_to']:.2f} |",
        "",
        "## H|h0 → H|h_C",
        "",
        f"| | retention | cross ratio |",
        f"|--|-----------|-------------|",
        f"| original 5C | 0.81 | 0.66 |",
        f"| **replicate** | {h_sum['retention']:.2f} | {h_sum['cross_ratio']:.2f} |",
        "",
        "## Decision",
        "",
        f"- Keep as trajectory-coupling finding: **{gate['keep_as_trajectory_coupling']}**",
        f"- Discard as unstable diagnostic: **{gate['discard_as_unstable']}**",
        f"- 8-way: **OPEN (not blocked by this)**",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "gate": gate, "C": c_sum, "H": h_sum}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
