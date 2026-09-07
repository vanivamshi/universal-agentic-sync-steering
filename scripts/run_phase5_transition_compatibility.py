#!/usr/bin/env python3
"""Phase 5C — Transition compatibility (not direction search).

5A/5B locked: failure ≠ wrong local direction (gradient adapt ✗, v_opt ✗;
C actuator retention ≈ 0.85–0.88).

Question:
  Does an intervention that improves one channel move the trajectory into a
  state where another channel's target becomes harder?

Protocol (frozen v_c, C→H→O sites):
  Measure M(h) = [M_C, M_H, M_O] at each state.
  Apply α e_j v_c^j; record ΔM^{(j)} = M(after) − M(before).
  Transition Jacobian T_ij ≈ ΔM_i / α when steering j.

Special contrasts:
  C|h0  vs  C|h_H
  H|h0  vs  H|h_C

  .venv/bin/python scripts/run_phase5_transition_compatibility.py --reps 4
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

from scripts.sync_channel_margins import (  # noqa: E402
    SPECS,
    margin_at_site,
    messages_for_channel,
    unit,
)
from scripts.sync_h_decision import LAYER_DEFAULT, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase5_transition_compatibility.json"
MD = ROOT / "data" / "results" / "sync_phase5_transition_compatibility.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"

SEED = 20260907
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
ORDER = ("C", "H", "O")


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


def measure_M(
    loaded,
    sc,
    *,
    active: dict[str, tuple[float, np.ndarray]] | None = None,
    alpha: float = ALPHA,
) -> dict[str, float]:
    """Full margin vector M at channel sites under current cumulative steer."""
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
    loaded,
    sc,
    *,
    j: str,
    e_j: float,
    v_j: np.ndarray,
    prior: dict[str, tuple[float, np.ndarray]],
    alpha: float,
) -> dict[str, Any]:
    """M before / after adding intervention j on top of prior; ΔM and T≈ΔM/α."""
    M0 = measure_M(loaded, sc, active=prior, alpha=alpha)
    after = dict(prior)
    after[j] = (float(e_j), v_j)
    M1 = measure_M(loaded, sc, active=after, alpha=alpha)
    dM = {k: float(M1[k] - M0[k]) for k in CHANNELS}
    T_col = {k: float(dM[k] / alpha) for k in CHANNELS}  # ∂M_k/∂α along signed e·v
    # intended sign: e_j * dM_j should be > 0 if actuator works in requested direction
    # Here e_j is the gain sign on v (positive e means +v increases M toward channel=1)
    return {
        "j": j,
        "e_j": float(e_j),
        "M0": M0,
        "M1": M1,
        "dM": dM,
        "T_col": T_col,
        "A_jj": float(dM[j] / alpha),
        "cross_abs_mean": float(np.mean([abs(dM[i]) for i in CHANNELS if i != j])),
        "diag_dominance": float(
            abs(dM[j]) / max(sum(abs(dM[i]) for i in CHANNELS if i != j), 1e-9)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=4)
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
    rng = np.random.default_rng(args.seed)

    # Use e_j = +1 for all for Jacobian sign convention (increase M_k)
    # Also run e_j = -1 suppress for C (persistent failure mode)
    signs = (+1.0, -1.0)

    print("=== sequential path C→H→O margin trajectory ===", flush=True)
    path_rows = []
    for r in range(args.reps):
        # suppress-oriented path (typical sync toward safer/lower channel bits)
        e_path = {"C": -1.0, "H": -1.0, "O": -1.0}
        active: dict[str, tuple[float, np.ndarray]] = {}
        M_traj = [measure_M(loaded, sc, active={}, alpha=args.alpha)]
        stage_deltas = []
        for j in ORDER:
            row = apply_and_delta(
                loaded, sc, j=j, e_j=e_path[j], v_j=vc[j], prior=active, alpha=args.alpha
            )
            active[j] = (e_path[j], vc[j])
            M_traj.append(row["M1"])
            stage_deltas.append(row)
            dM_str = ", ".join(f"{k}:{row['dM'][k]:+.2f}" for k in CHANNELS)
            print(
                f"  r={r} +{j}: dM={{{dM_str}}} DD={row['diag_dominance']:.2f}",
                flush=True,
            )
        path_rows.append({"rep": r, "e_path": e_path, "M_traj": M_traj, "stages": stage_deltas})

    print("=== special: C|h0 vs C|h_H ===", flush=True)
    c_contrast = []
    for r in range(args.reps):
        for ej in signs:
            c0 = apply_and_delta(
                loaded, sc, j="C", e_j=ej, v_j=vc["C"], prior={}, alpha=args.alpha
            )
            # first apply H, then C from h_H
            prior_H = {"H": (-1.0, vc["H"])}  # H suppress as typical upstream
            # also try H with same sign as ej for symmetry in one of the reps
            if r % 2 == 1:
                prior_H = {"H": (ej, vc["H"])}
            cH = apply_and_delta(
                loaded, sc, j="C", e_j=ej, v_j=vc["C"], prior=prior_H, alpha=args.alpha
            )
            c_contrast.append({"rep": r, "e_C": ej, "prior_H": prior_H["H"][0], "C_h0": c0, "C_hH": cH})
            print(
                f"  r={r} eC={ej:+.0f}: |dM_C| h0={abs(c0['dM']['C']):.2f} hH={abs(cH['dM']['C']):.2f} "
                f"cross_h0={c0['cross_abs_mean']:.2f} cross_hH={cH['cross_abs_mean']:.2f}",
                flush=True,
            )

    print("=== special: H|h0 vs H|h_C ===", flush=True)
    h_contrast = []
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
            h_contrast.append({"rep": r, "e_H": ej, "prior_C": prior_C["C"][0], "H_h0": h0, "H_hC": hC})
            print(
                f"  r={r} eH={ej:+.0f}: |dM_H| h0={abs(h0['dM']['H']):.2f} hC={abs(hC['dM']['H']):.2f} "
                f"cross_h0={h0['cross_abs_mean']:.2f} cross_hC={hC['cross_abs_mean']:.2f}",
                flush=True,
            )

    # Aggregate transition Jacobian from path stages (mean over reps, suppress path)
    # T_ij = mean dM_i when intervening on j (column j)
    T = {i: {j: [] for j in CHANNELS} for i in CHANNELS}
    DD = {j: [] for j in CHANNELS}
    for pr in path_rows:
        for st in pr["stages"]:
            j = st["j"]
            DD[j].append(st["diag_dominance"])
            for i in CHANNELS:
                T[i][j].append(st["dM"][i] / args.alpha)

    T_mean = {i: {j: float(np.mean(T[i][j])) for j in CHANNELS} for i in CHANNELS}
    DD_mean = {j: float(np.mean(DD[j])) for j in CHANNELS}

    def _mean_abs_dM(contrast_rows: list, key_from: str, key_to: str, channel: str) -> dict[str, float]:
        """Mean |dM| and cross for intervention from two starts."""
        a = [abs(r[key_from]["dM"][channel]) for r in contrast_rows]
        b = [abs(r[key_to]["dM"][channel]) for r in contrast_rows]
        ca = [r[key_from]["cross_abs_mean"] for r in contrast_rows]
        cb = [r[key_to]["cross_abs_mean"] for r in contrast_rows]
        # full dM vectors
        d_from = {k: float(np.mean([r[key_from]["dM"][k] for r in contrast_rows])) for k in CHANNELS}
        d_to = {k: float(np.mean([r[key_to]["dM"][k] for r in contrast_rows])) for k in CHANNELS}
        return {
            f"abs_dM_{channel}_from": float(np.mean(a)),
            f"abs_dM_{channel}_to": float(np.mean(b)),
            "retention": float(np.mean(b) / max(np.mean(a), 1e-9)),
            "cross_from": float(np.mean(ca)),
            "cross_to": float(np.mean(cb)),
            "cross_ratio": float(np.mean(cb) / max(np.mean(ca), 1e-9)),
            "dM_vec_from": d_from,
            "dM_vec_to": d_to,
        }

    c_sum = _mean_abs_dM(c_contrast, "C_h0", "C_hH", "C")
    h_sum = _mean_abs_dM(h_contrast, "H_h0", "H_hC", "H")

    # Classify
    off = [abs(T_mean[i][j]) for i in CHANNELS for j in CHANNELS if i != j]
    diag = [abs(T_mean[j][j]) for j in CHANNELS]
    mean_off, mean_diag = float(np.mean(off)), float(np.mean(diag))
    if mean_diag > 3 * mean_off:
        cls = "A_diagonal_transition"
    elif mean_off > 0.5 * mean_diag:
        cls = "B_strong_offdiag_coupling"
    else:
        cls = "mixed_coupling"

    # Cross increase after prior?
    cross_C_worsens = bool(c_sum["cross_ratio"] > 1.25)
    cross_H_worsens = bool(h_sum["cross_ratio"] > 1.25)

    gate = {
        "question": "What state transition does each successful intervention induce?",
        "T_class": cls,
        "mean_diag_abs_T": mean_diag,
        "mean_offdiag_abs_T": mean_off,
        "DD_per_channel": DD_mean,
        "C_after_H": {
            "retention_local": c_sum["retention"],
            "cross_ratio": c_sum["cross_ratio"],
            "cross_worsens": cross_C_worsens,
        },
        "H_after_C": {
            "retention_local": h_sum["retention"],
            "cross_ratio": h_sum["cross_ratio"],
            "cross_worsens": cross_H_worsens,
        },
        "implication": {
            "A_diagonal_transition": "look next at behavioral state machine / thresholds",
            "B_strong_offdiag_coupling": "pairwise geometric ≠ trajectory composition",
            "mixed_coupling": "characterize which pairs couple",
        }[cls if cls in ("A_diagonal_transition", "B_strong_offdiag_coupling") else "mixed_coupling"],
        "eight_way": "CLOSED",
        "do_not": "reopen direction search / opt / 8-way",
    }

    payload = {
        "protocol": "Phase 5C transition compatibility",
        "alpha": args.alpha,
        "reps": args.reps,
        "T_mean_dM_per_alpha": T_mean,
        "DD_mean": DD_mean,
        "path_rows": path_rows,
        "c_contrast_summary": c_sum,
        "h_contrast_summary": h_sum,
        "c_contrast": c_contrast,
        "h_contrast": h_contrast,
        "gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 5C — Transition compatibility",
        "",
        "> Failure ≠ wrong local direction (5A/5B).",
        "> **Question:** what state transition does each intervention induce?",
        "",
        f"Frozen \(v_c\). α={args.alpha}. Path C→H→O (suppress e=−1).",
        "",
        "## Transition Jacobian \(T_{ij}\\approx\\partial M_i/\\partial\\alpha_j\)",
        "",
        "| affected\\\\steer | C | H | O |",
        "|------------------|---|---|---|",
    ]
    for i in CHANNELS:
        cells = " | ".join(f"{T_mean[i][j]:+.2f}" for j in CHANNELS)
        lines.append(f"| **{i}** | {cells} |")

    lines += [
        "",
        f"| Channel | diag dominance \\|dM_j\\|/Σ\\_{{i≠j}}\\|dM_i\\| |",
        f"|---------|---------------------------------------------|",
    ]
    for j in CHANNELS:
        lines.append(f"| {j} | {DD_mean[j]:.2f} |")

    lines += [
        "",
        f"**T class:** `{cls}` (mean |diag|={mean_diag:.2f}, mean |off|={mean_off:.2f})",
        "",
        "## Special: C|h0 vs C|h_H",
        "",
        f"| | \\|ΔM_C\\| | mean cross \\|ΔM\\| |",
        f"|--|---------|-------------------|",
        f"| C from h0 | {c_sum['abs_dM_C_from']:.2f} | {c_sum['cross_from']:.2f} |",
        f"| C from h_H | {c_sum['abs_dM_C_to']:.2f} | {c_sum['cross_to']:.2f} |",
        f"| retention / cross ratio | {c_sum['retention']:.2f} | {c_sum['cross_ratio']:.2f} |",
        "",
        "ΔM vector C|h0: " + ", ".join(f"{k}={c_sum['dM_vec_from'][k]:+.2f}" for k in CHANNELS),
        "",
        "ΔM vector C|h_H: " + ", ".join(f"{k}={c_sum['dM_vec_to'][k]:+.2f}" for k in CHANNELS),
        "",
        "## Special: H|h0 vs H|h_C",
        "",
        f"| | \\|ΔM_H\\| | mean cross \\|ΔM\\| |",
        f"|--|---------|-------------------|",
        f"| H from h0 | {h_sum['abs_dM_H_from']:.2f} | {h_sum['cross_from']:.2f} |",
        f"| H from h_C | {h_sum['abs_dM_H_to']:.2f} | {h_sum['cross_to']:.2f} |",
        f"| retention / cross ratio | {h_sum['retention']:.2f} | {h_sum['cross_ratio']:.2f} |",
        "",
        "ΔM vector H|h0: " + ", ".join(f"{k}={h_sum['dM_vec_from'][k]:+.2f}" for k in CHANNELS),
        "",
        "ΔM vector H|h_C: " + ", ".join(f"{k}={h_sum['dM_vec_to'][k]:+.2f}" for k in CHANNELS),
        "",
        "## Gate",
        "",
        f"- Cross worsens for C after H: **{cross_C_worsens}**",
        f"- Cross worsens for H after C: **{cross_H_worsens}**",
        f"- Implication: {gate['implication']}",
        f"- 8-way: **CLOSED**",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "T_mean": T_mean}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
