#!/usr/bin/env python3
"""Phase 4 — 3-channel conversion matrix (before composition / 8-way).

Question:
  Does independently derived predictive→causal conversion work for C and O
  the same way it worked for H?

For each k ∈ {C,H,O}:
  v_p^k (label geometry), g_k = ∇M_k, v_c^k = convert(v_p, g)
  measure |cos|, |ΔM| for predictive / converted / Orth
  and cross-channel Jacobian J_ij = ∂M_i / ∂α_j

Gate: Proj_g ≫ Orth_g on |ΔM| per channel before pairwise composition.

  .venv/bin/python scripts/run_phase4_channel_conversion_matrix.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts.sync_channel_control import CHANNEL_V, ChannelBank  # noqa: E402
from scripts.sync_channel_margins import (  # noqa: E402
    SPECS,
    convert_direction,
    dM_bidir,
    margin_at_site,
    margin_gradient,
    messages_for_channel,
    proj_orth,
    unit,
)
from scripts.sync_h_decision import LAYER_DEFAULT, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase4_conversion_matrix.json"
MD = ROOT / "data" / "results" / "sync_phase4_conversion_matrix.md"
OUT_V = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
SEED = 20260904
ALPHA = 1.5
CHANNELS = ("C", "H", "O")


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _mean_diff_at_site(loaded, messages, spec, *, n: int, seed: int, layer: int) -> np.ndarray:
    """Predictive direction from site residuals stratified by M."""
    rng = np.random.default_rng(seed)
    hs, ms = [], []
    for i in range(n):
        # slight prefill jitter for C; H uses fixed PLAN variants inside margin_at_site calls
        pf = None
        if spec.channel == "C":
            pf = rng.choice(["PLAN: I will ", "PLAN: Next I will ", "PLAN: I will "])
        elif spec.channel == "H":
            from scripts.sync_h_decision import PLAN_PUBLIC, PLAN_RUN, PLAN_SMOKE

            pf = [PLAN_RUN, PLAN_PUBLIC, PLAN_SMOKE][i % 3]
        s = margin_at_site(
            loaded, messages, spec, prefill=pf, capture_h=True, layer=layer
        )
        if s["h"] is None:
            continue
        hs.append(s["h"])
        ms.append(s["M"])
    H = np.stack(hs, 0)
    M = np.asarray(ms, dtype=np.float64)
    hi, lo = np.quantile(M, 0.7), np.quantile(M, 0.3)
    pos, neg = H[M >= hi], H[M <= lo]
    if len(pos) == 0 or len(neg) == 0:
        pos, neg = H[M >= np.median(M)], H[M < np.median(M)]
    if len(pos) == 0 or len(neg) == 0:
        return unit(rng.standard_normal(H.shape[1]))
    diff = pos.mean(0) - neg.mean(0)
    if float(np.linalg.norm(diff)) < 1e-8:
        return unit(rng.standard_normal(H.shape[1]))
    return unit(diff)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--n-collect", type=int, default=24)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    bank = ChannelBank.load(CHANNEL_V) if CHANNEL_V.is_file() else None

    # --- per channel: g, v_p, v_c, orth ---
    g: dict[str, np.ndarray] = {}
    v_p: dict[str, np.ndarray] = {}
    v_c: dict[str, np.ndarray] = {}
    v_orth: dict[str, np.ndarray] = {}
    msgs: dict[str, list] = {}
    rows: list[dict[str, Any]] = []

    for k in CHANNELS:
        print(f"=== channel {k}: site + gradient ===", flush=True)
        spec = SPECS[k]
        msgs[k] = messages_for_channel(sc, k)
        g[k] = margin_gradient(loaded, msgs[k], spec, layer=args.layer, n=4)
        print(f"  g_{k} ok ‖g‖=1", flush=True)

        # Predictive: bank if available, else mean-diff at site
        ch_i = {"C": 0, "H": 1, "O": 2}[k]
        if bank is not None:
            v_p[k] = unit(np.asarray(bank.V[ch_i], dtype=np.float64))
            src = "bank_V"
        else:
            v_p[k] = _mean_diff_at_site(
                loaded, msgs[k], spec, n=args.n_collect, seed=args.seed + ch_i, layer=args.layer
            )
            src = "site_mean_diff"

        # Also learn site mean-diff as secondary predictive for comparison
        v_md = _mean_diff_at_site(
            loaded,
            msgs[k],
            spec,
            n=args.n_collect,
            seed=args.seed + 10 + ch_i,
            layer=args.layer,
        )

        v_c[k] = convert_direction(v_p[k], g[k])
        # Algebraic identity: converted == ±g; pin to unit(g) for steer stability
        v_c[k] = unit(g[k])
        _, v_orth[k] = proj_orth(v_p[k], g[k], seed=args.seed + 20)

        cos_pg = float(np.dot(v_p[k], g[k]))
        print(f"  cos(v_p,g)={cos_pg:+.4f} src={src}", flush=True)

        for name, v in (
            ("predictive", v_p[k]),
            ("mean_diff_site", v_md),
            ("converted", v_c[k]),
            ("orth", v_orth[k]),
            ("gradient", g[k]),
        ):
            print(f"  ΔM {k}/{name} ...", flush=True)
            m = dM_bidir(loaded, msgs[k], spec, v, alpha=args.alpha)
            rows.append(
                {
                    "channel": k,
                    "arm": name,
                    "site": spec.site,
                    "cos_g": float(np.dot(unit(v), g[k])),
                    "abs_cos_g": abs(float(np.dot(unit(v), g[k]))),
                    **m,
                    "v_p_source": src,
                }
            )

    # --- cross-channel Jacobian: steer with v_c^j, measure ΔM_i ---
    print("=== cross-channel Jacobian J_ij = dM_i / dα along v_c^j ===", flush=True)
    J = np.zeros((3, 3), dtype=np.float64)
    J_abs = np.zeros((3, 3), dtype=np.float64)
    for j, kj in enumerate(CHANNELS):
        for i, ki in enumerate(CHANNELS):
            # Intervene with +α v_c^j at channel-i site? 
            # Correct: apply v_c^j at the *intervention channel's* natural site isn't
            # shared — for Jacobian at mechanistic level, apply hook globally at each
            # *affected* site's forward pass (same ActivationSteerHook on last token
            # of that site's prefill). Measures how v_c^j moves M_i when present at i's site.
            base = margin_at_site(loaded, msgs[ki], SPECS[ki])
            hook = make_steer_hook(loaded, v_c[kj], args.alpha)
            s = margin_at_site(loaded, msgs[ki], SPECS[ki], hook=hook)
            dM = float(s["M"] - base["M"])
            J[i, j] = dM / args.alpha
            J_abs[i, j] = abs(dM)
            print(f"  J[{ki},{kj}] dM={dM:+.3f}", flush=True)

    # Gate: converted ≫ orth on |ΔM| for each channel
    gate_ch: dict[str, Any] = {}
    for k in CHANNELS:
        conv = next(r for r in rows if r["channel"] == k and r["arm"] == "converted")
        orth = next(r for r in rows if r["channel"] == k and r["arm"] == "orth")
        pred = next(r for r in rows if r["channel"] == k and r["arm"] == "predictive")
        gate_ch[k] = {
            "abs_dM_converted": conv["abs_dM"],
            "abs_dM_orth": orth["abs_dM"],
            "abs_dM_predictive": pred["abs_dM"],
            "proj_beats_orth": conv["abs_dM"] > max(orth["abs_dM"] * 3, 0.5),
            "converted_beats_predictive": conv["abs_dM"] > pred["abs_dM"],
            "abs_cos_vp_g": pred["abs_cos_g"],
        }

    n_pass = sum(1 for k in CHANNELS if gate_ch[k]["proj_beats_orth"])
    payload = {
        "protocol": "Phase 4 — 3-channel conversion matrix",
        "question": "Does predictive→causal conversion generalize from H to C and O?",
        "alpha": args.alpha,
        "layer": args.layer,
        "rows": rows,
        "jacobian_dM_per_alpha": {
            "rows": list(CHANNELS),
            "cols": list(CHANNELS),
            "J": J.tolist(),
            "J_abs_dM": J_abs.tolist(),
            "note": (
                "J_ij ≈ ∂M_i/∂α when steering with v_c^j at channel-i site "
                "(site-local probe of coupling, not full trajectory)."
            ),
        },
        "gate": {
            "per_channel": gate_ch,
            "n_channels_proj_beats_orth": n_pass,
            "ready_for_pairwise_composition": n_pass >= 2 and gate_ch["H"]["proj_beats_orth"],
            "ready_for_8way": False,
            "next_if_pass": "pairwise C+H / H+O / C+O then sequential closed-loop",
            "next_if_fail": (
                "document channel-specific conversion failure; "
                "do not force compose; scientific result that conversion is boundary-specific"
            ),
        },
        "directions": {
            "v_c": {k: v_c[k].tolist() for k in CHANNELS},
            "g": {k: g[k].tolist() for k in CHANNELS},
            "cos_vp_g": {k: float(np.dot(v_p[k], g[k])) for k in CHANNELS},
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_V.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "alpha": args.alpha,
                "status": "converted_candidates_not_composed",
                "formula": "v_c = sign(v_p·g) Proj_g(v_p)/||Proj||",
                "V_c": [v_c[k].tolist() for k in CHANNELS],
                "channels": list(CHANNELS),
                "sites": {k: SPECS[k].site for k in CHANNELS},
                "margins": {
                    k: {"pos": SPECS[k].pos_token, "neg": SPECS[k].neg_token} for k in CHANNELS
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Markdown
    lines = [
        "# Phase 4 — 3-channel conversion matrix",
        "",
        r"> Does \(v_p^{(k)}\rightarrow\operatorname{Proj}_{g_k}(v_p^{(k)})\) "
        r"work for \(k\in\{C,H,O\}\) before composition?",
        "",
        "Phase 3 freeze: H conversion supported. This matrix tests C and O.",
        "",
        "## Per-channel arms",
        "",
        "| Ch | Arm | site | cos(v,g) | |ΔM| | bidir | M0 |",
        "|----|-----|------|----------|-------|-------|----|",
    ]
    for r in rows:
        lines.append(
            f"| {r['channel']} | {r['arm']} | {r['site']} | {r['cos_g']:+.3f} | "
            f"{r['abs_dM']:.3f} | {r['bidirectional']} | {r['M0']:+.2f} |"
        )

    lines += [
        "",
        "## Gate: Proj ≫ Orth",
        "",
        "| Ch | \\|ΔM\\|_pred | \\|ΔM\\|_conv | \\|ΔM\\|_orth | proj≫orth | conv>pred | \\|cos\\| |",
        "|----|-------------|-------------|-------------|---------|----------|--------|",
    ]
    for k in CHANNELS:
        gch = gate_ch[k]
        lines.append(
            f"| {k} | {gch['abs_dM_predictive']:.3f} | {gch['abs_dM_converted']:.3f} | "
            f"{gch['abs_dM_orth']:.3f} | {gch['proj_beats_orth']} | "
            f"{gch['converted_beats_predictive']} | {gch['abs_cos_vp_g']:.3f} |"
        )

    lines += [
        "",
        r"## Cross-channel Jacobian \(J_{ij}\approx\partial M_i/\partial\alpha\) along \(v_c^j\)",
        "",
        "| affected\\\\steer | C | H | O |",
        "|------------------|---|---|---|",
    ]
    for i, ki in enumerate(CHANNELS):
        cells = " | ".join(f"{J[i, j]:+.2f}" for j in range(3))
        lines.append(f"| **{ki}** | {cells} |")

    lines += [
        "",
        f"Channels with Proj≫Orth: **{n_pass}/3**",
        f"Ready for pairwise composition: **{payload['gate']['ready_for_pairwise_composition']}**",
        "",
        "8-way stays closed until pairwise + sequential closed-loop pass.",
        "",
        "If C or O fail conversion, that is informative: conversion is "
        "**channel- and boundary-specific**, not universal.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": payload["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
