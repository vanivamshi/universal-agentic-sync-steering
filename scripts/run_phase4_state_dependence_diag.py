#!/usr/bin/env python3
"""Phase 4 diagnostic — state dependence of g_k after prior interventions.

Frozen conclusion: pairwise compositionality ≠ sequential behavioral sync.
Question: does local decision geometry change after preceding steers?

  For each target channel k and prior intervention j ≠ k:
    measure M_k(h0), g_k(h0)
    apply α v_j^c (converted) at j's site context / as activation steer
    measure M_k(h1), g_k(h1)
    report ΔM_k, cos(g_k(h0), g_k(h1)), ‖g_k(h1)−g_k(h0)‖

If g_k drifts, static v_c^k is insufficient → state-adaptive Proj_{g_k(h_t)}.

No new discovery, no 8-way.

  .venv/bin/python scripts/run_phase4_state_dependence_diag.py
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

from scripts.sync_h_decision import LAYER_DEFAULT, chat_prompt, make_steer_hook  # noqa: E402
from scripts.sync_channel_margins import (  # noqa: E402
    SPECS,
    margin_at_site,
    margin_gradient,
    messages_for_channel,
    token_id,
    unit,
)

OUT = ROOT / "data" / "results" / "sync_phase4_state_dependence.json"
MD = ROOT / "data" / "results" / "sync_phase4_state_dependence.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
SEED = 20260905
ALPHA = 1.5
CHANNELS = ("C", "H", "O")


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    blob = json.loads(VC_PATH.read_text())
    Vc = np.asarray(blob["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=ALPHA)
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
    vc = _load_vc()
    msgs = {k: messages_for_channel(sc, k) for k in CHANNELS}
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    # Baseline g_k(h0), M_k(h0)
    g0: dict[str, np.ndarray] = {}
    M0: dict[str, float] = {}
    print("=== baseline g_k(h0) ===", flush=True)
    for k in CHANNELS:
        g0[k] = margin_gradient(loaded, msgs[k], SPECS[k], layer=args.layer, n=4)
        M0[k] = float(margin_at_site(loaded, msgs[k], SPECS[k])["M"])
        print(f"  {k}: M0={M0[k]:+.3f}", flush=True)

    print("=== after prior intervention j, remeasure k ===", flush=True)
    for j in CHANNELS:
        for sign in (+1.0, -1.0):
            # Steer with v_j^c while evaluating each target channel k's site
            for k in CHANNELS:
                hook = make_steer_hook(loaded, vc[j], sign * args.alpha)
                Mk = float(margin_at_site(loaded, msgs[k], SPECS[k], hook=hook)["M"])
                # Gradient under steered forward (state after j-perturbation at k's site)
                # Approximate g_k(h1) by backprop with the same steer hooked during forward.
                g1 = _grad_with_hook(
                    loaded, msgs[k], SPECS[k], hook_dir=vc[j], hook_alpha=sign * args.alpha, layer=args.layer
                )
                cos = float(np.dot(g0[k], g1))
                dM = Mk - M0[k]
                drift = float(np.linalg.norm(g1 - g0[k]))
                row = {
                    "prior_j": j,
                    "sign_j": int(sign),
                    "target_k": k,
                    "self": j == k,
                    "M0": M0[k],
                    "M1": Mk,
                    "dM": dM,
                    "cos_g0_g1": cos,
                    "g_drift_l2": drift,
                    "abs_dM": abs(dM),
                }
                rows.append(row)
                if j != k:
                    print(
                        f"  j={j}({sign:+.0f})→k={k}: dM={dM:+.3f} cos(g0,g1)={cos:+.3f} ‖Δg‖={drift:.3f}",
                        flush=True,
                    )

    # Summaries: cross-effects (j≠k) vs self
    cross = [r for r in rows if not r["self"]]
    self_rows = [r for r in rows if r["self"]]

    def _mean(rs: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")

    # Per target k: how much g_k drifts when ANY other channel is steered
    by_k: dict[str, Any] = {}
    for k in CHANNELS:
        sub = [r for r in cross if r["target_k"] == k]
        by_k[k] = {
            "mean_cos_g_after_other": _mean(sub, "cos_g0_g1"),
            "mean_g_drift_l2": _mean(sub, "g_drift_l2"),
            "mean_abs_dM": _mean(sub, "abs_dM"),
            "min_cos": float(np.min([r["cos_g0_g1"] for r in sub])) if sub else float("nan"),
        }

    # Does H's geometry stay stable while C/O drift? (matches sequential story)
    gate = {
        "question": "Do g_k / M_k change after preceding interventions?",
        "mean_cross_cos": _mean(cross, "cos_g0_g1"),
        "mean_cross_g_drift": _mean(cross, "g_drift_l2"),
        "mean_self_cos": _mean(self_rows, "cos_g0_g1"),
        "geometry_drifts_cross": bool(_mean(cross, "cos_g0_g1") < 0.95 or _mean(cross, "g_drift_l2") > 0.2),
        "by_target": by_k,
        "implication_if_drift": (
            "static v_c^k insufficient; next controller is state-adaptive "
            "v_c^k(h_t) ∝ Proj_{g_k(h_t)}(v_p^k)"
        ),
        "eight_way": "CLOSED",
        "phase4_frozen": True,
    }

    payload = {
        "protocol": "Phase 4 state-dependence diagnostic",
        "alpha": args.alpha,
        "layer": args.layer,
        "M0": M0,
        "rows": rows,
        "gate": gate,
        "phase4_conclusion": (
            "pairwise compositionality ≠ sequential behavioral synchronization; "
            "only H is a reliable sequential behavioral actuator."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 4 diagnostic — state dependence of \(g_k\)",
        "",
        "> Pairwise compositionality ≠ sequential behavioral sync (frozen).",
        "> Does \(g_k(h_t)\) change after a prior intervention on channel \(j\)?",
        "",
        f"α={args.alpha}. Cross = j≠k.",
        "",
        f"- mean cos\\(g_k(h_0),g_k(h_1)\\) cross: **{gate['mean_cross_cos']:.3f}**",
        f"- mean ‖Δg‖₂ cross: **{gate['mean_cross_g_drift']:.3f}**",
        f"- mean cos self (j=k): **{gate['mean_self_cos']:.3f}**",
        f"- geometry drifts under cross-steer: **{gate['geometry_drifts_cross']}**",
        "",
        "## Per target channel (after other-channel steer)",
        "",
        "| k | mean cos(g0,g1) | min cos | mean ‖Δg‖ | mean \\|ΔM\\| |",
        "|---|-----------------|---------|-----------|-------------|",
    ]
    for k in CHANNELS:
        b = by_k[k]
        lines.append(
            f"| {k} | {b['mean_cos_g_after_other']:.3f} | {b['min_cos']:.3f} | "
            f"{b['mean_g_drift_l2']:.3f} | {b['mean_abs_dM']:.3f} |"
        )

    lines += [
        "",
        "## Cross pairs (detail)",
        "",
        "| prior j | sign | target k | ΔM | cos(g0,g1) | ‖Δg‖ |",
        "|---------|------|----------|----|------------|------|",
    ]
    for r in sorted(cross, key=lambda x: (x["target_k"], x["prior_j"], -x["sign_j"])):
        lines.append(
            f"| {r['prior_j']} | {r['sign_j']:+d} | {r['target_k']} | "
            f"{r['dM']:+.3f} | {r['cos_g0_g1']:+.3f} | {r['g_drift_l2']:.3f} |"
        )

    lines += [
        "",
        "## Implication",
        "",
        "If cross cos ≪ 1 or ‖Δg‖ large: static \(v_c^{(k)}\) is not trajectory-valid;",
        r"next step is state-adaptive \(v_c^{(k)}(h_t)\propto\operatorname{Proj}_{g_k(h_t)}(v_p^{(k)})\).",
        "",
        "**8-way CLOSED.** Phase 4 evidence ends at pairwise ✓ / sequential ✗.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2))
    return 0


def _grad_with_hook(loaded, messages, spec, *, hook_dir: np.ndarray, hook_alpha: float, layer: int) -> np.ndarray:
    """∇M_k at site while a prior-channel steer is active (approximate g_k(h1))."""
    from activation_pipeline.hooks import resolve_decoder_layers

    tok = loaded.tokenizer
    id_pos = token_id(tok, spec.pos_token)
    id_neg = token_id(tok, spec.neg_token)
    enc = tok(chat_prompt(tok, messages, spec.prefill), return_tensors="pt")
    device = next(loaded.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    steer = make_steer_hook(loaded, hook_dir, hook_alpha, layer=layer)
    dec = resolve_decoder_layers(loaded.model)
    captured: dict[str, torch.Tensor] = {}

    def make_hook():
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h_last = hidden[:, -1, :].detach().requires_grad_(True)
            captured["h"] = h_last
            h_new = hidden.clone()
            h_new[:, -1, :] = h_last
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        return hook

    handle = dec[layer].register_forward_hook(make_hook())
    if steer is not None:
        steer.register()
    try:
        loaded.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = loaded.model(**enc, use_cache=False)
            M = out.logits[0, -1][id_pos] - out.logits[0, -1][id_neg]
            M.backward()
        g = captured["h"].grad
        if g is None:
            raise RuntimeError("no grad")
        return unit(g.detach().float().cpu().numpy().reshape(-1))
    finally:
        handle.remove()
        if steer is not None:
            steer.remove()


if __name__ == "__main__":
    raise SystemExit(main())
