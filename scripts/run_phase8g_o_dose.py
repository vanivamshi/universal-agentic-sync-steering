#!/usr/bin/env python3
"""Phase 8G — Frozen v_c^O dose sweep with disclose ground truth.

Diagnosis: under-actuation + proxy/behavior mismatch (not timing).
No new v.

For α ∈ {1.5, 3, 5, 8} at the live post-tool O site, same seed:

  E[G_O]                         # target-sign analytical / hooked
  P(S_proxy− ≠ S_proxy+)         # During/docs auxiliary
  P(S_O− ≠ S_O+)                 # score_output_disclose ground truth
  P(W→C)_O                       # target-sign → disclose bit

Hierarchy:
  B_O → proxy movement   (auxiliary)
  B_O → S_O              (behavioral endpoint)

  .venv/bin/python scripts/run_phase8g_o_dose.py --n 12
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

from scripts.sync_channel_margins import SPECS, margin_at_site, unit  # noqa: E402
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8g_o_dose.json"
MD = ROOT / "data" / "results" / "sync_phase8g_o_dose.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

SEED = 20260915
ALPHAS = (1.5, 3.0, 5.0, 8.0)


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc_O() -> np.ndarray:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return unit(Vc[2])


def _load_boundary_O() -> tuple[np.ndarray, float]:
    b = json.loads(P7_PATH.read_text())["boundaries"]["O"]
    if not b.get("ok") or not b.get("w"):
        raise SystemExit("need Phase-7 O boundary")
    return np.asarray(b["w"], dtype=np.float64), float(b["b"])


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


def _stratified_schedule(sc, n: int, seed: int) -> list[tuple[str, str]]:
    plan = sc.iter_episode_plan()
    priv = [(c, tid) for c, tid, _f, sens in plan if sens != "public"]
    pub = [(c, tid) for c, tid, _f, sens in plan if sens == "public"]
    rng = np.random.default_rng(seed)
    rng.shuffle(priv)
    rng.shuffle(pub)
    out: list[tuple[str, str]] = []
    i_p = i_u = 0
    for i in range(n):
        if i % 2 == 0 and priv:
            out.append(priv[i_p % len(priv)])
            i_p += 1
        elif pub:
            out.append(pub[i_u % len(pub)])
            i_u += 1
        elif priv:
            out.append(priv[i_p % len(priv)])
            i_p += 1
    return out


def _proxy_bit(loaded, messages, *, signed_alpha: float, v: np.ndarray) -> dict[str, Any]:
    hook = make_steer_hook(loaded, v, float(signed_alpha)) if abs(signed_alpha) > 1e-12 else None
    try:
        s = margin_at_site(
            loaded, messages, SPECS["O"], prefill="FINAL: ", hook=hook, capture_h=True
        )
    finally:
        if hook is not None:
            hook.remove()
    return {
        "S_proxy": int(s["M"] > 0),
        "M": float(s["M"]),
        "h": np.asarray(s["h"], dtype=np.float64) if s.get("h") is not None else None,
    }


def _disclose_bit(
    sc,
    loaded,
    messages,
    *,
    signed_alpha: float,
    v: np.ndarray,
    seed: int,
    sensitive_paths: list[str],
) -> dict[str, Any]:
    """Generate FINAL continuation; ground-truth S_O = score_output_disclose."""
    from activation_pipeline.agent.loop import generate_assistant

    prefill = "FINAL: "
    hook = make_steer_hook(loaded, v, float(signed_alpha)) if abs(signed_alpha) > 1e-12 else None
    torch.manual_seed(seed)
    if hook is not None:
        hook.register()
    try:
        asst = generate_assistant(
            loaded,
            messages,
            max_new_tokens=sc.MAX_NEW_TOKENS,
            temperature=sc.TEMPERATURE,
            assistant_prefill=prefill,
        )
    finally:
        if hook is not None:
            hook.remove()
    S_O = int(sc.score_output_disclose(asst, sensitive_paths))
    cont = asst[len(prefill) :] if asst.startswith(prefill) else asst
    return {"S_O": S_O, "cont_head": cont[:100], "final": asst[:400]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--alphas",
        default="1.5,3,5,8",
        help="comma-separated α doses",
    )
    args = ap.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    vO = _load_vc_O()
    wO, bO = _load_boundary_O()
    schedule = _stratified_schedule(sc, args.n, args.seed)

    free_rows: list[dict[str, Any]] = []
    print(f"=== free runs n={args.n} (freeze live O site) ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        seed = args.seed + 17 * i
        torch.manual_seed(seed)
        state = sc.run_tools_phase(
            loaded, cls=cls, task_id=task_id, seed=seed, with_plan_format=True
        )
        row = sc.finalize_episode(loaded, state)
        msgs = state["messages_snapshot"]
        sens = list(state.get("sensitive_paths") or row.get("sensitive_paths") or [])
        S0 = int(row.get("s_output") or 0)
        # baseline site
        base = _proxy_bit(loaded, msgs, signed_alpha=0.0, v=vO)
        h0 = base["h"]
        assert h0 is not None
        B0 = B_score(h0, wO, bO)
        free_rows.append(
            {
                "i": i,
                "cls": cls,
                "task_id": task_id,
                "seed": seed,
                "S0": S0,
                "msgs": msgs,
                "sens": sens,
                "h0": h0,
                "B0": B0,
                "S_proxy0": base["S_proxy"],
                "M0": base["M"],
            }
        )
        print(
            f"  [{i}] S_O={S0} B0={B0:+.2f} Sprox0={base['S_proxy']} sens={len(sens)}",
            flush=True,
        )

    by_alpha: dict[str, Any] = {}
    print(f"=== dose sweep α={alphas} ===", flush=True)

    for alpha in alphas:
        rows_g: list[dict] = []
        flip_proxy: list[int] = []
        flip_SO: list[int] = []
        dB_abs: list[float] = []
        w2c_ts: list[dict] = []

        for fr in free_rows:
            h0, B0, msgs, seed, S0 = fr["h0"], fr["B0"], fr["msgs"], fr["seed"], fr["S0"]
            sens = fr["sens"]

            # geometry under ±α
            dB_p = B_score(h0 + alpha * vO, wO, bO) - B0
            dB_m = B_score(h0 - alpha * vO, wO, bO) - B0
            dB_abs.append(abs(dB_p))

            # proxy ±α (sign-agnostic flip)
            prox_p = _proxy_bit(loaded, msgs, signed_alpha=+alpha, v=vO)
            prox_m = _proxy_bit(loaded, msgs, signed_alpha=-alpha, v=vO)
            flip_proxy.append(int(prox_p["S_proxy"] != prox_m["S_proxy"]))

            # disclose ±α (ground truth flip) — early-FINAL intervention
            # Also covers target-sign / wrong-sign for both m*:
            #   m*=1 → TS=+α, WS=−α ;  m*=0 → TS=−α, WS=+α
            disc_p = _disclose_bit(
                sc, loaded, msgs, signed_alpha=+alpha, v=vO, seed=seed + 500, sensitive_paths=sens
            )
            disc_m = _disclose_bit(
                sc, loaded, msgs, signed_alpha=-alpha, v=vO, seed=seed + 501, sensitive_paths=sens
            )
            flip_SO.append(int(disc_p["S_O"] != disc_m["S_O"]))

            for mstar in (0, 1):
                s = 2 * mstar - 1
                B1 = B_score(h0 + s * alpha * vO, wO, bO)
                G = float(s * (B1 - B0))
                S_ts = disc_p["S_O"] if s > 0 else disc_m["S_O"]
                S_ws = disc_m["S_O"] if s > 0 else disc_p["S_O"]
                wrong0 = int(S0 != mstar)
                rows_g.append({"mstar": mstar, "G": G, "wrong0": wrong0})
                w2c_ts.append(
                    {
                        "mstar": mstar,
                        "wrong0": wrong0,
                        "W2C_TS": int(wrong0 and S_ts == mstar),
                        "W2C_WS": int(wrong0 and S_ws == mstar),
                        "S_TS": S_ts,
                        "S_WS": S_ws,
                        "S0": S0,
                    }
                )

        wrong_ts = [r for r in w2c_ts if r["wrong0"]]
        summary = {
            "alpha": alpha,
            "mean_abs_delta_B": float(np.mean(dB_abs)),
            "mean_G_TS": float(np.mean([r["G"] for r in rows_g])),
            "frac_G_pos": float(np.mean([r["G"] > 0 for r in rows_g])),
            "P_proxy_flip": float(np.mean(flip_proxy)),
            "P_SO_flip": float(np.mean(flip_SO)),
            "P_W2C_TS": float(np.mean([r["W2C_TS"] for r in wrong_ts])) if wrong_ts else float("nan"),
            "P_W2C_WS": float(np.mean([r["W2C_WS"] for r in wrong_ts])) if wrong_ts else float("nan"),
            "n_wrong0": len(wrong_ts),
            "n": len(free_rows),
        }
        by_alpha[str(alpha)] = summary
        print(
            f"  α={alpha:g}: |ΔB|={summary['mean_abs_delta_B']:.3f} "
            f"E[G]={summary['mean_G_TS']:+.3f} "
            f"P(proxy flip)={summary['P_proxy_flip']:.2f} "
            f"P(S_O flip)={summary['P_SO_flip']:.2f} "
            f"P(W→C)_TS={summary['P_W2C_TS']:.2f} WS={summary['P_W2C_WS']:.2f}",
            flush=True,
        )

    # --- gates ---
    alphas_s = [str(a) for a in alphas]
    abs_B = [by_alpha[a]["mean_abs_delta_B"] for a in alphas_s]
    p_proxy = [by_alpha[a]["P_proxy_flip"] for a in alphas_s]
    p_so = [by_alpha[a]["P_SO_flip"] for a in alphas_s]
    g_pos = all(by_alpha[a]["mean_G_TS"] > 0 for a in alphas_s)

    # monotone-ish increase: last > first
    gate = {
        "hypothesis": "O under-actuation: higher α raises |ΔB| and S_O flips; E[G]>0",
        "E_G_pos_all_alpha": g_pos,
        "abs_dB_increases": bool(abs_B[-1] > abs_B[0] + 0.05),
        "proxy_flip_increases": bool(p_proxy[-1] > p_proxy[0] + 0.05),
        "SO_flip_increases": bool(p_so[-1] > p_so[0] + 0.05),
        "SO_flip_at_max_alpha": p_so[-1],
        "proxy_flip_at_max_alpha": p_proxy[-1],
        "P_W2C_TS_at_max": by_alpha[alphas_s[-1]]["P_W2C_TS"],
        "P_W2C_WS_at_max": by_alpha[alphas_s[-1]]["P_W2C_WS"],
        "under_actuation": False,
        "proxy_behavior_mismatch": False,
        "gain_helps_behavior": False,
    }
    # under-actuation: |ΔB|↑ and S_O flip↑ with G>0
    gate["under_actuation"] = bool(
        gate["E_G_pos_all_alpha"]
        and gate["abs_dB_increases"]
        and gate["SO_flip_increases"]
    )
    # mismatch: |ΔB|↑ (and maybe proxy↑) but S_O flip stays flat
    gate["proxy_behavior_mismatch"] = bool(
        gate["abs_dB_increases"]
        and (not gate["SO_flip_increases"])
        and (p_so[-1] < 0.15)
    )
    gate["gain_helps_behavior"] = bool(
        gate["SO_flip_increases"]
        or (
            by_alpha[alphas_s[-1]]["P_W2C_TS"]
            > by_alpha[alphas_s[0]]["P_W2C_TS"] + 0.05
        )
    )
    gate["read"] = (
        "If |ΔB|↑ and P(S_O flip)↑ with E[G]>0: under-actuation → use higher α_O. "
        "If |ΔB|↑ but S_O flat: proxy≠behavior → align scoring / early-FINAL, not more gain. "
        "Only then reopen O conversion."
    )

    payload = {
        "protocol": "Phase 8G frozen v_c^O dose + disclose GT",
        "alphas": list(alphas),
        "n": args.n,
        "seed": args.seed,
        "ground_truth": "score_output_disclose",
        "proxy": "logit(During)-logit(docs)",
        "by_alpha": by_alpha,
        "gate": gate,
        "unchanged": ["v_c^O", "Phase7 O boundary"],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8G — Frozen $v_c^O$ dose sweep",
        "",
        "> Ground truth: `score_output_disclose`. Proxy: During/docs (auxiliary). No new $v$.",
        "",
        f"n={args.n}, seed={args.seed}, α∈{list(alphas)}.",
        "",
        "## Dose table",
        "",
        "| α | $E[\\|\\Delta B\\|]$ | $E[G_O]$ | P(proxy flip) | P($S_O$ flip) | P(W→C)$_{TS}$ | P(W→C)$_{WS}$ |",
        "|--:|------------------:|---------:|--------------:|--------------:|--------------:|--------------:|",
    ]
    for a in alphas_s:
        s = by_alpha[a]
        lines.append(
            f"| {s['alpha']:g} | {s['mean_abs_delta_B']:.3f} | {s['mean_G_TS']:+.3f} | "
            f"{s['P_proxy_flip']:.2f} | {s['P_SO_flip']:.2f} | "
            f"{s['P_W2C_TS']:.2f} | {s['P_W2C_WS']:.2f} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- $E[G_O]>0$ all α: **{gate['E_G_pos_all_alpha']}**",
        f"- $|\\Delta B|$ increases with α: **{gate['abs_dB_increases']}**",
        f"- Proxy flip increases: **{gate['proxy_flip_increases']}**",
        f"- **$S_O$ flip increases (disclose GT):** **{gate['SO_flip_increases']}**",
        f"- Under-actuation (gain → behavior): **{gate['under_actuation']}**",
        f"- Proxy/behavior mismatch: **{gate['proxy_behavior_mismatch']}**",
        f"- Gain helps W→C or $S_O$: **{gate['gain_helps_behavior']}**",
        "",
        gate["read"],
        "",
        "Next: if under-actuation → raise α_O in controller; "
        "if mismatch → disclose-aligned early-FINAL; "
        "only then re-convert O.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "by_alpha": by_alpha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
