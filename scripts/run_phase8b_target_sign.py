#!/usr/bin/env python3
"""Phase 8B — Target-signed live-boundary control (no new v).

Phase 8A: L_live ↔ ΔE, but G_H<0 and P(W→C)_H=0.11.
Hypothesis: actuator sign/timing wrong relative to m*_k.

Local Phase-7 fact: ΔB(+α)>0, ΔB(−α)<0 for frozen v_c.
Therefore target-conditioned direction:

  d_k(m*_k) = s_k v_c^k ,  s_k = 2 m*_k − 1

Single-channel tests (H primary):
  m*=0 → −α v_c    vs    m*=1 → +α v_c
  measure G_k = s_k (B_after − B_before) and P(W→C).

Also wrong-sign ablation: d = −s v_c.

  .venv/bin/python scripts/run_phase8b_target_sign.py --n 16
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
from scripts.sync_eq import extract_plan, score_plan  # noqa: E402
from scripts.sync_h_decision import decision_at_site, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8b_target_sign.json"
MD = ROOT / "data" / "results" / "sync_phase8b_target_sign.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

SEED = 20260914
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


def _load_boundaries() -> dict[str, dict[str, Any]]:
    blob = json.loads(P7_PATH.read_text())
    out = {}
    for k in CHANNELS:
        b = blob["boundaries"][k]
        if not b.get("ok") or not b.get("w"):
            raise SystemExit(f"need Phase-7 boundary for {k}")
        out[k] = {"w": np.asarray(b["w"], dtype=np.float64), "b": float(b["b"])}
    return out


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


def _base_messages(sc, *, cls: str, task_id: str) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def _S_from(row: dict, messages: list[dict]) -> list[int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    asst = [str(m.get("content") or "") for m in messages if m.get("role") == "assistant"]
    plan = extract_plan((row.get("final") or "") + "\n" + "\n\n".join(asst))
    c = score_plan(plan, H) if plan else None
    C = int(c) if c is not None else 0
    return [C, H, O]


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


def capture_live_hB(
    loaded,
    sc,
    *,
    state: dict,
    boundaries: dict,
    cls: str,
    task_id: str,
) -> dict[str, Any]:
    base = _base_messages(sc, cls=cls, task_id=task_id)
    msgs = state["messages_snapshot"]
    asst = [str(m.get("content") or "") for m in msgs if m.get("role") == "assistant"]
    plan = extract_plan("\n\n".join(asst)) or ""
    out: dict[str, Any] = {"plan": plan}

    sC = margin_at_site(loaded, base, SPECS["C"], capture_h=True)
    hC = np.asarray(sC["h"], dtype=np.float64)
    out["C"] = {
        "h": hC,
        "B": B_score(hC, boundaries["C"]["w"], boundaries["C"]["b"]),
        "M": float(sC["M"]),
        "ok": True,
    }

    if plan.strip():
        pf = plan if plan.endswith("\n") else plan + "\n"
        sH = decision_at_site(loaded, base, assistant_prefill=pf, capture_h=True)
        hH = np.asarray(sH["h"], dtype=np.float64)
        out["H"] = {
            "h": hH,
            "B": B_score(hH, boundaries["H"]["w"], boundaries["H"]["b"]),
            "M": float(sH["M_H"]),
            "ok": True,
            "prefill": pf,
        }
    else:
        out["H"] = {"ok": False}

    sO = margin_at_site(loaded, msgs, SPECS["O"], capture_h=True)
    hO = np.asarray(sO["h"], dtype=np.float64)
    out["O"] = {
        "h": hO,
        "B": B_score(hO, boundaries["O"]["w"], boundaries["O"]["b"]),
        "M": float(sO["M"]),
        "ok": True,
    }
    out["base"] = base
    out["msgs"] = msgs
    return out


def run_free(sc, loaded, *, cls: str, task_id: str, seed: int):
    state = sc.run_tools_phase(
        loaded, cls=cls, task_id=task_id, seed=seed, with_plan_format=True
    )
    row = sc.finalize_episode(loaded, state)
    S = _S_from(row, state["messages_snapshot"])
    return S, state, row


def run_single_channel_steer(
    sc,
    loaded,
    *,
    cls: str,
    task_id: str,
    seed: int,
    channel: str,
    signed_alpha: float,
    v: np.ndarray,
):
    """Steer only one channel during the matching phase."""

    def hook_for_turn(phase: str, _t: int):
        if channel in ("C", "H") and phase == "tool":
            return make_steer_hook(loaded, v, float(signed_alpha))
        if channel == "O" and phase == "report":
            return make_steer_hook(loaded, v, float(signed_alpha))
        return None

    state = sc.run_tools_phase(
        loaded,
        cls=cls,
        task_id=task_id,
        seed=seed,
        hook_for_turn=hook_for_turn,
        with_plan_format=True,
    )
    row = sc.finalize_episode(loaded, state, hook_for_turn=hook_for_turn)
    S = _S_from(row, state["messages_snapshot"])
    return S, state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16, help="free-run episodes")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--channels",
        default="H",
        help="H primary (default). Pass H,C,O for all channels (slower).",
    )
    args = ap.parse_args()
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    vc = _load_vc()
    boundaries = _load_boundaries()
    schedule = _stratified_schedule(sc, args.n, args.seed)

    free_rows: list[dict[str, Any]] = []
    print(f"=== free runs n={args.n} ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        seed = args.seed + 17 * i
        torch.manual_seed(seed)
        S, state, _ = run_free(sc, loaded, cls=cls, task_id=task_id, seed=seed)
        live = capture_live_hB(
            loaded, sc, state=state, boundaries=boundaries, cls=cls, task_id=task_id
        )
        free_rows.append(
            {
                "i": i,
                "cls": cls,
                "task_id": task_id,
                "seed": seed,
                "S": S,
                "live": live,
            }
        )
        print(f"  [{i}] S={S} B_H={live['H'].get('B', float('nan')):+.2f}", flush=True)

    # --- activation-space target-signed G (cheap, all free h) ---
    act_G: dict[str, Any] = {}
    print("=== activation-space G (target-signed vs wrong-sign) ===", flush=True)
    for k in channels:
        rows_ts, rows_ws = [], []
        for fr in free_rows:
            site = fr["live"].get(k)
            if not site or not site.get("ok"):
                continue
            h = site["h"]
            B0 = site["B"]
            w, b = boundaries[k]["w"], boundaries[k]["b"]
            for mstar_bit in (0, 1):
                s = 2 * mstar_bit - 1
                # target-signed: h' = h + s α v
                B_ts = B_score(h + s * args.alpha * vc[k], w, b)
                G_ts = s * (B_ts - B0)
                # wrong-sign
                B_ws = B_score(h - s * args.alpha * vc[k], w, b)
                G_ws = s * (B_ws - B0)
                wrong0 = int(fr["S"][CH_IDX[k]] != mstar_bit)
                rows_ts.append({"mstar": mstar_bit, "G": G_ts, "wrong0": wrong0, "S0": fr["S"][CH_IDX[k]]})
                rows_ws.append({"mstar": mstar_bit, "G": G_ws, "wrong0": wrong0, "S0": fr["S"][CH_IDX[k]]})
        act_G[k] = {
            "target_sign": {
                "mean_G": float(np.mean([r["G"] for r in rows_ts])),
                "frac_G_pos": float(np.mean([r["G"] > 0 for r in rows_ts])),
                "mean_G_wrong0": float(np.mean([r["G"] for r in rows_ts if r["wrong0"]]))
                if any(r["wrong0"] for r in rows_ts)
                else float("nan"),
            },
            "wrong_sign": {
                "mean_G": float(np.mean([r["G"] for r in rows_ws])),
                "frac_G_pos": float(np.mean([r["G"] > 0 for r in rows_ws])),
            },
        }
        print(
            f"  {k} act: TS mean_G={act_G[k]['target_sign']['mean_G']:+.3f} "
            f"frac>0={act_G[k]['target_sign']['frac_G_pos']:.2f} | "
            f"WS mean_G={act_G[k]['wrong_sign']['mean_G']:+.3f}",
            flush=True,
        )

    # --- behavioral single-channel (H primary: both m*; C/O: both m* but fewer) ---
    beh: dict[str, list[dict]] = {k: [] for k in channels}
    print("=== behavioral target-signed vs wrong-sign ===", flush=True)

    for k in channels:
        # all free runs × both targets for H; for C/O same but OK
        for fr in free_rows:
            if not fr["live"].get(k, {}).get("ok"):
                continue
            cls, task_id, seed0 = fr["cls"], fr["task_id"], fr["seed"]
            S0 = fr["S"]
            B0 = fr["live"][k]["B"]
            for mstar_bit in (0, 1):
                s = 2 * mstar_bit - 1
                for mode, sign_mult in (("target_sign", +1.0), ("wrong_sign", -1.0)):
                    signed_alpha = sign_mult * s * args.alpha
                    seed = seed0 + 200 + 10 * mstar_bit + (0 if mode == "target_sign" else 1)
                    torch.manual_seed(seed)
                    S1, state1 = run_single_channel_steer(
                        sc,
                        loaded,
                        cls=cls,
                        task_id=task_id,
                        seed=seed,
                        channel=k,
                        signed_alpha=signed_alpha,
                        v=vc[k],
                    )
                    live1 = capture_live_hB(
                        loaded,
                        sc,
                        state=state1,
                        boundaries=boundaries,
                        cls=cls,
                        task_id=task_id,
                    )
                    if not live1.get(k, {}).get("ok"):
                        B1 = float("nan")
                        G = float("nan")
                    else:
                        B1 = live1[k]["B"]
                        G = float(s * (B1 - B0))
                    wrong0 = S0[CH_IDX[k]] != mstar_bit
                    correct1 = S1[CH_IDX[k]] == mstar_bit
                    w2c = int(wrong0 and correct1)
                    c2w = int((not wrong0) and (not correct1))
                    beh[k].append(
                        {
                            "mode": mode,
                            "mstar": mstar_bit,
                            "s": s,
                            "signed_alpha": signed_alpha,
                            "S0": S0[CH_IDX[k]],
                            "S1": S1[CH_IDX[k]],
                            "B0": B0,
                            "B1": B1,
                            "G": G,
                            "wrong0": int(wrong0),
                            "W2C": w2c,
                            "C2W": c2w,
                            "seed0": seed0,
                        }
                    )
            # progress per free-run channel
            last = [r for r in beh[k] if r["seed0"] == seed0 and r["mode"] == "target_sign"]
            if last:
                print(
                    f"  {k} seed0={seed0} S0={S0[CH_IDX[k]]} "
                    + " ".join(
                        f"m*={r['mstar']}:G={r['G']:+.2f},S1={r['S1']},W2C={r['W2C']}" for r in last
                    ),
                    flush=True,
                )

    def summarize_beh(rows: list[dict], mode: str) -> dict[str, Any]:
        sub = [r for r in rows if r["mode"] == mode]
        subw = [r for r in sub if r["wrong0"]]
        gs = [r["G"] for r in sub if not np.isnan(r["G"])]
        gsw = [r["G"] for r in subw if not np.isnan(r["G"])]
        return {
            "n": len(sub),
            "n_wrong0": len(subw),
            "mean_G": float(np.mean(gs)) if gs else float("nan"),
            "frac_G_pos": float(np.mean([g > 0 for g in gs])) if gs else float("nan"),
            "mean_G_wrong0": float(np.mean(gsw)) if gsw else float("nan"),
            "P_W2C": float(np.mean([r["W2C"] for r in subw])) if subw else float("nan"),
            "P_C2W": float(
                np.mean([r["C2W"] for r in sub if not r["wrong0"]])
            )
            if any(not r["wrong0"] for r in sub)
            else float("nan"),
            "by_mstar": {
                str(m): {
                    "mean_G": float(np.mean([r["G"] for r in sub if r["mstar"] == m and not np.isnan(r["G"])]))
                    if any(r["mstar"] == m and not np.isnan(r["G"]) for r in sub)
                    else float("nan"),
                    "P_W2C": float(
                        np.mean([r["W2C"] for r in sub if r["mstar"] == m and r["wrong0"]])
                    )
                    if any(r["mstar"] == m and r["wrong0"] for r in sub)
                    else float("nan"),
                    "n_wrong0": sum(1 for r in sub if r["mstar"] == m and r["wrong0"]),
                }
                for m in (0, 1)
            },
        }

    beh_sum = {
        k: {
            "target_sign": summarize_beh(beh[k], "target_sign"),
            "wrong_sign": summarize_beh(beh[k], "wrong_sign"),
        }
        for k in channels
    }

    # gates — H primary
    h_ts = beh_sum.get("H", {}).get("target_sign", {})
    h_ws = beh_sum.get("H", {}).get("wrong_sign", {})
    gate = {
        "hypothesis": "target-signed s_k v_c fixes G_H and lifts P(W→C)_H",
        "H_act_TS_G_pos": bool(act_G.get("H", {}).get("target_sign", {}).get("mean_G", 0) > 0),
        "H_beh_TS_mean_G_pos": bool((h_ts.get("mean_G") or 0) > 0),
        "H_beh_TS_beats_WS_G": bool((h_ts.get("mean_G") or -1e9) > (h_ws.get("mean_G") or 1e9)),
        "H_P_W2C_TS": h_ts.get("P_W2C"),
        "H_P_W2C_WS": h_ws.get("P_W2C"),
        "H_P_W2C_improved_vs_0p11": bool((h_ts.get("P_W2C") or 0) > 0.11 + 0.05),
        "H_sign_matters_for_W2C": bool(
            (h_ts.get("P_W2C") or 0) > (h_ws.get("P_W2C") or 1) + 0.05
        ),
    }
    gate["controller_bug_is_sign"] = bool(
        gate["H_beh_TS_mean_G_pos"]
        and gate["H_beh_TS_beats_WS_G"]
        and gate["H_P_W2C_improved_vs_0p11"]
    )
    gate["read"] = (
        "If target-sign lifts P(W→C)_H ≫ 0.11 and G_H>0 vs wrong-sign: sign was the bug. "
        "If not: investigate H timing/capture, not representation."
    )

    # serialize free rows without huge h
    free_ser = []
    for fr in free_rows:
        free_ser.append(
            {
                "i": fr["i"],
                "cls": fr["cls"],
                "task_id": fr["task_id"],
                "seed": fr["seed"],
                "S": fr["S"],
                "B0": {k: fr["live"][k].get("B") for k in CHANNELS if fr["live"].get(k)},
            }
        )

    payload = {
        "protocol": "Phase 8B target-signed live-boundary",
        "alpha": args.alpha,
        "n": args.n,
        "formula": "d_k = s_k v_c^k , G_k = s_k (B_after - B_before)",
        "activation_G": act_G,
        "behavioral": beh_sum,
        "behavioral_rows": beh,
        "free_runs": free_ser,
        "gate": gate,
        "phase8a_ref": {"G_H": -0.50, "P_W2C_H": 0.11},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8B — Target-signed live-boundary control",
        "",
        r"> $d_k(m_k^*)=s_k v_c^k$, $s_k=2m_k^*-1$. No new $v$.",
        "",
        f"α={args.alpha}, n={args.n} free runs. H primary.",
        "",
        "## Activation-space $G$ (local $h\\pm s\\alpha v$)",
        "",
        "| Channel | TS mean $G$ | TS frac $G{>}0$ | WS mean $G$ |",
        "|---------|-------------|----------------|-------------|",
    ]
    for k in channels:
        ts, ws = act_G[k]["target_sign"], act_G[k]["wrong_sign"]
        lines.append(
            f"| {k} | {ts['mean_G']:+.3f} | {ts['frac_G_pos']:.2f} | {ws['mean_G']:+.3f} |"
        )

    lines += [
        "",
        "## Behavioral single-channel (episode re-run)",
        "",
        "| Channel | mode | mean $G$ | frac $G{>}0$ | P(W→C) | n_wrong |",
        "|---------|------|----------|--------------|--------|---------|",
    ]
    for k in channels:
        for mode in ("target_sign", "wrong_sign"):
            s = beh_sum[k][mode]
            lines.append(
                f"| {k} | {mode} | {s['mean_G']:+.3f} | {s['frac_G_pos']:.2f} | "
                f"{s['P_W2C']:.2f} | {s['n_wrong0']} |"
            )

    lines += [
        "",
        "### H by $m^*$ (target-sign)",
        "",
        "| $H^*$ | mean $G$ | P(W→C) | n_wrong |",
        "|-------|----------|--------|---------|",
    ]
    for m in ("0", "1"):
        b = beh_sum["H"]["target_sign"]["by_mstar"][m]
        lines.append(
            f"| {m} | {b['mean_G']:+.3f} | {b['P_W2C']:.2f} | {b['n_wrong0']} |"
        )

    lines += [
        "",
        "## Gate (H primary)",
        "",
        f"- Act TS mean $G_H>0$: **{gate['H_act_TS_G_pos']}**",
        f"- Beh TS mean $G_H>0$: **{gate['H_beh_TS_mean_G_pos']}**",
        f"- Beh TS beats wrong-sign on $G$: **{gate['H_beh_TS_beats_WS_G']}**",
        f"- P(W→C)_H target-sign: **{gate['H_P_W2C_TS']}** (Phase 8A was 0.11)",
        f"- P(W→C)_H wrong-sign: **{gate['H_P_W2C_WS']}**",
        f"- P(W→C) improved vs 0.11: **{gate['H_P_W2C_improved_vs_0p11']}**",
        f"- Sign matters for flips: **{gate['H_sign_matters_for_W2C']}**",
        f"- Controller bug is sign: **{gate['controller_bug_is_sign']}**",
        "",
        gate["read"],
        "",
        "No new $v$. No full 8-way yet.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate, "beh_H": beh_sum.get("H")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
