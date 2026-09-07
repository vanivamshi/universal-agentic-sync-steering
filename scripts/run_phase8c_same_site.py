#!/usr/bin/env python3
"""Phase 8C — Same-site H counterfactual (measurement / control-path correction).

Phase 8B: act-space G_H correct (+0.41), but full-episode re-run G inverted.
Root cause under test: counterfactual activation ≠ full-episode intervention
measurement.

Stage A (default) — frozen H decision state, same seed:
  B0 = wᵀ h0 + b
  B1 = wᵀ (h0 + s α v_c) + b     (analytical; expect |ΔB|≈0.41)
  G  = s (B1 − B0)
  H decision from the *same* forward at that site (hook on), not a later replay.

Stage B (--stage B|both) — one variable change:
  same seed + same plan prefill + decision-token-only hook
  (teacher-force PLAN, steer only post-PLAN generation).

Do not change v_c, α, target-sign logic, or H boundary model.
Do not run 8-way.

  .venv/bin/python scripts/run_phase8c_same_site.py --n 16
  .venv/bin/python scripts/run_phase8c_same_site.py --n 16 --stage both
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

from scripts.sync_eq import extract_plan  # noqa: E402
from scripts.sync_h_decision import decision_at_site, make_steer_hook, unit  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8c_same_site.json"
MD = ROOT / "data" / "results" / "sync_phase8c_same_site.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

SEED = 20260914  # match Phase 8B for comparability
ALPHA = 1.5
PHASE7_ACT_G = 0.412  # Phase 8B / 7 act-space |E[G]| reference


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc_H() -> np.ndarray:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return unit(Vc[1])  # H


def _load_boundary_H() -> tuple[np.ndarray, float]:
    b = json.loads(P7_PATH.read_text())["boundaries"]["H"]
    if not b.get("ok") or not b.get("w"):
        raise SystemExit("need Phase-7 H boundary")
    return np.asarray(b["w"], dtype=np.float64), float(b["b"])


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


def _base_messages(sc, *, cls: str, task_id: str) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def _S_H(row: dict, messages: list[dict]) -> int:
    return int(row.get("s_tool") or 0)


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


def _site_forward(loaded, base, prefill: str, *, signed_alpha: float, v: np.ndarray, capture_h: bool):
    hook = make_steer_hook(loaded, v, float(signed_alpha)) if abs(signed_alpha) > 1e-12 else None
    try:
        return decision_at_site(
            loaded,
            base,
            assistant_prefill=prefill,
            hook=hook,
            capture_h=capture_h,
        )
    finally:
        if hook is not None:
            hook.remove()


def _summarize_condition(rows: list[dict], *, g_key: str = "G") -> dict[str, Any]:
    gs = [r[g_key] for r in rows if not np.isnan(r.get(g_key, float("nan")))]
    wrong = [r for r in rows if r["wrong0"]]
    s_bits = [
        r["S_proxy"] if "S_proxy" in r else r["S1"]
        for r in rows
        if ("S_proxy" in r) or ("S1" in r)
    ]
    return {
        "n": len(rows),
        "n_wrong0": len(wrong),
        "mean_G": float(np.mean(gs)) if gs else float("nan"),
        "frac_G_pos": float(np.mean([g > 0 for g in gs])) if gs else float("nan"),
        "P_W2C": float(np.mean([r["W2C"] for r in wrong])) if wrong else float("nan"),
        "P_C2W": float(np.mean([r["C2W"] for r in rows if not r["wrong0"]]))
        if any(not r["wrong0"] for r in rows)
        else float("nan"),
        "mean_S_proxy": float(np.mean(s_bits)) if s_bits else float("nan"),
        "by_mstar": {
            str(m): {
                "mean_G": float(
                    np.mean([r[g_key] for r in rows if r["mstar"] == m and not np.isnan(r.get(g_key, float("nan")))])
                )
                if any(r["mstar"] == m for r in rows)
                else float("nan"),
                "P_W2C": float(
                    np.mean([r["W2C"] for r in rows if r["mstar"] == m and r["wrong0"]])
                )
                if any(r["mstar"] == m and r["wrong0"] for r in rows)
                else float("nan"),
                "n_wrong0": sum(1 for r in rows if r["mstar"] == m and r["wrong0"]),
            }
            for m in (0, 1)
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--stage",
        choices=("A", "B", "both"),
        default="A",
        help="A=same-site counterfactual (default); B=decision-token-only gen; both",
    )
    args = ap.parse_args()

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    vH = _load_vc_H()
    wH, bH = _load_boundary_H()
    schedule = _stratified_schedule(sc, args.n, args.seed)

    # --- free runs: freeze H site (plan prefill + S_H) ---
    free_rows: list[dict[str, Any]] = []
    print(f"=== free runs n={args.n} (freeze H site) ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        seed = args.seed + 17 * i  # match 8B free-run seeding
        torch.manual_seed(seed)
        state = sc.run_tools_phase(
            loaded, cls=cls, task_id=task_id, seed=seed, with_plan_format=True
        )
        row = sc.finalize_episode(loaded, state)
        msgs = state["messages_snapshot"]
        asst = [str(m.get("content") or "") for m in msgs if m.get("role") == "assistant"]
        plan = extract_plan("\n\n".join(asst)) or ""
        if not plan.strip():
            print(f"  [{i}] skip (no plan)", flush=True)
            continue
        prefill = plan if plan.endswith("\n") else plan + "\n"
        base = _base_messages(sc, cls=cls, task_id=task_id)
        S_H = _S_H(row, msgs)
        free_rows.append(
            {
                "i": i,
                "cls": cls,
                "task_id": task_id,
                "seed": seed,
                "S_H": S_H,
                "prefill": prefill,
                "base": base,
            }
        )
        print(f"  [{i}] S_H={S_H} plan={prefill[:48]!r}...", flush=True)

    # --- Stage A: same-site counterfactual ---
    stage_a_rows: dict[str, list[dict]] = {
        "no_steer": [],
        "target_sign": [],
        "wrong_sign": [],
    }
    delta_plus: list[float] = []
    delta_minus: list[float] = []

    print("=== Stage A: same-site counterfactual ===", flush=True)
    for fr in free_rows:
        base, prefill, S0 = fr["base"], fr["prefill"], fr["S_H"]
        # unsteered site capture (shared B0 / h0)
        s0 = _site_forward(loaded, base, prefill, signed_alpha=0.0, v=vH, capture_h=True)
        h0 = np.asarray(s0["h"], dtype=np.float64)
        B0 = B_score(h0, wH, bH)
        S_proxy0 = int(s0["M_H"] > 0)

        # Phase-7-style raw ±α on frozen h (sign-agnostic geometry check)
        delta_plus.append(B_score(h0 + args.alpha * vH, wH, bH) - B0)
        delta_minus.append(B_score(h0 - args.alpha * vH, wH, bH) - B0)

        for mstar in (0, 1):
            s = 2 * mstar - 1
            wrong0 = int(S0 != mstar)

            # analytical target / wrong G on frozen h0
            B_ts = B_score(h0 + s * args.alpha * vH, wH, bH)
            B_ws = B_score(h0 - s * args.alpha * vH, wH, bH)
            G_ts = float(s * (B_ts - B0))
            G_ws = float(s * (B_ws - B0))

            # decisions from same-site forwards
            conds = {
                "no_steer": (0.0, 0.0, S_proxy0),  # G=0; reuse unsteered decision
                "target_sign": (float(s * args.alpha), G_ts, None),
                "wrong_sign": (float(-s * args.alpha), G_ws, None),
            }
            for name, (signed_a, G_anal, S_reuse) in conds.items():
                if S_reuse is not None:
                    S_proxy = S_reuse
                    B_hook = B0
                    G_hook = 0.0
                    M = float(s0["M_H"])
                else:
                    sh = _site_forward(
                        loaded, base, prefill, signed_alpha=signed_a, v=vH, capture_h=True
                    )
                    S_proxy = int(sh["M_H"] > 0)
                    M = float(sh["M_H"])
                    h1 = np.asarray(sh["h"], dtype=np.float64) if sh.get("h") is not None else None
                    if h1 is not None:
                        B_hook = B_score(h1, wH, bH)
                        G_hook = float(s * (B_hook - B0))
                    else:
                        B_hook = float("nan")
                        G_hook = float("nan")

                correct1 = S_proxy == mstar
                stage_a_rows[name].append(
                    {
                        "i": fr["i"],
                        "seed": fr["seed"],
                        "mstar": mstar,
                        "s": s,
                        "S0": S0,
                        "S_proxy": S_proxy,
                        "B0": B0,
                        "G": G_anal,  # analytical acceptance G
                        "G_hook": G_hook,  # hooked residual G (diagnostic)
                        "B_hook": B_hook,
                        "M": M,
                        "wrong0": wrong0,
                        "W2C": int(wrong0 and correct1),
                        "C2W": int((not wrong0) and (not correct1)),
                    }
                )

        print(
            f"  seed={fr['seed']} S0={S0} B0={B0:+.2f} "
            f"ΔB(+α)={delta_plus[-1]:+.3f} ΔB(−α)={delta_minus[-1]:+.3f} "
            f"Sprox0={S_proxy0}",
            flush=True,
        )

    sum_a = {k: _summarize_condition(v) for k, v in stage_a_rows.items()}
    # also summarize hooked G for TS/WS
    sum_a_hook = {
        k: _summarize_condition(v, g_key="G_hook")
        for k, v in stage_a_rows.items()
        if k != "no_steer"
    }

    mean_dp = float(np.mean(delta_plus)) if delta_plus else float("nan")
    mean_dm = float(np.mean(delta_minus)) if delta_minus else float("nan")

    gate_a = {
        "hypothesis": "same-site counterfactual recovers E[G_H]>0 under target-sign",
        "mean_delta_B_plus": mean_dp,
        "mean_delta_B_minus": mean_dm,
        "delta_B_matches_phase7": bool(
            abs(mean_dp - PHASE7_ACT_G) < 0.15 and abs(mean_dm + PHASE7_ACT_G) < 0.15
        ),
        "TS_mean_G_pos": bool(sum_a["target_sign"]["mean_G"] > 0),
        "WS_mean_G_neg": bool(sum_a["wrong_sign"]["mean_G"] < 0),
        "TS_beats_WS_on_G": bool(
            sum_a["target_sign"]["mean_G"] > sum_a["wrong_sign"]["mean_G"]
        ),
        "TS_P_W2C": sum_a["target_sign"]["P_W2C"],
        "WS_P_W2C": sum_a["wrong_sign"]["P_W2C"],
        "no_steer_P_W2C": sum_a["no_steer"]["P_W2C"],
        "TS_improves_W2C_vs_WS": bool(
            (sum_a["target_sign"]["P_W2C"] or 0) > (sum_a["wrong_sign"]["P_W2C"] or 0) + 0.05
        ),
        "TS_improves_W2C_vs_none": bool(
            (sum_a["target_sign"]["P_W2C"] or 0) > (sum_a["no_steer"]["P_W2C"] or 0) + 0.05
        ),
    }
    gate_a["stage_A_pass"] = bool(
        gate_a["TS_mean_G_pos"]
        and gate_a["WS_mean_G_neg"]
        and gate_a["TS_beats_WS_on_G"]
    )
    gate_a["read"] = (
        "If Stage A passes: measurement path is fixed; proceed to decision-token-only (Stage B). "
        "If Stage A fails: H boundary / v_c geometry broken (should not happen). "
        "If A passes but B fails: token timing within H generation."
    )

    print(
        f"  ΔB(+α)={mean_dp:+.3f} ΔB(−α)={mean_dm:+.3f} "
        f"(Phase7 ref ±{PHASE7_ACT_G})",
        flush=True,
    )
    for name in ("no_steer", "wrong_sign", "target_sign"):
        s = sum_a[name]
        print(
            f"  A {name}: mean_G={s['mean_G']:+.3f} P(W→C)={s['P_W2C']:.3f} "
            f"n_wrong={s['n_wrong0']}",
            flush=True,
        )
    print(f"  Stage A pass: {gate_a['stage_A_pass']}", flush=True)

    # --- Stage B: decision-token-only generation ---
    sum_b: dict[str, Any] | None = None
    gate_b: dict[str, Any] | None = None
    stage_b_rows: dict[str, list[dict]] | None = None

    if args.stage in ("B", "both"):
        from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls

        stage_b_rows = {"no_steer": [], "target_sign": [], "wrong_sign": []}
        print("=== Stage B: decision-token-only (PLAN prefill + steered continuation) ===", flush=True)

        for fr in free_rows:
            base, prefill, S0, seed = fr["base"], fr["prefill"], fr["S_H"], fr["seed"]
            # B0 from Stage-A geometry on same site (recompute cheap)
            s0 = _site_forward(loaded, base, prefill, signed_alpha=0.0, v=vH, capture_h=True)
            h0 = np.asarray(s0["h"], dtype=np.float64)
            B0 = B_score(h0, wH, bH)

            for mstar in (0, 1):
                s = 2 * mstar - 1
                wrong0 = int(S0 != mstar)
                B_ts = B_score(h0 + s * args.alpha * vH, wH, bH)
                B_ws = B_score(h0 - s * args.alpha * vH, wH, bH)
                G_map = {
                    "no_steer": 0.0,
                    "target_sign": float(s * (B_ts - B0)),
                    "wrong_sign": float(s * (B_ws - B0)),
                }
                alpha_map = {
                    "no_steer": 0.0,
                    "target_sign": float(s * args.alpha),
                    "wrong_sign": float(-s * args.alpha),
                }
                for name, signed_a in alpha_map.items():
                    hook = (
                        make_steer_hook(loaded, vH, signed_a) if abs(signed_a) > 1e-12 else None
                    )
                    # same seed for all conditions at this (episode, m*) — isolate sign
                    torch.manual_seed(seed + 9000 + 10 * mstar)
                    if hook is not None:
                        hook.register()
                    try:
                        asst = generate_assistant(
                            loaded,
                            base,
                            max_new_tokens=sc.MAX_NEW_TOKENS,
                            temperature=sc.TEMPERATURE,
                            assistant_prefill=prefill,
                        )
                    finally:
                        if hook is not None:
                            hook.remove()
                    # returned string is prefill + continuation
                    cont = asst[len(prefill) :] if asst.startswith(prefill) else asst
                    has_tool = bool(parse_tool_calls(asst)) or "<tool_call>" in cont
                    has_final = cont.strip().startswith("FINAL") or "\nFINAL" in cont
                    if has_tool and not has_final:
                        S1 = 1
                    elif has_final and not has_tool:
                        S1 = 0
                    else:
                        # fallback: tool preferred if tool_call present
                        S1 = int(has_tool)
                    correct1 = S1 == mstar
                    stage_b_rows[name].append(
                        {
                            "i": fr["i"],
                            "seed": seed,
                            "mstar": mstar,
                            "s": s,
                            "S0": S0,
                            "S1": S1,
                            "G": G_map[name],
                            "wrong0": wrong0,
                            "W2C": int(wrong0 and correct1),
                            "C2W": int((not wrong0) and (not correct1)),
                            "cont_head": cont[:80],
                        }
                    )

            print(f"  seed={seed} S0={S0} Stage-B done", flush=True)

        sum_b = {k: _summarize_condition(v) for k, v in stage_b_rows.items()}
        gate_b = {
            "TS_mean_G_pos": bool(sum_b["target_sign"]["mean_G"] > 0),
            "WS_mean_G_neg": bool(sum_b["wrong_sign"]["mean_G"] < 0),
            "TS_P_W2C": sum_b["target_sign"]["P_W2C"],
            "WS_P_W2C": sum_b["wrong_sign"]["P_W2C"],
            "no_steer_P_W2C": sum_b["no_steer"]["P_W2C"],
            "TS_improves_W2C_vs_WS": bool(
                (sum_b["target_sign"]["P_W2C"] or 0)
                > (sum_b["wrong_sign"]["P_W2C"] or 0) + 0.05
            ),
            "TS_improves_W2C_vs_none": bool(
                (sum_b["target_sign"]["P_W2C"] or 0)
                > (sum_b["no_steer"]["P_W2C"] or 0) + 0.05
            ),
        }
        gate_b["stage_B_pass"] = bool(
            gate_b["TS_mean_G_pos"]
            and gate_b["TS_improves_W2C_vs_WS"]
            and gate_b["TS_improves_W2C_vs_none"]
        )
        for name in ("no_steer", "wrong_sign", "target_sign"):
            s = sum_b[name]
            print(
                f"  B {name}: mean_G={s['mean_G']:+.3f} P(W→C)={s['P_W2C']:.3f}",
                flush=True,
            )
        print(f"  Stage B pass: {gate_b['stage_B_pass']}", flush=True)

    payload = {
        "protocol": "Phase 8C same-site H counterfactual",
        "alpha": args.alpha,
        "n": args.n,
        "seed": args.seed,
        "stage": args.stage,
        "unchanged": ["v_c", "alpha", "target_sign s=2m*-1", "H boundary from Phase 7"],
        "phase7_act_G_ref": PHASE7_ACT_G,
        "delta_B": {"mean_plus": mean_dp, "mean_minus": mean_dm},
        "stage_A": {
            "summary": sum_a,
            "summary_hook_G": sum_a_hook,
            "gate": gate_a,
        },
        "stage_B": {"summary": sum_b, "gate": gate_b} if sum_b is not None else None,
        "rows_A": {k: v for k, v in stage_a_rows.items()},
        "rows_B": stage_b_rows,
    }
    # drop huge fields if any — rows are small
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8C — Same-site H counterfactual",
        "",
        "> Counterfactual activation test ≠ full-episode intervention. "
        "No change to $v_c$, α, target-sign, or H boundary.",
        "",
        f"α={args.alpha}, n={args.n}, seed={args.seed}, stage={args.stage}.",
        "",
        "## Geometry check (frozen $h_0$)",
        "",
        f"| $\\Delta B(+\\alpha)$ | $\\Delta B(-\\alpha)$ | Phase 7 / 8B ref |",
        f"|---------------------:|--------------------:|-----------------:|",
        f"| {mean_dp:+.3f} | {mean_dm:+.3f} | ±{PHASE7_ACT_G} |",
        "",
        "## Stage A — same-site decision (acceptance table)",
        "",
        "| condition | $E[G_H]$ (analytical) | $P(W\\to C)$ | n_wrong |",
        "|-----------|----------------------:|------------:|--------:|",
    ]
    for name in ("no_steer", "wrong_sign", "target_sign"):
        s = sum_a[name]
        lines.append(
            f"| {name} | {s['mean_G']:+.3f} | {s['P_W2C']:.3f} | {s['n_wrong0']} |"
        )
    lines += [
        "",
        f"- Stage A pass: **{gate_a['stage_A_pass']}**",
        f"- ΔB matches Phase 7: **{gate_a['delta_B_matches_phase7']}**",
        f"- TS improves W→C vs wrong-sign: **{gate_a['TS_improves_W2C_vs_WS']}**",
        f"- TS improves W→C vs no-steer: **{gate_a['TS_improves_W2C_vs_none']}**",
        "",
        "Hooked residual $G$ (diagnostic, not acceptance):",
        "",
        "| condition | $E[G_{hook}]$ |",
        "|-----------|--------------:|",
    ]
    for name in ("wrong_sign", "target_sign"):
        s = sum_a_hook[name]
        lines.append(f"| {name} | {s['mean_G']:+.3f} |")

    if sum_b is not None and gate_b is not None:
        lines += [
            "",
            "## Stage B — decision-token-only generation",
            "",
            "| condition | $E[G_H]$ | $P(W\\to C)$ | n_wrong |",
            "|-----------|---------:|------------:|--------:|",
        ]
        for name in ("no_steer", "wrong_sign", "target_sign"):
            s = sum_b[name]
            lines.append(
                f"| {name} | {s['mean_G']:+.3f} | {s['P_W2C']:.3f} | {s['n_wrong0']} |"
            )
        lines += [
            "",
            f"- Stage B pass: **{gate_b['stage_B_pass']}**",
            "",
        ]

    lines += [
        "",
        gate_a["read"],
        "",
        "8-way still paused.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "md": str(MD),
                "gate_A": gate_a,
                "gate_B": gate_b,
                "sum_A": sum_a,
                "sum_B": sum_b,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
