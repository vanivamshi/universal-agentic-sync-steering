#!/usr/bin/env python3
"""Phase 8F — Same-site causal control for O and live-decision C (vectors frozen).

After 8E: ΔE↑ but P(hit) flat; P(O)=0.44 weakest; G_C≡0 under fixed prompt.
H already passed 8C. This phase runs the *same* local test for O and a genuine
episode-conditioned C site.

Stage A (default): frozen live site, same seed
  B0 = wᵀ h0 + b
  B1 = wᵀ (h0 + s α v_c) + b
  G  = s (B1 − B0)
  S_proxy from same forward (hook on)

Stage B (--stage B|both): decision-token-only generation
  C: teacher-force episode mid-PLAN stem → steer continuation
  O: teacher-force FINAL: on post-tool msgs → steer continuation

  .venv/bin/python scripts/run_phase8f_co_same_site.py --n 12 --stage both
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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
from scripts.sync_h_decision import make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase8f_co_same_site.json"
MD = ROOT / "data" / "results" / "sync_phase8f_co_same_site.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

SEED = 20260915
ALPHA = 1.5
# Phase 7 act-space |ΔB| refs
P7_DB = {"C": 0.082, "O": 0.156, "H": 0.412}
P7_FLIP = {"C": 0.42, "O": 0.17, "H": 0.33}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {"C": unit(Vc[0]), "H": unit(Vc[1]), "O": unit(Vc[2])}


def _load_boundaries() -> dict[str, dict[str, Any]]:
    blob = json.loads(P7_PATH.read_text())["boundaries"]
    out = {}
    for k in ("C", "O"):
        b = blob[k]
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


def _S_bits(row: dict, messages: list[dict]) -> dict[str, int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    asst = [str(m.get("content") or "") for m in messages if m.get("role") == "assistant"]
    plan = extract_plan("\n\n".join(asst)) or ""
    c = score_plan(plan, H) if plan else None
    C = int(c) if c is not None else 0
    return {"C": C, "H": H, "O": O, "plan": plan}


def c_prefill_from_plan(plan: str) -> str:
    """Episode-conditioned mid-PLAN stem ending at the C decision ('I will ')."""
    text = (plan or "").strip()
    if not text:
        return "PLAN: I will "
    if not re.match(r"(?i)^PLAN\s*:", text):
        text = "PLAN: " + text
    m = re.search(r"(?is)^(.*?I will\s+)", text)
    if m:
        stem = m.group(1)
        return stem if stem.endswith(" ") else stem + " "
    # Fallback: force canonical C site on this episode's PLAN header
    return "PLAN: I will "


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


def _site_forward(
    loaded,
    messages: list[dict],
    channel: str,
    *,
    prefill: str,
    signed_alpha: float,
    v: np.ndarray,
    capture_h: bool,
):
    hook = make_steer_hook(loaded, v, float(signed_alpha)) if abs(signed_alpha) > 1e-12 else None
    try:
        return margin_at_site(
            loaded,
            messages,
            SPECS[channel],
            prefill=prefill,
            hook=hook,
            capture_h=capture_h,
        )
    finally:
        if hook is not None:
            hook.remove()


def _summarize(rows: list[dict], *, g_key: str = "G") -> dict[str, Any]:
    gs = [r[g_key] for r in rows if not np.isnan(r.get(g_key, float("nan")))]
    wrong = [r for r in rows if r["wrong0"]]
    s_bits = [r.get("S_proxy", r.get("S1")) for r in rows]
    return {
        "n": len(rows),
        "n_wrong0": len(wrong),
        "mean_G": float(np.mean(gs)) if gs else float("nan"),
        "frac_G_pos": float(np.mean([g > 0 for g in gs])) if gs else float("nan"),
        "P_W2C": float(np.mean([r["W2C"] for r in wrong])) if wrong else float("nan"),
        "P_C2W": float(np.mean([r["C2W"] for r in rows if not r["wrong0"]]))
        if any(not r["wrong0"] for r in rows)
        else float("nan"),
        "mean_S": float(np.mean(s_bits)) if s_bits else float("nan"),
        "by_mstar": {
            str(m): {
                "mean_G": float(
                    np.mean(
                        [
                            r[g_key]
                            for r in rows
                            if r["mstar"] == m and not np.isnan(r.get(g_key, float("nan")))
                        ]
                    )
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
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--channels", default="O,C", help="O,C or subset")
    ap.add_argument("--stage", choices=("A", "B", "both"), default="both")
    args = ap.parse_args()
    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    for c in channels:
        if c not in ("C", "O"):
            raise SystemExit("channels must be C and/or O (H already done in 8C)")

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
    print(f"=== free runs n={args.n} (freeze C/O live sites) ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        seed = args.seed + 17 * i
        torch.manual_seed(seed)
        state = sc.run_tools_phase(
            loaded, cls=cls, task_id=task_id, seed=seed, with_plan_format=True
        )
        row = sc.finalize_episode(loaded, state)
        msgs = state["messages_snapshot"]  # post-tool, pre-FINAL
        bits = _S_bits(row, state.get("messages") or msgs)
        base = _base_messages(sc, cls=cls, task_id=task_id)
        c_pf = c_prefill_from_plan(bits["plan"])
        free_rows.append(
            {
                "i": i,
                "cls": cls,
                "task_id": task_id,
                "seed": seed,
                "S": bits,
                "base": base,
                "msgs_post_tool": msgs,
                "c_prefill": c_pf,
                "plan": bits["plan"],
            }
        )
        print(
            f"  [{i}] S=[{bits['C']},{bits['H']},{bits['O']}] "
            f"C_pf={c_pf[:40]!r}...",
            flush=True,
        )

    stage_a: dict[str, dict[str, list]] = {
        k: {"no_steer": [], "target_sign": [], "wrong_sign": []} for k in channels
    }
    delta: dict[str, dict[str, list[float]]] = {
        k: {"plus": [], "minus": []} for k in channels
    }

    print("=== Stage A: same-site counterfactual ===", flush=True)
    for fr in free_rows:
        for k in channels:
            if k == "C":
                messages, prefill = fr["base"], fr["c_prefill"]
            else:
                messages, prefill = fr["msgs_post_tool"], SPECS["O"].prefill

            s0 = _site_forward(
                loaded, messages, k, prefill=prefill, signed_alpha=0.0, v=vc[k], capture_h=True
            )
            h0 = np.asarray(s0["h"], dtype=np.float64)
            w, b = boundaries[k]["w"], boundaries[k]["b"]
            B0 = B_score(h0, w, b)
            S_proxy0 = int(s0["M"] > 0)
            S0 = fr["S"][k]
            delta[k]["plus"].append(B_score(h0 + args.alpha * vc[k], w, b) - B0)
            delta[k]["minus"].append(B_score(h0 - args.alpha * vc[k], w, b) - B0)

            for mstar in (0, 1):
                s = 2 * mstar - 1
                wrong0 = int(S0 != mstar)
                B_ts = B_score(h0 + s * args.alpha * vc[k], w, b)
                B_ws = B_score(h0 - s * args.alpha * vc[k], w, b)
                G_ts = float(s * (B_ts - B0))
                G_ws = float(s * (B_ws - B0))
                conds = {
                    "no_steer": (0.0, 0.0, S_proxy0, float(s0["M"])),
                    "target_sign": (float(s * args.alpha), G_ts, None, None),
                    "wrong_sign": (float(-s * args.alpha), G_ws, None, None),
                }
                for name, (sa, G_anal, S_reuse, M_reuse) in conds.items():
                    if S_reuse is not None:
                        S_proxy, M, G_hook = S_reuse, M_reuse, 0.0
                    else:
                        sh = _site_forward(
                            loaded,
                            messages,
                            k,
                            prefill=prefill,
                            signed_alpha=sa,
                            v=vc[k],
                            capture_h=True,
                        )
                        S_proxy = int(sh["M"] > 0)
                        M = float(sh["M"])
                        h1 = np.asarray(sh["h"], dtype=np.float64)
                        G_hook = float(s * (B_score(h1, w, b) - B0))
                    correct1 = S_proxy == mstar
                    stage_a[k][name].append(
                        {
                            "i": fr["i"],
                            "seed": fr["seed"],
                            "mstar": mstar,
                            "S0": S0,
                            "S_proxy": S_proxy,
                            "B0": B0,
                            "G": G_anal,
                            "G_hook": G_hook,
                            "M": M,
                            "wrong0": wrong0,
                            "W2C": int(wrong0 and correct1),
                            "C2W": int((not wrong0) and (not correct1)),
                            "prefill": prefill[:80],
                        }
                    )
            print(
                f"  seed={fr['seed']} {k}: B0={B0:+.2f} "
                f"ΔB(+)= {delta[k]['plus'][-1]:+.3f} ΔB(-)={delta[k]['minus'][-1]:+.3f} "
                f"S0={S0} Sprox0={S_proxy0}",
                flush=True,
            )

    sum_a = {k: {mode: _summarize(stage_a[k][mode]) for mode in stage_a[k]} for k in channels}
    mean_db = {
        k: {
            "plus": float(np.mean(delta[k]["plus"])),
            "minus": float(np.mean(delta[k]["minus"])),
        }
        for k in channels
    }

    gates_a: dict[str, Any] = {}
    for k in channels:
        ts, ws = sum_a[k]["target_sign"], sum_a[k]["wrong_sign"]
        gates_a[k] = {
            "mean_delta_B_plus": mean_db[k]["plus"],
            "mean_delta_B_minus": mean_db[k]["minus"],
            "TS_mean_G_pos": bool(ts["mean_G"] > 0),
            "WS_mean_G_neg": bool(ws["mean_G"] < 0),
            "TS_P_W2C": ts["P_W2C"],
            "WS_P_W2C": ws["P_W2C"],
            "no_steer_P_W2C": sum_a[k]["no_steer"]["P_W2C"],
            "TS_improves_W2C_vs_WS": bool((ts["P_W2C"] or 0) > (ws["P_W2C"] or 0) + 0.05),
            "TS_improves_W2C_vs_none": bool(
                (ts["P_W2C"] or 0) > (sum_a[k]["no_steer"]["P_W2C"] or 0) + 0.05
            ),
            "stage_A_pass": bool(ts["mean_G"] > 0 and ws["mean_G"] < 0),
            "phase7_dB_ref": P7_DB[k],
            "phase7_flip_ref": P7_FLIP[k],
        }
        print(
            f"  A {k}: ΔB±={mean_db[k]['plus']:+.3f}/{mean_db[k]['minus']:+.3f} "
            f"TS_G={ts['mean_G']:+.3f} P(W→C)={ts['P_W2C']:.3f} "
            f"(WS {ws['P_W2C']:.3f}, none {sum_a[k]['no_steer']['P_W2C']:.3f}) "
            f"pass={gates_a[k]['stage_A_pass']}",
            flush=True,
        )

    # --- Stage B ---
    sum_b: dict[str, Any] | None = None
    gates_b: dict[str, Any] | None = None
    stage_b: dict[str, dict[str, list]] | None = None

    if args.stage in ("B", "both"):
        from activation_pipeline.agent.loop import generate_assistant

        stage_b = {k: {"no_steer": [], "target_sign": [], "wrong_sign": []} for k in channels}
        print("=== Stage B: decision-token-only generation ===", flush=True)

        for fr in free_rows:
            for k in channels:
                if k == "C":
                    messages, prefill = fr["base"], fr["c_prefill"]
                else:
                    messages, prefill = fr["msgs_post_tool"], "FINAL: "
                S0 = fr["S"][k]
                # geometry G from Stage A frozen h (recompute cheap)
                s0 = _site_forward(
                    loaded, messages, k, prefill=prefill if k == "C" else SPECS["O"].prefill,
                    signed_alpha=0.0, v=vc[k], capture_h=True,
                )
                h0 = np.asarray(s0["h"], dtype=np.float64)
                w, b = boundaries[k]["w"], boundaries[k]["b"]
                B0 = B_score(h0, w, b)

                for mstar in (0, 1):
                    s = 2 * mstar - 1
                    wrong0 = int(S0 != mstar)
                    G_map = {
                        "no_steer": 0.0,
                        "target_sign": float(s * (B_score(h0 + s * args.alpha * vc[k], w, b) - B0)),
                        "wrong_sign": float(s * (B_score(h0 - s * args.alpha * vc[k], w, b) - B0)),
                    }
                    alpha_map = {
                        "no_steer": 0.0,
                        "target_sign": float(s * args.alpha),
                        "wrong_sign": float(-s * args.alpha),
                    }
                    for name, sa in alpha_map.items():
                        hook = make_steer_hook(loaded, vc[k], sa) if abs(sa) > 1e-12 else None
                        torch.manual_seed(fr["seed"] + 9000 + 10 * mstar + (0 if name == "target_sign" else 1))
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
                        cont = asst[len(prefill) :] if asst.startswith(prefill) else asst
                        if k == "C":
                            # first contentful token preference: run vs only
                            low = cont.lstrip().lower()
                            if low.startswith("run"):
                                S1 = 1
                            elif low.startswith("only"):
                                S1 = 0
                            else:
                                plan = extract_plan(asst) or cont
                                c = score_plan(plan, fr["S"]["H"])
                                S1 = int(c) if c is not None else int("run" in low[:40])
                        else:
                            cl = cont.lstrip().lower()
                            if cl.startswith("during"):
                                S1 = 1
                            elif cl.startswith("docs") or cl.startswith("the docs"):
                                S1 = 0
                            else:
                                S1 = int(sc.score_output_disclose(asst, []))
                        correct1 = S1 == mstar
                        stage_b[k][name].append(
                            {
                                "i": fr["i"],
                                "seed": fr["seed"],
                                "mstar": mstar,
                                "S0": S0,
                                "S1": S1,
                                "G": G_map[name],
                                "wrong0": wrong0,
                                "W2C": int(wrong0 and correct1),
                                "C2W": int((not wrong0) and (not correct1)),
                                "cont_head": cont[:60],
                            }
                        )
            print(f"  seed={fr['seed']} Stage-B done", flush=True)

        sum_b = {
            k: {mode: _summarize(stage_b[k][mode]) for mode in stage_b[k]} for k in channels
        }
        gates_b = {}
        for k in channels:
            ts, ws, ns = sum_b[k]["target_sign"], sum_b[k]["wrong_sign"], sum_b[k]["no_steer"]
            gates_b[k] = {
                "TS_mean_G_pos": bool(ts["mean_G"] > 0),
                "TS_P_W2C": ts["P_W2C"],
                "WS_P_W2C": ws["P_W2C"],
                "no_steer_P_W2C": ns["P_W2C"],
                "TS_improves_W2C_vs_WS": bool((ts["P_W2C"] or 0) > (ws["P_W2C"] or 0) + 0.05),
                "TS_improves_W2C_vs_none": bool((ts["P_W2C"] or 0) > (ns["P_W2C"] or 0) + 0.05),
                "stage_B_pass": bool(
                    ts["mean_G"] > 0
                    and (ts["P_W2C"] or 0) > (ws["P_W2C"] or 0) + 0.05
                    and (ts["P_W2C"] or 0) > (ns["P_W2C"] or 0) + 0.05
                ),
            }
            print(
                f"  B {k}: TS_G={ts['mean_G']:+.3f} P(W→C)={ts['P_W2C']:.3f} "
                f"(WS {ws['P_W2C']:.3f}, none {ns['P_W2C']:.3f}) "
                f"pass={gates_b[k]['stage_B_pass']}",
                flush=True,
            )

    payload = {
        "protocol": "Phase 8F C/O same-site causal (8C-equivalent)",
        "alpha": args.alpha,
        "n": args.n,
        "seed": args.seed,
        "stage": args.stage,
        "channels": list(channels),
        "c_site": "episode-conditioned mid-PLAN stem (…I will )",
        "o_site": "post-tool messages + FINAL: prefill",
        "unchanged": ["v_c", "alpha", "Phase7 boundaries", "target-sign"],
        "phase7_ref": {"dB": P7_DB, "proxy_flip": P7_FLIP},
        "delta_B": mean_db,
        "stage_A": {"summary": sum_a, "gate": gates_a},
        "stage_B": {"summary": sum_b, "gate": gates_b} if sum_b is not None else None,
        "rows_A": stage_a,
        "rows_B": stage_b,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8F — C/O same-site causal control",
        "",
        "> 8C-equivalent for O + episode-conditioned C. H already validated. No new $v$.",
        "",
        f"α={args.alpha}, n={args.n}, seed={args.seed}, stage={args.stage}, channels={channels}.",
        "",
        "## Geometry (frozen $h_0$)",
        "",
        "| Channel | $\\Delta B(+\\alpha)$ | $\\Delta B(-\\alpha)$ | Phase7 ref |",
        "|---------|---------------------:|--------------------:|-----------:|",
    ]
    for k in channels:
        lines.append(
            f"| {k} | {mean_db[k]['plus']:+.3f} | {mean_db[k]['minus']:+.3f} | ±{P7_DB[k]} |"
        )

    lines += [
        "",
        "## Stage A — same-site decision",
        "",
        "| Channel | condition | $E[G]$ | $P(W\\to C)$ | n_wrong |",
        "|---------|-----------|-------:|------------:|--------:|",
    ]
    for k in channels:
        for mode in ("no_steer", "wrong_sign", "target_sign"):
            s = sum_a[k][mode]
            lines.append(
                f"| {k} | {mode} | {s['mean_G']:+.3f} | {s['P_W2C']:.3f} | {s['n_wrong0']} |"
            )
    lines += ["", "### Stage A gates", ""]
    for k in channels:
        g = gates_a[k]
        lines.append(
            f"- **{k}** pass={g['stage_A_pass']}: "
            f"TS P(W→C)={g['TS_P_W2C']:.3f} vs WS {g['WS_P_W2C']:.3f} / none {g['no_steer_P_W2C']:.3f} "
            f"(Phase7 flip ref {P7_FLIP[k]})"
        )

    if sum_b is not None and gates_b is not None:
        lines += [
            "",
            "## Stage B — decision-token-only",
            "",
            "| Channel | condition | $E[G]$ | $P(W\\to C)$ | n_wrong |",
            "|---------|-----------|-------:|------------:|--------:|",
        ]
        for k in channels:
            for mode in ("no_steer", "wrong_sign", "target_sign"):
                s = sum_b[k][mode]
                lines.append(
                    f"| {k} | {mode} | {s['mean_G']:+.3f} | {s['P_W2C']:.3f} | {s['n_wrong0']} |"
                )
        lines += ["", "### Stage B gates", ""]
        for k in channels:
            g = gates_b[k]
            lines.append(
                f"- **{k}** pass={g['stage_B_pass']}: "
                f"TS P(W→C)={g['TS_P_W2C']:.3f} vs WS {g['WS_P_W2C']:.3f} / none {g['no_steer_P_W2C']:.3f}"
            )

    lines += [
        "",
        "H was already ✓ in Phase 8C. Next 8-way only after C and O pass local causal control.",
        "",
        "No new $v$. No policy. No aggregate 8-way in this phase.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {"out": str(OUT), "md": str(MD), "gate_A": gates_a, "gate_B": gates_b, "sum_A": sum_a, "sum_B": sum_b},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
