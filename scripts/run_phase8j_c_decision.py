#!/usr/bin/env python3
"""Phase 8J — C actual decision control (8C-equivalent for C).

Phase 8I: C is the limiting actuator under H→C→O (ΔE_C < 0). Do NOT skip C
for the universal 8-way objective — that would abandon one required factor.

Phase 8F already showed the soft/live gap:
  Stage A (proxy): E[G_C]=+0.082, P(W→C)=0.92 ✓
  Stage B (gen):   P(W→C)=0.50 (flat across TS/WS/none) ✗

Same unresolved chain H had before 8C:
  B_C^live → S_C^proxy   but not yet   S_C^proxy → S_C^generation

This phase isolates C only:
  h_C^live → B_C^live → actual C decision tokens → S_C

Corrections vs 8F Stage B:
  - same seed across no_steer / target_sign / wrong_sign (8C protocol)
  - primary S_C^gen = first-token run vs only (aligned with soft margin)
  - secondary = score_plan on completed PLAN
  - optional α_C dose {1.5,3,5,8} if generation under-actuated

Frozen elsewhere: H decision-token ✓, O early-FINAL ✓, order H→C→O preferred.
No new v. No C-skip as success. No full 8-way reopen until C gen passes.

  .venv/bin/python scripts/run_phase8j_c_decision.py --n 16
  .venv/bin/python scripts/run_phase8j_c_decision.py --n 12 --alphas 1.5,3,5,8
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

OUT = ROOT / "data" / "results" / "sync_phase8j_c_decision.json"
MD = ROOT / "data" / "results" / "sync_phase8j_c_decision.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
P7_PATH = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"

SEED = 20260916
ALPHA_DEFAULT = 1.5
P7_DB_C = 0.082
REF_8F = {
    "A_G": 0.082,
    "A_W2C_TS": 0.917,
    "A_W2C_WS": 0.500,
    "B_W2C_TS": 0.500,
    "B_W2C_WS": 0.500,
}


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc_C() -> np.ndarray:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return unit(Vc[0])


def _load_boundary_C() -> tuple[np.ndarray, float]:
    b = json.loads(P7_PATH.read_text())["boundaries"]["C"]
    if not b.get("ok") or not b.get("w"):
        raise SystemExit("need Phase-7 C boundary")
    return np.asarray(b["w"], dtype=np.float64), float(b["b"])


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


def _base_messages(sc, *, cls: str, task_id: str) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def _S_bits(row: dict, messages: list[dict]) -> dict[str, Any]:
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


def _site_forward(loaded, messages, prefill: str, *, signed_alpha: float, v: np.ndarray, capture_h: bool):
    hook = make_steer_hook(loaded, v, float(signed_alpha)) if abs(signed_alpha) > 1e-12 else None
    try:
        return margin_at_site(
            loaded,
            messages,
            SPECS["C"],
            prefill=prefill,
            hook=hook,
            capture_h=capture_h,
        )
    finally:
        if hook is not None:
            hook.remove()


def _score_gen_C(asst: str, prefill: str, H: int) -> dict[str, Any]:
    """Primary: first-token run/only. Secondary: score_plan."""
    cont = asst[len(prefill) :] if asst.startswith(prefill) else asst
    low = cont.lstrip().lower()
    if low.startswith("run"):
        S_tok = 1
        tok = "run"
    elif low.startswith("only"):
        S_tok = 0
        tok = "only"
    else:
        # first alphanumeric word
        m = re.match(r"^\s*([A-Za-z]+)", cont)
        tok = (m.group(1).lower() if m else "")[:20]
        S_tok = -1  # unknown first token
    plan = extract_plan(asst) or (prefill + cont)
    c = score_plan(plan, H)
    S_plan = int(c) if c is not None else None
    # primary behavioral bit: token if clear, else plan score
    if S_tok >= 0:
        S_gen = S_tok
    elif S_plan is not None:
        S_gen = S_plan
    else:
        S_gen = int("run" in low[:40] and "only" not in low[:40])
    return {
        "S_gen": int(S_gen),
        "S_tok": int(S_tok),
        "S_plan": S_plan,
        "tok": tok,
        "cont_head": cont[:80],
    }


def _summarize(rows: list[dict], *, g_key: str = "G", s_key: str = "S_proxy") -> dict[str, Any]:
    gs = [r[g_key] for r in rows if not np.isnan(r.get(g_key, float("nan")))]
    wrong = [r for r in rows if r["wrong0"]]
    s_bits = [r[s_key] for r in rows if s_key in r and r[s_key] is not None]
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
        "frac_tok_clear": float(np.mean([r.get("S_tok", -1) >= 0 for r in rows]))
        if rows and "S_tok" in rows[0]
        else float("nan"),
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


def _gate_from_sums(sum_modes: dict[str, dict], *, require_G: bool = True) -> dict[str, Any]:
    ts, ws, ns = sum_modes["target_sign"], sum_modes["wrong_sign"], sum_modes["no_steer"]
    g = {
        "TS_mean_G_pos": bool(ts["mean_G"] > 0) if require_G else True,
        "TS_P_W2C": ts["P_W2C"],
        "WS_P_W2C": ws["P_W2C"],
        "no_steer_P_W2C": ns["P_W2C"],
        "TS_improves_W2C_vs_WS": bool((ts["P_W2C"] or 0) > (ws["P_W2C"] or 0) + 0.05),
        "TS_improves_W2C_vs_none": bool((ts["P_W2C"] or 0) > (ns["P_W2C"] or 0) + 0.05),
    }
    g["pass"] = bool(
        g["TS_mean_G_pos"]
        and g["TS_improves_W2C_vs_WS"]
        and g["TS_improves_W2C_vs_none"]
    )
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--alphas",
        default="1.5",
        help="comma-separated α_C for Stage B dose (default 1.5 only; try 1.5,3,5,8)",
    )
    ap.add_argument("--stage", choices=("A", "B", "both"), default="both")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="gen tokens after C stem")
    args = ap.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())

    from activation_pipeline.agent.loop import generate_assistant
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    vC = _load_vc_C()
    wC, bC = _load_boundary_C()
    schedule = _stratified_schedule(sc, args.n, args.seed)

    free_rows: list[dict[str, Any]] = []
    print(f"=== free runs n={args.n} (freeze live C sites) ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        seed = args.seed + 17 * i
        torch.manual_seed(seed)
        state = sc.run_tools_phase(
            loaded, cls=cls, task_id=task_id, seed=seed, with_plan_format=True
        )
        row = sc.finalize_episode(loaded, state)
        msgs = state.get("messages") or state["messages_snapshot"]
        bits = _S_bits(row, msgs)
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
                "c_prefill": c_pf,
                "plan": bits["plan"],
            }
        )
        print(
            f"  [{i}] S=[{bits['C']},{bits['H']},{bits['O']}] "
            f"C_pf={c_pf[:50]!r}...",
            flush=True,
        )

    # --- Stage A: soft/proxy at live C site ---
    sum_a: dict[str, Any] | None = None
    gate_a: dict[str, Any] | None = None
    stage_a: dict[str, list] | None = None
    mean_db = {"plus": float("nan"), "minus": float("nan")}

    if args.stage in ("A", "both"):
        stage_a = {"no_steer": [], "target_sign": [], "wrong_sign": []}
        d_plus: list[float] = []
        d_minus: list[float] = []
        print("=== Stage A: same-site soft/proxy (B_C → S_proxy) ===", flush=True)
        for fr in free_rows:
            messages, prefill = fr["base"], fr["c_prefill"]
            s0 = _site_forward(
                loaded, messages, prefill, signed_alpha=0.0, v=vC, capture_h=True
            )
            h0 = np.asarray(s0["h"], dtype=np.float64)
            B0 = B_score(h0, wC, bC)
            S_proxy0 = int(s0["M"] > 0)
            S0 = fr["S"]["C"]
            d_plus.append(B_score(h0 + ALPHA_DEFAULT * vC, wC, bC) - B0)
            d_minus.append(B_score(h0 - ALPHA_DEFAULT * vC, wC, bC) - B0)

            for mstar in (0, 1):
                s = 2 * mstar - 1
                wrong0 = int(S0 != mstar)
                for name, sa, G_anal in (
                    ("no_steer", 0.0, 0.0),
                    (
                        "target_sign",
                        float(s * ALPHA_DEFAULT),
                        float(s * (B_score(h0 + s * ALPHA_DEFAULT * vC, wC, bC) - B0)),
                    ),
                    (
                        "wrong_sign",
                        float(-s * ALPHA_DEFAULT),
                        float(s * (B_score(h0 - s * ALPHA_DEFAULT * vC, wC, bC) - B0)),
                    ),
                ):
                    if name == "no_steer":
                        S_proxy, M = S_proxy0, float(s0["M"])
                    else:
                        sh = _site_forward(
                            loaded,
                            messages,
                            prefill,
                            signed_alpha=sa,
                            v=vC,
                            capture_h=True,
                        )
                        S_proxy, M = int(sh["M"] > 0), float(sh["M"])
                    correct1 = S_proxy == mstar
                    stage_a[name].append(
                        {
                            "i": fr["i"],
                            "seed": fr["seed"],
                            "mstar": mstar,
                            "S0": S0,
                            "S_proxy": S_proxy,
                            "B0": B0,
                            "G": G_anal,
                            "M": M,
                            "wrong0": wrong0,
                            "W2C": int(wrong0 and correct1),
                            "C2W": int((not wrong0) and (not correct1)),
                        }
                    )
            print(
                f"  seed={fr['seed']} B0={B0:+.2f} "
                f"ΔB±={d_plus[-1]:+.3f}/{d_minus[-1]:+.3f} "
                f"S0={S0} Sprox0={S_proxy0}",
                flush=True,
            )

        mean_db = {"plus": float(np.mean(d_plus)), "minus": float(np.mean(d_minus))}
        sum_a = {mode: _summarize(stage_a[mode], s_key="S_proxy") for mode in stage_a}
        gate_a = _gate_from_sums(sum_a)
        gate_a["delta_B_matches_phase7"] = bool(
            abs(abs(mean_db["plus"]) - P7_DB_C) < 0.03
            and abs(abs(mean_db["minus"]) - P7_DB_C) < 0.03
        )
        for name in ("no_steer", "wrong_sign", "target_sign"):
            s = sum_a[name]
            print(
                f"  A {name}: G={s['mean_G']:+.3f} P(W→C)={s['P_W2C']:.3f} "
                f"n_wrong={s['n_wrong0']}",
                flush=True,
            )
        print(f"  Stage A pass: {gate_a['pass']}", flush=True)

    # --- Stage B: decision-token generation (same seed across conditions) ---
    by_alpha: dict[str, Any] = {}
    stage_b_all: dict[str, Any] = {}

    if args.stage in ("B", "both"):
        print(
            f"=== Stage B: C decision-token gen (same-seed) α∈{alphas} "
            f"max_new={args.max_new_tokens} ===",
            flush=True,
        )
        for alpha in alphas:
            stage_b = {"no_steer": [], "target_sign": [], "wrong_sign": []}
            print(f"--- α_C={alpha:g} ---", flush=True)
            for fr in free_rows:
                messages, prefill = fr["base"], fr["c_prefill"]
                S0 = fr["S"]["C"]
                H = fr["S"]["H"]
                # geometry G from frozen h0 at this α
                s0 = _site_forward(
                    loaded, messages, prefill, signed_alpha=0.0, v=vC, capture_h=True
                )
                h0 = np.asarray(s0["h"], dtype=np.float64)
                B0 = B_score(h0, wC, bC)

                for mstar in (0, 1):
                    s = 2 * mstar - 1
                    wrong0 = int(S0 != mstar)
                    G_map = {
                        "no_steer": 0.0,
                        "target_sign": float(s * (B_score(h0 + s * alpha * vC, wC, bC) - B0)),
                        "wrong_sign": float(s * (B_score(h0 - s * alpha * vC, wC, bC) - B0)),
                    }
                    alpha_map = {
                        "no_steer": 0.0,
                        "target_sign": float(s * alpha),
                        "wrong_sign": float(-s * alpha),
                    }
                    for name, sa in alpha_map.items():
                        hook = make_steer_hook(loaded, vC, sa) if abs(sa) > 1e-12 else None
                        # CRITICAL: same seed for all conditions (8C protocol; 8F bug fixed)
                        torch.manual_seed(fr["seed"] + 9000 + 10 * mstar)
                        if hook is not None:
                            hook.register()
                        try:
                            asst = generate_assistant(
                                loaded,
                                messages,
                                max_new_tokens=args.max_new_tokens,
                                temperature=sc.TEMPERATURE,
                                assistant_prefill=prefill,
                            )
                        finally:
                            if hook is not None:
                                hook.remove()
                        scored = _score_gen_C(asst, prefill, H)
                        S1 = scored["S_gen"]
                        correct1 = S1 == mstar
                        stage_b[name].append(
                            {
                                "i": fr["i"],
                                "seed": fr["seed"],
                                "mstar": mstar,
                                "alpha": alpha,
                                "S0": S0,
                                "S_gen": S1,
                                "S_tok": scored["S_tok"],
                                "S_plan": scored["S_plan"],
                                "G": G_map[name],
                                "wrong0": wrong0,
                                "W2C": int(wrong0 and correct1),
                                "C2W": int((not wrong0) and (not correct1)),
                                "tok": scored["tok"],
                                "cont_head": scored["cont_head"],
                            }
                        )
                print(f"  seed={fr['seed']} S0={S0} α={alpha:g} Stage-B done", flush=True)

            sum_b = {
                mode: _summarize(stage_b[mode], s_key="S_gen") for mode in stage_b
            }
            gate_b = _gate_from_sums(sum_b)
            # also token-clear W2C diagnostic
            for mode in stage_b:
                clear = [r for r in stage_b[mode] if r["S_tok"] >= 0 and r["wrong0"]]
                sum_b[mode]["P_W2C_tok_clear"] = (
                    float(np.mean([r["W2C"] for r in clear])) if clear else float("nan")
                )
                sum_b[mode]["n_wrong_tok_clear"] = len(clear)
            by_alpha[str(alpha)] = {"summary": sum_b, "gate": gate_b}
            stage_b_all[str(alpha)] = stage_b
            ts = sum_b["target_sign"]
            print(
                f"  >> α={alpha:g}: TS G={ts['mean_G']:+.3f} "
                f"P(W→C)_gen={ts['P_W2C']:.3f} "
                f"(WS {sum_b['wrong_sign']['P_W2C']:.3f}, "
                f"none {sum_b['no_steer']['P_W2C']:.3f}) "
                f"pass={gate_b['pass']}",
                flush=True,
            )

    # overall gate
    primary_a = str(alphas[0]) if alphas else "1.5"
    gate_b_primary = by_alpha.get(primary_a, {}).get("gate")
    best_alpha = None
    best_w2c = -1.0
    for a, blob in by_alpha.items():
        w2c = blob["summary"]["target_sign"]["P_W2C"] or 0.0
        if blob["gate"]["pass"] and w2c >= best_w2c:
            best_w2c = w2c
            best_alpha = float(a)
    if best_alpha is None and by_alpha:
        # pick highest TS W2C even if not pass
        for a, blob in by_alpha.items():
            w2c = blob["summary"]["target_sign"]["P_W2C"] or 0.0
            if w2c > best_w2c:
                best_w2c = w2c
                best_alpha = float(a)

    gate = {
        "hypothesis": (
            "C soft site valid; generation fails until decision-token path "
            "matches 8C (same-seed + run/only tokens). No C-skip for 8-way."
        ),
        "ref_8F": REF_8F,
        "stage_A": gate_a,
        "stage_B_by_alpha": {a: blob["gate"] for a, blob in by_alpha.items()},
        "stage_B_primary_pass": bool(gate_b_primary["pass"]) if gate_b_primary else None,
        "any_alpha_B_pass": bool(any(blob["gate"]["pass"] for blob in by_alpha.values())),
        "best_alpha_C": best_alpha,
        "best_TS_P_W2C": best_w2c if best_w2c >= 0 else None,
        "C_generation_fixed": bool(
            any(blob["gate"]["pass"] for blob in by_alpha.values())
        ),
        "read": (
            "If Stage B passes at some α_C: wire C decision-token (stem prefill) "
            "into H→C→O controller, then 8-way replication. "
            "If soft✓ gen✗ at all α: site/scoring still misaligned — not skip C, "
            "not new v yet. Conditional C-skip is diagnostic only, never 8-way success."
        ),
    }

    payload = {
        "protocol": "Phase 8J C actual decision control",
        "alpha_A": ALPHA_DEFAULT,
        "alphas_B": list(alphas),
        "n": args.n,
        "seed": args.seed,
        "stage": args.stage,
        "max_new_tokens": args.max_new_tokens,
        "c_site": "episode-conditioned mid-PLAN stem (…I will )",
        "gen_primary": "first-token run vs only",
        "gen_secondary": "score_plan",
        "same_seed_across_conditions": True,
        "unchanged": ["v_c", "Phase7 C boundary", "target-sign"],
        "frozen_elsewhere": {
            "H": "decision-token ✓ (8C)",
            "O": "early-FINAL α=1.5 ✓ (8H)",
            "order": "H→C→O preferred (8I)",
        },
        "delta_B": mean_db,
        "stage_A": {"summary": sum_a, "gate": gate_a} if sum_a is not None else None,
        "stage_B": by_alpha,
        "gate": gate,
        "rows_A": stage_a,
        "rows_B": stage_b_all,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 8J — C actual decision control",
        "",
        r"> Close $B_C^{\mathrm{live}}\!\to\!S_C^{\mathrm{proxy}}\!\to\!S_C^{\mathrm{generation}}$. "
        r"No C-skip for 8-way. No new $v$. Same-seed Stage B (8C protocol).",
        "",
        f"n={args.n}, seed={args.seed}, α_B={alphas}, max_new={args.max_new_tokens}.",
        "",
    ]
    if sum_a is not None and gate_a is not None:
        lines += [
            "## Stage A — soft/proxy",
            "",
            f"| $\\Delta B(+\\alpha)$ | $\\Delta B(-\\alpha)$ | Phase7 ref |",
            f"|---------------------:|--------------------:|-----------:|",
            f"| {mean_db['plus']:+.3f} | {mean_db['minus']:+.3f} | ±{P7_DB_C} |",
            "",
            "| condition | $E[G]$ | $P(W\\to C)$ | n_wrong |",
            "|-----------|-------:|------------:|--------:|",
        ]
        for mode in ("no_steer", "wrong_sign", "target_sign"):
            s = sum_a[mode]
            lines.append(
                f"| {mode} | {s['mean_G']:+.3f} | {s['P_W2C']:.3f} | {s['n_wrong0']} |"
            )
        lines += [
            "",
            f"- Stage A pass: **{gate_a['pass']}**",
            f"- 8F ref TS P(W→C)={REF_8F['A_W2C_TS']}",
            "",
        ]
    if by_alpha:
        lines += [
            "## Stage B — decision-token generation (same seed)",
            "",
            "| $\\alpha_C$ | condition | $E[G]$ | $P(W\\to C)_{gen}$ | n_wrong | pass |",
            "|-----------:|-----------|-------:|------------------:|--------:|:----:|",
        ]
        for a in alphas:
            blob = by_alpha[str(a)]
            for mode in ("no_steer", "wrong_sign", "target_sign"):
                s = blob["summary"][mode]
                pas = blob["gate"]["pass"] if mode == "target_sign" else ""
                lines.append(
                    f"| {a:g} | {mode} | {s['mean_G']:+.3f} | {s['P_W2C']:.3f} | "
                    f"{s['n_wrong0']} | {pas} |"
                )
        lines += [
            "",
            f"| 8F Stage B (buggy seed) | target_sign | +0.082 | "
            f"{REF_8F['B_W2C_TS']:.3f} | — | False |",
            "",
        ]
    lines += [
        "## Gate",
        "",
        f"- Stage A pass: **{gate.get('stage_A', {}).get('pass') if gate.get('stage_A') else 'n/a'}**",
        f"- Stage B pass (any α): **{gate['any_alpha_B_pass']}**",
        f"- Best α_C: **{gate.get('best_alpha_C')}** "
        f"(TS P(W→C)={gate.get('best_TS_P_W2C')})",
        f"- C generation fixed: **{gate['C_generation_fixed']}**",
        "",
        gate["read"],
        "",
        "C remains required for universal 8-way. Conditional skip = diagnostic only.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "md": str(MD),
                "gate": {k: v for k, v in gate.items() if k != "read"},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
