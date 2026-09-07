#!/usr/bin/env python3
"""Phase 7 — Episode-conditioned boundary discovery (frozen v_c).

Prior result: template M_k is deterministic and uncalibrated to episode S.
Question: is there a live decision-site feature that (1) predicts S_k and
(2) is moved by the frozen causal direction?

Protocol (no new v, no policy):
  1) Free-run episodes; capture h_k^live + M_k^live at C/H/O decision sites
     (episode-conditioned prefills — NOT fixed templates).
  2) Fit S_k ~ σ(w_kᵀ h + b); report AUC_live vs AUC_template.
  3) On live h: ΔB_k = B(h±α v_c) − B(h); sign consistency with channel=1.
  4) Local dose-response at live site: steered forward → soft bit from logits
     (S_proxy) at −α / 0 / +α.

  .venv/bin/python scripts/run_phase7_live_boundary.py --n 32 --causal-n 12
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
    margin_at_site,
    messages_for_channel,
    unit,
)
from scripts.sync_eq import extract_plan, score_plan  # noqa: E402
from scripts.sync_h_decision import LAYER_DEFAULT, decision_at_site, make_steer_hook  # noqa: E402

OUT = ROOT / "data" / "results" / "sync_phase7_live_boundary.json"
MD = ROOT / "data" / "results" / "sync_phase7_live_boundary.md"
VC_PATH = ROOT / "data" / "directions" / "sync_channel_Vc_L4.json"
CACHE = ROOT / "data" / "results" / "sync_phase7_live_boundary_cache.npz"

SEED = 20260912
ALPHA = 1.5
CHANNELS = ("C", "H", "O")
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _load_vc() -> dict[str, np.ndarray]:
    Vc = np.asarray(json.loads(VC_PATH.read_text())["V_c"], dtype=np.float64)
    return {ch: unit(Vc[i]) for i, ch in enumerate(CHANNELS)}


def _base_messages(sc, *, cls: str, task_id: str) -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {
            "role": "user",
            "content": sc.task_user_message(cls, task, with_plan_format=True),
        },
    ]


def _score_S(row: dict, messages: list[dict]) -> dict[str, int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    asst = [str(m.get("content") or "") for m in messages if m.get("role") == "assistant"]
    blob = (row.get("final") or "") + "\n" + "\n\n".join(asst)
    plan = extract_plan(blob)
    c = score_plan(plan, H) if plan else None
    C = int(c) if c is not None else -1
    return {"C": C, "H": H, "O": O, "plan": (plan or "")[:400]}


def capture_live_sites(
    loaded,
    sc,
    *,
    state: dict,
    cls: str,
    task_id: str,
) -> dict[str, Any]:
    """Episode-conditioned (h, M) at C/H/O decision sites."""
    base = _base_messages(sc, cls=cls, task_id=task_id)
    msgs = state["messages_snapshot"]
    asst = [str(m.get("content") or "") for m in msgs if m.get("role") == "assistant"]
    plan = extract_plan("\n\n".join(asst)) or ""

    out: dict[str, Any] = {}

    # C: mid-PLAN on episode task prompt
    sC = margin_at_site(loaded, base, SPECS["C"], capture_h=True)
    out["C"] = {
        "h": np.asarray(sC["h"], dtype=np.float64),
        "M": float(sC["M"]),
        "site": sC["site"],
        "prefill": sC["prefill"],
        "kind": "live_task_prompt",
    }

    # H: post-PLAN with *actual* generated PLAN
    if plan.strip():
        plan_pf = plan if plan.endswith("\n") else plan + "\n"
        sH = decision_at_site(loaded, base, assistant_prefill=plan_pf, capture_h=True)
        out["H"] = {
            "h": np.asarray(sH["h"], dtype=np.float64),
            "M": float(sH["M_H"]),
            "site": sH["site"],
            "prefill": plan_pf[:200],
            "kind": "live_post_PLAN",
        }
    else:
        out["H"] = None

    # O: post-tool messages + FINAL prefill
    sO = margin_at_site(loaded, msgs, SPECS["O"], capture_h=True)
    out["O"] = {
        "h": np.asarray(sO["h"], dtype=np.float64),
        "M": float(sO["M"]),
        "site": sO["site"],
        "prefill": sO["prefill"],
        "kind": "live_post_tool",
    }
    return out


def capture_template_sites(loaded, sc) -> dict[str, Any]:
    """Fixed-template probe (the broken baseline)."""
    out = {}
    for k in CHANNELS:
        msgs = messages_for_channel(sc, k)
        s = margin_at_site(loaded, msgs, SPECS[k], capture_h=True)
        out[k] = {
            "h": np.asarray(s["h"], dtype=np.float64),
            "M": float(s["M"]),
            "kind": "template",
        }
    return out


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def fit_live_boundary(
    H: np.ndarray, y: np.ndarray, *, seed: int = 0
) -> dict[str, Any]:
    """S ~ σ(wᵀh + b); held-out AUC + full-fit weights for ΔB tests."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    y = np.asarray(y, dtype=np.int64)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if n0 < 3 or n1 < 3:
        return {
            "ok": False,
            "n0": n0,
            "n1": n1,
            "auc_heldout": float("nan"),
            "auc_train": float("nan"),
            "w": None,
            "b": None,
            "reason": "insufficient_class_counts",
        }
    Xtr, Xte, ytr, yte = train_test_split(
        H, y, test_size=0.35, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(Xtr, ytr)
    auc_te = _auc(clf.decision_function(Xte), yte)
    auc_tr = _auc(clf.decision_function(Xtr), ytr)
    # refit on all for causal ΔB
    clf_all = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf_all.fit(H, y)
    w = clf_all.coef_.ravel().astype(np.float64)
    b = float(clf_all.intercept_[0])
    return {
        "ok": True,
        "n0": n0,
        "n1": n1,
        "auc_heldout": auc_te,
        "auc_train": auc_tr,
        "w": w,
        "b": b,
        "mean_acc_heldout": float((clf.predict(Xte) == yte).mean()),
    }


def B_score(h: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.dot(w, h) + b)


def soft_bit_from_site(loaded, messages, channel: str, *, prefill: str | None, hook=None) -> dict:
    """Local S_proxy from site logits under optional steer."""
    if channel == "H":
        s = decision_at_site(loaded, messages, assistant_prefill=prefill, hook=hook)
        # S=1 if tool preferred
        return {
            "S_proxy": int(s["M_H"] > 0),
            "M": float(s["M_H"]),
            "P_pos": float(s.get("P_tool", float("nan"))),
        }
    s = margin_at_site(loaded, messages, SPECS[channel], prefill=prefill, hook=hook)
    return {
        "S_proxy": int(s["M"] > 0),
        "M": float(s["M"]),
        "P_pos": float(s["P_pos"]),
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32, help="free-run episodes for live boundary")
    ap.add_argument("--causal-n", type=int, default=12, help="episodes for ±α local dose-response")
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
    schedule = _stratified_schedule(sc, args.n, args.seed)

    print("=== template baseline (expect std≈0) ===", flush=True)
    tmpl = capture_template_sites(loaded, sc)
    for k in CHANNELS:
        print(f"  template {k}: M={tmpl[k]['M']:+.3f}", flush=True)

    records: list[dict[str, Any]] = []
    print(f"=== collect live sites n={args.n} ===", flush=True)
    for i, (cls, task_id) in enumerate(schedule):
        seed = args.seed + 13 * i
        torch.manual_seed(seed)
        state = sc.run_tools_phase(
            loaded, cls=cls, task_id=task_id, seed=seed, with_plan_format=True
        )
        row = sc.finalize_episode(loaded, state)
        S = _score_S(row, state["messages_snapshot"])
        live = capture_live_sites(loaded, sc, state=state, cls=cls, task_id=task_id)
        rec = {
            "i": i,
            "cls": cls,
            "task_id": task_id,
            "seed": seed,
            "S": {k: S[k] for k in CHANNELS},
            "plan": S["plan"],
            "live": {},
            "base_msgs": _base_messages(sc, cls=cls, task_id=task_id),
            "plan_prefill": (live["H"] or {}).get("prefill") if live.get("H") else None,
            "msgs_len": len(state["messages_snapshot"]),
        }
        # store messages snapshot lightly for causal (O needs full msgs — keep in side list)
        rec["_msgs"] = state["messages_snapshot"]
        for k in CHANNELS:
            site = live.get(k)
            if site is None or site.get("h") is None:
                rec["live"][k] = None
                continue
            rec["live"][k] = {
                "M": site["M"],
                "kind": site["kind"],
                "site": site["site"],
                "h": site["h"],
            }
        records.append(rec)
        Mc = rec["live"]["C"]["M"] if rec["live"].get("C") else None
        Mh = rec["live"]["H"]["M"] if rec["live"].get("H") else None
        Mo = rec["live"]["O"]["M"] if rec["live"].get("O") else None
        print(
            f"  [{i}] cls={cls} task={task_id} S={rec['S']} "
            f"M_live={{C:{Mc if Mc is None else f'{Mc:+.2f}'},"
            f"H:{Mh if Mh is None else f'{Mh:+.2f}'},"
            f"O:{Mo if Mo is None else f'{Mo:+.2f}'}}}",
            flush=True,
        )

    # --- predictive AUC ---
    boundaries: dict[str, Any] = {}
    print("=== fit live boundaries ===", flush=True)
    for k in CHANNELS:
        hs, ys, Ms_live, Ms_tmpl = [], [], [], []
        for r in records:
            if r["live"].get(k) is None:
                continue
            if r["S"][k] not in (0, 1):
                continue
            hs.append(r["live"][k]["h"])
            ys.append(r["S"][k])
            Ms_live.append(r["live"][k]["M"])
            Ms_tmpl.append(tmpl[k]["M"])
        Hmat = np.stack(hs, axis=0) if hs else np.zeros((0, 1))
        y = np.asarray(ys, dtype=np.int64)
        fit = fit_live_boundary(Hmat, y, seed=args.seed + hash(k) % 1000) if len(y) else {
            "ok": False, "n0": 0, "n1": 0, "auc_heldout": float("nan"), "reason": "empty"
        }
        auc_M_live = _auc(np.asarray(Ms_live, dtype=np.float64), y) if len(y) else float("nan")
        # template M is constant → AUC undefined / chance
        auc_M_tmpl = _auc(np.asarray(Ms_tmpl, dtype=np.float64), y) if len(y) else float("nan")
        # template h is constant — use same vector for all → AUC nan
        Ht = np.stack([tmpl[k]["h"]] * len(y), axis=0) if len(y) else np.zeros((0, 1))
        if len(y) and float(np.std(Ht)) > 1e-12:
            fit_t = fit_live_boundary(Ht, y, seed=args.seed)
            auc_h_tmpl = fit_t.get("auc_heldout", float("nan"))
        else:
            auc_h_tmpl = float("nan")
        std_M_live = float(np.std(Ms_live)) if Ms_live else float("nan")
        std_h_live = float(np.mean([np.std(h) for h in hs])) if hs else float("nan")
        # variation across episodes of h
        if len(hs) > 1:
            Hstack = np.stack(hs, axis=0)
            std_h_across = float(np.mean(np.std(Hstack, axis=0)))
        else:
            std_h_across = float("nan")

        boundaries[k] = {
            **{kk: vv for kk, vv in fit.items() if kk != "w"},
            "w": fit.get("w").tolist() if fit.get("w") is not None else None,
            "b": fit.get("b"),
            "auc_M_live": auc_M_live,
            "auc_M_template": auc_M_tmpl,
            "auc_h_template": auc_h_tmpl,
            "std_M_live": std_M_live,
            "std_M_template": 0.0,
            "std_h_across_episodes": std_h_across,
            "n": int(len(y)),
            "live_beats_template_auc": bool(
                fit.get("ok")
                and (fit.get("auc_heldout") or 0) > 0.65
                and (
                    np.isnan(auc_M_tmpl)
                    or (fit.get("auc_heldout") or 0) > (auc_M_tmpl or 0) + 0.05
                )
            ),
        }
        # keep w in memory for ΔB
        boundaries[k]["_w"] = fit.get("w")
        print(
            f"  {k}: n0={boundaries[k].get('n0')} n1={boundaries[k].get('n1')} "
            f"AUC_live_h={boundaries[k].get('auc_heldout')} "
            f"AUC_M_live={auc_M_live:.3f} AUC_M_tmpl={auc_M_tmpl} "
            f"std_M_live={std_M_live:.3f} std_h_ep={std_h_across:.4f}",
            flush=True,
        )

    # --- ΔB causal on activations ---
    print("=== ΔB from frozen v_c on live h ===", flush=True)
    delta_B: dict[str, Any] = {}
    for k in CHANNELS:
        w = boundaries[k].get("_w")
        b = boundaries[k].get("b")
        if w is None:
            delta_B[k] = {"ok": False, "reason": "no_boundary"}
            continue
        d_plus, d_minus = [], []
        for r in records:
            if r["live"].get(k) is None:
                continue
            h = r["live"][k]["h"]
            B0 = B_score(h, w, b)
            Bp = B_score(h + args.alpha * vc[k], w, b)
            Bm = B_score(h - args.alpha * vc[k], w, b)
            d_plus.append(Bp - B0)
            d_minus.append(Bm - B0)
        delta_B[k] = {
            "ok": True,
            "mean_delta_B_plus": float(np.mean(d_plus)),
            "mean_delta_B_minus": float(np.mean(d_minus)),
            "frac_plus_increases_B": float(np.mean([d > 0 for d in d_plus])),
            "frac_minus_decreases_B": float(np.mean([d < 0 for d in d_minus])),
            "monotonic_dose": bool(
                float(np.mean(d_plus)) > 0 and float(np.mean(d_minus)) < 0
            ),
        }
        print(
            f"  {k}: ΔB(+α)={delta_B[k]['mean_delta_B_plus']:+.3f} "
            f"ΔB(−α)={delta_B[k]['mean_delta_B_minus']:+.3f} "
            f"mono={delta_B[k]['monotonic_dose']}",
            flush=True,
        )

    # --- local dose-response at live site (logit proxy) ---
    print(f"=== local dose-response causal-n={args.causal_n} ===", flush=True)
    dose: dict[str, Any] = {k: [] for k in CHANNELS}
    causal_recs = [r for r in records if r["live"].get("H") is not None][: args.causal_n]
    for r in causal_recs:
        base = r["base_msgs"]
        for k in CHANNELS:
            if r["live"].get(k) is None:
                continue
            if k == "H":
                msgs, pf = base, r["plan_prefill"]
            elif k == "C":
                msgs, pf = base, SPECS["C"].prefill
            else:
                msgs, pf = r["_msgs"], SPECS["O"].prefill
            row_dose = {"seed": r["seed"], "S_episode": r["S"][k]}
            for tag, a in (("m", -args.alpha), ("0", 0.0), ("p", args.alpha)):
                hook = None
                if abs(a) > 1e-12:
                    hook = make_steer_hook(loaded, vc[k], float(a))
                sb = soft_bit_from_site(loaded, msgs, k, prefill=pf, hook=hook)
                row_dose[f"S_proxy_{tag}"] = sb["S_proxy"]
                row_dose[f"M_{tag}"] = sb["M"]
                if hook is not None:
                    hook.remove()
            # B scores if available
            w = boundaries[k].get("_w")
            b0 = boundaries[k].get("b")
            if w is not None:
                h = r["live"][k]["h"]
                row_dose["B_m"] = B_score(h - args.alpha * vc[k], w, b0)
                row_dose["B_0"] = B_score(h, w, b0)
                row_dose["B_p"] = B_score(h + args.alpha * vc[k], w, b0)
            dose[k].append(row_dose)
        print(
            f"  causal seed={r['seed']} H: M-={dose['H'][-1]['M_m']:+.2f} "
            f"M0={dose['H'][-1]['M_0']:+.2f} M+={dose['H'][-1]['M_p']:+.2f} "
            f"Sprox={dose['H'][-1]['S_proxy_m']}/{dose['H'][-1]['S_proxy_0']}/{dose['H'][-1]['S_proxy_p']}",
            flush=True,
        )

    dose_summary: dict[str, Any] = {}
    for k in CHANNELS:
        rows = dose[k]
        if not rows:
            dose_summary[k] = {"ok": False}
            continue
        flip = [int(r["S_proxy_m"] != r["S_proxy_p"]) for r in rows]
        mono_M = [
            int(r["M_m"] < r["M_0"] < r["M_p"] or r["M_m"] > r["M_0"] > r["M_p"])
            for r in rows
        ]
        mono_B = []
        for r in rows:
            if "B_0" not in r:
                continue
            mono_B.append(int(r["B_m"] < r["B_0"] < r["B_p"] or r["B_m"] > r["B_0"] > r["B_p"]))
        dose_summary[k] = {
            "ok": True,
            "n": len(rows),
            "P_S_proxy_flip_pm": float(np.mean(flip)),
            "P_M_monotonic": float(np.mean(mono_M)),
            "P_B_monotonic": float(np.mean(mono_B)) if mono_B else float("nan"),
            "mean_M_m": float(np.mean([r["M_m"] for r in rows])),
            "mean_M_0": float(np.mean([r["M_0"] for r in rows])),
            "mean_M_p": float(np.mean([r["M_p"] for r in rows])),
        }

    # gates
    gate = {
        "question": "Does frozen v_c move the episode-conditioned decision boundary?",
        "live_auc_beats_template": {
            k: boundaries[k].get("live_beats_template_auc") for k in CHANNELS
        },
        "live_h_varies": {
            k: bool((boundaries[k].get("std_h_across_episodes") or 0) > 1e-4) for k in CHANNELS
        },
        "vc_moves_B_monotonic": {k: delta_B.get(k, {}).get("monotonic_dose") for k in CHANNELS},
        "vc_flips_S_proxy": {
            k: bool((dose_summary.get(k, {}).get("P_S_proxy_flip_pm") or 0) >= 0.25)
            for k in CHANNELS
        },
    }
    gate["any_live_boundary"] = any(
        boundaries[k].get("ok") and (boundaries[k].get("auc_heldout") or 0) >= 0.65
        for k in CHANNELS
    )
    gate["any_vc_moves_live_B"] = any(delta_B.get(k, {}).get("monotonic_dose") for k in CHANNELS)
    gate["chain_ok_any"] = bool(
        gate["any_live_boundary"] and gate["any_vc_moves_live_B"]
    )
    gate["read"] = (
        "If AUC_live ≫ template and v_c moves B monotonically with some S_proxy flips: "
        "missing map is recoverable. If AUC_live good but v_c does not move B: deeper problem."
    )

    # serialize (drop large msgs / raw w duplicates)
    ser_records = []
    for r in records:
        ser_records.append(
            {
                "i": r["i"],
                "cls": r["cls"],
                "task_id": r["task_id"],
                "seed": r["seed"],
                "S": r["S"],
                "plan": r["plan"],
                "live": {
                    k: (
                        None
                        if r["live"].get(k) is None
                        else {
                            "M": r["live"][k]["M"],
                            "kind": r["live"][k]["kind"],
                            "site": r["live"][k]["site"],
                            # omit full h from JSON (in npz)
                        }
                    )
                    for k in CHANNELS
                },
            }
        )

    # save cache of live h
    cache_obj = {}
    for k in CHANNELS:
        hs = [r["live"][k]["h"] for r in records if r["live"].get(k) is not None]
        ys = [r["S"][k] for r in records if r["live"].get(k) is not None and r["S"][k] in (0, 1)]
        if hs:
            cache_obj[f"h_{k}"] = np.stack(hs, axis=0)
            cache_obj[f"S_{k}"] = np.asarray(
                [r["S"][k] for r in records if r["live"].get(k) is not None], dtype=np.int64
            )
    if cache_obj:
        np.savez_compressed(CACHE, **cache_obj)

    payload = {
        "protocol": "Phase 7 episode-conditioned boundary",
        "alpha": args.alpha,
        "n": args.n,
        "causal_n": args.causal_n,
        "template_M": {k: tmpl[k]["M"] for k in CHANNELS},
        "boundaries": {
            k: {kk: vv for kk, vv in boundaries[k].items() if not kk.startswith("_")}
            for k in CHANNELS
        },
        "delta_B": delta_B,
        "dose_summary": dose_summary,
        "dose_rows": {k: dose[k] for k in CHANNELS},
        "records": ser_records,
        "gate": gate,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 7 — Episode-conditioned boundary",
        "",
        "> Does frozen $v_c$ move the *live* decision boundary $h^{\\mathrm{live}}\\to S$?",
        "",
        f"n={args.n}, causal-n={args.causal_n}, α={args.alpha}, layer={LAYER_DEFAULT}.",
        "",
        "## Template vs live probe variance",
        "",
        "| Channel | $M$ template | std $M$ live | std $h$ across episodes |",
        "|---------|--------------|--------------|-------------------------|",
    ]
    for k in CHANNELS:
        lines.append(
            f"| {k} | {tmpl[k]['M']:+.3f} | {boundaries[k].get('std_M_live', float('nan')):.3f} | "
            f"{boundaries[k].get('std_h_across_episodes', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## A — Predictive: $h^{\\mathrm{live}}\\to S$ vs template",
        "",
        "| Channel | n0/n1 | AUC live $h$ | AUC live $M$ | AUC template $M$ | live ≫ template |",
        "|---------|-------|--------------|--------------|------------------|-----------------|",
    ]
    for k in CHANNELS:
        b = boundaries[k]
        lines.append(
            f"| {k} | {b.get('n0')}/{b.get('n1')} | {b.get('auc_heldout', float('nan')):.3f} | "
            f"{b.get('auc_M_live', float('nan')):.3f} | {b.get('auc_M_template', float('nan'))} | "
            f"**{b.get('live_beats_template_auc')}** |"
        )

    lines += [
        "",
        "## B — Causal on activations: $\\Delta B(h\\pm\\alpha v_c)$",
        "",
        "| Channel | mean ΔB(+α) | mean ΔB(−α) | frac +↑B | mono dose |",
        "|---------|-------------|-------------|----------|-----------|",
    ]
    for k in CHANNELS:
        d = delta_B.get(k, {})
        if not d.get("ok"):
            lines.append(f"| {k} | — | — | — | — |")
            continue
        lines.append(
            f"| {k} | {d['mean_delta_B_plus']:+.3f} | {d['mean_delta_B_minus']:+.3f} | "
            f"{d['frac_plus_increases_B']:.2f} | **{d['monotonic_dose']}** |"
        )

    lines += [
        "",
        "## C — Local dose-response at live site ($S_{\\mathrm{proxy}}$ from site logits)",
        "",
        "| Channel | n | P(flip $S^-$↔$S^+$) | P(M mono) | P(B mono) | mean $M^-/M^0/M^+$ |",
        "|---------|---|---------------------|-----------|-----------|---------------------|",
    ]
    for k in CHANNELS:
        d = dose_summary.get(k, {})
        if not d.get("ok"):
            lines.append(f"| {k} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {k} | {d['n']} | {d['P_S_proxy_flip_pm']:.2f} | {d['P_M_monotonic']:.2f} | "
            f"{d['P_B_monotonic']:.2f} | "
            f"{d['mean_M_m']:+.2f}/{d['mean_M_0']:+.2f}/{d['mean_M_p']:+.2f} |"
        )

    lines += [
        "",
        "## Gate",
        "",
        f"- Any recoverable live boundary (AUC≥0.65): **{gate['any_live_boundary']}**",
        f"- Any $v_c$ moves live $B$ monotonically: **{gate['any_vc_moves_live_B']}**",
        f"- Chain recoverable on ≥1 channel: **{gate['chain_ok_any']}**",
        "",
        gate["read"],
        "",
        "Directions frozen. No new $v$. No policy.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
