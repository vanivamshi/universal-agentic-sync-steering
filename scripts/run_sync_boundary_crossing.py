#!/usr/bin/env python3
"""Boundary-crossing + calibration diagnostic (frozen v_c).

After L_8way: vc moves joint margins strongly but hit/ΔE stay weak.
Question: is that margin movement failing to cross the *behavioral* boundary?

A) Per-bit transitions: W→C / C→W (from existing surrogate JSON or fresh)
B) P(W→C | tilde_M0) stratification
C) L_wrong / D_wrong vs ΔE
D) Calibration: P(S_k=1 | M_k) on free-run (+ optional steered) data

  # reanalyze prior surrogate + quick calibration
  .venv/bin/python scripts/run_sync_boundary_crossing.py --cal-reps 24

  # skip GPU calibration (JSON-only A/B/C)
  .venv/bin/python scripts/run_sync_boundary_crossing.py --no-calibrate
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

from scripts.sync_channel_margins import SPECS, margin_at_site, messages_for_channel  # noqa: E402
from scripts.sync_eq import extract_plan, score_plan  # noqa: E402

SURR = ROOT / "data" / "results" / "sync_8way_surrogate.json"
OUT = ROOT / "data" / "results" / "sync_boundary_crossing.json"
MD = ROOT / "data" / "results" / "sync_boundary_crossing.md"

CHANNELS = ("C", "H", "O")
IDX = {"C": 0, "H": 1, "O": 2}
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"
SEED = 20260911
BETA = 1.0


def softplus(x: float) -> float:
    x = float(x)
    if x > 20:
        return x
    if x < -20:
        return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _S_from_row(row: dict) -> list[int]:
    H = int(row.get("s_tool") or 0)
    O = int(row.get("s_output") or 0)
    blob = (row.get("final") or "") + "\n" + (row.get("messages_text") or "")
    plan = extract_plan(blob)
    if plan:
        c = score_plan(plan, H)
        C = int(c) if c is not None else 0
    else:
        C = 0
    return [C, H, O]


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def classify(s0: list[int], s1: list[int], mstar: list[int], k: str) -> str:
    i = IDX[k]
    w0 = s0[i] != mstar[i]
    c1 = s1[i] == mstar[i]
    if w0 and c1:
        return "W2C"
    if (not w0) and c1:
        return "C2C"
    if w0 and (not c1):
        return "W2W"
    return "C2W"


def analyze_surrogate(blob: dict) -> dict[str, Any]:
    rows = []
    for arm, trials in blob["trials"].items():
        for t in trials:
            mstar = t["mstar"]
            s0, s1 = t["S0"], t["S_final"]
            M0, M1 = t["M0"], t["M1"]
            Lw0 = Lw1 = Dw0 = Dw1 = 0.0
            n_wrong = 0
            tilde0 = {}
            for k in CHANNELS:
                i = IDX[k]
                sign = 2 * int(mstar[i]) - 1
                t0 = sign * float(M0[k])
                t1 = sign * float(M1[k])
                tilde0[k] = t0
                if s0[i] != mstar[i]:
                    n_wrong += 1
                    Lw0 += softplus(-BETA * t0)
                    Lw1 += softplus(-BETA * t1)
                    Dw0 += -t0
                    Dw1 += -t1
            rows.append(
                {
                    "arm": arm,
                    "mstar": mstar,
                    "s0": s0,
                    "s1": s1,
                    "n_wrong0": n_wrong,
                    "delta_E": t["delta_E"],
                    "neg_delta_L": t["neg_delta_L"],
                    "neg_delta_L_wrong": float(-(Lw1 - Lw0)),
                    "neg_delta_D_wrong": float(-(Dw1 - Dw0)),
                    "cls": {k: classify(s0, s1, mstar, k) for k in CHANNELS},
                    "tilde0": tilde0,
                    "dtilde": t["delta_tilde_M"],
                    "M0": M0,
                    "M1": M1,
                    # margin sign vs behavioral bit at S0 (calibration snapshot)
                    "signM_eq_S0": {
                        k: int((1 if M0[k] > 0 else 0) == s0[IDX[k]]) for k in CHANNELS
                    },
                }
            )

    arms = sorted({r["arm"] for r in rows})
    transitions: dict[str, Any] = {}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        transitions[arm] = {}
        for k in CHANNELS:
            counts = {c: 0 for c in ("W2C", "C2C", "W2W", "C2W")}
            for r in sub:
                counts[r["cls"][k]] += 1
            n_w = counts["W2C"] + counts["W2W"]
            n_c = counts["C2C"] + counts["C2W"]
            transitions[arm][k] = {
                **counts,
                "n_wrong0": n_w,
                "n_correct0": n_c,
                "P_W2C": float(counts["W2C"] / n_w) if n_w else float("nan"),
                "P_C2W": float(counts["C2W"] / n_c) if n_c else float("nan"),
            }

    wrong_vs_correct_dM: dict[str, Any] = {}
    for arm in arms:
        wrong_vs_correct_dM[arm] = {}
        for k in CHANNELS:
            dw, dc = [], []
            for r in rows:
                if r["arm"] != arm:
                    continue
                i = IDX[k]
                if r["s0"][i] != r["mstar"][i]:
                    dw.append(float(r["dtilde"][k]))
                else:
                    dc.append(float(r["dtilde"][k]))
            wrong_vs_correct_dM[arm][k] = {
                "mean_dtilde_wrong0": float(np.mean(dw)) if dw else float("nan"),
                "mean_dtilde_correct0": float(np.mean(dc)) if dc else float("nan"),
                "n_wrong0": len(dw),
                "n_correct0": len(dc),
            }

    surrogates: dict[str, Any] = {}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        subw = [r for r in sub if r["n_wrong0"] > 0]
        surrogates[arm] = {
            "n": len(sub),
            "mean_neg_delta_L": float(np.mean([r["neg_delta_L"] for r in sub])),
            "mean_neg_delta_L_wrong": float(np.mean([r["neg_delta_L_wrong"] for r in sub])),
            "mean_neg_delta_D_wrong": float(np.mean([r["neg_delta_D_wrong"] for r in sub])),
            "mean_delta_E": float(np.mean([r["delta_E"] for r in sub])),
            "corr_neg_dL_delta_E": _corr(
                [r["neg_delta_L"] for r in sub], [r["delta_E"] for r in sub]
            ),
            "corr_neg_dL_wrong_delta_E": _corr(
                [r["neg_delta_L_wrong"] for r in subw], [r["delta_E"] for r in subw]
            ),
            "corr_neg_dD_wrong_delta_E": _corr(
                [r["neg_delta_D_wrong"] for r in subw], [r["delta_E"] for r in subw]
            ),
            "P_signM_eq_S0": {
                k: float(np.mean([r["signM_eq_S0"][k] for r in sub])) for k in CHANNELS
            },
        }

    # B: flip vs tilde0 for converted (and others)
    strata: dict[str, Any] = {}
    for arm in arms:
        strata[arm] = {}
        for k in CHANNELS:
            pts = []
            for r in rows:
                if r["arm"] != arm:
                    continue
                i = IDX[k]
                if r["s0"][i] == r["mstar"][i]:
                    continue
                pts.append((float(r["tilde0"][k]), int(r["s1"][i] == r["mstar"][i])))
            if len(pts) < 2:
                strata[arm][k] = {"n_wrong": len(pts), "bins": []}
                continue
            xs = np.asarray([p[0] for p in pts], dtype=np.float64)
            ys = np.asarray([p[1] for p in pts], dtype=np.float64)
            # fixed bins by target-signed margin (distance to boundary)
            edges = [-np.inf, -2.0, 0.0, 2.0, np.inf]
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                if lo == -np.inf:
                    mask = xs <= hi
                    label = f"(-inf,{hi}]"
                elif hi == np.inf:
                    mask = xs > lo
                    label = f"({lo},inf)"
                else:
                    mask = (xs > lo) & (xs <= hi)
                    label = f"({lo},{hi}]"
                bins.append(
                    {
                        "bin": label,
                        "n": int(mask.sum()),
                        "P_W2C": float(ys[mask].mean()) if mask.any() else float("nan"),
                        "mean_tilde0": float(xs[mask].mean()) if mask.any() else float("nan"),
                    }
                )
            strata[arm][k] = {
                "n_wrong": len(pts),
                "mean_tilde0": float(xs.mean()),
                "P_W2C": float(ys.mean()),
                "bins": bins,
            }

    return {
        "transitions": transitions,
        "wrong_vs_correct_dM": wrong_vs_correct_dM,
        "surrogates": surrogates,
        "strata_W2C_by_tilde0": strata,
        "n_rows": len(rows),
    }


def run_calibration(sc, loaded, *, reps: int, seed: int) -> dict[str, Any]:
    """Free-run episodes: measure M_k (no steer) and observed S_k."""
    msgs = {k: messages_for_channel(sc, k) for k in CHANNELS}
    pairs: list[dict[str, Any]] = []
    for r in range(reps):
        s = seed + 17 * r
        torch.manual_seed(s)
        row = sc.run_episode(
            loaded,
            cls=NEUTRAL_CLS,
            task_id=NEUTRAL_TASK,
            seed=s,
            with_plan_format=True,
            capture_activations=False,
        )
        S = _S_from_row(row)
        M = {k: float(margin_at_site(loaded, msgs[k], SPECS[k])["M"]) for k in CHANNELS}
        pairs.append({"rep": r, "seed": s, "S": S, "M": M})
        print(
            f"  cal r={r} S={S} M={{C:{M['C']:+.2f},H:{M['H']:+.2f},O:{M['O']:+.2f}}}",
            flush=True,
        )

    cal: dict[str, Any] = {"n": len(pairs), "channels": {}}
    for k in CHANNELS:
        Ms = np.asarray([p["M"][k] for p in pairs], dtype=np.float64)
        Ss = np.asarray([p["S"][IDX[k]] for p in pairs], dtype=np.float64)
        # agreement with sign convention M>0 ⇒ bit=1
        pred = (Ms > 0).astype(np.float64)
        acc = float(np.mean(pred == Ss)) if len(Ss) else float("nan")
        # bins around 0
        edges = [-np.inf, -4, -2, -0.5, 0.5, 2, 4, np.inf]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo == -np.inf:
                mask = Ms <= hi
                label = f"(-inf,{hi}]"
            elif hi == np.inf:
                mask = Ms > lo
                label = f"({lo},inf)"
            else:
                mask = (Ms > lo) & (Ms <= hi)
                label = f"({lo},{hi}]"
            bins.append(
                {
                    "bin": label,
                    "n": int(mask.sum()),
                    "P_S1": float(Ss[mask].mean()) if mask.any() else float("nan"),
                    "mean_M": float(Ms[mask].mean()) if mask.any() else float("nan"),
                }
            )
        # logistic-ish: rate for M<0 vs M>0
        cal["channels"][k] = {
            "P_S1_given_M_neg": float(Ss[Ms < 0].mean()) if (Ms < 0).any() else float("nan"),
            "P_S1_given_M_pos": float(Ss[Ms > 0].mean()) if (Ms > 0).any() else float("nan"),
            "P_S1_given_M_near0": float(Ss[np.abs(Ms) <= 0.5].mean())
            if (np.abs(Ms) <= 0.5).any()
            else float("nan"),
            "sign_accuracy": acc,
            "mean_M": float(Ms.mean()),
            "mean_S": float(Ss.mean()),
            "corr_M_S": _corr(Ms.tolist(), Ss.tolist()),
            "bins": bins,
        }
    cal["pairs"] = pairs
    return cal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surrogate-json", type=Path, default=SURR)
    ap.add_argument("--cal-reps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    if not args.surrogate_json.is_file():
        raise SystemExit(f"missing {args.surrogate_json}; run surrogate first")

    surr = json.loads(args.surrogate_json.read_text())
    print("=== A/B/C from surrogate JSON ===", flush=True)
    abc = analyze_surrogate(surr)

    cal = None
    if not args.no_calibrate:
        from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
        from activation_pipeline.loader import load_model_and_tokenizer

        print(f"=== D calibration free-runs reps={args.cal_reps} ===", flush=True)
        sc = _load_sc()
        sc.ensure_sandbox()
        assert_model_fits_machine(LOCAL_MODEL_KEY)
        loaded = load_model_and_tokenizer(
            LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
        )
        cal = run_calibration(sc, loaded, reps=args.cal_reps, seed=args.seed)

    # Gate / interpretation
    tr_c = abc["transitions"].get("converted", {})
    dM_c = abc["wrong_vs_correct_dM"].get("converted", {})
    su_c = abc["surrogates"].get("converted", {})
    deepening = all(
        (dM_c.get(k, {}).get("mean_dtilde_correct0", 0) or 0)
        > (dM_c.get(k, {}).get("mean_dtilde_wrong0", 0) or 0) + 0.5
        for k in CHANNELS
        if dM_c.get(k, {}).get("n_correct0", 0) and dM_c.get(k, {}).get("n_wrong0", 0)
    )
    moves_wrong = all(
        (dM_c.get(k, {}).get("mean_dtilde_wrong0", 0) or 0)
        > (dM_c.get(k, {}).get("mean_dtilde_correct0", 0) or 0) + 0.5
        for k in CHANNELS
        if dM_c.get(k, {}).get("n_wrong0", 0)
    )
    mean_w2c = float(
        np.nanmean([tr_c[k]["P_W2C"] for k in CHANNELS if k in tr_c])
    ) if tr_c else float("nan")
    Lwrong_helps = bool(
        abs(su_c.get("corr_neg_dL_wrong_delta_E", 0) or 0) >= 0.3
        and (su_c.get("corr_neg_dL_wrong_delta_E") or 0)
        > (su_c.get("corr_neg_dL_delta_E") or -1) + 0.1
    )
    cal_ok = None
    if cal is not None:
        # calibrated if P(S=1|M>0) >> P(S=1|M<0) for each channel
        cal_ok = all(
            (cal["channels"][k]["P_S1_given_M_pos"] or 0)
            - (cal["channels"][k]["P_S1_given_M_neg"] or 1)
            >= 0.25
            for k in CHANNELS
        )

    gate = {
        "hypothesis": "margin improvement ≠ boundary crossing",
        "mean_P_W2C_converted": mean_w2c,
        "margin_deepening_on_already_correct": deepening,
        "margin_movement_concentrated_on_wrong_bits": moves_wrong,
        "L_wrong_predicts_delta_E_better": Lwrong_helps,
        "M_calibrated_to_S": cal_ok,
        "read": (
            "If wrong-bit Δ~M large but P(W→C) low → continuous move away from / through "
            "probe boundary without reliable discrete flip. "
            "If M not calibrated to S → M is mechanistic probe ≠ decision variable."
        ),
    }

    payload = {
        "protocol": "boundary-crossing + calibration",
        "source_surrogate": str(args.surrogate_json),
        "abc": abc,
        "calibration": {k: v for k, v in (cal or {}).items() if k != "pairs"}
        if cal
        else None,
        "calibration_pairs": (cal or {}).get("pairs") if cal else None,
        "gate": gate,
    }
    # keep pairs in separate lighter structure already in calibration_pairs
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Boundary-crossing diagnostic",
        "",
        "> Target-signed margin movement ⇏ behavioral boundary crossing.",
        "",
        f"Source: `{args.surrogate_json.name}` (frozen $v_c$).",
        "",
        "## A — Per-bit transitions (converted)",
        "",
        "| Channel | P(W→C) | P(C→W) | W2C | W2W | C2C | C2W |",
        "|---------|--------|--------|-----|-----|-----|-----|",
    ]
    for k in CHANNELS:
        t = tr_c.get(k, {})
        lines.append(
            f"| {k} | {t.get('P_W2C', float('nan')):.2f} | {t.get('P_C2W', float('nan')):.2f} | "
            f"{t.get('W2C', 0)} | {t.get('W2W', 0)} | {t.get('C2C', 0)} | {t.get('C2W', 0)} |"
        )

    lines += ["", "### All arms P(W→C)", "", "| Arm | C | H | O |", "|-----|---|---|---|"]
    for arm in ("none", "predictive", "converted", "random"):
        if arm not in abc["transitions"]:
            continue
        cells = [
            f"{abc['transitions'][arm][k]['P_W2C']:.2f}"
            if not np.isnan(abc["transitions"][arm][k]["P_W2C"])
            else "—"
            for k in CHANNELS
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Δ~M on wrong vs already-correct bits (converted)",
        "",
        "| Channel | Δ~M \\| wrong₀ | Δ~M \\| correct₀ |",
        "|---------|----------------|------------------|",
    ]
    for k in CHANNELS:
        d = dM_c.get(k, {})
        lines.append(
            f"| {k} | {d.get('mean_dtilde_wrong0', float('nan')):+.2f} (n={d.get('n_wrong0', 0)}) | "
            f"{d.get('mean_dtilde_correct0', float('nan')):+.2f} (n={d.get('n_correct0', 0)}) |"
        )

    lines += [
        "",
        "## C — $\\mathcal{L}_{8\\mathrm{way}}$ vs $\\mathcal{L}_{\\mathrm{wrong}}$ / $D_{\\mathrm{wrong}}$",
        "",
        "| Arm | E[−ΔL] | E[−ΔL_wrong] | E[−ΔD_wrong] | corr(−ΔL,ΔE) | corr(−ΔL_w,ΔE) | corr(−ΔD_w,ΔE) |",
        "|-----|--------|---------------|---------------|--------------|----------------|----------------|",
    ]
    for arm in ("none", "predictive", "converted", "random"):
        if arm not in abc["surrogates"]:
            continue
        s = abc["surrogates"][arm]
        lines.append(
            f"| {arm} | {s['mean_neg_delta_L']:+.3f} | {s['mean_neg_delta_L_wrong']:+.3f} | "
            f"{s['mean_neg_delta_D_wrong']:+.3f} | {s['corr_neg_dL_delta_E']:+.3f} | "
            f"{s['corr_neg_dL_wrong_delta_E']:+.3f} | {s['corr_neg_dD_wrong_delta_E']:+.3f} |"
        )

    lines += ["", "## B — P(W→C \\| $\\tilde M_0$) bins (converted)", ""]
    for k in CHANNELS:
        st = abc["strata_W2C_by_tilde0"].get("converted", {}).get(k, {})
        lines.append(f"### {k} (n_wrong={st.get('n_wrong', 0)}, P(W→C)={st.get('P_W2C', float('nan')):.2f})")
        lines.append("")
        lines.append("| $\\tilde M_0$ bin | n | P(W→C) |")
        lines.append("|------------------|---|--------|")
        for b in st.get("bins", []):
            lines.append(f"| {b['bin']} | {b['n']} | {b['P_W2C']:.2f} |" if b["n"] else f"| {b['bin']} | 0 | — |")
        lines.append("")

    if cal is not None:
        lines += [
            "## D — Calibration $P(S_k=1\\mid M_k)$ (free-run)",
            "",
            f"n={cal['n']}",
            "",
            "| Channel | P(S=1\\|M<0) | P(S=1\\|M>0) | sign accuracy | corr(M,S) |",
            "|---------|-------------|-------------|---------------|-----------|",
        ]
        for k in CHANNELS:
            c = cal["channels"][k]
            lines.append(
                f"| {k} | {c['P_S1_given_M_neg']:.2f} | {c['P_S1_given_M_pos']:.2f} | "
                f"{c['sign_accuracy']:.2f} | {c['corr_M_S']:+.2f} |"
            )
        lines.append("")
        for k in CHANNELS:
            lines.append(f"### {k} bins")
            lines.append("")
            lines.append("| M bin | n | P(S=1) |")
            lines.append("|-------|---|--------|")
            for b in cal["channels"][k]["bins"]:
                p = f"{b['P_S1']:.2f}" if b["n"] else "—"
                lines.append(f"| {b['bin']} | {b['n']} | {p} |")
            lines.append("")

    lines += [
        "## Gate",
        "",
        f"- mean P(W→C) converted: **{mean_w2c:.2f}**",
        f"- Δ~M concentrated on wrong bits: **{moves_wrong}**",
        f"- Δ~M deepening already-correct: **{deepening}**",
        f"- L_wrong predicts ΔE better: **{Lwrong_helps}**",
        f"- M calibrated to S (free-run): **{cal_ok}**",
        "",
        gate["read"],
        "",
        "No new $v$. No policy.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "md": str(MD), "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
