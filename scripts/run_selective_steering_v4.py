#!/usr/bin/env python3
"""Selective steering v4: decision/logit-level selective controllability.

Do NOT differentiate discrete sampled actions (v3 null). Differentiate the
continuous next-token decision margin that produces the action:

  m_T = logit(<tool_call>) - max(logit(no-tool prefixes))
  D_T(u) = [m_T(h+αu) - m_T(h-αu)] / (2α)
  D_C(u) = RMS of unrelated next-token logit derivatives
  S(u)  = |D_T(u)| / (λ + D_C(u))

Then: freeze u* = argmax S on train → held-out ±α → ΔP(tool).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

_v1 = importlib.util.spec_from_file_location("sel_v1", ROOT / "scripts" / "run_selective_steering.py")
assert _v1 and _v1.loader
sel_v1 = importlib.util.module_from_spec(_v1)
sys.modules[_v1.name] = sel_v1
_v1.loader.exec_module(sel_v1)

LAYER = sel_v1.LAYER
SEED = 20260827
N_REPS = 4
ALPHA = 0.25
LAMBDA = 0.05
N_THETA = 32  # dense sweep on Exp1 2D circle

TRAIN_TASKS = sel_v1.TRAIN_TASKS
TEST_TASKS = sel_v1.TEST_TASKS
WORKSPACE = sel_v1.WORKSPACE
DIR = sel_v1.DIR
NEUTRAL = sel_v1.NEUTRAL
TEMPERATURE = sel_v1.TEMPERATURE
MAX_TURNS = sel_v1.MAX_TURNS
MAX_NEW_TOKENS = sel_v1.MAX_NEW_TOKENS

OUT = ROOT / "data" / "results" / "selective_steering_v4.json"
MD = ROOT / "data" / "results" / "selective_steering_v4.md"
PROTO = ROOT / "docs" / "selective_steering_protocol.md"

# Locked after tokenizer probe (Qwen3 special tokens).
TOOL_TOKEN_STR = "<tool_call>"
NO_TOOL_STRS = ("The", "I", "Based", "Okay")


def _chat_prompt(tok: Any, messages: list[dict[str, str]]) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def _resolve_token_ids(tok: Any) -> dict[str, Any]:
    tool_ids = tok.encode(TOOL_TOKEN_STR, add_special_tokens=False)
    assert len(tool_ids) == 1, f"expected single <tool_call> id, got {tool_ids}"
    tool_id = int(tool_ids[0])
    notool: list[int] = []
    for s in NO_TOOL_STRS:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            notool.append(int(ids[0]))
    eos = tok.eos_token_id
    if eos is not None:
        notool.append(int(eos))
    notool = sorted(set(notool) - {tool_id})
    return {"tool_id": tool_id, "notool_ids": notool}


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = v.float().reshape(-1)
    n = torch.linalg.norm(v)
    if float(n) < 1e-12:
        raise ValueError("zero vector")
    return v / n


def _forward_logits(
    loaded: Any,
    *,
    messages: list[dict[str, str]],
    direction: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    """Return next-token logits at the generation prompt (last position)."""
    from activation_pipeline.steering import ActivationSteerHook

    tok = loaded.tokenizer
    prompt = _chat_prompt(tok, messages)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(loaded.model.parameters()).device
    ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)

    hook = None
    if direction is not None and abs(alpha) > 1e-12:
        hook = ActivationSteerHook(
            loaded.model,
            layer=LAYER,
            direction=direction,
            alpha=alpha,
            pos_mode="last",
            collect_stats=False,
        )
        hook.register()
    try:
        with torch.inference_mode():
            out = loaded.model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[0, -1, :].float().cpu()
    finally:
        if hook is not None:
            hook.remove()
    return logits


def _m_T(logits: torch.Tensor, tool_id: int, notool_ids: list[int]) -> float:
    l_tool = float(logits[tool_id])
    l_nt = float(logits[notool_ids].max())
    return l_tool - l_nt


def _d_from_pair(
    lp: torch.Tensor,
    lm: torch.Tensor,
    *,
    tool_id: int,
    notool_ids: list[int],
    alpha: float,
) -> dict[str, float]:
    """Centered finite difference of decision margins / unrelated logits."""
    dlogit = (lp - lm) / (2.0 * alpha)
    d_t = float(dlogit[tool_id] - dlogit[notool_ids].max())
    # Collateral: RMS of logit derivatives excluding tool + no-tool ids.
    mask = torch.ones(dlogit.numel(), dtype=torch.bool)
    mask[tool_id] = False
    for i in notool_ids:
        mask[i] = False
    resid = dlogit[mask]
    d_c = float(torch.sqrt((resid * resid).mean()))
    # Also report absolute tool / notool derivatives for diagnostics.
    return {
        "D_T": d_t,
        "D_C": d_c,
        "d_logit_tool": float(dlogit[tool_id]),
        "d_logit_notool_max": float(dlogit[notool_ids].max()),
        "S": abs(d_t) / (LAMBDA + d_c),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum())) + 1e-12
    return float((rx * ry).sum() / denom)


def main() -> int:
    from activation_pipeline.agent.loop import generate_assistant
    from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    dspec = importlib.util.spec_from_file_location(
        "run_gap_deception_collect", ROOT / "scripts" / "run_gap_deception_collect.py"
    )
    assert dspec and dspec.loader
    dcol = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(dcol)

    e21_spec = importlib.util.spec_from_file_location(
        "run_exp21_replicate_neg_u2", ROOT / "scripts" / "run_exp21_replicate_neg_u2.py"
    )
    assert e21_spec and e21_spec.loader
    e21 = importlib.util.module_from_spec(e21_spec)
    sys.modules[e21_spec.name] = e21
    e21_spec.loader.exec_module(e21)
    e21.generate_assistant = generate_assistant
    run_episode = e21.run_episode_with_steer

    drow = json.loads(DIR.read_text().splitlines()[0])
    U = torch.tensor(drow["basis"], dtype=torch.float32)
    u1 = _unit(U[0])
    u2 = _unit(U[1])
    k = int(U.shape[0])

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    tok_ids = _resolve_token_ids(loaded.tokenizer)
    tool_id = tok_ids["tool_id"]
    notool_ids = tok_ids["notool_ids"]
    print(f"tool_id={tool_id} notool_ids={notool_ids}", flush=True)

    system = sel_v1._system()
    known = {t["name"] for t in tools_for_prompt()}
    registry = ToolRegistry(WORKSPACE)
    rng = np.random.default_rng(SEED)

    # Candidate bank: dense circle in Exp1 plane + signed bases + random + orth.
    candidates: dict[str, torch.Tensor] = {}
    for i, th in enumerate(np.linspace(0.0, 2.0 * math.pi, N_THETA, endpoint=False)):
        c = math.cos(th) * u1 + math.sin(th) * u2
        candidates[f"theta_{i:02d}"] = _unit(c)
    candidates["u1"] = u1
    candidates["u2"] = u2
    candidates["neg_u1"] = -u1
    candidates["neg_u2"] = -u2
    r_full = _unit(torch.tensor(rng.standard_normal(u1.numel()), dtype=torch.float32))
    candidates["rand_full"] = r_full
    orth = r_full - float(torch.dot(r_full, u2)) * u2
    candidates["orth_u2"] = _unit(orth)

    print("=== Phase 1: decision-level S(u) on train ===", flush=True)
    per_dir: dict[str, dict[str, float]] = {}
    for did, vec in candidates.items():
        dts, dcs, ss, m_base = [], [], [], []
        for tid, task in TRAIN_TASKS:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{NEUTRAL}\n\nTask: {task}"},
            ]
            l0 = _forward_logits(loaded, messages=messages, direction=None, alpha=0.0)
            lp = _forward_logits(loaded, messages=messages, direction=vec, alpha=+ALPHA)
            lm = _forward_logits(loaded, messages=messages, direction=vec, alpha=-ALPHA)
            d = _d_from_pair(lp, lm, tool_id=tool_id, notool_ids=notool_ids, alpha=ALPHA)
            dts.append(d["D_T"])
            dcs.append(d["D_C"])
            ss.append(d["S"])
            m_base.append(_m_T(l0, tool_id, notool_ids))
        per_dir[did] = {
            "D_T": float(np.mean(dts)),
            "D_C": float(np.mean(dcs)),
            "S": float(np.mean(ss)),
            "m_T_baseline_mean": float(np.mean(m_base)),
            "D_T_abs": float(np.mean(np.abs(dts))),
        }
        print(
            f"  {did}: S={per_dir[did]['S']:.4f} D_T={per_dir[did]['D_T']:+.4f} "
            f"D_C={per_dir[did]['D_C']:.4f}",
            flush=True,
        )

    # u* = argmax S (equation defines the direction); freeze before test.
    best_id = max(per_dir, key=lambda d: per_dir[d]["S"])
    u_star = candidates[best_id]
    print(
        f"u* = {best_id}  S={per_dir[best_id]['S']:.4f}  "
        f"D_T={per_dir[best_id]['D_T']:+.4f}",
        flush=True,
    )

    # Frozen eval arms (held-out).
    arms: dict[str, torch.Tensor] = {
        "u_star": u_star,
        "u1": u1,
        "u2": u2,
        "rand_full": candidates["rand_full"],
        "orth_u2": candidates["orth_u2"],
    }

    print("=== Phase 2: held-out ±α steering → ΔP(tool) ===", flush=True)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        aid: {"baseline": [], "plus": [], "minus": []} for aid in arms
    }
    # Also record decision margins on test prompts (no generation) for ±α.
    margin_test: dict[str, dict[str, float]] = {}

    for aid, vec in arms.items():
        # Decision margins on test (continuous).
        dts, dcs = [], []
        for tid, task in TEST_TASKS:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{NEUTRAL}\n\nTask: {task}"},
            ]
            lp = _forward_logits(loaded, messages=messages, direction=vec, alpha=+ALPHA)
            lm = _forward_logits(loaded, messages=messages, direction=vec, alpha=-ALPHA)
            d = _d_from_pair(lp, lm, tool_id=tool_id, notool_ids=notool_ids, alpha=ALPHA)
            dts.append(d["D_T"])
            dcs.append(d["D_C"])
        margin_test[aid] = {
            "D_T": float(np.mean(dts)),
            "D_C": float(np.mean(dcs)),
            "S": abs(float(np.mean(dts))) / (LAMBDA + float(np.mean(dcs))),
            "bidirectional_decision": bool(float(np.mean(dts)) != 0.0),
        }

        for rep in range(N_REPS):
            for ti, (tid, task) in enumerate(TEST_TASKS):
                seed = int(SEED + 11003 * rep + 97 * ti + (abs(hash(aid)) % 997))
                common = dict(
                    loaded=loaded,
                    dcol=dcol,
                    registry=registry,
                    system=system,
                    known=known,
                    task=task,
                    seed=seed,
                )
                rb = run_episode(**common, direction=None, alpha=0.0)
                rp = run_episode(**common, direction=vec, alpha=+ALPHA)
                rm = run_episode(**common, direction=vec, alpha=-ALPHA)
                for arm, rrow in (("baseline", rb), ("plus", rp), ("minus", rm)):
                    rrow.update({"arm_id": aid, "arm": arm, "rep": rep, "task_id": tid})
                    rows[aid][arm].append(rrow)

        print(
            f"  {aid}: test D_T={margin_test[aid]['D_T']:+.4f} S={margin_test[aid]['S']:.4f}",
            flush=True,
        )

    def _behavior_score(aid: str) -> dict[str, float]:
        base = rows[aid]["baseline"]
        plus = rows[aid]["plus"]
        minus = rows[aid]["minus"]
        key = {(r["task_id"], r["rep"]): r for r in base}

        def paired(steer: list[dict[str, Any]], metric: str) -> np.ndarray:
            return np.array(
                [float(r[metric]) - float(key[(r["task_id"], r["rep"])][metric]) for r in steer],
                dtype=np.float64,
            )

        d_tool_p = paired(plus, "tool_emission")
        d_tool_m = paired(minus, "tool_emission")
        d_extra_p = paired(plus, "n_extra_paths")
        d_extra_m = paired(minus, "n_extra_paths")
        d_calls_p = paired(plus, "n_calls")
        d_calls_m = paired(minus, "n_calls")

        p_base = float(np.mean([r["tool_emission"] for r in base]))
        p_plus = float(np.mean([r["tool_emission"] for r in plus]))
        p_minus = float(np.mean([r["tool_emission"] for r in minus]))
        dP_plus = p_plus - p_base
        dP_minus = p_minus - p_base
        # Bidirectional: opposite signs on ΔP (or on Δextra as secondary).
        bidir = (dP_plus * dP_minus < 0) and (abs(dP_plus) > 1e-9 or abs(dP_minus) > 1e-9)
        return {
            "P_tool_base": p_base,
            "P_tool_plus": p_plus,
            "P_tool_minus": p_minus,
            "dP_plus": dP_plus,
            "dP_minus": dP_minus,
            "abs_dP_max": max(abs(dP_plus), abs(dP_minus)),
            "mean_d_extra_plus": float(d_extra_p.mean()),
            "mean_d_extra_minus": float(d_extra_m.mean()),
            "mean_d_calls_plus": float(d_calls_p.mean()),
            "mean_d_calls_minus": float(d_calls_m.mean()),
            "bidirectional_behavior": bidir,
            "mean_d_tool_plus": float(d_tool_p.mean()),
            "mean_d_tool_minus": float(d_tool_m.mean()),
        }

    scores = {aid: {**margin_test[aid], **_behavior_score(aid)} for aid in arms}

    # Prediction: train S → held-out |ΔP(tool)| for the frozen arms that also have train S.
    # For u_star use its train S; for named arms use their train S.
    pred_ids = ["u_star", "u1", "u2", "rand_full", "orth_u2"]
    train_S = []
    held_dP = []
    for aid in pred_ids:
        sid = best_id if aid == "u_star" else aid
        train_S.append(per_dir[sid]["S"])
        held_dP.append(scores[aid]["abs_dP_max"])
    rho = _spearman(np.array(train_S), np.array(held_dP))

    s_star = scores["u_star"]
    s_rand = scores["rand_full"]
    s_spread = float(np.std([per_dir[d]["S"] for d in per_dir]))
    decision_alive = s_spread > 1e-4 and abs(per_dir[best_id]["D_T"]) > 1e-3

    if (
        decision_alive
        and rho >= 0.5
        and s_star["abs_dP_max"] > s_rand["abs_dP_max"]
        and (
            s_star["bidirectional_behavior"]
            or (s_star["dP_plus"] * s_star["dP_minus"] < 0)
            or (margin_test["u_star"]["D_T"] != 0 and abs(margin_test["u_star"]["D_T"]) > abs(margin_test["rand_full"]["D_T"]))
        )
        and s_star["abs_dP_max"] > 0.05
    ):
        decision = "SELECTIVE_V4_HIT"
    elif decision_alive and (s_star["abs_dP_max"] > 0.05 or abs(per_dir[best_id]["D_T"]) > 0.1):
        decision = "SELECTIVE_V4_WEAK"
    elif not decision_alive:
        decision = "SELECTIVE_V4_NULL"  # continuous signal also flat
    else:
        decision = "SELECTIVE_V4_NULL"

    payload = {
        "protocol": str(PROTO),
        "experiment": "SELECTIVE_STEERING_V4",
        "decision": decision,
        "layer": LAYER,
        "alpha": ALPHA,
        "lambda": LAMBDA,
        "token_ids": tok_ids,
        "subspace_dim_k": k,
        "u_star_id": best_id,
        "train_S_spread": s_spread,
        "prediction": {
            "spearman_trainS_vs_held_abs_dP": rho,
            "train_S": {aid: (per_dir[best_id]["S"] if aid == "u_star" else per_dir[aid]["S"]) for aid in pred_ids},
            "held_abs_dP": {aid: scores[aid]["abs_dP_max"] for aid in pred_ids},
        },
        "per_direction_train": per_dir,
        "heldout_scores": scores,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Selective steering v4 — decision/logit controllability",
        "",
        f"- Decision: **`{decision}`**",
        f"- Protocol: `docs/selective_steering_protocol.md`",
        f"- m_T = logit(`{TOOL_TOKEN_STR}`) − max(logit(no-tool))",
        f"- u* = `{best_id}` (argmax train S); S_spread={s_spread:.4f}",
        f"- Spearman(train S → held |ΔP(tool)|) = **{rho:.3f}**",
        "",
        "## Train decision scores (selected)",
        "",
        "| direction | S | D_T | D_C |",
        "|---|---:|---:|---:|",
    ]
    show = sorted(
        ["u1", "u2", "neg_u1", "neg_u2", "rand_full", "orth_u2", best_id],
        key=lambda d: per_dir[d]["S"],
        reverse=True,
    )
    seen = set()
    for did in show:
        if did in seen:
            continue
        seen.add(did)
        p = per_dir[did]
        mark = " ← u*" if did == best_id else ""
        lines.append(
            f"| `{did}`{mark} | {p['S']:.4f} | {p['D_T']:+.4f} | {p['D_C']:.4f} |"
        )

    lines += [
        "",
        "## Held-out ±α (frozen)",
        "",
        "| arm | train S | D_T (test) | |ΔP(tool)| | ΔP(+) | ΔP(−) | bidir |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for aid in pred_ids:
        sid = best_id if aid == "u_star" else aid
        sc = scores[aid]
        lines.append(
            f"| `{aid}` | {per_dir[sid]['S']:.4f} | {sc['D_T']:+.4f} | "
            f"{sc['abs_dP_max']:.3f} | {sc['dP_plus']:+.3f} | {sc['dP_minus']:+.3f} | "
            f"{sc['bidirectional_behavior']} |"
        )
    lines += [
        "",
        "v3 failed because discrete `n_extra_paths` has no local gradient. "
        "v4 differentiates the **decision logits** that produce the action.",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "u_star": best_id,
                "rho": rho,
                "S_star": per_dir[best_id]["S"],
                "dP_star": scores["u_star"]["abs_dP_max"],
                "dP_rand": scores["rand_full"]["abs_dP_max"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
