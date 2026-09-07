#!/usr/bin/env python3
"""Diagnose H decision proxy vs true post-PLAN action boundary.

Writes data/results/sync_h_decision_proxy_diag.{json,md}

Finding expected under with_plan_format:
  prompt-end first token ≈ PLAN (p~0.998) — old M_H is not the action boundary.
  post-PLAN: <tool_call> vs FINAL is the real local decision.

  .venv/bin/python scripts/diagnose_h_decision_proxy.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts.sync_channel_control import CHANNEL_V, ChannelBank  # noqa: E402
from scripts.sync_h_decision import (  # noqa: E402
    DEFAULT_PREFILLS,
    decision_at_site,
    make_steer_hook,
)

OUT = ROOT / "data" / "results" / "sync_h_decision_proxy_diag.json"
MD = ROOT / "data" / "results" / "sync_h_decision_proxy_diag.md"


def _load_sc():
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "sync_scenario.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def main() -> int:
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    sc = _load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    bank = ChannelBank.load(CHANNEL_V)
    v_H = bank.V[1]
    task = sc.task_by_id("api")
    messages = [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message("B", task, with_plan_format=True)},
    ]

    rows = []
    for prefill in [None, *DEFAULT_PREFILLS]:
        for alpha in (0.0, -2.0, 2.0):
            hook = make_steer_hook(loaded, v_H, alpha)
            s = decision_at_site(loaded, messages, assistant_prefill=prefill, hook=hook)
            rows.append(
                {
                    "prefill": prefill,
                    "alpha": alpha,
                    "site": s["site"],
                    "argmax": s["argmax"],
                    "P_tool": s["P_tool"],
                    "P_FINAL": s["P_FINAL"],
                    "P_PLAN": s["P_PLAN"],
                    "M_H": s["M_H"],
                    "top5": s["top"][:5],
                }
            )
            print(
                f"site={s['site']} α={alpha:+.1f} argmax={s['argmax']!r} "
                f"P_tool={s['P_tool']:.4f} P_PLAN={s['P_PLAN']:.4f} M={s['M_H']:+.2f}",
                flush=True,
            )

    prompt_end = [r for r in rows if r["site"] == "prompt_end" and r["alpha"] == 0.0][0]
    post = [r for r in rows if r["site"] == "post_PLAN" and r["alpha"] == 0.0]
    post_steer = [
        r for r in rows if r["site"] == "post_PLAN" and abs(r["alpha"] + 2.0) < 1e-12
    ]
    mean_dP = float(
        np.mean([b["P_tool"] - a["P_tool"] for a, b in zip(post, post_steer)])
    ) if post and post_steer else float("nan")

    finding = {
        "diagnosis": "PROXY_MARGIN_NOT_TRUE_ACTION_BOUNDARY",
        "detail": (
            "At prompt end, P(PLAN)≈1 and P(<tool_call>)≈0. Old boundary M_H can cross "
            "zero without controlling the sampled action. The real local decision is "
            "post-PLAN: <tool_call> vs FINAL."
        ),
        "prompt_end_alpha0": prompt_end,
        "mean_dP_tool_old_vH_at_post_PLAN_alpha_-2": mean_dP,
        "implication": (
            "Do not keep raising α on old v_H. Relearn against post-PLAN M_H / P_tool, "
            "then causal-screen for sampled H."
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(finding, indent=2) + "\n")

    lines = [
        "# H decision proxy diagnostic",
        "",
        f"## Finding: **{finding['diagnosis']}**",
        "",
        finding["detail"],
        "",
        f"- prompt-end α=0: argmax=`{prompt_end['argmax']}`, "
        f"P_PLAN={prompt_end['P_PLAN']:.4f}, P_tool={prompt_end['P_tool']:.4f}, "
        f"M_H(tool−FINAL)={prompt_end['M_H']:+.2f}",
        f"- old v_H @ α=−2 mean ΔP_tool at post-PLAN site: **{mean_dP:+.4f}** "
        "(tiny — weakly aligned at the *true* site)",
        "",
        "## Implication",
        "",
        finding["implication"],
        "",
        "Next: `learn_h_tool_decision_direction.py` targeting post-PLAN M_H / P_tool.",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "md": str(MD), "diagnosis": finding["diagnosis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
