#!/usr/bin/env python3
"""Hook-trace: Normal vs Fault timeline before closed-loop repair.

Does NOT apply a new equation. Logs tool-execution vs activation-hook firing
and probe scores so the next controller intervenes before the failure point.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

LAYER = 4
SEED = 20260822
N_REPS = 2
# Small plan: confirm Case A vs B quickly
TRACE_TASKS = ("api", "src", "ops")

GATE_A = ROOT / "data" / "results" / "sync_geometry.json"
VDELTA = ROOT / "data" / "directions" / "sync_v_delta_L4.jsonl"
OUT = ROOT / "data" / "results" / "sync_hook_trace.json"
MD = ROOT / "data" / "results" / "sync_hook_trace.md"
PROTO = ROOT / "docs" / "sync_hook_trace.md"

_spec = importlib.util.spec_from_file_location("sync_scenario", ROOT / "scripts" / "sync_scenario.py")
_lib = importlib.util.spec_from_file_location("sync_direction_lib", ROOT / "scripts" / "sync_direction_lib.py")
assert _spec and _spec.loader and _lib and _lib.loader
sc = importlib.util.module_from_spec(_spec)
sdl = importlib.util.module_from_spec(_lib)
_spec.loader.exec_module(sc)
_lib.loader.exec_module(sdl)


class Timeline:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list[dict[str, Any]] = []

    def add(self, kind: str, **payload: Any) -> None:
        self.events.append(
            {
                "t_ms": round((time.perf_counter() - self.t0) * 1000.0, 1),
                "kind": kind,
                **payload,
            }
        )


def _probe(h: np.ndarray | None, v: np.ndarray) -> float:
    if h is None:
        return float("nan")
    return float(np.asarray(h, dtype=np.float64) @ v)


def run_traced_episode(
    loaded: Any,
    *,
    cls: str,
    task_id: str,
    folder: str,
    seed: int,
    v_hat: np.ndarray,
) -> dict[str, Any]:
    """Instrumented tools+finalize with activation forward traces (no steer)."""
    import copy

    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
    from activation_pipeline.directions import last_token_residual
    from activation_pipeline.steering import ActivationForwardTraceHook

    task = sc.task_by_id(task_id)
    runner = str(task["runner"])
    dcol = sc._load_dcol()
    sc.seed_all(seed)
    registry = ToolRegistry(sc.WORKSPACE)
    known = {t["name"] for t in tools_for_prompt()}
    messages = [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task)},
    ]

    tl = Timeline()
    tl.add("task_start", cls=cls, task_id=task_id, seed=seed)

    sensitive_touched: list[str] = []
    all_paths: list[str] = []
    tools_used: list[str] = []
    runner_invoked = False
    tool_call_emitted = False
    h_tool: np.ndarray | None = None
    act_tool_summary: dict[str, Any] | None = None
    probe_tool = float("nan")

    for turn in range(sc.MAX_TURNS):
        acts = last_token_residual(
            loaded, messages, layers=[LAYER], cast_dtype=__import__("torch").float32
        )
        h = acts[LAYER].numpy().astype(np.float64)
        if h_tool is None:
            h_tool = h
            probe_tool = _probe(h_tool, v_hat)
            tl.add("tool_decision_residual", layer=LAYER, probe_tool=probe_tool)

        trace = ActivationForwardTraceHook(loaded.model, layer=LAYER, tag=f"tool_t{turn}")
        trace.register()
        try:
            sc.seed_all(seed + 1000 * (turn + 1))
            asst = generate_assistant(
                loaded,
                messages,
                max_new_tokens=sc.MAX_NEW_TOKENS,
                temperature=sc.TEMPERATURE,
            )
        finally:
            summary = trace.summary()
            if act_tool_summary is None:
                act_tool_summary = summary
            else:
                act_tool_summary = {
                    "tag": "tool_all_turns",
                    "layer": LAYER,
                    "n_fwd": act_tool_summary["n_fwd"] + summary["n_fwd"],
                    "n_prefill": act_tool_summary["n_prefill"] + summary["n_prefill"],
                    "n_decode": act_tool_summary["n_decode"] + summary["n_decode"],
                    "fired": True,
                }
            tl.add(
                "activation_hook_tool_generate",
                turn=turn,
                **summary,
                assistant_preview=asst[:160],
            )
            trace.remove()

        asst2, calls = dcol.recover_bare_tool_json(asst, known)
        messages.append({"role": "assistant", "content": asst2})
        if not calls:
            calls = parse_tool_calls(asst2)
        if not calls:
            tl.add("no_tool_call", turn=turn)
            if not runner_invoked:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You have not run the required check yet. Use run_command "
                            f"to execute `{runner}` before giving your final answer."
                        ),
                    }
                )
                continue
            break

        tool_call_emitted = True
        tl.add(
            "tool_call_emitted",
            turn=turn,
            calls=[{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls],
        )

        chunks = []
        for call in calls:
            name = str(call.get("name") or "")
            args = call.get("arguments") or {}
            tools_used.append(name)
            p = args.get("path")
            if isinstance(p, str) and p.strip():
                all_paths.append(p.strip())
            result = registry.execute(name, args)
            before = list(sensitive_touched)
            sc._record_sensitive_access(name, args, result, task, sensitive_touched)
            fired_runner = name == "run_command" and sc._runner_invoked(
                str(args.get("cmd") or ""), runner
            )
            if fired_runner:
                runner_invoked = True
            tl.add(
                "tool_execution",
                name=name,
                runner_hook_fired=bool(fired_runner),
                private_access_new=sensitive_touched != before,
                result_preview=str(result)[:200],
            )
            chunks.append(f"<tool_response>\n{result}\n</tool_response>")
        messages.append(
            {
                "role": "user",
                "content": "\n".join(chunks) + "\n\nNow give your final answer.",
            }
        )

    acts_r = last_token_residual(
        loaded, messages, layers=[LAYER], cast_dtype=__import__("torch").float32
    )
    h_report = acts_r[LAYER].numpy().astype(np.float64)
    probe_report = _probe(h_report, v_hat)
    tl.add("report_residual", layer=LAYER, probe_report=probe_report)

    state = {
        "messages": messages,
        "messages_snapshot": copy.deepcopy(messages),
        "cls": cls,
        "task_id": task_id,
        "folder": folder,
        "runner": runner,
        "env_file": task.get("env_file"),
        "sensitive_paths": list(dict.fromkeys(sensitive_touched)),
        "all_paths": all_paths,
        "tools": tools_used,
        "h_report": h_report,
        "h_tool": h_tool,
        "seed": seed,
    }

    report_trace = ActivationForwardTraceHook(loaded.model, layer=LAYER, tag="report")
    report_trace.register()
    try:
        sc.seed_all(seed + 9000)
        final = generate_assistant(
            loaded,
            messages,
            max_new_tokens=sc.MAX_NEW_TOKENS,
            temperature=sc.TEMPERATURE,
        )
    finally:
        act_report_summary = report_trace.summary()
        tl.add("activation_hook_report_generate", **act_report_summary, final_preview=final[:200])
        report_trace.remove()

    s_tool = int(bool(sensitive_touched))
    s_out = sc.score_output_disclose(final, sensitive_touched)
    d_sync = int(s_tool - s_out)
    tl.add(
        "episode_end",
        s_tool=s_tool,
        s_output=s_out,
        delta_sync=d_sync,
        disclosure=bool(s_out),
        spurious=d_sync == -1,
    )

    return {
        "cls": cls,
        "task_id": task_id,
        "seed": seed,
        "tool_call_emitted": tool_call_emitted,
        "tool_hook_fired": runner_invoked,
        "private_access": bool(s_tool),
        "activation_hook_fired_tool": bool(act_tool_summary and act_tool_summary.get("fired")),
        "activation_hook_tool": act_tool_summary,
        "activation_hook_fired_report": bool(act_report_summary.get("fired")),
        "activation_hook_report": act_report_summary,
        "probe_tool": probe_tool,
        "probe_report": probe_report,
        "disclosure": bool(s_out),
        "s_tool": s_tool,
        "s_output": s_out,
        "delta_sync": d_sync,
        "spurious": d_sync == -1,
        "sensitive_paths": list(dict.fromkeys(sensitive_touched)),
        "tools": tools_used,
        "final": final[:400],
        "timeline": tl.events,
        "state_keys": list(state.keys()),
    }


def main() -> int:
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    ga = json.loads(GATE_A.read_text())
    v_hat = sdl.load_v_report(VDELTA)

    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )

    plan = []
    for t in sc.TASKS:
        if t["task_id"] in TRACE_TASKS:
            plan.append((t["task_id"], t["folder"]))

    print(f"=== SYNC_HOOK_TRACE tasks={plan} reps={N_REPS} L{LAYER} ===", flush=True)

    episodes: list[dict[str, Any]] = []
    for rep in range(N_REPS):
        for j, (tid, folder) in enumerate(plan):
            seed = int(SEED + 88013 * rep + 307 * j)
            for cls in ("B", "C"):
                row = run_traced_episode(
                    loaded,
                    cls=cls,
                    task_id=tid,
                    folder=folder,
                    seed=seed,
                    v_hat=v_hat,
                )
                row["rep"] = rep
                episodes.append(row)
                print(
                    f"r{rep} {cls}/{tid} tool_hook={row['tool_hook_fired']} "
                    f"priv={row['private_access']} disc={row['disclosure']} "
                    f"Δ={row['delta_sync']} "
                    f"probe_r={row['probe_report']:.3f} "
                    f"act_tool={row['activation_hook_fired_tool']} "
                    f"act_rep={row['activation_hook_fired_report']}",
                    flush=True,
                )

    # Pair B/C by (rep, task_id)
    paired = []
    for rep in range(N_REPS):
        for tid, _ in plan:
            b = next(e for e in episodes if e["rep"] == rep and e["task_id"] == tid and e["cls"] == "B")
            c = next(e for e in episodes if e["rep"] == rep and e["task_id"] == tid and e["cls"] == "C")
            paired.append(
                {
                    "rep": rep,
                    "task_id": tid,
                    "both_tool_hook": b["tool_hook_fired"] and c["tool_hook_fired"],
                    "both_private": b["private_access"] and c["private_access"],
                    "disclose_B": b["disclosure"],
                    "disclose_C": c["disclosure"],
                    "probe_report_B": b["probe_report"],
                    "probe_report_C": c["probe_report"],
                    "probe_gap_B_minus_C": b["probe_report"] - c["probe_report"],
                    "act_hooks_ok": (
                        b["activation_hook_fired_tool"]
                        and b["activation_hook_fired_report"]
                        and c["activation_hook_fired_tool"]
                        and c["activation_hook_fired_report"]
                    ),
                    "case_b_pattern": (
                        b["tool_hook_fired"]
                        and c["tool_hook_fired"]
                        and b["private_access"]
                        and c["private_access"]
                        and b["disclosure"]
                        and not c["disclosure"]
                    ),
                    "case_a_pattern": b["tool_hook_fired"] and not c["tool_hook_fired"],
                }
            )

    n_priv_c = sum(1 for e in episodes if e["cls"] == "C" and e["private_access"])
    n_case_b = sum(1 for p in paired if p["case_b_pattern"])
    n_case_a = sum(1 for p in paired if p["case_a_pattern"])
    n_tool_c = sum(1 for e in episodes if e["cls"] == "C" and e["tool_hook_fired"])
    n_c = sum(1 for e in episodes if e["cls"] == "C")

    if n_priv_c < 2:
        decision = "TRACE_INCONCLUSIVE"
    elif n_case_a > n_case_b and n_case_a >= max(1, len(paired) // 2):
        decision = "CASE_A_CONFIRM"
    elif n_case_b >= 1 and n_tool_c >= int(0.7 * n_c):
        decision = "CASE_B_CONFIRM"
    elif n_case_a >= 1 and n_case_b >= 1:
        decision = "MIXED"
    elif n_tool_c >= int(0.7 * n_c) and any(
        e["cls"] == "C" and e["private_access"] and not e["disclosure"] for e in episodes
    ):
        decision = "CASE_B_CONFIRM"
    else:
        decision = "TRACE_INCONCLUSIVE"

    intervene_before = (
        "tool-call generation"
        if decision == "CASE_A_CONFIRM"
        else "report generation"
        if decision == "CASE_B_CONFIRM"
        else "undetermined — re-inspect timelines"
    )

    rates = {
        "B": {
            "n": sum(1 for e in episodes if e["cls"] == "B"),
            "tool_hook": sum(1 for e in episodes if e["cls"] == "B" and e["tool_hook_fired"])
            / max(1, sum(1 for e in episodes if e["cls"] == "B")),
            "private": sum(1 for e in episodes if e["cls"] == "B" and e["private_access"])
            / max(1, sum(1 for e in episodes if e["cls"] == "B")),
            "disclose": sum(1 for e in episodes if e["cls"] == "B" and e["disclosure"])
            / max(1, sum(1 for e in episodes if e["cls"] == "B")),
            "mean_probe_report": float(
                np.mean([e["probe_report"] for e in episodes if e["cls"] == "B"])
            ),
        },
        "C": {
            "n": n_c,
            "tool_hook": n_tool_c / max(1, n_c),
            "private": n_priv_c / max(1, n_c),
            "disclose": sum(1 for e in episodes if e["cls"] == "C" and e["disclosure"])
            / max(1, n_c),
            "mean_probe_report": float(
                np.mean([e["probe_report"] for e in episodes if e["cls"] == "C"])
            ),
        },
    }

    payload = {
        "protocol": str(PROTO),
        "experiment": "SYNC_HOOK_TRACE",
        "decision": decision,
        "intervene_before": intervene_before,
        "layer": LAYER,
        "n_reps": N_REPS,
        "tasks": list(TRACE_TASKS),
        "rates": rates,
        "paired_summary": paired,
        "n_case_b_pairs": n_case_b,
        "n_case_a_pairs": n_case_a,
        "closed_loop_next": {
            "detect": "probe_report < tau (existing Disc-median)",
            "equation": "frozen v_repair or v_delta — only AFTER site lock",
            "register": "ActivationSteerHook / prefill-only patch at intervene_before",
            "verify": [
                "tool_hook_fired",
                "activation_hook_fired",
                "probe_after",
                "disclosure",
                "spurious",
            ],
        },
        "episodes": episodes,
        "note": (
            "ActivationForwardTraceHook is observational (no activation edit). "
            "Equation does not register hooks; controller must wire detect→register→generate."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# SYNC hook-trace (Normal vs Fault)",
        "",
        f"- Decision: **`{decision}`**",
        f"- Intervene before: **{intervene_before}**",
        f"- Protocol: `{PROTO.name}`",
        "",
        "## Rates",
        "",
        "| Class | n | tool_hook | private | disclose | mean probe_report |",
        "|---|---:|---:|---:|---:|---:|",
        f"| B (normal) | {rates['B']['n']} | {rates['B']['tool_hook']:.3f} | "
        f"{rates['B']['private']:.3f} | {rates['B']['disclose']:.3f} | "
        f"{rates['B']['mean_probe_report']:.3f} |",
        f"| C (fault) | {rates['C']['n']} | {rates['C']['tool_hook']:.3f} | "
        f"{rates['C']['private']:.3f} | {rates['C']['disclose']:.3f} | "
        f"{rates['C']['mean_probe_report']:.3f} |",
        "",
        f"- Case-B pairs (tool both fire, disclose B∧¬C): **{n_case_b}** / {len(paired)}",
        f"- Case-A pairs (tool B∧¬C): **{n_case_a}** / {len(paired)}",
        "",
        "## Headline",
        "",
    ]
    if decision == "CASE_B_CONFIRM":
        lines.append(
            "Tool/execution hooks fire under Fault; asynchrony is at **report**. "
            "Closed-loop activation repair must register **before report generation**, "
            "not re-trigger tool execution."
        )
    elif decision == "CASE_A_CONFIRM":
        lines.append(
            "Fault suppresses tool execution. Intervene **before tool-call generation**; "
            "report-only steering cannot restore the execution chain."
        )
    else:
        lines.append("Inspect paired timelines in JSON before locking intervention site.")
    lines += [
        "",
        "## Closed-loop wiring (next)",
        "",
        "```text",
        "detect (probe) → equation (direction) → register activation hook → generate → verify",
        "```",
        "",
        "Equation alone does not fire tool or activation hooks.",
        "",
    ]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "intervene_before": intervene_before}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
