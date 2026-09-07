#!/usr/bin/env python3
"""Live Agent experiment protocol + log summarizer (no replay, no _apply_adapt).

Hooks append to data/results/sync_agent_live.jsonl on each Agent response/stop.

  # Start fresh log + print 8 Agent paste blocks
  .venv/bin/python scripts/run_sync_agent_live_record.py --init

  # After running chats in Agent, summarize observations
  .venv/bin/python scripts/run_sync_agent_live_record.py --summarize

Research claims supported: detection (1), measurement (2), policy repair (3).
NOT activation causal control — that requires Track 1 Qwen separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_eq import all_m_star_masks  # noqa: E402
from scripts.sync_cot_prompts import task_prompt  # noqa: E402
from scripts.sync_live_log import LOG, MD, clear_log, load_records  # noqa: E402

ARM = ROOT / "scripts" / "arm_sync_eq.py"


def _paste_blocks(*, style: str) -> list[dict]:
    blocks = []
    for m in all_m_star_masks():
        task = "api" if m[1] else "docs"
        blocks.append(
            {
                "m_star": m,
                "task": task,
                "prompt_style": style,
                "paste": (
                    f".venv/bin/python scripts/arm_sync_eq.py --m-star {m[0]},{m[1]},{m[2]}\n\n"
                    f"{task_prompt(task, style=style)}"
                ),
            }
        )
    return blocks


def cmd_init(*, style: str) -> int:
    clear_log()
    blocks = _paste_blocks(style=style)
    out = ROOT / "data" / "results" / "sync_agent_live_protocol.md"
    lines = [
        "# Live Agent protocol — 8 benchmark cells",
        "",
        f"**PLAN / CoT-proxy elicitation:** `{style}` "
        "(see `scripts/sync_cot_prompts.py`). "
        "Scores reported PLAN vs hooks — not internal CoT.",
        "",
        "Run each block in a **new Agent chat**. Hooks log to "
        f"`{LOG}` automatically.",
        "",
        "**No m* / hide-disclose instructions in the task.**",
        "",
    ]
    for i, b in enumerate(blocks, 1):
        lines += [
            f"## {i}. m*={tuple(b['m_star'])} ({b['task']})",
            "",
            "```text",
            b["paste"],
            "```",
            "",
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"cleared {LOG}")
    print(f"wrote {out} (prompt_style={style})")
    return 0


def cmd_summarize() -> int:
    recs = load_records()
    if not recs:
        print(f"No records in {LOG}. Run Agent chats after --init.")
        return 1

    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        key = json.dumps(r.get("m_star"))
        by_m[key].append(r)

    lines = [
        "# Live Agent observations",
        "",
        f"- Source: `{LOG}`",
        f"- Records: {len(recs)}",
        "- Claim: policy-followup repair (not activation control)",
        "",
    ]

    for key, items in sorted(by_m.items()):
        m = json.loads(key)
        lines.append(f"## m*={tuple(m)}")
        responses = [x for x in items if x.get("kind") == "after_response"]
        repairs = [x for x in items if x.get("kind") == "repair_issued"]
        synced = [x for x in items if x.get("kind") == "synced"]
        for i, r in enumerate(responses):
            lines += [
                f"### Response {i}",
                f"- tool_hook_this_turn: {r.get('tool_hook_this_turn')}",
                f"- S_raw: {r.get('S_raw')}",
                f"- e: {r.get('e')}",
                f"- plan: {str(r.get('plan', ''))[:200]}",
                f"- final: {str(r.get('final', ''))[:200]}",
                "",
            ]
        if repairs:
            lines.append(f"- repairs issued: {len(repairs)}")
            pre = next((x for x in responses if (x.get("e") or [0, 0, 0]) != [0, 0, 0]), None)
            post = synced[-1] if synced else next(
                (x for x in reversed(responses) if (x.get("e") or []) == [0, 0, 0]),
                None,
            )
            if pre and post:
                from scripts.sync_eq import delta_sync_error, sync_error_norm  # noqa: E402

                e_before = pre.get("e") or [0, 0, 0]
                e_after = post.get("e") or [0, 0, 0]
                dE = delta_sync_error(e_before, e_after)
                lines.append(
                    f"- post-repair: S_raw={post.get('S_raw')} e={e_after} "
                    f"ΔE_sync={dE} (||e||: {sync_error_norm(e_before)} → {sync_error_norm(e_after)})"
                )
        if synced and not repairs:
            lines.append(f"- synced: S_raw={synced[-1].get('S_raw')} e={synced[-1].get('e')}")
        elif synced and repairs:
            lines.append(f"- synced: yes")
        lines.append("")

    MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {MD} ({len(recs)} records)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", action="store_true", help="clear log + write 8 paste blocks")
    ap.add_argument(
        "--prompt-style",
        choices=("bare", "zero_shot", "few_shot"),
        default="zero_shot",
        help="PLAN/CoT-proxy elicitation style (default: zero_shot)",
    )
    ap.add_argument("--summarize", action="store_true", help="summarize live jsonl log")
    args = ap.parse_args()
    if args.init:
        return cmd_init(style=args.prompt_style)
    if args.summarize:
        return cmd_summarize()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
