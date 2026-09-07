#!/usr/bin/env python3
"""Collect real agentic transcripts via Cursor-like tool loop (Hermes <tool_call>)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Default: qwen3-0.6b on MacBook (MPS)")
    ap.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "data" / "agent_workspace",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "transcripts" / "agentic" / "coding.jsonl",
    )
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--device", default=None, help="Default: mps on MacBook")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--task-id", action="append", default=None, help="Filter transcript_id")
    args = ap.parse_args()

    import importlib.util

    from activation_pipeline.agent.loop import run_agent_task
    from activation_pipeline.agent.tasks import CODING_TASKS
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    _vs = importlib.util.spec_from_file_location(
        "validate_transcripts", ROOT / "scripts" / "validate_transcripts.py"
    )
    assert _vs and _vs.loader
    _mod = importlib.util.module_from_spec(_vs)
    _vs.loader.exec_module(_mod)
    annotate_transcript = _mod.annotate_transcript

    model_key = args.model or LOCAL_MODEL_KEY
    assert_model_fits_machine(model_key)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    print(f"agent model={model_key} device={device_map}")

    tasks = CODING_TASKS
    if args.task_id:
        want = set(args.task_id)
        tasks = [t for t in tasks if t.transcript_id in want]
    if not tasks:
        print("No tasks selected", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("w") as f:
        for task in tasks:
            print(f"=== {task.transcript_id}: {task.prompt[:80]}...")
            result = run_agent_task(
                loaded,
                transcript_id=task.transcript_id,
                domain=task.domain,
                task=task.prompt,
                workspace=args.workspace,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            tr = result.to_transcript()
            try:
                tr = annotate_transcript(tr)
                # drop underscore stats before write
                tr = {k: v for k, v in tr.items() if not k.startswith("_")}
            except ValueError as e:
                print(f"WARN annotate: {e}")
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
            n_tool = result.n_tool_calls
            print(
                f"  stop={result.stop_reason} tool_calls={n_tool} "
                f"messages={len(result.messages)}"
            )
            if n_tool > 0:
                n_ok += 1

    print(f"---\nwrote {args.out}  tasks={len(tasks)} with_tools={n_ok}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
