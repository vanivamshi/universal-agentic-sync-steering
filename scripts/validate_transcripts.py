#!/usr/bin/env python3
"""Validate agentic transcripts and attach tool/prose character spans."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


def find_tool_calls(content: str) -> list[dict]:
    spans = []
    for m in TOOL_CALL_RE.finditer(content):
        full_start, full_end = m.start(), m.end()
        body = m.group(1)
        body_start = m.start(1)
        body_end = m.end(1)
        name = ""
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                name = str(obj.get("name") or obj.get("tool") or obj.get("function") or "")
        except json.JSONDecodeError:
            name = ""
        spans.append(
            {
                "start": full_start,
                "end": full_end,
                "name": name,
                "body_start": body_start,
                "body_end": body_end,
            }
        )
    return spans


def prose_regions(content: str, tool_spans: list[dict]) -> list[dict]:
    if not tool_spans:
        text = content.strip()
        if not text:
            return []
        return [{"start": 0, "end": len(content)}]

    regions = []
    cursor = 0
    for t in sorted(tool_spans, key=lambda x: x["start"]):
        if t["start"] > cursor:
            chunk = content[cursor : t["start"]]
            if chunk.strip():
                regions.append({"start": cursor, "end": t["start"]})
        cursor = t["end"]
    if cursor < len(content) and content[cursor:].strip():
        regions.append({"start": cursor, "end": len(content)})
    return regions


def annotate_transcript(obj: dict) -> dict:
    required = ("transcript_id", "domain", "tool_format", "tools", "messages")
    for key in required:
        if key not in obj:
            raise ValueError(f"{obj.get('transcript_id', '?')}: missing field {key}")

    if obj["domain"] not in {"coding", "therapy", "writing"}:
        raise ValueError(f"{obj['transcript_id']}: bad domain {obj['domain']}")

    if obj["tool_format"] != "qwen3_hermes":
        raise ValueError(
            f"{obj['transcript_id']}: expected tool_format=qwen3_hermes, "
            f"got {obj['tool_format']}"
        )

    spans = []
    n_tool = 0
    n_prose_with_tool = 0
    for i, msg in enumerate(obj["messages"]):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        tools = find_tool_calls(content)
        prose = prose_regions(content, tools)
        if tools:
            n_tool += len(tools)
            if prose:
                n_prose_with_tool += 1
            for t in tools:
                if not t["name"]:
                    t["name"] = "unknown"
                # Interior window must be non-empty
                if t["body_end"] <= t["body_start"]:
                    raise ValueError(
                        f"{obj['transcript_id']} msg {i}: empty tool body window"
                    )
        spans.append(
            {
                "message_index": i,
                "prose": prose,
                "tool_calls": tools,
            }
        )

    if n_tool == 0:
        raise ValueError(f"{obj['transcript_id']}: no <tool_call> spans found")

    obj = dict(obj)
    obj["spans"] = spans
    obj["_stats"] = {
        "n_tool_calls": n_tool,
        "n_assistant_turns_with_prose_and_tool": n_prose_with_tool,
    }
    return obj


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("data/transcripts")],
        help="JSONL files or directories (default: data/transcripts)",
    )
    ap.add_argument(
        "--write-spans",
        action="store_true",
        help="Rewrite JSONL files with computed spans field",
    )
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        else:
            files.append(p)

    if not files:
        print("No JSONL files found.", file=sys.stderr)
        return 1

    total = 0
    by_domain: dict[str, int] = {}
    errors = 0

    for path in files:
        try:
            rows = load_jsonl(path)
            annotated = []
            for obj in rows:
                ann = annotate_transcript(obj)
                annotated.append(ann)
                total += 1
                by_domain[ann["domain"]] = by_domain.get(ann["domain"], 0) + 1
                print(
                    f"OK  {ann['transcript_id']:20s}  "
                    f"tools={ann['_stats']['n_tool_calls']}  "
                    f"prose+tool_turns={ann['_stats']['n_assistant_turns_with_prose_and_tool']}"
                )
            if args.write_spans:
                with path.open("w") as f:
                    for ann in annotated:
                        out = {k: v for k, v in ann.items() if not k.startswith("_")}
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        except ValueError as e:
            errors += 1
            print(f"FAIL {path}: {e}", file=sys.stderr)

    print("---")
    print(f"transcripts={total}  by_domain={by_domain}  errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
