"""Convert τ-bench historical trajectories → Hermes qwen3 transcript schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOMAIN_MAP = {
    "retail": "writing",  # customer-service / open-ended intermediate
    "airline": "writing",
}


def _hermes_tool_call(name: str, arguments: dict[str, Any] | str) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


def convert_traj(traj: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten OpenAI-style tool_calls into Hermes text + tool_response user msgs."""
    out: list[dict[str, str]] = []
    tools_meta: list[dict[str, Any]] = []
    seen_tools: set[str] = set()

    for msg in traj:
        role = msg.get("role")
        if role == "system":
            out.append({"role": "system", "content": msg.get("content") or ""})
            continue
        if role == "user":
            out.append({"role": "user", "content": msg.get("content") or ""})
            continue
        if role == "assistant":
            parts: list[str] = []
            text = (msg.get("content") or "").strip()
            if text:
                parts.append(text)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or "unknown"
                args = fn.get("arguments") or {}
                parts.append(_hermes_tool_call(name, args))
                if name not in seen_tools:
                    seen_tools.add(name)
                    tools_meta.append(
                        {
                            "name": name,
                            "description": f"τ-bench tool {name}",
                            "parameters": {},
                        }
                    )
            if parts:
                out.append({"role": "assistant", "content": "\n".join(parts)})
            continue
        if role == "tool":
            name = msg.get("name") or "tool"
            content = msg.get("content") or ""
            out.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n[{name}] {content}\n</tool_response>",
                }
            )
    return out, tools_meta


def convert_file(
    path: Path,
    *,
    domain_key: str,
    limit: int,
    require_tool: bool = True,
) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    domain = DOMAIN_MAP.get(domain_key, "writing")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if limit and len(out) >= limit:
            break
        traj = row.get("traj") or []
        messages, tools = convert_traj(traj)
        has_tool = any("<tool_call>" in (m.get("content") or "") for m in messages)
        if require_tool and not has_tool:
            continue
        out.append(
            {
                "transcript_id": f"tau_{domain_key}_{row.get('task_id', i)}_{row.get('trial', 0)}",
                "domain": domain,
                "tool_format": "qwen3_hermes",
                "source": "tau_bench_historical",
                "tools": tools,
                "messages": messages,
                "meta": {
                    "tau_domain": domain_key,
                    "task_id": row.get("task_id"),
                    "reward": row.get("reward"),
                    "trial": row.get("trial"),
                    "mapped_domain": domain,
                },
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--retail",
        type=Path,
        default=ROOT / "data" / "external" / "tau_gpt4o_retail.json",
    )
    ap.add_argument(
        "--airline",
        type=Path,
        default=ROOT / "data" / "external" / "tau_gpt4o_airline.json",
    )
    ap.add_argument("--limit-per-domain", type=int, default=15)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "transcripts" / "real" / "tau_bench.jsonl",
    )
    args = ap.parse_args()

    import importlib.util

    _vs = importlib.util.spec_from_file_location(
        "validate_transcripts", ROOT / "scripts" / "validate_transcripts.py"
    )
    assert _vs and _vs.loader
    mod = importlib.util.module_from_spec(_vs)
    _vs.loader.exec_module(mod)

    all_rows: list[dict[str, Any]] = []
    for path, key in [(args.retail, "retail"), (args.airline, "airline")]:
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        chunk = convert_file(path, domain_key=key, limit=args.limit_per_domain)
        print(f"{key}: converted {len(chunk)} trajectories from {path.name}")
        all_rows.extend(chunk)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("w") as f:
        for row in all_rows:
            try:
                ann = mod.annotate_transcript(row)
                ann = {k: v for k, v in ann.items() if not k.startswith("_")}
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")
                n_ok += 1
            except ValueError as e:
                print(f"WARN {row['transcript_id']}: {e}")

    print(f"wrote {args.out} n={n_ok}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
