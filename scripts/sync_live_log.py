"""Append-only live Agent observation log (hooks write; summarize reads)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "results" / "sync_agent_live.jsonl"
MD = ROOT / "data" / "results" / "sync_agent_live.md"


def append_record(record: dict) -> None:
    rec = dict(record)
    rec.setdefault("t", datetime.now(timezone.utc).isoformat())
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def clear_log() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")


def load_records() -> list[dict]:
    if not LOG.is_file():
        return []
    out: list[dict] = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
