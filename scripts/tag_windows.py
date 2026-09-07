#!/usr/bin/env python3
"""Tag assistant turns at token level; write JSON summary + sanity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcripts",
        type=Path,
        default=ROOT / "data" / "transcripts",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Registry key for tokenizer (default: qwen3-0.6b on MacBook)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "windows" / "token_windows.jsonl",
    )
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine
    from activation_pipeline.registry import get_model_spec
    from activation_pipeline.windows import (
        embed_windows_in_chat,
        sanity_check_tagged,
        tag_transcript_assistant_turns,
    )

    model_key = args.model or LOCAL_MODEL_KEY
    assert_model_fits_machine(model_key)
    spec = get_model_spec(model_key)
    tok = AutoTokenizer.from_pretrained(
        spec.hf_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    transcripts = load_transcripts(args.transcripts)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_turns = 0
    n_prose = 0
    n_tool = 0
    n_warn = 0

    with args.out.open("w") as f:
        for tr in transcripts:
            for tagged in tag_transcript_assistant_turns(tr, tok):
                embedded = embed_windows_in_chat(
                    tok, tr["messages"], tagged.message_index, tagged
                )
                warns = sanity_check_tagged(tagged) + sanity_check_tagged(embedded)
                n_turns += 1
                n_prose += len(embedded.prose_windows())
                n_tool += len(embedded.tool_windows())
                n_warn += len(warns)
                row = embedded.to_dict()
                row["warnings"] = warns
                # keep payload smaller
                row.pop("input_ids", None)
                row["n_input_tokens"] = len(embedded.input_ids)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = "WARN" if warns else "OK"
                print(
                    f"{status} {tr['transcript_id']} msg={tagged.message_index} "
                    f"prose={len(embedded.prose_windows())} "
                    f"tool={len(embedded.tool_windows())} "
                    f"seq={len(embedded.input_ids)}"
                    + (f"  {warns}" if warns else "")
                )

    print(
        f"---\nwrote {args.out}  turns={n_turns} "
        f"prose_windows={n_prose} tool_windows={n_tool} warnings={n_warn}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
