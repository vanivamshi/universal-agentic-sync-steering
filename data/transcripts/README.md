# Agentic transcripts (seed corpus)

Domain-stratified multi-turn agent traces with **visible Qwen3 Hermes tool-call delimiters**, for window tagging and activation caching.

| File | Domain | Count |
|---|---|---|
| `coding.jsonl` | coding (low-drift baseline) | 4 |
| `therapy.jsonl` | therapy (high-drift) | 4 |
| `writing.jsonl` | open-ended writing (intermediate) | 4 |

Format lock: `tool_format = qwen3_hermes` (see `../../preregistration.md` and `schema.md`).

## Validate / refresh spans

```bash
python3 scripts/validate_transcripts.py --write-spans data/transcripts
```

Each assistant `<tool_call>` gets `body_start`/`body_end` (delimiter-excluded tool window) and flanking `prose` regions for same-turn contrasts.

## Expanding later

Add more JSONL lines with the same schema. Prefer turns that include **both** planning prose and a tool call. Keep tool JSON single-line objects inside `<tool_call>` … `</tool_call>` so the validator’s non-greedy parse stays reliable.
