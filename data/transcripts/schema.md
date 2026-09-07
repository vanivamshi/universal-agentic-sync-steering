# Transcript schema

Each line in `*.jsonl` is one transcript object.

## Required fields

| Field | Type | Description |
|---|---|---|
| `transcript_id` | string | Unique id, e.g. `coding_001` |
| `domain` | `"coding"` \| `"therapy"` \| `"writing"` | Domain stratum |
| `tool_format` | string | Locked: `qwen3_hermes` for primary collection |
| `tools` | array | JSON-schema-like tool definitions shown to the agent |
| `messages` | array | Chat messages (`system` / `user` / `assistant`) |

## Message content rules

- Assistant tool calls use Qwen3 Hermes delimiters exactly:
  - open: `<tool_call>`
  - body: single JSON object `{"name": "...", "arguments": {...}}`
  - close: `</tool_call>`
- Tool results appear as **user** messages wrapped in `<tool_response>...</tool_response>`.
- Assistant turns used for geometry should contain **both** some prose and at least one tool call when possible (same-turn prose→tool contrast).

## Optional `spans` field

If present, character offsets into that message's `content` (Unicode code points / Python string indices):

```json
"spans": [
  {
    "message_index": 2,
    "prose": [{"start": 0, "end": 42}],
    "tool_calls": [
      {
        "start": 43,
        "end": 120,
        "name": "read_file",
        "body_start": 54,
        "body_end": 109
      }
    ]
  }
]
```

- `tool_calls[].start` / `end`: full span including `<tool_call>` … `</tool_call>`
- `body_start` / `body_end`: interior only (activation **tool window** per preregistration — excludes delimiters)
- If `spans` omitted, derive via `scripts/validate_transcripts.py` (delimiter parse)
