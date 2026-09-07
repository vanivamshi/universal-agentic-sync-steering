"""Token-level prose / tool-call window tagging (preregistration §4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from transformers import PreTrainedTokenizerBase


@dataclass
class TokenSpan:
    """Inclusive token index range in a tokenized sequence [start, end)."""

    start: int
    end: int
    kind: str  # prose | tool_call
    name: str = ""
    char_start: int = 0
    char_end: int = 0

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"empty/inverted span {self.start}:{self.end}")

    @property
    def n_tokens(self) -> int:
        return self.end - self.start

    def token_indices(self) -> list[int]:
        return list(range(self.start, self.end))


@dataclass
class TaggedAssistantTurn:
    """Token windows for one assistant message (optionally embedded in a prompt)."""

    transcript_id: str
    domain: str
    message_index: int
    text: str
    input_ids: list[int]
    windows: list[TokenSpan] = field(default_factory=list)
    assistant_token_offset: int = 0  # where assistant content starts in full prompt
    meta: dict[str, Any] = field(default_factory=dict)

    def prose_windows(self) -> list[TokenSpan]:
        return [w for w in self.windows if w.kind == "prose"]

    def tool_windows(self) -> list[TokenSpan]:
        return [w for w in self.windows if w.kind == "tool_call"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _offsets_to_token_range(
    offset_mapping: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
    *,
    seq_offset: int = 0,
) -> tuple[int, int] | None:
    """Map a half-open char span to token indices that overlap it.

    Skips special tokens with (0, 0) offsets when they are not content.
    Returns None if no overlapping content tokens.
    """
    indices: list[int] = []
    for i, (a, b) in enumerate(offset_mapping):
        if a == b == 0 and i > 0:
            # Leading BOS often (0,0); trailing specials too — skip non-overlapping empties
            continue
        # overlap of [a,b) with [char_start, char_end)
        if b <= char_start or a >= char_end:
            continue
        indices.append(i)
    if not indices:
        return None
    return seq_offset + indices[0], seq_offset + indices[-1] + 1


def char_spans_to_token_windows(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    prose_chars: Sequence[dict[str, int]],
    tool_chars: Sequence[dict[str, Any]],
    *,
    transcript_id: str = "",
    domain: str = "",
    message_index: int = -1,
) -> TaggedAssistantTurn:
    """Tag windows from character spans on ``text`` alone (assistant content).

    Tool windows use ``body_start``/``body_end`` (delimiter-excluded) per preregistration.
    """
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    ids = list(encoded["input_ids"])
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    windows: list[TokenSpan] = []
    for p in prose_chars:
        rng = _offsets_to_token_range(offsets, int(p["start"]), int(p["end"]))
        if rng is None or rng[1] <= rng[0]:
            continue
        windows.append(
            TokenSpan(
                start=rng[0],
                end=rng[1],
                kind="prose",
                char_start=int(p["start"]),
                char_end=int(p["end"]),
            )
        )

    for t in tool_chars:
        body_s = int(t.get("body_start", t["start"]))
        body_e = int(t.get("body_end", t["end"]))
        rng = _offsets_to_token_range(offsets, body_s, body_e)
        if rng is None or rng[1] <= rng[0]:
            continue
        windows.append(
            TokenSpan(
                start=rng[0],
                end=rng[1],
                kind="tool_call",
                name=str(t.get("name", "")),
                char_start=body_s,
                char_end=body_e,
            )
        )

    windows.sort(key=lambda w: (w.start, w.end))
    return TaggedAssistantTurn(
        transcript_id=transcript_id,
        domain=domain,
        message_index=message_index,
        text=text,
        input_ids=ids,
        windows=windows,
        assistant_token_offset=0,
        meta={"tokenizer": getattr(tokenizer, "name_or_path", "")},
    )


def tag_transcript_assistant_turns(
    transcript: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> list[TaggedAssistantTurn]:
    """Tag every assistant turn that has character ``spans`` on the transcript."""
    span_by_msg = {s["message_index"]: s for s in transcript.get("spans", [])}
    out: list[TaggedAssistantTurn] = []
    for i, msg in enumerate(transcript["messages"]):
        if msg.get("role") != "assistant":
            continue
        span = span_by_msg.get(i)
        if span is None:
            continue
        tagged = char_spans_to_token_windows(
            tokenizer,
            msg["content"],
            span.get("prose", []),
            span.get("tool_calls", []),
            transcript_id=transcript["transcript_id"],
            domain=transcript["domain"],
            message_index=i,
        )
        out.append(tagged)
    return out


def embed_windows_in_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    message_index: int,
    tagged: TaggedAssistantTurn,
) -> TaggedAssistantTurn:
    """Shift token windows into a chat-templated prompt ending at ``message_index``.

    Locates the assistant content substring in the rendered chat string and
    remaps token indices via offset mapping on the full prompt.
    """
    prefix = messages[: message_index + 1]
    # Prefer chat template when available
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in prefix)

    assistant_text = messages[message_index]["content"]
    # Last occurrence — assistant turn is at the end of prefix
    pos = rendered.rfind(assistant_text)
    if pos < 0:
        raise ValueError(
            f"assistant text not found in chat template render "
            f"(transcript={tagged.transcript_id} msg={message_index})"
        )

    encoded = tokenizer(
        rendered,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    ids = list(encoded["input_ids"])
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    new_windows: list[TokenSpan] = []
    for w in tagged.windows:
        abs_s = pos + w.char_start
        abs_e = pos + w.char_end
        rng = _offsets_to_token_range(offsets, abs_s, abs_e)
        if rng is None or rng[1] <= rng[0]:
            continue
        new_windows.append(
            TokenSpan(
                start=rng[0],
                end=rng[1],
                kind=w.kind,
                name=w.name,
                char_start=w.char_start,
                char_end=w.char_end,
            )
        )

    # Approximate assistant content token start
    asst_rng = _offsets_to_token_range(offsets, pos, pos + len(assistant_text))
    asst_off = asst_rng[0] if asst_rng else 0

    return TaggedAssistantTurn(
        transcript_id=tagged.transcript_id,
        domain=tagged.domain,
        message_index=message_index,
        text=rendered,
        input_ids=ids,
        windows=new_windows,
        assistant_token_offset=asst_off,
        meta={
            **tagged.meta,
            "chat_embedded": True,
            "assistant_char_pos": pos,
        },
    )


def sanity_check_tagged(tagged: TaggedAssistantTurn) -> list[str]:
    """Return list of warning strings (empty if clean)."""
    warnings: list[str] = []
    n = len(tagged.input_ids)
    for w in tagged.windows:
        if w.start < 0 or w.end > n:
            warnings.append(f"span {w.kind} {w.start}:{w.end} outside seq len {n}")
        if w.n_tokens == 0:
            warnings.append(f"empty {w.kind} window")
    # tool and prose should not heavily overlap
    for i, a in enumerate(tagged.windows):
        for b in tagged.windows[i + 1 :]:
            if a.kind == b.kind:
                continue
            overlap = min(a.end, b.end) - max(a.start, b.start)
            if overlap > 0:
                warnings.append(
                    f"overlap {a.kind}[{a.start}:{a.end}] vs {b.kind}[{b.start}:{b.end}]"
                )
    if not tagged.tool_windows() and not tagged.prose_windows():
        warnings.append("no windows tagged")
    return warnings
