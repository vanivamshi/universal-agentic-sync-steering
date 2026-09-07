"""Build activation caches from tagged transcript windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .cache_io import (
    CACHE_FORMAT_VERSION,
    ActivationCache,
    WindowActivationRecord,
    pool_window,
)
from .collect import default_layer_indices
from .hooks import ResidualStreamHooks, resolve_decoder_layers
from .loader import LoadedModel
from .windows import (
    embed_windows_in_chat,
    sanity_check_tagged,
    tag_transcript_assistant_turns,
)


def load_transcripts(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.inference_mode()
def cache_transcript_windows(
    loaded: LoadedModel,
    transcripts: Sequence[dict[str, Any]],
    *,
    layer_indices: Sequence[int] | None = None,
    domains: Sequence[str] | None = None,
    require_both_modes: bool = True,
    max_length: int = 4096,
    cast_dtype: torch.dtype | None = torch.float32,
) -> ActivationCache:
    """Forward each relevant assistant turn; store mean activations per window."""
    layers = (
        list(layer_indices)
        if layer_indices is not None
        else default_layer_indices(loaded.spec)
    )
    n_live = len(resolve_decoder_layers(loaded.model))
    for idx in layers:
        if idx < 0 or idx >= n_live:
            raise IndexError(f"layer {idx} out of range ({n_live})")

    domain_set = set(domains) if domains else None
    records: list[WindowActivationRecord] = []
    first_param = next(loaded.model.parameters())
    device = first_param.device

    for tr in transcripts:
        if domain_set and tr["domain"] not in domain_set:
            continue
        tagged_list = tag_transcript_assistant_turns(tr, loaded.tokenizer)
        for tagged in tagged_list:
            if require_both_modes and not (tagged.prose_windows() and tagged.tool_windows()):
                continue
            warnings = sanity_check_tagged(tagged)
            embedded = embed_windows_in_chat(
                loaded.tokenizer,
                tr["messages"],
                tagged.message_index,
                tagged,
            )
            warn2 = sanity_check_tagged(embedded)
            all_warn = warnings + warn2
            if len(embedded.input_ids) > max_length:
                continue

            input_ids = torch.tensor([embedded.input_ids], device=device)
            attn = torch.ones_like(input_ids)

            hooks = ResidualStreamHooks(
                loaded.model,
                layers,
                cast_dtype=cast_dtype,
                store_cpu=True,
            )
            with hooks.capture():
                _ = loaded.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    use_cache=False,
                )

            for w in embedded.windows:
                pooled = pool_window(hooks.activations, 0, w.start, w.end)
                records.append(
                    WindowActivationRecord(
                        transcript_id=tr["transcript_id"],
                        domain=tr["domain"],
                        message_index=tagged.message_index,
                        window_kind=w.kind,
                        window_name=w.name,
                        token_start=w.start,
                        token_end=w.end,
                        n_tokens=w.n_tokens,
                        layer_means={
                            str(L): pooled[L].cpu().float().tolist() for L in layers
                        },
                        model_key=loaded.spec.key,
                        hf_id=loaded.spec.hf_id,
                        meta={"warnings": all_warn},
                    )
                )

    return ActivationCache(
        format_version=CACHE_FORMAT_VERSION,
        model_key=loaded.spec.key,
        hf_id=loaded.spec.hf_id,
        layer_indices=layers,
        records=records,
        meta={
            "n_transcripts": len(transcripts),
            "require_both_modes": require_both_modes,
            "n_live_layers": n_live,
        },
    )
