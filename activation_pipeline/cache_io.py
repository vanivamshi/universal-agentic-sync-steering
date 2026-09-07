"""Activation cache format and I/O for tagged windows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


CACHE_FORMAT_VERSION = "1.0"


@dataclass
class WindowActivationRecord:
    """Mean (and optional token-wise) residual stream for one tagged window."""

    transcript_id: str
    domain: str
    message_index: int
    window_kind: str  # prose | tool_call
    window_name: str
    token_start: int
    token_end: int
    n_tokens: int
    layer_means: dict[str, list[float]]  # layer -> mean vector (JSON-friendly)
    model_key: str
    hf_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def mean_tensor(self, layer: int) -> torch.Tensor:
        return torch.tensor(self.layer_means[str(layer)], dtype=torch.float32)


@dataclass
class ActivationCache:
    """Collection of window records + provenance."""

    format_version: str
    model_key: str
    hf_id: str
    layer_indices: list[int]
    records: list[WindowActivationRecord]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.format_version,
            "model_key": self.model_key,
            "hf_id": self.hf_id,
            "layer_indices": self.layer_indices,
            "meta": self.meta,
            "records": [asdict(r) for r in self.records],
        }
        # Sidecar tensors for faster reload (optional binary)
        pt_path = path.with_suffix(".pt")
        tensor_blob: dict[str, torch.Tensor] = {}
        for i, rec in enumerate(self.records):
            for layer_s, vec in rec.layer_means.items():
                tensor_blob[f"{i}:{layer_s}"] = torch.tensor(vec, dtype=torch.float32)

        path.write_text(json.dumps(payload, indent=2))
        torch.save(
            {
                "format_version": self.format_version,
                "layer_indices": self.layer_indices,
                "means": tensor_blob,
                "index": [
                    {
                        "transcript_id": r.transcript_id,
                        "domain": r.domain,
                        "message_index": r.message_index,
                        "window_kind": r.window_kind,
                        "window_name": r.window_name,
                        "token_start": r.token_start,
                        "token_end": r.token_end,
                        "n_tokens": r.n_tokens,
                        "model_key": r.model_key,
                        "hf_id": r.hf_id,
                        "meta": r.meta,
                    }
                    for r in self.records
                ],
                "meta": self.meta,
                "model_key": self.model_key,
                "hf_id": self.hf_id,
            },
            pt_path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ActivationCache":
        path = Path(path)
        pt_path = path.with_suffix(".pt") if path.suffix == ".json" else path
        json_path = path.with_suffix(".json") if path.suffix == ".pt" else path

        if pt_path.exists() and pt_path.suffix == ".pt":
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
            records = []
            for i, info in enumerate(blob["index"]):
                layer_means = {}
                for layer in blob["layer_indices"]:
                    key = f"{i}:{layer}"
                    if key in blob["means"]:
                        layer_means[str(layer)] = blob["means"][key].float().tolist()
                records.append(
                    WindowActivationRecord(
                        transcript_id=info["transcript_id"],
                        domain=info["domain"],
                        message_index=info["message_index"],
                        window_kind=info["window_kind"],
                        window_name=info.get("window_name", ""),
                        token_start=info["token_start"],
                        token_end=info["token_end"],
                        n_tokens=info["n_tokens"],
                        layer_means=layer_means,
                        model_key=info["model_key"],
                        hf_id=info["hf_id"],
                        meta=info.get("meta", {}),
                    )
                )
            return cls(
                format_version=blob.get("format_version", CACHE_FORMAT_VERSION),
                model_key=blob["model_key"],
                hf_id=blob["hf_id"],
                layer_indices=list(blob["layer_indices"]),
                records=records,
                meta=blob.get("meta", {}),
            )

        payload = json.loads(json_path.read_text())
        records = [WindowActivationRecord(**r) for r in payload["records"]]
        return cls(
            format_version=payload["format_version"],
            model_key=payload["model_key"],
            hf_id=payload["hf_id"],
            layer_indices=payload["layer_indices"],
            records=records,
            meta=payload.get("meta", {}),
        )


def pool_window(
    activations: dict[int, torch.Tensor],
    batch_index: int,
    token_start: int,
    token_end: int,
) -> dict[int, torch.Tensor]:
    """Mean-pool residual stream over [token_start, token_end) for one batch row."""
    out: dict[int, torch.Tensor] = {}
    for layer, act in activations.items():
        # act: (batch, seq, hidden)
        sl = act[batch_index, token_start:token_end, :]
        if sl.numel() == 0:
            raise ValueError(f"empty slice layer={layer} [{token_start}:{token_end}]")
        out[layer] = sl.float().mean(dim=0)
    return out
