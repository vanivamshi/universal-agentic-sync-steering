"""Safety direction extraction (Arditi-style mean-diff) + stability checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from activation_pipeline.hooks import ResidualStreamHooks
from activation_pipeline.loader import LoadedModel


@dataclass
class DirectionVector:
    direction_id: str
    kind: str
    mode: str  # prose | tool_call
    layer: int
    vector: list[float]
    n_pos: int
    n_neg: int
    meta: dict[str, Any] = field(default_factory=dict)

    def tensor(self) -> torch.Tensor:
        return torch.tensor(self.vector, dtype=torch.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + 1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(_unit(a.float()), _unit(b.float())))


@torch.inference_mode()
def last_token_residual(
    loaded: LoadedModel,
    messages: list[dict[str, str]],
    *,
    layers: Sequence[int],
    cast_dtype: torch.dtype | None = torch.float32,
) -> dict[int, torch.Tensor]:
    """Residual at the last prompt token for each layer (Arditi-style readout)."""
    tok = loaded.tokenizer
    if hasattr(tok, "apply_chat_template"):
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    enc = tok(rendered, return_tensors="pt", add_special_tokens=False)
    device = next(loaded.model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    hooks = ResidualStreamHooks(
        loaded.model, list(layers), cast_dtype=cast_dtype, store_cpu=True
    )
    with hooks.capture():
        _ = loaded.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    out: dict[int, torch.Tensor] = {}
    last = int(input_ids.shape[-1] - 1)
    for layer in layers:
        h = hooks.activations[layer]
        if h.dim() == 3:
            h = h[0]
        out[layer] = h[last].float().cpu()
    return out


def mean_difference_direction(
    pos_vecs: Sequence[torch.Tensor],
    neg_vecs: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not pos_vecs or not neg_vecs:
        raise ValueError("need positive and negative activations")
    pos = torch.stack([v.float().reshape(-1) for v in pos_vecs], dim=0).mean(0)
    neg = torch.stack([v.float().reshape(-1) for v in neg_vecs], dim=0).mean(0)
    return _unit(pos - neg)


def extract_mean_diff_from_pairs(
    loaded: LoadedModel,
    pairs: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    direction_id: str,
    kind: str,
    mode: str = "prose",
    system: str | None = None,
) -> list[DirectionVector]:
    """Extract per-layer mean-diff directions from prose contrast pairs.

    Each pair needs ``positive`` and ``negative`` user strings.
    """
    by_layer_pos: dict[int, list[torch.Tensor]] = {int(l): [] for l in layers}
    by_layer_neg: dict[int, list[torch.Tensor]] = {int(l): [] for l in layers}
    sys_msg = system or "You are a helpful assistant."

    for pair in pairs:
        for arm, key in (("pos", "positive"), ("neg", "negative")):
            text = pair[key]
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text},
            ]
            acts = last_token_residual(loaded, messages, layers=layers)
            bucket = by_layer_pos if arm == "pos" else by_layer_neg
            for layer, vec in acts.items():
                bucket[layer].append(vec)

    out: list[DirectionVector] = []
    for layer in layers:
        d = mean_difference_direction(by_layer_pos[layer], by_layer_neg[layer])
        out.append(
            DirectionVector(
                direction_id=f"{direction_id}_L{layer}",
                kind=kind,
                mode=mode,
                layer=int(layer),
                vector=d.tolist(),
                n_pos=len(by_layer_pos[layer]),
                n_neg=len(by_layer_neg[layer]),
                meta={"system": sys_msg},
            )
        )
    return out


def extract_tool_mean_diff(
    loaded: LoadedModel,
    tool_rows: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    direction_id: str,
    kind: str = "refusal",
) -> list[DirectionVector]:
    """Mean-diff on teacher-forced tool-call assistant turns (harmful vs benign args)."""
    by_layer_pos: dict[int, list[torch.Tensor]] = {int(l): [] for l in layers}
    by_layer_neg: dict[int, list[torch.Tensor]] = {int(l): [] for l in layers}

    for row in tool_rows:
        messages = [
            {
                "role": "system",
                "content": "You are an agent that uses tools. Prefer structured tool calls.",
            },
            {"role": "user", "content": row.get("user") or "Use the tool."},
            {"role": "assistant", "content": row["assistant_prefix"]},
        ]
        # Readout at last token of the forced tool-call assistant message
        tok = loaded.tokenizer
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = tok(rendered, return_tensors="pt", add_special_tokens=False)
        device = next(loaded.model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        hooks = ResidualStreamHooks(
            loaded.model, list(layers), cast_dtype=torch.float32, store_cpu=True
        )
        with torch.inference_mode(), hooks.capture():
            _ = loaded.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        last = int(input_ids.shape[-1] - 1)
        bucket = by_layer_pos if row.get("arm") == "positive" else by_layer_neg
        for layer in layers:
            h = hooks.activations[layer]
            if h.dim() == 3:
                h = h[0]
            bucket[int(layer)].append(h[last].float().cpu())

    out: list[DirectionVector] = []
    for layer in layers:
        d = mean_difference_direction(by_layer_pos[layer], by_layer_neg[layer])
        out.append(
            DirectionVector(
                direction_id=f"{direction_id}_tool_L{layer}",
                kind=kind,
                mode="tool_call",
                layer=int(layer),
                vector=d.tolist(),
                n_pos=len(by_layer_pos[layer]),
                n_neg=len(by_layer_neg[layer]),
                meta={"extraction": "teacher_forced_tool_call"},
            )
        )
    return out


@dataclass
class StabilityGateResult:
    kind: str
    passed: bool
    threshold: float
    layer_cosines: dict[str, float]
    mean_cosine: float
    band_layers: list[int]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stability_gate(
    prose_dirs: Sequence[DirectionVector],
    tool_dirs: Sequence[DirectionVector],
    *,
    kind: str,
    threshold: float = 0.70,
    band_layers: Sequence[int] | None = None,
) -> StabilityGateResult:
    """Cosine(prose_dir, tool_dir) per layer; pass if mean over band ≥ threshold."""
    prose_by = {d.layer: d for d in prose_dirs}
    tool_by = {d.layer: d for d in tool_dirs}
    layers = sorted(set(prose_by) & set(tool_by))
    if band_layers is not None:
        layers = [l for l in layers if l in set(band_layers)]
    cosines: dict[str, float] = {}
    vals: list[float] = []
    for layer in layers:
        c = cosine(prose_by[layer].tensor(), tool_by[layer].tensor())
        cosines[str(layer)] = c
        vals.append(c)
    mean_c = float(sum(vals) / len(vals)) if vals else float("nan")
    passed = bool(vals) and mean_c >= threshold
    note = (
        f"PASS: mean cos={mean_c:.3f} ≥ {threshold} over layers {layers}"
        if passed
        else (
            f"FAIL: mean cos={mean_c:.3f} < {threshold} over layers {layers}. "
            "Do not run refusal privilege arm; analyze instability instead."
            if vals
            else "FAIL: no overlapping layers between prose and tool directions."
        )
    )
    return StabilityGateResult(
        kind=kind,
        passed=passed,
        threshold=threshold,
        layer_cosines=cosines,
        mean_cosine=mean_c,
        band_layers=list(layers),
        notes=note,
    )


@dataclass
class SplitHalfStabilityResult:
    kind: str
    mode: str
    n_pairs: int
    n_splits: int
    layer_mean_cosine: dict[str, float]
    layer_std_cosine: dict[str, float]
    band_mean_cosine: float
    band_layers: list[int]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_half_prose_stability(
    loaded: LoadedModel,
    pairs: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int],
    kind: str = "refusal",
    n_splits: int = 20,
    seed: int = 0,
    system: str | None = None,
) -> SplitHalfStabilityResult:
    """Same-mode baseline: extract mean-diff on two random halves of prose pairs.

    Distinguishes "tool mode breaks refusal" from "model cannot hold the direction
    stably even within prose" (critical at small scales like 0.6B).
    """
    import random

    if len(pairs) < 4:
        raise ValueError("need ≥4 pairs for split-half")
    rng = random.Random(seed)
    per_layer: dict[int, list[float]] = {int(l): [] for l in layers}
    half = len(pairs) // 2

    for _ in range(n_splits):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        a, b = shuffled[:half], shuffled[half : 2 * half]
        dirs_a = extract_mean_diff_from_pairs(
            loaded,
            a,
            layers=layers,
            direction_id=f"{kind}_half_a",
            kind=kind,
            mode="prose",
            system=system,
        )
        dirs_b = extract_mean_diff_from_pairs(
            loaded,
            b,
            layers=layers,
            direction_id=f"{kind}_half_b",
            kind=kind,
            mode="prose",
            system=system,
        )
        by_a = {d.layer: d for d in dirs_a}
        by_b = {d.layer: d for d in dirs_b}
        for layer in layers:
            per_layer[int(layer)].append(
                cosine(by_a[int(layer)].tensor(), by_b[int(layer)].tensor())
            )

    mean_c: dict[str, float] = {}
    std_c: dict[str, float] = {}
    band_vals: list[float] = []
    for layer in layers:
        xs = per_layer[int(layer)]
        m = float(sum(xs) / len(xs))
        var = float(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
        mean_c[str(layer)] = m
        std_c[str(layer)] = var**0.5
        band_vals.append(m)
    band_mean = float(sum(band_vals) / len(band_vals)) if band_vals else float("nan")
    note = (
        f"prose↔prose split-half mean cos={band_mean:.3f} over layers {list(layers)} "
        f"({n_splits} splits, {len(pairs)} pairs). "
        "Compare to prose↔tool stability before attributing failures to tool mode."
    )
    return SplitHalfStabilityResult(
        kind=kind,
        mode="prose",
        n_pairs=len(pairs),
        n_splits=n_splits,
        layer_mean_cosine=mean_c,
        layer_std_cosine=std_c,
        band_mean_cosine=band_mean,
        band_layers=[int(l) for l in layers],
        notes=note,
    )


def save_directions(path: Path, directions: Sequence[DirectionVector]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(d.to_dict()) for d in directions) + "\n"
    )


def load_directions(path: Path) -> list[DirectionVector]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(DirectionVector(**d))
    return rows
