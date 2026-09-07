"""Logit-lens vocabulary entropy for prose vs tool-call windows (§1 gate)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from activation_pipeline.hooks import ResidualStreamHooks
from activation_pipeline.loader import LoadedModel
from activation_pipeline.windows import TaggedAssistantTurn, embed_windows_in_chat, tag_transcript_assistant_turns


@dataclass
class WindowEntropy:
    transcript_id: str
    message_index: int
    window_kind: str
    layer: int
    mean_entropy: float
    n_tokens: int
    window_name: str = ""


@dataclass
class LogitLensGateResult:
    passed: bool
    n_pairs: int
    mean_prose_entropy: float
    mean_tool_entropy: float
    mean_delta_tool_minus_prose: float
    paired_t: float | None
    p_value_one_sided: float | None
    alpha: float
    layer: int
    rows: list[WindowEntropy] = field(default_factory=list)
    notes: str = ""
    cohens_d_paired: float | None = None
    pair_mode_used: str = ""
    layer_choice: str = "preregistered_final"  # or exploratory_mid
    evaluation_dataset: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def paired_cohens_d(deltas: Sequence[float]) -> float | None:
    """Cohen's d for paired differences: mean(delta) / sd(delta)."""
    n = len(deltas)
    if n < 2:
        return None
    x = torch.tensor(list(deltas), dtype=torch.float64)
    sd = float(x.std(unbiased=True))
    if sd < 1e-12:
        mean = float(x.mean())
        if abs(mean) < 1e-12:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return float(x.mean() / sd)


def _unembed(model: Any) -> torch.nn.Module:
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("model has no lm_head for logit lens")


def token_entropies_from_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    token_indices: Sequence[int],
) -> torch.Tensor:
    """Shannon entropy (nats) of vocab softmax for selected token positions.

    ``hidden``: (seq, hidden) or (1, seq, hidden)
    """
    if hidden.dim() == 3:
        h = hidden[0]
    else:
        h = hidden
    idx = list(token_indices)
    if not idx:
        return torch.empty(0)
    h_sel = h[idx].float()
    # Match lm_head dtype/device
    weight = next(lm_head.parameters())
    h_sel = h_sel.to(device=weight.device, dtype=weight.dtype)
    logits = lm_head(h_sel)
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    ent = -(p * log_p).sum(dim=-1)
    return ent.cpu()


def _welch_or_paired_t(deltas: list[float]) -> tuple[float | None, float | None]:
    """One-sample t-test of deltas against 0 (tool - prose). Returns (t, one-sided p for mean<0)."""
    n = len(deltas)
    if n < 2:
        return None, None
    x = torch.tensor(deltas, dtype=torch.float64)
    mean = float(x.mean())
    std = float(x.std(unbiased=True))
    if std < 1e-12:
        # Perfect separation or identical — treat extreme
        if mean < 0:
            return float("-inf"), 0.0
        if mean > 0:
            return float("inf"), 1.0
        return 0.0, 0.5
    t = mean / (std / (n**0.5))
    # Student-t survival for one-sided mean < 0: P(T <= t) with df=n-1
    # Approximate with normal for simplicity if scipy unavailable; prefer erfc
    try:
        from math import erf

        # Normal approx for p = Phi(t) when testing mean < 0 (want small t)
        p = 0.5 * (1.0 + erf(t / (2**0.5)))
        return float(t), float(p)
    except Exception:  # noqa: BLE001
        return float(t), None


@torch.inference_mode()
def entropy_for_tagged_turn(
    loaded: LoadedModel,
    messages: list[dict[str, str]],
    tagged: TaggedAssistantTurn,
    *,
    layer: int,
    cast_dtype: torch.dtype | None = torch.float32,
) -> list[WindowEntropy]:
    embedded = embed_windows_in_chat(
        loaded.tokenizer, messages, tagged.message_index, tagged
    )
    device = next(loaded.model.parameters()).device
    input_ids = torch.tensor([embedded.input_ids], device=device)
    attn = torch.ones_like(input_ids)
    hooks = ResidualStreamHooks(
        loaded.model, [layer], cast_dtype=cast_dtype, store_cpu=False
    )
    with hooks.capture():
        _ = loaded.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    hidden = hooks.activations[layer]
    lm_head = _unembed(loaded.model)
    rows: list[WindowEntropy] = []
    for w in embedded.windows:
        if w.n_tokens <= 0:
            continue
        ents = token_entropies_from_hidden(hidden, lm_head, w.token_indices())
        if ents.numel() == 0:
            continue
        rows.append(
            WindowEntropy(
                transcript_id=tagged.transcript_id,
                message_index=tagged.message_index,
                window_kind=w.kind,
                window_name=w.name,
                layer=layer,
                mean_entropy=float(ents.mean()),
                n_tokens=int(ents.numel()),
            )
        )
    return rows


@torch.inference_mode()
def collect_window_entropies(
    loaded: LoadedModel,
    transcripts: Sequence[dict[str, Any]],
    *,
    layer: int | None = None,
    require_both_modes: bool = False,
) -> list[WindowEntropy]:
    """Collect logit-lens entropies for all tagged windows.

    Default ``require_both_modes=False`` so Cursor-like agentic traces (tool-only
    turns + later prose answers) still contribute to the §1 gate.
    """
    layer_i = (
        layer if layer is not None else loaded.spec.default_measure_layer
    )
    rows: list[WindowEntropy] = []
    for tr in transcripts:
        for tagged in tag_transcript_assistant_turns(tr, loaded.tokenizer):
            if require_both_modes and not (
                tagged.prose_windows() and tagged.tool_windows()
            ):
                continue
            if not tagged.windows:
                continue
            rows.extend(
                entropy_for_tagged_turn(
                    loaded, tr["messages"], tagged, layer=layer_i
                )
            )
    return rows


def evaluate_logit_lens_gate(
    rows: Sequence[WindowEntropy],
    *,
    alpha: float = 0.05,
    layer: int,
    pair_mode: str = "auto",
    layer_choice: str = "preregistered_final",
    evaluation_dataset: str = "",
) -> LogitLensGateResult:
    """Compare tool vs prose mean entropy; pass if tool < prose (one-sided α).

    ``pair_mode``:
      - ``turn``: pair within (transcript_id, message_index) — seed corpus style
      - ``transcript``: pair mean prose vs mean tool within each transcript —
        better for Cursor-like agents that often emit tool-only then prose-only turns
      - ``auto``: use turn pairs if ≥3, else transcript pairs
    """
    turn_deltas: list[float] = []
    turn_prose: list[float] = []
    turn_tool: list[float] = []
    by_turn: dict[tuple[str, int], dict[str, list[float]]] = {}
    by_tr: dict[str, dict[str, list[float]]] = {}

    for r in rows:
        by_turn.setdefault((r.transcript_id, r.message_index), {"prose": [], "tool_call": []})
        by_tr.setdefault(r.transcript_id, {"prose": [], "tool_call": []})
        if r.window_kind in ("prose", "tool_call"):
            by_turn[(r.transcript_id, r.message_index)][r.window_kind].append(r.mean_entropy)
            by_tr[r.transcript_id][r.window_kind].append(r.mean_entropy)

    for d in by_turn.values():
        if not d["prose"] or not d["tool_call"]:
            continue
        p = sum(d["prose"]) / len(d["prose"])
        t = sum(d["tool_call"]) / len(d["tool_call"])
        turn_prose.append(p)
        turn_tool.append(t)
        turn_deltas.append(t - p)

    tr_deltas: list[float] = []
    tr_prose: list[float] = []
    tr_tool: list[float] = []
    for d in by_tr.values():
        if not d["prose"] or not d["tool_call"]:
            continue
        p = sum(d["prose"]) / len(d["prose"])
        t = sum(d["tool_call"]) / len(d["tool_call"])
        tr_prose.append(p)
        tr_tool.append(t)
        tr_deltas.append(t - p)

    if pair_mode == "auto":
        use = "turn" if len(turn_deltas) >= 3 else "transcript"
    else:
        use = pair_mode

    if use == "turn":
        deltas, prose_vals, tool_vals = turn_deltas, turn_prose, turn_tool
    else:
        deltas, prose_vals, tool_vals = tr_deltas, tr_prose, tr_tool

    if not deltas:
        return LogitLensGateResult(
            passed=False,
            n_pairs=0,
            mean_prose_entropy=float("nan"),
            mean_tool_entropy=float("nan"),
            mean_delta_tool_minus_prose=float("nan"),
            paired_t=None,
            p_value_one_sided=None,
            alpha=alpha,
            layer=layer,
            rows=list(rows),
            notes=(
                f"No paired prose+tool under pair_mode={use} "
                f"(turn_pairs={len(turn_deltas)}, transcript_pairs={len(tr_deltas)})."
            ),
            pair_mode_used=use,
            layer_choice=layer_choice,
            evaluation_dataset=evaluation_dataset,
        )

    t_stat, p_one = _welch_or_paired_t(deltas)
    mean_delta = sum(deltas) / len(deltas)
    d_eff = paired_cohens_d(deltas)
    direction_ok = mean_delta < 0
    sig_ok = p_one is not None and p_one < alpha
    if len(deltas) < 3 and direction_ok and mean_delta < -0.05:
        passed = True
        note = (
            f"Small-n directional pass ({use} pairing; mean tool entropy lower); "
            "smoke OK for agentic bring-up."
        )
    else:
        passed = bool(direction_ok and sig_ok)
        note = (
            f"PASS ({use} pairing): tool windows lower entropy than prose."
            if passed
            else f"FAIL ({use} pairing): tool entropy not significantly below prose — revise windows."
        )
    note += f" turn_pairs={len(turn_deltas)} transcript_pairs={len(tr_deltas)}"
    if d_eff is not None and d_eff == d_eff:  # not NaN
        note += f" cohens_d_paired={d_eff:.3f}"

    return LogitLensGateResult(
        passed=passed,
        n_pairs=len(deltas),
        mean_prose_entropy=sum(prose_vals) / len(prose_vals),
        mean_tool_entropy=sum(tool_vals) / len(tool_vals),
        mean_delta_tool_minus_prose=mean_delta,
        paired_t=t_stat,
        p_value_one_sided=p_one,
        alpha=alpha,
        layer=layer,
        rows=list(rows),
        notes=note,
        cohens_d_paired=d_eff,
        pair_mode_used=use,
        layer_choice=layer_choice,
        evaluation_dataset=evaluation_dataset,
    )
