#!/usr/bin/env python3
"""Channel decision margins M_k and gradients g_k for Phase 4 conversion.

C → H → O temporal sites (agent causal timeline):

  C: mid-PLAN intent after ``PLAN: I will ``
     M_C = logit(run) − logit(only)     # access vs public-only plan

  H: post-PLAN tool selection (Phase 3)
     M_H = logit(<tool_call>) − logit(FINAL)

  O: post-tool FINAL disclosure
     M_O = logit(During) − logit(docs)  # disclose vs public-summary opener

Conversion:
  v_c = sign(v_pᵀ g) · Proj_g(v_p) / ‖Proj_g(v_p)‖
      = unit(g)   (algebraically, once Proj is nonzero)

Phase 3 freeze: H conversion supported mechanistically.
Phase 4: test whether C and O convert the same way before composing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from scripts.sync_h_decision import (
    LAYER_DEFAULT,
    PLAN_PUBLIC,
    PLAN_RUN,
    PLAN_SMOKE,
    chat_prompt,
    make_steer_hook,
    unit,
)


@dataclass(frozen=True)
class ChannelMarginSpec:
    channel: str  # C | H | O
    site: str
    prefill: str
    pos_token: str  # increases M_k (toward channel=1 side of margin)
    neg_token: str
    n_grad_contexts: int = 4


SPECS: dict[str, ChannelMarginSpec] = {
    "C": ChannelMarginSpec(
        channel="C",
        site="mid_PLAN_intent",
        prefill="PLAN: I will ",
        pos_token="run",
        neg_token="only",
    ),
    "H": ChannelMarginSpec(
        channel="H",
        site="post_PLAN_tool",
        prefill=PLAN_RUN,
        pos_token="<tool_call>",
        neg_token="FINAL",
    ),
    "O": ChannelMarginSpec(
        channel="O",
        site="post_tool_FINAL",
        prefill="FINAL: ",
        pos_token="During",
        neg_token="docs",
    ),
}

# Prefill variants for averaging g_H (same as Phase 3)
H_PREFILLS = (PLAN_RUN, PLAN_PUBLIC, PLAN_SMOKE)
C_PREFILLS = (
    "PLAN: I will ",
    "PLAN: I will ",
    "PLAN: Next I will ",
)


def token_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty tokenize for {text!r}")
    return int(ids[0])


def messages_base(sc, *, cls: str = "B", task_id: str = "api") -> list[dict]:
    task = sc.task_by_id(task_id)
    return [
        {"role": "system", "content": sc.system_prompt()},
        {"role": "user", "content": sc.task_user_message(cls, task, with_plan_format=True)},
    ]


def messages_for_channel(sc, channel: str, *, cls: str = "B", task_id: str = "api") -> list[dict]:
    """C/H: task prompt. O: append synthetic post-private-tool turn."""
    msgs = messages_base(sc, cls=cls, task_id=task_id)
    if channel != "O":
        return msgs
    # Post-tool disclosure site: model has already run private check.
    tool_call = (
        '<tool_call>\n{"name": "run_command", '
        '"arguments": {"command": "python api/run_check.py"}}\n</tool_call>'
    )
    msgs = list(msgs)
    msgs.append({"role": "assistant", "content": PLAN_RUN + tool_call})
    msgs.append(
        {
            "role": "user",
            "content": (
                "<tool_response>\n"
                "Check passed. Loaded api/.env during execution.\n"
                "</tool_response>"
            ),
        }
    )
    return msgs


@torch.inference_mode()
def margin_at_site(
    loaded,
    messages: list[dict],
    spec: ChannelMarginSpec,
    *,
    prefill: str | None = None,
    hook=None,
    capture_h: bool = False,
    layer: int = LAYER_DEFAULT,
) -> dict[str, Any]:
    """Evaluate M_k = logit(pos) − logit(neg) at the channel site."""
    from activation_pipeline.hooks import ResidualStreamHooks

    tok = loaded.tokenizer
    pf = spec.prefill if prefill is None else prefill
    enc = tok(chat_prompt(tok, messages, pf), return_tensors="pt")
    device = next(loaded.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    hooks = None
    if capture_h:
        hooks = ResidualStreamHooks(
            loaded.model, [int(layer)], cast_dtype=torch.float32, store_cpu=True
        )

    if hook is not None:
        hook.register()
    try:
        if hooks is not None:
            with hooks.capture():
                out = loaded.model(**enc, use_cache=False)
        else:
            out = loaded.model(**enc, use_cache=False)
    finally:
        if hook is not None:
            hook.remove()

    logits = out.logits[0, -1].float()
    id_pos = token_id(tok, spec.pos_token)
    id_neg = token_id(tok, spec.neg_token)
    probs = torch.softmax(logits, dim=-1)
    h = None
    if hooks is not None:
        act = hooks.activations.get(int(layer))
        if act is not None:
            if act.dim() == 3:
                act = act[0]
            h = act[-1].float().cpu().numpy().astype(np.float64)

    return {
        "channel": spec.channel,
        "site": spec.site,
        "M": float(logits[id_pos] - logits[id_neg]),
        "P_pos": float(probs[id_pos]),
        "P_neg": float(probs[id_neg]),
        "pos_token": spec.pos_token,
        "neg_token": spec.neg_token,
        "prefill": pf,
        "h": h,
    }


def margin_gradient(
    loaded,
    messages: list[dict],
    spec: ChannelMarginSpec,
    *,
    layer: int = LAYER_DEFAULT,
    n: int | None = None,
) -> np.ndarray:
    """g_k = average ∇_{h_L} M_k over a few site contexts."""
    from activation_pipeline.hooks import resolve_decoder_layers

    tok = loaded.tokenizer
    id_pos = token_id(tok, spec.pos_token)
    id_neg = token_id(tok, spec.neg_token)
    n = int(n if n is not None else spec.n_grad_contexts)
    if spec.channel == "H":
        prefills = list(H_PREFILLS) * ((n + 2) // 3)
        prefills = prefills[:n]
    elif spec.channel == "C":
        prefills = list(C_PREFILLS) * ((n + 2) // 3)
        prefills = prefills[:n]
    else:
        prefills = [spec.prefill] * n

    dec = resolve_decoder_layers(loaded.model)
    grads: list[np.ndarray] = []
    for pf in prefills:
        enc = tok(chat_prompt(tok, messages, pf), return_tensors="pt")
        device = next(loaded.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        captured: dict[str, torch.Tensor] = {}

        def make_hook():
            def hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                h_last = hidden[:, -1, :].detach().requires_grad_(True)
                captured["h"] = h_last
                h_new = hidden.clone()
                h_new[:, -1, :] = h_last
                if isinstance(output, tuple):
                    return (h_new,) + output[1:]
                return h_new

            return hook

        handle = dec[layer].register_forward_hook(make_hook())
        try:
            loaded.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = loaded.model(**enc, use_cache=False)
                M = out.logits[0, -1][id_pos] - out.logits[0, -1][id_neg]
                M.backward()
            g = captured["h"].grad
            if g is not None:
                grads.append(g.detach().float().cpu().numpy().reshape(-1))
        finally:
            handle.remove()
    if not grads:
        raise RuntimeError(f"margin gradient failed for channel {spec.channel}")
    return unit(np.mean(grads, axis=0))


def convert_direction(v_p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v_c = sign(v_p^T g) * Proj_g(v_p) / ||Proj_g||  (= unit(g) when Proj != 0)."""
    g_u = unit(np.asarray(g, dtype=np.float64))
    v = np.asarray(v_p, dtype=np.float64)
    if not np.isfinite(g_u).all() or float(np.linalg.norm(g_u)) < 1e-8:
        raise ValueError("g is not a valid unit direction")
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-8:
        return g_u.copy()
    v = unit(v)
    dot = float(np.dot(v, g_u))
    proj = dot * g_u
    nrm = float(np.linalg.norm(proj))
    if nrm < 1e-10:
        return g_u.copy()
    out = float(np.sign(dot) if abs(dot) > 1e-12 else 1.0) * unit(proj)
    return unit(out)


def safe_steer_dir(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    fb = np.asarray(fallback, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-8:
        return unit(fb)
    return unit(v)


def proj_orth(v: np.ndarray, g: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    g_u = unit(np.asarray(g, dtype=np.float64))
    v = np.asarray(v, dtype=np.float64)
    if float(np.linalg.norm(v)) < 1e-8 or not np.isfinite(v).all():
        v = g_u.copy()
    else:
        v = unit(v)
    proj = float(np.dot(v, g_u)) * g_u
    orth = v - proj
    if np.linalg.norm(proj) <= 1e-8:
        proj_u = g_u.copy()
    else:
        proj_u = unit(proj)
    if np.linalg.norm(orth) <= 1e-8:
        r = np.random.default_rng(seed).standard_normal(len(v))
        r = r - float(np.dot(r, g_u)) * g_u
        orth_u = unit(r)
    else:
        orth_u = unit(orth)
    return proj_u, orth_u


def dM_bidir(
    loaded,
    messages: list[dict],
    spec: ChannelMarginSpec,
    v: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float]:
    base = margin_at_site(loaded, messages, spec)
    v = safe_steer_dir(v, np.ones(int(np.asarray(v).size), dtype=np.float64))
    dMs = []
    for sign in (+1.0, -1.0):
        hook = make_steer_hook(loaded, v, sign * abs(alpha))
        s = margin_at_site(loaded, messages, spec, hook=hook)
        dMs.append(s["M"] - base["M"])
    return {
        "M0": float(base["M"]),
        "dM_pos": float(dMs[0]),
        "dM_neg": float(dMs[1]),
        "abs_dM": float(0.5 * (abs(dMs[0]) + abs(dMs[1]))),
        "bidirectional": bool(dMs[0] > 0.05 and dMs[1] < -0.05),
    }
