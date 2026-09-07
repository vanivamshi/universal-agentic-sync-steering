#!/usr/bin/env python3
"""Phase 1 — persona PCA basis on Qwen3-0.6B (Lu et al. method, Mac-scale).

1.1 Extract role vectors (subset of Lu roles) in prose mode + tool-call mode
1.2 Fit PCA separately per mode; compare variance + principal angles
1.3 Report near-90° subspace divergence if present
1.4 Score top PCs + assistant axis with Heimersheim-style ε* privilege on GAP windows

Full 275×1200 Lu pipeline is not runnable on MPS; this uses a documented subset
(--n-roles / --n-questions) that preserves the method, not HF 32B drop-in vectors.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ASSISTANT_AXIS_ROLES = Path.home() / "assistant-axis" / "data" / "roles" / "instructions"
DEFAULT_QUESTIONS = Path.home() / "assistant-axis" / "data" / "extraction_questions.jsonl"

PROSE_SYSTEM_DEFAULT = "You are a helpful assistant."
TOOL_SYSTEM_SUFFIX = (
    " You may use tools via Hermes format: "
    '<tool_call>\n{"name":"lookup","arguments":{"query":"..."}}\n</tool_call> '
    "When a lookup would help, emit a tool call first."
)


def _load_roles(roles_dir: Path) -> list[tuple[str, str]]:
    rows = []
    for path in sorted(roles_dir.glob("*.json")):
        data = json.loads(path.read_text())
        instr = data.get("instruction") or []
        if not instr:
            continue
        # first pos prompt
        text = instr[0].get("pos") or instr[0].get("instruction") or ""
        if text:
            rows.append((path.stem, text))
    return rows


def _load_questions(path: Path, n: int, seed: int) -> list[str]:
    qs = [json.loads(line)["question"] for line in path.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    if n >= len(qs):
        return qs
    return rng.sample(qs, n)


@torch.inference_mode()
def _generate(loaded, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
    from activation_pipeline.agent.loop import generate_assistant

    return generate_assistant(loaded, messages, max_new_tokens=max_new_tokens, temperature=0.7)


@torch.inference_mode()
def _response_mean_residual(
    loaded,
    messages: list[dict[str, str]],
    response: str,
    *,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Mean residual over assistant response tokens (Lu-style response pooling)."""
    from activation_pipeline.hooks import ResidualStreamHooks

    tok = loaded.tokenizer
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prompt + response
    enc = tok(full, return_tensors="pt", add_special_tokens=False)
    prompt_enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(loaded.model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    prompt_len = int(prompt_enc["input_ids"].shape[-1])
    hooks = ResidualStreamHooks(loaded.model, layers, cast_dtype=torch.float32, store_cpu=True)
    with hooks.capture():
        loaded.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        h = hooks.activations[layer]
        if h.dim() == 3:
            h = h[0]
        # response span
        if h.shape[0] <= prompt_len:
            out[layer] = h[-1].float().cpu()
        else:
            out[layer] = h[prompt_len:].float().mean(dim=0).cpu()
    return out


def collect_role_matrix(
    loaded,
    roles: list[tuple[str, str]],
    questions: list[str],
    *,
    layers: list[int],
    mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Return {layer: (n_roles, hidden) array} plus default vector and role names."""
    layer_stacks: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    default_stacks: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    role_names: list[str] = []

    # default assistant vectors (no role)
    for qi, q in enumerate(questions):
        if mode == "prose":
            sys_msg = PROSE_SYSTEM_DEFAULT
            user = q
        else:
            sys_msg = PROSE_SYSTEM_DEFAULT + TOOL_SYSTEM_SUFFIX
            user = q + " If useful, look up a relevant fact with the lookup tool before answering."
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        print(f"  [{mode}] default q{qi+1}/{len(questions)}", flush=True)
        resp = _generate(loaded, messages, max_new_tokens=max_new_tokens)
        acts = _response_mean_residual(loaded, messages, resp, layers=layers)
        for L in layers:
            default_stacks[L].append(acts[L])

    for ri, (name, instruction) in enumerate(roles):
        role_names.append(name)
        per_q: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        for qi, q in enumerate(questions):
            if mode == "prose":
                sys_msg = instruction
                user = q
            else:
                sys_msg = instruction + TOOL_SYSTEM_SUFFIX
                user = q + " If useful, look up a relevant fact with the lookup tool before answering."
            messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
            print(f"  [{mode}] role {ri+1}/{len(roles)} {name} q{qi+1}/{len(questions)}", flush=True)
            resp = _generate(loaded, messages, max_new_tokens=max_new_tokens)
            acts = _response_mean_residual(loaded, messages, resp, layers=layers)
            for L in layers:
                per_q[L].append(acts[L])
        for L in layers:
            layer_stacks[L].append(torch.stack(per_q[L]).mean(dim=0))

    matrices = {L: torch.stack(layer_stacks[L]).numpy() for L in layers}
    defaults = {L: torch.stack(default_stacks[L]).mean(dim=0).numpy() for L in layers}
    return {"matrices": matrices, "defaults": defaults, "role_names": role_names}


def write_directions_jsonl(
    path: Path,
    *,
    mode: str,
    layer: int,
    pca_components: np.ndarray,
    assistant_axis: np.ndarray,
    explained: np.ndarray,
    n_pcs: int,
) -> None:
    rows = []
    aa = assistant_axis / (np.linalg.norm(assistant_axis) + 1e-12)
    rows.append(
        {
            "direction_id": f"assistant_axis_{mode}",
            "kind": "assistant_axis",
            "mode": mode,
            "layer": layer,
            "vector": aa.astype(float).tolist(),
            "meta": {"phase": 1, "source": "persona_pca"},
        }
    )
    for i in range(min(n_pcs, pca_components.shape[0])):
        v = pca_components[i]
        v = v / (np.linalg.norm(v) + 1e-12)
        rows.append(
            {
                "direction_id": f"persona_pc{i}_{mode}",
                "kind": "persona_pc",
                "mode": mode,
                "layer": layer,
                "vector": v.astype(float).tolist(),
                "meta": {
                    "phase": 1,
                    "pc_index": i,
                    "explained_variance_ratio": float(explained[i]),
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roles-dir", type=Path, default=ASSISTANT_AXIS_ROLES)
    ap.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--n-roles", type=int, default=32)
    ap.add_argument("--n-questions", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--layers", type=int, nargs="*", default=[4])
    ap.add_argument("--n-pcs-export", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--skip-tool-mode", action="store_true")
    ap.add_argument("--skip-privilege", action="store_true", default=True)
    ap.add_argument("--max-privilege-transcripts", type=int, default=8)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "phase1_persona_pca.json",
    )
    args = ap.parse_args()

    from activation_pipeline.analysis.persona_pca import (
        component_alignment,
        fit_pca,
        principal_angles_deg,
    )
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    if not args.roles_dir.is_dir():
        raise SystemExit(f"roles dir missing: {args.roles_dir}")

    all_roles = _load_roles(args.roles_dir)
    rng = random.Random(args.seed)
    prefer = ["analyst", "therapist", "ghost", "hermit", "consultant", "coach", "pirate", "robot"]
    chosen = []
    by_name = dict(all_roles)
    for p in prefer:
        if p in by_name:
            chosen.append((p, by_name[p]))
    rest = [(n, t) for n, t in all_roles if n not in {c[0] for c in chosen}]
    rng.shuffle(rest)
    roles = (chosen + rest)[: args.n_roles]
    questions = _load_questions(args.questions_file, args.n_questions, args.seed)
    print(f"roles={len(roles)} questions={len(questions)} layers={args.layers}", flush=True)

    t0 = time.time()
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )

    modes = ["prose"] if args.skip_tool_mode else ["prose", "tool"]
    mode_data: dict[str, Any] = {}
    for mode in modes:
        print(f"\n=== collect {mode} ===", flush=True)
        mode_data[mode] = collect_role_matrix(
            loaded,
            roles,
            questions,
            layers=args.layers,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
        )

    report: dict[str, Any] = {
        "model": LOCAL_MODEL_KEY,
        "n_roles": len(roles),
        "role_names": [r[0] for r in roles],
        "n_questions": len(questions),
        "layers": args.layers,
        "protocol": (
            "Lu-style mean response activations over role system prompts; "
            "PCA via SVD; assistant_axis = mean(default) - mean(roles); "
            "Mac-scale subset (not full 275×1200)"
        ),
        "by_layer": {},
    }

    primary_layer = args.layers[0]
    for L in args.layers:
        layer_rep: dict[str, Any] = {}
        pcs = {}
        for mode in modes:
            X = mode_data[mode]["matrices"][L]
            default = mode_data[mode]["defaults"][L]
            role_mean = X.mean(axis=0)
            axis = default - role_mean
            pca = fit_pca(X)
            pcs[mode] = pca
            out_dir = ROOT / "data" / "directions" / f"persona_pca_{mode}_L{L}.jsonl"
            write_directions_jsonl(
                out_dir,
                mode=mode,
                layer=L,
                pca_components=pca.components,
                assistant_axis=axis,
                explained=pca.explained_variance_ratio,
                n_pcs=args.n_pcs_export,
            )
            layer_rep[mode] = {
                "pca": pca.to_dict(),
                "assistant_axis_norm": float(np.linalg.norm(axis)),
                "directions_out": str(out_dir),
                "n_roles_matrix": int(X.shape[0]),
                "hidden": int(X.shape[1]),
            }
            print(
                f"L{L} {mode}: dims@90%={pca.to_dict()['dims_for_90']} "
                f"PC1 var={pca.explained_variance_ratio[0]:.3f}",
                flush=True,
            )

        if "prose" in pcs and "tool" in pcs:
            r = min(8, pcs["prose"].n_components, pcs["tool"].n_components)
            angles = principal_angles_deg(
                pcs["prose"].components[:r], pcs["tool"].components[:r], rank=min(4, r)
            )
            align = component_alignment(
                pcs["prose"].components, pcs["tool"].components, n=min(5, r)
            )
            axis_p = mode_data["prose"]["defaults"][L] - mode_data["prose"]["matrices"][L].mean(0)
            axis_t = mode_data["tool"]["defaults"][L] - mode_data["tool"]["matrices"][L].mean(0)
            axis_p = axis_p / (np.linalg.norm(axis_p) + 1e-12)
            axis_t = axis_t / (np.linalg.norm(axis_t) + 1e-12)
            aa_cos = float(np.dot(axis_p, axis_t))
            layer_rep["prose_vs_tool"] = {
                "principal_angles": angles,
                "pc_alignment": align,
                "assistant_axis_cosine": aa_cos,
                "divergence_claim": (
                    "NEAR_90_DIVERGENCE"
                    if angles["near_orthogonal_90"]
                    else "ALIGNED_OR_ACUTE"
                ),
            }
            print(
                f"L{L} prose↔tool mean∠={angles['mean_angle_deg']:.1f}° "
                f"max∠={angles['max_angle_deg']:.1f}° "
                f"AA cos={aa_cos:.3f} → {layer_rep['prose_vs_tool']['divergence_claim']}",
                flush=True,
            )
        report["by_layer"][str(L)] = layer_rep

    if not args.skip_privilege:
        print(
            "privilege pilot removed — see docs/archived_pca_shortlist.md",
            flush=True,
        )

    report["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out} elapsed={report['elapsed_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
