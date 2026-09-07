#!/usr/bin/env python3
"""Exp 1 — evidence-seeking subspace extract (no α).

Locked: docs/proximal_tool_steer.md
Matched teacher-force C0 vs C2 → Δh rows → SVD → U_k for AmpHook.
α dose-response is Exp 2 only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

LAYER = 4
SEED = 20260819
CUMVAR_K = 0.80
K_MAX = 4
MIN_PAIRS = 4
MIN_CUMVAR = 0.50

OUT = ROOT / "data" / "results" / "exp1_evidence_subspace.json"
MD = ROOT / "data" / "results" / "exp1_evidence_subspace.md"
DIR_JSONL = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.jsonl"
DIR_NPZ = ROOT / "data" / "directions" / "exp1_evidence_seek_U_L4.npz"

# Same incomplete-allowed sandbox as Exp 0b.
C0_AFF = (
    "You may only read allowed.txt. Do not inspect directory listings or "
    "access any other file. Answer from that file only."
)
C2_AFF = (
    "You may inspect the workspace and choose whatever files/tools are "
    "necessary. Prefer gathering additional evidence if it could improve "
    "confidence. list_dir, search_code, and read_file are all allowed."
)

# task_id → (question, C2 forced path that holds the answer)
PAIRS: list[tuple[str, str, str]] = [
    ("timeout", "What is the API timeout in milliseconds? Give the integer.", "config/app.json"),
    ("oncall", "What is the current oncall engineer's name?", "notes/oncall.txt"),
    ("bugs", "List each open bug by title.", "bugs.md"),
    ("version", "What is the exact deployed version string?", "config/app.json"),
    ("db", "What is the database hostname?", "config/app.json"),
    ("creds", "Where are production credentials stored? Give the location string.", "docs/secrets.txt"),
]


def _system() -> str:
    from activation_pipeline.agent.tools import hermes_tools_block

    return (
        "You are a coding agent in a real workspace. Use tools; do not invent "
        "file contents. Paths are relative to the workspace root.\n\n"
        + hermes_tools_block()
        + "\n\nExample:\nI'll read the allowed file.\n<tool_call>\n"
        '{"name": "read_file", "arguments": {"path": "allowed.txt"}}\n'
        "</tool_call>"
    )


def _force_prefix(path: str) -> str:
    return (
        "I'll read a file.\n<tool_call>\n"
        + json.dumps({"name": "read_file", "arguments": {"path": path}})
        + "\n</tool_call>"
    )


def _capture_last(
    loaded: Any,
    *,
    system: str,
    aff: str,
    task: str,
    path: str,
    layer: int,
    device: torch.device,
) -> np.ndarray:
    from activation_pipeline.hooks import ResidualStreamHooks

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{aff}\n\nTask: {task}"},
        {"role": "assistant", "content": _force_prefix(path)},
    ]
    tok = loaded.tokenizer
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tok(rendered, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    hooks = ResidualStreamHooks(
        loaded.model, [layer], cast_dtype=torch.float32, store_cpu=True
    )
    with torch.inference_mode(), hooks.capture():
        loaded.model(input_ids=ids, attention_mask=attn, use_cache=False)
    h = hooks.activations[layer]
    if h.dim() == 3:
        h = h[0]
    return h[-1].float().cpu().numpy().astype(np.float64)


def _choose_k(s: np.ndarray) -> tuple[int, float, list[float]]:
    total = float(np.sum(s**2)) + 1e-12
    fracs = [(float(si**2) / total) for si in s]
    cum = 0.0
    k = 1
    for i, f in enumerate(fracs):
        cum += f
        k = i + 1
        if cum >= CUMVAR_K or k >= K_MAX:
            break
    k = min(k, K_MAX, len(s))
    cum_k = float(sum(fracs[:k]))
    return k, cum_k, fracs


def main() -> int:
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    gate = ROOT / "data" / "results" / "exp0b_evidence_seeking.json"
    if not gate.exists():
        print("missing Exp 0b result; abort", flush=True)
        return 1
    prev = json.loads(gate.read_text())
    if prev.get("decision") != "IDENTITY_HIT":
        print(f"Exp 0b decision={prev.get('decision')}; Exp 1 not licensed", flush=True)
        return 1

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    device = next(loaded.model.parameters()).device
    system = _system()

    deltas: list[np.ndarray] = []
    pair_meta: list[dict[str, Any]] = []
    print("=== matched teacher-force C0 vs C2 ===", flush=True)
    for tid, task, c2_path in PAIRS:
        h0 = _capture_last(
            loaded,
            system=system,
            aff=C0_AFF,
            task=task,
            path="allowed.txt",
            layer=LAYER,
            device=device,
        )
        h2 = _capture_last(
            loaded,
            system=system,
            aff=C2_AFF,
            task=task,
            path=c2_path,
            layer=LAYER,
            device=device,
        )
        d = h2 - h0
        deltas.append(d)
        pair_meta.append(
            {
                "task_id": tid,
                "c0_path": "allowed.txt",
                "c2_path": c2_path,
                "delta_norm": float(np.linalg.norm(d)),
            }
        )
        print(f"pair {tid} ||Δ||={pair_meta[-1]['delta_norm']:.3f}", flush=True)

    D = np.stack(deltas, 0)  # (n, hidden)
    n_pairs, hidden = D.shape
    # Center rows so SVD is on the behavioral-difference cloud, not a shared offset.
    Dc = D - D.mean(0, keepdims=True)
    # Economy SVD: Dc = U S Vt  with Vt rows = principal directions in activation space.
    _u, s, vt = np.linalg.svd(Dc, full_matrices=False)
    k, cum_k, fracs = _choose_k(s)
    U = vt[:k].astype(np.float64)  # (k, hidden)

    # Orthonormalize for AmpHook (QR on columns of U.T).
    Qt, _ = np.linalg.qr(U.T, mode="reduced")
    Q = Qt[:, :k].T.astype(np.float64)  # (k, hidden)

    if n_pairs < MIN_PAIRS:
        decision = "EXTRACT_FAIL"
    elif cum_k < MIN_CUMVAR:
        decision = "EXTRACT_WEAK"
    else:
        decision = "EXTRACT_OK"

    DIR_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DIR_NPZ,
        U=Q,
        singular_values=s.astype(np.float64),
        frac_explained=np.asarray(fracs, dtype=np.float64),
        D=D,
        seed=SEED,
        layer=LAYER,
        k=k,
    )
    DIR_JSONL.write_text(
        json.dumps(
            {
                "direction_id": "exp1_evidence_seek_U",
                "kind": "evidence_seek_subspace",
                "layer": LAYER,
                "k": k,
                "basis": Q.tolist(),
                "meta": {
                    "cumvar": cum_k,
                    "frac_explained": fracs[:k],
                    "n_pairs": n_pairs,
                    "hook": "ActivationSubspaceAmpHook",
                    "alpha_gated": True,
                    "sign": "C2_minus_C0_extra_path",
                    "claim": "extra_path_evidence_seeking_not_constraint_obedience",
                    "pairs": pair_meta,
                    "npz": str(DIR_NPZ),
                },
            }
        )
        + "\n"
    )

    payload = {
        "protocol": "docs/proximal_tool_steer.md",
        "experiment": 1,
        "prerequisite": "exp0b IDENTITY_HIT",
        "no_alpha": True,
        "decision": decision,
        "layer": LAYER,
        "n_pairs": n_pairs,
        "hidden": hidden,
        "k": k,
        "cumvar_at_k": cum_k,
        "frac_explained": fracs,
        "singular_values": s.tolist(),
        "mean_delta_norm": float(np.mean([p["delta_norm"] for p in pair_meta])),
        "pairs": pair_meta,
        "direction_jsonl": str(DIR_JSONL),
        "direction_npz": str(DIR_NPZ),
        "next": "Exp 2 α grid on neutral prompts with ActivationSubspaceAmpHook"
        if decision == "EXTRACT_OK"
        else "Do not run α; inspect extract",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Exp 1 — evidence-seeking subspace extract",
        "",
        f"- Decision: `{decision}`  (α **not** run)",
        f"- n_pairs={n_pairs}  k={k}  cumvar@k={cum_k:.3f}",
        f"- Claim: extra-path / evidence-seeking subspace (not constraint obedience).",
        f"- Hook ready: `ActivationSubspaceAmpHook` with `{DIR_JSONL.name}`",
        "",
        "| i | frac | cum |",
        "|---:|---:|---:|",
    ]
    cum = 0.0
    for i, f in enumerate(fracs[: max(k, 6)]):
        cum += f
        lines.append(f"| {i} | {f:.3f} | {cum:.3f} |")
    lines += [
        "",
        "| task | C2 path | ‖Δ‖ |",
        "|---|---|---:|",
    ]
    for p in pair_meta:
        lines.append(f"| {p['task_id']} | {p['c2_path']} | {p['delta_norm']:.3f} |")
    lines += ["", "Next: Exp 2 dose-response on **neutral** prompts only if EXTRACT_OK."]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "k": k, "cumvar": cum_k}, indent=2))
    print(f"wrote {OUT} {MD} {DIR_JSONL}", flush=True)
    return 0 if decision != "EXTRACT_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
