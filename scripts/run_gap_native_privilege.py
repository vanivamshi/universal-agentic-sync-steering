#!/usr/bin/env python3
"""ε* privilege of GAP-native v vs locked L4 controls (prose vs tool).

Does not retune p90. Does not steer. Locked: docs/gap_native_privilege.md.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 20260814
LAYER_K = 4
LAYER_L = 22
THRESHOLD = 0.5
MAX_ITER = 20
N_RANDOM = 8
NATIVE = ROOT / "data" / "directions" / "gap_native_tool_sg_L4.jsonl"
CONTROLS = ROOT / "data" / "directions" / "controls_l4.jsonl"
RANDOMS = ROOT / "data" / "directions" / "random_controls_l4.jsonl"
TRANSCRIPTS = ROOT / "data" / "transcripts" / "real" / "gap_agentic.jsonl"
OUT = ROOT / "data" / "results" / "gap_native_privilege.json"
MD = ROOT / "data" / "results" / "gap_native_privilege.md"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bank() -> list[dict[str, Any]]:
    native = _load_jsonl(NATIVE)[0]
    rows: list[dict[str, Any]] = [
        {
            "direction_id": native["direction_id"],
            "kind": "gap_native",
            "direction_type": "native",
            "domain": None,
            "vector": torch.tensor(native["vector"], dtype=torch.float32),
        }
    ]
    for r in _load_jsonl(CONTROLS):
        if int(r["layer"]) != LAYER_K:
            continue
        rows.append(
            {
                "direction_id": r["direction_id"],
                "kind": r["kind"],
                "direction_type": "learned_control",
                "domain": (r.get("meta") or {}).get("domain"),
                "vector": torch.tensor(r["vector"], dtype=torch.float32),
            }
        )
    for r in _load_jsonl(RANDOMS)[:N_RANDOM]:
        rows.append(
            {
                "direction_id": r["direction_id"],
                "kind": "random_control",
                "direction_type": "random_control",
                "domain": None,
                "vector": torch.tensor(r["vector"], dtype=torch.float32),
            }
        )
    return rows


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def main() -> int:
    from activation_pipeline.analysis.sensitivity import LayerPerturbHooks
    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.windows import embed_windows_in_chat, tag_transcript_assistant_turns
    from scripts.run_rq1_privilege_pilot import epsilon_star_from_clean

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    device = next(loaded.model.parameters()).device
    bank = _bank()
    transcripts = load_transcripts(TRANSCRIPTS)
    preferred = {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}
    transcripts = [t for t in transcripts if t.get("source") in preferred] or transcripts

    sens_rows: list[dict[str, Any]] = []
    for tr_i, tr in enumerate(transcripts):
        tagged_list = tag_transcript_assistant_turns(tr, loaded.tokenizer)
        for tagged in tagged_list:
            embedded = embed_windows_in_chat(
                loaded.tokenizer, tr["messages"], tagged.message_index, tagged
            )
            if not embedded.windows:
                continue
            input_ids = torch.tensor([embedded.input_ids], device=device)
            attn = torch.ones_like(input_ids)
            for window in embedded.windows:
                idxs = window.token_indices()
                if not idxs:
                    continue
                clean_hooks = LayerPerturbHooks(
                    loaded.model,
                    layer_k=LAYER_K,
                    direction=torch.ones(loaded.spec.hidden_size),
                    token_indices=idxs,
                    read_layers=[LAYER_L],
                )
                clean_acts = clean_hooks.run(input_ids, attn, epsilon=0.0)

                def pool(t: torch.Tensor, token_idxs=idxs) -> torch.Tensor:
                    return t[0, token_idxs, :].float().mean(dim=0)

                clean_L = pool(clean_acts[LAYER_L])
                applicable = []
                for dinfo in bank:
                    if (
                        dinfo["direction_type"] == "learned_control"
                        and dinfo["domain"] is not None
                        and dinfo["domain"] != tr.get("domain")
                    ):
                        continue
                    applicable.append(dinfo)
                for dinfo in applicable:
                    hooks = LayerPerturbHooks(
                        loaded.model,
                        layer_k=LAYER_K,
                        direction=dinfo["vector"],
                        token_indices=idxs,
                        read_layers=[LAYER_L],
                    )

                    def forward_pert(eps: float, h=hooks, ids=input_ids, a=attn):
                        return h.run(ids, a, epsilon=eps)

                    eps_star, blowup, converged = epsilon_star_from_clean(
                        clean_L=clean_L,
                        forward_pert=forward_pert,
                        layer_L=LAYER_L,
                        pool=pool,
                        threshold=THRESHOLD,
                        max_iter=MAX_ITER,
                        direction_id=dinfo["direction_id"],
                        layer_k=LAYER_K,
                    )
                    sens_rows.append(
                        {
                            "transcript_id": tr["transcript_id"],
                            "domain": tr.get("domain"),
                            "window_kind": window.kind,
                            "direction_id": dinfo["direction_id"],
                            "direction_type": dinfo["direction_type"],
                            "epsilon_star": eps_star,
                            "blowup": blowup,
                            "converged": converged,
                        }
                    )
                print(
                    f"priv [{tr_i+1}/{len(transcripts)}] {tr['transcript_id']} "
                    f"{window.kind} ntok={len(idxs)}",
                    flush=True,
                )

    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"prose": [], "tool_call": []})
    meta: dict[str, str] = {}
    for r in sens_rows:
        by[r["direction_id"]][r["window_kind"]].append(float(r["epsilon_star"]))
        meta[r["direction_id"]] = r["direction_type"]

    per_direction = []
    for did, modes in sorted(by.items()):
        mp, mt = _mean(modes["prose"]), _mean(modes["tool_call"])
        per_direction.append(
            {
                "direction_id": did,
                "direction_type": meta[did],
                "mean_eps_prose": mp,
                "mean_eps_tool": mt,
                "delta_tool_minus_prose": mt - mp,
                "n_prose": len(modes["prose"]),
                "n_tool": len(modes["tool_call"]),
            }
        )

    native = next(d for d in per_direction if d["direction_type"] == "native")
    learned = [d for d in per_direction if d["direction_type"] == "learned_control"]
    randoms = [d for d in per_direction if d["direction_type"] == "random_control"]
    d_nat = float(native["delta_tool_minus_prose"])
    d_learn = [float(d["delta_tool_minus_prose"]) for d in learned]
    d_rand = [float(d["delta_tool_minus_prose"]) for d in randoms]
    i_learn = d_nat - _mean(d_learn)
    i_rand = d_nat - _mean(d_rand)
    n_prose = int(native["n_prose"])
    n_tool = int(native["n_tool"])
    if n_prose < 8 or n_tool < 8:
        decision = "UNDERPOWERED"
    elif not d_rand:
        decision = "NO_RANDOM"
    else:
        p10 = float(np.percentile(d_rand, 10))
        p90 = float(np.percentile(d_rand, 90))
        if d_nat > p90 and i_learn > 0:
            decision = "NATIVE_PRIVILEGE_GAIN"
        elif d_nat < p10 and i_learn < 0:
            decision = "NATIVE_PRIVILEGE_LOSS"
        else:
            decision = "NULL_VS_CONTROLS"

    payload = {
        "protocol": "docs/gap_native_privilege.md",
        "p90_native_hit_unchanged": True,
        "layer_k": LAYER_K,
        "layer_L": LAYER_L,
        "threshold": THRESHOLD,
        "n_transcripts": len(transcripts),
        "n_random": N_RANDOM,
        "sign_note": "Δ=ε*_tool−ε*_prose; positive ⇒ privilege gain in tool",
        "native_delta": d_nat,
        "mean_learned_delta": _mean(d_learn),
        "mean_random_delta": _mean(d_rand),
        "I_native_minus_learned": i_learn,
        "I_native_minus_random": i_rand,
        "random_delta_p10": float(np.percentile(d_rand, 10)) if d_rand else float("nan"),
        "random_delta_p90": float(np.percentile(d_rand, 90)) if d_rand else float("nan"),
        "decision": decision,
        "exploratory": True,
        "per_direction": per_direction,
        "n_rows": len(sens_rows),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# GAP-native v privilege vs locked controls",
        "",
        f"- Decision: `{decision}`  (exploratory; n_random={N_RANDOM})",
        f"- Δ_native={d_nat:+.3f}  mean Δ_learned={_mean(d_learn):+.3f}  mean Δ_random={_mean(d_rand):+.3f}",
        f"- I vs learned={i_learn:+.3f}  I vs random={i_rand:+.3f}",
        "- Sign: Δ=ε*_tool−ε*_prose; positive ⇒ privilege gain in tool.",
        "- p90 of the native HIT was not changed.",
        "",
        "| id | type | ε*_prose | ε*_tool | Δ | n_prose | n_tool |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for d in per_direction:
        lines.append(
            f"| {d['direction_id']} | {d['direction_type']} | "
            f"{d['mean_eps_prose']:.3f} | {d['mean_eps_tool']:.3f} | "
            f"{d['delta_tool_minus_prose']:+.3f} | {d['n_prose']} | {d['n_tool']} |"
        )
    lines += ["", f"Artifact: `{OUT}`"]
    MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "native_delta": d_nat, "I_learned": i_learn}, indent=2))
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
