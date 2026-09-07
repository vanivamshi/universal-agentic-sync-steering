#!/usr/bin/env python3
"""Step 5 / RQ1 pilot: safety vs control directional sensitivity (prose↔tool).

Primary metric: ε* at layer k such that relative L2 blowup at layer L exceeds τ.
Relative privilege loss = Δε* = ε*_tool − ε*_prose (positive ⇒ less privilege in tool).

H1: mean Δε*_safety > mean Δε*_control.
Null: safety and controls lose privilege equally.

Pilot notes:
- Model = Qwen3-0.6B; confirmatory claims require Qwen3-32B.
- Refusal arm is blocked by the §3 stability gate; numbers are exploratory only.
- Assistant Axis / harmlessness are reported as provisional (no full tool-mode
  stability gate completed for them on this pilot).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_direction_at_layer(path: Path, *, layer: int) -> dict[str, Any] | None:
    for row in load_jsonl(path):
        if int(row["layer"]) == layer:
            return row
    return None


@dataclass
class WindowSens:
    transcript_id: str
    domain: str
    message_index: int
    window_kind: str
    window_name: str
    direction_id: str
    direction_type: str  # safety | learned_control | random_control
    epsilon_star: float
    blowup: float
    converged: bool
    n_tokens: int


def epsilon_star_from_clean(
    *,
    clean_L: Any,
    forward_pert,
    layer_L: int,
    pool,
    threshold: float,
    max_iter: int,
    direction_id: str,
    layer_k: int,
    eps_lo: float = 1e-4,
    eps_hi: float = 50.0,
) -> tuple[float, float, bool]:
    """Binary-search ε* reusing a precomputed clean late-layer pooled activation."""
    from activation_pipeline.analysis.sensitivity import residual_l2_blowup

    def blow(eps: float) -> float:
        pert = forward_pert(eps)
        return residual_l2_blowup(clean_L, pool(pert[layer_L]), relative=True)

    hi = eps_hi
    b_hi = blow(hi)
    if b_hi < threshold:
        return hi, b_hi, False

    lo = eps_lo
    b_lo = blow(lo)
    if b_lo >= threshold:
        return lo, b_lo, True

    best_eps = hi
    best_blow = b_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        b_mid = blow(mid)
        if b_mid >= threshold:
            hi = mid
            best_eps = mid
            best_blow = b_mid
        else:
            lo = mid
    return best_eps, best_blow, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transcripts",
        type=Path,
        default=ROOT / "data" / "transcripts" / "real" / "gap_agentic.jsonl",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--layer-k", type=int, default=None, help="Default: model default_perturb_layer")
    ap.add_argument("--layer-L", type=int, default=None, help="Default: model default_measure_layer")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=16)
    ap.add_argument("--n-random", type=int, default=4, help="Subset of random controls (speed)")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--max-transcripts", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "rq1_privilege_pilot.json",
    )
    args = ap.parse_args()

    import torch

    from activation_pipeline.analysis.sensitivity import LayerPerturbHooks
    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.controls import (
        build_random_orthogonal_controls,
        collect_pair_activations,
        full_directions_from_activations,
    )
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.windows import embed_windows_in_chat, tag_transcript_assistant_turns

    model_key = args.model or LOCAL_MODEL_KEY
    assert_model_fits_machine(model_key)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    layer_k = args.layer_k if args.layer_k is not None else loaded.spec.default_perturb_layer
    layer_L = args.layer_L if args.layer_L is not None else loaded.spec.default_measure_layer
    device = next(loaded.model.parameters()).device
    print(f"rq1 model={model_key} device={device_map} k={layer_k} L={layer_L} τ={args.threshold}")

    pairs_dir = ROOT / "data" / "contrast_pairs"
    dirs_dir = ROOT / "data" / "directions"

    direction_bank: list[dict[str, Any]] = []

    safety_files = [
        ("refusal", dirs_dir / "refusal_prose.jsonl", "blocked_by_section3_stability_gate"),
        ("assistant_axis", dirs_dir / "assistant_axis_prose.jsonl", "provisional_no_tool_stability_gate"),
        ("harmlessness", dirs_dir / "harmlessness_prose.jsonl", "provisional_no_tool_stability_gate"),
    ]
    safety_tensors: list[torch.Tensor] = []
    for kind, path, status in safety_files:
        row = load_direction_at_layer(path, layer=layer_k)
        if row is None:
            print(f"WARN missing {kind} at L{layer_k}")
            continue
        vec = torch.tensor(row["vector"], dtype=torch.float32)
        safety_tensors.append(vec)
        direction_bank.append(
            {
                "direction_id": f"{kind}_L{layer_k}",
                "kind": kind,
                "direction_type": "safety",
                "domain": None,
                "status": status,
                "vector": vec,
            }
        )

    control_specs = [
        ("syntax", "syntax", None, pairs_dir / "syntax_control_pilot.jsonl"),
        (
            "domain_content_coding",
            "domain_content",
            "coding",
            pairs_dir / "domain_content_coding_pilot.jsonl",
        ),
        (
            "domain_content_therapy",
            "domain_content",
            "therapy",
            pairs_dir / "domain_content_therapy_pilot_v2.jsonl",
        ),
        (
            "domain_content_writing",
            "domain_content",
            "writing",
            pairs_dir / "domain_content_writing_pilot.jsonl",
        ),
    ]
    learned_tensors: list[torch.Tensor] = []
    for direction_id, kind, domain, path in control_specs:
        pairs = load_jsonl(path)
        print(f"extracting control {direction_id} at L{layer_k} from {path.name} (n={len(pairs)})")
        pos, neg = collect_pair_activations(loaded, pairs, layers=[layer_k])
        dirs = full_directions_from_activations(
            pos,
            neg,
            layers=[layer_k],
            direction_id=direction_id,
            kind=kind,
            domain=domain,
        )
        vec = dirs[0].tensor()
        learned_tensors.append(vec)
        direction_bank.append(
            {
                "direction_id": dirs[0].direction_id,
                "kind": kind,
                "direction_type": "learned_control",
                "domain": domain,
                "status": "step4_eligible_at_Fmid_extracted_at_k",
                "vector": vec,
            }
        )

    random_dirs, random_diag = build_random_orthogonal_controls(
        layer=layer_k,
        hidden_size=loaded.spec.hidden_size,
        fixed_basis=[*safety_tensors, *learned_tensors],
        n_random=max(args.n_random, 4),
        seed=args.seed,
    )
    for d in random_dirs[: args.n_random]:
        direction_bank.append(
            {
                "direction_id": d.direction_id,
                "kind": "random_control",
                "direction_type": "random_control",
                "domain": None,
                "status": "random_orthogonal",
                "vector": d.tensor(),
            }
        )
    print(
        f"directions: safety={sum(1 for d in direction_bank if d['direction_type']=='safety')} "
        f"learned={sum(1 for d in direction_bank if d['direction_type']=='learned_control')} "
        f"random={sum(1 for d in direction_bank if d['direction_type']=='random_control')} "
        f"random_orth_ok={random_diag.passed}"
    )

    transcripts = load_transcripts(args.transcripts)
    preferred = {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}
    transcripts = [t for t in transcripts if t.get("source") in preferred] or transcripts
    if args.max_transcripts is not None:
        transcripts = transcripts[: args.max_transcripts]

    rows: list[WindowSens] = []
    n_windows = 0
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
                n_windows += 1

                # Clean forward once per window (ε=0); direction unused at ε=0.
                clean_hooks = LayerPerturbHooks(
                    loaded.model,
                    layer_k=layer_k,
                    direction=torch.ones(loaded.spec.hidden_size),
                    token_indices=idxs,
                    read_layers=[layer_L],
                )
                clean_acts = clean_hooks.run(input_ids, attn, epsilon=0.0)

                def pool(t: torch.Tensor, token_idxs=idxs) -> torch.Tensor:
                    return t[0, token_idxs, :].float().mean(dim=0)

                clean_L = pool(clean_acts[layer_L])

                applicable = []
                for dinfo in direction_bank:
                    if (
                        dinfo["direction_type"] == "learned_control"
                        and dinfo["domain"] is not None
                        and dinfo["domain"] != tr["domain"]
                    ):
                        continue
                    applicable.append(dinfo)

                for dinfo in applicable:
                    hooks = LayerPerturbHooks(
                        loaded.model,
                        layer_k=layer_k,
                        direction=dinfo["vector"],
                        token_indices=idxs,
                        read_layers=[layer_L],
                    )

                    def forward_pert(eps: float, h=hooks, ids=input_ids, a=attn):
                        return h.run(ids, a, epsilon=eps)

                    eps_star, blowup, converged = epsilon_star_from_clean(
                        clean_L=clean_L,
                        forward_pert=forward_pert,
                        layer_L=layer_L,
                        pool=pool,
                        threshold=args.threshold,
                        max_iter=args.max_iter,
                        direction_id=dinfo["direction_id"],
                        layer_k=layer_k,
                    )
                    rows.append(
                        WindowSens(
                            transcript_id=tr["transcript_id"],
                            domain=tr["domain"],
                            message_index=tagged.message_index,
                            window_kind=window.kind,
                            window_name=window.name or "",
                            direction_id=dinfo["direction_id"],
                            direction_type=dinfo["direction_type"],
                            epsilon_star=eps_star,
                            blowup=blowup,
                            converged=converged,
                            n_tokens=len(idxs),
                        )
                    )
                print(
                    f"[{tr_i+1}/{len(transcripts)}] {tr['transcript_id']} "
                    f"msg={tagged.message_index} {window.kind} "
                    f"ntok={len(idxs)} dirs={len(applicable)}"
                )

    # Transcript-level prose vs tool means per direction
    by_dir_tr: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"prose": [], "tool_call": []})
    )
    meta_by_dir: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_dir_tr[r.direction_id][r.transcript_id][r.window_kind].append(r.epsilon_star)
        if r.direction_id not in meta_by_dir:
            meta_by_dir[r.direction_id] = next(
                d for d in direction_bank if d["direction_id"] == r.direction_id
            )

    per_direction = []
    # Store transcript-level deltas for bootstrap
    deltas_by_dir: dict[str, dict[str, float]] = defaultdict(dict)
    for direction_id, tr_map in by_dir_tr.items():
        deltas = []
        prose_vals = []
        tool_vals = []
        for tr_id, modes in tr_map.items():
            if not modes["prose"] or not modes["tool_call"]:
                continue
            p = sum(modes["prose"]) / len(modes["prose"])
            t = sum(modes["tool_call"]) / len(modes["tool_call"])
            prose_vals.append(p)
            tool_vals.append(t)
            delta = t - p
            deltas.append(delta)
            deltas_by_dir[direction_id][tr_id] = delta
        if not deltas:
            continue
        bank = meta_by_dir[direction_id]
        per_direction.append(
            {
                "direction_id": direction_id,
                "kind": bank["kind"],
                "direction_type": bank["direction_type"],
                "domain": bank["domain"],
                "status": bank["status"],
                "n_transcript_pairs": len(deltas),
                "mean_eps_prose": sum(prose_vals) / len(prose_vals),
                "mean_eps_tool": sum(tool_vals) / len(tool_vals),
                "mean_delta_tool_minus_prose": sum(deltas) / len(deltas),
                "deltas": deltas,
            }
        )

    def mean_delta(types: set[str], *, exclude_refusal: bool) -> tuple[float | None, list[str]]:
        subset = [
            d
            for d in per_direction
            if d["direction_type"] in types and not (exclude_refusal and d["kind"] == "refusal")
        ]
        if not subset:
            return None, []
        return (
            sum(d["mean_delta_tool_minus_prose"] for d in subset) / len(subset),
            [d["direction_id"] for d in subset],
        )

    safety_main_mean, safety_main_ids = mean_delta({"safety"}, exclude_refusal=True)
    safety_all_mean, safety_all_ids = mean_delta({"safety"}, exclude_refusal=False)
    controls_mean, control_ids = mean_delta(
        {"learned_control", "random_control"}, exclude_refusal=False
    )
    learned_mean, learned_ids = mean_delta({"learned_control"}, exclude_refusal=False)
    random_mean, random_ids = mean_delta({"random_control"}, exclude_refusal=False)

    interaction = None
    if safety_main_mean is not None and controls_mean is not None:
        interaction = safety_main_mean - controls_mean
    interaction_learned = (
        None
        if safety_main_mean is None or learned_mean is None
        else safety_main_mean - learned_mean
    )
    interaction_random = (
        None
        if safety_main_mean is None or random_mean is None
        else safety_main_mean - random_mean
    )

    # Bootstrap interaction over transcripts (resample transcript ids)
    all_tr_ids = sorted({tid for dmap in deltas_by_dir.values() for tid in dmap})
    rng = random.Random(args.seed)
    boot_interactions: list[float] = []
    for _ in range(400):
        sample = [rng.choice(all_tr_ids) for _ in all_tr_ids] if all_tr_ids else []

        def boot_mean_for(direction_ids: list[str]) -> float | None:
            vals = []
            for did in direction_ids:
                dmap = deltas_by_dir.get(did, {})
                sampled = [dmap[tid] for tid in sample if tid in dmap]
                if sampled:
                    vals.append(sum(sampled) / len(sampled))
            if not vals:
                return None
            return sum(vals) / len(vals)

        s = boot_mean_for(safety_main_ids)
        c = boot_mean_for(control_ids)
        if s is not None and c is not None:
            boot_interactions.append(s - c)

    boot_interactions.sort()
    if boot_interactions:
        lo = boot_interactions[int(0.025 * len(boot_interactions))]
        hi = boot_interactions[min(len(boot_interactions) - 1, int(0.975 * len(boot_interactions)))]
        boot_mean = sum(boot_interactions) / len(boot_interactions)
    else:
        lo = hi = boot_mean = None

    h1_support = bool(
        interaction is not None and lo is not None and interaction > 0 and lo > 0
    )
    # Learned-only is the cleaner Step-4-normalized contrast; do not overclaim from
    # pooled randoms if they alone drive the interaction.
    h1_vs_learned = bool(
        interaction_learned is not None and interaction_learned > 0.5
    )
    pilot_interpretation = (
        "Report pooled interaction for completeness, but interpret H1 against "
        "learned controls first. If pooled≫learned because random Δ is largely "
        "negative, treat pooled positivity as fragile on this pilot."
    )

    payload = {
        "model": model_key,
        "pilot": True,
        "evaluation_dataset": "mind_the_gap",
        "layer_k": layer_k,
        "layer_L": layer_L,
        "threshold": args.threshold,
        "n_windows": n_windows,
        "n_transcripts": len(transcripts),
        "primary_metric": "directional_sensitivity_epsilon_star",
        "interpretation_rule": (
            "Lower ε* ⇒ more geometric privilege. "
            "Δ = ε*_tool − ε*_prose; positive Δ ⇒ privilege loss in tool mode. "
            "H1: mean Δ_safety > mean Δ_control."
        ),
        "per_direction": per_direction,
        "aggregates": {
            "safety_excluding_blocked_refusal": {
                "n_directions": len(safety_main_ids),
                "mean_delta": safety_main_mean,
                "directions": safety_main_ids,
            },
            "safety_including_refusal_exploratory": {
                "n_directions": len(safety_all_ids),
                "mean_delta": safety_all_mean,
                "directions": safety_all_ids,
            },
            "controls_learned_and_random": {
                "n_directions": len(control_ids),
                "mean_delta": controls_mean,
                "directions": control_ids,
            },
            "controls_learned_only": {
                "n_directions": len(learned_ids),
                "mean_delta": learned_mean,
                "directions": learned_ids,
            },
            "controls_random_only": {
                "n_directions": len(random_ids),
                "mean_delta": random_mean,
                "directions": random_ids,
            },
            "interaction_safety_minus_controls": interaction,
            "interaction_safety_minus_learned": interaction_learned,
            "interaction_safety_minus_random": interaction_random,
            "interaction_boot_mean": boot_mean,
            "interaction_percentile_ci95": [lo, hi] if lo is not None else None,
            "h1_supported_by_pilot_ci": h1_support,
            "h1_supported_vs_learned_controls": h1_vs_learned,
            "pilot_interpretation": pilot_interpretation,
        },
        "refusal_arm": {
            "status": "blocked_by_section3_stability_gate",
            "note": (
                "Preregistration §5.2: do not run confirmatory refusal privilege "
                "comparison after stability FAIL. Refusal numbers above are exploratory."
            ),
        },
        "window_rows": [asdict(r) for r in rows],
        "status": (
            "pilot_complete_not_confirmatory"
            if interaction is not None
            else "pilot_incomplete"
        ),
        "caveats": [
            "Qwen3-0.6B bring-up only; confirmatory RQ1 requires Qwen3-32B.",
            "Step 1 mode gate is not validated; windows are usable but not confirmed.",
            "Refusal confirmatory arm blocked by §3 stability gate.",
            "Assistant Axis / harmlessness lack completed tool-mode stability gates on this pilot.",
            "Learned controls were Step-4 gated at F_mid and re-extracted at layer k for perturbation.",
            "Small n transcripts; interaction CI is percentile bootstrap over transcripts.",
            "Interpret learned-control interaction before pooled random+learned.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print("---")
    for d in per_direction:
        print(
            f"{d['direction_id']}: eps_prose={d['mean_eps_prose']:.4f} "
            f"eps_tool={d['mean_eps_tool']:.4f} "
            f"Δ={d['mean_delta_tool_minus_prose']:+.4f} "
            f"type={d['direction_type']} status={d['status']}"
        )
    print(
        f"interaction pooled={interaction} learned={interaction_learned} "
        f"random={interaction_random} CI95=[{lo},{hi}] "
        f"h1_pooled={h1_support} h1_learned={h1_vs_learned}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
