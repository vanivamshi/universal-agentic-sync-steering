#!/usr/bin/env python3
"""0.6B watch-list close-out (no 32B): power, blowup variants, writing L4.

1) Writing control: one documented pair expansion + L4 split-half BCa.
2) Random-Δ: recompute ε* with relative, absolute, and norm-matched blowup.
3) Power: GAP + τ-bench with one prose and one tool window per transcript
   (raises cluster n while keeping compute tractable), then BCa+joint CIs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
    direction_id: str
    direction_type: str
    epsilon_star: float
    blowup: float
    converged: bool
    n_tokens: int
    clean_norm: float
    blowup_mode: str


def epsilon_star(
    *,
    clean_L,
    forward_pert,
    layer_L: int,
    pool,
    threshold: float,
    max_iter: int,
    blowup_mode: str,
    ref_norm: float,
    eps_lo: float = 1e-4,
    eps_hi: float = 50.0,
) -> tuple[float, float, bool]:
    from activation_pipeline.analysis.sensitivity import residual_l2_blowup

    def blow(eps: float) -> float:
        pert = forward_pert(eps)
        pooled = pool(pert[layer_L])
        if blowup_mode == "relative":
            return residual_l2_blowup(clean_L, pooled, relative=True)
        if blowup_mode == "absolute":
            # threshold is absolute L2; compare raw distance
            return residual_l2_blowup(clean_L, pooled, relative=False)
        if blowup_mode == "norm_matched":
            return residual_l2_blowup(clean_L, pooled, denom=ref_norm)
        raise ValueError(blowup_mode)

    hi = eps_hi
    b_hi = blow(hi)
    if b_hi < threshold:
        return hi, b_hi, False
    lo = eps_lo
    b_lo = blow(lo)
    if b_lo >= threshold:
        return lo, b_lo, True
    best_eps, best_blow = hi, b_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        b_mid = blow(mid)
        if b_mid >= threshold:
            hi = mid
            best_eps, best_blow = mid, b_mid
        else:
            lo = mid
    return best_eps, best_blow, True


def bca_joint_from_rows(
    rows: list[WindowSens],
    meta: dict[str, dict],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    from activation_pipeline.analysis.bootstrap_ci import bca_ci, percentile_ci

    by_dir_tr: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"prose": [], "tool_call": []})
    )
    for r in rows:
        by_dir_tr[r.direction_id][r.transcript_id][r.window_kind].append(r.epsilon_star)

    deltas_by_dir: dict[str, dict[str, float]] = {}
    for did, tr_map in by_dir_tr.items():
        dmap = {}
        for tid, modes in tr_map.items():
            if modes["prose"] and modes["tool_call"]:
                p = sum(modes["prose"]) / len(modes["prose"])
                t = sum(modes["tool_call"]) / len(modes["tool_call"])
                dmap[tid] = t - p
        deltas_by_dir[did] = dmap

    all_tr = sorted({tid for m in deltas_by_dir.values() for tid in m})

    def mean_delta(ids: list[str], tr_ids: list[str]) -> float | None:
        vals = []
        for did in ids:
            sampled = [deltas_by_dir[did][t] for t in tr_ids if t in deltas_by_dir[did]]
            if sampled:
                vals.append(sum(sampled) / len(sampled))
        return sum(vals) / len(vals) if vals else None

    safety = [
        d
        for d, m in meta.items()
        if m["direction_type"] == "safety" and m["kind"] != "refusal"
    ]
    learned = [d for d, m in meta.items() if m["direction_type"] == "learned_control"]
    randoms = [d for d, m in meta.items() if m["direction_type"] == "random_control"]

    def pack(stat_fn, label: str) -> dict[str, Any]:
        point = stat_fn(all_tr)
        boots = []
        for b in range(n_boot):
            brng = random.Random(seed + 10_000 + b)
            sample = [brng.choice(all_tr) for _ in all_tr]
            val = stat_fn(sample)
            if val is not None:
                boots.append(val)
        jacks = []
        for leave in all_tr:
            val = stat_fn([t for t in all_tr if t != leave])
            if val is not None:
                jacks.append(val)
        lo_b, hi_b, diag = bca_ci(point, boots, jacks)
        lo_p, hi_p = percentile_ci(boots)
        return {
            "label": label,
            "point": point,
            "n_transcripts": len(all_tr),
            "bca_ci95": [lo_b, hi_b],
            "percentile_ci95": [lo_p, hi_p],
            "bca_excludes_zero": bool(lo_b > 0 or hi_b < 0),
            "bca_diagnostics": diag,
        }

    return {
        "n_transcripts": len(all_tr),
        "group_mean_delta": {
            "safety_ex_refusal": pack(lambda tr: mean_delta(safety, tr), "safety"),
            "learned": pack(lambda tr: mean_delta(learned, tr), "learned"),
            "random": pack(lambda tr: mean_delta(randoms, tr), "random"),
        },
        "interaction": {
            "safety_minus_learned": pack(
                lambda tr: (
                    None
                    if mean_delta(safety, tr) is None or mean_delta(learned, tr) is None
                    else mean_delta(safety, tr) - mean_delta(learned, tr)
                ),
                "safety_minus_learned",
            ),
            "safety_minus_random": pack(
                lambda tr: (
                    None
                    if mean_delta(safety, tr) is None or mean_delta(randoms, tr) is None
                    else mean_delta(safety, tr) - mean_delta(randoms, tr)
                ),
                "safety_minus_random",
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--layer-k", type=int, default=4)
    ap.add_argument("--layer-L", type=int, default=22)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=14)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "rq1_06b_watchlist.json",
    )
    args = ap.parse_args()

    import torch

    from activation_pipeline.analysis.sensitivity import LayerPerturbHooks
    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.controls import (
        build_random_orthogonal_controls,
        collect_pair_activations,
        full_directions_from_activations,
        learned_control_stability,
    )
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.windows import embed_windows_in_chat, tag_transcript_assistant_turns

    pairs_dir = ROOT / "data" / "contrast_pairs"
    dirs_dir = ROOT / "data" / "directions"

    # --- writing v2 pairs (do not run build_control_pairs as __main__: it SystemExits) ---
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_control_pairs", ROOT / "scripts" / "build_control_pairs.py"
    )
    bcp = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bcp)
    writing_v2 = pairs_dir / "domain_content_writing_pilot_v2.jsonl"
    bcp.write_pairs(
        writing_v2,
        [*bcp.DOMAIN_PAIRS["writing"], *bcp.WRITING_V2_ADDITIONS],
        kind="domain_content",
        positive_label="narrative",
        negative_label="expository",
        domain="writing",
    )

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    device = next(loaded.model.parameters()).device
    layer_k, layer_L = args.layer_k, args.layer_L
    print(f"watchlist model={LOCAL_MODEL_KEY} device={device_map} k={layer_k} L={layer_L}", flush=True)

    # --- writing L4 stability (v1 vs v2) ---
    writing_stability = {}
    for label, path in (
        ("v1", pairs_dir / "domain_content_writing_pilot.jsonl"),
        ("v2", writing_v2),
    ):
        pairs = load_jsonl(path)
        print(f"writing {label}: collecting activations n_pairs={len(pairs)}", flush=True)
        pos, neg = collect_pair_activations(loaded, pairs, layers=[layer_k])
        stab = learned_control_stability(
            pos,
            neg,
            layers=[layer_k],
            direction_id=f"domain_content_writing_{label}",
            kind="domain_content",
            domain="writing",
            n_boot=200,
            n_splits_per_stat=10,
            seed=args.seed,
            threshold=0.70,
            required_layers=1,
        )
        writing_stability[label] = stab.to_dict()
        row = stab.layers[0]
        print(
            f"writing {label}: L{layer_k} cos={row['split_half_cosine']:.3f} "
            f"BCa={row['bca_ci95']} pass={stab.passed}",
            flush=True,
        )


    v2_lo = float(writing_stability["v2"]["layers"][0]["bca_ci95"][0])
    writing_path = (
        writing_v2 if v2_lo >= 0.70 else pairs_dir / "domain_content_writing_pilot.jsonl"
    )
    print(f"writing pairs for privilege: {writing_path.name} (v2 BCa lo={v2_lo:.3f})")

    direction_bank: list[dict[str, Any]] = []
    safety_tensors = []
    for kind, path, status in [
        ("refusal", dirs_dir / "refusal_prose.jsonl", "blocked_exploratory"),
        ("assistant_axis", dirs_dir / "assistant_axis_prose.jsonl", "provisional"),
        ("harmlessness", dirs_dir / "harmlessness_prose.jsonl", "provisional"),
    ]:
        row = load_direction_at_layer(path, layer=layer_k)
        if row is None:
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
        ("domain_content_writing", "domain_content", "writing", writing_path),
    ]
    learned_tensors = []
    for direction_id, kind, domain, path in control_specs:
        pairs = load_jsonl(path)
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
                "status": "learned_control",
                "vector": vec,
            }
        )

    random_dirs, _ = build_random_orthogonal_controls(
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

    def select_jobs(transcripts: list[dict]) -> list[dict]:
        out = []
        for tr in transcripts:
            chosen = {"prose": None, "tool_call": None}
            for tagged in tag_transcript_assistant_turns(tr, loaded.tokenizer):
                emb = embed_windows_in_chat(
                    loaded.tokenizer, tr["messages"], tagged.message_index, tagged
                )
                for window in emb.windows:
                    idxs = window.token_indices()
                    if not idxs or window.kind not in chosen:
                        continue
                    if chosen[window.kind] is not None:
                        continue
                    chosen[window.kind] = (tagged, emb, window, idxs)
                if chosen["prose"] and chosen["tool_call"]:
                    break
            if not (chosen["prose"] and chosen["tool_call"]):
                continue
            for kind in ("prose", "tool_call"):
                tagged, emb, window, idxs = chosen[kind]
                out.append(
                    {
                        "transcript_id": tr["transcript_id"],
                        "domain": tr["domain"],
                        "tagged": tagged,
                        "emb": emb,
                        "window": window,
                        "idxs": idxs,
                    }
                )
        return out

    def cache_cleans(job_list: list[dict]):
        clean_cache = {}
        prose_norms, tool_norms = [], []
        for job in job_list:
            key = (job["transcript_id"], job["window"].kind, job["tagged"].message_index)
            ids = torch.tensor([job["emb"].input_ids], device=device)
            attn = torch.ones_like(ids)
            hooks = LayerPerturbHooks(
                loaded.model,
                layer_k=layer_k,
                direction=torch.ones(loaded.spec.hidden_size),
                token_indices=job["idxs"],
                read_layers=[layer_L],
            )
            clean = hooks.run(ids, attn, 0.0)
            pooled = clean[layer_L][0, job["idxs"], :].float().mean(0)
            nrm = float(torch.linalg.norm(pooled))
            clean_cache[key] = (ids, attn, pooled, nrm)
            (prose_norms if job["window"].kind == "prose" else tool_norms).append(nrm)
        mean_all = sum(prose_norms + tool_norms) / len(prose_norms + tool_norms)
        return (
            clean_cache,
            mean_all,
            sum(prose_norms) / len(prose_norms),
            sum(tool_norms) / len(tool_norms),
        )

    def run_mode(job_list, clean_cache, *, blowup_mode, threshold, ref, label):
        print(f"=== {label} mode={blowup_mode} thr={threshold:.4f} n_win={len(job_list)} ===")
        rows: list[WindowSens] = []
        for job in job_list:
            key = (job["transcript_id"], job["window"].kind, job["tagged"].message_index)
            ids, attn, clean_L, nrm = clean_cache[key]
            idxs = job["idxs"]

            def pool(t, token_idxs=idxs):
                return t[0, token_idxs, :].float().mean(0)

            for dinfo in direction_bank:
                if (
                    dinfo["direction_type"] == "learned_control"
                    and dinfo["domain"] is not None
                    and dinfo["domain"] != job["domain"]
                ):
                    continue
                hooks = LayerPerturbHooks(
                    loaded.model,
                    layer_k=layer_k,
                    direction=dinfo["vector"],
                    token_indices=idxs,
                    read_layers=[layer_L],
                )

                def forward_pert(eps, h=hooks, i=ids, a=attn):
                    return h.run(i, a, epsilon=eps)

                eps_star, blowup, converged = epsilon_star(
                    clean_L=clean_L,
                    forward_pert=forward_pert,
                    layer_L=layer_L,
                    pool=pool,
                    threshold=threshold,
                    max_iter=args.max_iter,
                    blowup_mode=blowup_mode,
                    ref_norm=ref,
                )
                rows.append(
                    WindowSens(
                        transcript_id=job["transcript_id"],
                        domain=job["domain"],
                        message_index=job["tagged"].message_index,
                        window_kind=job["window"].kind,
                        direction_id=dinfo["direction_id"],
                        direction_type=dinfo["direction_type"],
                        epsilon_star=eps_star,
                        blowup=blowup,
                        converged=converged,
                        n_tokens=len(idxs),
                        clean_norm=nrm,
                        blowup_mode=blowup_mode,
                    )
                )
            print(f"  {job['transcript_id']} {job['window'].kind}")

        meta = {d["direction_id"]: d for d in direction_bank}
        per_direction = []
        for did in {r.direction_id for r in rows}:
            tr_map = defaultdict(lambda: {"prose": [], "tool_call": []})
            for r in rows:
                if r.direction_id == did:
                    tr_map[r.transcript_id][r.window_kind].append(r.epsilon_star)
            deltas = []
            for modes_ in tr_map.values():
                if modes_["prose"] and modes_["tool_call"]:
                    p = sum(modes_["prose"]) / len(modes_["prose"])
                    t = sum(modes_["tool_call"]) / len(modes_["tool_call"])
                    deltas.append(t - p)
            if not deltas:
                continue
            m = meta[did]
            per_direction.append(
                {
                    "direction_id": did,
                    "kind": m["kind"],
                    "direction_type": m["direction_type"],
                    "mean_delta": sum(deltas) / len(deltas),
                    "n_transcript_pairs": len(deltas),
                }
            )
        cis = bca_joint_from_rows(rows, meta, n_boot=args.n_boot, seed=args.seed)
        inter = cis["interaction"]["safety_minus_learned"]
        rand = cis["group_mean_delta"]["random"]
        print(
            f"  learned_inter={inter['point']:+.3f} BCa={inter['bca_ci95']} "
            f"excl0={inter['bca_excludes_zero']}"
        )
        print(
            f"  random_Δ={rand['point']:+.3f} BCa={rand['bca_ci95']} "
            f"excl0={rand['bca_excludes_zero']}"
        )
        return {
            "threshold": threshold,
            "ref_norm": ref,
            "n_windows": len(job_list),
            "per_direction": per_direction,
            "aggregates": cis,
            "window_rows": [asdict(r) for r in rows],
        }

    gap = load_transcripts(ROOT / "data" / "transcripts" / "real" / "gap_agentic.jsonl")
    tau = load_transcripts(ROOT / "data" / "transcripts" / "real" / "tau_bench.jsonl")
    preferred = {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}
    gap = [t for t in gap if t.get("source") in preferred] or gap
    gap_jobs = select_jobs(gap)
    power_jobs = select_jobs(gap + tau)
    print(f"gap_jobs={len(gap_jobs)} power_jobs={len(power_jobs)}")

    gap_cache, gap_mean, gap_prose, gap_tool = cache_cleans(gap_jobs)
    power_cache, power_mean, power_prose, power_tool = cache_cleans(power_jobs)

    blowup_on_gap = {
        "relative": run_mode(
            gap_jobs, gap_cache, blowup_mode="relative",
            threshold=args.threshold, ref=gap_prose, label="gap_variants",
        ),
        "absolute": run_mode(
            gap_jobs, gap_cache, blowup_mode="absolute",
            threshold=args.threshold * gap_mean, ref=gap_mean, label="gap_variants",
        ),
        "norm_matched": run_mode(
            gap_jobs, gap_cache, blowup_mode="norm_matched",
            threshold=args.threshold, ref=gap_prose, label="gap_variants",
        ),
    }
    power_relative = run_mode(
        power_jobs, power_cache, blowup_mode="relative",
        threshold=args.threshold, ref=power_prose, label="power_gap_plus_tau",
    )

    payload = {
        "model": LOCAL_MODEL_KEY,
        "scope": "06b_watchlist_no_32b",
        "layer_k": layer_k,
        "layer_L": layer_L,
        "writing_control_l4": writing_stability,
        "writing_pairs_used": str(writing_path.relative_to(ROOT)),
        "writing_v2_bca_lo": v2_lo,
        "writing_v2_bca_lo_clears_0_70": bool(v2_lo >= 0.70),
        "norm_summary": {
            "gap": {
                "mean_all": gap_mean,
                "mean_prose": gap_prose,
                "mean_tool": gap_tool,
                "tool_over_prose": gap_tool / gap_prose,
            },
            "power": {
                "mean_all": power_mean,
                "mean_prose": power_prose,
                "mean_tool": power_tool,
                "tool_over_prose": power_tool / power_prose,
            },
        },
        "power_design": {
            "datasets": ["gap_agentic", "tau_bench"],
            "window_rule": "one_prose_and_one_tool_window_per_transcript",
            "n_windows": len(power_jobs),
            "n_transcript_pairs": len(power_jobs) // 2,
        },
        "blowup_variants_on_gap": {
            k: {kk: vv for kk, vv in v.items() if kk != "window_rows"}
            for k, v in blowup_on_gap.items()
        },
        "power_relative": {
            kk: vv for kk, vv in power_relative.items() if kk != "window_rows"
        },
        "verdicts": {
            "learned_interaction_power": power_relative["aggregates"]["interaction"][
                "safety_minus_learned"
            ],
            "random_delta_by_blowup_mode": {
                mode: blowup_on_gap[mode]["aggregates"]["group_mean_delta"]["random"]
                for mode in blowup_on_gap
            },
        },
    }
    # keep full window rows only for relative GAP in a sidecar? skip to save space
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
