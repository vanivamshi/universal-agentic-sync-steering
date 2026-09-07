#!/usr/bin/env python3
"""Cheap causal diagnostics for three unexplained findings.

1) PC1 sign flip — relative vs absolute vs norm_matched Δ
2) L4 near-90° / rank collapse — equal-n prose subsample + norm-normalized PCA
3) rand_02 residual — draw n_random≥64 and see how often outliers occur

Run order matches the design: all three cheap checks before expensive follow-ups.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dims_at_frac(explained: np.ndarray, frac: float = 0.90) -> int:
    c = np.cumsum(explained)
    return int(np.searchsorted(c, frac) + 1)


def fit_pca_dims(X: np.ndarray) -> dict[str, Any]:
    from activation_pipeline.analysis.persona_pca import fit_pca, principal_angles_deg

    pca = fit_pca(X)
    return {
        "n": int(X.shape[0]),
        "hidden": int(X.shape[1]),
        "dims_for_90": pca.to_dict()["dims_for_90"],
        "dims_for_80": pca.to_dict()["dims_for_80"],
        "pc1_var": float(pca.explained_variance_ratio[0]),
        "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        "components": pca.components,
    }


# ---------------------------------------------------------------------------
# 2) Rank / divergence controls (no model — activation cache)
# ---------------------------------------------------------------------------
def run_rank_equal_n(
    cache_path: Path,
    *,
    layers: list[int],
    n_tool_target: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    from activation_pipeline.analysis.persona_pca import principal_angles_deg
    from activation_pipeline.cache_io import ActivationCache

    cache = ActivationCache.load(cache_path)
    out: dict[str, Any] = {
        "cache": str(cache_path),
        "n_boot": n_boot,
        "protocol": (
            "Window-mean PCA on GAP cache. Equal-n: subsample prose to n_tool "
            "and recompute dims@90%. Also norm-normalize rows before PCA. "
            "Note: Phase-1 persona PCA already used equal n_roles=32; this "
            "tests the window-count version of H_b."
        ),
        "by_layer": {},
    }
    rng = random.Random(seed)

    for layer in layers:
        prose = [
            r.mean_tensor(layer).numpy()
            for r in cache.records
            if r.window_kind == "prose" and str(layer) in r.layer_means
        ]
        tool = [
            r.mean_tensor(layer).numpy()
            for r in cache.records
            if r.window_kind == "tool_call" and str(layer) in r.layer_means
        ]
        if len(prose) < 4 or len(tool) < 4:
            out["by_layer"][str(layer)] = {"error": "too few windows"}
            continue
        Xp = np.stack(prose)
        Xt = np.stack(tool)
        n_eq = min(n_tool_target, len(tool), len(prose))

        def row_norm(X: np.ndarray) -> np.ndarray:
            nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / nrm

        full_p = fit_pca_dims(Xp)
        full_t = fit_pca_dims(Xt)
        full_p_n = fit_pca_dims(row_norm(Xp))
        full_t_n = fit_pca_dims(row_norm(Xt))

        # equal-n bootstrap: subsample prose (and tool) to n_eq
        prose_dims = []
        tool_dims = []
        prose_dims_n = []
        tool_dims_n = []
        angles = []
        angles_n = []
        for b in range(n_boot):
            pi = rng.sample(range(len(prose)), n_eq)
            ti = rng.sample(range(len(tool)), n_eq)
            Xp_b = Xp[pi]
            Xt_b = Xt[ti]
            pp = fit_pca_dims(Xp_b)
            tt = fit_pca_dims(Xt_b)
            prose_dims.append(pp["dims_for_90"])
            tool_dims.append(tt["dims_for_90"])
            ang = principal_angles_deg(pp["components"][:4], tt["components"][:4], rank=4)
            angles.append(ang["mean_angle_deg"])
            ppn = fit_pca_dims(row_norm(Xp_b))
            ttn = fit_pca_dims(row_norm(Xt_b))
            prose_dims_n.append(ppn["dims_for_90"])
            tool_dims_n.append(ttn["dims_for_90"])
            ang_n = principal_angles_deg(ppn["components"][:4], ttn["components"][:4], rank=4)
            angles_n.append(ang_n["mean_angle_deg"])

        gap_full = full_p["dims_for_90"] - full_t["dims_for_90"]
        gap_eq_mean = float(np.mean(prose_dims) - np.mean(tool_dims))
        # Does equal-n close most of the rank gap?
        frac_explained = 1.0 - (gap_eq_mean / gap_full) if abs(gap_full) > 1e-9 else 0.0

        layer_out = {
            "n_prose": len(prose),
            "n_tool": len(tool),
            "n_equal": n_eq,
            "full": {
                "prose_dims90": full_p["dims_for_90"],
                "tool_dims90": full_t["dims_for_90"],
                "prose_pc1": full_p["pc1_var"],
                "tool_pc1": full_t["pc1_var"],
                "rank_gap": gap_full,
            },
            "full_normed": {
                "prose_dims90": full_p_n["dims_for_90"],
                "tool_dims90": full_t_n["dims_for_90"],
                "rank_gap": full_p_n["dims_for_90"] - full_t_n["dims_for_90"],
            },
            "equal_n_boot": {
                "prose_dims90_mean": float(np.mean(prose_dims)),
                "prose_dims90_std": float(np.std(prose_dims)),
                "tool_dims90_mean": float(np.mean(tool_dims)),
                "tool_dims90_std": float(np.std(tool_dims)),
                "rank_gap_mean": gap_eq_mean,
                "mean_angle_mean": float(np.mean(angles)),
                "mean_angle_std": float(np.std(angles)),
                "frac_of_full_gap_closed_by_equal_n": float(frac_explained),
            },
            "equal_n_boot_normed": {
                "prose_dims90_mean": float(np.mean(prose_dims_n)),
                "tool_dims90_mean": float(np.mean(tool_dims_n)),
                "rank_gap_mean": float(np.mean(prose_dims_n) - np.mean(tool_dims_n)),
                "mean_angle_mean": float(np.mean(angles_n)),
            },
            "persona_pca_note": (
                "Phase-1 persona PCA used equal n_roles=32 in both modes, so "
                "unequal window counts cannot explain that rank collapse. "
                "This block tests window-mean PCA H_b only."
            ),
        }
        # Decision
        if frac_explained >= 0.7:
            layer_out["decision"] = "H_b_LIKELY — equal-n closes ≥70% of rank gap"
        elif abs(gap_eq_mean) < 2 and abs(gap_full) >= 4:
            layer_out["decision"] = "H_b_LIKELY — equal-n nearly eliminates gap"
        elif gap_eq_mean >= 0.5 * gap_full and gap_full > 0:
            layer_out["decision"] = "H_b_PARTIAL — equal-n explains some but not most"
        else:
            layer_out["decision"] = "H_b_UNLIKELY — rank gap survives equal-n"
        out["by_layer"][str(layer)] = layer_out
        print(
            f"[rank] L{layer}: full gap={gap_full} → equal-n gap≈{gap_eq_mean:.2f} "
            f"({frac_explained:.0%} closed) → {layer_out['decision']}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# 1) PC1 metric check + 3) expanded randoms (shared model load)
# ---------------------------------------------------------------------------
def run_privilege_blowup_modes(
    loaded,
    *,
    directions: list[dict],
    transcripts: list[dict],
    layer_k: int,
    layer_L: int,
    max_transcripts: int,
    threshold_rel: float,
    max_iter: int,
    modes: list[str],
) -> dict[str, Any]:
    """Compute Δ under relative / absolute / norm_matched for given directions."""
    from activation_pipeline.analysis.sensitivity import LayerPerturbHooks, residual_l2_blowup
    from activation_pipeline.windows import embed_windows_in_chat, tag_transcript_assistant_turns
    from scripts.run_06b_watchlist import epsilon_star

    device = next(loaded.model.parameters()).device
    preferred = {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}
    transcripts = [t for t in transcripts if t.get("source") in preferred] or transcripts
    transcripts = transcripts[:max_transcripts]

    # Build jobs + clean cache once
    jobs = []
    for tr in transcripts:
        for tagged in tag_transcript_assistant_turns(tr, loaded.tokenizer):
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
                jobs.append(
                    {
                        "transcript_id": tr["transcript_id"],
                        "domain": tr["domain"],
                        "window_kind": window.kind,
                        "idxs": idxs,
                        "input_ids": input_ids,
                        "attn": attn,
                        "message_index": tagged.message_index,
                    }
                )

    clean_cache = {}
    prose_norms, tool_norms = [], []
    for job in jobs:
        key = (job["transcript_id"], job["window_kind"], job["message_index"])
        hooks = LayerPerturbHooks(
            loaded.model,
            layer_k=layer_k,
            direction=torch.ones(loaded.spec.hidden_size),
            token_indices=job["idxs"],
            read_layers=[layer_L],
        )
        clean = hooks.run(job["input_ids"], job["attn"], 0.0)
        pooled = clean[layer_L][0, job["idxs"], :].float().mean(0)
        nrm = float(torch.linalg.norm(pooled))
        clean_cache[key] = (pooled, nrm)
        (prose_norms if job["window_kind"] == "prose" else tool_norms).append(nrm)
    mean_all = sum(prose_norms + tool_norms) / max(len(prose_norms) + len(tool_norms), 1)
    mean_prose = sum(prose_norms) / max(len(prose_norms), 1)

    bank = []
    for d in directions:
        bank.append(
            {
                "direction_id": d["direction_id"],
                "kind": d.get("kind", "unknown"),
                "vector": torch.tensor(d["vector"], dtype=torch.float32),
            }
        )

    mode_results: dict[str, Any] = {}
    for blowup_mode in modes:
        if blowup_mode == "relative":
            thr, ref = threshold_rel, mean_prose
        elif blowup_mode == "absolute":
            thr, ref = threshold_rel * mean_all, mean_all
        else:  # norm_matched
            thr, ref = threshold_rel, mean_prose

        print(f"[priv] mode={blowup_mode} thr={thr:.4f} n_jobs={len(jobs)} n_dirs={len(bank)}", flush=True)
        by_dir: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: {"prose": [], "tool_call": []})
        )
        for ji, job in enumerate(jobs):
            key = (job["transcript_id"], job["window_kind"], job["message_index"])
            clean_L, nrm = clean_cache[key]
            idxs = job["idxs"]

            def pool(t, token_idxs=idxs):
                return t[0, token_idxs, :].float().mean(0)

            for dinfo in bank:
                hooks = LayerPerturbHooks(
                    loaded.model,
                    layer_k=layer_k,
                    direction=dinfo["vector"],
                    token_indices=idxs,
                    read_layers=[layer_L],
                )

                def forward_pert(eps, h=hooks, i=job["input_ids"], a=job["attn"]):
                    return h.run(i, a, epsilon=eps)

                eps_star, blowup, converged = epsilon_star(
                    clean_L=clean_L,
                    forward_pert=forward_pert,
                    layer_L=layer_L,
                    pool=pool,
                    threshold=thr,
                    max_iter=max_iter,
                    blowup_mode=blowup_mode,
                    ref_norm=ref,
                )
                by_dir[dinfo["direction_id"]][job["transcript_id"]][job["window_kind"]].append(
                    eps_star
                )
            if (ji + 1) % 4 == 0 or ji + 1 == len(jobs):
                print(f"  {ji+1}/{len(jobs)} windows", flush=True)

        per_direction = []
        for did, tr_map in by_dir.items():
            deltas, prose_vals, tool_vals = [], [], []
            for modes_ in tr_map.values():
                if modes_["prose"] and modes_["tool_call"]:
                    p = sum(modes_["prose"]) / len(modes_["prose"])
                    t = sum(modes_["tool_call"]) / len(modes_["tool_call"])
                    prose_vals.append(p)
                    tool_vals.append(t)
                    deltas.append(t - p)
            if not deltas:
                continue
            kind = next(d["kind"] for d in bank if d["direction_id"] == did)
            per_direction.append(
                {
                    "direction_id": did,
                    "kind": kind,
                    "mean_eps_prose": sum(prose_vals) / len(prose_vals),
                    "mean_eps_tool": sum(tool_vals) / len(tool_vals),
                    "mean_delta": sum(deltas) / len(deltas),
                    "deltas": deltas,
                    "n_pairs": len(deltas),
                    "sign": "positive" if sum(deltas) / len(deltas) > 0 else "negative",
                }
            )
        per_direction.sort(key=lambda d: d["direction_id"])
        mode_results[blowup_mode] = {
            "threshold": thr,
            "ref_norm": ref,
            "mean_clean_norm_all": mean_all,
            "mean_clean_norm_prose": mean_prose,
            "tool_over_prose_norm": (sum(tool_norms) / max(len(tool_norms), 1)) / max(mean_prose, 1e-12),
            "per_direction": per_direction,
        }
        for d in per_direction:
            print(f"  {d['direction_id']}: Δ={d['mean_delta']:+.3f} ({d['sign']})", flush=True)

    # PC1 decision across modes
    pc1_signs = {}
    for mode, blk in mode_results.items():
        row = next((d for d in blk["per_direction"] if "pc1" in d["direction_id"]), None)
        if row:
            pc1_signs[mode] = row["mean_delta"]
    relative_neg = pc1_signs.get("relative", 0) < 0
    abs_pos = pc1_signs.get("absolute", 0) > 0
    nm_pos = pc1_signs.get("norm_matched", 0) > 0
    if relative_neg and abs_pos and nm_pos:
        pc1_decision = "H_b_CONFIRMED — PC1 sign flips under absolute/norm_matched (same as AA)"
    elif relative_neg and (pc1_signs.get("absolute", 0) < 0) and (pc1_signs.get("norm_matched", 0) < 0):
        pc1_decision = "H_b_REJECTED — PC1 Δ<0 survives preferred metrics; run depth/content/stability"
    else:
        pc1_decision = "H_b_UNCLEAR — mixed signs across metrics; inspect per_direction"

    return {
        "layer_k": layer_k,
        "layer_L": layer_L,
        "n_jobs": len(jobs),
        "n_transcripts": len(transcripts),
        "pc1_delta_by_mode": pc1_signs,
        "pc1_decision": pc1_decision,
        "by_blowup_mode": mode_results,
    }


def run_expanded_randoms(
    loaded,
    *,
    transcripts: list[dict],
    layer_k: int,
    layer_L: int,
    n_random: int,
    max_transcripts: int,
    threshold_rel: float,
    max_iter: int,
    seed: int,
    safety_dirs: list[dict],
) -> dict[str, Any]:
    """Draw n_random orthogonal controls; Δ under absolute + norm_matched."""
    from activation_pipeline.controls import build_random_orthogonal_controls
    from activation_pipeline.directions import DirectionVector

    # Fixed basis = safety / persona dirs at layer_k
    fixed = []
    for d in safety_dirs:
        if int(d["layer"]) != layer_k:
            continue
        fixed.append(
            DirectionVector(
                direction_id=d["direction_id"],
                kind=d.get("kind", "fixed"),
                mode="prose",
                layer=layer_k,
                vector=d["vector"],
                n_pos=0,
                n_neg=0,
            )
        )
    directions, diagnostics = build_random_orthogonal_controls(
        n_random=n_random,
        hidden_size=loaded.spec.hidden_size,
        layer=layer_k,
        seed=seed,
        fixed_basis=[torch.tensor(f.vector, dtype=torch.float32) for f in fixed],
    )
    normed = [
        {
            "direction_id": d.direction_id,
            "kind": "random_control",
            "vector": d.vector,
            "layer": layer_k,
        }
        for d in directions
    ]

    print(f"[random] n={len(normed)} orth_ok={diagnostics.passed}", flush=True)
    # Absolute first (preferred metric); closes frequency question without 2× cost.
    priv = run_privilege_blowup_modes(
        loaded,
        directions=normed,
        transcripts=transcripts,
        layer_k=layer_k,
        layer_L=layer_L,
        max_transcripts=max_transcripts,
        threshold_rel=threshold_rel,
        max_iter=max_iter,
        modes=["absolute"],
    )

    # Prior watchlist (absolute): rand_02 Δ≈+1.08 while rand_01/03 were ≈−4 to −5.4.
    # "rand_02-like" = |Δ − mean| ≥ 2σ OR residual vs mean ≥ prior residual scale (~2).
    summary = {}
    for mode in ("absolute",):
        deltas = [d["mean_delta"] for d in priv["by_blowup_mode"][mode]["per_direction"]]
        arr = np.array(deltas, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        # Extreme: ≥2σ from mean, or |Δ−mean|≥2.0 (prior residual scale)
        extreme_mask = (np.abs(arr - mean) >= max(2.0 * std, 1e-9)) | (np.abs(arr - mean) >= 2.0)
        extreme = [float(x) for x in arr[extreme_mask]]
        n_extreme = len(extreme)
        summary[mode] = {
            "n": len(arr),
            "mean_delta": mean,
            "std_delta": std,
            "min": float(arr.min()) if len(arr) else None,
            "max": float(arr.max()) if len(arr) else None,
            "n_extreme_ge_2sigma_or_residual_ge_2": n_extreme,
            "frac_extreme": n_extreme / max(len(arr), 1),
            "extreme_deltas": sorted(extreme, key=abs, reverse=True)[:10],
            "percentiles": {
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            },
        }
        print(
            f"[random] {mode}: meanΔ={mean:+.3f} sd={std:.3f} "
            f"extreme={n_extreme}/{len(arr)} ({100*n_extreme/max(len(arr),1):.1f}%)",
            flush=True,
        )

    abs_frac = summary["absolute"]["frac_extreme"]
    if abs_frac <= 0.15:
        decision = (
            "SAMPLING_VARIANCE — extremes are uncommon (~"
            f"{100*abs_frac:.0f}%); rand_02-class residuals are in the tail of small-n luck"
        )
    elif abs_frac >= 0.35:
        decision = (
            "CLUSTER — many directions behave like rand_02; not pure sampling noise; "
            "needs content/alignment probe"
        )
    else:
        decision = "HEAVY_TAIL — moderate extreme rate; more draws or content probe warranted"

    return {
        "n_random": len(normed),
        "orthogonality": diagnostics.to_dict(),
        "summary_by_mode": summary,
        "decision": decision,
        "n_jobs": priv["n_jobs"],
        "n_transcripts": priv["n_transcripts"],
        "per_direction_absolute": priv["by_blowup_mode"]["absolute"]["per_direction"],
    }


def asdict_dir(d) -> dict:
    return {
        "direction_id": d.direction_id,
        "vector": d.vector if isinstance(d.vector, list) else list(d.vector),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-rank", action="store_true")
    ap.add_argument("--skip-pc1", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--n-random", type=int, default=64)
    ap.add_argument("--max-transcripts", type=int, default=8)
    ap.add_argument("--n-boot-rank", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=18)
    ap.add_argument("--layer-k", type=int, default=4)
    ap.add_argument("--layer-L", type=int, default=22)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "unexplained_diagnostics.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    report: dict[str, Any] = {
        "model": "qwen3-0.6b",
        "date": "2026-08-09",
        "purpose": "Cheap causal probes for PC1 sign, rank collapse, rand_02 residual",
    }

    # --- 2) Rank (no model) ---
    if not args.skip_rank:
        print("\n===== RANK / EQUAL-N =====", flush=True)
        report["rank_equal_n"] = run_rank_equal_n(
            ROOT / "data" / "activations" / "real_gap.json",
            layers=[args.layer_k, args.layer_L],
            n_tool_target=24,
            n_boot=args.n_boot_rank,
            seed=args.seed,
        )
        # Persona PCA already equal-n note from phase1 artifact
        p1 = ROOT / "data" / "results" / "phase1_persona_pca.json"
        if p1.exists():
            phase1 = json.loads(p1.read_text())
            report["persona_pca_equal_n_roles"] = {
                "n_roles": phase1.get("n_roles"),
                "note": (
                    "Persona PCA prose and tool matrices both used the same n_roles; "
                    "H_b (unequal sample size) does NOT explain Phase-1 dims@90% collapse. "
                    "Window equal-n block above is a separate test."
                ),
                "phase1_dims90": {
                    L: {
                        "prose": blk.get("prose", {}).get("pca", {}).get("dims_for_90"),
                        "tool": blk.get("tool", {}).get("pca", {}).get("dims_for_90"),
                    }
                    for L, blk in phase1.get("by_layer", {}).items()
                },
            }

    need_model = not (args.skip_pc1 and args.skip_random)
    if need_model:
        from activation_pipeline.caching import load_transcripts
        from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
        from activation_pipeline.loader import load_model_and_tokenizer

        print("\n===== LOAD MODEL =====", flush=True)
        loaded = load_model_and_tokenizer(
            LOCAL_MODEL_KEY,
            device_map=resolve_device_map(None),
            dtype="float32",
            local_files_only=True,
        )
        transcripts = load_transcripts(
            ROOT / "data" / "transcripts" / "real" / "gap_agentic.jsonl"
        )
        persona_dirs = load_jsonl(
            ROOT / "data" / "directions" / f"persona_pca_prose_L{args.layer_k}.jsonl"
        )
        # AA + PC0-4
        persona_bank = [
            d
            for d in persona_dirs
            if int(d["layer"]) == args.layer_k
            and (
                d.get("kind") == "assistant_axis"
                or (
                    d.get("kind") == "persona_pc"
                    and int(d.get("meta", {}).get("pc_index", 99)) < 5
                )
            )
        ]

        if not args.skip_pc1:
            print("\n===== PC1 METRIC CHECK =====", flush=True)
            report["pc1_metric"] = run_privilege_blowup_modes(
                loaded,
                directions=persona_bank,
                transcripts=transcripts,
                layer_k=args.layer_k,
                layer_L=args.layer_L,
                max_transcripts=args.max_transcripts,
                threshold_rel=args.threshold,
                max_iter=args.max_iter,
                modes=["relative", "absolute", "norm_matched"],
            )
            print(f"PC1 decision: {report['pc1_metric']['pc1_decision']}", flush=True)

        if not args.skip_random:
            print("\n===== EXPANDED RANDOMS =====", flush=True)
            report["random_expand"] = run_expanded_randoms(
                loaded,
                transcripts=transcripts,
                layer_k=args.layer_k,
                layer_L=args.layer_L,
                n_random=args.n_random,
                max_transcripts=args.max_transcripts,
                threshold_rel=args.threshold,
                max_iter=args.max_iter,
                seed=args.seed,
                safety_dirs=persona_bank,
            )
            print(f"Random decision: {report['random_expand']['decision']}", flush=True)

    report["elapsed_s"] = time.time() - t0
    # Strip bulky components arrays from rank if any leaked
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "components"}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    args.out.write_text(json.dumps(_clean(report), indent=2) + "\n")
    print(f"\nwrote {args.out} elapsed={report['elapsed_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
