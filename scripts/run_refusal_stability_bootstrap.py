#!/usr/bin/env python3
"""Joint BCa bootstrap CIs for layer-matched refusal stability (prose↔tool + gap)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    ap.add_argument("--no-local-files-only", action="store_false", dest="local_files_only")
    ap.add_argument("--layers", type=int, nargs="*", default=[4, 14, 22])
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prose-pairs",
        type=Path,
        default=ROOT / "data" / "contrast_pairs" / "refusal_pilot.jsonl",
    )
    ap.add_argument(
        "--tool-rows",
        type=Path,
        default=ROOT / "data" / "contrast_pairs" / "refusal_tool_pilot.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "refusal_stability_layer_matched.json",
    )
    args = ap.parse_args()

    import torch

    from activation_pipeline.analysis.bootstrap_ci import bca_ci, percentile_ci
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.directions import cosine, last_token_residual, mean_difference_direction
    from activation_pipeline.hooks import ResidualStreamHooks
    from activation_pipeline.loader import load_model_and_tokenizer

    model_key = args.model or LOCAL_MODEL_KEY
    assert_model_fits_machine(model_key)
    device_map = resolve_device_map(args.device)
    loaded = load_model_and_tokenizer(
        model_key,
        device_map=device_map,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    layers = [int(l) for l in args.layers]
    prose_pairs = [json.loads(l) for l in args.prose_pairs.read_text().splitlines() if l.strip()]
    tool_rows = [json.loads(l) for l in args.tool_rows.read_text().splitlines() if l.strip()]
    tool_pos = [r for r in tool_rows if r.get("arm") == "positive"]
    tool_neg = [r for r in tool_rows if r.get("arm") == "negative"]
    print(
        f"joint-BCa model={model_key} layers={layers} "
        f"n_prose={len(prose_pairs)} tool={len(tool_pos)}/{len(tool_neg)} n_boot={args.n_boot}"
    )

    prose_pos_acts: list[dict[int, torch.Tensor]] = []
    prose_neg_acts: list[dict[int, torch.Tensor]] = []
    for pair in prose_pairs:
        for text, bucket in (
            (pair["positive"], prose_pos_acts),
            (pair["negative"], prose_neg_acts),
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ]
            bucket.append(last_token_residual(loaded, messages, layers=layers))

    def tool_act(row: dict) -> dict[int, torch.Tensor]:
        messages = [
            {
                "role": "system",
                "content": "You are an agent that uses tools. Prefer structured tool calls.",
            },
            {"role": "user", "content": row.get("user") or "Use the tool."},
            {"role": "assistant", "content": row["assistant_prefix"]},
        ]
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
            loaded.model, layers, cast_dtype=torch.float32, store_cpu=True
        )
        with torch.inference_mode(), hooks.capture():
            _ = loaded.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        last = int(input_ids.shape[-1] - 1)
        return {
            layer: (
                hooks.activations[layer][0, last].float().cpu()
                if hooks.activations[layer].dim() == 3
                else hooks.activations[layer][last].float().cpu()
            )
            for layer in layers
        }

    tool_pos_acts = [tool_act(r) for r in tool_pos]
    tool_neg_acts = [tool_act(r) for r in tool_neg]

    def dirs_from(pos_list, neg_list, layer: int):
        return mean_difference_direction(
            [a[layer] for a in pos_list],
            [a[layer] for a in neg_list],
        )

    def split_half_cos(pos_list, neg_list, layer: int, rng: random.Random) -> float:
        n = len(pos_list)
        idx = list(range(n))
        rng.shuffle(idx)
        half = n // 2
        if half < 1 or n - half < 1:
            return float("nan")
        a, b = idx[:half], idx[half : 2 * half]
        da = dirs_from([pos_list[i] for i in a], [neg_list[i] for i in a], layer)
        db = dirs_from([pos_list[i] for i in b], [neg_list[i] for i in b], layer)
        return float(cosine(da, db))

    n_pp = len(prose_pos_acts)
    n_tp = len(tool_pos_acts)
    n_tn = len(tool_neg_acts)
    rng = random.Random(args.seed)

    # Point estimates
    point_pt = {}
    point_pp = {}
    point_gap = {}
    for layer in layers:
        point_pt[layer] = float(
            cosine(
                dirs_from(prose_pos_acts, prose_neg_acts, layer),
                dirs_from(tool_pos_acts, tool_neg_acts, layer),
            )
        )
        # Deterministic split for point pp (seeded)
        point_pp[layer] = split_half_cos(
            prose_pos_acts, prose_neg_acts, layer, random.Random(args.seed + layer)
        )
        point_gap[layer] = point_pp[layer] - point_pt[layer]

    boot_pt: dict[int, list[float]] = {l: [] for l in layers}
    boot_pp: dict[int, list[float]] = {l: [] for l in layers}
    boot_gap: dict[int, list[float]] = {l: [] for l in layers}

    for b in range(args.n_boot):
        brng = random.Random(args.seed + 10_000 + b)
        pi = [brng.randrange(n_pp) for _ in range(n_pp)]
        tip = [brng.randrange(n_tp) for _ in range(n_tp)]
        tin = [brng.randrange(n_tn) for _ in range(n_tn)]
        ppos = [prose_pos_acts[i] for i in pi]
        pneg = [prose_neg_acts[i] for i in pi]
        tpos = [tool_pos_acts[i] for i in tip]
        tneg = [tool_neg_acts[i] for i in tin]
        for layer in layers:
            c_pt = float(
                cosine(dirs_from(ppos, pneg, layer), dirs_from(tpos, tneg, layer))
            )
            c_pp = split_half_cos(ppos, pneg, layer, brng)
            boot_pt[layer].append(c_pt)
            boot_pp[layer].append(c_pp)
            boot_gap[layer].append(c_pp - c_pt)

    # Jackknife for BCa acceleration: leave-one prose pair + leave-one tool (approx)
    def jackknife(stat_fn) -> list[float]:
        vals = []
        for i in range(n_pp):
            ppos = [prose_pos_acts[j] for j in range(n_pp) if j != i]
            pneg = [prose_neg_acts[j] for j in range(n_pp) if j != i]
            vals.append(stat_fn(ppos, pneg, tool_pos_acts, tool_neg_acts))
        for i in range(n_tp):
            tpos = [tool_pos_acts[j] for j in range(n_tp) if j != i]
            vals.append(stat_fn(prose_pos_acts, prose_neg_acts, tpos, tool_neg_acts))
        for i in range(n_tn):
            tneg = [tool_neg_acts[j] for j in range(n_tn) if j != i]
            vals.append(stat_fn(prose_pos_acts, prose_neg_acts, tool_pos_acts, tneg))
        return vals

    layers_out = []
    for layer in layers:
        def pt_stat(ppos, pneg, tpos, tneg, _l=layer):
            return float(cosine(dirs_from(ppos, pneg, _l), dirs_from(tpos, tneg, _l)))

        def gap_stat(ppos, pneg, tpos, tneg, _l=layer):
            c_pt = pt_stat(ppos, pneg, tpos, tneg)
            c_pp = split_half_cos(ppos, pneg, _l, random.Random(args.seed + _l + 999))
            return c_pp - c_pt

        jack_pt = jackknife(pt_stat)
        jack_gap = jackknife(gap_stat)
        pt_lo, pt_hi, pt_diag = bca_ci(point_pt[layer], boot_pt[layer], jack_pt)
        gap_lo, gap_hi, gap_diag = bca_ci(point_gap[layer], boot_gap[layer], jack_gap)
        pct_pt = percentile_ci(boot_pt[layer])
        pct_gap = percentile_ci(boot_gap[layer])

        row = {
            "layer": layer,
            "prose_prose_split_half_point": point_pp[layer],
            "prose_tool_point": point_pt[layer],
            "gap_point": point_gap[layer],
            "prose_tool_bca_ci95": [pt_lo, pt_hi],
            "prose_tool_percentile_ci95": list(pct_pt),
            "prose_tool_bca_diag": pt_diag,
            "gap_bca_ci95": [gap_lo, gap_hi],
            "gap_percentile_ci95": list(pct_gap),
            "gap_bca_diag": gap_diag,
            "gap_bca_excludes_zero": bool(gap_lo > 0),
            "ci_method": "bca_joint",
            "n_prose_pairs": n_pp,
            "n_tool_pos": n_tp,
            "n_tool_neg": n_tn,
            "n_boot": args.n_boot,
            "note": (
                "Gap is joint: each bootstrap replicate resamples prose+tool and "
                "recomputes both split-half pp and pt. Percentile shown for pathology check."
            ),
        }
        layers_out.append(row)
        print(
            f"L{layer}: pt={point_pt[layer]:.3f} BCa[{pt_lo:.3f},{pt_hi:.3f}] "
            f"(pct[{pct_pt[0]:.3f},{pct_pt[1]:.3f}]) "
            f"gap={point_gap[layer]:.3f} BCa[{gap_lo:.3f},{gap_hi:.3f}] "
            f"excl0={row['gap_bca_excludes_zero']}"
        )

    reading = (
        "Joint BCa CIs (prereg §5.2). Percentile at small n can look skewed "
        "(e.g. L4 pt CI straddling the point); prefer BCa for 32B writeups. "
        "Mid/late gaps remain suggestive; scale-confounded at 0.6B."
    )
    payload = {
        "model": model_key,
        "ci_method": "bca_joint",
        "comparison": {
            "layers": layers_out,
            "mid_late_layers": [14, 22],
            "reading": reading,
        },
        "status": "suggestive_midlate_gap_with_joint_bca_ci_scale_confounded",
        "prereg_note": (
            "32B: BCa + joint gap required; depth-grid correction family is §3-own "
            "(not folded into §1 GAP⊥τ); stability gate = majority of fixed F_mid."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    pilot = ROOT / "data" / "results" / "direction_stability_pilot.json"
    if pilot.exists():
        d = json.loads(pilot.read_text())
        d["layer_matched_comparison"] = payload["comparison"]
        d["status"] = payload["status"]
        d["interpretation"] = reading
        d["ci_method"] = "bca_joint"
        pilot.write_text(json.dumps(d, indent=2))
        print(f"annotated {pilot}")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
