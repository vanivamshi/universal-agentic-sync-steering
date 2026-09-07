#!/usr/bin/env python3
"""Heimersheim-style activation graphs for agentic prose↔tool windows.

Extends the plateau / directional-sensitivity method to GAP tool-call contexts:
sweep ε along a unit direction at layer k, plot late-layer L2 blowup vs ε.

Primary display: relative blowup (matches RQ1 ε* at τ). Absolute blowup also
saved. Curves are mean ± SEM over sampled windows, prose vs tool_call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Writable matplotlib cache in sandbox / CI
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_direction_at_layer(path: Path, *, layer: int) -> dict[str, Any] | None:
    for row in load_jsonl(path):
        if int(row["layer"]) == layer:
            return row
    return None


def mean_sem(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, (var**0.5) / (len(xs) ** 0.5)


def interpolate_eps_at_threshold(eps: list[float], blow: list[float], thr: float) -> float | None:
    """Linear interpolate first crossing of thr (Heim ε* approx from curve)."""
    for i in range(1, len(eps)):
        if blow[i - 1] < thr <= blow[i]:
            a, b = blow[i - 1], blow[i]
            if b == a:
                return eps[i]
            t = (thr - a) / (b - a)
            return eps[i - 1] + t * (eps[i] - eps[i - 1])
        if blow[i] >= thr and blow[i - 1] >= thr:
            return eps[i - 1] if i == 1 else None
    return None


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
    ap.add_argument("--layer-k", type=int, default=None)
    ap.add_argument("--layer-L", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5, help="Relative τ for annotation")
    ap.add_argument("--n-eps", type=int, default=18)
    ap.add_argument("--eps-hi", type=float, default=40.0)
    ap.add_argument("--max-transcripts", type=int, default=4)
    ap.add_argument("--max-windows-per-mode", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "heim_style_plateau_curves.json",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=ROOT / "docs" / "figures" / "step_5_heim_style_plateau_curves.png",
    )
    args = ap.parse_args()

    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from activation_pipeline.analysis.sensitivity import (
        LayerPerturbHooks,
        blowup_vs_epsilon_curve,
        default_epsilon_grid,
    )
    from activation_pipeline.caching import load_transcripts
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
    eps_grid = default_epsilon_grid(eps_hi=args.eps_hi, n=args.n_eps)
    print(
        f"heim-curves model={model_key} device={device_map} k={layer_k} L={layer_L} "
        f"n_eps={len(eps_grid)} τ={args.threshold}"
    )

    dirs_dir = ROOT / "data" / "directions"
    # Representative set: named central-prediction dir, harmlessness, learned, random
    specs = [
        ("assistant_axis", "safety", dirs_dir / "assistant_axis_prose.jsonl", None),
        ("harmlessness", "safety", dirs_dir / "harmlessness_prose.jsonl", None),
        ("syntax", "learned_control", dirs_dir / "controls_l4.jsonl", "syntax_L4"),
        ("rand_00", "random_control", dirs_dir / "random_controls_l4.jsonl", "rand_00_L4"),
    ]
    direction_bank: list[dict[str, Any]] = []
    for kind, dtype, path, want_id in specs:
        if want_id is None:
            row = load_direction_at_layer(path, layer=layer_k)
        else:
            row = next(
                (
                    r
                    for r in load_jsonl(path)
                    if int(r["layer"]) == layer_k and r["direction_id"] == want_id
                ),
                None,
            )
        if row is None:
            print(f"WARN missing {kind} at L{layer_k} from {path}")
            continue
        direction_bank.append(
            {
                "direction_id": row["direction_id"],
                "kind": kind,
                "direction_type": dtype,
                "vector": torch.tensor(row["vector"], dtype=torch.float32),
            }
        )
    print("directions:", [d["direction_id"] for d in direction_bank])

    transcripts = load_transcripts(args.transcripts)
    preferred = {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}
    transcripts = [t for t in transcripts if t.get("source") in preferred] or transcripts
    transcripts = transcripts[: args.max_transcripts]

    # Collect windows: up to max_windows_per_mode of each kind
    window_jobs: list[dict[str, Any]] = []
    counts = {"prose": 0, "tool_call": 0}
    for tr in transcripts:
        if counts["prose"] >= args.max_windows_per_mode and counts["tool_call"] >= args.max_windows_per_mode:
            break
        for tagged in tag_transcript_assistant_turns(tr, loaded.tokenizer):
            embedded = embed_windows_in_chat(
                loaded.tokenizer, tr["messages"], tagged.message_index, tagged
            )
            if not embedded.windows:
                continue
            input_ids = torch.tensor([embedded.input_ids], device=device)
            attn = torch.ones_like(input_ids)
            for window in embedded.windows:
                kind = window.kind
                if kind not in counts:
                    continue
                if counts[kind] >= args.max_windows_per_mode:
                    continue
                idxs = window.token_indices()
                if not idxs:
                    continue
                window_jobs.append(
                    {
                        "transcript_id": tr["transcript_id"],
                        "domain": tr["domain"],
                        "message_index": tagged.message_index,
                        "window_kind": kind,
                        "window_name": window.name or "",
                        "input_ids": input_ids,
                        "attn": attn,
                        "token_indices": idxs,
                    }
                )
                counts[kind] += 1
            if counts["prose"] >= args.max_windows_per_mode and counts["tool_call"] >= args.max_windows_per_mode:
                break

    print(f"windows: prose={counts['prose']} tool_call={counts['tool_call']} total={len(window_jobs)}")

    # curves[direction_id][window_kind] -> list of relative/absolute series
    rel_store: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    abs_store: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    raw_rows: list[dict[str, Any]] = []

    n_fwd = 0
    for wi, job in enumerate(window_jobs):
        idxs = job["token_indices"]
        clean_hooks = LayerPerturbHooks(
            loaded.model,
            layer_k=layer_k,
            direction=torch.ones(loaded.spec.hidden_size),
            token_indices=idxs,
            read_layers=[layer_L],
        )
        clean_acts = clean_hooks.run(job["input_ids"], job["attn"], epsilon=0.0)
        n_fwd += 1

        def pool(t: torch.Tensor, token_idxs=idxs) -> torch.Tensor:
            return t[0, token_idxs, :].float().mean(dim=0)

        clean_L = pool(clean_acts[layer_L])
        clean_norm = float(torch.linalg.norm(clean_L))

        for dinfo in direction_bank:
            hooks = LayerPerturbHooks(
                loaded.model,
                layer_k=layer_k,
                direction=dinfo["vector"],
                token_indices=idxs,
                read_layers=[layer_L],
            )

            def forward_pert(eps: float, h=hooks, ids=job["input_ids"], a=job["attn"]):
                return h.run(ids, a, epsilon=eps)

            curve = blowup_vs_epsilon_curve(
                clean_L=clean_L,
                forward_pert=forward_pert,
                layer_L=layer_L,
                pool=pool,
                eps_grid=eps_grid,
            )
            n_fwd += max(0, len(eps_grid) - 1)  # ε=0 skipped inside
            rel_store[dinfo["direction_id"]][job["window_kind"]].append(curve["blowup_relative"])
            abs_store[dinfo["direction_id"]][job["window_kind"]].append(curve["blowup_absolute"])
            raw_rows.append(
                {
                    "transcript_id": job["transcript_id"],
                    "domain": job["domain"],
                    "message_index": job["message_index"],
                    "window_kind": job["window_kind"],
                    "window_name": job["window_name"],
                    "direction_id": dinfo["direction_id"],
                    "direction_type": dinfo["direction_type"],
                    "kind": dinfo["kind"],
                    "clean_norm_L": clean_norm,
                    "epsilon": curve["epsilon"],
                    "blowup_relative": curve["blowup_relative"],
                    "blowup_absolute": curve["blowup_absolute"],
                    "eps_star_rel_approx": interpolate_eps_at_threshold(
                        curve["epsilon"], curve["blowup_relative"], args.threshold
                    ),
                }
            )
        print(
            f"[{wi+1}/{len(window_jobs)}] {job['transcript_id']} "
            f"{job['window_kind']} ntok={len(idxs)}"
        )

    # Aggregate mean±sem curves
    aggregates = []
    for dinfo in direction_bank:
        did = dinfo["direction_id"]
        entry: dict[str, Any] = {
            "direction_id": did,
            "kind": dinfo["kind"],
            "direction_type": dinfo["direction_type"],
            "epsilon": eps_grid,
            "modes": {},
        }
        for mode in ("prose", "tool_call"):
            rel_series = rel_store[did].get(mode, [])
            abs_series = abs_store[did].get(mode, [])
            if not rel_series:
                continue
            rel_arr = np.array(rel_series, dtype=float)
            abs_arr = np.array(abs_series, dtype=float)
            rel_mean = rel_arr.mean(axis=0).tolist()
            rel_sem = (rel_arr.std(axis=0, ddof=1) / (len(rel_series) ** 0.5)).tolist() if len(rel_series) > 1 else [0.0] * len(eps_grid)
            abs_mean = abs_arr.mean(axis=0).tolist()
            abs_sem = (abs_arr.std(axis=0, ddof=1) / (len(abs_series) ** 0.5)).tolist() if len(abs_series) > 1 else [0.0] * len(eps_grid)
            entry["modes"][mode] = {
                "n_windows": len(rel_series),
                "blowup_relative_mean": rel_mean,
                "blowup_relative_sem": rel_sem,
                "blowup_absolute_mean": abs_mean,
                "blowup_absolute_sem": abs_sem,
                "eps_star_rel_approx": interpolate_eps_at_threshold(
                    eps_grid, rel_mean, args.threshold
                ),
            }
        aggregates.append(entry)

    payload = {
        "model": model_key,
        "method": "heimersheim_style_blowup_vs_epsilon",
        "extension": "agentic_prose_vs_tool_call_windows",
        "layer_k": layer_k,
        "layer_L": layer_L,
        "threshold_relative": args.threshold,
        "epsilon_grid": eps_grid,
        "n_forwards_approx": n_fwd,
        "n_windows": counts,
        "aggregates": aggregates,
        "window_curves": raw_rows,
        "notes": [
            "Heimersheim/Mendel-style activation graph: late-layer L2 blowup vs ε.",
            "Extended here to paired agentic prose vs tool_call residual windows.",
            "Relative blowup matches RQ1 primary τ; absolute also recorded.",
            "ε*_approx from curve interpolation is diagnostic; RQ1 binary-search is canonical.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")

    # ---- Figure ----
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
    panel_order = [
        ("assistant_axis", "Assistant Axis (safety)"),
        ("harmlessness", "Harmlessness (safety)"),
        ("syntax", "Syntax (learned control)"),
        ("rand_00", "Random (control)"),
    ]
    by_kind = {a["kind"]: a for a in aggregates}
    colors = {"prose": "#2c7fb8", "tool_call": "#d95f0e"}

    for ax, (kind, title) in zip(axes.flat, panel_order):
        agg = by_kind.get(kind)
        ax.set_title(title, fontsize=11)
        ax.axhline(args.threshold, color="#666666", ls=":", lw=1.0, label=f"τ={args.threshold}")
        if agg is None:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            continue
        for mode, label in (("prose", "prose"), ("tool_call", "tool")):
            m = agg["modes"].get(mode)
            if not m:
                continue
            x = np.array(agg["epsilon"], dtype=float)
            y = np.array(m["blowup_relative_mean"], dtype=float)
            sem = np.array(m["blowup_relative_sem"], dtype=float)
            ax.plot(x, y, color=colors[mode], lw=2.0, label=f"{label} (n={m['n_windows']})")
            ax.fill_between(x, y - sem, y + sem, color=colors[mode], alpha=0.18, linewidth=0)
            est = m.get("eps_star_rel_approx")
            if est is not None:
                ax.axvline(est, color=colors[mode], ls="--", lw=0.9, alpha=0.7)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Heimersheim-style activation graphs — agentic prose vs tool (Qwen3-0.6B)\n"
        f"relative L2 blowup at L={layer_L} after perturbing along d at k={layer_k}",
        fontsize=12,
    )
    fig.supxlabel("perturbation magnitude ε", fontsize=11)
    fig.supylabel("relative residual L2 blowup  ‖Δh_L‖ / ‖h_L‖", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Solid: mean over windows; band: SEM. Dashed vertical: ε*≈τ crossing on the mean curve. "
        "Method extension of Heimersheim & Mendel (2024) to agentic tool-call windows.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(args.fig, dpi=160)
    fig.savefig(args.fig.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {args.fig}")
    print(f"wrote {args.fig.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
