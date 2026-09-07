#!/usr/bin/env python3
"""Classic Heimersheim & Mendel (2024) activation graphs on agentic windows.

Reproduces the two core contrasts from the paper (not prose-vs-tool on a fixed
safety direction):

1) Activation plateaus — random-direction push on a *real* base vs a
   *Gaussian random* base (mean/cov-matched to real activations at layer k).
2) Sensitive directions — from a *real* base, push toward another *real*
   activation (“real-other”) vs a covariance-adjusted *random* direction.

Rows: prose windows / tool_call windows (agentic extension of the setting).
Metric: absolute residual L2 blowup at late layer L (Heim also used KL/L2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


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
    ap.add_argument("--max-transcripts", type=int, default=6)
    ap.add_argument("--max-windows-per-mode", type=int, default=5)
    ap.add_argument("--n-curves", type=int, default=4, help="Overlay curves per condition (Heim-style)")
    ap.add_argument("--n-eps", type=int, default=20)
    ap.add_argument("--eps-hi", type=float, default=50.0, help="Heim ~typical inter-activation distance scale")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "heim_classic_graphs.json",
    )
    ap.add_argument(
        "--fig",
        type=Path,
        default=ROOT / "docs" / "figures" / "step_5_heim_classic_graphs.png",
    )
    args = ap.parse_args()

    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from activation_pipeline.analysis.sensitivity import LayerPerturbHooks, residual_l2_blowup
    from activation_pipeline.caching import load_transcripts
    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer
    from activation_pipeline.windows import embed_windows_in_chat, tag_transcript_assistant_turns

    rng = np.random.default_rng(args.seed)
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
    hidden = loaded.spec.hidden_size
    # Linear ε grid like Heim step sweeps (include 0)
    eps_grid = [0.0] + list(np.linspace(args.eps_hi / args.n_eps, args.eps_hi, args.n_eps))
    print(f"heim-classic model={model_key} k={layer_k} L={layer_L} n_eps={len(eps_grid)}")

    transcripts = load_transcripts(args.transcripts)
    preferred = {"mind_the_gap_tool_replay", "mind_the_gap_scenarios"}
    transcripts = [t for t in transcripts if t.get("source") in preferred] or transcripts
    transcripts = transcripts[: args.max_transcripts]

    # ---- Collect real windows + layer-k pooled activations ----
    jobs: dict[str, list[dict[str, Any]]] = {"prose": [], "tool_call": []}
    act_bank: dict[str, list[torch.Tensor]] = {"prose": [], "tool_call": []}

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
                kind = window.kind
                if kind not in jobs or len(jobs[kind]) >= args.max_windows_per_mode:
                    continue
                idxs = window.token_indices()
                if not idxs:
                    continue
                hooks = LayerPerturbHooks(
                    loaded.model,
                    layer_k=layer_k,
                    direction=torch.ones(hidden),
                    token_indices=idxs,
                    read_layers=[layer_k, layer_L],
                )
                acts = hooks.run(input_ids, attn, epsilon=0.0)
                h_k = acts[layer_k][0, idxs, :].float().mean(dim=0).cpu()
                h_L = acts[layer_L][0, idxs, :].float().mean(dim=0).cpu()
                jobs[kind].append(
                    {
                        "transcript_id": tr["transcript_id"],
                        "window_kind": kind,
                        "input_ids": input_ids,
                        "attn": attn,
                        "token_indices": idxs,
                        "h_k": h_k,
                        "h_L": h_L,
                    }
                )
                act_bank[kind].append(h_k)
        if all(len(jobs[m]) >= args.max_windows_per_mode for m in jobs):
            break

    print({m: len(jobs[m]) for m in jobs})

    def fit_gaussian(acts: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        X = torch.stack(acts, dim=0).numpy()
        mu = X.mean(axis=0)
        # ridge covariance; fall back to diagonal if too few samples
        if X.shape[0] >= 3:
            C = np.cov(X, rowvar=False)
        else:
            C = np.diag(X.var(axis=0) + 1e-4)
        C = C + 1e-3 * np.eye(C.shape[0])
        # diagonalize for stable sampling (full Cholesky can be flaky in high-d)
        # use diagonal cov for sampling stability at d=1024 with n≈5
        var = np.maximum(np.diag(C), 1e-4)
        return mu, var

    def sample_random_base(mu: np.ndarray, var: np.ndarray) -> torch.Tensor:
        z = mu + rng.normal(0.0, 1.0, size=mu.shape) * np.sqrt(var)
        return torch.tensor(z, dtype=torch.float32)

    def sample_random_dir(var: np.ndarray) -> torch.Tensor:
        # covariance-adjusted random direction (Heim preference vs isotropic)
        g = rng.normal(0.0, 1.0, size=var.shape) * np.sqrt(var)
        v = torch.tensor(g, dtype=torch.float32)
        return v / (torch.linalg.norm(v) + 1e-8)

    def curve_blowup_abs(
        *,
        job: dict[str, Any],
        direction: torch.Tensor,
        base_override: torch.Tensor | None,
        ref_L: torch.Tensor,
    ) -> list[float]:
        hooks = LayerPerturbHooks(
            loaded.model,
            layer_k=layer_k,
            direction=direction,
            token_indices=job["token_indices"],
            read_layers=[layer_L],
            base_override=base_override,
        )
        out = []
        for eps in eps_grid:
            if eps <= 0.0 and base_override is None:
                out.append(0.0)
                continue
            if eps <= 0.0 and base_override is not None:
                # unperturbed random base: still need a forward for ref if ref not set
                acts = hooks.run(job["input_ids"], job["attn"], epsilon=0.0)
                # blowup vs natural clean is not meaningful for plateau fig;
                # Heim measures change from the unperturbed base of that condition.
                # For ε=0 relative to itself → 0
                _ = acts
                out.append(0.0)
                continue
            acts = hooks.run(job["input_ids"], job["attn"], epsilon=float(eps))
            pert_L = acts[layer_L][0, job["token_indices"], :].float().mean(dim=0)
            # For replace mode, ref_L should be the ε=0 replace forward
            out.append(residual_l2_blowup(ref_L, pert_L.cpu(), relative=False))
        return out

    def get_replace_ref_L(job: dict[str, Any], base: torch.Tensor) -> torch.Tensor:
        hooks = LayerPerturbHooks(
            loaded.model,
            layer_k=layer_k,
            direction=torch.ones(hidden),
            token_indices=job["token_indices"],
            read_layers=[layer_L],
            base_override=base,
        )
        acts = hooks.run(job["input_ids"], job["attn"], epsilon=0.0)
        return acts[layer_L][0, job["token_indices"], :].float().mean(dim=0).cpu()

    results: dict[str, Any] = {
        "model": model_key,
        "layer_k": layer_k,
        "layer_L": layer_L,
        "epsilon": eps_grid,
        "n_curves": args.n_curves,
        "modes": {},
        "notes": [
            "Classic Heim contrasts: (1) real vs random base plateaus; (2) real-other vs random dir.",
            "Colors follow Heim: orange=real/real-other, blue=random base/random dir.",
            "Absolute L2 blowup at L; agentic rows = prose / tool_call windows.",
        ],
    }

    # Precompute Gaussian fits per mode (and pooled)
    gauss = {}
    for mode in ("prose", "tool_call"):
        if act_bank[mode]:
            gauss[mode] = fit_gaussian(act_bank[mode])
    all_acts = act_bank["prose"] + act_bank["tool_call"]
    gauss["pooled"] = fit_gaussian(all_acts)

    for mode in ("prose", "tool_call"):
        mode_jobs = jobs[mode]
        if len(mode_jobs) < 2:
            print(f"skip {mode}: need >=2 windows for real-other")
            continue
        mu, var = gauss[mode]
        plateau_real: list[list[float]] = []
        plateau_randbase: list[list[float]] = []
        sens_realother: list[list[float]] = []
        sens_randomdir: list[list[float]] = []

        # pick curve bases
        idxs = list(range(len(mode_jobs)))
        rng.shuffle(idxs)
        chosen = idxs[: args.n_curves]

        for ci, ji in enumerate(chosen):
            job = mode_jobs[ji]
            # other real activation (different window)
            other = mode_jobs[(ji + 1) % len(mode_jobs)]
            d_realother = other["h_k"] - job["h_k"]
            if float(torch.linalg.norm(d_realother)) < 1e-6:
                other = mode_jobs[(ji + 2) % len(mode_jobs)]
                d_realother = other["h_k"] - job["h_k"]
            d_realother = d_realother / (torch.linalg.norm(d_realother) + 1e-8)
            d_rand = sample_random_dir(var)
            rand_base = sample_random_base(mu, var)

            # 1a real base + random dir
            plateau_real.append(
                curve_blowup_abs(
                    job=job,
                    direction=d_rand,
                    base_override=None,
                    ref_L=job["h_L"],
                )
            )
            # 1b random base + same random dir family (new dir for independence)
            d_rand2 = sample_random_dir(var)
            ref_rand = get_replace_ref_L(job, rand_base)
            plateau_randbase.append(
                curve_blowup_abs(
                    job=job,
                    direction=d_rand2,
                    base_override=rand_base,
                    ref_L=ref_rand,
                )
            )
            # 2a real-other
            sens_realother.append(
                curve_blowup_abs(
                    job=job,
                    direction=d_realother,
                    base_override=None,
                    ref_L=job["h_L"],
                )
            )
            # 2b random dir
            sens_randomdir.append(
                curve_blowup_abs(
                    job=job,
                    direction=sample_random_dir(var),
                    base_override=None,
                    ref_L=job["h_L"],
                )
            )
            print(f"{mode} curve {ci+1}/{len(chosen)} {job['transcript_id']}")

        results["modes"][mode] = {
            "n_windows": len(mode_jobs),
            "plateau_real_base_random_dir": plateau_real,
            "plateau_random_base_random_dir": plateau_randbase,
            "sensitive_real_other_dir": sens_realother,
            "sensitive_random_dir": sens_randomdir,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # store without tensors
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")

    # ---- Figure: 2x2 like Heim's two experiments × agentic modes ----
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True)
    # Heim colors
    c_real = "#e67e22"  # orange
    c_rand = "#2980b9"  # blue
    mode_rows = [("prose", "Prose windows"), ("tool_call", "Tool-call windows")]

    for row, (mode, mode_title) in enumerate(mode_rows):
        ax_p = axes[row, 0]
        ax_s = axes[row, 1]
        m = results["modes"].get(mode)
        x = np.array(eps_grid, dtype=float)
        if not m:
            for ax in (ax_p, ax_s):
                ax.text(0.5, 0.5, f"insufficient {mode} windows", transform=ax.transAxes, ha="center")
            continue

        # Panel A: plateaus
        ax_p.set_title(f"{mode_title}: activation plateaus", fontsize=11)
        for series in m["plateau_real_base_random_dir"]:
            ax_p.plot(x, series, color=c_real, alpha=0.55, lw=1.6)
        for series in m["plateau_random_base_random_dir"]:
            ax_p.plot(x, series, color=c_rand, alpha=0.55, lw=1.6)
        # mean overlays
        ax_p.plot(x, np.mean(m["plateau_real_base_random_dir"], axis=0), color=c_real, lw=2.4, label="real base")
        ax_p.plot(x, np.mean(m["plateau_random_base_random_dir"], axis=0), color=c_rand, lw=2.4, label="random base")
        ax_p.set_ylabel(r"$\|\Delta h_L\|_2$ (absolute)")
        ax_p.legend(fontsize=8, loc="upper left")
        ax_p.grid(True, alpha=0.25)
        ax_p.text(
            0.98,
            0.05,
            "same random-dir push;\nbase differs",
            transform=ax_p.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#444444",
        )

        # Panel B: sensitive directions
        ax_s.set_title(f"{mode_title}: sensitive directions", fontsize=11)
        for series in m["sensitive_real_other_dir"]:
            ax_s.plot(x, series, color=c_real, alpha=0.55, lw=1.6)
        for series in m["sensitive_random_dir"]:
            ax_s.plot(x, series, color=c_rand, alpha=0.55, lw=1.6)
        ax_s.plot(x, np.mean(m["sensitive_real_other_dir"], axis=0), color=c_real, lw=2.4, label="real-other dir")
        ax_s.plot(x, np.mean(m["sensitive_random_dir"], axis=0), color=c_rand, lw=2.4, label="random dir")
        ax_s.legend(fontsize=8, loc="upper left")
        ax_s.grid(True, alpha=0.25)
        ax_s.text(
            0.98,
            0.05,
            "same real base;\ndirection differs",
            transform=ax_s.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#444444",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel(r"perturbation magnitude $\varepsilon$")

    fig.suptitle(
        "Heimersheim & Mendel–style graphs on agentic windows (Qwen3-0.6B)\n"
        f"Left: real vs random base (plateaus). Right: real-other vs random dir.  k={layer_k} → L={layer_L}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Orange = real / real-other (Heim). Blue = random base / random dir. "
        "These are the paper’s contrasts — not prose-vs-tool on a fixed safety direction. "
        "Rows extend the setting to GAP prose and tool-call windows.",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(args.fig, dpi=160)
    fig.savefig(args.fig.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
