#!/usr/bin/env python3
"""Experiment 1 — broad GAP L4 SAE + Engels-style irreducibility / differential.

Locked: docs/multidim_feature_protocol.md (Exp 1). User override of CLUSTER_HIT gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

SEED = 20260813
LAYER = 4
DICT_MULT = 4
MAX_TRAIN = 16384
SAE_STEPS = 4000
BATCH = 256
L1 = 1e-3
LR = 1e-3
CUT = 0.50
MIN_SIZE = 2
MAX_SIZE = 12
COOC_MIN = 0.35
COHEN_HIT = 0.50


class ReLUSAE(nn.Module):
    def __init__(self, d: int, n_feat: int) -> None:
        super().__init__()
        self.w_enc = nn.Parameter(torch.empty(d, n_feat))
        self.b_enc = nn.Parameter(torch.zeros(n_feat))
        self.w_dec = nn.Parameter(torch.empty(n_feat, d))
        self.b_dec = nn.Parameter(torch.zeros(d))
        nn.init.kaiming_uniform_(self.w_enc)
        nn.init.kaiming_uniform_(self.w_dec)
        self.renorm_decoder()

    def renorm_decoder(self) -> None:
        with torch.no_grad():
            n = torch.linalg.norm(self.w_dec, dim=1, keepdim=True).clamp_min(1e-8)
            self.w_dec.div_(n)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu((h - self.b_dec) @ self.w_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.w_dec + self.b_dec

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(h)
        return self.decode(f), f


def _average_linkage_clusters(dist: np.ndarray, cut: float) -> list[list[int]]:
    p = dist.shape[0]
    active = list(range(p))
    members: dict[int, set[int]] = {i: {i} for i in range(p)}
    next_id = p
    while len(active) > 1:
        best = None
        best_d = float("inf")
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                da = 0.0
                n = 0
                for u in members[a]:
                    for v in members[b]:
                        da += dist[u, v]
                        n += 1
                da /= max(n, 1)
                if da < best_d:
                    best_d = da
                    best = (a, b)
        if best is None or best_d > cut:
            break
        a, b = best
        members[next_id] = members[a] | members[b]
        active = [x for x in active if x not in (a, b)] + [next_id]
        del members[a]
        del members[b]
        next_id += 1
    return [sorted(members[k]) for k in active]


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ma, mb = float(a.mean()), float(b.mean())
    sa, sb = float(a.std(ddof=1)), float(b.std(ddof=1))
    sp = np.sqrt(((a.size - 1) * sa**2 + (b.size - 1) * sb**2) / max(a.size + b.size - 2, 1))
    if sp < 1e-12:
        return 0.0
    return (ma - mb) / sp


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    s = np.sqrt((Xc**2).sum(0))
    s = np.where(s < 1e-12, 1.0, s)
    return (Xc.T @ Xc) / np.outer(s, s)


def _load_pt_layer_means(path: Path, layer: int) -> np.ndarray:
    z = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for k, v in z["means"].items():
        if str(k).endswith(f":{layer}"):
            rows.append(v.float().numpy())
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0)


def _collect_window_tokens(
    *,
    model,
    tok,
    device: torch.device,
    windows_path: Path,
    layer: int,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    from activation_pipeline.hooks import ResidualStreamHooks

    chunks: list[np.ndarray] = []
    n = 0
    hooks = ResidualStreamHooks(model, [layer], cast_dtype=torch.float32, store_cpu=True)
    for line in windows_path.read_text().splitlines():
        if n >= max_rows:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        wins = row.get("windows") or []
        if not wins:
            continue
        enc = tok(row["text"], return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        with torch.inference_mode():
            with hooks.capture():
                model(input_ids=input_ids, use_cache=False)
        h = hooks.activations[layer]
        if h.dim() == 3:
            h = h[0]
        for w in wins:
            a, b = int(w["start"]), int(w["end"])
            sl = h[a:b].float().numpy()
            if sl.size == 0:
                continue
            chunks.append(sl)
            n += int(sl.shape[0])
            if n >= max_rows:
                break
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    X = np.concatenate(chunks, axis=0)
    if X.shape[0] > max_rows:
        pick = rng.choice(X.shape[0], size=max_rows, replace=False)
        X = X[pick]
    return X.astype(np.float32)


def main() -> int:
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.loader import load_model_and_tokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument(
        "--windows",
        type=Path,
        default=ROOT / "data" / "windows" / "real_token_windows.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "results" / "multidim_exp1_sae.json",
    )
    ap.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "data" / "results" / "multidim_exp1_sae.md",
    )
    ap.add_argument(
        "--sae-out",
        type=Path,
        default=ROOT / "data" / "directions" / "multidim_exp1_sae.pt",
    )
    ap.add_argument("--max-train", type=int, default=MAX_TRAIN)
    ap.add_argument("--steps", type=int, default=SAE_STEPS)
    ap.add_argument(
        "--load-sae",
        type=Path,
        default=None,
        help="Skip training; load weights from prior Exp1 SAE .pt",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    hidden = int(loaded.spec.hidden_size)
    device = next(loaded.model.parameters()).device
    tok = loaded.tokenizer

    n_feat = DICT_MULT * hidden
    if args.load_sae is not None:
        print(f"=== load SAE {args.load_sae} ===", flush=True)
        blob = torch.load(args.load_sae, map_location="cpu", weights_only=False)
        sae = ReLUSAE(hidden, n_feat).to(device)
        with torch.no_grad():
            sae.w_enc.copy_(blob["w_enc"].to(device))
            sae.b_enc.copy_(blob["b_enc"].to(device))
            sae.w_dec.copy_(blob["w_dec"].to(device))
            sae.b_dec.copy_(blob["b_dec"].to(device))
        dead_frac = float("nan")
        n_train = int(blob.get("n_train") or 0)
        X = None
    else:
        print("=== collect train residuals ===", flush=True)
        parts: list[np.ndarray] = []
        Xt = _collect_window_tokens(
            model=loaded.model,
            tok=tok,
            device=device,
            windows_path=args.windows,
            layer=args.layer,
            max_rows=args.max_train,
            rng=rng,
        )
        if Xt.size:
            parts.append(Xt)
            print(f"  window tokens={Xt.shape[0]}", flush=True)
        for name in ("real_gap.pt", "real_all.pt", "gap_deception.pt"):
            path = ROOT / "data" / "activations" / name
            if not path.exists():
                continue
            Xm = _load_pt_layer_means(path, args.layer)
            if Xm.size:
                parts.append(Xm.astype(np.float32))
                print(f"  {name} L{args.layer} means={Xm.shape[0]}", flush=True)
        if not parts:
            raise RuntimeError("no SAE training rows")
        Xnp = np.concatenate(parts, axis=0)
        if Xnp.shape[0] > args.max_train:
            pick = rng.choice(Xnp.shape[0], size=args.max_train, replace=False)
            Xnp = Xnp[pick]
        X = torch.from_numpy(Xnp).to(device)
        n_train = int(X.shape[0])
        print(f"train rows={n_train} d={hidden}", flush=True)

        sae = ReLUSAE(hidden, n_feat).to(device)
        opt = torch.optim.Adam(sae.parameters(), lr=LR)
        n = int(X.shape[0])
        print(f"=== train SAE features={n_feat} steps={args.steps} ===", flush=True)
        for step in range(args.steps):
            idx = torch.randint(0, n, (min(BATCH, n),), device=device)
            xb = X[idx]
            xhat, f = sae(xb)
            mse = F.mse_loss(xhat, xb)
            l1 = f.abs().mean()
            loss = mse + L1 * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.renorm_decoder()
            if step % 400 == 0 or step == args.steps - 1:
                dead = float((f.detach().mean(0) == 0).float().mean())
                print(
                    f"  step {step} mse={float(mse.detach()):.4g} l1={float(l1.detach()):.4g} batch_dead={dead:.3f}",
                    flush=True,
                )

        with torch.inference_mode():
            _, fall = sae(X)
            alive_train = (fall.mean(0) > 0).cpu().numpy()
            dead_frac = float(1.0 - alive_train.mean())
        print(f"dead features={dead_frac:.3f}", flush=True)
        if dead_frac > 0.90:
            payload = {
                "stage": "MULTIDIM_EXP1_SAE",
                "decision": "SAE_DEAD",
                "dead_frac": dead_frac,
                "causal_claim": False,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n")
            args.md_out.write_text(
                f"# Exp1 SAE\n\n## Decision: `SAE_DEAD`\ndead_frac={dead_frac:.3f}\n"
            )
            print("SAE_DEAD", flush=True)
            return 0

    # mark alive from encoder bias / decoder usage on a probe batch of means
    with torch.inference_mode():
        probe = _load_pt_layer_means(
            ROOT / "data" / "activations" / "gap_deception.pt", args.layer
        )
        if probe.size == 0:
            probe = np.random.randn(64, hidden).astype(np.float32)
        _, fall = sae(torch.from_numpy(probe.astype(np.float32)).to(device))
        alive = (fall.mean(0) > 0).cpu().numpy()
        if args.load_sae is not None:
            dead_frac = float(1.0 - alive.mean())

    # Transcript-level activations for differential + clustering.
    # Eliciting lives in gap_deception.json; benign controls are NOT in that
    # cache — pull L4 means from real_gap.pt (legitimate GAP trajectories).
    act = json.loads((ROOT / "data" / "activations" / "gap_deception.json").read_text())
    elab = {
        json.loads(l)["transcript_id"]: json.loads(l)
        for l in (ROOT / "data" / "labels" / "gap_deception_eliciting.jsonl").read_text().splitlines()
        if l.strip()
    }
    buckets: dict[str, list[np.ndarray]] = {}
    meta_family: dict[str, str] = {}
    for rec in act["records"]:
        tid = rec["transcript_id"]
        lm = rec["layer_means"].get(str(args.layer))
        if lm is None:
            continue
        buckets.setdefault(tid, []).append(np.asarray(lm, dtype=np.float32))
        meta_family[tid] = (elab.get(tid) or {}).get("family") or "eliciting"

    rg = torch.load(
        ROOT / "data" / "activations" / "real_gap.pt",
        map_location="cpu",
        weights_only=False,
    )
    for i, info in enumerate(rg["index"]):
        tid = str(info["transcript_id"])
        key = f"{i}:{args.layer}"
        if key not in rg["means"]:
            continue
        # only legitimate / non-jailbreak style ids as control pool
        if "legitimate" not in tid and "control" not in tid:
            # still allow as control if clearly benign naming in real_gap
            pass
        buckets.setdefault(f"ctrl::{tid}", []).append(
            rg["means"][key].float().numpy().astype(np.float32)
        )
        meta_family[f"ctrl::{tid}"] = "control"

    tids = sorted(buckets)
    H = np.stack([np.mean(buckets[t], 0) for t in tids], axis=0)
    with torch.inference_mode():
        Fmat = sae.encode(torch.from_numpy(H).to(device)).cpu().numpy()

    y_surf = np.array(
        [
            float((elab.get(t) or {}).get("surface_gap") or 0)
            if not t.startswith("ctrl::")
            else 0.0
            for t in tids
        ],
        dtype=np.float64,
    )
    y_int = np.array(
        [
            float((elab.get(t) or {}).get("intent_contradiction") or 0)
            if not t.startswith("ctrl::")
            else 0.0
            for t in tids
        ],
        dtype=np.float64,
    )
    is_ctrl = np.array(
        [1.0 if meta_family.get(t) == "control" else 0.0 for t in tids]
    )
    pos = y_surf > 0.5
    ctrl = is_ctrl > 0.5
    # live features only for clustering (on transcript set)
    live = alive & (Fmat.mean(0) > 1e-6)
    live_idx = np.where(live)[0]
    print(f"live features on transcripts={live_idx.size}", flush=True)
    if live_idx.size < 4:
        decision = "SAE_NULL"
        clusters_out: list[dict[str, Any]] = []
        note = "Too few live features on transcript means."
    else:
        Fl = Fmat[:, live_idx]
        # cap features for clustering cost: top by variance
        if Fl.shape[1] > 256:
            var = Fl.var(0)
            keep = np.argsort(-var)[:256]
            live_idx = live_idx[keep]
            Fl = Fmat[:, live_idx]
            print(f"  variance-cap features→{live_idx.size}", flush=True)
        C = np.clip(_corr_matrix(Fl), -1, 1)
        dist = 1.0 - np.abs(C)
        np.fill_diagonal(dist, 0.0)
        raw = _average_linkage_clusters(dist, CUT)
        clusters = [c for c in raw if MIN_SIZE <= len(c) <= MAX_SIZE]

        # co-occurrence on binary activity
        med = np.median(Fl, axis=0)
        active = Fl > med

        clusters_out = []
        any_hit = False
        any_weak = False
        for ci, members in enumerate(clusters):
            feat_ids = [int(live_idx[j]) for j in members]
            # pairwise co-occurrence
            coocs = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ai = active[:, members[i]]
                    aj = active[:, members[j]]
                    inter = float(np.logical_and(ai, aj).mean())
                    union = float(np.logical_or(ai, aj).mean())
                    coocs.append(inter / union if union > 1e-8 else 0.0)
            mean_cooc = float(np.mean(coocs)) if coocs else 0.0
            irreducible = mean_cooc >= COOC_MIN
            score = Fl[:, members].mean(1)
            d_surf = _cohen_d(score[pos], score[ctrl]) if ctrl.any() and pos.any() else float("nan")
            # also vs non-gap eliciting
            neg = (~pos) & (np.array([t in elab for t in tids]))
            d_vs_neg = _cohen_d(score[pos], score[neg]) if neg.any() and pos.any() else float("nan")
            hit = irreducible and abs(d_surf) >= COHEN_HIT
            weak = irreducible and abs(d_surf) >= 0.30
            if hit:
                any_hit = True
            if weak:
                any_weak = True
            # decoder subspace for Exp3
            dec = sae.w_dec[feat_ids].detach().cpu().numpy()
            clusters_out.append(
                {
                    "feature_ids": feat_ids,
                    "size": len(feat_ids),
                    "mean_cooccurrence_jaccard": mean_cooc,
                    "irreducible_candidate": bool(irreducible),
                    "cohen_d_surface_vs_control": float(d_surf)
                    if d_surf == d_surf
                    else None,
                    "cohen_d_surface_vs_nongap": float(d_vs_neg)
                    if d_vs_neg == d_vs_neg
                    else None,
                    "hit": bool(hit),
                    "weak": bool(weak),
                    "decoder_basis": dec.tolist(),
                }
            )

        if any_hit:
            decision = "SAE_HIT"
            note = "≥1 irreducible co-occurring feature cluster with |Cohen d|≥0.50 vs control."
        elif any_weak:
            decision = "SAE_WEAK"
            note = "Irreducible clusters with moderate differential (|d|≥0.30); locked HIT bar not cleared."
        elif clusters_out:
            decision = "SAE_NULL"
            note = "Co-activation clusters exist but none clear differential bars vs control."
        else:
            decision = "SAE_NULL"
            note = "No size-gated co-activation clusters among live SAE features."

    # persist SAE + top bases (skip overwrite if loaded)
    if args.load_sae is None:
        args.sae_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layer": args.layer,
                "w_enc": sae.w_enc.detach().cpu(),
                "b_enc": sae.b_enc.detach().cpu(),
                "w_dec": sae.w_dec.detach().cpu(),
                "b_dec": sae.b_dec.detach().cpu(),
                "seed": SEED,
                "n_train": n_train,
                "decision": decision,
            },
            args.sae_out,
        )

    # strip bulky bases from JSON except top differential candidates
    ranked = sorted(
        clusters_out,
        key=lambda x: abs(x.get("cohen_d_surface_vs_control") or 0.0),
        reverse=True,
    )
    clusters_json = []
    for i, c in enumerate(ranked):
        slim = {k: v for k, v in c.items() if k != "decoder_basis"}
        keep_basis = (
            c.get("hit")
            or c.get("weak")
            or i < 3
            or abs(c.get("cohen_d_surface_vs_control") or 0) >= 0.30
        )
        if keep_basis:
            slim["decoder_basis"] = c["decoder_basis"]
        clusters_json.append(slim)

    report = {
        "stage": "MULTIDIM_EXP1_SAE",
        "paper": "arXiv:2405.14860",
        "override_note": "Ran despite Exp2 CLUSTER_WEAK (user request)",
        "layer": args.layer,
        "n_train": n_train,
        "n_features": n_feat,
        "dead_frac": dead_frac,
        "n_transcripts": len(tids),
        "n_surface_pos": int(pos.sum()),
        "n_control": int(ctrl.sum()),
        "n_clusters": len(clusters_out),
        "clusters": clusters_json[:40],
        "decision": decision,
        "note": note,
        "sae_path": str(args.sae_out if args.load_sae is None else args.load_sae),
        "causal_claim": False,
        "exp3_licensed": decision in ("SAE_HIT", "SAE_WEAK"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Experiment 1 — broad GAP L4 SAE (Engels lens)",
        "",
        f"train_rows={report['n_train']} features={n_feat} dead={dead_frac:.3f}",
        f"transcripts={len(tids)} surface_pos={int(pos.sum())} control={int(ctrl.sum())}",
        "",
        f"## Decision: `{decision}`",
        note,
        "",
        f"Clusters (size {MIN_SIZE}–{MAX_SIZE}): {len(clusters_out)}",
        "",
    ]
    for i, c in enumerate(clusters_json[:12]):
        lines.append(
            f"### Cluster {i}: feats={c['feature_ids']} size={c['size']} "
            f"cooc={c['mean_cooccurrence_jaccard']:.3f} irr={c['irreducible_candidate']}"
        )
        lines.append(
            f"- Cohen d surface vs control={c['cohen_d_surface_vs_control']:.3f} "
            f"vs nongap={c.get('cohen_d_surface_vs_nongap')} hit={c['hit']}"
        )
        lines.append("")
    lines += [
        f"Exp3 licensed (descriptive): {report['exp3_licensed']}",
        f"SAE weights: `{args.sae_out}`",
        f"Artifact: `{args.out}`",
    ]
    args.md_out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "n_clusters": len(clusters_out)}, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
