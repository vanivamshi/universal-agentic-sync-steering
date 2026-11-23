#!/usr/bin/env python3
"""Phase 9Z — maintain 111 across the whole agent loop (preregistered; docs/sync_channel_control.md).

Stages
  z1  recording-only A40 baseline: arm B (frozen 9X arm B), T=5, read-only margin probes,
      determinism check on the first 5 acquired conversations (re-run with probes off).
  z3  evaluation: paired forks B vs BH (B + H steering on in-round retry turns), T=5, probes on.
  report  re-render markdown (z1: Z2 validation; z3: decision) from saved JSON.

  python -u scripts/run_phase9z_retention.py --stage z1 --device cuda [--resume]
  python -u scripts/run_phase9z_retention.py --stage z3 --device cuda [--resume]
  python -u scripts/run_phase9z_retention.py --stage z1 --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_phase9x_111_confirm as p9x  # noqa: E402
from scripts import sync_adaptive_retention as sar  # noqa: E402
from scripts import sync_retention_9z as z  # noqa: E402

RESULTS = ROOT / "data" / "results"
TARGET = z.TARGET
T_ROUNDS = z.T_ROUNDS
SEED_STRIDE = p9x.SEED_STRIDE
ROUND_STRIDE = p9x.ROUND_STRIDE
N_ACQ = 100
MAX_TRIES = 700
N_DETERMINISM = 5

STAGES: dict[str, dict[str, Any]] = {
    "z1": {"seed": 61000000, "arms": ("B",)},
    "z3": {"seed": 71000000, "arms": ("B", "BH")},
}
Z4_SEED = 81000000  # reserved; only via a committed addendum

EXPECTED_ENV = {"torch": "2.8.0+cu129", "transformers": "5.24.0", "cuda": "12.9",
                "device_name": "NVIDIA A40", "dtype": "float32"}


OUT_DIR = RESULTS


def out_paths(stage: str) -> tuple[Path, Path]:
    return OUT_DIR / f"sync_phase9z_{stage}.json", OUT_DIR / f"sync_phase9z_{stage}.md"


def acq_seed(seed: int, i: int) -> int:
    return seed + SEED_STRIDE * i


def round_seeds(s0: int, rounds: int) -> list[int]:
    return [s0 + ROUND_STRIDE * k for k in range(1, rounds + 1)]


def envelope(seed: int, max_tries: int = MAX_TRIES, rounds: int = T_ROUNDS) -> tuple[int, int]:
    return acq_seed(seed, 0), round_seeds(acq_seed(seed, max_tries - 1), rounds)[-1]


def prior_envelopes() -> dict[str, tuple[int, int]]:
    p9w = p9x.p9w
    return {"9U": (p9w._9U_LO, p9w._9U_HI), "9V": (p9w._9V_LO, p9w._9V_HI), "9W": (p9x._9W_LO, p9x._9W_HI),
            "9X/9Y": envelope(p9x.SEED, p9x.MAX_TRIES, 2)}


def assert_new_seeds(stage: str, seed: int, max_tries: int, rounds: int) -> None:
    lo, hi = envelope(seed, max_tries, rounds)
    others = dict(prior_envelopes())
    for st, cfg in STAGES.items():
        if st != stage:
            others[st] = envelope(cfg["seed"])
    others["z4"] = envelope(Z4_SEED)
    for name, (a, b) in others.items():
        assert hi < a or lo > b, f"9Z {stage} seeds [{lo},{hi}] overlap {name} [{a},{b}]"


def arm_order(s0: int, arms: tuple[str, ...]) -> list[str]:
    if len(arms) == 1:
        return list(arms)
    return list(arms) if random.Random(s0 ^ 0x9A9A).random() < 0.5 else list(reversed(arms))


def environment() -> dict[str, Any]:
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "float32"}


# --------------------------------------------------------------------------- recorder

class MarginRecorder:
    """Read-only logit margins at the decision sites; no generation state is touched."""

    PAIRS = {"C_stem": ("run", "only"), "H_post_plan": ("<tool_call>", "FINAL"),
             "H_retry": ("<tool_call>", "FINAL"), "O_final": ("During", "docs")}

    # Space-aware first-token sets at the C stem (fixed in the 9Z amendment before any run).
    C_COMMIT = (" run", " load", " read", " execute", " open", " access", " use")
    C_NONCOMMIT = (" not", " only", " check", " verify", " review", " first", " ask", " look", " confirm", " avoid")

    def __init__(self, loaded):
        from scripts.sync_channel_margins import token_id
        self.loaded = loaded
        tok = loaded.tokenizer
        self.ids = {s: (token_id(tok, a), token_id(tok, b)) for s, (a, b) in self.PAIRS.items()}
        self.id_tool = token_id(tok, "<tool_call>")
        self.c_commit = self._single_ids(tok, self.C_COMMIT)
        self.c_noncommit = self._single_ids(tok, self.C_NONCOMMIT)
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _single_ids(tok, words) -> list[int]:
        ids = []
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            assert len(t) == 1, f"C set word {w!r} is not a single token: {t}"
            ids.append(int(t[0]))
        return ids

    def _forward(self, messages, prefill, hooks, site):
        import torch
        from scripts.sync_h_decision import chat_prompt
        tok, model = self.loaded.tokenizer, self.loaded.model
        enc = tok(chat_prompt(tok, messages, prefill), return_tensors="pt")
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        active = [h for h in hooks if h is not None]
        for h in active:
            h.register()
        try:
            with torch.inference_mode():
                logits = model(**enc, use_cache=False).logits[0, -1].float()
        finally:
            for h in reversed(active):
                h.remove()
        probs = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        ip, ineg = self.ids[site]
        top = torch.topk(probs, 10)
        row = {"M_legacy": float(logits[ip] - logits[ineg]), "P_pos": float(probs[ip]), "P_neg": float(probs[ineg]),
               "P_tool": float(probs[self.id_tool]), "argmax": tok.decode([int(logits.argmax())]),
               "top": [[tok.decode([int(i)]), round(float(p), 5)] for p, i in zip(top.values.tolist(), top.indices.tolist())],
               "n_hooks": len(active)}
        if site == "C_stem":
            lc = torch.logsumexp(logp[self.c_commit], 0)
            ln = torch.logsumexp(logp[self.c_noncommit], 0)
            row["M"] = float(lc - ln)
            row["P_commit"], row["P_noncommit"] = float(lc.exp()), float(ln.exp())
        else:
            row["M"] = row["M_legacy"]
        return row

    def probe(self, site, turn, messages, prefill, hooks):
        self.rows.append({"site": site, "turn": int(turn),
                          "as_run": self._forward(messages, prefill, hooks, site),
                          "unsteered": self._forward(messages, prefill, [], site)})

    def drain(self) -> list[dict[str, Any]]:
        out, self.rows = self.rows, []
        return out


# --------------------------------------------------------------------------- run

def _sig(trial_arm: dict[str, Any]) -> list:
    return [(r["after"], r["info"].get("plan"), r["info"].get("final"), r["info"].get("tools"),
             r["info"].get("sensitive_paths")) for r in trial_arm["rounds"]]


def run(args) -> int:
    from scripts.run_adaptive_retention import _load_mod, make_round_fn
    from scripts.run_phase9u_retention_gains import run_fixed
    from activation_pipeline.device import LOCAL_MODEL_KEY, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    stage = args.stage
    arms = STAGES[stage]["arms"]
    seed = args.seed or STAGES[stage]["seed"]
    assert_new_seeds(stage, seed, args.max_tries, args.rounds)
    cfg = p9x.load_frozen()
    action_B = p9x.arm_action(cfg, "B")
    assert action_B.gains() == {"C": 0.5, "H": 1.5, "O": 1.5}
    env = environment()
    if not args.allow_env_mismatch:
        assert env == EXPECTED_ENV, f"environment differs from preregistered A40 cohort: {env}"
    out_json, out_md = out_paths(stage)

    payload: dict[str, Any] = {
        "protocol": f"9Z {stage}: " + ("recording-only B baseline (calibration)" if stage == "z1"
                                       else "paired B vs BH (H on in-round retries)"),
        "target": list(TARGET), "stage": stage, "seed": seed, "rounds": args.rounds,
        "n_acq_wanted": args.n_acq, "max_tries": args.max_tries, "arms": list(arms),
        "frozen_config": str(p9x.FROZEN.relative_to(ROOT)), "frozen_config_sha256": p9x._sha(p9x.FROZEN),
        "action_B": {"name": action_B.name, "gains": action_B.gains()},
        "environment": env, "attempts": [], "trials": [], "determinism": [], "final": False,
    }
    if args.resume and out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        for k in ("arms", "seed", "rounds", "frozen_config_sha256", "environment"):
            assert prev[k] == payload[k], f"resume refused: {k} differs"
        for k in ("attempts", "trials", "determinism"):
            payload[k] = prev.get(k, [])
        print(f"resume: {len(payload['attempts'])} attempts, {len(payload['trials'])} trials", flush=True)

    def save():
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    assert p8k._alphas() == sar.GAINS
    sc = p8k._load_sc()
    if args.sandbox:
        sc.WORKSPACE = Path(args.sandbox).resolve()
    sc.ensure_sandbox()
    loaded = load_model_and_tokenizer(LOCAL_MODEL_KEY, device_map=resolve_device_map(args.device or None),
                                      dtype="float32", local_files_only=args.local_files_only)
    dirs = p8k._load_vc()
    rec = MarginRecorder(loaded)
    rf_plain = make_round_fn(sc, p8k, loaded, dirs)
    rf = {"B": make_round_fn(sc, p8k, loaded, dirs, recorder=rec),
          "BH": make_round_fn(sc, p8k, loaded, dirs, h_on_retry=True, recorder=rec)}

    print(f"=== 9Z {stage} arms={arms} seed={seed} T={args.rounds} need={args.n_acq} env={env} ===", flush=True)
    i = len(payload["attempts"])
    while len(payload["trials"]) < args.n_acq and i < args.max_tries:
        sd = acq_seed(seed, i)
        i += 1
        msgs0, S0, _ = rf_plain(None, sar.CONTINUOUS, list(TARGET), sd)
        ok = S0 == list(TARGET)
        payload["attempts"].append({"i": i - 1, "seed": sd, "S0": list(S0), "acquired": ok})
        print(f"  acquire try={i} seed={sd} S0={sar.key(S0)} {'KEEP' if ok else ''}", flush=True)
        if not ok:
            save()
            continue
        rs = round_seeds(sd, args.rounds)
        order = arm_order(sd, arms)
        trial: dict[str, Any] = {"seed": sd, "round_seeds": rs, "order": order}
        for a in order:
            trial[a] = run_fixed(rf[a], msgs0, action_B, TARGET, rs)
        payload["trials"].append(trial)
        print(f"    order={'/'.join(order)} " + " | ".join(
            f"{a} {'>'.join(trial[a]['path'])} {'hold' if trial[a]['hold'] else 'FAIL'}"
            + (f" retryH@{[r['k'] for r in trial[a]['rounds'] if r['info'].get('h_retry_fired')]}" if a == "BH" else "")
            for a in arms), flush=True)
        if stage == "z1" and len(payload["determinism"]) < N_DETERMINISM and len(payload["trials"]) <= N_DETERMINISM:
            plain = run_fixed(rf_plain, msgs0, action_B, TARGET, rs)
            same = _sig(plain) == _sig(trial["B"])
            payload["determinism"].append({"seed": sd, "identical": same})
            print(f"    determinism (probes off vs on): {'identical' if same else 'MISMATCH'}", flush=True)
            if not same:
                save()
                raise SystemExit("determinism check failed: probes changed generation; stopping (preregistered)")
        save()

    payload["final"] = len(payload["trials"]) >= args.n_acq or i >= args.max_tries
    payload["summary"] = summarize(payload)
    save()
    write_md(out_md, payload)
    print(json.dumps({"out": str(out_json), "summary_head": _head(payload)}, default=str), flush=True)
    return 0


# --------------------------------------------------------------------------- analysis

def summarize(p: dict[str, Any]) -> dict[str, Any]:
    trials = p["trials"]
    n_att = len(p["attempts"])
    from scripts.sync_retention_controller import wilson
    s: dict[str, Any] = {"n_acq": len(trials), "tries": n_att,
                         "P_acq": len(trials) / n_att if n_att else None, "P_acq_ci": wilson(len(trials), n_att),
                         "determinism": {"n": len(p.get("determinism", [])),
                                         "identical": sum(1 for d in p.get("determinism", []) if d["identical"])}}
    if p["stage"] == "z1":
        bt = [t["B"] for t in trials]
        s["B"] = z.arm_summary(bt)
        s["z2"] = {"C_gate": z.c_gate(bt), "H_diagnostics": z.h_diagnostics(bt),
                   "mechanism_all_failed_rounds_entered_on_target": _mech_counts(bt)}
    else:
        s["z3"] = z.z3_summary(trials)
        s["z3"]["mechanisms_entered_on_target"] = {a: _mech_counts([t[a] for t in trials]) for a in p["arms"]}
    return s


def _mech_counts(trials) -> dict[str, int]:
    from collections import Counter
    c = Counter()
    for _ci, r, info in z.rounds_entered_on_target(trials):
        m = z.mechanism(r["after"], info.get("plan", ""))
        if m:
            c[m] += 1
    return dict(c)


def _head(p):
    s = p["summary"]
    if p["stage"] == "z1":
        return {"hold_B": s["B"]["n_hold"], "n": s["n_acq"], "C_gate": s["z2"]["C_gate"]["reason"]}
    d = s["z3"]
    return {a: d["arms"][a]["n_hold"] for a in p["arms"]} | {"verdict": d["decision"]["verdict"]}


def _ci(ci) -> str:
    return "—" if not ci or ci[0] is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _f(x, nd=2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _arm_rows(name: str, a: dict[str, Any]) -> list[str]:
    L = [f"### {name}: hold all {len(a['per_round'])} rounds = **{a['n_hold']}/{a['n']}** {_ci(a['ci'])}", "",
         "| round k | entered at 111 | stayed | $P(S_k{=}111\\mid S_{k-1}{=}111)$ | 95% CI | survive to k | retry rounds |",
         "|---:|---:|---:|---:|---|---:|---|"]
    for r in a["per_round"]:
        L.append(f"| {r['k']} | {r['entered_on_target']} | {r['stayed']} | {_f(r['P_cond'])} | {_ci(r['P_cond_ci'])} | "
                 f"{_f(r['P_survive_to_k'])} | {a['retry_rate_by_round'].get(r['k'], a['retry_rate_by_round'].get(str(r['k']), '—'))} |")
    L += ["", f"- first drift round: {a['first_drift_round'] or '—'}",
          f"- departures per conversation: {a['departures']}; returns to 111: {a['n_returns']}",
          f"- first failing bits: {a['first_bits'] or '—'}",
          f"- first-failure mechanism: {a['first_mechanism'] or '—'}", ""]
    return L


def write_md(path: Path, p: dict[str, Any]) -> None:
    s = p.get("summary") or summarize(p)
    env = p["environment"]
    L = [f"# Phase 9Z {p['stage'].upper()} — {p['protocol'].split(': ', 1)[1]}", "",
         f"> Preregistered (docs/sync_channel_control.md, Phase 9Z). Target 111, T={p['rounds']} retention rounds, "
         f"seed base {p['seed']}. Arm B = frozen 9X B {p['action_B']['gains']}. Rounds continue after a failure; "
         "no repair. Hardware cohort: " f"{env['device_name']}, torch {env['torch']}, transformers {env['transformers']}, "
         f"CUDA {env['cuda']}, {env['dtype']} — not pooled with 9W–9Y (RTX A6000).", "",
         f"Acquired {s['n_acq']}/{s['tries']} (P={_f(s['P_acq'])} {_ci(s['P_acq_ci'])}); "
         f"{'complete' if p.get('final') else 'PARTIAL'}.", ""]
    if p["stage"] == "z1":
        d = s["determinism"]
        L += [f"Determinism check (probes off vs on): {d['identical']}/{d['n']} identical.", ""]
        L += _arm_rows("B (A40 baseline)", s["B"])
        g, h = s["z2"]["C_gate"], s["z2"]["H_diagnostics"]
        L += ["## Z2 — signal validation (calibration only)", "",
              f"Failed-round mechanisms (rounds entered at 111): {s['z2']['mechanism_all_failed_rounds_entered_on_target'] or '—'}", "",
              "**C gate** (positive = `C_contradict`, negative = held; risk = −M_C as run):", "",
              f"- n_pos = {g['n_pos']}, n_neg = {g['n_neg']}, missing margin = {g['n_missing_margin']}",
              f"- AUROC = {_f(g['auroc'])}, bootstrap 95% lower = {_f(g['auroc_boot_lb95'])}",
              f"- **gate: {'PASS' if g['passed'] else 'FAIL'}** ({g['reason']})",
              f"- transparency only: AUROC vs any scorer-C failure = {_f(g['transparency_auroc_vs_scorer_C'])} "
              f"(n_pos={g['transparency_n_scorer_C_pos']}); cannot pass the gate", "",
              "**H diagnostics** (no gate; the BH trigger is the observed missing tool call):", "",
              f"- turn-0 tool-call rate by round: {h['turn0_tool_call_rate_by_round']}",
              f"- P(H fails | no tool call on turn 0) = {h['P_Hfail_given_turn0_nocall']}; "
              f"P(no tool call on turn 0 | H fails) = {h['P_turn0_nocall_given_Hfail']}",
              f"- AUROC of −M_H (turn 0, as run) for no tool call on turn 0 = {_f(h['auroc_MH_turn0_for_nocall'])}; "
              f"for H failure = {_f(h['auroc_MH_turn0_for_Hfail'])}", ""]
    else:
        d = s["z3"]
        L += _arm_rows("B", d["arms"]["B"]) + _arm_rows("BH (B + H on retries)", d["arms"]["BH"])
        pr, at, dec = d["paired"], d["attribution"], d["decision"]
        L += ["## Primary: paired BH vs B on $P(S_1{=}\\dots{=}S_5{=}111\\mid S_0{=}111)$", "",
              f"Both hold {pr['both']}, BH-only **{pr['arm_only']}**, B-only **{pr['ref_only']}**. "
              f"Exact McNemar p = **{pr['mcnemar_p']:.3f}**.", "",
              "## Steering attribution (first round in which the BH trigger fires)", "",
              f"- conversations where it fired: {at['n_conversations_trigger_fired']} "
              f"(identical to B before the trigger: {at['identical_before_trigger']})",
              f"- outcomes in that round: {at['outcomes'] or '—'}",
              f"- BH lost C or O while H held (and B kept it): {at['bh_lost_C_or_O_with_H_held']}", "",
              f"Mechanisms of failed rounds entered at 111: {d['mechanisms_entered_on_target']}", "",
              "## Decision (preregistered)", "",
              f"- hold(BH) ≥ hold(B): {dec['hold_BH_ge_B']}; BH-only > B-only: {dec['BH_only_gt_B_only']}; "
              f"rescued > broken: {dec['rescued_gt_broken']}",
              f"- **{dec['verdict']}**", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def report(args) -> int:
    out_json, out_md = out_paths(args.stage)
    p = json.loads(out_json.read_text(encoding="utf-8"))
    p["summary"] = summarize(p)
    out_json.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")
    write_md(out_md, p)
    print(out_md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--rounds", type=int, default=T_ROUNDS)
    ap.add_argument("--n-acq", type=int, default=N_ACQ)
    ap.add_argument("--max-tries", type=int, default=MAX_TRIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--device", default="")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--allow-env-mismatch", action="store_true", help="smoke tests only; never for 9Z results")
    ap.add_argument("--out-dir", default="", help="smoke tests only: write outside data/results")
    args = ap.parse_args()
    if args.out_dir:
        global OUT_DIR
        OUT_DIR = Path(args.out_dir).resolve()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        return report(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
