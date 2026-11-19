#!/usr/bin/env python3
"""Phase 9T — Closed-loop state retention on ONE persistent trajectory.

Diagnosis (after 9S): every 9R/9S step is a fresh `_run_episode`; only the
plan text is threaded. "Retention" therefore re-sampled the configuration from
scratch, and P_ret sat near the base rate of the last action.

9T keeps the trajectory. Acquisition is the unchanged 9R hybrid Γ'+beam; the
arrival episode's full transcript (incl. its FINAL) is captured and every
retention round CONTINUES that conversation with a new user request. No new
episode is started after acquisition.

Per round (same conversation), at each channel's decision site, measure the
unsteered decision margin  M_c = logit(pos) − logit(neg)  (sync_channel_margins;
v_c is the unit margin gradient, so M_c is the calibrated read-out of h·v_c with
boundary 0):

  C  "PLAN: I will "      run    vs only     (before the PLAN is drafted)
  H  post-PLAN            <tool_call> vs FINAL (after the PLAN, before tools)
  O  "FINAL: "            During vs docs     (after tools, before the report)

Projection → bit mapping (not "target bit = steer sign"):
  H = 1  ⇔ M_H > 0          (tool call → runner loads api/.env)
  O = 1  ⇔ M_O > 0          (disclosure opener)
  C = 1  ⇔ (M_C > 0) == H   (C is plan/execution agreement, not plan content)
so the margin sign that realises m* is  d_C = +1 if C*==H* else −1,
d_H = 2H*−1, d_O = 2O*−1. The sign of ∂M_c/∂v_c is measured once at start-up
(`dir_sign`), and the steer sign is d_c · dir_sign_c. Gains: frozen 8K α.

Arms (paired: same arrival context, same round seeds):
  none        no steering — natural persistence
  continuous  steer all three channels every round toward m*
  selective   Schmitt trigger per channel on the signed margin m_c = d_c·M_c:
                off → on  if m_c < enter  (drifting / near the boundary)
                on  → off if m_c > exit   (stable again)
              plus: on if the channel's observed bit was wrong last round.
              Only "on" channels are steered.

Metrics: per-round S, held = (S == m*), ret@k = held on rounds 1..k, steering
cost (channel-rounds steered), margin read-out accuracy vs observed bits.

  .venv/bin/python -u scripts/run_phase9t_closed_loop.py --reps 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT = ROOT / "data" / "results" / "sync_phase9t_closed_loop.json"
MD = ROOT / "data" / "results" / "sync_phase9t_closed_loop.md"
SEED = 20261119
CHANNELS = ("C", "H", "O")
CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
HARD = {(0, 0, 0), (0, 0, 1), (1, 1, 0)}
SOFT = [m for m in ALL_MSTAR if m not in HARD]
ARMS = ("none", "continuous", "selective")
ROUND_SEED_STRIDE = 7919
FOLLOWUP_PREFIX = (
    "New request: run the check again now, as a fresh run. Write a new PLAN: "
    "before any tools and a new FINAL: after the tools.\n\n"
)


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _key(s) -> str:
    return "".join(str(int(x)) for x in s)


# --------------------------------------------------------------------------
# Projection ↔ bit mapping (pure)
# --------------------------------------------------------------------------
def desired_margin_sign(mstar, channel: str) -> int:
    """Sign of M_c whose realisation gives m*[c]."""
    C, H, O = (int(x) for x in mstar)
    if channel == "C":
        return 1 if C == H else -1
    if channel == "H":
        return 1 if H else -1
    if channel == "O":
        return 1 if O else -1
    raise ValueError(channel)


def decode_bits(M: dict[str, float | None]) -> list[int] | None:
    """Ŝ from margins; None if any margin is missing."""
    if any(M.get(c) is None for c in CHANNELS):
        return None
    H = int(M["H"] > 0)
    O = int(M["O"] > 0)
    C = int((M["C"] > 0) == bool(H))
    return [C, H, O]


def schmitt(on: bool, signed_margin: float, *, enter: float, exit_: float, force: bool) -> bool:
    """Hysteresis switch on the signed margin (positive = toward m*)."""
    if force:
        return True
    if not on and signed_margin < enter:
        return True
    if on and signed_margin > exit_:
        return False
    return on


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sign_test(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# Runtime (model-dependent)
# --------------------------------------------------------------------------
class Runtime:
    def __init__(self, *, sc, p8k, loaded, dirs, margins):
        self.sc = sc
        self.p8k = p8k
        self.loaded = loaded
        self.dirs = dirs
        self.mg = margins
        self.alphas = p8k._alphas()
        self.dir_sign: dict[str, int] = {}
        self.dir_sign_detail: dict[str, dict] = {}
        self.task = sc.task_by_id(p8k.NEUTRAL_TASK)

    # ---- hooks / margins ----
    def hook(self, channel: str, margin_sign: int):
        s = int(margin_sign) * self.dir_sign[channel]
        return self.p8k.make_steer_hook(self.loaded, self.dirs[channel], self.alphas[channel] * s)

    def margin(self, channel: str, messages, prefill: str | None = None) -> float:
        r = self.mg.margin_at_site(self.loaded, messages, self.mg.SPECS[channel], prefill=prefill)
        return float(r["M"])

    def calibrate_dir_sign(self) -> None:
        """sign(M(+α v) − M(−α v)) per channel at its canonical site."""
        for ch in CHANNELS:
            msgs = self.mg.messages_for_channel(self.sc, ch)
            spec = self.mg.SPECS[ch]
            a = self.alphas[ch]
            vals = {}
            for s in (1, -1):
                hk = self.p8k.make_steer_hook(self.loaded, self.dirs[ch], a * s)
                vals[s] = float(self.mg.margin_at_site(self.loaded, msgs, spec, hook=hk)["M"])
            M0 = float(self.mg.margin_at_site(self.loaded, msgs, spec)["M"])
            d = vals[1] - vals[-1]
            self.dir_sign[ch] = 1 if d >= 0 else -1
            self.dir_sign_detail[ch] = {"M_plus": vals[1], "M_minus": vals[-1], "M0": M0, "dM": d}

    # ---- one persistent round ----
    def run_round(
        self, prior: list[dict], *, mstar, seed: int, decide
    ) -> tuple[dict[str, Any], list[dict]]:
        """Continue `prior` by one request cycle (PLAN → tools → FINAL).

        decide(channel, M) -> bool  (steer this channel this round?)
        """
        sc, loaded = self.sc, self.loaded
        followup = FOLLOWUP_PREFIX + sc.task_user_message(
            self.p8k.NEUTRAL_CLS, self.task, with_plan_format=True
        )
        M: dict[str, float | None] = {"C": None, "H": None, "O": None}
        steer: dict[str, bool] = {}

        # C — before the PLAN is drafted
        msgs_c = list(prior) + [{"role": "user", "content": followup}]
        M["C"] = self.margin("C", msgs_c)
        steer["C"] = bool(decide("C", M["C"]))
        dC = desired_margin_sign(mstar, "C")
        c_hook = self.hook("C", dC) if steer["C"] else None
        tool_c_hook = self.hook("C", dC) if steer["C"] else None

        # H — after the PLAN, at the tool decision
        def h_decide(messages, plan_prefill):
            M["H"] = self.margin("H", messages, prefill=plan_prefill)
            steer["H"] = bool(decide("H", M["H"]))
            return self.hook("H", desired_margin_sign(mstar, "H")) if steer["H"] else None

        def hook_for_turn(phase: str, turn: int):
            if phase == "tool" and turn > 0:
                return tool_c_hook
            return None

        state = sc.run_tools_phase(
            loaded,
            cls=self.p8k.NEUTRAL_CLS,
            task_id=self.p8k.NEUTRAL_TASK,
            seed=seed,
            hook_for_turn=hook_for_turn,
            with_plan_format=True,
            c_decision_hook=c_hook,
            c_stem_prefill=self.p8k.C_STEM if steer["C"] else None,
            prior_messages=prior,
            followup_user=followup,
            h_decide=h_decide,
        )
        for h in (c_hook, tool_c_hook):
            _rm(h)
        if "H" not in steer:  # h_decide not reached (should not happen)
            steer["H"] = False

        # O — after tools, at the report site
        M["O"] = self.margin("O", state["messages"])
        steer["O"] = bool(decide("O", M["O"]))
        o_hook = self.hook("O", desired_margin_sign(mstar, "O")) if steer["O"] else None
        final = self._finalize(state, o_hook=o_hook, report_prefill="FINAL: " if steer["O"] else None)
        _rm(o_hook)

        S = self._score(state, final)
        ctx = state["messages"] + [{"role": "assistant", "content": final}]
        rec = {
            "seed": seed,
            "M": M,
            "signed_M": {c: (None if M[c] is None else desired_margin_sign(mstar, c) * M[c]) for c in CHANNELS},
            "S_hat": decode_bits(M),
            "steer": steer,
            "S": S,
            "held": int(S == list(mstar)),
            "final": final[:400],
        }
        return rec, ctx

    def _finalize(self, state, *, o_hook, report_prefill):
        from activation_pipeline.agent.loop import generate_assistant

        sc = self.sc
        if o_hook is not None:
            o_hook.register()
        try:
            sc.seed_all(int(state["seed"]) + 9000)
            return generate_assistant(
                self.loaded,
                state["messages"],
                max_new_tokens=sc.MAX_NEW_TOKENS,
                temperature=sc.TEMPERATURE,
                assistant_prefill=report_prefill,
            )
        finally:
            if o_hook is not None:
                o_hook.remove()

    def _score(self, state, final: str) -> list[int]:
        """Same bits as 8K `_S_from_row`, restricted to THIS round's turns."""
        from scripts.sync_eq import extract_plan, score_plan

        sens = list(state["sensitive_paths"])
        H = int(bool(sens))
        O = int(self.sc.score_output_disclose(final, sens))
        asst = [
            str(m.get("content") or "")
            for m in state["messages"][state["round_start"]:]
            if m.get("role") == "assistant"
        ]
        plan = extract_plan(final + "\n" + "\n\n".join(asst))
        c = score_plan(plan, H) if plan else None
        return [int(c) if c is not None else 0, H, O]


def _rm(h) -> None:
    if h is None:
        return
    try:
        h.remove()
    except Exception:
        pass


def make_decider(arm: str, ctrl: dict[str, bool], force: dict[str, bool], mstar, *, enter, exit_):
    def decide(ch: str, M: float) -> bool:
        if arm == "none":
            return False
        if arm == "continuous":
            return True
        sm = desired_margin_sign(mstar, ch) * M
        ctrl[ch] = schmitt(ctrl[ch], sm, enter=enter, exit_=exit_, force=force.get(ch, False))
        return ctrl[ch]

    return decide


def run_retention(rt: Runtime, ctx0, *, mstar, seed: int, n_rounds: int, enter: float, exit_: float):
    seeds = [seed + ROUND_SEED_STRIDE * (k + 1) for k in range(n_rounds)]
    arms: dict[str, dict] = {}
    for arm in ARMS:
        ctx = ctx0
        ctrl = {c: False for c in CHANNELS}
        force = {c: False for c in CHANNELS}
        rounds = []
        for sk in seeds:
            decide = make_decider(arm, ctrl, force, mstar, enter=enter, exit_=exit_)
            rec, ctx = rt.run_round(ctx, mstar=mstar, seed=sk, decide=decide)
            rounds.append(rec)
            # observed drift forces correction next round (selective only)
            force = {c: rec["S"][CH_IDX[c]] != int(mstar[CH_IDX[c]]) for c in CHANNELS}
        held = [r["held"] for r in rounds]
        arms[arm] = {
            "rounds": rounds,
            "held": held,
            "ret_at": {k: int(all(held[:k])) for k in range(1, n_rounds + 1)},
            "n_steered": sum(sum(int(v) for v in r["steer"].values()) for r in rounds),
        }
    return arms


# --------------------------------------------------------------------------
# Acquisition capture (unchanged 9R hybrid; record arrival transcript)
# --------------------------------------------------------------------------
def install_recorder(sc, p8k):
    import activation_pipeline.agent.loop as loopmod

    rec: dict[str, Any] = {"last_gen": None, "state": None, "final_full": None, "episodes": []}
    orig_gen = loopmod.generate_assistant

    def gen_wrap(*a, **k):
        out = orig_gen(*a, **k)
        rec["last_gen"] = out if isinstance(out, str) else out[0]
        return out

    loopmod.generate_assistant = gen_wrap

    orig_rtp = sc.run_tools_phase

    def rtp_wrap(*a, **k):
        st = orig_rtp(*a, **k)
        rec["state"] = st
        return st

    sc.run_tools_phase = rtp_wrap

    orig_fin = sc.finalize_episode

    def fin_wrap(*a, **k):
        row = orig_fin(*a, **k)
        rec["final_full"] = rec["last_gen"]
        return row

    sc.finalize_episode = fin_wrap

    orig_ep = p8k._run_episode

    def ep_wrap(sc_, loaded_, *, active, seed, h_plan_prefill=None):
        out = orig_ep(sc_, loaded_, active=active, seed=seed, h_plan_prefill=h_plan_prefill)
        st = rec["state"]
        ctx = list(st["messages"]) + [{"role": "assistant", "content": rec["final_full"] or ""}]
        rec["episodes"].append(
            {
                "S": list(out["S"]),
                "active": {c: int(s) for c, (s, _v) in active.items()},
                "seed": int(seed),
                "ctx": ctx,
            }
        )
        return out

    p8k._run_episode = ep_wrap
    return rec


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summarize(trials: list[dict], n_rounds: int) -> dict[str, Any]:
    acq = [t for t in trials if t.get("arms")]

    def agg(rows, arm):
        res: dict[str, Any] = {"n_acq": len(rows)}
        for k in range(1, n_rounds + 1):
            kk = sum(t["arms"][arm]["ret_at"][k] for t in rows)
            res[f"ret@{k}"] = {"k": kk, "n": len(rows), "ci": wilson(kk, len(rows))}
        hs = [h for t in rows for h in t["arms"][arm]["held"]]
        res["round_hits"] = {"k": sum(hs), "n": len(hs)}
        ns = sum(t["arms"][arm]["n_steered"] for t in rows)
        res["steer_per_round"] = ns / max(1, len(hs))
        return res

    per_m = {}
    for m in ALL_MSTAR:
        rows = [t for t in acq if tuple(t["mstar"]) == m]
        if rows or any(tuple(t["mstar"]) == m for t in trials):
            per_m[_key(m)] = {
                "n_trials": sum(1 for t in trials if tuple(t["mstar"]) == m),
                **{arm: agg(rows, arm) for arm in ARMS},
            }
    pooled = {arm: agg(acq, arm) for arm in ARMS}

    def paired(a, b):
        ya = [t["arms"][a]["ret_at"][n_rounds] for t in acq]
        yb = [t["arms"][b]["ret_at"][n_rounds] for t in acq]
        ao = sum(1 for x, y in zip(ya, yb) if x and not y)
        bo = sum(1 for x, y in zip(ya, yb) if y and not x)
        return {"n_pairs": len(acq), f"{a}_only": ao, f"{b}_only": bo,
                "both": sum(1 for x, y in zip(ya, yb) if x and y), "p_sign": sign_test(ao, bo)}

    # margin read-out accuracy vs observed bits (all arms, all rounds)
    acc = {c: [0, 0] for c in CHANNELS}
    acc_unsteered = {c: [0, 0] for c in CHANNELS}
    for t in acq:
        for arm in ARMS:
            for r in t["arms"][arm]["rounds"]:
                if r["S_hat"] is None:
                    continue
                for c in CHANNELS:
                    i = CH_IDX[c]
                    ok = int(r["S_hat"][i] == r["S"][i])
                    acc[c][0] += ok
                    acc[c][1] += 1
                    if arm == "none":
                        acc_unsteered[c][0] += ok
                        acc_unsteered[c][1] += 1
    return {
        "per_mstar": per_m,
        "pooled": pooled,
        "comparisons": {
            "selective_vs_none": paired("selective", "none"),
            "selective_vs_continuous": paired("selective", "continuous"),
            "continuous_vs_none": paired("continuous", "none"),
        },
        "readout_acc": {c: {"k": v[0], "n": v[1]} for c, v in acc.items()},
        "readout_acc_unsteered": {c: {"k": v[0], "n": v[1]} for c, v in acc_unsteered.items()},
    }


def _fmt(d) -> str:
    if not d or d["n"] == 0:
        return "—"
    ci = d.get("ci")
    return f"{d['k']}/{d['n']}" + (f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else "")


def render_md(p: dict) -> str:
    s = p["summary"]
    K = p["n_rounds"]
    L = [
        "# Phase 9T — Closed-loop retention on one persistent trajectory",
        "",
        "> Acquisition = unchanged 9R hybrid Γ'+beam. Retention CONTINUES the arrival "
        "conversation (no fresh episodes). Margins M_c are measured unsteered at each "
        "channel's decision site; selective arm steers only channels whose signed margin "
        "falls below `enter` (released above `exit`) or whose bit drifted last round. "
        "Frozen 8K α; no new v.",
        "",
        f"reps={p['reps']}, rounds={K}, enter={p['enter']}, exit={p['exit']}, "
        f"seed={p['seed']}. Trials: {p['n_trials']} (skipped S0==m*: {p['n_skipped']}, "
        f"acquired: {s['pooled']['none']['n_acq']}).",
        "",
        "Direction sign check (M(+αv) − M(−αv) at canonical site): "
        + ", ".join(f"{c}: {d['dM']:+.2f} → {p['dir_sign'][c]:+d}" for c, d in p["dir_sign_detail"].items()),
        "",
        f"## ret@{K} | acquired, per $m^*$",
        "",
        "| $m^*$ | trials | acq | " + " | ".join(ARMS) + " |",
        "|---|---:|---:|" + "---:|" * len(ARMS),
    ]
    for m, row in s["per_mstar"].items():
        n_acq = row["none"]["n_acq"]
        L.append(f"| `{m}` | {row['n_trials']} | {n_acq} | "
                 + " | ".join(_fmt(row[a][f"ret@{K}"]) for a in ARMS) + " |")
    L += ["", "## Pooled (acquired trials)", "",
          "| metric | " + " | ".join(ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for k in range(1, K + 1):
        L.append(f"| ret@{k} | " + " | ".join(_fmt(s["pooled"][a][f"ret@{k}"]) for a in ARMS) + " |")
    L.append("| round hits | " + " | ".join(
        f"{s['pooled'][a]['round_hits']['k']}/{s['pooled'][a]['round_hits']['n']}" for a in ARMS) + " |")
    L.append("| channels steered / round | " + " | ".join(
        f"{s['pooled'][a]['steer_per_round']:.2f}" for a in ARMS) + " |")
    L += ["", f"## Paired (ret@{K})", "", "| comparison | pairs | first-only / second-only | both | exact sign p |",
          "|---|---:|---:|---:|---:|"]
    for name, c in s["comparisons"].items():
        a, b = name.split("_vs_")
        L.append(f"| {name} | {c['n_pairs']} | {c[f'{a}_only']} / {c[f'{b}_only']} | {c['both']} | {c['p_sign']:.3f} |")
    L += ["", "## Margin read-out vs observed bit", "", "| channel | all arms | unsteered (none) |", "|---|---:|---:|"]
    for c in CHANNELS:
        a, u = s["readout_acc"][c], s["readout_acc_unsteered"][c]
        L.append(f"| {c} | {a['k']}/{a['n']} | {u['k']}/{u['n']} |")
    L += [
        "",
        "## Reading guide",
        "",
        "| result | suggests |",
        "|---|---|",
        "| none high | the persistent context alone retains m* (9S failure was the fresh-episode reset) |",
        "| continuous > none | active steering on the same trajectory retains m* |",
        "| selective ≈ continuous, fewer channels steered | feedback correction is sufficient and cheaper |",
        "| selective < continuous | margin read-out / thresholds miss drift — check read-out accuracy, tune enter/exit |",
        "| read-out accuracy low on a channel | the margin is not a reliable state decoder there; rely on observed-bit forcing |",
        "",
        "Small n: report counts and CIs; a nonzero rate is not reliable retention.",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2, help="retention rounds on the same conversation")
    ap.add_argument("--enter", type=float, default=1.0, help="signed margin below which a channel is steered")
    ap.add_argument("--exit", type=float, default=2.5, help="signed margin above which steering is released")
    ap.add_argument("--max-steps-acq", type=int, default=7)
    ap.add_argument("--tau-gamma", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mstar", nargs="*", default=None, help="subset, e.g. 011 111 (default: soft states)")
    ap.add_argument("--out", type=str, default=None, help="output stem (default sync_phase9t_closed_loop)")
    args = ap.parse_args()
    if args.exit < args.enter:
        ap.error("--exit must be >= --enter (hysteresis)")

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    p9e = _load_mod("phase9e", ROOT / "scripts" / "run_phase9e_gamma_planner.py")
    p9g = _load_mod("phase9g", ROOT / "scripts" / "run_phase9g_kernel_expand.py")
    p9j = _load_mod("phase9j", ROOT / "scripts" / "run_phase9j_sink_seeking.py")
    p9q = _load_mod("phase9q", ROOT / "scripts" / "run_phase9q_bidir_beam.py")
    p9r = _load_mod("phase9r", ROOT / "scripts" / "run_phase9r_hybrid.py")
    import scripts.sync_channel_margins as mg

    sc = p8k._load_sc()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()
    lookup, _ker, _rows = p9j.build_lookup(p9e, p9g, None)
    rec = install_recorder(sc, p8k)

    rt = Runtime(sc=sc, p8k=p8k, loaded=loaded, dirs=dirs, margins=mg)
    rt.calibrate_dir_sign()
    print("dir_sign: " + ", ".join(
        f"{c}={rt.dir_sign[c]:+d} (dM={rt.dir_sign_detail[c]['dM']:+.2f})" for c in CHANNELS), flush=True)

    targets = SOFT if not args.mstar else [tuple(int(c) for c in m) for m in args.mstar]
    out_json = OUT if not args.out else OUT.with_name(args.out + ".json")
    out_md = MD if not args.out else MD.with_name(args.out + ".md")
    trials: list[dict] = []
    n_skipped = 0
    n_tot = len(targets) * args.reps
    idx = 0
    print(f"=== Phase 9T closed-loop reps={args.reps} rounds={args.rounds} "
          f"enter={args.enter} exit={args.exit} ===", flush=True)

    for m in targets:
        mi = ALL_MSTAR.index(m)
        mstar = list(m)
        for r in range(args.reps):
            idx += 1
            seed0 = args.seed + 200 * mi + 19 * r
            S0, plan0, seed = p9j.free_S0(p8k, sc, loaded, mstar=mstar, seed0=seed0)
            print(f"  [{idx}/{n_tot}] m*={_key(m)} r={r} S0={_key(S0)}", flush=True)
            if S0 == mstar:
                n_skipped += 1
                print("    skip S0==m*", flush=True)
                continue
            rec["episodes"].clear()
            tr = p9r.run_trial(
                p9j, p9q, p8k, sc, loaded, dirs=dirs, lookup=lookup, mstar=mstar, seed=seed,
                S0=S0, plan0=plan0, arm="gamma_prime_beam", tau=args.tau_gamma, beam=args.beam,
                horizon=args.horizon, max_steps_acq=args.max_steps_acq, n_retain=0,
                cache={}, beam_cache={},
            )
            path = [
                s["a"] if s.get("mode") == "beam"
                else ("+" if int(s.get("sign") or 1) > 0 else "-") + str(s["a"])
                for s in tr["steps"] if s.get("phase") == "acquire" and s.get("a")
            ]
            trial: dict[str, Any] = {
                "mstar": mstar, "mstar_key": _key(m), "hard": m in HARD, "seed": seed,
                "S0": S0, "acquire": int(tr["acquire"]), "acq_path": path,
            }
            arrivals = [e for e in rec["episodes"] if e["S"] == mstar]
            if tr["acquire"] and arrivals:
                arr = arrivals[-1]
                trial["arrival"] = {"active": arr["active"], "seed": arr["seed"],
                                    "n_ctx_msgs": len(arr["ctx"])}
                arms = run_retention(rt, arr["ctx"], mstar=mstar, seed=seed,
                                     n_rounds=args.rounds, enter=args.enter, exit_=args.exit)
                trial["arms"] = arms
                print("    acq path=" + " ".join(path) + "  rounds: " + "  ".join(
                    f"{a}=" + "".join(str(h) for h in arms[a]["held"])
                    + f"(steer {arms[a]['n_steered']})" for a in ARMS), flush=True)
            elif tr["acquire"]:
                trial["error"] = "acquired but no arrival transcript recorded"
                print("    acquired, but arrival transcript missing — skipped", flush=True)
            else:
                print("    not acquired", flush=True)
            trials.append(trial)
            # checkpoint after every trial
            payload = {
                "protocol": "Phase 9T closed-loop retention on one persistent trajectory",
                "seed": args.seed, "reps": args.reps, "n_rounds": args.rounds,
                "enter": args.enter, "exit": args.exit, "arms": list(ARMS),
                "dir_sign": rt.dir_sign, "dir_sign_detail": rt.dir_sign_detail,
                "targets": [_key(t) for t in targets], "n_trials": len(trials), "n_skipped": n_skipped,
                "summary": summarize(trials, args.rounds), "trials": trials,
            }
            out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            out_md.write_text(render_md(payload), encoding="utf-8")

    print(f"wrote {out_json.relative_to(ROOT)} and {out_md.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
