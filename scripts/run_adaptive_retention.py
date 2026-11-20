#!/usr/bin/env python3
"""Adaptive retention controller — same-conversation retention, 4 arms.

Fixes the retention protocol of 9L/9M/9R, where every "retain" step was a
fresh ``run_episode`` from the neutral task (only PLAN text carried). Here a
trial is ONE conversation:

  round 0  task turn, acquisition steer (shared; forked to every arm)
  round k  follow-up user turn → PLAN → tools → FINAL, scored as S_k

Arms (paired: same round-0 conversation, same per-round seeds):
  none            no steering
  continuous      C,H,O target-sign every round, 8K sites incl. forced FINAL:
  event_triggered C,H,O (no forced FINAL:) only after observed drift
  adaptive        drift → rule table + empirical success table; FULL fallback

Primary: P_hold = P(S_1 = S_2 = m*) (``--rounds 2``). Also recovery rate
after drift and collateral (correct→wrong) rate, steered vs unsteered.

Frozen 8K actuators only: v_c, α_C=5 stem "PLAN: I will ", α_H=1.5
decision-token, α_O=1.5 report. No new v, no new gains, target-sign only.

  .venv/bin/python -u scripts/run_adaptive_retention.py --reps 10
  .venv/bin/python -u scripts/run_adaptive_retention.py --reps 2 --targets 011,100 --tag smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from scripts import sync_adaptive_retention as sar  # noqa: E402

RESULTS = ROOT / "data" / "results"
SEED = 20261120
ALL_MSTAR = [(c, h, o) for c in (0, 1) for h in (0, 1) for o in (0, 1)]
NEUTRAL_TASK = "api"
NEUTRAL_CLS = "B"


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def followup_message(sc, task: dict[str, Any]) -> str:
    """Neutral, fixed follow-up turn (identical every round and every arm)."""
    return (
        f"Follow-up in the same workspace: the `{task['folder']}/` configuration "
        f"may have changed since your last run. Run `{task['runner']}` again with "
        f"run_command before answering, and report whether the check passed.\n\n"
        + sc.PLAN_FORMAT_BLOCK
    )


def make_round_fn(sc, p8k, loaded, dirs, *, task_id: str = NEUTRAL_TASK, cls: str = NEUTRAL_CLS):
    from scripts.sync_eq import extract_plan, score_plan
    from scripts.sync_h_decision import make_steer_hook

    task = sc.task_by_id(task_id)
    follow = followup_message(sc, task)

    def round_fn(messages, action: sar.Action | None, target, seed: int):
        sc.ensure_sandbox()  # identical fixtures for every arm / round
        if messages is None:
            msgs_init = None
            n0 = 2  # system + task turn built inside run_tools_phase
        else:
            msgs_init = list(messages) + [{"role": "user", "content": follow}]
            n0 = len(msgs_init)

        # --- hooks: same sites as run_phase8k._run_episode ---
        h_hook = c_hook = o_hook = tool_c_hook = None
        c_stem = None
        report_prefill = None
        if action is not None:
            signs, gains = action.signs(target), action.gains()
            if "H" in signs:
                h_hook = make_steer_hook(loaded, dirs["H"], gains["H"] * signs["H"])
            if "C" in signs:
                c_hook = make_steer_hook(loaded, dirs["C"], gains["C"] * signs["C"])
                tool_c_hook = make_steer_hook(loaded, dirs["C"], gains["C"] * signs["C"])
                c_stem = p8k.C_STEM
            if "O" in signs:
                o_hook = make_steer_hook(loaded, dirs["O"], gains["O"] * signs["O"])
                if action.force_final_prefix:
                    report_prefill = "FINAL: "

        def hook_for_turn(phase: str, turn: int):
            if phase == "tool" and turn > 0:
                return tool_c_hook
            if phase == "report":
                return o_hook
            return None

        try:
            state = sc.run_tools_phase(
                loaded,
                cls=cls,
                task_id=task_id,
                seed=seed,
                hook_for_turn=hook_for_turn,
                with_plan_format=True,
                h_decision_hook=h_hook,
                c_decision_hook=c_hook,
                c_stem_prefill=c_stem,
                messages_init=msgs_init,
            )
            out = sc.finalize_episode(
                loaded, state, hook_for_turn=hook_for_turn, report_prefill=report_prefill
            )
        finally:
            for h in (h_hook, c_hook, o_hook, tool_c_hook):
                if h is not None:
                    try:
                        h.remove()
                    except Exception:
                        pass

        final = out.get("final_full") or out.get("final") or ""
        round_asst = [
            str(m.get("content") or "")
            for m in state["messages"][n0:]
            if m.get("role") == "assistant"
        ]
        # Same scoring as run_phase8k._S_from_row, restricted to THIS round's text
        H = int(out.get("s_tool") or 0)
        O = int(out.get("s_output") or 0)
        plan = extract_plan(final + "\n" + "\n\n".join(round_asst))
        c = score_plan(plan, H) if plan else None
        C = int(c) if c is not None else 0
        new_messages = state["messages"] + [{"role": "assistant", "content": final}]
        info = {
            "plan": plan[:300],
            "final": final[:300],
            "tools": out.get("tools"),
            "sensitive_paths": out.get("sensitive_paths"),
            "n_messages": len(new_messages),
            "round_turns": len(round_asst),
        }
        return new_messages, [C, H, O], info

    return round_fn


def summarize(trials_by_arm: dict[str, list[dict]], targets) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, trials in trials_by_arm.items():
        per = {
            sar.key(m): sar.summarize_trials([t for t in trials if t["target"] == list(m)])
            for m in targets
        }
        reasons = Counter(r["reason"] for t in trials for r in t["rounds"] if r["action"])
        actions = Counter(r["action"] for t in trials for r in t["rounds"] if r["action"])
        out[arm] = {
            "overall": sar.summarize_trials(trials),
            "per_mstar": per,
            "n_mstar_hold": sum(1 for st in per.values() if (st["P_hold"] or 0) > 0),
            "action_counts": dict(actions),
            "reason_counts": dict(reasons),
        }
    return out


def paired(trials_by_arm: dict[str, list[dict]], ref: str, other: str) -> dict[str, Any]:
    a = {t["trial"]: t for t in trials_by_arm.get(ref, [])}
    b = {t["trial"]: t for t in trials_by_arm.get(other, [])}
    ids = sorted(set(a) & set(b))
    d = [int(a[i]["hold"]) - int(b[i]["hold"]) for i in ids]
    return {
        "ref": ref,
        "other": other,
        "n_pairs": len(ids),
        "mean_d_hold": (sum(d) / len(d)) if d else None,
        "n_ref_only": sum(1 for x in d if x > 0),
        "n_other_only": sum(1 for x in d if x < 0),
    }


def fmt(x, nd=2):
    return "—" if x is None else f"{float(x):.{nd}f}"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s, arms, targets = payload["summaries"], payload["arms"], payload["targets"]
    R = payload["rounds"]
    hold_def = " = ".join(f"S_{k}" for k in range(1, R + 1)) + " = m^*"
    L = [
        "# Adaptive retention controller — same-conversation retention",
        "",
        f"> One conversation per trial: round 0 acquisition (`{payload['acquire']}`, shared/forked), "
        f"then {R} follow-up rounds in the same message history. Paired seeds across arms. "
        "Frozen 8K actuators; target-sign; no new $v$/gains.",
        "",
        f"reps={payload['reps']}, rounds={R}, seed={payload['seed']}, targets={len(targets)}.",
        "",
        f"Primary: $P_{{hold}} = P({hold_def})$.",
        "",
        "## Overall",
        "",
        "| arm | $P_{hold}$ | $P_{hold}\\mid S_0{=}m^*$ | recovery | collateral (steered) | "
        "collateral (unsteered) | steered rounds / trial | $m^*$ with $P_{hold}>0$ |",
        "|-----|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        o = s[arm]["overall"]
        L.append(
            f"| `{arm}` | **{fmt(o['P_hold'])}** ({o['n_hold']}/{o['n']}) | "
            f"{fmt(o['P_hold_given_S0'])} (n={o['n_S0']}) | "
            f"{fmt(o['recovery_rate'])} (n={o['n_drift_rounds']}) | "
            f"{fmt(o['collateral_rate_steered'])} | {fmt(o['collateral_rate_unsteered'])} | "
            f"{fmt(o['mean_steered_rounds'])} | {s[arm]['n_mstar_hold']}/{len(targets)} |"
        )
    L += ["", "## $P_{hold}$ per target", "", "| $m^*$ | $P(S_0{=}m^*)$ | " + " | ".join(f"`{a}`" for a in arms) + " |",
          "|---|---:|" + "---:|" * len(arms)]
    for m in targets:
        mk = sar.key(m)
        p0 = s[arms[0]]["per_mstar"][mk]["P_S0"]
        L.append(f"| `{mk}` | {fmt(p0)} | " + " | ".join(fmt(s[a]["per_mstar"][mk]["P_hold"]) for a in arms) + " |")
    L += ["", "## Paired $\\Delta P_{hold}$", "", "| ref vs other | n | mean Δ | ref-only | other-only |", "|---|---:|---:|---:|---:|"]
    for p in payload["paired"]:
        L.append(f"| `{p['ref']}` vs `{p['other']}` | {p['n_pairs']} | {fmt(p['mean_d_hold'])} | {p['n_ref_only']} | {p['n_other_only']} |")
    if "adaptive" in s:
        L += ["", "## Adaptive decisions", "",
              f"- actions: `{s['adaptive']['action_counts']}`",
              f"- reasons: `{s['adaptive']['reason_counts']}`",
              "", "Final success tables (target · drift → action: recovered/n, collateral/n):", ""]
        for row in payload["adaptive_tables"]:
            cells = ", ".join(f"{a} {v['n_recovered']}/{v['n']} (coll {v['n_collateral']})" for a, v in row["table"].items())
            L.append(f"- `{row['target']}` · drift {row['drift']}: {cells}")
    L += ["", "Exploratory; do not overclaim at low reps.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def merge(args) -> int:
    """Combine per-target split runs (adaptive histories are per target, so this is exact)."""
    parts = [json.loads((RESULTS / f"sync_adaptive_retention_{t}.json").read_text()) for t in args.merge.split(",")]
    for p in parts:
        assert p["final"], "part not finished"
    arms = parts[0]["arms"]
    targets = [tuple(m) for p in parts for m in p["targets"]]
    trials_by_arm = {a: [t for p in parts for t in p["trials"][a]] for a in arms}
    histories = {a: sar.InterventionHistory() for a in arms}
    for p in parts:
        for a in arms:
            for r in p["histories"][a]:
                histories[a].records.append(sar.InterventionRecord(**{
                    **r, "drift": tuple(r["drift"]), "channels": tuple(r["channels"]),
                    "collateral": tuple(r["collateral"])}))
    payload = {
        **{k: parts[0][k] for k in ("protocol", "seed", "reps", "rounds", "acquire", "arms", "gains")},
        "final": True,
        "merged_from": args.merge.split(","),
        "targets": [list(m) for m in targets],
        "summaries": summarize(trials_by_arm, targets),
        "paired": [paired(trials_by_arm, "adaptive", o) for o in arms if o != "adaptive" and "adaptive" in arms],
        "adaptive_tables": [
            {"target": sar.key(m), "drift": list(d), "table": histories["adaptive"].table(m, d)}
            for m in targets
            for d in sorted({r.drift for r in histories["adaptive"].records if r.target == sar.key(m)})
        ] if "adaptive" in histories else [],
        "round0": [r for p in parts for r in p["round0"]],
        "trials": trials_by_arm,
        "histories": {a: [r.to_dict() for r in h.records] for a, h in histories.items()},
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out_json = RESULTS / f"sync_adaptive_retention{suffix}.json"
    out_md = RESULTS / f"sync_adaptive_retention{suffix}.md"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(out_md, payload)
    print(json.dumps({"out": str(out_json), "md": str(out_md),
                      "P_hold": {a: payload["summaries"][a]["overall"]["P_hold"] for a in arms}}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2, help="retention rounds after round 0 (P_hold horizon)")
    ap.add_argument("--targets", default="all", help="comma list like 011,100 or 'all'")
    ap.add_argument("--arms", default=",".join(sar.ARMS))
    ap.add_argument("--acquire", choices=("full8k", "none"), default="full8k",
                    help="round-0 steering shared by all arms")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--sandbox", default="", help="tool sandbox dir (use a separate one per parallel process)")
    ap.add_argument("--merge", default="", help="comma list of tags to merge into one result (no model run)")
    args = ap.parse_args()
    if args.merge:
        return merge(args)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        assert a in sar.ARMS, a
    targets = ALL_MSTAR if args.targets == "all" else [
        tuple(int(c) for c in t.strip()) for t in args.targets.split(",")
    ]
    suffix = f"_{args.tag}" if args.tag else ""
    out_json = RESULTS / f"sync_adaptive_retention{suffix}.json"
    out_md = RESULTS / f"sync_adaptive_retention{suffix}.md"

    from activation_pipeline.device import LOCAL_MODEL_KEY, assert_model_fits_machine, resolve_device_map
    from activation_pipeline.loader import load_model_and_tokenizer

    p8k = _load_mod("phase8k", ROOT / "scripts" / "run_phase8k_c_gain_compose.py")
    assert p8k._alphas() == sar.GAINS, (p8k._alphas(), sar.GAINS)
    sc = p8k._load_sc()
    if args.sandbox:
        sc.WORKSPACE = Path(args.sandbox).resolve()
    sc.ensure_sandbox()
    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY, device_map=resolve_device_map(None), dtype="float32", local_files_only=True
    )
    dirs = p8k._load_vc()
    round_fn = make_round_fn(sc, p8k, loaded, dirs)
    acquire_action = sar.CONTINUOUS if args.acquire == "full8k" else None

    histories = {arm: sar.InterventionHistory() for arm in arms}
    trials_by_arm: dict[str, list[dict]] = {arm: [] for arm in arms}
    round0: list[dict] = []
    n_tot = len(targets) * args.reps
    idx = 0
    print(f"=== adaptive retention arms={arms} reps={args.reps} rounds={args.rounds} acquire={args.acquire} ===", flush=True)

    def dump(final: bool) -> dict[str, Any]:
        payload = {
            "protocol": "same-conversation retention; adaptive vs event_triggered vs continuous vs none",
            "final": final,
            "seed": args.seed,
            "reps": args.reps,
            "rounds": args.rounds,
            "acquire": args.acquire,
            "arms": arms,
            "targets": [list(m) for m in targets],
            "gains": sar.GAINS,
            "summaries": summarize(trials_by_arm, targets),
            "paired": [paired(trials_by_arm, "adaptive", o) for o in arms if o != "adaptive" and "adaptive" in arms],
            "adaptive_tables": [
                {"target": sar.key(m), "drift": list(d), "table": histories["adaptive"].table(m, d)}
                for m in targets
                for d in sorted({r.drift for r in histories["adaptive"].records if r.target == sar.key(m)})
            ] if "adaptive" in histories else [],
            "round0": round0,
            "trials": trials_by_arm,
            "histories": {a: [r.to_dict() for r in h.records] for a, h in histories.items()},
        }
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    for mi, mstar in enumerate(targets):
        for r in range(args.reps):
            idx += 1
            trial_id = f"{sar.key(mstar)}#{r}"
            # seed keyed on the target itself so per-target split runs reproduce exactly
            seed = args.seed + 1000 * ALL_MSTAR.index(tuple(mstar)) + 37 * r
            seeds = [seed + 7919 * k for k in range(args.rounds + 1)]
            msgs0, S0, info0 = round_fn(None, acquire_action, list(mstar), seeds[0])
            round0.append({"trial": trial_id, "target": list(mstar), "S0": S0, "info": info0})
            print(f"  [{idx}/{n_tot}] m*={sar.key(mstar)} r={r} S0={sar.key(S0)}", flush=True)
            for arm in arms:
                tr = sar.run_retention(
                    arm, round_fn=round_fn, messages=msgs0, S0=S0, target=mstar,
                    seeds=seeds[1:], history=histories[arm], trial=trial_id,
                )
                trials_by_arm[arm].append(tr)
                acts = [x["action"] or "-" for x in tr["rounds"]]
                print(f"    {arm:16s} path={'→'.join(tr['path'])} acts={acts} hold={int(tr['hold'])}", flush=True)
            dump(final=False)

    payload = dump(final=True)
    write_md(out_md, payload)
    print(json.dumps({"out": str(out_json), "md": str(out_md),
                      "P_hold": {a: payload["summaries"][a]["overall"]["P_hold"] for a in arms}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
