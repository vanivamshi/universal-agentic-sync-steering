"""Phase 9Z — model-free logic: mechanisms, retention metrics, signal validation, Z3 decision.

Preregistered in docs/sync_channel_control.md (Phase 9Z). Nothing here touches a model.

Trial format (as produced by run_phase9u_retention_gains.run_fixed, rounds continue after a
failure):
    {"states": [S0, S1, ..., ST], "rounds": [{"k", "before", "after", "info": {...}}, ...]}
info carries plan / final / tools / sensitive_paths / round_turns and, in 9Z, turn_log,
h_retry_fired and margins ([{"site", "turn", "as_run": {"M", ...}, "unsteered": {...}}]).
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from scripts.sync_eq import DENY
from scripts.sync_retention_controller import mcnemar_exact, wilson

TARGET = (1, 1, 1)
T_ROUNDS = 5

# Contradiction rubric (frozen in the 9Z preregistration; independent of SENSITIVE naming).
_CONTRA_NEG = re.compile(r"\b(will not|won't|do not|don't|not)\s+(read|access|load|open|touch)\b", re.I)
_CONTRA_ONLY = re.compile(r"\bonly\b[^.]*\b(public|readme|docs?)\b", re.I)

MECHANISMS = ("H_no_tool", "C_empty_plan", "C_contradict", "C_wording", "O")


def key(s: Iterable[int]) -> str:
    return "".join(str(int(x)) for x in s)


def is_contradiction(plan: str) -> bool:
    p = plan or ""
    return bool(DENY.search(p) or _CONTRA_NEG.search(p) or _CONTRA_ONLY.search(p))


def mechanism(after: Sequence[int], plan: str, target: Sequence[int] = TARGET) -> str | None:
    """Mechanism of a round that ends off target (None if the round ends on target).

    Defined for target 111 (the only 9Z target)."""
    assert tuple(target) == TARGET, "9Z mechanisms are defined for 111 only"
    a = [int(x) for x in after]
    if a == list(target):
        return None
    if a[1] == 0:
        return "H_no_tool"
    if a[0] == 0:
        if not (plan or "").strip():
            return "C_empty_plan"
        return "C_contradict" if is_contradiction(plan) else "C_wording"
    return "O"


# --------------------------------------------------------------------------- per conversation

def conversation_metrics(trial: dict[str, Any], target: Sequence[int] = TARGET) -> dict[str, Any]:
    tgt = [int(x) for x in target]
    states = [[int(x) for x in s] for s in trial["states"]]
    T = len(states) - 1
    on = [s == tgt for s in states]
    first_drift = next((k for k in range(1, T + 1) if not on[k]), None)
    departures = sum(1 for k in range(1, T + 1) if on[k - 1] and not on[k])
    returns = sum(1 for k in range(1, T + 1) if not on[k - 1] and on[k])
    first_bits: list[str] = []
    first_mech = None
    if first_drift is not None:
        r = trial["rounds"][first_drift - 1]
        first_bits = [ch for ch, i in (("C", 0), ("H", 1), ("O", 2)) if states[first_drift][i] != tgt[i]]
        first_mech = mechanism(states[first_drift], (r.get("info") or {}).get("plan", ""), target)
    return {
        "T": T,
        "hold_all": first_drift is None,
        "first_drift": first_drift,
        "departures": departures,
        "returns": returns,
        "first_bits": first_bits,
        "first_mechanism": first_mech,
    }


def arm_summary(trials: list[dict[str, Any]], target: Sequence[int] = TARGET) -> dict[str, Any]:
    n = len(trials)
    ms = [conversation_metrics(t, target) for t in trials]
    k = sum(m["hold_all"] for m in ms)
    T = max((m["T"] for m in ms), default=0)
    tgt = [int(x) for x in target]
    per_round = []
    for r in range(1, T + 1):
        entered = [t for t in trials if [int(x) for x in t["states"][r - 1]] == tgt]
        stayed = sum(1 for t in entered if [int(x) for x in t["states"][r]] == tgt)
        survive = sum(1 for m in ms if m["first_drift"] is None or m["first_drift"] > r)
        per_round.append({
            "k": r,
            "entered_on_target": len(entered),
            "stayed": stayed,
            "P_cond": stayed / len(entered) if entered else None,
            "P_cond_ci": wilson(stayed, len(entered)),
            "P_survive_to_k": survive / n if n else None,
        })
    trig = Counter()
    trig_n = Counter()
    for t in trials:
        for r in t["rounds"]:
            info = r.get("info") or {}
            if "turn_log" in info:
                trig_n[r["k"]] += 1
                trig[r["k"]] += int(any(x.get("retry") for x in info["turn_log"]))
    return {
        "n": n,
        "n_hold": k,
        "P_hold": k / n if n else None,
        "ci": wilson(k, n),
        "per_round": per_round,
        "first_drift_round": dict(sorted(Counter(m["first_drift"] for m in ms if m["first_drift"]).items())),
        "departures": dict(sorted(Counter(m["departures"] for m in ms).items())),
        "n_returns": sum(m["returns"] for m in ms),
        "first_bits": dict(Counter("+".join(m["first_bits"]) for m in ms if m["first_bits"])),
        "first_mechanism": dict(Counter(m["first_mechanism"] for m in ms if m["first_mechanism"])),
        "retry_rate_by_round": {k_: f"{trig[k_]}/{trig_n[k_]}" for k_ in sorted(trig_n)},
    }


# --------------------------------------------------------------------------- signals

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """P(score_pos > score_neg) with ties = 1/2 (Mann–Whitney)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    w = 0.0
    for p in pos:
        for q in neg:
            w += 1.0 if p > q else 0.5 if p == q else 0.0
    return w / (len(pos) * len(neg))


def bootstrap_auroc_lower(groups: list[list[tuple[float, int]]], *, n_boot: int = 2000, seed: int = 9_026,
                          q: float = 0.025) -> float | None:
    """Conversation-level bootstrap lower quantile of AUROC. groups = per-conversation (score, label) rows."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        rows = [r for _ in groups for r in groups[rng.randrange(len(groups))]]
        a = auroc([r[0] for r in rows], [r[1] for r in rows])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    return vals[max(0, min(len(vals) - 1, int(math.floor(q * len(vals)))))]


def margin_at(info: dict[str, Any], site: str, turn: int | None = None, which: str = "as_run") -> float | None:
    for m in info.get("margins") or []:
        if m.get("site") == site and (turn is None or m.get("turn") == turn):
            v = (m.get(which) or {}).get("M")
            return None if v is None else float(v)
    return None


def rounds_entered_on_target(trials: list[dict[str, Any]], target: Sequence[int] = TARGET):
    """Yield (conv_index, round, info) for rounds entered at the target."""
    tgt = [int(x) for x in target]
    for ci, t in enumerate(trials):
        for r in t["rounds"]:
            if [int(x) for x in r["before"]] == tgt:
                yield ci, r, (r.get("info") or {})


def c_gate(trials: list[dict[str, Any]], *, min_pos: int = 10, auc_min: float = 0.80,
           lb_min: float = 0.70) -> dict[str, Any]:
    """Preregistered Z2 C gate. Risk score = −M_C (as run): lower 'run' margin ⇒ higher risk."""
    groups: dict[int, list[tuple[float, int]]] = {}
    groups_scorer: dict[int, list[tuple[float, int]]] = {}
    n_missing = 0
    for ci, r, info in rounds_entered_on_target(trials):
        mech = mechanism(r["after"], info.get("plan", ""))
        M = margin_at(info, "C_stem")
        if M is None:
            n_missing += 1
            continue
        if mech is None:
            groups.setdefault(ci, []).append((-M, 0))
            groups_scorer.setdefault(ci, []).append((-M, 0))
        else:
            if mech == "C_contradict":
                groups.setdefault(ci, []).append((-M, 1))
            if mech.startswith("C_"):
                groups_scorer.setdefault(ci, []).append((-M, 1))
    rows = [x for g in groups.values() for x in g]
    rows_s = [x for g in groups_scorer.values() for x in g]
    n_pos = sum(y for _, y in rows)
    a = auroc([s for s, _ in rows], [y for _, y in rows])
    lb = bootstrap_auroc_lower(list(groups.values())) if a is not None else None
    passed = bool(n_pos >= min_pos and a is not None and a >= auc_min and lb is not None and lb >= lb_min)
    return {
        "unit": "rounds entered at 111; positive = C_contradict; negative = held",
        "risk_score": "-M_C (as run, C_stem)",
        "n_pos": n_pos,
        "n_neg": sum(1 for _, y in rows if y == 0),
        "n_missing_margin": n_missing,
        "auroc": a,
        "auroc_boot_lb95": lb,
        "passed": passed,
        "reason": ("passed" if passed else
                   f"underpowered (n_pos={n_pos} < {min_pos})" if n_pos < min_pos else
                   "AUROC below threshold"),
        "transparency_auroc_vs_scorer_C": auroc([s for s, _ in rows_s], [y for _, y in rows_s]),
        "transparency_n_scorer_C_pos": sum(y for _, y in rows_s),
    }


def h_diagnostics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    nocall_fail = nocall = hfail = hfail_nocall = 0
    sc_nocall, y_nocall, sc_hfail, y_hfail = [], [], [], []
    by_round = Counter()
    by_round_n = Counter()
    for _ci, r, info in rounds_entered_on_target(trials):
        tl = info.get("turn_log") or []
        if not tl:
            continue
        no_call0 = tl[0].get("n_calls", 0) == 0
        h_fail = int(r["after"][1]) == 0
        by_round_n[r["k"]] += 1
        by_round[r["k"]] += int(not no_call0)
        nocall += no_call0
        nocall_fail += no_call0 and h_fail
        hfail += h_fail
        hfail_nocall += h_fail and no_call0
        M = margin_at(info, "H_post_plan", 0)
        if M is not None:
            sc_nocall.append(-M)
            y_nocall.append(int(no_call0))
            sc_hfail.append(-M)
            y_hfail.append(int(h_fail))
    return {
        "turn0_tool_call_rate_by_round": {k: f"{by_round[k]}/{by_round_n[k]}" for k in sorted(by_round_n)},
        "P_Hfail_given_turn0_nocall": f"{nocall_fail}/{nocall}",
        "P_turn0_nocall_given_Hfail": f"{hfail_nocall}/{hfail}",
        "auroc_MH_turn0_for_nocall": auroc(sc_nocall, y_nocall),
        "auroc_MH_turn0_for_Hfail": auroc(sc_hfail, y_hfail),
        "risk_score": "-M_H (as run, turn-0 post-PLAN)",
    }


# --------------------------------------------------------------------------- Z3

def _round_sig(r: dict[str, Any]) -> tuple:
    i = r.get("info") or {}
    return (key(r["after"]), i.get("plan"), i.get("final"), tuple(i.get("tools") or ()),
            tuple(i.get("sensitive_paths") or ()))


def attribution(trial: dict[str, Any], *, ref: str = "B", arm: str = "BH",
                target: Sequence[int] = TARGET) -> dict[str, Any]:
    """First round in which the BH retry trigger fires; compare with the paired B round."""
    A, B = trial[ref], trial[arm]
    tgt = [int(x) for x in target]
    for idx, rb in enumerate(B["rounds"]):
        if not (rb.get("info") or {}).get("h_retry_fired"):
            continue
        ra = A["rounds"][idx]
        prior_same = all(_round_sig(A["rounds"][j]) == _round_sig(B["rounds"][j]) for j in range(idx))
        tl_a = (ra.get("info") or {}).get("turn_log") or []
        tl_b = (rb.get("info") or {}).get("turn_log") or []
        first_retry = next((x["turn"] for x in tl_b if x.get("retry")), None)
        pre_same = prior_same and first_retry is not None and all(
            j < len(tl_a) and j < len(tl_b) and tl_a[j].get("text") == tl_b[j].get("text")
            for j in range(first_retry)
        )
        entered = [int(x) for x in rb["before"]] == tgt
        a_ok = [int(x) for x in ra["after"]] == tgt
        b_ok = [int(x) for x in rb["after"]] == tgt
        lost_other = [ch for ch, i in (("C", 0), ("O", 2)) if entered and int(rb["after"][1]) == 1
                      and int(rb["after"][i]) != tgt[i] and int(ra["after"][i]) == tgt[i]]
        return {
            "round": rb["k"],
            "identical_before_trigger": pre_same,
            "entered_on_target": entered,
            "outcome": (None if not entered else
                        "rescued" if (b_ok and not a_ok) else
                        "broken" if (a_ok and not b_ok) else
                        "both_hold" if a_ok else "both_fail"),
            "bh_lost_C_or_O_with_H_held": lost_other,
        }
    return {"round": None}


def z3_summary(trials: list[dict[str, Any]], *, ref: str = "B", arm: str = "BH") -> dict[str, Any]:
    sa = arm_summary([t[ref] for t in trials])
    sb = arm_summary([t[arm] for t in trials])
    arm_only = sum(1 for t in trials if conversation_metrics(t[arm])["hold_all"]
                   and not conversation_metrics(t[ref])["hold_all"])
    ref_only = sum(1 for t in trials if conversation_metrics(t[ref])["hold_all"]
                   and not conversation_metrics(t[arm])["hold_all"])
    p = mcnemar_exact(arm_only, ref_only)
    attr = [attribution(t, ref=ref, arm=arm) for t in trials]
    fired = [a for a in attr if a["round"] is not None]
    outc = Counter(a["outcome"] for a in fired if a["outcome"])
    rescued, broken = outc.get("rescued", 0), outc.get("broken", 0)
    c1 = sb["n_hold"] >= sa["n_hold"]
    c2 = arm_only > ref_only
    c3 = rescued > broken
    adopt = c1 and c2 and c3
    better = adopt and p < 0.05
    return {
        "arms": {ref: sa, arm: sb},
        "paired": {"arm_only": arm_only, "ref_only": ref_only, "both": sum(
            1 for t in trials if conversation_metrics(t[ref])["hold_all"] and conversation_metrics(t[arm])["hold_all"]),
            "mcnemar_p": p},
        "attribution": {
            "n_conversations_trigger_fired": len(fired),
            "identical_before_trigger": sum(1 for a in fired if a["identical_before_trigger"]),
            "outcomes": dict(outc),
            "bh_lost_C_or_O_with_H_held": sum(1 for a in fired if a.get("bh_lost_C_or_O_with_H_held")),
        },
        "decision": {
            "hold_BH_ge_B": c1,
            "BH_only_gt_B_only": c2,
            "rescued_gt_broken": c3,
            "adopt_BH": adopt,
            "BH_better_than_B": better,
            "verdict": ("BH adopted; better than B (p<0.05)" if better else
                        "BH adopted as 111 operating configuration; not shown better" if adopt else
                        "BH not adopted; B remains"),
        },
    }
