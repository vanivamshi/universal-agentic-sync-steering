"""Adaptive retention controller — pure decision logic (no model, no torch).

Principle: observe first, intervene only when necessary, learn from outcomes.

Per retention round k (same conversation throughout; never a fresh episode):

  1. observe S_{k-1}
  2. S_{k-1} == m*  → run round k unsteered
  3. else           → drift = channels where S_{k-1} != m*
                      action = choose_best_action(drift, m*, history)
                      run round k under that action
  4. record (before=S_{k-1}, action, after=S_k) in the history

Actions are *compositions of the frozen 8K actuators only* — target-sign
polarity and locked gains (α_C=5 stem, α_H=1.5 decision-token, α_O=1.5
report). The controller never invents signs or gains; it only chooses which
calibrated channels to steer. C is never steered without H (C/H coupling).

Arms:
  none            never steer
  continuous      steer C,H,O every round (existing 8K config, forced FINAL:)
  event_triggered steer C,H,O (no forced FINAL:) only after observed drift
  adaptive        rule table + empirical success table, FULL as safe fallback
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

# Locked Phase 8K gains (runner asserts these equal run_phase8k._alphas()).
GAINS = {"C": 5.0, "H": 1.5, "O": 1.5}

ARMS = ("none", "continuous", "event_triggered", "adaptive")

# Untried candidates are scored at this optimistic prior so the rule default is
# tried first, and an alternative is tried as soon as the default has failed.
UNTRIED_PRIOR = 0.5


@dataclass(frozen=True)
class Action:
    name: str
    channels: tuple[str, ...]
    force_final_prefix: bool = False
    # Per-channel multiplier on the frozen 8K gains (C, H, O). Default = locked 8K.
    gain_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def signs(self, target: Sequence[int]) -> dict[str, int]:
        """Target-sign polarity (s* = 2·m*_k − 1), as frozen in 8K."""
        return {ch: 2 * int(target[CH_IDX[ch]]) - 1 for ch in self.channels}

    def gains(self) -> dict[str, float]:
        return {ch: GAINS[ch] * float(self.gain_scale[CH_IDX[ch]]) for ch in self.channels}


def scaled_continuous(scale: Sequence[float], name: str | None = None) -> Action:
    """Continuous 8K steering (forced FINAL:) with per-channel gain multipliers.

    A channel with scale 0 is dropped (not steered at all).
    """
    sc = tuple(float(x) for x in scale)
    assert len(sc) == 3 and all(x >= 0 for x in sc), sc
    chans = tuple(ch for ch in CHANNELS if sc[CH_IDX[ch]] > 0)
    nm = name or "CONT[" + ",".join(f"{ch}x{sc[CH_IDX[ch]]:g}" for ch in CHANNELS) + "]"
    return Action(nm, chans, force_final_prefix=True, gain_scale=sc)


# Calibrated compositions. No C-without-H action exists (C/H coupling rule).
CATALOG: dict[str, Action] = {
    "H": Action("H", ("H",)),
    "O": Action("O", ("O",)),
    "CH": Action("CH", ("C", "H")),
    "HO": Action("HO", ("H", "O")),
    "FULL": Action("FULL", ("C", "H", "O")),
}
FALLBACK = "FULL"
# Existing continuous steering = the 8K configuration incl. forced "FINAL: ".
CONTINUOUS = Action("FULL_8K", ("C", "H", "O"), force_final_prefix=True)


def key(s: Iterable[int]) -> str:
    return "".join(str(int(x)) for x in s)


def identify_drift(observed: Sequence[int], target: Sequence[int]) -> tuple[str, ...]:
    """Channels whose observed bit differs from the target, in C,H,O order."""
    return tuple(ch for ch in CHANNELS if int(observed[CH_IDX[ch]]) != int(target[CH_IDX[ch]]))


def collateral(before: Sequence[int], after: Sequence[int], target: Sequence[int]) -> tuple[str, ...]:
    """Channels that were correct before and are wrong after (unintended change)."""
    return tuple(
        ch
        for ch in CHANNELS
        if int(before[CH_IDX[ch]]) == int(target[CH_IDX[ch]])
        and int(after[CH_IDX[ch]]) != int(target[CH_IDX[ch]])
    )


def rule_candidates(drift: Sequence[str]) -> list[str]:
    """Decision table → ordered candidate actions (first = rule default).

    | drift      | default | alternatives |
    |------------|---------|--------------|
    | C          | CH      | FULL         |   C drifts → steer C and H together
    | H          | H       | CH, FULL     |
    | O          | O       | FULL         |   O without forced FINAL:
    | C,H        | FULL    | CH           |   multiple → full-channel
    | H,O        | FULL    | HO           |
    | C,O / CHO  | FULL    | —            |
    """
    d = frozenset(drift)
    if not d:
        return []
    table: dict[frozenset, list[str]] = {
        frozenset({"C"}): ["CH", "FULL"],
        frozenset({"H"}): ["H", "CH", "FULL"],
        frozenset({"O"}): ["O", "FULL"],
        frozenset({"C", "H"}): ["FULL", "CH"],
        frozenset({"H", "O"}): ["FULL", "HO"],
    }
    return list(table.get(d, ["FULL"]))


@dataclass
class InterventionRecord:
    target: str
    drift: tuple[str, ...]
    action: str
    channels: tuple[str, ...]
    signs: dict[str, int]
    gains: dict[str, float]
    force_final_prefix: bool
    before: list[int]
    after: list[int]
    recovered: bool
    collateral: tuple[str, ...]
    trial: Any = None
    round: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["drift"] = list(self.drift)
        d["channels"] = list(self.channels)
        d["collateral"] = list(self.collateral)
        return d


@dataclass
class InterventionHistory:
    """Per-arm log of every corrective intervention, keyed by target."""

    records: list[InterventionRecord] = field(default_factory=list)

    def record(
        self,
        *,
        target: Sequence[int],
        drift: Sequence[str],
        action: Action,
        before: Sequence[int],
        after: Sequence[int],
        trial: Any = None,
        round: int | None = None,
        reason: str = "",
    ) -> InterventionRecord:
        rec = InterventionRecord(
            target=key(target),
            drift=tuple(drift),
            action=action.name,
            channels=tuple(action.channels),
            signs=action.signs(target),
            gains=action.gains(),
            force_final_prefix=action.force_final_prefix,
            before=[int(x) for x in before],
            after=[int(x) for x in after],
            recovered=list(map(int, after)) == list(map(int, target)),
            collateral=collateral(before, after, target),
        )
        rec.trial, rec.round, rec.reason = trial, round, reason
        self.records.append(rec)
        return rec

    def table(self, target: Sequence[int], drift: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Empirical success table for (target, drift pattern): per action n / recovery / collateral."""
        tk, dk = key(target), tuple(drift)
        out: dict[str, dict[str, Any]] = {}
        for r in self.records:
            if r.target != tk or r.drift != dk:
                continue
            st = out.setdefault(r.action, {"n": 0, "n_recovered": 0, "n_collateral": 0})
            st["n"] += 1
            st["n_recovered"] += int(r.recovered)
            st["n_collateral"] += int(bool(r.collateral))
        for st in out.values():
            st["p_recovered"] = st["n_recovered"] / st["n"]
            st["p_collateral"] = st["n_collateral"] / st["n"]
        return out


def choose_best_action(
    *,
    drift: Sequence[str],
    target: Sequence[int],
    history: InterventionHistory,
    avoid: str | None = None,
) -> tuple[Action, dict[str, Any]]:
    """Pick among rule candidates using the empirical success table.

    score(a) = P(recovered | target, drift, a) if tried, else UNTRIED_PRIOR.
    Ties → lower collateral rate, then rule order. ``avoid`` (the action that
    just caused new drift in this conversation) is excluded when an
    alternative exists. If every candidate has been tried and none ever
    recovered, fall back to FULL.
    """
    cands = rule_candidates(drift)
    tbl = history.table(target, drift)
    pool = [c for c in cands if c != avoid] or cands
    scored = []
    for i, c in enumerate(pool):
        st = tbl.get(c)
        if st is None:
            scored.append((UNTRIED_PRIOR, 0.0, -i, c, "untried"))
        else:
            scored.append((st["p_recovered"], -st["p_collateral"], -i, c, "empirical"))
    scored.sort(reverse=True)
    best_score, _, _, best, src = scored[0]
    meta: dict[str, Any] = {
        "candidates": cands,
        "avoid": avoid,
        "table": tbl,
        "scores": {c: s for s, _, _, c, _ in scored},
    }
    if best_score <= 0.0:
        meta["reason"] = "fallback_full"
        return CATALOG[FALLBACK], meta
    if src == "untried":
        meta["reason"] = "rule_default" if best == cands[0] else "explore_alternative"
    else:
        meta["reason"] = "empirical"
    return CATALOG[best], meta


def decide(
    arm: str,
    *,
    observed: Sequence[int],
    target: Sequence[int],
    history: InterventionHistory,
    last_record: InterventionRecord | None = None,
) -> tuple[Action | None, dict[str, Any]]:
    """Return the action for the next round of the same conversation (None = no steer)."""
    intact = list(map(int, observed)) == list(map(int, target))
    drift = identify_drift(observed, target)
    if arm == "none":
        return None, {"reason": "never_steer", "drift": list(drift)}
    if arm == "continuous":
        return CONTINUOUS, {"reason": "always_steer", "drift": list(drift)}
    if intact:
        return None, {"reason": "target_intact", "drift": []}
    if arm == "event_triggered":
        return CATALOG[FALLBACK], {"reason": "drift_full_correction", "drift": list(drift)}
    if arm == "adaptive":
        avoid = None
        if last_record is not None and last_record.collateral:
            avoid = last_record.action  # previous correction caused new drift
        action, meta = choose_best_action(drift=drift, target=target, history=history, avoid=avoid)
        meta["drift"] = list(drift)
        return action, meta
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------
# Metrics (pure; trials are dicts produced by the runner)
# --------------------------------------------------------------------------


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Trial: {target, states:[S0..SR], rounds:[{before, after, action|None}]}.

    P_hold        = P(S_1 = … = S_R = m*)       (primary; R=2 → P(S1=S2=m*))
    P_hold|S0     = same, conditioned on S_0 = m*
    recovery      = P(S_k = m* | S_{k-1} != m*)
    collateral    = flips correct→wrong / correct channels at k−1, split by
                    steered vs unsteered rounds
    """
    n = len(trials)
    hold = [all(list(s) == list(t["target"]) for s in t["states"][1:]) for t in trials]
    s0 = [list(t["states"][0]) == list(t["target"]) for t in trials]
    hold_s0 = [h for h, a in zip(hold, s0) if a]
    n_drift = n_rec = 0
    coll = {"steered": [0, 0], "unsteered": [0, 0]}
    n_steer_rounds = n_steer_ch = 0
    for t in trials:
        tgt = list(t["target"])
        for r in t["rounds"]:
            b, a = list(r["before"]), list(r["after"])
            if b != tgt:
                n_drift += 1
                n_rec += int(a == tgt)
            steered = bool(r.get("action"))
            bucket = coll["steered" if steered else "unsteered"]
            bucket[0] += len(collateral(b, a, tgt))
            bucket[1] += sum(int(b[i]) == tgt[i] for i in range(3))
            if steered:
                n_steer_rounds += 1
                n_steer_ch += len(r.get("channels") or [])

    def rate(k: int, m: int) -> float | None:
        return k / m if m else None

    return {
        "n": n,
        "P_hold": rate(sum(hold), n),
        "n_hold": sum(hold),
        "P_S0": rate(sum(s0), n),
        "P_hold_given_S0": rate(sum(hold_s0), len(hold_s0)),
        "n_S0": len(hold_s0),
        "recovery_rate": rate(n_rec, n_drift),
        "n_drift_rounds": n_drift,
        "collateral_rate_steered": rate(*coll["steered"]),
        "collateral_rate_unsteered": rate(*coll["unsteered"]),
        "mean_steered_rounds": rate(n_steer_rounds, n),
        "mean_steered_channel_rounds": rate(n_steer_ch, n),
    }


# --------------------------------------------------------------------------
# Same-conversation retention loop
# --------------------------------------------------------------------------


def run_retention(
    arm: str,
    *,
    round_fn,
    messages: Any,
    S0: Sequence[int],
    target: Sequence[int],
    seeds: Sequence[int],
    history: InterventionHistory,
    trial: Any = None,
) -> dict[str, Any]:
    """Continue ONE conversation for len(seeds) retention rounds.

    ``round_fn(messages, action|None, target, seed) -> (messages', S, info)``
    must extend the given conversation (never start a fresh episode).
    """
    target = [int(x) for x in target]
    S = [int(x) for x in S0]
    states = [list(S)]
    rounds: list[dict[str, Any]] = []
    last: InterventionRecord | None = None
    for k, seed in enumerate(seeds, start=1):
        action, meta = decide(arm, observed=S, target=target, history=history, last_record=last)
        drift = identify_drift(S, target)
        messages, S_new, info = round_fn(messages, action, target, seed)
        S_new = [int(x) for x in S_new]
        last = None
        if action is not None and drift:
            last = history.record(
                target=target,
                drift=drift,
                action=action,
                before=S,
                after=S_new,
                trial=trial,
                round=k,
                reason=str(meta.get("reason", "")),
            )
        rounds.append(
            {
                "k": k,
                "seed": int(seed),
                "before": list(S),
                "after": list(S_new),
                "drift": list(drift),
                "action": action.name if action else None,
                "channels": list(action.channels) if action else [],
                "signs": action.signs(target) if action else {},
                "gains": action.gains() if action else {},
                "force_final_prefix": bool(action.force_final_prefix) if action else False,
                "reason": meta.get("reason"),
                "avoid": meta.get("avoid"),
                "recovered": bool(drift) and S_new == target,
                "collateral": list(collateral(S, S_new, target)),
                "info": info,
            }
        )
        S = S_new
        states.append(list(S))
    return {
        "arm": arm,
        "trial": trial,
        "target": target,
        "states": states,
        "path": [key(s) for s in states],
        "rounds": rounds,
        "hold": all(s == target for s in states[1:]),
    }
