#!/usr/bin/env python3
"""Unit tests for the adaptive retention controller (no model)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sync_adaptive_retention as sar  # noqa: E402

T = [1, 0, 1]


def test_rules() -> None:
    assert sar.identify_drift([1, 0, 1], T) == ()
    assert sar.identify_drift([0, 0, 1], T) == ("C",)
    assert sar.identify_drift([0, 1, 0], T) == ("C", "H", "O")
    assert sar.rule_candidates(("C",))[0] == "CH"  # C/H coupling
    assert sar.rule_candidates(("O",))[0] == "O"
    assert sar.rule_candidates(("H",))[0] == "H"
    assert sar.rule_candidates(("C", "O"))[0] == "FULL"
    assert sar.rule_candidates(("C", "H", "O")) == ["FULL"]
    # No action ever steers C without H
    for a in sar.CATALOG.values():
        assert "C" not in a.channels or "H" in a.channels, a
    # O correction does not force FINAL:, existing continuous does
    assert not sar.CATALOG["O"].force_final_prefix
    assert not sar.CATALOG["FULL"].force_final_prefix
    assert sar.CONTINUOUS.force_final_prefix
    # Signs/gains come only from target + frozen gains
    assert sar.CATALOG["FULL"].signs([1, 0, 1]) == {"C": 1, "H": -1, "O": 1}
    assert sar.CATALOG["CH"].gains() == {"C": 5.0, "H": 1.5}
    assert sar.collateral([1, 0, 0], [1, 1, 1], T) == ("H",)


def test_decide_arms() -> None:
    h = sar.InterventionHistory()
    a, _ = sar.decide("none", observed=[0, 0, 0], target=T, history=h)
    assert a is None
    a, _ = sar.decide("continuous", observed=T, target=T, history=h)
    assert a is sar.CONTINUOUS
    for arm in ("event_triggered", "adaptive"):
        a, m = sar.decide(arm, observed=T, target=T, history=h)
        assert a is None and m["reason"] == "target_intact"
    a, _ = sar.decide("event_triggered", observed=[0, 0, 1], target=T, history=h)
    assert a.name == "FULL"
    a, m = sar.decide("adaptive", observed=[0, 0, 1], target=T, history=h)
    assert a.name == "CH" and m["reason"] == "rule_default"
    a, m = sar.decide("adaptive", observed=[1, 0, 0], target=T, history=h)
    assert a.name == "O"


def test_learning() -> None:
    h = sar.InterventionHistory()
    drift = ("C",)
    # Default CH fails with collateral → next time explore alternative FULL
    h.record(target=T, drift=drift, action=sar.CATALOG["CH"], before=[0, 0, 1], after=[0, 1, 1])
    a, m = sar.choose_best_action(drift=drift, target=T, history=h)
    assert a.name == "FULL" and m["reason"] == "explore_alternative", m
    # FULL recovers → preferred empirically
    h.record(target=T, drift=drift, action=sar.CATALOG["FULL"], before=[0, 0, 1], after=T)
    a, m = sar.choose_best_action(drift=drift, target=T, history=h)
    assert a.name == "FULL" and m["reason"] == "empirical"
    # Tables are per target: another target still starts at the rule default
    a, m = sar.choose_best_action(drift=drift, target=[1, 1, 1], history=h)
    assert a.name == "CH" and m["reason"] == "rule_default"
    # All candidates tried, none recovered → conservative FULL fallback
    h2 = sar.InterventionHistory()
    for name in ("O", "FULL"):
        h2.record(target=T, drift=("O",), action=sar.CATALOG[name], before=[1, 0, 0], after=[1, 0, 0])
    a, m = sar.choose_best_action(drift=("O",), target=T, history=h2)
    assert a.name == "FULL" and m["reason"] == "fallback_full"
    # Avoid: action that just caused new drift is skipped when alternatives exist
    a, m = sar.choose_best_action(drift=("H",), target=T, history=sar.InterventionHistory(), avoid="H")
    assert a.name == "CH"
    rec = h.records[0].to_dict()
    for k in ("before", "after", "channels", "signs", "gains", "recovered", "collateral"):
        assert k in rec


def test_run_retention_same_conversation() -> None:
    """Fake model: round_fn must receive the growing conversation each round."""
    script = {
        # (incoming state after action) deterministic toy dynamics:
        None: lambda S: [S[0], S[1], 0],  # unsteered: O decays
    }

    def round_fn(messages, action, target, seed):
        assert messages is not None and messages[0] == "root"
        S = messages[-1]
        if action is None:
            S2 = script[None](S)
        else:
            S2 = list(S)
            for ch, s in action.signs(target).items():
                S2[sar.CH_IDX[ch]] = int(s > 0)
        return messages + [S2], S2, {"len": len(messages)}

    h = sar.InterventionHistory()
    tr = sar.run_retention("adaptive", round_fn=round_fn, messages=["root", T], S0=T,
                           target=T, seeds=[1, 2, 3], history=h, trial="x")
    # r1: intact → no steer → O decays; r2: drift O → steer O → recovered; r3: intact → no steer → decays
    assert [r["action"] for r in tr["rounds"]] == [None, "O", None], tr["rounds"]
    assert tr["path"] == ["101", "100", "101", "100"]
    assert [r["info"]["len"] for r in tr["rounds"]] == [2, 3, 4]  # same conversation grows
    assert not tr["hold"]
    assert len(h.records) == 1 and h.records[0].recovered

    trn = sar.run_retention("none", round_fn=round_fn, messages=["root", T], S0=T,
                            target=T, seeds=[1, 2], history=sar.InterventionHistory())
    trc = sar.run_retention("continuous", round_fn=round_fn, messages=["root", T], S0=T,
                            target=T, seeds=[1, 2], history=sar.InterventionHistory())
    assert trc["hold"] and not trn["hold"]

    s = sar.summarize_trials([trn, trc, tr])
    assert s["n"] == 3 and s["n_hold"] == 1
    assert abs(s["recovery_rate"] - 1 / 2) < 1e-9, s  # none: 100→100 fail; adaptive: 100→101 ok
    assert s["collateral_rate_steered"] == 0.0


def main() -> int:
    test_rules()
    test_decide_arms()
    test_learning()
    test_run_retention_same_conversation()
    print("OK test_sync_adaptive_retention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
