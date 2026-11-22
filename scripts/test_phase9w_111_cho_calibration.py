#!/usr/bin/env python3
"""Model-free tests for the Phase 9W runner (arms, seeds, failure anatomy, summary)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase9w_111_cho_calibration as p9w  # noqa: E402


def test_arm_gains() -> None:
    g = {a: p9w.arm_action(a).gains() for a in p9w.ARMS}
    assert g["A"] == {"H": 1.5, "O": 1.5}
    assert g["B"] == {"C": 0.5, "H": 1.5, "O": 1.5}
    assert g["C"] == {"C": 1.0, "H": 1.5, "O": 1.5}
    assert g["D"] == {"H": 0.75, "O": 0.75}
    for a in p9w.ARMS:
        act = p9w.arm_action(a)
        assert act.force_final_prefix
        assert act.gains().get("C", 0.0) < 5.0  # never full C gain
        assert all(v == 1 for v in act.signs([1, 1, 1]).values())


def test_control_is_frozen_9v() -> None:
    p9w.assert_control_is_frozen_9v()


def test_seeds_are_new() -> None:
    p9w.assert_new_seeds(p9w.SEED, p9w.MAX_TRIES, 2)
    try:
        p9w.assert_new_seeds(31000000, 300, 2)  # the 9V base must be rejected
        raise AssertionError("9V seed base accepted")
    except AssertionError as e:
        assert "overlap" in str(e)


def _res(*states: str) -> dict:
    st = [[int(c) for c in s] for s in states]
    return {"states": st, "hold": all(s == [1, 1, 1] for s in st[1:])}


def test_failure_anatomy() -> None:
    assert p9w.failure_anatomy(_res("111", "111", "111")) is None
    a = p9w.failure_anatomy(_res("111", "011", "011"))
    assert a["first_flip"] == "C" and a["final"] == "011" and a["first_fail_round"] == 1
    b = p9w.failure_anatomy(_res("111", "111", "001"))
    assert b["first_flip"] == "C+H" and b["first_fail_round"] == 2
    c = p9w.failure_anatomy(_res("111", "011", "001"))
    assert c["first_flip"] == "C" and c["later_collateral"] == ["H"]
    d = p9w.failure_anatomy(_res("111", "011", "111"))
    assert d["reverted"] == ["C"] and d["final"] == "111"


def test_summary_and_holm() -> None:
    hold, cflip, chflip = ("111", "111", "111"), ("111", "011", "011"), ("111", "111", "001")
    rows = [  # (A, B, C, D)
        (cflip, hold, hold, chflip),
        (chflip, hold, cflip, hold),
        (hold, hold, hold, hold),
        (hold, cflip, hold, hold),
    ]
    p = {"control": "A", "attempts": [{}] * 8,
         "arms": {a: {"gains": p9w.arm_action(a).gains()} for a in p9w.ARMS},
         "trials": [{a: _res(*r[i]) for i, a in enumerate(p9w.ARMS)} for r in rows]}
    s = p9w.summarize(p)
    assert s["n_acq"] == 4 and s["P_acq"] == 0.5
    assert s["arms"]["A"]["n_hold"] == 2 and s["arms"]["B"]["n_hold"] == 3
    assert s["arms"]["A"]["C_alone"] == 1 and s["arms"]["A"]["C_with_H"] == 1
    assert s["arms"]["A"]["C_involved"] == 2
    assert s["paired_vs_control"]["B"] == {**s["paired_vs_control"]["B"], "arm_only": 2, "control_only": 1}
    for v in s["paired_vs_control"].values():
        assert v["holm_p"] >= v["mcnemar_p"]
    h = p9w.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert {k: round(v, 9) for k, v in h.items()} == {"x": 0.03, "z": 0.06, "y": 0.06}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
