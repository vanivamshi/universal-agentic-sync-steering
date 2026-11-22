#!/usr/bin/env python3
"""Model-free tests for the Phase 9X confirmation runner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase9x_111_confirm as p9x  # noqa: E402


def test_frozen_arms() -> None:
    cfg = p9x.load_frozen()
    assert p9x.arm_action(cfg, "A").gains() == {"H": 1.5, "O": 1.5}
    assert p9x.arm_action(cfg, "B").gains() == {"C": 0.5, "H": 1.5, "O": 1.5}
    assert p9x.build_frozen() == cfg  # freeze is reproducible from the SHA-checked sources


def test_seeds_are_new() -> None:
    p9x.assert_new_seeds(p9x.SEED, p9x.MAX_TRIES, 2)
    for base in (31000000, 41000000):
        try:
            p9x.assert_new_seeds(base, 300, 2)
            raise AssertionError(f"seed base {base} accepted")
        except AssertionError as e:
            assert "overlap" in str(e)


def test_arm_order_deterministic_and_balanced() -> None:
    seeds = [p9x.acq_seed(p9x.SEED, i) for i in range(400)]
    assert all(p9x.arm_order(s) == p9x.arm_order(s) for s in seeds)
    n_ab = sum(p9x.arm_order(s) == ["A", "B"] for s in seeds)
    assert 150 < n_ab < 250, n_ab


def _res(*states: str) -> dict:
    st = [[int(c) for c in s] for s in states]
    return {"states": st, "hold": all(s == [1, 1, 1] for s in st[1:])}


def _payload(rows):
    cfg = p9x.load_frozen()
    return {"control": "A", "attempts": [{}] * (2 * len(rows)),
            "arms": {a: {"gains": p9x.arm_action(cfg, a).gains()} for a in p9x.ARMS},
            "trials": [{"order": ["A", "B"], "A": _res(*a), "B": _res(*b)} for a, b in rows]}


def test_decision_rule() -> None:
    h, c, ch = ("111", "111", "111"), ("111", "011", "011"), ("111", "111", "001")
    s = p9x.summarize(_payload([(c, h), (ch, h), (h, h), (h, c)]))
    assert s["decision"]["selection_rule_met"] and not s["decision"]["B_better_than_A"]
    s = p9x.summarize(_payload([(h, c), (h, h)]))
    assert s["decision"]["verdict"] == "B not confirmed"
    s = p9x.summarize(_payload([(h, h), (h, h)]))  # tie with no discordance: not confirmed
    assert not s["decision"]["selection_rule_met"]
    s = p9x.summarize(_payload([(c, h)] * 8))
    assert s["decision"]["B_better_than_A"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
