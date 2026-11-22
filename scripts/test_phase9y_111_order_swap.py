#!/usr/bin/env python3
"""Model-free tests for the Phase 9Y order-swap analysis (uses the saved 9X result)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase9y_111_order_swap as p9y  # noqa: E402


def _fake_9y(x, mutate=None):
    trials = []
    for xt in x["trials"]:
        t = {"seed": xt["seed"], "round_seeds": xt["round_seeds"], "order": p9y.swap(xt["order"]),
             "order_9x": xt["order"], "reacquired": True, "S0": [1, 1, 1],
             "A": copy.deepcopy(xt["A"]), "B": copy.deepcopy(xt["B"])}
        if mutate:
            mutate(t)
        trials.append(t)
    return {"arms": x["arms"], "trials": trials}


def test_swap() -> None:
    assert p9y.swap(["A", "B"]) == ["B", "A"] and p9y.swap(["B", "A"]) == ["A", "B"]


def test_perfect_reproduction_has_no_order_effect() -> None:
    x = p9y.load_9x()
    s = p9y.summarize(_fake_9y(x), x)
    n = len(x["trials"])
    assert s["n_reacquired"] == n
    assert s["arms"]["A"]["n_hold"] == x["summary"]["arms"]["A"]["n_hold"]
    assert s["arms"]["B"]["n_hold"] == x["summary"]["arms"]["B"]["n_hold"]
    for a in p9y.ARMS:
        assert s["reproduce"][a] == {"n": n, "identical": n, "same_states": n, "same_hold": n}
        q = s["position_by_arm"][a]
        assert q["first_only"] == q["second_only"] == 0 and q["first_hold"] == q["second_hold"]
    assert s["order_counts"] == {"AB": x["summary"]["order_counts"]["BA"], "BA": x["summary"]["order_counts"]["AB"]}
    assert sum(v["9Y"]["n"] for v in s["by_9x_order"].values()) == n


def test_detects_position_effect() -> None:
    x = p9y.load_9x()
    fail = {"states": [[1, 1, 1], [0, 1, 1], [0, 1, 1]], "hold": False, "rounds": [], "path": []}

    def second_fails(t):  # the second-executed fork in 9Y always fails
        t[t["order"][1]] = copy.deepcopy(fail)

    s = p9y.summarize(_fake_9y(x, second_fails), x)
    assert s["position_9y"]["second_hold"] == 0 and s["position_9y"]["mcnemar_p"] < 1e-6
    assert s["reproduce"]["A"]["identical"] < len(x["trials"])


def test_skips_unreacquired() -> None:
    x = p9y.load_9x()
    first = x["trials"][0]["seed"]

    def drop_first(t):
        if t["seed"] == first:
            t["reacquired"] = False

    s = p9y.summarize(_fake_9y(x, drop_first), x)
    assert s["n_reacquired"] == len(x["trials"]) - 1 and s["n_trials"] == len(x["trials"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
