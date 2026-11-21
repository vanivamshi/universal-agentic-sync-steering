#!/usr/bin/env python3
"""Unit tests for the frozen 9U retention controller and Phase 9V helpers (no model)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_phase9v_heldout_retention as p9v  # noqa: E402
from scripts import sync_retention_controller as rc  # noqa: E402

EXPECTED = {  # exact 9U selections (sync_phase9u_retention_gains.md)
    "100": {"O": 0.75},
    "101": {"O": 0.75},
    "110": {"H": 0.75, "O": 0.75},
    "111": {"H": 1.5, "O": 1.5},
}


def test_frozen_matches_9u() -> None:
    cfg = rc.load_frozen(p9v.FROZEN, ROOT)
    for t, g in EXPECTED.items():
        a = rc.retention_action(cfg, t)
        assert a.gains() == g, (t, a.gains())
        assert a.force_final_prefix
        assert "C" not in a.channels  # C dropped for every calibrated target
    assert rc.retention_action(cfg, [1, 1, 1]).signs([1, 1, 1]) == {"H": 1, "O": 1}
    assert rc.retention_action(cfg, "101").signs([1, 0, 1]) == {"O": 1}
    try:
        rc.retention_action(cfg, "000")
        raise AssertionError("uncalibrated target must raise")
    except KeyError:
        pass


def test_tamper_detected() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        cfg = json.loads(p9v.FROZEN.read_text())
        for t in cfg["targets"].values():
            dst = root / t["source"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / t["source"], dst)
        path = root / "frozen.json"
        path.write_text(json.dumps(cfg))
        rc.load_frozen(path, root)
        src = root / cfg["targets"]["110"]["source"]
        src.write_text(src.read_text() + " ")
        try:
            rc.load_frozen(path, root)
            raise AssertionError("modified 9U source must be rejected")
        except AssertionError as e:
            assert "changed since freeze" in str(e)


def test_seeds_disjoint() -> None:
    cfg = rc.load_frozen(p9v.FROZEN, ROOT)
    for t in p9v.TARGETS:
        ti = p9v.ALL_MSTAR.index(tuple(int(c) for c in t))
        n, mt = p9v.DEFAULTS[t]
        p9v.assert_disjoint_from_9u(p9v.SEED, ti, mt, 2, cfg)
    try:
        p9v.assert_disjoint_from_9u(20261121, 4, 60, 2, cfg)  # the 9U seed itself
        raise AssertionError("9U seed base must be rejected")
    except AssertionError as e:
        assert "overlap" in str(e)


def test_stats() -> None:
    lo, hi = rc.wilson(12, 12)
    assert hi == 1.0 and 0.75 < lo < 0.76
    assert rc.wilson(0, 0) == (None, None)
    assert rc.mcnemar_exact(0, 0) == 1.0
    assert abs(rc.mcnemar_exact(1, 0) - 1.0) < 1e-12
    assert abs(rc.mcnemar_exact(0, 6) - 0.03125) < 1e-12


def test_retain_and_summary() -> None:
    cfg = rc.load_frozen(p9v.FROZEN, ROOT)
    seen = []

    def fake_round(msgs, action, target, seed):
        seen.append((action.name, seed))
        return msgs, (list(target) if seed % 2 else [0, 0, 0]), {}

    r = rc.retain(fake_round, [], cfg, [1, 0, 0], [1, 3])
    assert r["hold"] and seen == [("CAL9U_100", 1), ("CAL9U_100", 3)]
    assert not rc.retain(fake_round, [], cfg, [1, 0, 0], [1, 2])["hold"]

    tr = lambda a, b: {"A": {"hold": a}, "B": {"hold": b}}  # noqa: E731
    p = {"attempts": [{}] * 10, "trials": [tr(1, 1), tr(1, 0), tr(0, 1), tr(0, 1)],
         "arms": {"A": {"gains": {"C": 5.0, "H": 1.5, "O": 1.5}}, "B": {"gains": {"O": 0.75}}}}
    s = p9v.summarize(p)
    assert s["n_acq"] == 4 and s["P_acq"] == 0.4
    assert s["A"]["n_hold"] == 2 and s["B"]["n_hold"] == 3
    assert s["paired"]["B_only"] == 2 and s["paired"]["A_only"] == 1
    assert s["A"]["total_gain"] == 8.0 and s["B"]["total_gain"] == 0.75
    assert abs(s["B"]["P_e2e"] - 0.3) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
