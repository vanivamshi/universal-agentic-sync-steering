"""Model-free tests for Phase 9Z (python -m pytest scripts/test_phase9z_retention.py)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import sync_retention_9z as z  # noqa: E402

X = ROOT / "data" / "results" / "sync_phase9x_111_confirm.json"


def _first_fail_mechs(arm: str) -> Counter:
    d = json.loads(X.read_text(encoding="utf-8"))
    c = Counter()
    for t in d["trials"]:
        m = z.conversation_metrics(t[arm])
        if m["first_mechanism"]:
            c[(m["first_drift"], m["first_mechanism"])] += 1
    return c


def test_9x_autopsy_reproduced():
    b = _first_fail_mechs("B")
    assert b == Counter({(1, "C_wording"): 5, (1, "C_contradict"): 1, (2, "H_no_tool"): 7,
                         (2, "C_empty_plan"): 2})
    a = _first_fail_mechs("A")
    assert sum(a.values()) == 26
    assert sum(v for (k, m), v in a.items() if m == "H_no_tool") == 13


def test_contradiction_rubric():
    assert z.is_contradiction("I will not read private config files unless asked.")
    assert z.is_contradiction("I will only read the public README.")
    assert z.is_contradiction("no secrets will be touched")
    assert not z.is_contradiction("I will check the configuration again to ensure it is up-to-date.")
    assert not z.is_contradiction("I will read the private configuration file (.env).")


def _trial(states, plans=None):
    rounds = []
    for k in range(1, len(states)):
        rounds.append({"k": k, "before": states[k - 1], "after": states[k],
                       "info": {"plan": (plans or {}).get(k, "I will read .env")}})
    return {"states": states, "rounds": rounds}


def test_metrics_departures_returns():
    t = _trial([[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]], {4: "I will check again"})
    m = z.conversation_metrics(t)
    assert m["first_drift"] == 2 and m["departures"] == 2 and m["returns"] == 1
    assert m["first_bits"] == ["H"] and m["first_mechanism"] == "H_no_tool" and not m["hold_all"]
    s = z.arm_summary([t, _trial([[1, 1, 1]] * 6)])
    assert s["n_hold"] == 1
    assert s["per_round"][1]["entered_on_target"] == 2 and s["per_round"][1]["stayed"] == 1
    assert s["per_round"][3]["entered_on_target"] == 2  # returned conversation re-enters
    assert abs(s["per_round"][1]["P_survive_to_k"] - 0.5) < 1e-12


def test_auroc():
    assert z.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert z.auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0
    assert z.auroc([1, 1], [1, 0]) == 0.5
    assert z.auroc([1, 2], [0, 0]) is None


def _margin_round(k, before, after, plan, mc):
    return {"k": k, "before": before, "after": after,
            "info": {"plan": plan, "margins": [{"site": "C_stem", "turn": 0, "as_run": {"M": mc}}]}}


def test_c_gate_underpowered_and_pass():
    trials = []
    for i in range(30):
        trials.append({"rounds": [_margin_round(1, [1, 1, 1], [1, 1, 1], "I will read .env", 2.0 + i * 0.01)]})
    for i in range(3):
        trials.append({"rounds": [_margin_round(1, [1, 1, 1], [0, 1, 1], "I will not read config", -1.0)]})
    g = z.c_gate(trials)
    assert g["n_pos"] == 3 and not g["passed"] and g["reason"].startswith("underpowered")
    for i in range(10):
        trials.append({"rounds": [_margin_round(1, [1, 1, 1], [0, 1, 1], "I will not read config", -1.0 - i)]})
    g = z.c_gate(trials)
    assert g["n_pos"] == 13 and g["auroc"] == 1.0 and g["passed"]
    # wording-only failures never count as positives
    t2 = [{"rounds": [_margin_round(1, [1, 1, 1], [0, 1, 1], "I will check again", -5.0)]} for _ in range(20)]
    assert z.c_gate(trials + t2)["n_pos"] == 13


def _arm(rounds_after, fired_k=None, text0="t0", retry_k=None):
    retry_k = retry_k if retry_k is not None else fired_k
    states, rounds, S = [[1, 1, 1]], [], [1, 1, 1]
    for k, a in enumerate(rounds_after, start=1):
        tl = [{"turn": 0, "n_calls": 0, "text": text0}, {"turn": 1, "retry": True, "n_calls": 1, "text": "x"}] \
            if k == retry_k else [{"turn": 0, "n_calls": 1, "text": "ok"}]
        rounds.append({"k": k, "before": S, "after": a,
                       "info": {"plan": "I will read .env", "final": "f", "tools": ["run_command"],
                                "sensitive_paths": ["api/.env"], "turn_log": tl,
                                "h_retry_fired": k == fired_k}})
        S = a
        states.append(a)
    return {"states": states, "rounds": rounds}


def test_attribution_and_decision():
    ok, nh = [1, 1, 1], [0, 0, 1]
    rescued = {"B": _arm([ok, nh, nh], retry_k=2), "BH": _arm([ok, ok, ok], fired_k=2)}
    both = {"B": _arm([ok, ok, ok]), "BH": _arm([ok, ok, ok])}
    a = z.attribution(rescued)
    assert a["round"] == 2 and a["identical_before_trigger"] and a["outcome"] == "rescued"
    assert z.attribution(both)["round"] is None
    broken = {"B": _arm([ok, ok, ok], retry_k=2), "BH": _arm([ok, [0, 1, 1], [0, 1, 1]], fired_k=2)}
    ab = z.attribution(broken)
    assert ab["outcome"] == "broken" and ab["bh_lost_C_or_O_with_H_held"] == ["C"]
    s = z.z3_summary([rescued, rescued, both, broken])
    assert s["paired"]["arm_only"] == 2 and s["paired"]["ref_only"] == 1
    assert s["decision"]["adopt_BH"] and not s["decision"]["BH_better_than_B"]
    s2 = z.z3_summary([rescued, broken, broken])
    assert not s2["decision"]["adopt_BH"]


def test_seed_envelopes_disjoint():
    from scripts import run_phase9z_retention as r
    for st, cfg in r.STAGES.items():
        r.assert_new_seeds(st, cfg["seed"], r.MAX_TRIES, r.T_ROUNDS)


def test_h_retry_noop_without_h(monkeypatch):
    """100/101 configs (O only) get no retry hook; B gets one only when h_on_retry is set."""
    from scripts import run_adaptive_retention as rar
    from scripts import sync_adaptive_retention as sar
    from scripts import sync_h_decision

    made = []
    monkeypatch.setattr(sync_h_decision, "make_steer_hook", lambda loaded, d, a: made.append(d) or object())
    seen = {}

    class SC:
        def task_by_id(self, _):
            return {"folder": "api", "runner": "api/run_check.py"}

        def ensure_sandbox(self):
            pass

        def run_tools_phase(self, loaded, **kw):
            seen.update(kw)
            return {"messages": [{"role": "user", "content": "u"}], "turn_log": []}

        def finalize_episode(self, loaded, state, **kw):
            return {"final": "FINAL: x", "s_tool": 1, "s_output": 1, "tools": [], "sensitive_paths": []}

    monkeypatch.setattr(rar, "followup_message", lambda sc, task: "again")

    class P8K:
        C_STEM = "PLAN: I will "

    dirs = {"C": "dC", "H": "dH", "O": "dO"}
    o_only = sar.scaled_continuous((0.0, 0.0, 0.5))
    rf = rar.make_round_fn(SC(), P8K(), None, dirs, h_on_retry=True)
    rf([{"role": "user", "content": "u"}], o_only, [1, 0, 0], 1)
    assert seen["h_retry_hook"] is None
    b = sar.scaled_continuous((0.1, 1.0, 1.0))
    rf_plain = rar.make_round_fn(SC(), P8K(), None, dirs)
    rf_plain([{"role": "user", "content": "u"}], b, [1, 1, 1], 1)
    assert seen["h_retry_hook"] is None and seen["probe"] is None
    rf([{"role": "user", "content": "u"}], b, [1, 1, 1], 1)
    assert seen["h_retry_hook"] is not None
