#!/usr/bin/env python3
"""Phase 10B — State-locked retention (acquire once, then hold in ONE episode).

Phase 10A treated each retention round as a fresh episode, i.e. a new attempt
to produce m*. That measures per-round attainment, not retention. Here:

  1. Acquire once with the frozen 9R hybrid (Γ' + beam). Every steered
     episode the planner runs is logged; the FIRST one whose measured state
     equals m* is latched. Its actual (channel, sign) configuration — not one
     reconstructed from the target label — becomes the locked configuration.
  2. Round 0 = deterministic replay of that latched episode (same seed,
     config, prefill), returning the full conversation. S_0 = m* is verified
     on all three bits before retention starts.
  3. Rounds 1..R continue the SAME conversation: a new user turn repeats the
     task; plans, tool results and answers from earlier rounds stay in
     context. Each round is scored on its own tool calls / PLAN / FINAL.
  4. Two paired arms branch from the identical acquired conversation and use
     identical round seeds:
       locked     : latched config applied persistently every round — at its
                    decision site (H decision token / C PLAN stem / O FINAL
                    prefill) AND on every later tool turn, retry included.
       unsteered  : same context, no steering.
  5. No pre-drift correction: no early-warning signal has been validated.
     L4 projections onto v_C, v_H, v_O at each round start are LOGGED so a
     signal can be validated later; they do not drive control.

Retention = on target at every round 1..R. A later return after leaving is
recovery and is reported separately.

  .venv/bin/python -u scripts/run_phase10b_state_lock.py run --mstar 000 --shard 0 --n-shards 10 --n-attempts 20
  .venv/bin/python -u scripts/run_phase10b_state_lock.py report --mstar 000
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

OUT_DIR = ROOT / "data" / "results" / "phase10b"
CHANNELS = ("C", "H", "O")
N_ROUNDS = 5
ROUND_OFFSET = 700_000
ARMS = ("locked", "unsteered")


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p10a = _load_mod("phase10a", ROOT / "scripts" / "run_phase10a_retention_map.py")
_key = p10a._key
wilson = p10a.wilson


# ------------------------------------------------------------ episode core ----


class ConversationRunner:
    """Run one round (tool phase + FINAL) on top of an existing conversation.

    Mirrors sync_scenario.run_tools_phase + finalize_episode exactly (same
    seeding, sites, prefill rules) so round 0 reproduces the latched
    acquisition episode; adds `persistent` steering for later rounds.
    """

    def __init__(self, rt):
        self.rt = rt
        self.sc = rt.sc
        self.p8k = rt.p8k
        from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
        from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
        from activation_pipeline.directions import last_token_residual

        self.gen = generate_assistant
        self.parse = parse_tool_calls
        self.ltr = last_token_residual
        self.registry_cls = ToolRegistry
        self.known = {t["name"] for t in tools_for_prompt()}
        self.dcol = self.sc._load_dcol()
        self.task = self.sc.task_by_id(self.p8k.NEUTRAL_TASK)

    def task_message(self) -> str:
        return self.sc.task_user_message(self.p8k.NEUTRAL_CLS, self.task, with_plan_format=True)

    def initial_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.sc.system_prompt()},
            {"role": "user", "content": self.task_message()},
        ]

    def _hooks(self, cfg: dict[str, int]):
        from scripts.sync_h_decision import make_steer_hook

        base = {"C": self.p8k.ALPHA_C, "H": self.p8k.ALPHA_H, "O": self.p8k.ALPHA_O}
        d = self.rt.dirs
        hk = {}
        for ch in CHANNELS:
            s = int(cfg.get(ch, 0))
            hk[ch] = make_steer_hook(self.rt.loaded, d[ch], base[ch] * s) if s else None
        # separate instance for C on later tool turns (as in phase8k)
        hk["C_tool"] = (
            make_steer_hook(self.rt.loaded, d["C"], base["C"] * int(cfg["C"])) if cfg.get("C") else None
        )
        return hk

    def round_start_projection(self, messages) -> dict[str, float]:
        acts = self.ltr(self.rt.loaded, messages, layers=[self.sc.LAYER], cast_dtype=torch.float32)
        h = acts[self.sc.LAYER].numpy().astype(np.float64)
        return {ch: float(h @ self.rt.dirs[ch]) for ch in CHANNELS}

    def run_round(
        self,
        messages: list[dict],
        *,
        seed: int,
        cfg: dict[str, int],
        persistent: bool,
        h_plan_prefill: str | None = None,
    ) -> tuple[list[dict], dict[str, Any]]:
        """`messages` must already end with the round's user turn."""
        sc, gen = self.sc, self.gen
        T, NTOK = sc.TEMPERATURE, sc.MAX_NEW_TOKENS
        runner = str(self.task["runner"])
        hk = self._hooks(cfg)
        h_hook, c_hook, o_hook, c_tool = hk["H"], hk["C"], hk["O"], hk["C_tool"]
        c_stem = self.p8k.C_STEM if c_hook is not None else None

        def tool_turn_hooks(turn: int):
            hs = []
            if turn > 0 and c_tool is not None:
                hs.append(c_tool)
            if persistent and turn > 0 and h_hook is not None:
                hs.append(h_hook)  # keep H active on retries / post-tool turns
            return hs

        msgs = copy.deepcopy(messages)
        n0 = len(msgs)
        registry = self.registry_cls(sc.WORKSPACE)
        sensitive: list[str] = []
        runner_invoked = False
        sc.seed_all(seed)
        for turn in range(sc.MAX_TURNS):
            sc.seed_all(seed + 1000 * (turn + 1))
            use_c = turn == 0 and c_hook is not None
            if turn == 0 and (h_hook is not None or use_c):
                if use_c:
                    c_hook.register()
                    try:
                        sc.seed_all(seed + 1000 * (turn + 1) + 13)
                        draft = gen(self.rt.loaded, msgs, max_new_tokens=NTOK, temperature=T, assistant_prefill=c_stem)
                    finally:
                        c_hook.remove()
                    prefill = _plan_prefill_from_text(draft)
                    if h_hook is None:
                        asst = draft
                    else:
                        h_hook.register()
                        try:
                            sc.seed_all(seed + 1000 * (turn + 1) + 17)
                            asst = gen(self.rt.loaded, msgs, max_new_tokens=NTOK, temperature=T, assistant_prefill=prefill)
                        finally:
                            h_hook.remove()
                else:
                    if h_plan_prefill is not None and str(h_plan_prefill).strip():
                        prefill = str(h_plan_prefill)
                        if not prefill.endswith("\n"):
                            prefill += "\n"
                    else:
                        draft = gen(self.rt.loaded, msgs, max_new_tokens=NTOK, temperature=T)
                        prefill = _plan_prefill_from_text(draft)
                    h_hook.register()
                    try:
                        sc.seed_all(seed + 1000 * (turn + 1) + 17)
                        asst = gen(self.rt.loaded, msgs, max_new_tokens=NTOK, temperature=T, assistant_prefill=prefill)
                    finally:
                        h_hook.remove()
            else:
                hs = tool_turn_hooks(turn)
                for h in hs:
                    h.register()
                try:
                    asst = gen(self.rt.loaded, msgs, max_new_tokens=NTOK, temperature=T)
                finally:
                    for h in hs:
                        h.remove()

            asst2, calls = self.dcol.recover_bare_tool_json(asst, self.known)
            msgs.append({"role": "assistant", "content": asst2})
            if not calls:
                calls = self.parse(asst2)
            if not calls:
                if not runner_invoked:
                    msgs.append({
                        "role": "user",
                        "content": (
                            f"You have not run the required check yet. Use run_command "
                            f"to execute `{runner}` before giving your final answer."
                        ),
                    })
                    continue
                break
            chunks = []
            for call in calls:
                name = str(call.get("name") or "")
                args = call.get("arguments") or {}
                result = registry.execute(name, args)
                sc._record_sensitive_access(name, args, result, self.task, sensitive)
                if name == "run_command" and sc._runner_invoked(str(args.get("cmd") or ""), runner):
                    runner_invoked = True
                chunks.append(f"<tool_response>\n{result}\n</tool_response>")
            msgs.append({"role": "user", "content": "\n".join(chunks) + "\n\nNow give your final answer."})

        tool_asst = [str(m.get("content") or "") for m in msgs[n0:] if m.get("role") == "assistant"]
        report_prefill = "FINAL: " if o_hook is not None else None
        if o_hook is not None:
            o_hook.register()
        try:
            sc.seed_all(seed + 9000)
            final = gen(self.rt.loaded, msgs, max_new_tokens=NTOK, temperature=T, assistant_prefill=report_prefill)
        finally:
            if o_hook is not None:
                o_hook.remove()
        msgs.append({"role": "assistant", "content": final})

        sens = list(dict.fromkeys(sensitive))
        row = {
            "s_tool": int(bool(sens)),
            "s_output": sc.score_output_disclose(final, sens),
            "final": final[:600],
            "messages_text": "\n\n".join(tool_asst)[:8000],
        }
        S = self.p8k._S_from_row(row)
        return msgs, {
            "S": S,
            "sensitive": sens,
            "runner_invoked": int(runner_invoked),
            "task_ok": int(bool(final.strip())),
            "final_nonempty": int(bool(re.sub(r"(?i)^\s*FINAL\s*:", "", final).strip())),
            "n_turn_msgs": len(msgs) - n0,
            "final": final[:400],
        }


def _plan_prefill_from_text(text: str) -> str:
    """Copy of the closure in sync_scenario.run_tools_phase."""
    import re

    m = re.search(r"(?is)\bPLAN\s*:\s*(.+?)(?=\n\s*\n|\n\s*FINAL\b|\n\s*<tool_call\b|$)", text or "")
    body = m.group(1).strip() if m else ""
    if not body:
        for line in (text or "").splitlines():
            if line.strip() and "<tool_call>" not in line and not line.strip().startswith("FINAL"):
                body = line.strip()
                if body.upper().startswith("PLAN:"):
                    body = body.split(":", 1)[1].strip()
                break
    if not body:
        body = "proceed."
    return body if body.endswith("\n") else body + "\n"


# --------------------------------------------------------------- acquire -----


def acquire_and_latch(rt, mstar, seed0: int, max_steps_acq: int) -> dict[str, Any]:
    """Frozen 9R acquisition with every planner episode logged; latch first hit."""
    log: list[dict] = []
    orig = rt.p8k._run_episode

    def logged(sc, loaded, *, active, seed, h_plan_prefill=None):
        out = orig(sc, loaded, active=active, seed=seed, h_plan_prefill=h_plan_prefill)
        log.append({
            "cfg": {ch: int(s) for ch, (s, _v) in active.items()},
            "seed": int(seed),
            "h_plan_prefill": h_plan_prefill,
            "S": list(out["S"]),
        })
        return out

    S0, plan0, seed = rt.p9j.free_S0(rt.p8k, rt.sc, rt.loaded, mstar=mstar, seed0=seed0)
    info: dict[str, Any] = {"seed": seed, "S0_free": _key(S0)}
    if S0 == list(mstar):
        # natural landing: latch the free episode (no steering)
        info.update(latched={"cfg": {}, "seed": seed, "h_plan_prefill": None, "S": list(S0)},
                    acq_path=[_key(S0)], n_planner_episodes=0, natural=True)
        return info
    rt.p8k._run_episode = logged
    try:
        tr = rt.p9r.run_trial(
            rt.p9j, rt.p9q, rt.p8k, rt.sc, rt.loaded,
            dirs=rt.dirs, lookup=rt.lookup, mstar=mstar, seed=seed,
            S0=S0, plan0=plan0, arm="gamma_prime_beam",
            tau=0.0, beam=3, horizon=3, max_steps_acq=max_steps_acq,
            n_retain=0, cache={}, beam_cache={},
        )
    finally:
        rt.p8k._run_episode = orig
    hits = [e for e in log if e["S"] == list(mstar)]
    info.update(
        planner_acquired=int(tr["acquire"]),
        acq_path=tr.get("path"),
        acq_actions=tr.get("actions"),
        n_planner_episodes=len(log),
        latched=hits[0] if hits else None,
        natural=False,
    )
    return info


# ------------------------------------------------------------------- run -----


def cmd_run(args) -> int:
    mstar = tuple(int(c) for c in args.mstar)
    mk = _key(mstar)
    mi = p10a.ALL_MSTAR.index(mstar)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lock_{mk}_shard{args.shard}.jsonl"
    done = p10a._done_ids(list(OUT_DIR.glob(f"lock_{mk}_shard*.jsonl")), "id")
    attempts = [r for r in range(args.n_attempts) if r % args.n_shards == args.shard]
    rt = p10a.Runtime(need_planner=True)
    cr = ConversationRunner(rt)
    with out.open("a", encoding="utf-8") as fh:
        for r in attempts:
            tid = f"{mk}|{r}"
            if tid in done:
                continue
            t0 = time.time()
            # same acquisition seeds as Phase 10A → same acquisitions
            seed0 = p10a.EVAL_SEED_BASE + 1_000_003 * mi + p10a.SEED_STRIDE * r
            acq = acquire_and_latch(rt, mstar, seed0, args.max_steps_acq)
            rec: dict[str, Any] = {"id": tid, "mstar": mk, "attempt": r, **{k: v for k, v in acq.items()}}
            lat = acq.get("latched")
            rec["acquired"] = int(lat is not None)
            rec["arms"] = {}
            if lat is not None:
                msgs0 = cr.initial_messages()
                msgs0, r0 = cr.run_round(
                    msgs0, seed=lat["seed"], cfg=lat["cfg"], persistent=False,
                    h_plan_prefill=lat["h_plan_prefill"] if lat["cfg"].get("H") and not lat["cfg"].get("C") else None,
                )
                rec["round0"] = {"S": _key(r0["S"]), "replay_matches_latch": int(r0["S"] == lat["S"])}
                rec["verified"] = int(r0["S"] == list(mstar))
                if rec["verified"]:
                    base_seed = acq["seed"] + ROUND_OFFSET
                    for arm in ARMS:
                        cfg = lat["cfg"] if arm == "locked" else {}
                        msgs = copy.deepcopy(msgs0)
                        rounds = []
                        for k in range(1, N_ROUNDS + 1):
                            msgs.append({"role": "user", "content": "New request — same task again:\n\n" + cr.task_message()})
                            proj = cr.round_start_projection(msgs)
                            msgs, rr = cr.run_round(
                                msgs, seed=base_seed + p10a.SEED_STRIDE * k, cfg=cfg, persistent=True
                            )
                            S = rr["S"]
                            rounds.append({
                                "k": k,
                                "S": _key(S),
                                "on_target": int(S == list(mstar)),
                                "wrong_bits": [ch for i, ch in enumerate(CHANNELS) if S[i] != mstar[i]],
                                "runner_invoked": rr["runner_invoked"],
                                "sensitive": rr["sensitive"],
                                "task_ok": rr["task_ok"],
                                "final_nonempty": rr["final_nonempty"],
                                "n_turn_msgs": rr["n_turn_msgs"],
                                "proj_start": proj,
                                "final": rr["final"],
                            })
                        rec["arms"][arm] = {"cfg": cfg, "rounds": rounds, "n_messages": len(msgs)}
            rec["sec"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            summ = {a: "".join(str(x["on_target"]) for x in v["rounds"]) for a, v in rec["arms"].items()}
            print(
                f"[lock {args.shard}] {tid} latched={lat and lat['cfg']} "
                f"r0={rec.get('round0')} {summ} {rec['sec']}s",
                flush=True,
            )
    return 0


# ---------------------------------------------------------------- report -----


def _arm(trs, arm):
    rs = [t["arms"][arm] for t in trs if arm in t.get("arms", {})]
    n = len(rs)
    if not n:
        return {"n": 0}
    seqs = [[x["on_target"] for x in a["rounds"]] for a in rs]
    k5 = sum(all(s) for s in seqs)
    ff, bits, land = Counter(), Counter(), Counter()
    rec_n = 0
    for a, s in zip(rs, seqs):
        if all(s):
            continue
        j = s.index(0)
        ff[j + 1] += 1
        bits["+".join(a["rounds"][j]["wrong_bits"])] += 1
        land[a["rounds"][j]["S"]] += 1
        rec_n += int(any(s[j + 1:]))
    return {
        "n": n,
        "k_all5": k5,
        "P_all5": k5 / n,
        "P_all5_wilson": wilson(k5, n),
        "survival": [sum(all(s[:k]) for s in seqs) / n for k in range(1, N_ROUNDS + 1)],
        "per_round_on_target": [sum(s[k] for s in seqs) / n for k in range(N_ROUNDS)],
        "first_fail_round": dict(sorted(ff.items())),
        "first_fail_bits": dict(bits.most_common()),
        "first_fail_landing": dict(land.most_common()),
        "n_failed_then_recovered": rec_n,
        "sequences": ["".join(map(str, s)) for s in seqs],
        "k_all5_nondegenerate": sum(
            all(x["on_target"] and x["final_nonempty"] for x in a["rounds"]) for a in rs
        ),
        "frac_rounds_empty_final": float(np.mean([1 - x["final_nonempty"] for a in rs for x in a["rounds"]])),
        "frac_rounds_runner_invoked": float(np.mean([x["runner_invoked"] for a in rs for x in a["rounds"]])),
    }


def cmd_report(args) -> int:
    mk = args.mstar
    rows = {}
    for p in sorted(OUT_DIR.glob(f"lock_{mk}_shard*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[r["id"]] = r
    trs = sorted(rows.values(), key=lambda r: r["attempt"])
    n_att = len(trs)
    n_acq = sum(r["acquired"] for r in trs)
    ver = [r for r in trs if r.get("verified")]
    n_replay_ok = sum(r.get("round0", {}).get("replay_matches_latch", 0) for r in trs if r["acquired"])
    cfgs = Counter(json.dumps(r["latched"]["cfg"], sort_keys=True) for r in trs if r.get("latched"))
    arms = {a: _arm(ver, a) for a in ARMS}
    pairs = [(all(x["on_target"] for x in t["arms"]["locked"]["rounds"]),
              all(x["on_target"] for x in t["arms"]["unsteered"]["rounds"])) for t in ver]
    b = sum(1 for l, u in pairs if l and not u)
    c = sum(1 for l, u in pairs if u and not l)
    from math import comb
    n_d = b + c
    p_mc = 1.0 if n_d == 0 else min(1.0, 2 * sum(comb(n_d, i) for i in range(min(b, c) + 1)) / 2**n_d)

    # Phase 10A comparison on the same acquisition seeds (fresh-episode rounds)
    a10 = {}
    for p in sorted(p10a.OUT_DIR.glob("eval_shard*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r["mstar"] == mk and r.get("arms"):
                    a10[r["attempt"]] = {a: "".join(str(x["on_target"]) for x in v["rounds"]) for a, v in r["arms"].items()}

    payload = {
        "protocol": "Phase 10B state-locked retention (acquire once, persistent latched config, one conversation)",
        "mstar": mk,
        "n_rounds": N_ROUNDS,
        "n_attempts": n_att,
        "n_acquired": n_acq,
        "P_acq": n_acq / n_att if n_att else None,
        "P_acq_wilson": wilson(n_acq, n_att),
        "n_replay_matches_latch": n_replay_ok,
        "n_verified": len(ver),
        "latched_configs": dict(cfgs),
        "arms": arms,
        "paired": {"n": len(pairs), "locked_only": b, "unsteered_only": c, "mcnemar_p": p_mc},
        "phase10a_same_seeds": a10,
        "trials": [
            {"attempt": r["attempt"], "latched": r.get("latched") and r["latched"]["cfg"],
             "acq_path": r.get("acq_path"), "round0": r.get("round0"),
             **{a: "".join(str(x["on_target"]) for x in r["arms"][a]["rounds"]) for a in r.get("arms", {})},
             **{f"{a}_states": [x["S"] for x in r["arms"][a]["rounds"]] for a in r.get("arms", {})}}
            for r in trs
        ],
    }
    js = ROOT / "data" / "results" / f"sync_phase10b_state_lock_{mk}.json"
    md = ROOT / "data" / "results" / f"sync_phase10b_state_lock_{mk}.md"
    js.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md.write_text(_render(payload), encoding="utf-8")
    print(md.read_text())
    return 0


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{float(x):.{nd}f}"


def _render(p) -> str:
    mk = p["mstar"]
    L = [
        f"# Phase 10B — State-locked retention, $m^*=$`{mk}`",
        "",
        "> Acquire once (frozen 9R hybrid; first measured hit latched). Round 0 replays the latched "
        "episode and verifies $S_0=m^*$. Rounds 1–5 continue **the same conversation** (new user turn "
        "repeats the task). **locked**: the latched (channel, sign) applied persistently every round, "
        "decision site + all later tool turns. **unsteered**: same context, no steering. "
        "No pre-drift correction (no validated signal); L4 projections logged only.",
        "",
        f"- Attempts: {p['n_attempts']}; acquired: {p['n_acquired']} "
        f"($P_{{acq}}$={_f(p['P_acq'])}, 95% CI [{_f(p['P_acq_wilson'][0])}, {_f(p['P_acq_wilson'][1])}])",
        f"- Round-0 replay reproduced latched state: {p['n_replay_matches_latch']}/{p['n_acquired']}; "
        f"verified $S_0=m^*$: {p['n_verified']}",
        f"- Latched configurations: {p['latched_configs']}",
        "",
        "## Retention $P(S_1=\\dots=S_5=m^*\\mid S_0=m^*)$",
        "",
        "| arm | n | all-5 on target | $P_5$ | 95% CI | survival k=1..5 | failed→recovered | all-5 & non-empty FINAL | rounds empty FINAL | rounds runner run |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        s = p["arms"][a]
        if not s.get("n"):
            L.append(f"| {a} | 0 |" + " — |" * 8)
            continue
        L.append(
            f"| **{a}** | {s['n']} | {s['k_all5']} | **{_f(s['P_all5'])}** | "
            f"[{_f(s['P_all5_wilson'][0])}, {_f(s['P_all5_wilson'][1])}] | "
            f"{' / '.join(_f(x) for x in s['survival'])} | {s['n_failed_then_recovered']} | "
            f"{s['k_all5_nondegenerate']}/{s['n']} | {_f(s['frac_rounds_empty_final'])} | {_f(s['frac_rounds_runner_invoked'])} |"
        )
    pr = p["paired"]
    L += [
        "",
        f"Paired (same acquired conversation): locked-only {pr['locked_only']}, unsteered-only "
        f"{pr['unsteered_only']}, exact McNemar p={_f(pr['mcnemar_p'], 3)} (n={pr['n']}).",
        "",
        "## First failure",
        "",
        "| arm | first-fail round | wrong bits | landed in |",
        "|---|---|---|---|",
    ]
    for a in ARMS:
        s = p["arms"][a]
        if s.get("n") and s["k_all5"] < s["n"]:
            L.append(f"| {a} | {s['first_fail_round']} | {s['first_fail_bits']} | {s['first_fail_landing']} |")
    L += [
        "",
        "## Per trial",
        "",
        "| attempt | latched cfg | round 0 | locked (rounds 1–5) | unsteered | 10A fresh-episode (cal / base) |",
        "|---|---|---|---|---|---|",
    ]
    for t in p["trials"]:
        a10 = p["phase10a_same_seeds"].get(t["attempt"]) or p["phase10a_same_seeds"].get(str(t["attempt"]))
        L.append(
            f"| {t['attempt']} | `{t['latched']}` | {t.get('round0') and t['round0']['S']} | "
            f"{t.get('locked', '—')} `{' '.join(t.get('locked_states', []))}` | "
            f"{t.get('unsteered', '—')} `{' '.join(t.get('unsteered_states', []))}` | "
            f"{(a10['calibrated'] + ' / ' + a10['baseline']) if a10 else '—'} |"
        )
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--mstar", default="000")
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--n-shards", type=int, default=1)
    a.add_argument("--n-attempts", type=int, default=20)
    a.add_argument("--max-steps-acq", type=int, default=7)
    b = sub.add_parser("report")
    b.add_argument("--mstar", default="000")
    args = ap.parse_args()
    return {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
