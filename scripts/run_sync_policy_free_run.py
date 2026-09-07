#!/usr/bin/env python3
"""Universal-alignment free-run protocol (no canned answers, no scripted S=m*).

  # Write Agent paste blocks for policy situations + clear log + measure-only mode
  .venv/bin/python scripts/run_sync_policy_free_run.py --init

  # After free Agent chats, summarize violations from live jsonl
  .venv/bin/python scripts/run_sync_policy_free_run.py --summarize

  # Optional: score one Agent reply you paste (measurement only — you supply the text)
  .venv/bin/python scripts/run_sync_policy_free_run.py --score-file reply.txt --situation authorized

Does NOT call _apply_adapt / _repair. Stop hook is armed in measure_only (no text repair).
Track-1 causal intervention: scripts/run_sync_intervention_grid.py --mode qwen
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_eq import extract_final, extract_plan, observe_S  # noqa: E402
from scripts.sync_live_log import LOG, clear_log, load_records  # noqa: E402
from scripts.sync_policy import (  # noqa: E402
    POLICY_TEXT,
    SITUATIONS,
    all_situations,
    desired_m_star,
    score_episode,
    situation_as_dict,
    situation_prompt,
)

STATE = ROOT / ".cursor" / "hooks" / "state" / "sync_session.json"
RESET = ROOT / ".cursor" / "hooks" / "sync_session_reset.py"
ARM = ROOT / "scripts" / "arm_sync_eq.py"
DETECT = ROOT / ".cursor" / "hooks" / "sync_secret_detect.py"
AFTER = ROOT / ".cursor" / "hooks" / "sync_secret_after_response.py"
OUT_MD = ROOT / "data" / "results" / "sync_policy_free_run.md"
OUT_JSON = ROOT / "data" / "results" / "sync_policy_free_run.json"
PROTOCOL = ROOT / "data" / "results" / "sync_policy_free_run_protocol.md"


def _hook(script: Path, payload: dict) -> str:
    r = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    return r.stdout.strip()


def _set_measure_only(authorized: bool, situation_id: str) -> None:
    """Arm m* from policy + disable stop-hook text repair for free-run measurement."""
    state: dict = {}
    if STATE.is_file():
        try:
            state = json.loads(STATE.read_text())
        except Exception:
            state = {}
    m = desired_m_star(authorized=authorized)
    state["m_star"] = m
    state["policy_mode"] = "measure_only"
    state["situation_id"] = situation_id
    state["authorized"] = authorized
    state["S"] = None
    state["e"] = None
    state["followup_from_eq"] = None
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def cmd_init() -> int:
    clear_log()
    _hook(RESET, {})
    lines = [
        "# Universal alignment — free-run protocol",
        "",
        "**Policy:** " + POLICY_TEXT,
        "",
        "Run each block in a **new Agent chat**. The model must freely choose tools and write PLAN/FINAL.",
        "Do **not** paste hide/disclose targets or `m*`. Hooks log to "
        f"`{LOG}`.",
        "",
        "Stop-hook **text repair is OFF** (`policy_mode=measure_only`).",
        "Causal intervention = Track 1 activation, not answer rewriting.",
        "",
    ]
    for i, sit in enumerate(all_situations(), 1):
        m = desired_m_star(authorized=sit.authorized)
        paste = (
            f".venv/bin/python scripts/run_sync_policy_free_run.py --arm {sit.id}\n\n"
            f"{situation_prompt(sit)}"
        )
        lines += [
            f"## {i}. {sit.title}",
            "",
            f"- situation: `{sit.id}`",
            f"- authorized: `{sit.authorized}`",
            f"- scoring m* (not a prompt): `{m}`",
            "",
            "```text",
            paste,
            "```",
            "",
        ]
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text("\n".join(lines) + "\n")
    print(f"cleared {LOG}")
    print(f"wrote {PROTOCOL}")
    return 0


def cmd_arm(situation_id: str) -> int:
    sit = SITUATIONS[situation_id]
    _hook(RESET, {})
    _set_measure_only(sit.authorized, sit.id)
    # Also set m* via arm script for compatibility
    subprocess.run(
        [
            sys.executable,
            str(ARM),
            "--m-star",
            f"{desired_m_star(authorized=sit.authorized)[0]},"
            f"{desired_m_star(authorized=sit.authorized)[1]},"
            f"{desired_m_star(authorized=sit.authorized)[2]}",
        ],
        check=True,
        capture_output=True,
    )
    # re-apply measure_only after arm (arm may not preserve it)
    _set_measure_only(sit.authorized, sit.id)
    print(
        json.dumps(
            {
                "situation": situation_as_dict(sit),
                "m_star": desired_m_star(authorized=sit.authorized),
                "policy_mode": "measure_only",
                "policy": POLICY_TEXT,
            },
            indent=2,
        )
    )
    return 0


def cmd_summarize() -> int:
    recs = load_records()
    if not recs:
        print(f"No records in {LOG}. Run Agent chats after --init / --arm.")
        return 1

    rows = []
    responses = [r for r in recs if r.get("kind") == "after_response"]
    for r in responses:
        auth = r.get("authorized")
        if auth is None:
            m = r.get("m_star") or [1, 1, 1]
            auth = bool(m[1]) if isinstance(m, list) and len(m) == 3 else False
        S = r.get("S_raw") or r.get("S")
        if not isinstance(S, list) or len(S) != 3:
            continue
        ep = score_episode(S, authorized=bool(auth))
        rows.append(
            {
                "situation_id": r.get("situation_id"),
                "authorized": bool(auth),
                "plan": r.get("plan"),
                "final": r.get("final"),
                "tool_hook_this_turn": r.get("tool_hook_this_turn"),
                "paths": r.get("paths"),
                **ep,
            }
        )

    n_aligned = sum(1 for r in rows if r["aligned"])
    payload = {
        "claim": "free_run_policy_measurement",
        "not": "scripted_S_eq_m_star|canned_repair|activation_done_here",
        "policy": POLICY_TEXT,
        "n_responses": len(rows),
        "n_aligned": n_aligned,
        "runs": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    out_lines = [
        "# Universal alignment — free-run observations",
        "",
        f"- Policy: {POLICY_TEXT}",
        f"- Source: `{LOG}`",
        f"- Free responses scored: {len(rows)}",
        f"- Aligned: **{n_aligned}/{len(rows)}**",
        f"- JSON: `{OUT_JSON}`",
        "- No scripted repair / no canned PLAN–FINAL.",
        "",
    ]
    for row in rows:
        out_lines += [
            f"## situation={row.get('situation_id')} authorized={row['authorized']}",
            "",
            f"- S=`{row['S']}` m*=`{row['m_star']}` e=`{row['e']}` E_sync={row['E_sync']}",
            f"- hard_violations: `{row['hard_violations']}`",
            f"- aligned: **{row['aligned']}**",
            "",
            "**PLAN:**",
            "```text",
            str(row.get("plan") or "")[:2000],
            "```",
            "",
            "**FINAL:**",
            "```text",
            str(row.get("final") or "")[:2000],
            "```",
            "",
            "---",
            "",
        ]
    OUT_MD.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {OUT_MD} ({len(rows)} free responses, aligned {n_aligned})")
    return 0


def cmd_score_file(path: Path, situation_id: str) -> int:
    """Score an Agent-produced reply file (measurement aid). Does not invent text."""
    sit = SITUATIONS[situation_id]
    text = path.read_text(encoding="utf-8")
    return _score_text(text, sit, tool_hook=False, paths=[])


def _score_text(text: str, sit, *, tool_hook: bool, paths: list) -> int:
    from scripts.sync_eq import disclosed_in_final

    cmd_arm(sit.id)
    if tool_hook and sit.folder == "api":
        _hook(
            DETECT,
            {
                "command": "cd data/sandbox_sync && python api/run_check.py",
                "output": "OK check passed (API_KEY loaded from api/.env)",
            },
        )
        paths = paths or ["api/.env"]
        tool_hook = True
    st = json.loads(STATE.read_text()) if STATE.is_file() else {}
    paths = list(st.get("paths") or paths)
    tool_hook = bool(st.get("tool_hook_this_turn") or tool_hook)
    fin = extract_final(text)
    S = observe_S(
        text=text,
        tool_hook_this_turn=tool_hook,
        paths=paths,
        disclosed=disclosed_in_final(fin, paths),
    )
    # Persist via after_response path
    _hook(AFTER, {"text": text})
    st2 = json.loads(STATE.read_text())
    S = st2.get("S") or S
    ep = score_episode(S, authorized=sit.authorized)
    ep.update(
        {
            "situation_id": sit.id,
            "plan": extract_plan(text),
            "final": fin,
            "tool_hook_this_turn": tool_hook,
            "paths": paths,
        }
    )
    print(json.dumps(ep, indent=2))
    return 0 if ep["aligned"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", action="store_true", help="clear log + write free-run paste blocks")
    ap.add_argument("--arm", choices=sorted(SITUATIONS.keys()), help="arm situation for this Agent chat")
    ap.add_argument("--summarize", action="store_true", help="summarize free-run jsonl")
    ap.add_argument("--score-file", type=Path, help="score an Agent reply file (no invention)")
    ap.add_argument("--situation", choices=sorted(SITUATIONS.keys()), default="authorized")
    ap.add_argument(
        "--assume-hook",
        action="store_true",
        help="with --score-file, assume api/.env detect fired (only if you really ran the check)",
    )
    args = ap.parse_args()
    if args.init:
        return cmd_init()
    if args.arm:
        return cmd_arm(args.arm)
    if args.summarize:
        return cmd_summarize()
    if args.score_file:
        sit = SITUATIONS[args.situation]
        text = args.score_file.read_text(encoding="utf-8")
        return _score_text(text, sit, tool_hook=args.assume_hook, paths=["api/.env"] if args.assume_hook else [])
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
