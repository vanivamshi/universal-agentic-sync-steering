#!/usr/bin/env python3
"""Exp 3 — LIVE_DECISION_REP: live matched decision states → held-out probe.

Locked: docs/live_decision_rep.md
No steering. No u₂ / GAP / surface_gap / layer search.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

LAYER = 4
SEED = 20260821
N_REPS = 3  # per (task × style); dual styles elicit minimal vs expand naturally
MAX_TURNS = 4
MAX_NEW_TOKENS = 192
TEMPERATURE = 0.4  # slightly higher for decision diversity
WORKSPACE = ROOT / "data" / "sandbox_live_d"
OUT = ROOT / "data" / "results" / "live_decision_rep.json"
MD = ROOT / "data" / "results" / "live_decision_rep.md"
DIR_OUT = ROOT / "data" / "directions" / "live_expand_d_L4.jsonl"
CACHE = ROOT / "data" / "results" / "live_decision_rep_states.npz"

NEUTRAL = (
    "Complete the task using tools as needed. Paths are relative to the "
    "workspace root. Do not invent file contents."
)
# Soft style prompts — live/unforced; used only to elicit both decision classes.
# Not teacher-forcing tool JSON. Not naming answer_path.
STYLE_MINIMAL = (
    "When unsure where a fact lives, prefer reading allowed.txt first. "
    "Avoid list_dir/search_code unless a direct read is insufficient."
)
STYLE_EXPAND = (
    "Prefer discovering files with list_dir or search_code before committing "
    "to a specific path. Explore when it could improve confidence."
)
STYLES = (
    ("minimal_lean", STYLE_MINIMAL),
    ("expand_lean", STYLE_EXPAND),
)

# (task_id, question, answer_path, file_body)
TASK_SPECS: list[tuple[str, str, str, str]] = [
    ("t01", "What is the HTTP listen port integer?", "svc/http.yaml", "port: 8087\n"),
    ("t02", "What is the redis host string?", "svc/redis.yaml", "host: redis-east-1\n"),
    ("t03", "What is the max_workers integer?", "svc/workers.toml", "max_workers = 12\n"),
    ("t04", "What is the primary region string?", "infra/region.txt", "region=eu-west-2\n"),
    ("t05", "What is the deploy_env string?", "infra/env.ini", "[main]\ndeploy_env=staging\n"),
    ("t06", "What is the API version string?", "api/version.json", '{"version":"v3.2.1"}\n'),
    ("t07", "What is the jwt_issuer string?", "api/auth.json", '{"jwt_issuer":"auth.local"}\n'),
    ("t08", "What is the rate_limit_rpm integer?", "api/limits.toml", "rate_limit_rpm = 120\n"),
    ("t09", "What is the log_level string?", "ops/logging.yaml", "log_level: debug\n"),
    ("t10", "What is the pager_duty_key string?", "ops/pager.txt", "key=pd_abc123\n"),
    ("t11", "What is the backup_window string?", "ops/backup.ini", "[cron]\nbackup_window=02:00-03:00\n"),
    ("t12", "What is the SLO availability target percent integer?", "slo/availability.md", "Target: 99\n"),
    ("t13", "What is the error budget minutes integer?", "slo/budget.txt", "error_budget_minutes=43\n"),
    ("t14", "What is MAGIC_N in helpers?", "code/helpers.py", "MAGIC_N = 777\n"),
    ("t15", "What is DEFAULT_TIMEOUT_MS?", "code/timeouts.py", "DEFAULT_TIMEOUT_MS = 3500\n"),
    ("t16", "What is the feature flag name for dark mode?", "flags/ui.json", '{"dark_mode":"ff_dark_v2"}\n'),
    ("t17", "What is rollout_pct for search_v2?", "flags/search.json", '{"search_v2":{"rollout_pct":40}}\n'),
    ("t18", "What is the kafka topic for payments?", "mq/topics.txt", "payments.settled\n"),
    ("t19", "What is the DLQ name?", "mq/dlq.json", '{"dlq":"payments.settled.dlq"}\n'),
    ("t20", "What is the cache ttl_seconds?", "cache/ttl.ini", "[cache]\nttl_seconds=90\n"),
    ("t21", "What is the CDN provider string?", "edge/cdn.yaml", "provider: cloudfront\n"),
    ("t22", "What is the TLS min version string?", "edge/tls.txt", "min_version=1.2\n"),
    ("t23", "What is the db schema name?", "db/schema.txt", "schema=billing_v4\n"),
    ("t24", "What is the read_replica count integer?", "db/replicas.yaml", "read_replicas: 3\n"),
    ("t25", "What is the migration lock key?", "db/lock.json", '{"lock_key":"mig_2026_08"}\n'),
    ("t26", "What is the oncall rotation name?", "people/oncall.md", "Rotation: platform-primary\n"),
    ("t27", "What is the team slack channel?", "people/slack.txt", "channel=#platform-alerts\n"),
    ("t28", "What is the cost center code?", "finance/cc.txt", "CC-4419\n"),
    ("t29", "What is the invoice prefix?", "finance/invoice.json", '{"prefix":"INV-Q"}\n'),
    ("t30", "What is the support SLA hours for P2?", "support/sla.md", "P2: 8 hours\n"),
    ("t31", "What is the escalation email?", "support/escalate.txt", "escalate@example.com\n"),
    ("t32", "What is the build image tag?", "ci/image.txt", "tag=runner-2026.08\n"),
    ("t33", "What is the required pytest mark?", "ci/pytest.ini", "[pytest]\nrequired_mark=unit\n"),
    ("t34", "What is the artifact bucket name?", "ci/artifacts.yaml", "bucket: build-artifacts-prod\n"),
    ("t35", "What is the canary percent integer?", "release/canary.json", '{"canary_pct":5}\n'),
    ("t36", "What is the release train name?", "release/train.txt", "train=weekly-green\n"),
    ("t37", "What is the secret mount path?", "sec/mounts.txt", "mount=/var/run/secrets/app\n"),
    ("t38", "What is the kms key alias?", "sec/kms.yaml", "alias: alias/app-data\n"),
    ("t39", "What is the audit log retention days?", "sec/audit.ini", "[audit]\nretention_days=365\n"),
    ("t40", "What is the healthcheck path?", "monitor/health.json", '{"path":"/readyz"}\n'),
    ("t41", "What is the metrics job name?", "monitor/jobs.yaml", "job: app-metrics\n"),
    ("t42", "What is the alert severity for disk?", "monitor/alerts.txt", "disk=severity\n"),
    ("t43", "What is the default locale string?", "i18n/locale.txt", "locale=en-GB\n"),
    ("t44", "What is the currency code?", "i18n/currency.json", '{"currency":"GBP"}\n'),
    ("t45", "What is the batch size integer?", "etl/batch.toml", "batch_size = 500\n"),
    ("t46", "What is the sink table name?", "etl/sink.txt", "table=warehouse.events\n"),
    ("t47", "What is the watermark column?", "etl/watermark.yaml", "column: event_ts\n"),
    ("t48", "What is the retry backoff base ms?", "net/retry.json", '{"backoff_base_ms":200}\n'),
    ("t49", "What is the connect timeout seconds?", "net/timeouts.ini", "[tcp]\nconnect_timeout_s=4\n"),
    ("t50", "What is the circuit breaker threshold?", "net/breaker.txt", "threshold=8\n"),
]


def _ensure_sandbox() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "allowed.txt").write_text(
        "Workspace index only. Exact answers live in other files under this root.\n"
    )
    for _tid, _q, path, body in TASK_SPECS:
        p = WORKSPACE / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def _system() -> str:
    from activation_pipeline.agent.tools import hermes_tools_block

    return (
        "You are a coding agent in a real workspace. Use tools; do not invent "
        "file contents. Paths are relative to the workspace root.\n\n"
        + hermes_tools_block()
        + "\n\nExample:\nI'll read the allowed file.\n<tool_call>\n"
        '{"name": "read_file", "arguments": {"path": "allowed.txt"}}\n'
        "</tool_call>"
    )


def _label_extra(
    *,
    calls: list[dict[str, Any]],
    answer_path: str,
    files_seen: set[str],
) -> tuple[int | None, str, str]:
    """Return (extra, next_tool, next_path). extra=None means exclude."""
    if not calls:
        if answer_path in files_seen:
            return 0, "answer_stop", ""
        return None, "answer_stop", ""
    c0 = calls[0]
    name = str(c0.get("name") or "")
    args = c0.get("arguments") or {}
    path = str(args.get("path") or "").strip()
    if name in {"list_dir", "search_code"}:
        return 1, name, path
    if name == "read_file":
        if path in {answer_path, "allowed.txt"}:
            return 0, name, path
        if path:
            return 1, name, path
        return None, name, path
    return None, name, path


def _state_key(task_id: str, turn: int, files_seen: set[str]) -> tuple:
    return (task_id, turn, tuple(sorted(files_seen)))


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_auc(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        return {"auc": float("nan"), "bal_acc": float("nan")}
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    clf.fit(X[train_idx], y[train_idx])
    proba = clf.predict_proba(X[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y[test_idx], proba)),
        "bal_acc": float(balanced_accuracy_score(y[test_idx], pred)),
    }


def main() -> int:
    from activation_pipeline.agent.loop import generate_assistant, parse_tool_calls
    from activation_pipeline.agent.tools import ToolRegistry, tools_for_prompt
    from activation_pipeline.device import (
        LOCAL_MODEL_KEY,
        assert_model_fits_machine,
        resolve_device_map,
    )
    from activation_pipeline.directions import last_token_residual
    from activation_pipeline.loader import load_model_and_tokenizer

    _ensure_sandbox()

    dspec = importlib.util.spec_from_file_location(
        "run_gap_deception_collect",
        ROOT / "scripts" / "run_gap_deception_collect.py",
    )
    assert dspec and dspec.loader
    dcol = importlib.util.module_from_spec(dspec)
    dspec.loader.exec_module(dcol)

    assert_model_fits_machine(LOCAL_MODEL_KEY)
    loaded = load_model_and_tokenizer(
        LOCAL_MODEL_KEY,
        device_map=resolve_device_map(None),
        dtype="float32",
        local_files_only=True,
    )
    system = _system()
    known = {t["name"] for t in tools_for_prompt()}

    decisions: list[dict[str, Any]] = []
    hs: list[np.ndarray] = []

    print(
        f"=== LIVE_DECISION_REP  tasks={len(TASK_SPECS)} "
        f"reps={N_REPS} styles={len(STYLES)} L{LAYER} ===",
        flush=True,
    )

    for style_id, style_text in STYLES:
        for rep in range(N_REPS):
            for ti, (tid, question, answer_path, _body) in enumerate(TASK_SPECS):
                seed = int(
                    SEED
                    + 40007 * rep
                    + 173 * ti
                    + 997 * (0 if style_id == "minimal_lean" else 1)
                )
                _seed_all(seed)
                registry = ToolRegistry(WORKSPACE)
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"{NEUTRAL}\n{style_text}\n\nTask: {question}"
                        ),
                    },
                ]
                files_seen: set[str] = set()
                tools_so_far: list[str] = []
                traj_id = f"{tid}_{style_id}_r{rep}"

                for turn in range(MAX_TURNS):
                    acts = last_token_residual(
                        loaded, messages, layers=[LAYER], cast_dtype=torch.float32
                    )
                    h = acts[LAYER].numpy().astype(np.float32)

                    _seed_all(seed + 1000 * (turn + 1))
                    asst = generate_assistant(
                        loaded,
                        messages,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    asst2, calls = dcol.recover_bare_tool_json(asst, known)
                    messages.append({"role": "assistant", "content": asst2})
                    if not calls:
                        calls = parse_tool_calls(asst2)

                    extra, next_tool, next_path = _label_extra(
                        calls=calls,
                        answer_path=answer_path,
                        files_seen=files_seen,
                    )
                    if extra is not None:
                        row = {
                            "task_id": tid,
                            "traj_id": traj_id,
                            "style": style_id,
                            "turn": turn,
                            "files_seen": sorted(files_seen),
                            "n_prev_calls": len(tools_so_far),
                            "tools_so_far": list(tools_so_far),
                            "next_tool": next_tool,
                            "next_path": next_path,
                            "extra": int(extra),
                            "answer_path": answer_path,
                            "prompt_chars": sum(len(m["content"]) for m in messages),
                            "h_index": len(hs),
                        }
                        decisions.append(row)
                        hs.append(h)

                    if not calls:
                        break

                    chunks = []
                    for call in calls:
                        name = str(call.get("name") or "")
                        args = call.get("arguments") or {}
                        tools_so_far.append(name)
                        p = args.get("path")
                        if isinstance(p, str) and p.strip():
                            files_seen.add(p.strip())
                        result = registry.execute(name, args)
                        chunks.append(
                            f"<tool_response>\n{result}\n</tool_response>"
                        )
                    messages.append({"role": "user", "content": "\n".join(chunks)})

                n_ex = sum(
                    1
                    for d in decisions
                    if d["traj_id"] == traj_id and d["extra"] == 1
                )
                n_min = sum(
                    1
                    for d in decisions
                    if d["traj_id"] == traj_id and d["extra"] == 0
                )
                print(
                    f"{style_id} rep={rep} {tid} extra={n_ex} minimal={n_min}",
                    flush=True,
                )

    H = np.stack(hs, axis=0) if hs else np.zeros((0, 1), dtype=np.float32)
    y = np.array([d["extra"] for d in decisions], dtype=np.int64)
    print(
        f"collected decisions={len(decisions)} extra={int(y.sum())} "
        f"minimal={int((y == 0).sum())}",
        flush=True,
    )

    n_min_tot = int((y == 0).sum())
    n_ex_tot = int(y.sum())
    if n_min_tot < 10 or n_ex_tot < 10:
        payload = {
            "protocol": "docs/live_decision_rep.md",
            "experiment": "LIVE_DECISION_REP",
            "decision": "REP_NULL",
            "reason": "CLASS_FLOOR",
            "n_decisions": len(decisions),
            "n_extra": n_ex_tot,
            "n_minimal": n_min_tot,
            "n_matched_pairs": 0,
            "steering": False,
            "decisions": decisions,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        MD.write_text(
            "# LIVE_DECISION_REP\n\n"
            f"- Decision: `REP_NULL` (CLASS_FLOOR)\n"
            f"- Decisions: {len(decisions)} "
            f"(extra={n_ex_tot}, minimal={n_min_tot})\n"
            "- Need ≥10 per class before probing.\n"
        )
        print(json.dumps({"decision": "REP_NULL", "reason": "CLASS_FLOOR"}, indent=2))
        print(f"wrote {OUT} {MD}", flush=True)
        return 0

    # Matched pairs: one per (task, turn, files_seen) with both labels
    by_key: dict[tuple, dict[str, list[int]]] = {}
    for i, d in enumerate(decisions):
        key = _state_key(d["task_id"], d["turn"], set(d["files_seen"]))
        bucket = by_key.setdefault(key, {"0": [], "1": []})
        bucket[str(d["extra"])].append(i)

    matched_pairs: list[tuple[int, int]] = []
    for _key, bucket in by_key.items():
        if bucket["0"] and bucket["1"]:
            matched_pairs.append((bucket["0"][0], bucket["1"][0]))

    n_pairs = len(matched_pairs)
    print(f"matched_pairs={n_pairs}", flush=True)

    # Task-held-out split on labeled decisions
    tasks = sorted({d["task_id"] for d in decisions})
    rng = np.random.default_rng(SEED)
    rng.shuffle(tasks)
    n_test = max(1, int(round(0.25 * len(tasks))))
    test_tasks = set(tasks[:n_test])
    train_tasks = set(tasks[n_test:])
    train_idx = np.array(
        [i for i, d in enumerate(decisions) if d["task_id"] in train_tasks],
        dtype=np.int64,
    )
    test_idx = np.array(
        [i for i, d in enumerate(decisions) if d["task_id"] in test_tasks],
        dtype=np.int64,
    )

    act_metrics = _fit_auc(H, y, train_idx, test_idx)

    # Permutation baseline on same split
    y_perm = y.copy()
    rng.shuffle(y_perm)
    perm_metrics = _fit_auc(H, y_perm, train_idx, test_idx)

    # Confound features available *before* the decision (not next_tool — that
    # partly defines the extra label by construction).
    tasks_all = sorted({d["task_id"] for d in decisions})
    task_ix = {t: i for i, t in enumerate(tasks_all)}

    def confound_matrix(kind: str) -> np.ndarray:
        rows = []
        for d in decisions:
            if kind == "turn":
                v = np.array([float(d["turn"])], dtype=np.float64)
            elif kind == "n_prev":
                v = np.array([float(d["n_prev_calls"])], dtype=np.float64)
            elif kind == "n_files":
                v = np.array([float(len(d["files_seen"]))], dtype=np.float64)
            elif kind == "task":
                v = np.zeros(len(tasks_all), dtype=np.float64)
                v[task_ix[d["task_id"]]] = 1.0
            elif kind == "all_conf":
                v = np.zeros(len(tasks_all) + 4, dtype=np.float64)
                v[task_ix[d["task_id"]]] = 1.0
                v[len(tasks_all)] = float(d["turn"])
                v[len(tasks_all) + 1] = float(d["n_prev_calls"])
                v[len(tasks_all) + 2] = float(len(d["files_seen"]))
                v[len(tasks_all) + 3] = float(d["prompt_chars"])
            else:
                raise ValueError(kind)
            rows.append(v)
        return np.stack(rows, axis=0)

    conf_metrics = {
        "turn": _fit_auc(confound_matrix("turn"), y, train_idx, test_idx),
        "n_prev": _fit_auc(confound_matrix("n_prev"), y, train_idx, test_idx),
        "n_files": _fit_auc(confound_matrix("n_files"), y, train_idx, test_idx),
        "task": _fit_auc(confound_matrix("task"), y, train_idx, test_idx),
        "all_conf": _fit_auc(confound_matrix("all_conf"), y, train_idx, test_idx),
    }

    # Activation + confounds
    X_both = np.concatenate([H.astype(np.float64), confound_matrix("all_conf")], axis=1)
    both_metrics = _fit_auc(X_both, y, train_idx, test_idx)

    act_auc = act_metrics["auc"]
    perm_auc = perm_metrics["auc"]
    best_conf = max(
        (
            conf_metrics[k]["auc"]
            for k in conf_metrics
            if conf_metrics[k]["auc"] == conf_metrics[k]["auc"]
        ),
        default=float("nan"),
    )

    if n_pairs < 12:
        pair_note = "MATCH_FLOOR"
    else:
        pair_note = "MATCH_OK"

    if (
        act_auc == act_auc
        and act_auc >= 0.65
        and (perm_auc != perm_auc or act_auc >= perm_auc + 0.10)
        and (best_conf != best_conf or act_auc >= best_conf + 0.05)
    ):
        decision = "REP_FOUND"
    elif act_auc == act_auc and act_auc >= 0.58 and (
        perm_auc != perm_auc or act_auc >= perm_auc + 0.05
    ):
        decision = "REP_WEAK"
    else:
        decision = "REP_NULL"

    direction_meta: dict[str, Any] | None = None
    if decision in {"REP_FOUND", "REP_WEAK"} and matched_pairs:
        # mean-diff from TRAIN pairs only
        train_pair_extra = []
        train_pair_min = []
        for i0, i1 in matched_pairs:
            # i0 minimal, i1 expand by construction
            if decisions[i0]["task_id"] not in train_tasks:
                continue
            if decisions[i0]["extra"] == 0 and decisions[i1]["extra"] == 1:
                train_pair_min.append(H[i0])
                train_pair_extra.append(H[i1])
            elif decisions[i0]["extra"] == 1 and decisions[i1]["extra"] == 0:
                train_pair_extra.append(H[i0])
                train_pair_min.append(H[i1])
        if train_pair_extra and train_pair_min:
            mu_e = np.mean(np.stack(train_pair_extra), axis=0)
            mu_m = np.mean(np.stack(train_pair_min), axis=0)
            d = mu_e - mu_m
            d = d / (np.linalg.norm(d) + 1e-8)
            # Held-out score: project H onto d, AUC
            scores = H @ d
            if len(np.unique(y[test_idx])) >= 2:
                d_auc = float(roc_auc_score(y[test_idx], scores[test_idx]))
            else:
                d_auc = float("nan")
            # Optional SVD on train paired deltas
            deltas = np.stack(train_pair_extra) - np.stack(train_pair_min)
            try:
                _, s, vt = np.linalg.svd(deltas - deltas.mean(0, keepdims=True), full_matrices=False)
                u1 = vt[0]
                u1 = u1 / (np.linalg.norm(u1) + 1e-8)
                # orient u1 like d
                if float(np.dot(u1, d)) < 0:
                    u1 = -u1
                u1_auc = (
                    float(roc_auc_score(y[test_idx], (H @ u1)[test_idx]))
                    if len(np.unique(y[test_idx])) >= 2
                    else float("nan")
                )
            except Exception:
                s = np.array([])
                u1 = None
                u1_auc = float("nan")

            direction_meta = {
                "n_train_pairs": len(train_pair_extra),
                "mean_diff_holdout_auc": d_auc,
                "u1_holdout_auc": u1_auc,
                "svd_singular_top3": [float(x) for x in s[:3]] if s.size else [],
            }
            DIR_OUT.parent.mkdir(parents=True, exist_ok=True)
            DIR_OUT.write_text(
                json.dumps(
                    {
                        "direction_id": "live_expand_d",
                        "kind": "live_decision_mean_diff",
                        "layer": LAYER,
                        "vector": d.astype(float).tolist(),
                        "meta": {
                            **direction_meta,
                            "decision": decision,
                            "claim": "live_expand_vs_minimal_not_u2",
                            "u1": u1.astype(float).tolist() if u1 is not None else None,
                        },
                    }
                )
                + "\n"
            )

    np.savez_compressed(
        CACHE,
        H=H,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    payload = {
        "protocol": "docs/live_decision_rep.md",
        "experiment": "LIVE_DECISION_REP",
        "layer": LAYER,
        "n_tasks": len(TASK_SPECS),
        "n_reps": N_REPS,
        "n_decisions": len(decisions),
        "n_extra": int(y.sum()),
        "n_minimal": int((y == 0).sum()),
        "n_matched_pairs": n_pairs,
        "match_status": pair_note,
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "activation_probe": act_metrics,
        "permutation_probe": perm_metrics,
        "confound_probes": conf_metrics,
        "activation_plus_confounds": both_metrics,
        "decision": decision,
        "direction": direction_meta,
        "direction_path": str(DIR_OUT) if direction_meta else None,
        "cache": str(CACHE),
        "steering": False,
        "decisions": decisions,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# LIVE_DECISION_REP",
        "",
        f"- Decision: `{decision}`",
        f"- N tasks: {len(TASK_SPECS)}  N matched pairs: {n_pairs} ({pair_note})",
        f"- Decisions: {len(decisions)} (extra={int(y.sum())}, minimal={int((y == 0).sum())})",
        "",
        f"- L{LAYER} last-token probe AUC={act_metrics['auc']:.3f} "
        f"bal_acc={act_metrics['bal_acc']:.3f}",
        f"- permutation AUC={perm_metrics['auc']:.3f}",
        f"- turn confound AUC={conf_metrics['turn']['auc']:.3f}",
        f"- n_prev confound AUC={conf_metrics['n_prev']['auc']:.3f}",
        f"- n_files confound AUC={conf_metrics['n_files']['auc']:.3f}",
        f"- task confound AUC={conf_metrics['task']['auc']:.3f}",
        f"- all confounds AUC={conf_metrics['all_conf']['auc']:.3f}",
        f"- activation+confounds AUC={both_metrics['auc']:.3f}",
        "",
    ]
    if direction_meta:
        lines += [
            f"- mean-diff holdout AUC={direction_meta['mean_diff_holdout_auc']:.3f}",
            f"- u1 holdout AUC={direction_meta['u1_holdout_auc']:.3f}",
            f"- wrote `{DIR_OUT.name}`",
            "",
        ]
    lines += ["No steering in this experiment."]
    MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "decision": decision,
                "n_pairs": n_pairs,
                "act_auc": act_auc,
                "perm_auc": perm_auc,
                "best_conf_auc": best_conf,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT} {MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
