"""Target-conditioned calibrated retention controller (frozen 9U gains) — no model, no torch.

After acquisition (S_0 = m*), steer every round of the SAME conversation with the
per-target gain multipliers selected in Phase 9U. Nothing here adapts, infers
drift, or tunes: the controller is a lookup from target -> frozen continuous action.

    action = retention_action(frozen, m*)          # every retention round
    run round k under `action`

The frozen config is written once from the 9U per-target JSONs (`freeze_from_9u`)
and records the SHA-256 of each source file, so a later edit to the 9U results
cannot silently change the evaluated gains (`load_frozen` re-verifies).

Also: Wilson intervals and the exact McNemar (sign) test used by Phase 9V.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from scripts import sync_adaptive_retention as sar

FROZEN_SCHEMA = "phase9v-frozen-gains/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_from_9u(sources: dict[str, Path], root: Path) -> dict[str, Any]:
    """Build the frozen-gains config from 9U per-target result files.

    `sources` maps target key ("100") -> path of sync_phase9u_retention_gains_t<key>.json.
    Only final 9U runs are accepted; the chosen scale is copied verbatim.
    """
    targets: dict[str, Any] = {}
    for tkey, path in sorted(sources.items()):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        assert p.get("final") is True, f"9U run for {tkey} is not final: {path}"
        assert sar.key(p["target"]) == tkey, (tkey, p["target"])
        scale = [float(x) for x in p["chosen_scale"]]
        act = sar.scaled_continuous(scale)
        assert act.gains() == {k: float(v) for k, v in p["chosen_gains"].items()}, (tkey, act.gains(), p["chosen_gains"])
        ev = p.get("eval", {})
        seeds_9u = sorted({int(t["seed"]) for arm in ev.get("trials", {}).values() for t in arm})
        targets[tkey] = {
            "scale": scale,
            "gains": act.gains(),
            "total_gain": sum(act.gains().values()),
            "source": str(Path(path).resolve().relative_to(root.resolve())),
            "source_sha256": _sha256(Path(path)),
            "seed_9u": int(p["seed"]),
            "eval_seeds_9u": seeds_9u,
        }
    return {"schema": FROZEN_SCHEMA, "base_gains": dict(sar.GAINS), "targets": targets}


def load_frozen(path: Path, root: Path, *, verify_sources: bool = True) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    assert cfg.get("schema") == FROZEN_SCHEMA, cfg.get("schema")
    assert cfg["base_gains"] == dict(sar.GAINS), "locked 8K gains changed since freeze"
    for tkey, t in cfg["targets"].items():
        act = sar.scaled_continuous(t["scale"])
        assert act.gains() == t["gains"], f"frozen gains inconsistent for {tkey}"
        if verify_sources:
            src = root / t["source"]
            got = _sha256(src)
            assert got == t["source_sha256"], f"9U source changed since freeze: {src}"
    return cfg


def retention_action(cfg: dict[str, Any], target: Sequence[int] | str) -> sar.Action:
    """Frozen calibrated continuous action for this target (forced FINAL:, target-sign)."""
    tkey = target if isinstance(target, str) else sar.key(target)
    if tkey not in cfg["targets"]:
        raise KeyError(f"no frozen 9U gains for target {tkey}; calibrate in a separate phase first")
    return sar.scaled_continuous(cfg["targets"][tkey]["scale"], name=f"CAL9U_{tkey}")


def retain(round_fn, msgs0, cfg, target, seeds) -> dict[str, Any]:
    """Same conversation; frozen action every round; a flip is recorded, never repaired."""
    action = retention_action(cfg, target)
    msgs, S = msgs0, list(target)
    states = [list(S)]
    for sd in seeds:
        msgs, S, _ = round_fn(msgs, action, list(target), sd)
        states.append(list(S))
    return {"states": states, "hold": all(s == list(target) for s in states[1:])}


# --------------------------------------------------------------------------- stats

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (B-only) and c (A-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
