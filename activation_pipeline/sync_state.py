"""Explicit controller state for sync-channel steering (Phase 9S+).

Before 9S the controller carried only the discrete bits ``S`` (plus the H plan
prefill) between steps. Everything that *produced* ``S`` -- which actuators
fired, with which polarity, in what order -- was discarded, so "retention" was
measured as an independent re-acquisition.

``SyncState`` keeps that history in two views:

* ``path``   -- ordered signed interventions, exactly as executed. Faithful even
               when a channel is hit repeatedly with different signs.
* ``active`` -- composed configuration ``{channel: sign}`` derived from ``path``
               (last applied sign per channel wins). This is the view that
               can be applied *simultaneously* in one episode.

Comparing replay of ``active`` against replay of ``path`` separates "the
accumulated configuration matters" from "the arrival sequence matters".

Stdlib only (no torch) so it can be imported and tested anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHANNELS = ("C", "H", "O")
CH_IDX = {"C": 0, "H": 1, "O": 2}

Bits = tuple[int, int, int]
SignedAct = tuple[str, int]


def act_name(channel: str, sign: int) -> str:
    """('C', -1) -> '-C'."""
    return f"{'+' if int(sign) > 0 else '-'}{channel}"


def parse_act(act: str) -> SignedAct:
    """'-C' -> ('C', -1)."""
    if len(act) != 2 or act[0] not in "+-" or act[1] not in CHANNELS:
        raise ValueError(f"bad signed action: {act!r}")
    return act[1], (+1 if act[0] == "+" else -1)


def target_sign(mstar, channel: str) -> int:
    """Target-conditioned polarity: +1 if m*_k == 1 else -1 (matches 8K `_s_star`)."""
    return 2 * int(mstar[CH_IDX[channel]]) - 1


def compose(path: list[SignedAct]) -> dict[str, int]:
    """Composed configuration from an ordered path: last sign per channel wins."""
    active: dict[str, int] = {}
    for ch, sign in path:
        if ch not in CH_IDX:
            raise ValueError(f"unknown channel {ch!r}")
        active[ch] = 1 if int(sign) > 0 else -1
    return active


@dataclass
class SyncState:
    """Discrete sync bits plus the intervention history that produced them."""

    S: Bits
    plan: str = ""
    path: list[SignedAct] = field(default_factory=list)
    traj: list[Bits] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.S = tuple(int(x) for x in self.S)  # type: ignore[assignment]
        if not self.traj:
            self.traj = [self.S]

    # --- updates -----------------------------------------------------------
    def record(self, channel: str, sign: int, S_after, plan_after: str | None = None) -> None:
        """Append one executed intervention and its observed outcome."""
        if channel not in CH_IDX:
            raise ValueError(f"unknown channel {channel!r}")
        sign = 1 if int(sign) > 0 else -1
        self.path.append((channel, sign))
        self.S = tuple(int(x) for x in S_after)  # type: ignore[assignment]
        self.traj.append(self.S)
        if plan_after:
            self.plan = plan_after

    # --- views -------------------------------------------------------------
    @property
    def active(self) -> dict[str, int]:
        return compose(self.path)

    def acts(self) -> list[str]:
        return [act_name(ch, s) for ch, s in self.path]

    def at(self, mstar) -> bool:
        return list(self.S) == [int(x) for x in mstar]

    def polarity_stats(self, mstar) -> dict[str, int]:
        """How many path steps used target vs opposite polarity."""
        n_t = sum(1 for ch, s in self.path if s == target_sign(mstar, ch))
        return {"n_steps": len(self.path), "n_target": n_t, "n_opposite": len(self.path) - n_t}

    def target_signed_active(self, mstar) -> dict[str, int]:
        """Same channels as ``active`` but every sign forced to target polarity."""
        return {ch: target_sign(mstar, ch) for ch in self.active}

    # --- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "S": list(self.S),
            "plan": self.plan,
            "path": [act_name(ch, s) for ch, s in self.path],
            "active": {ch: s for ch, s in self.active.items()},
            "traj": [list(s) for s in self.traj],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyncState":
        return cls(
            S=tuple(d["S"]),
            plan=d.get("plan") or "",
            path=[parse_act(a) for a in d.get("path") or []],
            traj=[tuple(s) for s in d.get("traj") or []],
        )

    @classmethod
    def from_trial_steps(cls, S0, plan0: str, steps: list[dict[str, Any]]) -> "SyncState":
        """Rebuild state from a 9R ``run_trial`` acquire-phase step log.

        Uses every executed acquire step (Γ' and beam), including no-op moves,
        because they are part of the arrival sequence as actually run.
        """
        st = cls(S=tuple(S0), plan=plan0 or "")
        for s in steps:
            if s.get("phase") != "acquire" or s.get("a") is None or s.get("empty_beam"):
                continue
            ch = s.get("channel") or s["a"][-1]
            st.record(ch, int(s["sign"]), s["S_after"])
        return st
