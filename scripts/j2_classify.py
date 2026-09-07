"""J2 causal classification helper (shared; jailbreak surface script removed)."""

from __future__ import annotations

J2_SUGGESTIVE_DELTA = -0.15
J2_SUGGESTIVE_CI_HI = 0.10
J2_TOO_WIDE = 0.60


def classify_j2(delta: float, ci_lo: float, ci_hi: float) -> str:
    width = ci_hi - ci_lo
    if ci_hi < 0:
        return "PASS_CAUSAL"
    if ci_lo > 0:
        return "WRONG_SIGN"
    if delta <= J2_SUGGESTIVE_DELTA and ci_hi < J2_SUGGESTIVE_CI_HI:
        return "SUGGESTIVE"
    if width > J2_TOO_WIDE:
        return "TOO_WIDE_NULL"
    return "NULL_OTHER"
