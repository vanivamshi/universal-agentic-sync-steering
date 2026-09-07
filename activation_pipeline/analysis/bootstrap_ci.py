"""Bootstrap CI helpers (percentile + BCa)."""

from __future__ import annotations

import math
from typing import Callable, Sequence


def percentile_ci(samples: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    xs = sorted(float(x) for x in samples)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    lo_i = int((alpha / 2) * n)
    hi_i = min(n - 1, int((1 - alpha / 2) * n))
    return xs[lo_i], xs[hi_i]


def _phi_inv(p: float) -> float:
    """Approx inverse CDF of standard normal (Acklam / Beasley-Springer style)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    # Rational approximation for central region
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bca_ci(
    theta_hat: float,
    boot_samples: Sequence[float],
    jackknife_samples: Sequence[float],
    alpha: float = 0.05,
) -> tuple[float, float, dict]:
    """Bias-corrected and accelerated (BCa) bootstrap CI.

    Returns (lo, hi, diagnostics).
    """
    boots = [float(x) for x in boot_samples]
    n_boot = len(boots)
    if n_boot < 10:
        lo, hi = percentile_ci(boots, alpha)
        return lo, hi, {"method": "percentile_fallback", "reason": "n_boot<10"}

    # Bias-correction factor
    prop = sum(1 for x in boots if x < theta_hat) / n_boot
    # Clamp to avoid ±inf at extremes
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = _phi_inv(prop)

    jacks = [float(x) for x in jackknife_samples]
    n_j = len(jacks)
    if n_j < 2:
        lo, hi = percentile_ci(boots, alpha)
        return lo, hi, {"method": "percentile_fallback", "reason": "n_jack<2", "z0": z0}

    j_bar = sum(jacks) / n_j
    num = sum((j_bar - j) ** 3 for j in jacks)
    den = sum((j_bar - j) ** 2 for j in jacks)
    if den <= 0:
        a = 0.0
    else:
        a = num / (6.0 * (den ** 1.5))

    def adj_alpha(a_level: float) -> float:
        z_a = _phi_inv(a_level)
        num_a = z0 + z_a
        den_a = 1.0 - a * num_a
        if abs(den_a) < 1e-12:
            return a_level
        return _phi(z0 + num_a / den_a)

    a_lo = adj_alpha(alpha / 2)
    a_hi = adj_alpha(1 - alpha / 2)
    boots_sorted = sorted(boots)
    lo_i = min(n_boot - 1, max(0, int(a_lo * n_boot)))
    hi_i = min(n_boot - 1, max(0, int(a_hi * n_boot)))
    return (
        boots_sorted[lo_i],
        boots_sorted[hi_i],
        {
            "method": "bca",
            "z0": z0,
            "acceleration": a,
            "a_lo": a_lo,
            "a_hi": a_hi,
            "prop_below_hat": prop,
        },
    )
