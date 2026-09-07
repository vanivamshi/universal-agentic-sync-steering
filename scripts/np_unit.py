"""Tiny shared vector helpers (avoid importing heavy experiment scripts)."""

from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-12)
