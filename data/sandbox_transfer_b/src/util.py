"""Shared helpers."""

MAGIC_CONST = 4242


def clamp(x: int) -> int:
    return max(0, min(x, MAGIC_CONST))
