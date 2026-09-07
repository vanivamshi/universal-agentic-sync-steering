"""Benchmark domain registry."""
from __future__ import annotations

from .base import BenchmarkDomain, ForbiddenAction

__all__ = ["BenchmarkDomain", "ForbiddenAction", "get_domain", "register", "list_domains"]

_REGISTRY: dict[str, type[BenchmarkDomain]] = {}
_LOADED = False  # Track if we've done the full domain scan


def register(cls: type[BenchmarkDomain]) -> type[BenchmarkDomain]:
    """Class decorator — register a domain implementation."""
    _REGISTRY[cls().name] = cls
    return cls


def get_domain(name: str) -> BenchmarkDomain:
    """Instantiate a registered domain by name."""
    _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown domain '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_domains() -> list[str]:
    """Return names of all registered domains."""
    _ensure_loaded()
    return sorted(_REGISTRY)


def _ensure_loaded():
    """Import all domain sub-packages so their @register decorators fire."""
    global _LOADED
    if _LOADED:  # Changed: check if we've done the full scan, not just if registry has anything
        return
    import importlib
    import pkgutil
    from pathlib import Path

    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.ispkg:
            importlib.import_module(f".{info.name}", __package__)
    _LOADED = True
