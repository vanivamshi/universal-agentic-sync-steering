"""Analysis helpers: sensitivity / plateau metrics + subspace compare + logit lens."""

from .logit_lens import (
    LogitLensGateResult,
    WindowEntropy,
    collect_window_entropies,
    evaluate_logit_lens_gate,
)
from .sensitivity import (
    LayerPerturbHooks,
    SensitivityResult,
    blowup_vs_epsilon_curve,
    default_epsilon_grid,
    directional_sensitivity,
    plateau_depth,
    residual_l2_blowup,
    top_k_sensitive_directions,
)
from .subspace import SubspaceCompareResult, compare_prose_tool_subspaces, orthogonal_procrustes

__all__ = [
    "LayerPerturbHooks",
    "LogitLensGateResult",
    "SensitivityResult",
    "SubspaceCompareResult",
    "WindowEntropy",
    "blowup_vs_epsilon_curve",
    "default_epsilon_grid",
    "collect_window_entropies",
    "compare_prose_tool_subspaces",
    "directional_sensitivity",
    "evaluate_logit_lens_gate",
    "orthogonal_procrustes",
    "plateau_depth",
    "residual_l2_blowup",
    "top_k_sensitive_directions",
]
