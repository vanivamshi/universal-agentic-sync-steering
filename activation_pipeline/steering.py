"""Activation steering: add α · d to residual stream at a chosen layer."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

from activation_pipeline.hooks import resolve_decoder_layers


def _apply_pos_mask(hidden: torch.Tensor, pos_mode: str) -> torch.Tensor:
    """Return a view used for stats / edits. ``pos_mode``: all | last."""
    if pos_mode == "all":
        return hidden
    if pos_mode == "last":
        return hidden[:, -1:, :]
    raise ValueError(f"unknown pos_mode={pos_mode!r}")


class ActivationSteerHook:
    """Add ``alpha * direction`` at ``layer``.

    ``pos_mode``: ``all`` (default) or ``last`` (final token only — matches
    last-token extraction under generate prefill / decode steps).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        direction: torch.Tensor,
        alpha: float,
        pos_mode: str = "all",
        collect_stats: bool = False,
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        d = direction.float().reshape(-1)
        n = torch.linalg.norm(d)
        if float(n) < 1e-12:
            raise ValueError("zero direction")
        self.direction = d / n
        self.alpha = float(alpha)
        self.pos_mode = str(pos_mode)
        self.collect_stats = bool(collect_stats)
        self.stats = {
            "n_fwd": 0,
            "n_tok": 0,
            "sum_abs_coord": 0.0,
            "sum_delta_norm": 0.0,
        }
        self._handle = None

    def reset_stats(self) -> None:
        self.stats = {
            "n_fwd": 0,
            "n_tok": 0,
            "sum_abs_coord": 0.0,
            "sum_delta_norm": 0.0,
        }

    def mean_stats(self) -> dict[str, float]:
        n = max(int(self.stats["n_tok"]), 1)
        return {
            "n_fwd": float(self.stats["n_fwd"]),
            "n_tok": float(self.stats["n_tok"]),
            "mean_abs_coord": self.stats["sum_abs_coord"] / n,
            "mean_delta_norm": self.stats["sum_delta_norm"] / n,
        }

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        delta = self.alpha * d
        if self.pos_mode == "all":
            if self.collect_stats:
                flat = hidden.reshape(-1, hidden.shape[-1])
                coord = flat @ d
                self.stats["n_fwd"] += 1
                self.stats["n_tok"] += int(flat.shape[0])
                self.stats["sum_abs_coord"] += float(coord.abs().sum().item())
                self.stats["sum_delta_norm"] += float(
                    torch.linalg.norm(delta) * flat.shape[0]
                )
            h = hidden + delta
        else:
            h = hidden.clone()
            slice_h = _apply_pos_mask(h, self.pos_mode)
            if self.collect_stats:
                flat = slice_h.reshape(-1, h.shape[-1])
                coord = flat @ d
                self.stats["n_fwd"] += 1
                self.stats["n_tok"] += int(flat.shape[0])
                self.stats["sum_abs_coord"] += float(coord.abs().sum().item())
                self.stats["sum_delta_norm"] += float(
                    torch.linalg.norm(delta) * flat.shape[0]
                )
            slice_h.add_(delta)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ActivationSubspaceAmpHook:
    """Amplify (or shrink) the projection onto an orthonormal subspace.

    ``basis`` is ``(k, hidden)``. At selected tokens::

        h ← h + α Q Qᵀ h

    Note: ``Q`` and ``−Q`` yield the same projector; sign of the SVD basis is
    invisible to this hook. Use ``ActivationSteerHook`` for oriented ±u tests.

    ``pos_mode``: ``all`` (default) or ``last``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        basis: torch.Tensor,
        alpha: float,
        pos_mode: str = "all",
        collect_stats: bool = False,
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.pos_mode = str(pos_mode)
        self.collect_stats = bool(collect_stats)
        B = basis.float()
        if B.ndim == 1:
            B = B.reshape(1, -1)
        if B.shape[0] < 1:
            raise ValueError("empty subspace basis")
        Qt, _ = torch.linalg.qr(B.T, mode="reduced")
        k = min(B.shape[0], Qt.shape[1])
        self.Q = Qt[:, :k].contiguous()
        self.stats = {
            "n_fwd": 0,
            "n_tok": 0,
            "sum_proj_norm": 0.0,
            "sum_delta_norm": 0.0,
        }
        self._handle = None

    def reset_stats(self) -> None:
        self.stats = {
            "n_fwd": 0,
            "n_tok": 0,
            "sum_proj_norm": 0.0,
            "sum_delta_norm": 0.0,
        }

    def mean_stats(self) -> dict[str, float]:
        n = max(int(self.stats["n_tok"]), 1)
        return {
            "n_fwd": float(self.stats["n_fwd"]),
            "n_tok": float(self.stats["n_tok"]),
            "mean_proj_norm": self.stats["sum_proj_norm"] / n,
            "mean_delta_norm": self.stats["sum_delta_norm"] / n,
        }

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Q = self.Q.to(device=hidden.device, dtype=hidden.dtype)

        def _edit(slice_h: torch.Tensor) -> torch.Tensor:
            coords = slice_h @ Q
            proj = coords @ Q.T
            delta = self.alpha * proj
            if self.collect_stats:
                flat_c = coords.reshape(-1, coords.shape[-1])
                flat_d = delta.reshape(-1, delta.shape[-1])
                self.stats["n_fwd"] += 1
                self.stats["n_tok"] += int(flat_c.shape[0])
                self.stats["sum_proj_norm"] += float(
                    torch.linalg.norm(flat_c, dim=-1).sum().item()
                )
                self.stats["sum_delta_norm"] += float(
                    torch.linalg.norm(flat_d, dim=-1).sum().item()
                )
            return slice_h + delta

        if self.pos_mode == "all":
            h = _edit(hidden)
        else:
            h = hidden.clone()
            slice_h = _apply_pos_mask(h, self.pos_mode)
            slice_h.copy_(_edit(slice_h))
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ActivationAblateHook:
    """Project residual onto the orthogonal complement of an orthonormal basis.

    ``basis`` is ``(k, hidden)`` — typically k persona-PCA components.
    At every token: ``h ← h − QQᵀ h``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        basis: torch.Tensor,
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        B = basis.float()
        if B.ndim == 1:
            B = B.reshape(1, -1)
        if B.shape[0] < 1:
            raise ValueError("empty ablation basis")
        # Orthonormalize rows → Q is (hidden, k)
        Qt, _ = torch.linalg.qr(B.T, mode="reduced")
        k = min(B.shape[0], Qt.shape[1])
        self.Q = Qt[:, :k].contiguous()
        self._handle = None

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Q = self.Q.to(device=hidden.device, dtype=hidden.dtype)
        proj = (hidden @ Q) @ Q.T
        h = hidden - proj
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ActivationForwardTraceHook:
    """No-op residual hook that records every forward at ``layer``.

    Distinguishes prefill (seq_len > 1) from decode (seq_len == 1 under
    ``generate`` with KV cache). Use to verify whether an activation intervention
    *would* fire and how many times — does not modify activations.
    """

    def __init__(self, model: nn.Module, *, layer: int, tag: str = "trace") -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        self.tag = str(tag)
        self.events: list[dict] = []
        self._handle = None
        self._n = 0

    def reset(self) -> None:
        self.events = []
        self._n = 0

    @property
    def n_fwd(self) -> int:
        return self._n

    @property
    def n_prefill(self) -> int:
        return sum(1 for e in self.events if e.get("phase") == "prefill")

    @property
    def n_decode(self) -> int:
        return sum(1 for e in self.events if e.get("phase") == "decode")

    def summary(self) -> dict:
        return {
            "tag": self.tag,
            "layer": self.layer,
            "n_fwd": self.n_fwd,
            "n_prefill": self.n_prefill,
            "n_decode": self.n_decode,
            "fired": self.n_fwd > 0,
        }

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._n += 1
        seq = int(hidden.shape[1]) if hidden.dim() >= 2 else 1
        # Under generate: first call is full prompt (prefill); later are 1-token decode.
        phase = "prefill" if seq > 1 else "decode"
        self.events.append(
            {
                "i": self._n,
                "phase": phase,
                "seq_len": seq,
                "hidden": int(hidden.shape[-1]),
            }
        )
        return output

    def register(self) -> None:
        self.remove()
        self.reset()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ActivationResidualPatchHook:
    """Replace last-token (or all-token) residual with a cached donor vector.

    ``blend=1`` is a full replace; ``0 < blend < 1`` is a convex mix toward the donor.
    Used for causal activation patching (normal → fault) at a chosen layer.

    ``prefill_only=True`` (recommended): apply on the first forward of a generate
    call (full-prompt prefill), then pass through on decode steps. Without this,
    full residual replace corrupts every generated token.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        donor: torch.Tensor,
        pos_mode: str = "last",
        blend: float = 1.0,
        prefill_only: bool = True,
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        d = donor.float().reshape(-1)
        if d.numel() < 1:
            raise ValueError("empty donor")
        self.donor = d
        self.pos_mode = str(pos_mode)
        b = float(blend)
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"blend must be in [0,1], got {blend}")
        self.blend = b
        self.prefill_only = bool(prefill_only)
        self._n_fwd = 0
        self._n_applied = 0
        self._handle = None

    def reset_stats(self) -> None:
        self._n_fwd = 0
        self._n_applied = 0

    def stats(self) -> dict:
        return {
            "n_fwd": self._n_fwd,
            "n_applied": self._n_applied,
            "prefill_only": self.prefill_only,
            "fired": self._n_applied > 0,
        }

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._n_fwd += 1
        seq = int(hidden.shape[1]) if hidden.dim() >= 2 else 1
        # Prefill = full prompt (seq>1). Decode under KV cache is seq==1.
        if self.prefill_only and seq <= 1:
            return output
        if self.prefill_only and self._n_applied >= 1:
            return output

        d = self.donor.to(device=hidden.device, dtype=hidden.dtype)
        if d.shape[-1] != hidden.shape[-1]:
            raise RuntimeError(
                f"donor dim {d.shape[-1]} != hidden {hidden.shape[-1]} at L{self.layer}"
            )
        h = hidden.clone()
        b = self.blend
        if self.pos_mode == "all":
            mixed = (1.0 - b) * h + b * d.view(1, 1, -1)
            h = mixed
        elif self.pos_mode == "last":
            h[:, -1, :] = (1.0 - b) * h[:, -1, :] + b * d
        else:
            raise ValueError(f"unknown pos_mode={self.pos_mode!r}")
        self._n_applied += 1
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self.reset_stats()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ActivationSubspaceCoordPatchHook:
    """Replace selected coordinates in an orthonormal subspace with a donor vector.

    ``basis`` is ``(k, hidden)`` (rows = directions). At every token::

        c = Qᵀ h
        c[idx] = Qᵀ h_donor  (same indices)
        h ← h + Q (c_new − c)

    Use to patch Assistant-Axis coords while holding the 31-dim persona span
    (or the reverse) — causal check beyond geometric orthogonality.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        basis: torch.Tensor,
        donor: torch.Tensor,
        coord_indices: list[int],
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        B = basis.float()
        if B.ndim == 1:
            B = B.reshape(1, -1)
        if B.shape[0] < 1:
            raise ValueError("empty patch basis")
        Qt, _ = torch.linalg.qr(B.T, mode="reduced")
        k = min(B.shape[0], Qt.shape[1])
        self.Q = Qt[:, :k].contiguous()  # (hidden, k)
        d = donor.float().reshape(-1)
        if d.numel() != self.Q.shape[0]:
            raise ValueError("donor dim mismatch")
        self.donor = d
        idx = sorted(set(int(i) for i in coord_indices))
        if not idx or idx[-1] >= k or idx[0] < 0:
            raise ValueError(f"bad coord_indices={coord_indices} for k={k}")
        self.idx = torch.tensor(idx, dtype=torch.long)
        self._handle = None

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Q = self.Q.to(device=hidden.device, dtype=hidden.dtype)
        donor = self.donor.to(device=hidden.device, dtype=hidden.dtype)
        idx = self.idx.to(device=hidden.device)
        # (batch, seq, k)
        c = hidden @ Q
        c_d = donor @ Q
        c_new = c.clone()
        c_new[..., idx] = c_d[idx]
        delta = (c_new - c) @ Q.T
        h = hidden + delta
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ChannelZeroHook:
    """Zero selected residual-stream channels at ``layer`` (all token positions)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        channel_indices: list[int],
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        if not channel_indices:
            raise ValueError("empty channel list")
        self.idx = torch.tensor(sorted(set(int(i) for i in channel_indices)), dtype=torch.long)
        self._handle = None

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h = hidden.clone()
        idx = self.idx.to(device=h.device)
        h[..., idx] = 0
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()


class ActivationDiagGainHook:
    """Diagonal gain in an orthonormal subspace (Exp B sync controller).

    ``Q`` is ``(hidden, k)``. At selected tokens::

        z = Qᵀ h
        Δz = d ⊙ (z − z_safe)
        h ← h + Q Δz

    ``pos_mode``: all | last.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        Q: torch.Tensor,
        gains: torch.Tensor,
        z_safe: torch.Tensor,
        pos_mode: str = "last",
    ) -> None:
        self.model = model
        self.layers = resolve_decoder_layers(model)
        self.layer = int(layer)
        self.pos_mode = str(pos_mode)
        self.Q = Q.float().contiguous()
        if self.Q.ndim != 2:
            raise ValueError("Q must be (hidden, k)")
        k = self.Q.shape[1]
        g = gains.float().reshape(-1)
        zs = z_safe.float().reshape(-1)
        if g.numel() != k or zs.numel() != k:
            raise ValueError(f"gains/z_safe length must be k={k}")
        self.gains = g.contiguous()
        self.z_safe = zs.contiguous()
        self._handle = None

    def _hook(self, _module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Q = self.Q.to(device=hidden.device, dtype=hidden.dtype)
        g = self.gains.to(device=hidden.device, dtype=hidden.dtype)
        zs = self.z_safe.to(device=hidden.device, dtype=hidden.dtype)

        def _edit(slice_h: torch.Tensor) -> torch.Tensor:
            z = slice_h @ Q
            delta_z = g * (z - zs)
            return slice_h + delta_z @ Q.T

        if self.pos_mode == "all":
            h = _edit(hidden)
        elif self.pos_mode == "last":
            h = hidden.clone()
            h[:, -1:, :] = _edit(h[:, -1:, :])
        else:
            raise ValueError(f"unknown pos_mode={self.pos_mode!r}")
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def register(self) -> None:
        self.remove()
        self._handle = self.layers[self.layer].register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.register()
        try:
            yield
        finally:
            self.remove()
