"""Sampler-state layout for target-band partitioned continuation.

Target-band continuation keeps the low/probe sampler state on the uniform
target grid. Frames ``[0, head_t)`` (the protected prefix plus the band) hold
ordinary target-grid values. Every later frame stores its reduced-grid state in
the top-left ``source_h x source_w`` window of its target-sized frame; the rest
of that frame is masked padding excluded from transformer blocks. Elementwise
sampler updates preserve the stored values' spatial ownership. Stochastic raw
padding is masked before model calls and discarded from handoff views.

At the handoff the band keeps its own target-grid clean prediction, and every
generated token, band included, is re-noised with the same independent noise.
"""

from __future__ import annotations

import torch

from .geometry import H3_DENSE_PATCH_CENTER_LATTICE, resize_spatial_5d, resize_spatial_5d_h3_patch_lattice
from .partitioned_stage import PartitionedTargetBandGeometry

TARGET_BAND_HANDOFF_POLICY = "target_band_native_clean_shared_renoise_v1"


def _validate_video(video: torch.Tensor, geometry: PartitionedTargetBandGeometry, name: str) -> None:
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise TypeError(f"{name} must be a BxCxTxHxW tensor")
    if tuple(video.shape[2:]) != (geometry.temporal, geometry.target_h, geometry.target_w):
        raise ValueError(
            f"{name} geometry {tuple(video.shape[2:])} does not match the target-band layout "
            f"{(geometry.temporal, geometry.target_h, geometry.target_w)}"
        )


def pack_target_band_video(
    head: torch.Tensor,
    tail: torch.Tensor,
    geometry: PartitionedTargetBandGeometry,
) -> torch.Tensor:
    """Pack target-grid head frames and reduced-grid tail frames into one target-sized tensor."""
    if head.ndim != 5 or tail.ndim != 5 or head.shape[:2] != tail.shape[:2]:
        raise ValueError("target-band head and tail must be BxCxTxHxW tensors with equal batch/channels")
    if tuple(head.shape[2:]) != (geometry.head_t, geometry.target_h, geometry.target_w):
        raise ValueError("target-band head does not match the target-grid head geometry")
    if tuple(tail.shape[2:]) != (geometry.temporal - geometry.head_t, geometry.source_h, geometry.source_w):
        raise ValueError("target-band tail does not match the reduced-grid tail geometry")
    packed = head.new_zeros((*head.shape[:2], geometry.temporal, geometry.target_h, geometry.target_w))
    packed[:, :, : geometry.head_t] = head
    packed[:, :, geometry.head_t :, : geometry.source_h, : geometry.source_w] = tail.to(packed)
    return packed


def target_band_tail(video: torch.Tensor, geometry: PartitionedTargetBandGeometry) -> torch.Tensor:
    """Return the reduced-grid tail values stored in a target-band tensor."""
    _validate_video(video, geometry, "target-band video")
    return video[:, :, geometry.head_t :, : geometry.source_h, : geometry.source_w]


def target_band_padding_max_abs(video: torch.Tensor, geometry: PartitionedTargetBandGeometry) -> float:
    """Largest magnitude stored outside the tail windows (zero for a well-formed state)."""
    _validate_video(video, geometry, "target-band video")
    tail = video[:, :, geometry.head_t :]
    right = tail[:, :, :, : geometry.source_h, geometry.source_w :]
    below = tail[:, :, :, geometry.source_h :]
    values = [part.abs().amax() for part in (right, below) if part.numel()]
    if not values:
        return 0.0
    return float(torch.stack(values).amax().item())


def target_band_source_view(
    video: torch.Tensor,
    geometry: PartitionedTargetBandGeometry,
    *,
    lattice: str = H3_DENSE_PATCH_CENTER_LATTICE,
) -> torch.Tensor:
    """Express a target-band tensor on the uniform reduced grid.

    Head frames are projected with H3's physical patch-lattice resample, the same
    operator and lattice used for the protected prefix's reduced-grid carrier and
    the learned transfer; tail frames are their stored values.
    """
    _validate_video(video, geometry, "target-band video")
    head = resize_spatial_5d_h3_patch_lattice(
        video[:, :, : geometry.head_t], geometry.source_h, geometry.source_w, lattice=lattice
    )
    return torch.cat((head.to(video), target_band_tail(video, geometry)), dim=2)


def target_band_target_preview(video: torch.Tensor, geometry: PartitionedTargetBandGeometry) -> torch.Tensor:
    """Express a target-band tensor on the uniform target grid for previews."""
    _validate_video(video, geometry, "target-band video")
    tail = resize_spatial_5d(target_band_tail(video, geometry), geometry.target_h, geometry.target_w, mode="bicubic")
    return torch.cat((video[:, :, : geometry.head_t], tail.to(video)), dim=2)


__all__ = [
    "TARGET_BAND_HANDOFF_POLICY",
    "pack_target_band_video",
    "target_band_padding_max_abs",
    "target_band_source_view",
    "target_band_tail",
    "target_band_target_preview",
]
