"""Sampler-state layout for target-band partitioned continuation.

Target-band continuation keeps the low/probe sampler state on the uniform
target grid. Frames ``[0, head_t)`` (the protected prefix plus the band) hold
ordinary target-grid values. Every later frame stores its reduced-grid state in
the top-left ``source_h x source_w`` window of its target-sized frame; the rest
of that frame is masked padding excluded from transformer blocks. Elementwise
sampler updates preserve the stored values' spatial ownership. Stochastic raw
padding is masked before model calls and discarded from handoff views.

At the handoff the band is the overlap region in which the generated video has
two clean representations of the same frames: the band's own target-grid
prediction and the learned transfer of its reduced-grid projection. The
high-stage clean operand crossfades from the former to the latter across the
band, and every generated token is re-noised with the same independent noise.
"""

from __future__ import annotations

import torch

from .geometry import resize_spatial_5d, resize_spatial_5d_h3_patch_lattice
from .partitioned_stage import PartitionedTargetBandGeometry

TARGET_BAND_HANDOFF_POLICY = "target_band_clean_crossfade_shared_renoise_v1"


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


def target_band_source_view(video: torch.Tensor, geometry: PartitionedTargetBandGeometry) -> torch.Tensor:
    """Express a target-band tensor on the uniform reduced grid.

    Head frames are projected with H3's physical patch-lattice resample, the same
    operator used for the protected prefix's reduced-grid carrier; tail frames are
    their stored values.
    """
    _validate_video(video, geometry, "target-band video")
    head = resize_spatial_5d_h3_patch_lattice(video[:, :, : geometry.head_t], geometry.source_h, geometry.source_w)
    return torch.cat((head.to(video), target_band_tail(video, geometry)), dim=2)


def target_band_target_preview(video: torch.Tensor, geometry: PartitionedTargetBandGeometry) -> torch.Tensor:
    """Express a target-band tensor on the uniform target grid for previews."""
    _validate_video(video, geometry, "target-band video")
    tail = resize_spatial_5d(target_band_tail(video, geometry), geometry.target_h, geometry.target_w, mode="bicubic")
    return torch.cat((video[:, :, : geometry.head_t], tail.to(video)), dim=2)


def target_band_crossfade_weights(band_t: int) -> tuple[float, ...]:
    """Learned-transfer weight of each band token in the high-stage clean operand.

    Token ``j`` of an ``n``-token band receives ``j / n``: the token next to the
    protected prefix keeps its own target-grid prediction, and the weight rises
    in equal steps so that the first tail token (weight one) continues the ramp.
    """
    if type(band_t) is not int or band_t < 1:
        raise ValueError("target-band crossfade requires a positive integer band length")
    return tuple(index / band_t for index in range(band_t))


def blend_target_band_clean(
    native_band: torch.Tensor,
    transferred_band: torch.Tensor,
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Crossfade two clean representations of the same band frames.

    ``native_band`` is the band's target-grid prediction and ``transferred_band``
    the learned transfer of the same frames; both are BxCxTxHxW with ``T`` equal
    to ``len(weights)``. The result has ``transferred_band``'s dtype and device.
    """
    if native_band.ndim != 5 or native_band.shape != transferred_band.shape:
        raise ValueError("target-band crossfade requires matching BxCxTxHxW band tensors")
    if int(native_band.shape[2]) != len(weights):
        raise ValueError("target-band crossfade weights do not match the band length")
    if any(not 0.0 <= float(weight) < 1.0 for weight in weights):
        raise ValueError("target-band crossfade weights must lie in [0, 1)")
    native = native_band.to(device=transferred_band.device, dtype=torch.float32)
    transferred = transferred_band.to(torch.float32)
    weight = torch.tensor(weights, dtype=torch.float32, device=transferred_band.device).view(1, 1, -1, 1, 1)
    return (native + weight * (transferred - native)).to(transferred_band.dtype)


__all__ = [
    "TARGET_BAND_HANDOFF_POLICY",
    "blend_target_band_clean",
    "pack_target_band_video",
    "target_band_crossfade_weights",
    "target_band_padding_max_abs",
    "target_band_source_view",
    "target_band_tail",
    "target_band_target_preview",
]
