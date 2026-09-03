from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import normalize_target_geometry, pack_streams, unpack_streams, validate_av
from .guidance import conditional_renoise_alignment
from .sigma import H3_VIDEO_SHIFT, normalized_coordinate


@dataclass(frozen=True, slots=True)
class ProgressiveHandoffConfig:
    target_latent_h: int | None = None
    target_latent_w: int | None = None
    target_scale: float | None = None
    handoff_coordinate: float = 0.35
    handoff_selection: str = "fixed"
    auto_min_coordinate: float = 0.2
    auto_max_coordinate: float = 0.55
    transfer_mode: str = "bicubic"
    matching_mode: str = "conditional_renoise"
    seed_offset: int = 0x4833464C4F57
    min_high_steps: int = 2

    def __post_init__(self) -> None:
        explicit = self.target_latent_h is not None or self.target_latent_w is not None
        if explicit == (self.target_scale is not None):
            raise ValueError("provide either target latent H/W or target scale")
        if explicit:
            if self.target_latent_h is None or self.target_latent_h < 2 or self.target_latent_h % 2:
                raise ValueError("target latent H must be positive and even")
            if self.target_latent_w is None or self.target_latent_w < 2 or self.target_latent_w % 2:
                raise ValueError("target latent W must be positive and even")
        elif not math.isfinite(float(self.target_scale)) or float(self.target_scale) <= 1.0:
            raise ValueError("target scale must be finite and greater than 1")
        if not 0 < self.handoff_coordinate < 1 or not math.isfinite(self.handoff_coordinate):
            raise ValueError("handoff coordinate must be finite and inside (0, 1)")
        if self.handoff_selection not in {"fixed", "auto_compute"}:
            raise ValueError("handoff selection must be fixed or auto_compute")
        if not 0 < self.auto_min_coordinate <= self.auto_max_coordinate < 1:
            raise ValueError("automatic handoff bounds must lie inside (0, 1)")
        if self.matching_mode != "conditional_renoise":
            raise ValueError("only the derived conditional_renoise handoff is currently supported")
        if self.min_high_steps < 1:
            raise ValueError("min_high_steps must be positive")

    def resolve_target(self, source_h: int, source_w: int) -> tuple[int, int]:
        if self.target_scale is not None:
            target_h, target_w = normalize_target_geometry(
                source_h=source_h,
                source_w=source_w,
                scale=self.target_scale,
                policy="nearest",
            )
        else:
            target_h, target_w = int(self.target_latent_h), int(self.target_latent_w)
        if target_h < int(source_h) or target_w < int(source_w):
            raise ValueError("progressive handoff target must not shrink either video axis")
        if target_h == int(source_h) and target_w == int(source_w):
            raise ValueError("progressive handoff target must increase at least one video axis")
        return target_h, target_w

    def resolve_coordinate(self, source_h: int, source_w: int, target_h: int, target_w: int) -> float:
        if self.handoff_selection == "fixed":
            return self.handoff_coordinate
        area_ratio = (target_h * target_w) / (source_h * source_w)
        estimate = self.handoff_coordinate / math.sqrt(area_ratio)
        return min(self.auto_max_coordinate, max(self.auto_min_coordinate, estimate))


@dataclass(frozen=True, slots=True)
class ProgressiveTargetInputConfig:
    """Run early denoising on a smaller grid while the workflow stays target-sized.

    This mode is designed for Continuum and other pipelines whose latent/session
    contract must remain on the final output grid. The wrapper derives a low-grid
    sampler invocation internally and returns to the original target grid at the
    handoff, so downstream spatial contracts never observe a geometry change.
    """

    source_latent_h: int | None = None
    source_latent_w: int | None = None
    source_scale: float | None = None
    handoff_coordinate: float = 0.35
    handoff_selection: str = "fixed"
    auto_min_coordinate: float = 0.2
    auto_max_coordinate: float = 0.55
    transfer_mode: str = "bicubic"
    matching_mode: str = "conditional_renoise"
    seed_offset: int = 0x4833464C4F57
    source_noise_offset: int = 0x48334C4F574C52
    min_high_steps: int = 2

    def __post_init__(self) -> None:
        explicit = self.source_latent_h is not None or self.source_latent_w is not None
        if explicit == (self.source_scale is not None):
            raise ValueError("provide either source latent H/W or source scale")
        if explicit:
            if self.source_latent_h is None or self.source_latent_h < 2 or self.source_latent_h % 2:
                raise ValueError("source latent H must be positive and even")
            if self.source_latent_w is None or self.source_latent_w < 2 or self.source_latent_w % 2:
                raise ValueError("source latent W must be positive and even")
        elif not math.isfinite(float(self.source_scale)) or not 0.0 < float(self.source_scale) < 1.0:
            raise ValueError("source scale must be finite and inside (0, 1)")
        if not 0 < self.handoff_coordinate < 1 or not math.isfinite(self.handoff_coordinate):
            raise ValueError("handoff coordinate must be finite and inside (0, 1)")
        if self.handoff_selection not in {"fixed", "auto_compute"}:
            raise ValueError("handoff selection must be fixed or auto_compute")
        if not 0 < self.auto_min_coordinate <= self.auto_max_coordinate < 1:
            raise ValueError("automatic handoff bounds must lie inside (0, 1)")
        if self.matching_mode != "conditional_renoise":
            raise ValueError("only the derived conditional_renoise handoff is currently supported")
        if self.min_high_steps < 1:
            raise ValueError("min_high_steps must be positive")

    def resolve_source(self, target_h: int, target_w: int) -> tuple[int, int]:
        if self.source_scale is not None:
            source_h, source_w = normalize_target_geometry(
                source_h=target_h,
                source_w=target_w,
                scale=self.source_scale,
                policy="nearest",
            )
        else:
            source_h, source_w = int(self.source_latent_h), int(self.source_latent_w)
        if source_h > int(target_h) or source_w > int(target_w):
            raise ValueError("target-input progressive source must not exceed either target video axis")
        if source_h == int(target_h) and source_w == int(target_w):
            raise ValueError("target-input progressive source must reduce at least one video axis")
        return source_h, source_w

    def resolve_coordinate(self, source_h: int, source_w: int, target_h: int, target_w: int) -> float:
        if self.handoff_selection == "fixed":
            return self.handoff_coordinate
        area_ratio = (target_h * target_w) / (source_h * source_w)
        estimate = self.handoff_coordinate / math.sqrt(area_ratio)
        return min(self.auto_max_coordinate, max(self.auto_min_coordinate, estimate))


def select_handoff_index(
    sigmas: torch.Tensor,
    coordinate: float,
    *,
    min_high_steps: int = 2,
    video_shift: float = H3_VIDEO_SHIFT,
) -> int:
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise ValueError("progressive handoff requires at least three sampling intervals")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("progressive handoff requires finite sigma values")
    if not math.isfinite(float(coordinate)) or not 0.0 < float(coordinate) < 1.0:
        raise ValueError("progressive handoff coordinate must be finite and inside (0, 1)")
    if not math.isfinite(float(video_shift)) or float(video_shift) <= 0.0:
        raise ValueError("progressive handoff video shift must be finite and positive")
    if bool((sigmas[1:] >= sigmas[:-1]).any()):
        raise ValueError("progressive handoff requires a strictly descending sigma schedule")
    candidates = torch.arange(1, sigmas.numel() - min_high_steps, device=sigmas.device)
    if candidates.numel() == 0:
        raise ValueError("sigma schedule leaves no valid handoff interval")
    base_coordinates = normalized_coordinate(sigmas[candidates], video_shift=video_shift)
    distances = (base_coordinates - float(coordinate)).abs()
    return int(candidates[int(distances.argmin())].item())


def deterministic_video_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").to(device=device, dtype=dtype)


def build_handoff_state(
    *,
    source_packed_state: torch.Tensor,
    source_x0_packed: torch.Tensor,
    source_shapes: list[tuple[int, ...]],
    sigma: float,
    target_h: int,
    target_w: int,
    seed: int,
    transfer_mode: str = "bicubic",
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    if len(source_shapes) != 2:
        raise ValueError("progressive H3 handoff requires exactly video and audio streams")
    if not 0 < sigma < 1:
        raise ValueError("handoff sigma must be strictly inside (0, 1)")
    source_video, source_audio = unpack_streams(source_packed_state, source_shapes)
    x0_video, _x0_audio = unpack_streams(source_x0_packed, source_shapes)
    validate_av(source_video, source_audio)
    if x0_video.shape != source_video.shape:
        raise ValueError("source x0 video geometry does not match the handoff state")
    if (target_h, target_w) == tuple(source_video.shape[-2:]):
        return source_packed_state.clone(), list(source_shapes)
    noise = deterministic_video_noise(
        (source_video.shape[0], source_video.shape[1], source_video.shape[2], target_h, target_w),
        seed=seed,
        device=source_video.device,
        dtype=source_video.dtype,
    )
    target_video = conditional_renoise_alignment(
        x0_video,
        target_h=target_h,
        target_w=target_w,
        sigma=float(sigma),
        noise=noise,
        transfer_mode=transfer_mode,
    )
    # The packed sampler carries audio on the video sigma schedule. Its state is
    # preserved byte-for-byte across a purely spatial video transition.
    target_packed, target_shapes = pack_streams((target_video, source_audio.clone()))
    return target_packed, target_shapes
