from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F

from .contracts import TrajectoryRun, TrajectorySample
from .geometry import resize_video
from .sigma import interpolate_coordinate


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
    mode: str = "direction"
    direction_weight: float = 0.35
    acceleration_weight: float = 0.0
    consistency_weight: float = 0.0
    cutoff: float = 0.25
    schedule_power: float = 1.0
    transfer_mode: str = "bicubic"
    max_correction_rms_ratio: float = 0.5

    def __post_init__(self) -> None:
        modes = {"off", "direction", "direction+acceleration", "downsample_consistency"}
        if self.mode not in modes:
            raise ValueError(f"unsupported guidance mode {self.mode!r}")
        for name in ("direction_weight", "acceleration_weight", "consistency_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        if self.schedule_power < 0 or not math.isfinite(self.schedule_power):
            raise ValueError("schedule_power must be finite and non-negative")
        if not math.isfinite(float(self.max_correction_rms_ratio)) or self.max_correction_rms_ratio <= 0:
            raise ValueError("max_correction_rms_ratio must be finite and positive")


@dataclass(slots=True)
class GuidanceState:
    start_coordinate: float | None = None
    previous_coordinate: float | None = None
    previous_high_x0: torch.Tensor | None = None
    previous_reference: torch.Tensor | None = None
    last_schedule: float | None = None
    last_correction_rms: float | None = None
    last_baseline_rms: float | None = None
    last_correction_rms_ratio: float | None = None
    last_clamp_scale: float | None = None

    def reset(self) -> None:
        self.start_coordinate = None
        self.previous_coordinate = None
        self.previous_high_x0 = None
        self.previous_reference = None
        self.last_schedule = None
        self.last_correction_rms = None
        self.last_baseline_rms = None
        self.last_correction_rms_ratio = None
        self.last_clamp_scale = None


_PHASE_PRIORITY = {
    "handoff_probe": 4,
    "corrected": 3,
    "single": 2,
    "predicted": 1,
}


def trustworthy_samples(run: TrajectoryRun) -> tuple[TrajectorySample, ...]:
    if not run.complete:
        raise RuntimeError("incomplete trajectory runs cannot guide regeneration")
    exact = [sample for sample in run.samples if sample.provenance == "actual"]
    if not exact:
        raise RuntimeError("trajectory has no exact anchors")
    if any(not math.isfinite(float(sample.coordinate)) for sample in exact):
        raise RuntimeError("trajectory contains a non-finite exact coordinate")
    exact.sort(key=lambda sample: sample.coordinate, reverse=True)

    # PECE evaluates predicted and corrected states at the same sigma. Keep one
    # exact anchor per coordinate, preferring the corrected endpoint; a dedicated
    # handoff probe is the strongest anchor because it evaluates the actual
    # stopped low-stage state at the split coordinate.
    deduplicated: list[TrajectorySample] = []
    for sample in exact:
        if deduplicated and math.isclose(
            float(sample.coordinate),
            float(deduplicated[-1].coordinate),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            current = deduplicated[-1]
            if _PHASE_PRIORITY.get(sample.phase, 0) > _PHASE_PRIORITY.get(current.phase, 0):
                deduplicated[-1] = sample
            continue
        deduplicated.append(sample)
    return tuple(deduplicated)


def time_matched_reference(run: TrajectoryRun, coordinate: float) -> torch.Tensor:
    samples = trustworthy_samples(run)
    if coordinate >= samples[0].coordinate:
        return samples[0].video_x0
    if coordinate <= samples[-1].coordinate:
        return samples[-1].video_x0
    for left, right in pairwise(samples):
        if left.coordinate >= coordinate >= right.coordinate:
            weight = interpolate_coordinate(left.coordinate, right.coordinate, coordinate)
            return torch.lerp(left.video_x0, right.video_x0, weight)
    raise RuntimeError("trajectory interpolation support is inconsistent")


def low_frequency_projection(video: torch.Tensor, cutoff: float) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("low-frequency projection expects BxCxTxHxW")
    if cutoff >= 1.0:
        return video
    h, w = video.shape[-2:]
    radius = max(1, round(0.5 / cutoff))
    kernel = 2 * radius + 1
    work = video.permute(0, 2, 1, 3, 4).reshape(-1, video.shape[1], h, w).float()
    padded = F.pad(work, (radius, radius, radius, radius), mode="replicate")
    filtered = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    return filtered.reshape(video.shape[0], video.shape[2], video.shape[1], h, w).permute(0, 2, 1, 3, 4).to(video)


def _bounded(
    correction: torch.Tensor,
    reference: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    dims = tuple(range(1, correction.ndim))
    corr_rms = correction.float().square().mean(dim=dims, keepdim=True).sqrt()
    ref_rms = reference.float().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    scale = torch.clamp(ref_rms * ratio / corr_rms.clamp_min(1e-8), max=1.0)
    bounded_rms = corr_rms * scale
    summary = (
        torch.stack(
            (
                bounded_rms.mean(),
                ref_rms.mean(),
                (bounded_rms / ref_rms).mean(),
                scale.mean(),
            )
        )
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    correction_rms, baseline_rms, correction_rms_ratio, clamp_scale = map(float, summary.tolist())
    stats = {
        "correction_rms": correction_rms,
        "baseline_rms": baseline_rms,
        "correction_rms_ratio": correction_rms_ratio,
        "clamp_scale": clamp_scale,
    }
    return correction * scale.to(correction), stats


def apply_guidance(
    high_x0: torch.Tensor,
    *,
    run: TrajectoryRun,
    coordinate: float,
    config: GuidanceConfig,
    state: GuidanceState,
) -> torch.Tensor:
    if config.mode == "off":
        return high_x0
    if config.mode == "direction+acceleration" and config.acceleration_weight > 0 and len(trustworthy_samples(run)) < 3:
        raise RuntimeError("acceleration guidance requires at least three exact trajectory anchors")
    coordinate = float(coordinate)
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise ValueError("guidance coordinate must be finite and inside [0, 1]")
    source_ref = time_matched_reference(run, coordinate).to(device=high_x0.device, dtype=high_x0.dtype)
    ref = resize_video(source_ref, high_x0.shape[-2], high_x0.shape[-1], mode=config.transfer_mode)
    if state.start_coordinate is None:
        state.start_coordinate = coordinate
    start = max(float(state.start_coordinate), 1e-8)
    schedule = min(1.0, max(0.0, coordinate / start)) ** config.schedule_power
    correction = torch.zeros_like(high_x0)
    if config.mode in {"direction", "direction+acceleration"}:
        residual = low_frequency_projection(ref - high_x0, config.cutoff)
        correction = correction + schedule * config.direction_weight * residual
    if config.mode == "downsample_consistency":
        source_size = (run.geometry.latent_h, run.geometry.latent_w)
        work = high_x0.permute(0, 2, 1, 3, 4).reshape(-1, high_x0.shape[1], *high_x0.shape[-2:]).float()
        down = F.interpolate(work, size=source_size, mode="area")
        down = down.reshape(high_x0.shape[0], high_x0.shape[2], high_x0.shape[1], *source_size).permute(0, 2, 1, 3, 4)
        source_error = source_ref - down.to(source_ref)
        error = resize_video(source_error, *high_x0.shape[-2:], mode=config.transfer_mode)
        correction = correction + schedule * config.consistency_weight * error
    if (
        config.mode == "direction+acceleration"
        and config.acceleration_weight > 0
        and state.previous_coordinate is not None
        and state.previous_high_x0 is not None
        and state.previous_reference is not None
        and abs(coordinate - state.previous_coordinate) > 1e-8
    ):
        ref_delta = low_frequency_projection(ref - state.previous_reference, config.cutoff)
        high_delta = low_frequency_projection(high_x0 - state.previous_high_x0, config.cutoff)
        correction = correction + schedule * config.acceleration_weight * (ref_delta - high_delta)
    correction, correction_stats = _bounded(correction, high_x0, config.max_correction_rms_ratio)
    state.last_schedule = float(schedule)
    state.last_correction_rms = correction_stats["correction_rms"]
    state.last_baseline_rms = correction_stats["baseline_rms"]
    state.last_correction_rms_ratio = correction_stats["correction_rms_ratio"]
    state.last_clamp_scale = correction_stats["clamp_scale"]
    if config.mode == "direction+acceleration" and config.acceleration_weight > 0:
        state.previous_coordinate = float(coordinate)
        state.previous_high_x0 = high_x0.detach()
        state.previous_reference = ref.detach()
    else:
        state.previous_coordinate = None
        state.previous_high_x0 = None
        state.previous_reference = None
    return high_x0 + correction


def conditional_renoise_alignment(
    reference_x0: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    sigma: float,
    noise: torch.Tensor,
    transfer_mode: str = "bicubic",
) -> torch.Tensor:
    if not 0 <= sigma <= 1:
        raise ValueError("sigma must be in [0, 1]")
    ref = resize_video(reference_x0, target_h, target_w, mode=transfer_mode).to(noise)
    if ref.shape != noise.shape:
        raise ValueError("initialization reference and noise shapes differ")
    return (1.0 - sigma) * ref + sigma * noise
