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
    temporal_weight: float = 0.20
    consistency_weight: float = 0.0
    cutoff: float = 0.25
    temporal_search_radius: int = 4
    temporal_min_similarity: float = 0.35
    temporal_min_margin: float = 0.02
    temporal_cycle_tolerance: float = 1.0
    schedule_power: float = 1.0
    transfer_mode: str = "bicubic"
    max_correction_rms_ratio: float = 0.5

    def __post_init__(self) -> None:
        modes = {"off", "direction", "direction+acceleration", "direction+temporal", "downsample_consistency"}
        if self.mode not in modes:
            raise ValueError(f"unsupported guidance mode {self.mode!r}")
        for name in ("direction_weight", "acceleration_weight", "temporal_weight", "consistency_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        if type(self.temporal_search_radius) is not int or self.temporal_search_radius < 1:
            raise ValueError("temporal_search_radius must be a positive integer")
        if (
            not math.isfinite(float(self.temporal_min_similarity))
            or not -1.0 <= self.temporal_min_similarity < 1.0
        ):
            raise ValueError("temporal_min_similarity must be finite and in [-1, 1)")
        if not math.isfinite(float(self.temporal_min_margin)) or self.temporal_min_margin <= 0:
            raise ValueError("temporal_min_margin must be finite and positive")
        if not math.isfinite(float(self.temporal_cycle_tolerance)) or self.temporal_cycle_tolerance < 0:
            raise ValueError("temporal_cycle_tolerance must be finite and non-negative")
        if self.schedule_power < 0 or not math.isfinite(self.schedule_power):
            raise ValueError("schedule_power must be finite and non-negative")
        if not math.isfinite(float(self.max_correction_rms_ratio)) or self.max_correction_rms_ratio <= 0:
            raise ValueError("max_correction_rms_ratio must be finite and positive")


@dataclass(slots=True)
class _TemporalCorrespondence:
    coordinate: float
    backward_flow: torch.Tensor
    forward_flow: torch.Tensor
    backward_confidence: torch.Tensor
    forward_confidence: torch.Tensor
    confidence_mean: float
    valid_fraction: float
    similarity_mean: float
    margin_mean: float
    flow_magnitude_mean: float
    flow_magnitude_max: float


@dataclass(slots=True)
class GuidanceState:
    start_coordinate: float | None = None
    current_coordinate: float | None = None
    current_high_velocity: torch.Tensor | None = None
    current_reference_velocity: torch.Tensor | None = None
    previous_coordinate: float | None = None
    previous_high_velocity: torch.Tensor | None = None
    previous_reference_velocity: torch.Tensor | None = None
    last_schedule: float | None = None
    last_correction_rms: float | None = None
    last_baseline_rms: float | None = None
    last_correction_rms_ratio: float | None = None
    last_clamp_scale: float | None = None
    last_direction_rms_ratio: float | None = None
    last_acceleration_rms_ratio: float | None = None
    last_acceleration_applied: bool = False
    last_same_coordinate_refinement: bool = False
    last_acceleration_anchor_coordinate: float | None = None
    temporal_cache: _TemporalCorrespondence | None = None
    last_temporal_rms_ratio: float | None = None
    last_temporal_confidence_mean: float | None = None
    last_temporal_valid_fraction: float | None = None
    last_temporal_disocclusion_fraction: float | None = None
    last_temporal_similarity_mean: float | None = None
    last_temporal_margin_mean: float | None = None
    last_temporal_flow_magnitude_mean: float | None = None
    last_temporal_flow_magnitude_max: float | None = None
    last_temporal_cache_hit: bool = False

    def reset(self) -> None:
        self.start_coordinate = None
        self.current_coordinate = None
        self.current_high_velocity = None
        self.current_reference_velocity = None
        self.previous_coordinate = None
        self.previous_high_velocity = None
        self.previous_reference_velocity = None
        self.last_schedule = None
        self.last_correction_rms = None
        self.last_baseline_rms = None
        self.last_correction_rms_ratio = None
        self.last_clamp_scale = None
        self.last_direction_rms_ratio = None
        self.last_acceleration_rms_ratio = None
        self.last_acceleration_applied = False
        self.last_same_coordinate_refinement = False
        self.last_acceleration_anchor_coordinate = None
        self.temporal_cache = None
        self.last_temporal_rms_ratio = None
        self.last_temporal_confidence_mean = None
        self.last_temporal_valid_fraction = None
        self.last_temporal_disocclusion_fraction = None
        self.last_temporal_similarity_mean = None
        self.last_temporal_margin_mean = None
        self.last_temporal_flow_magnitude_mean = None
        self.last_temporal_flow_magnitude_max = None
        self.last_temporal_cache_hit = False


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


def _rms_ratio(correction: torch.Tensor, reference: torch.Tensor) -> float:
    dims = tuple(range(1, correction.ndim))
    corr_rms = correction.float().square().mean(dim=dims, keepdim=True).sqrt()
    ref_rms = reference.float().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    return float((corr_rms / ref_rms).mean().detach().to(device="cpu", dtype=torch.float64).item())


def _local_correspondence(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    radius: int,
    min_similarity: float,
    min_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a local query->key displacement with conservative match confidence."""
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("temporal correspondence expects matching NxCxHxW tensors")
    normalized_query = F.normalize(query.float(), dim=1, eps=1e-6)
    normalized_key = F.normalize(key.float(), dim=1, eps=1e-6)
    batch, _, height, width = normalized_query.shape
    padded = F.pad(normalized_key, (radius, radius, radius, radius))
    best = torch.full((batch, height, width), -float("inf"), device=query.device)
    second = torch.full_like(best, -float("inf"))
    best_dx = torch.zeros((batch, height, width), device=query.device, dtype=torch.int64)
    best_dy = torch.zeros_like(best_dx)
    yy = torch.arange(height, device=query.device).view(1, height, 1)
    xx = torch.arange(width, device=query.device).view(1, 1, width)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[
                :,
                :,
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            score = (normalized_query * shifted).sum(dim=1)
            valid = (yy + dy >= 0) & (yy + dy < height) & (xx + dx >= 0) & (xx + dx < width)
            score = score.masked_fill(~valid, -float("inf"))
            better = score > best
            second = torch.where(better, best, torch.maximum(second, score))
            best = torch.where(better, score, best)
            best_dx = torch.where(better, torch.full_like(best_dx, dx), best_dx)
            best_dy = torch.where(better, torch.full_like(best_dy, dy), best_dy)

    margin = best - second
    similarity_confidence = ((best - min_similarity) / (1.0 - min_similarity)).clamp(0.0, 1.0)
    margin_confidence = (margin / min_margin).clamp(0.0, 1.0)
    confidence = similarity_confidence * margin_confidence
    flow = torch.stack((best_dx, best_dy), dim=1).float()
    return flow, confidence, best, margin


def _gather_at_integer_flow(value: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("integer-flow gather expects NxCxHxW values and Nx2xHxW flow")
    if value.shape[0] != flow.shape[0] or value.shape[-2:] != flow.shape[-2:]:
        raise ValueError("integer-flow gather geometry differs")
    batch, channels, height, width = value.shape
    yy = torch.arange(height, device=value.device).view(1, height, 1).expand(batch, height, width)
    xx = torch.arange(width, device=value.device).view(1, 1, width).expand(batch, height, width)
    px = xx + flow[:, 0].round().long()
    py = yy + flow[:, 1].round().long()
    if bool(((px < 0) | (px >= width) | (py < 0) | (py >= height)).any().item()):
        raise RuntimeError("temporal correspondence produced an out-of-bounds matched coordinate")
    index = (py * width + px).reshape(batch, 1, height * width).expand(batch, channels, height * width)
    return torch.gather(value.reshape(batch, channels, height * width), 2, index).reshape(
        batch, channels, height, width
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().mean().detach().to(device="cpu", dtype=torch.float64).item())


def _build_temporal_correspondence(
    reference: torch.Tensor,
    *,
    coordinate: float,
    config: GuidanceConfig,
) -> _TemporalCorrespondence | None:
    if reference.ndim != 5:
        raise ValueError("temporal correspondence expects BxCxTxHxW")
    batch, channels, frames, height, width = reference.shape
    if frames < 2:
        return None

    # Match a lightly smoothed clean-state latent. This keeps the matcher H3-native
    # and dependency-free while avoiding pixel/detail noise as the correspondence key.
    features = low_frequency_projection(reference, 0.5)
    left = features[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    right = features[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

    backward, backward_base, backward_similarity, backward_margin = _local_correspondence(
        right,
        left,
        radius=config.temporal_search_radius,
        min_similarity=config.temporal_min_similarity,
        min_margin=config.temporal_min_margin,
    )
    forward, forward_base, forward_similarity, forward_margin = _local_correspondence(
        left,
        right,
        radius=config.temporal_search_radius,
        min_similarity=config.temporal_min_similarity,
        min_margin=config.temporal_min_margin,
    )

    reverse_forward = _gather_at_integer_flow(forward, backward)
    reverse_forward_confidence = _gather_at_integer_flow(forward_base.unsqueeze(1), backward).squeeze(1)
    backward_cycle = (backward + reverse_forward).abs().amax(dim=1) <= config.temporal_cycle_tolerance
    backward_confidence = (
        (backward_base * reverse_forward_confidence).clamp_min(0.0).sqrt() * backward_cycle.float()
    )

    reverse_backward = _gather_at_integer_flow(backward, forward)
    reverse_backward_confidence = _gather_at_integer_flow(backward_base.unsqueeze(1), forward).squeeze(1)
    forward_cycle = (forward + reverse_backward).abs().amax(dim=1) <= config.temporal_cycle_tolerance
    forward_confidence = (
        (forward_base * reverse_backward_confidence).clamp_min(0.0).sqrt() * forward_cycle.float()
    )

    all_confidence = torch.cat((backward_confidence.reshape(-1), forward_confidence.reshape(-1)))
    valid = all_confidence > 0.0
    all_similarity = torch.cat((backward_similarity.reshape(-1), forward_similarity.reshape(-1)))
    all_margin = torch.cat((backward_margin.reshape(-1), forward_margin.reshape(-1)))
    all_flow = torch.cat(
        (
            backward.square().sum(dim=1).sqrt().reshape(-1),
            forward.square().sum(dim=1).sqrt().reshape(-1),
        )
    )
    confidence_mean = float(all_confidence.float().mean().detach().to(device="cpu", dtype=torch.float64).item())
    valid_fraction = float(valid.float().mean().detach().to(device="cpu", dtype=torch.float64).item())
    flow_magnitude_max = (
        float(all_flow[valid].max().detach().to(device="cpu", dtype=torch.float64).item()) if bool(valid.any().item()) else 0.0
    )

    pair_count = frames - 1
    return _TemporalCorrespondence(
        coordinate=float(coordinate),
        backward_flow=backward.reshape(batch, pair_count, 2, height, width).detach(),
        forward_flow=forward.reshape(batch, pair_count, 2, height, width).detach(),
        backward_confidence=backward_confidence.reshape(batch, pair_count, 1, height, width).detach(),
        forward_confidence=forward_confidence.reshape(batch, pair_count, 1, height, width).detach(),
        confidence_mean=confidence_mean,
        valid_fraction=valid_fraction,
        similarity_mean=_masked_mean(all_similarity, valid),
        margin_mean=_masked_mean(all_margin, valid),
        flow_magnitude_mean=_masked_mean(all_flow, valid),
        flow_magnitude_max=flow_magnitude_max,
    )


def _temporal_correspondence(
    reference: torch.Tensor,
    *,
    coordinate: float,
    config: GuidanceConfig,
    state: GuidanceState,
) -> tuple[_TemporalCorrespondence | None, bool]:
    cached = state.temporal_cache
    expected_pairs = max(reference.shape[2] - 1, 0)
    if (
        cached is not None
        and math.isclose(cached.coordinate, coordinate, rel_tol=0.0, abs_tol=1e-8)
        and cached.backward_flow.shape
        == (reference.shape[0], expected_pairs, 2, reference.shape[-2], reference.shape[-1])
        and cached.backward_flow.device == reference.device
    ):
        return cached, True
    built = _build_temporal_correspondence(reference, coordinate=coordinate, config=config)
    state.temporal_cache = built
    return built, False


def _resize_pairwise_flow(flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    batch, pairs, _, source_h, source_w = flow.shape
    work = flow.reshape(batch * pairs, 2, source_h, source_w).float()
    work = F.interpolate(work, size=(target_h, target_w), mode="bilinear", align_corners=False)
    work[:, 0].mul_(target_w / source_w)
    work[:, 1].mul_(target_h / source_h)
    return work.reshape(batch, pairs, 2, target_h, target_w)


def _resize_pairwise_confidence(confidence: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    batch, pairs, _, source_h, source_w = confidence.shape
    work = confidence.reshape(batch * pairs, 1, source_h, source_w).float()
    work = F.interpolate(work, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return work.reshape(batch, pairs, 1, target_h, target_w).clamp(0.0, 1.0)


def _warp_video_pairs(video: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5 or flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("temporal warp expects BxCxTxHxW video and BxTx2xHxW flow")
    batch, channels, pairs, height, width = video.shape
    if flow.shape != (batch, pairs, 2, height, width):
        raise ValueError("temporal warp flow geometry differs from video")
    frames = video.permute(0, 2, 1, 3, 4).reshape(batch * pairs, channels, height, width).float()
    offsets = flow.reshape(batch * pairs, 2, height, width).float()
    yy, xx = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=torch.float32),
        torch.arange(width, device=video.device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = ((xx.unsqueeze(0) + offsets[:, 0] + 0.5) * (2.0 / width)) - 1.0
    grid_y = ((yy.unsqueeze(0) + offsets[:, 1] + 0.5) * (2.0 / height)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (
        warped.reshape(batch, pairs, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .to(video)
    )


def _temporal_alignment_correction(
    high: torch.Tensor,
    reference: torch.Tensor,
    correspondence: _TemporalCorrespondence,
    *,
    transfer_mode: str,
) -> torch.Tensor:
    if high.ndim != 5 or reference.ndim != 5:
        raise ValueError("temporal alignment expects BxCxTxHxW tensors")
    if high.shape[:3] != reference.shape[:3]:
        raise ValueError("temporal alignment source/target batch-channel-time geometry differs")
    _, _, frames, target_h, target_w = high.shape
    if frames < 2:
        return torch.zeros_like(high)

    source_h, source_w = reference.shape[-2:]
    backward_source = correspondence.backward_flow
    forward_source = correspondence.forward_flow
    backward_target = _resize_pairwise_flow(backward_source, target_h, target_w)
    forward_target = _resize_pairwise_flow(forward_source, target_h, target_w)

    work_high = high.float()
    work_reference = reference.float()
    previous_reference = _warp_video_pairs(work_reference[:, :, :-1], backward_source)
    previous_innovation = work_reference[:, :, 1:] - previous_reference
    previous_target = _warp_video_pairs(work_high[:, :, :-1], backward_target)
    previous_target = previous_target + resize_video(
        previous_innovation, target_h, target_w, mode=transfer_mode
    ).float()
    previous_delta = previous_target - work_high[:, :, 1:]

    next_reference = _warp_video_pairs(work_reference[:, :, 1:], forward_source)
    next_innovation = work_reference[:, :, :-1] - next_reference
    next_target = _warp_video_pairs(work_high[:, :, 1:], forward_target)
    next_target = next_target + resize_video(next_innovation, target_h, target_w, mode=transfer_mode).float()
    next_delta = next_target - work_high[:, :, :-1]

    backward_confidence = _resize_pairwise_confidence(
        correspondence.backward_confidence, target_h, target_w
    ).permute(0, 2, 1, 3, 4)
    forward_confidence = _resize_pairwise_confidence(
        correspondence.forward_confidence, target_h, target_w
    ).permute(0, 2, 1, 3, 4)

    weighted = torch.zeros_like(work_high)
    weight = torch.zeros(
        (high.shape[0], 1, frames, target_h, target_w),
        device=high.device,
        dtype=torch.float32,
    )
    weighted[:, :, 1:] += backward_confidence * previous_delta
    weight[:, :, 1:] += backward_confidence
    weighted[:, :, :-1] += forward_confidence * next_delta
    weight[:, :, :-1] += forward_confidence

    # Below unit total confidence, keep the correction proportionally attenuated.
    # Above one, normalize the two-sided estimate instead of doubling its strength.
    correction = weighted / weight.clamp_min(1.0)
    return correction.to(high)


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
    high_state: torch.Tensor | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    if config.mode == "off":
        return high_x0
    acceleration_active = config.mode == "direction+acceleration" and config.acceleration_weight > 0
    temporal_active = config.mode == "direction+temporal" and config.temporal_weight > 0
    if acceleration_active and len(trustworthy_samples(run)) < 3:
        raise RuntimeError("acceleration guidance requires at least three exact trajectory anchors")
    coordinate = float(coordinate)
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise ValueError("guidance coordinate must be finite and inside [0, 1]")
    if acceleration_active:
        if high_state is None:
            raise ValueError("acceleration guidance requires the current high-grid sampler state")
        if high_state.shape != high_x0.shape:
            raise ValueError("acceleration sampler state and predicted-clean video shapes differ")
        if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
            raise ValueError("acceleration guidance requires a finite positive sampler sigma")
        high_state = high_state.to(device=high_x0.device, dtype=high_x0.dtype)
        sigma_value = float(sigma)
    else:
        sigma_value = None

    source_ref = time_matched_reference(run, coordinate).to(device=high_x0.device, dtype=high_x0.dtype)
    ref = resize_video(source_ref, high_x0.shape[-2], high_x0.shape[-1], mode=config.transfer_mode)
    if state.start_coordinate is None:
        state.start_coordinate = coordinate
    start = max(float(state.start_coordinate), 1e-8)
    schedule = min(1.0, max(0.0, coordinate / start)) ** config.schedule_power

    correction = torch.zeros_like(high_x0)
    if config.mode in {"direction", "direction+acceleration", "direction+temporal"}:
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

    direction_correction = correction
    temporal_correction = torch.zeros_like(high_x0)
    temporal_match = None
    temporal_cache_hit = False
    guided = high_x0 + direction_correction
    if temporal_active:
        temporal_match, temporal_cache_hit = _temporal_correspondence(
            source_ref,
            coordinate=coordinate,
            config=config,
            state=state,
        )
        if temporal_match is not None:
            temporal_delta = _temporal_alignment_correction(
                guided,
                source_ref,
                temporal_match,
                transfer_mode=config.transfer_mode,
            )
            temporal_correction = schedule * config.temporal_weight * temporal_delta
            guided = guided + temporal_correction
        correction = guided - high_x0
    else:
        state.temporal_cache = None

    acceleration_correction = torch.zeros_like(high_x0)
    reference_velocity = None
    same_coordinate_refinement = False
    acceleration_applied = False
    if acceleration_active:
        assert high_state is not None
        assert sigma_value is not None
        reference_velocity = (high_state - ref) / sigma_value
        high_velocity = (high_state - guided) / sigma_value

        if state.current_coordinate is not None:
            same_coordinate_refinement = math.isclose(
                coordinate,
                state.current_coordinate,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            if not same_coordinate_refinement:
                state.previous_coordinate = state.current_coordinate
                state.previous_high_velocity = state.current_high_velocity
                state.previous_reference_velocity = state.current_reference_velocity

        if (
            state.previous_coordinate is not None
            and state.previous_high_velocity is not None
            and state.previous_reference_velocity is not None
        ):
            acceleration_delta = (
                reference_velocity - state.previous_reference_velocity - high_velocity + state.previous_high_velocity
            )
            high_velocity = high_velocity + schedule * config.acceleration_weight * acceleration_delta
            acceleration_guided = high_state - sigma_value * high_velocity
            acceleration_correction = acceleration_guided - guided
            guided = acceleration_guided
            acceleration_applied = True
        correction = guided - high_x0

    correction, correction_stats = _bounded(correction, high_x0, config.max_correction_rms_ratio)
    result = high_x0 + correction
    state.last_schedule = float(schedule)
    state.last_correction_rms = correction_stats["correction_rms"]
    state.last_baseline_rms = correction_stats["baseline_rms"]
    state.last_correction_rms_ratio = correction_stats["correction_rms_ratio"]
    state.last_clamp_scale = correction_stats["clamp_scale"]
    state.last_direction_rms_ratio = _rms_ratio(direction_correction, high_x0)
    state.last_temporal_rms_ratio = _rms_ratio(temporal_correction, high_x0)
    state.last_acceleration_rms_ratio = _rms_ratio(acceleration_correction, high_x0)
    state.last_acceleration_applied = acceleration_applied
    state.last_same_coordinate_refinement = same_coordinate_refinement
    state.last_acceleration_anchor_coordinate = state.previous_coordinate if acceleration_applied else None
    state.last_temporal_cache_hit = temporal_cache_hit
    if temporal_match is None:
        state.last_temporal_confidence_mean = 0.0
        state.last_temporal_valid_fraction = 0.0
        state.last_temporal_disocclusion_fraction = 1.0 if temporal_active else 0.0
        state.last_temporal_similarity_mean = 0.0
        state.last_temporal_margin_mean = 0.0
        state.last_temporal_flow_magnitude_mean = 0.0
        state.last_temporal_flow_magnitude_max = 0.0
    else:
        state.last_temporal_confidence_mean = temporal_match.confidence_mean
        state.last_temporal_valid_fraction = temporal_match.valid_fraction
        state.last_temporal_disocclusion_fraction = 1.0 - temporal_match.valid_fraction
        state.last_temporal_similarity_mean = temporal_match.similarity_mean
        state.last_temporal_margin_mean = temporal_match.margin_mean
        state.last_temporal_flow_magnitude_mean = temporal_match.flow_magnitude_mean
        state.last_temporal_flow_magnitude_max = temporal_match.flow_magnitude_max

    if acceleration_active:
        assert high_state is not None
        assert sigma_value is not None
        assert reference_velocity is not None
        state.current_coordinate = float(coordinate)
        state.current_high_velocity = ((high_state - result) / sigma_value).detach()
        state.current_reference_velocity = reference_velocity.detach()
    else:
        state.current_coordinate = None
        state.current_high_velocity = None
        state.current_reference_velocity = None
        state.previous_coordinate = None
        state.previous_high_velocity = None
        state.previous_reference_velocity = None
    return result

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
