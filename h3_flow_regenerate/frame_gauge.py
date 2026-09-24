from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

FRAME_GAUGE_POLICY_VERSION = "paired_prefix_rigid_v1"


@dataclass(frozen=True, slots=True)
class FrameGaugePolicy:
    min_prefix_frames: int = 4
    max_prefix_frames: int = 6
    min_valid_channels: int = 8
    margin: int = 5
    max_support_axis: int = 96
    integer_radius: int = 3
    accepted_bound: float = 2.0
    identity_bound: float = 0.0625
    min_ncc: float = 0.75
    min_rms_improvement: float = 0.15
    min_runner_margin: float = 0.05
    consistency_tolerance: float = 0.25
    conflict_tolerance: float = 0.5
    max_invalid_fraction: float = 0.08


DEFAULT_POLICY = FrameGaugePolicy()


@dataclass(frozen=True, slots=True)
class FrameGaugeEstimate:
    status: str
    reason: str
    dx: float = 0.0
    dy: float = 0.0
    metrics: dict[str, Any] | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def identity(self) -> bool:
        return self.status == "identity"

    @property
    def rejected(self) -> bool:
        return self.status == "rejected"

    def telemetry(self) -> dict[str, Any]:
        fields = dict(self.metrics or {})
        fields.update(
            status=self.status,
            reason=self.reason,
            dx=float(self.dx),
            dy=float(self.dy),
            units="target_latent_cells",
            sign_convention="W_d Z(y,x)=Z(y-dy,x-dx); positive dx right, positive dy down",
            policy_version=FRAME_GAUGE_POLICY_VERSION,
        )
        return fields


@dataclass(frozen=True, slots=True)
class TranslationApplication:
    video: torch.Tensor
    valid_mask: torch.Tensor
    invalid_fraction: float
    transformed_frames: int


@dataclass(slots=True)
class _Prepared:
    learned: torch.Tensor
    exact: torch.Tensor
    valid_channels: torch.Tensor
    frame_indices: tuple[int, ...]
    fit: tuple[int, ...]
    validation: tuple[int, ...]
    ys: torch.Tensor
    xs: torch.Tensor
    stride: int
    gradient_rms: float
    height: int
    width: int


def _validate_pair(learned: torch.Tensor, exact: torch.Tensor) -> tuple[int, int, int, int, int]:
    if not torch.is_tensor(learned) or not torch.is_tensor(exact):
        raise TypeError("frame-gauge registration requires tensor inputs")
    if learned.ndim != 5 or exact.ndim != 5 or tuple(learned.shape) != tuple(exact.shape):
        raise ValueError("frame-gauge registration requires matching BxCxTxHxW tensors")
    if not learned.is_floating_point() or not exact.is_floating_point():
        raise TypeError("frame-gauge registration requires floating-point tensors")
    if learned.shape[0] != 1 or learned.shape[1] != 24:
        raise ValueError("frame-gauge v1 requires B=1 and C=24")
    if not bool(torch.isfinite(learned).all().item()) or not bool(torch.isfinite(exact).all().item()):
        raise ValueError("frame-gauge registration requires finite tensors")
    return tuple(int(value) for value in learned.shape)


def _prepare(
    learned: torch.Tensor,
    exact: torch.Tensor,
    policy: FrameGaugePolicy,
) -> _Prepared | str:
    _, _, prefix_t, height, width = _validate_pair(learned, exact)
    if prefix_t < policy.min_prefix_frames:
        return "insufficient_prefix_support"
    count = min(prefix_t, policy.max_prefix_frames)
    frame_indices = tuple(range(prefix_t - count, prefix_t))
    positions = tuple(range(count))
    validation = tuple(index for index in positions if index % 2 == 1)
    if positions[-1] not in validation:
        validation = tuple(sorted((*validation, positions[-1])))
    fit = tuple(index for index in positions if index not in validation)
    if len(fit) < 2 or len(validation) < 2:
        return "insufficient_fit_validation_support"

    stride = max(1, math.ceil(max(height, width) / policy.max_support_axis))
    ys = torch.arange(policy.margin, height - policy.margin, stride, dtype=torch.int64)
    xs = torch.arange(policy.margin, width - policy.margin, stride, dtype=torch.int64)
    if ys.numel() < 16 or xs.numel() < 16:
        return "insufficient_spatial_support"

    learned_work = learned[0, :, frame_indices].permute(1, 0, 2, 3).detach().float()
    exact_work = exact[0, :, frame_indices].permute(1, 0, 2, 3).detach().float()

    def feature(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames, channels, h, w = value.shape
        work = value.reshape(frames * channels, 1, h, w)
        work = F.avg_pool2d(
            F.pad(work, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        ).reshape(frames, channels, h, w)
        centered = work - work.mean(dim=(-2, -1), keepdim=True)
        rms = centered.square().mean(dim=(-2, -1)).sqrt()
        return centered / rms.clamp_min(1e-12)[..., None, None], rms

    learned_feature, learned_rms = feature(learned_work)
    exact_feature, exact_rms = feature(exact_work)
    learned_channel_rms = learned_rms.mean(dim=0)
    exact_channel_rms = exact_rms.mean(dim=0)
    learned_positive = learned_channel_rms[learned_channel_rms > 0]
    exact_positive = exact_channel_rms[exact_channel_rms > 0]
    if learned_positive.numel() == 0 or exact_positive.numel() == 0:
        return "ambiguous_low_energy"
    learned_threshold = max(
        1e-6,
        0.01 * float(torch.median(learned_positive).item()),
    )
    exact_threshold = max(
        1e-6,
        0.01 * float(torch.median(exact_positive).item()),
    )
    valid_channels = (learned_channel_rms > learned_threshold) & (exact_channel_rms > exact_threshold)
    if int(valid_channels.sum().item()) < policy.min_valid_channels:
        return "insufficient_valid_channels"

    combined = torch.cat(
        (learned_feature[:, valid_channels], exact_feature[:, valid_channels]),
        dim=0,
    )
    gradients = []
    if width > 1:
        gradients.append((combined[..., 1:] - combined[..., :-1]).square().mean())
    if height > 1:
        gradients.append((combined[..., 1:, :] - combined[..., :-1, :]).square().mean())
    gradient_rms = float(torch.stack(gradients).mean().sqrt().item()) if gradients else 0.0
    if not math.isfinite(gradient_rms) or gradient_rms <= 1e-6:
        return "ambiguous_low_gradient"

    return _Prepared(
        learned=learned_feature.cpu().to(torch.float64),
        exact=exact_feature.cpu().to(torch.float64),
        valid_channels=valid_channels.cpu(),
        frame_indices=frame_indices,
        fit=fit,
        validation=validation,
        ys=ys,
        xs=xs,
        stride=stride,
        gradient_rms=gradient_rms,
        height=height,
        width=width,
    )


def _coords(
    prepared: _Prepared,
    *,
    region: tuple[int, int, int, int] | None = None,
    parity: tuple[int, int] | None = None,
    min_axis: int = 8,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if parity is None:
        ys = prepared.ys
        xs = prepared.xs
    else:
        # The primary bounded support may use an even stride on large grids.
        # Construct each parity support independently so all four H3 latent
        # phases remain testable rather than inheriting one sampled parity.
        py, px = parity
        margin = int(prepared.ys[0].item())
        parity_stride = max(2, 2 * int(prepared.stride))
        y_start = margin + ((int(py) - margin) % 2)
        x_start = margin + ((int(px) - margin) % 2)
        ys = torch.arange(
            y_start,
            prepared.height - margin,
            parity_stride,
            dtype=torch.int64,
        )
        xs = torch.arange(
            x_start,
            prepared.width - margin,
            parity_stride,
            dtype=torch.int64,
        )
    if region is not None:
        y0, y1, x0, x1 = region
        ys = ys[(ys >= y0) & (ys < y1)]
        xs = xs[(xs >= x0) & (xs < x1)]
    if ys.numel() < min_axis or xs.numel() < min_axis:
        return None
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1).to(torch.float64), xx.reshape(-1).to(torch.float64)


def _gather_exact(value: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
    flat = value.reshape(value.shape[0], value.shape[1], -1)
    index = yy.to(torch.int64) * value.shape[-1] + xx.to(torch.int64)
    index = index.view(1, 1, -1).expand(value.shape[0], value.shape[1], -1)
    return torch.gather(flat, 2, index)


def _sample(
    value: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    sy = yy - float(dy)
    sx = xx - float(dx)
    y0 = torch.floor(sy).to(torch.int64)
    x0 = torch.floor(sx).to(torch.int64)
    y1 = (y0 + 1).clamp(max=value.shape[-2] - 1)
    x1 = (x0 + 1).clamp(max=value.shape[-1] - 1)
    wy = (sy - y0.to(torch.float64)).view(1, 1, -1)
    wx = (sx - x0.to(torch.float64)).view(1, 1, -1)
    flat = value.reshape(value.shape[0], value.shape[1], -1)
    width = value.shape[-1]

    def gather(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        index = (y * width + x).view(1, 1, -1).expand(value.shape[0], value.shape[1], -1)
        return torch.gather(flat, 2, index)

    top = gather(y0, x0) * (1.0 - wx) + gather(y0, x1) * wx
    bottom = gather(y1, x0) * (1.0 - wx) + gather(y1, x1) * wx
    return top * (1.0 - wy) + bottom * wy


def _metrics(
    prepared: _Prepared,
    frames: tuple[int, ...],
    coords: tuple[torch.Tensor, torch.Tensor],
    dx: float,
    dy: float,
) -> tuple[float, float, float]:
    yy, xx = coords
    channels = prepared.valid_channels
    learned = prepared.learned[list(frames)][:, channels]
    exact = prepared.exact[list(frames)][:, channels]
    shifted = _sample(learned, yy, xx, dx, dy)
    target = _gather_exact(exact, yy, xx)
    residual = shifted - target
    absolute = residual.abs()
    huber = torch.where(absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5)
    rms = float(residual.square().mean().sqrt().item())

    shifted_centered = shifted - shifted.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    denominator = shifted_centered.square().sum(dim=-1).sqrt() * target_centered.square().sum(dim=-1).sqrt()
    numerator = (shifted_centered * target_centered).sum(dim=-1)
    valid = denominator > 1e-12
    ncc = float((numerator[valid] / denominator[valid]).mean().item()) if bool(valid.any().item()) else -1.0
    return float(huber.mean().item()), rms, ncc


def _candidate_key(candidate: tuple[float, float, float]) -> tuple[float, float, float, float]:
    dx, dy, loss = candidate
    return loss, dx * dx + dy * dy, dy, dx


def _search(
    prepared: _Prepared,
    frames: tuple[int, ...],
    coords: tuple[torch.Tensor, torch.Tensor],
    policy: FrameGaugePolicy,
) -> tuple[tuple[float, float, float], tuple[float, float, float] | None, float, bool]:
    evaluated: dict[tuple[float, float], tuple[float, float, float]] = {}

    def evaluate(dx: float, dy: float) -> tuple[float, float, float]:
        key = round(float(dx), 8), round(float(dy), 8)
        if key not in evaluated:
            loss, _, _ = _metrics(prepared, frames, coords, key[0], key[1])
            evaluated[key] = key[0], key[1], loss
        return evaluated[key]

    integers = [
        evaluate(dx, dy)
        for dy in range(-policy.integer_radius, policy.integer_radius + 1)
        for dx in range(-policy.integer_radius, policy.integer_radius + 1)
    ]
    integer_best = min(integers, key=_candidate_key)
    saturated = abs(integer_best[0]) == policy.integer_radius or abs(integer_best[1]) == policy.integer_radius

    coarse = [
        evaluate(integer_best[0] + 0.25 * ix, integer_best[1] + 0.25 * iy) for iy in range(-4, 5) for ix in range(-4, 5)
    ]
    coarse_best = min(coarse, key=_candidate_key)
    fine = [
        evaluate(coarse_best[0] + 0.0625 * ix, coarse_best[1] + 0.0625 * iy)
        for iy in range(-4, 5)
        for ix in range(-4, 5)
    ]
    best = min(fine, key=_candidate_key)
    separated = [
        candidate
        for candidate in evaluated.values()
        if math.hypot(candidate[0] - best[0], candidate[1] - best[1]) >= 0.5 - 1e-12
    ]
    runner = min(separated, key=_candidate_key) if separated else None
    return best, runner, evaluate(0.0, 0.0)[2], saturated


def _check(
    prepared: _Prepared,
    frames: tuple[int, ...],
    coords: tuple[torch.Tensor, torch.Tensor] | None,
    policy: FrameGaugePolicy,
    dx: float,
    dy: float,
    name: str,
) -> dict[str, Any]:
    if coords is None:
        return {"name": name, "informative": False, "reason": "insufficient_support"}
    best, runner, zero_loss, saturated = _search(prepared, frames, coords, policy)
    runner_loss = math.inf if runner is None else runner[2]
    margin = (runner_loss - best[2]) / max(zero_loss, 1e-8)
    informative = not saturated and zero_loss > 1e-8 and math.isfinite(best[2]) and margin >= policy.min_runner_margin
    delta_x = abs(best[0] - dx)
    delta_y = abs(best[1] - dy)
    return {
        "name": name,
        "informative": informative,
        "dx": best[0],
        "dy": best[1],
        "runner_margin_ratio": margin,
        "supports_global": bool(
            informative and delta_x <= policy.consistency_tolerance and delta_y <= policy.consistency_tolerance
        ),
        "strong_conflict": bool(
            informative and (delta_x > policy.conflict_tolerance or delta_y > policy.conflict_tolerance)
        ),
    }


def translation_valid_mask(
    height: int,
    width: int,
    *,
    dx: float,
    dy: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (
        (xx - float(dx) >= 0.0)
        & (xx - float(dx) <= float(width - 1))
        & (yy - float(dy) >= 0.0)
        & (yy - float(dy) <= float(height - 1))
    )


def estimate_paired_prefix_translation(
    learned: torch.Tensor,
    exact: torch.Tensor,
    *,
    policy: FrameGaugePolicy = DEFAULT_POLICY,
) -> FrameGaugeEstimate:
    _, channels, prefix_t, height, width = _validate_pair(learned, exact)
    if prefix_t < policy.min_prefix_frames:
        return FrameGaugeEstimate("rejected", "insufficient_prefix_support")
    if torch.equal(learned, exact):
        return FrameGaugeEstimate(
            "identity",
            "already_aligned",
            metrics={
                "prefix_frames": tuple(range(max(0, prefix_t - policy.max_prefix_frames), prefix_t)),
                "valid_channels": channels,
                "validation_ncc": 1.0,
                "validation_rms": 0.0,
                "zero_validation_rms": 0.0,
                "invalid_fraction": 0.0,
                "support_y": max(0, height - 2 * policy.margin),
                "support_x": max(0, width - 2 * policy.margin),
            },
        )

    prepared_or_reason = _prepare(learned, exact, policy)
    if isinstance(prepared_or_reason, str):
        return FrameGaugeEstimate("rejected", prepared_or_reason)
    prepared = prepared_or_reason
    coords = _coords(prepared, min_axis=16)
    assert coords is not None

    best, runner, zero_fit_loss, saturated = _search(
        prepared,
        prepared.fit,
        coords,
        policy,
    )
    dx, dy, fit_loss = best
    _, validation_rms, validation_ncc = _metrics(
        prepared,
        prepared.validation,
        coords,
        dx,
        dy,
    )
    _, zero_rms, zero_ncc = _metrics(
        prepared,
        prepared.validation,
        coords,
        0.0,
        0.0,
    )
    improvement = (zero_rms - validation_rms) / max(zero_rms, 1e-8)
    runner_loss = math.inf if runner is None else runner[2]
    runner_margin = (runner_loss - fit_loss) / max(zero_fit_loss, 1e-8)
    valid_mask = translation_valid_mask(prepared.height, prepared.width, dx=dx, dy=dy)
    invalid_fraction = float(1.0 - valid_mask.float().mean().item())

    base_metrics: dict[str, Any] = {
        "prefix_frames": prepared.frame_indices,
        "fit_frames": tuple(prepared.frame_indices[index] for index in prepared.fit),
        "validation_frames": tuple(prepared.frame_indices[index] for index in prepared.validation),
        "stride": prepared.stride,
        "support_y": int(prepared.ys.numel()),
        "support_x": int(prepared.xs.numel()),
        "valid_channels": int(prepared.valid_channels.sum().item()),
        "gradient_rms": prepared.gradient_rms,
        "fit_loss": fit_loss,
        "validation_rms": validation_rms,
        "zero_validation_rms": zero_rms,
        "validation_ncc": validation_ncc,
        "zero_validation_ncc": zero_ncc,
        "rms_improvement": improvement,
        "runner_loss": runner_loss,
        "runner_margin_ratio": runner_margin,
        "invalid_fraction": invalid_fraction,
    }

    def reject(reason: str, **extra: Any) -> FrameGaugeEstimate:
        metrics = dict(base_metrics)
        metrics.update(extra)
        return FrameGaugeEstimate("rejected", reason, dx, dy, metrics)

    if saturated:
        return reject("search_saturated")
    if max(abs(dx), abs(dy)) > policy.accepted_bound + 1e-12:
        return reject("over_bound_shift")
    if invalid_fraction > policy.max_invalid_fraction:
        return reject("excessive_invalid_area")
    if max(abs(dx), abs(dy)) <= policy.identity_bound + 1e-12:
        if zero_ncc >= policy.min_ncc and runner_margin >= policy.min_runner_margin:
            identity_metrics = dict(base_metrics)
            identity_metrics.update(
                identity_candidate_dx=float(dx),
                identity_candidate_dy=float(dy),
            )
            return FrameGaugeEstimate(
                "identity",
                "already_aligned",
                0.0,
                0.0,
                identity_metrics,
            )
        return reject("ambiguous")
    if validation_ncc < policy.min_ncc:
        return reject("validation_ncc")
    if improvement < policy.min_rms_improvement:
        return reject("insufficient_validation_improvement")
    if runner_margin < policy.min_runner_margin:
        return reject("ambiguous_runner_up")

    frame_checks = []
    frame_estimates = []
    for index in prepared.validation:
        check = _check(
            prepared,
            (index,),
            coords,
            policy,
            dx,
            dy,
            f"frame_{prepared.frame_indices[index]}",
        )
        frame_checks.append(check)
        if check.get("informative"):
            frame_estimates.append((float(check["dx"]), float(check["dy"])))
            if not check.get("supports_global"):
                reason = "heldout_frame_conflict" if check.get("strong_conflict") else "heldout_frame_disagreement"
                return reject(reason, frame_checks=tuple(frame_checks))
    if len(frame_estimates) < 2:
        return reject(
            "insufficient_informative_holdout_frames",
            frame_checks=tuple(frame_checks),
        )

    dx_values = [value[0] for value in frame_estimates]
    dy_values = [value[1] for value in frame_estimates]
    frame_summary = {
        "frame_median_dx": float(statistics.median(dx_values)),
        "frame_median_dy": float(statistics.median(dy_values)),
        "frame_dx_range": max(dx_values) - min(dx_values),
        "frame_dy_range": max(dy_values) - min(dy_values),
        "max_frame_disagreement": max(
            max(abs(frame_dx - dx), abs(frame_dy - dy)) for frame_dx, frame_dy in frame_estimates
        ),
    }

    height, width = prepared.height, prepared.width
    regions = (
        ("upper", (0, height // 2, 0, width)),
        ("lower", (height // 2, height, 0, width)),
        ("left", (0, height, 0, width // 2)),
        ("right", (0, height, width // 2, width)),
    )
    region_checks = []
    for name, region in regions:
        check = _check(
            prepared,
            prepared.validation,
            _coords(prepared, region=region, min_axis=8),
            policy,
            dx,
            dy,
            name,
        )
        region_checks.append(check)
        if check.get("strong_conflict"):
            return reject(
                "regional_conflict",
                frame_checks=tuple(frame_checks),
                region_checks=tuple(region_checks),
                **frame_summary,
            )

    region_by_name = {check["name"]: check for check in region_checks}
    vertical_support = any(
        bool(region_by_name[name].get("informative")) and bool(region_by_name[name].get("supports_global"))
        for name in ("upper", "lower")
    )
    horizontal_support = any(
        bool(region_by_name[name].get("informative")) and bool(region_by_name[name].get("supports_global"))
        for name in ("left", "right")
    )
    if not vertical_support or not horizontal_support:
        return reject(
            "insufficient_regional_support",
            frame_checks=tuple(frame_checks),
            region_checks=tuple(region_checks),
            vertical_support=vertical_support,
            horizontal_support=horizontal_support,
            **frame_summary,
        )

    parity_checks = []
    for parity_y in (0, 1):
        for parity_x in (0, 1):
            check = _check(
                prepared,
                prepared.validation,
                _coords(
                    prepared,
                    parity=(parity_y, parity_x),
                    min_axis=8,
                ),
                policy,
                dx,
                dy,
                f"parity_{parity_y}{parity_x}",
            )
            parity_checks.append(check)
            if check.get("informative") and not check.get("supports_global"):
                return reject(
                    ("phase_dependent_conflict" if check.get("strong_conflict") else "phase_dependent_disagreement"),
                    frame_checks=tuple(frame_checks),
                    region_checks=tuple(region_checks),
                    parity_checks=tuple(parity_checks),
                    **frame_summary,
                )

    last = prepared.validation[-1]
    _, last_rms, last_ncc = _metrics(prepared, (last,), coords, dx, dy)
    _, last_zero_rms, _ = _metrics(prepared, (last,), coords, 0.0, 0.0)
    last_improvement = (last_zero_rms - last_rms) / max(last_zero_rms, 1e-8)
    if last_ncc < policy.min_ncc or last_improvement < policy.min_rms_improvement:
        return reject(
            "last_holdout_residual",
            frame_checks=tuple(frame_checks),
            region_checks=tuple(region_checks),
            parity_checks=tuple(parity_checks),
            last_holdout_rms=last_rms,
            last_holdout_zero_rms=last_zero_rms,
            last_holdout_ncc=last_ncc,
            last_holdout_improvement=last_improvement,
            **frame_summary,
        )

    accepted_metrics = dict(base_metrics)
    accepted_metrics.update(
        frame_checks=tuple(frame_checks),
        region_checks=tuple(region_checks),
        parity_checks=tuple(parity_checks),
        last_holdout_rms=last_rms,
        last_holdout_zero_rms=last_zero_rms,
        last_holdout_ncc=last_ncc,
        last_holdout_improvement=last_improvement,
        **frame_summary,
    )
    return FrameGaugeEstimate("accepted", "accepted", dx, dy, accepted_metrics)


def translate_video_cells(
    video: torch.Tensor,
    *,
    dx: float,
    dy: float,
    start_frame: int = 0,
    batch_frames: int = 4,
) -> TranslationApplication:
    if not torch.is_tensor(video) or video.ndim != 5 or not video.is_floating_point():
        raise TypeError("frame-gauge translation expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise ValueError("frame-gauge translation requires finite video")
    if type(start_frame) is not int or not 0 <= start_frame <= int(video.shape[2]):
        raise ValueError("frame-gauge translation start_frame is out of range")
    if type(batch_frames) is not int or batch_frames < 1:
        raise ValueError("frame-gauge translation batch_frames must be positive")
    if not math.isfinite(float(dx)) or not math.isfinite(float(dy)):
        raise ValueError("frame-gauge translation displacement must be finite")

    height, width = map(int, video.shape[-2:])
    valid = translation_valid_mask(
        height,
        width,
        dx=float(dx),
        dy=float(dy),
        device=video.device,
    )
    invalid_fraction = float(1.0 - valid.float().mean().detach().cpu().item())
    transformed_frames = int(video.shape[2]) - start_frame
    if transformed_frames == 0 or (float(dx) == 0.0 and float(dy) == 0.0):
        return TranslationApplication(video, valid, invalid_fraction, 0)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=torch.float32),
        torch.arange(width, device=video.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack(
        (
            2.0 * (xx - float(dx) + 0.5) / float(width) - 1.0,
            2.0 * (yy - float(dy) + 0.5) / float(height) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    output = video.clone()
    batch, channels, _, _, _ = video.shape
    for first in range(start_frame, int(video.shape[2]), batch_frames):
        last = min(int(video.shape[2]), first + batch_frames)
        work = video[:, :, first:last].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width).float()
        shifted = F.grid_sample(
            work,
            grid.expand(work.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        output[:, :, first:last] = (
            shifted.reshape(batch, last - first, channels, height, width).permute(0, 2, 1, 3, 4).to(video)
        )
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("frame-gauge translation produced non-finite output")
    return TranslationApplication(
        output,
        valid,
        invalid_fraction,
        transformed_frames,
    )


__all__ = [
    "DEFAULT_POLICY",
    "FRAME_GAUGE_POLICY_VERSION",
    "FrameGaugeEstimate",
    "FrameGaugePolicy",
    "TranslationApplication",
    "estimate_paired_prefix_translation",
    "translate_video_cells",
    "translation_valid_mask",
]
