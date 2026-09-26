"""Diagnostic-only local content continuity measurements for exact-prefix handoffs."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import torch
import torch.nn.functional as F

_EPS = 1e-12
BOUNDARY_CONTENT_DIAGNOSTIC_POLICY = "partitioned_boundary_content_continuity_v1"
PROVIDER_BOUNDARY_PREDICTOR_POLICY = "partitioned_provider_boundary_temporal_predictor_v1"
PROVIDER_BOUNDARY_CALIBRATION_POLICY = "partitioned_provider_boundary_temporal_calibration_v1"
PROVIDER_BOUNDARY_STABILIZATION_SHADOW_POLICY = "partitioned_provider_boundary_stabilization_shadow_v1"
PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY = "partitioned_provider_boundary_soft_support_shadow_v1"
PROVIDER_BOUNDARY_POST_HIGH_SHADOW_POLICY = "partitioned_provider_boundary_post_high_shadow_v1"
PROVIDER_BOUNDARY_STABILIZATION_POLICY = "partitioned_provider_boundary_soft_support_production_v1"
_RESIDUAL_FIELDS = (
    "raw_rms",
    "lowpass_rms",
    "centered_lowpass_rms",
    "gradient_rms",
    "spatial_mean_rms",
)
_DELTA_FIELDS = (
    "ncc",
    "high_similarity_fraction",
    "unique_fraction",
    "cycle_support_fraction",
    "ambiguity_fraction",
    "similarity_mean",
    "margin_mean",
    "flow_magnitude_mean",
)


def _finite(value: torch.Tensor) -> float:
    result = float(value.detach().to(device="cpu", dtype=torch.float64).item())
    if not math.isfinite(result):
        raise RuntimeError("boundary-content diagnostic produced a non-finite value")
    return result


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return _finite(value.float().square().mean().sqrt())


def _safe_ratio(numerator: float, denominator: float) -> float:
    value = float(numerator) / max(float(denominator), _EPS)
    if not math.isfinite(value):
        raise RuntimeError("boundary-content diagnostic produced a non-finite ratio")
    return value


def _lowpass(frame: torch.Tensor, kernel: int) -> torch.Tensor:
    if frame.ndim != 4:
        raise ValueError("boundary-content low-pass expects BxCxHxW")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("boundary-content low-pass kernel must be a positive odd integer")
    if kernel == 1:
        return frame.float()
    radius = kernel // 2
    work = F.pad(frame.float(), (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(work, kernel_size=kernel, stride=1)


def _ncc(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.detach().to(device="cpu").item()) <= _EPS:
        return 1.0 if _rms(left - right) <= _EPS else 0.0
    return _finite((left * right).sum() / denominator)


def _gradient_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    left_dx = left[..., :, 1:] - left[..., :, :-1]
    right_dx = right[..., :, 1:] - right[..., :, :-1]
    left_dy = left[..., 1:, :] - left[..., :-1, :]
    right_dy = right[..., 1:, :] - right[..., :-1, :]
    squared_sum = (right_dx - left_dx).float().square().sum()
    squared_sum = squared_sum + (right_dy - left_dy).float().square().sum()
    count = right_dx.numel() + right_dy.numel()
    if count == 0:
        return 0.0
    return _finite((squared_sum / float(count)).sqrt())


def _best_local_match(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    radius: int,
    min_similarity: float,
    min_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("boundary-content correspondence expects matching BxCxHxW tensors")
    query = F.normalize(query.float(), dim=1, eps=1e-6)
    key = F.normalize(key.float(), dim=1, eps=1e-6)
    batch, _, height, width = query.shape
    padded = F.pad(key, (radius, radius, radius, radius))
    best = torch.full((batch, height, width), -float("inf"), device=query.device)
    second = torch.full_like(best, -float("inf"))
    best_dx = torch.zeros((batch, height, width), dtype=torch.int64, device=query.device)
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
            score = (query * shifted).sum(dim=1)
            valid = (yy + dy >= 0) & (yy + dy < height) & (xx + dx >= 0) & (xx + dx < width)
            score = score.masked_fill(~valid, -float("inf"))
            better = score > best
            second = torch.where(better, best, torch.maximum(second, score))
            best = torch.where(better, score, best)
            best_dx = torch.where(better, torch.full_like(best_dx, dx), best_dx)
            best_dy = torch.where(better, torch.full_like(best_dy, dy), best_dy)

    margin = best - second
    unique = (best >= min_similarity) & (margin >= min_margin)
    flow = torch.stack((best_dx, best_dy), dim=1)
    return flow, best, margin, unique


def _gather_map(value: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("boundary-content integer gather geometry is invalid")
    batch, height, width = value.shape
    yy = torch.arange(height, device=value.device).view(1, height, 1).expand(batch, height, width)
    xx = torch.arange(width, device=value.device).view(1, 1, width).expand(batch, height, width)
    px = xx + flow[:, 0]
    py = yy + flow[:, 1]
    if bool(((px < 0) | (px >= width) | (py < 0) | (py >= height)).any().item()):
        raise RuntimeError("boundary-content correspondence produced an out-of-bounds match")
    index = (py * width + px).reshape(batch, height * width)
    return torch.gather(value.reshape(batch, height * width), 1, index).reshape(batch, height, width)


def _gather_flow(flow_value: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    gathered = [_gather_map(flow_value[:, axis], flow) for axis in range(2)]
    return torch.stack(gathered, dim=1)


def _correspondence_maps(
    left_low: torch.Tensor,
    right_low: torch.Tensor,
    *,
    radius: int,
    min_similarity: float,
    min_margin: float,
) -> dict[str, torch.Tensor]:
    forward, forward_best, forward_margin, forward_unique = _best_local_match(
        left_low,
        right_low,
        radius=radius,
        min_similarity=min_similarity,
        min_margin=min_margin,
    )
    backward, _, _, backward_unique = _best_local_match(
        right_low,
        left_low,
        radius=radius,
        min_similarity=min_similarity,
        min_margin=min_margin,
    )
    sampled_backward = _gather_flow(backward, forward)
    sampled_backward_unique = _gather_map(backward_unique.to(torch.uint8), forward).bool()
    cycle = (forward + sampled_backward).abs().amax(dim=1) == 0
    support = forward_unique & sampled_backward_unique & cycle
    high_similarity = forward_best >= min_similarity
    ambiguous = high_similarity & (forward_margin < min_margin)
    return {
        "forward": forward,
        "best": forward_best,
        "margin": forward_margin,
        "high_similarity": high_similarity,
        "unique": forward_unique,
        "cycle": cycle,
        "support": support,
        "ambiguous": ambiguous,
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    return _finite(selected.float().mean()) if selected.numel() else 0.0


def _region_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    left_low: torch.Tensor,
    right_low: torch.Tensor,
    correspondence: dict[str, torch.Tensor],
    bounds: tuple[int, int, int, int],
) -> dict[str, float]:
    y0, y1, x0, x1 = bounds
    left_region = left[..., y0:y1, x0:x1]
    right_region = right[..., y0:y1, x0:x1]
    left_low_region = left_low[..., y0:y1, x0:x1]
    right_low_region = right_low[..., y0:y1, x0:x1]
    left_centered = left_low_region - left_low_region.mean(dim=(-2, -1), keepdim=True)
    right_centered = right_low_region - right_low_region.mean(dim=(-2, -1), keepdim=True)

    support = correspondence["support"][..., y0:y1, x0:x1]
    best = correspondence["best"][..., y0:y1, x0:x1]
    margin = correspondence["margin"][..., y0:y1, x0:x1]
    flow = correspondence["forward"][..., y0:y1, x0:x1].float()
    magnitude = flow.square().sum(dim=1).sqrt()
    high_similarity = correspondence["high_similarity"][..., y0:y1, x0:x1]
    unique = correspondence["unique"][..., y0:y1, x0:x1]
    cycle = correspondence["cycle"][..., y0:y1, x0:x1]
    ambiguous = correspondence["ambiguous"][..., y0:y1, x0:x1]

    return {
        "raw_rms": _rms(right_region - left_region),
        "lowpass_rms": _rms(right_low_region - left_low_region),
        "centered_lowpass_rms": _rms(right_centered - left_centered),
        "gradient_rms": _gradient_rms(left_low_region, right_low_region),
        "spatial_mean_rms": _rms(right_region.float().mean(dim=(-2, -1)) - left_region.float().mean(dim=(-2, -1))),
        "ncc": _ncc(left_low_region, right_low_region),
        "high_similarity_fraction": _finite(high_similarity.float().mean()),
        "unique_fraction": _finite(unique.float().mean()),
        "cycle_consistent_fraction": _finite(cycle.float().mean()),
        "cycle_support_fraction": _finite(support.float().mean()),
        "ambiguity_fraction": _finite(ambiguous.float().mean()),
        "similarity_mean": _masked_mean(best, support),
        "margin_mean": _masked_mean(margin, support),
        "flow_magnitude_mean": _masked_mean(magnitude, support),
        "flow_magnitude_max": _finite(magnitude[support].max()) if bool(support.any().item()) else 0.0,
    }


def _tile_bounds(height: int, width: int, rows: int, cols: int) -> dict[str, tuple[int, int, int, int]]:
    if rows < 1 or cols < 1 or rows > height or cols > width:
        raise ValueError("boundary-content tile geometry is invalid")
    result: dict[str, tuple[int, int, int, int]] = {}
    for row in range(rows):
        y0 = row * height // rows
        y1 = (row + 1) * height // rows
        for col in range(cols):
            x0 = col * width // cols
            x1 = (col + 1) * width // cols
            result[f"r{row}c{col}"] = (y0, y1, x0, x1)
    return result


def _pair_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    lowpass_kernel: int,
    radius: int,
    min_similarity: float,
    min_margin: float,
    tiles: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    left_low = _lowpass(left, lowpass_kernel)
    right_low = _lowpass(right, lowpass_kernel)
    correspondence = _correspondence_maps(
        left_low,
        right_low,
        radius=radius,
        min_similarity=min_similarity,
        min_margin=min_margin,
    )
    height, width = map(int, left.shape[-2:])
    return {
        "global": _region_metrics(
            left,
            right,
            left_low,
            right_low,
            correspondence,
            (0, height, 0, width),
        ),
        "tiles": {
            tile_id: {
                "bounds": list(bounds),
                **_region_metrics(
                    left,
                    right,
                    left_low,
                    right_low,
                    correspondence,
                    bounds,
                ),
            }
            for tile_id, bounds in tiles.items()
        },
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def _baseline_summary(pairs: list[dict[str, Any]], *, tile_id: str | None = None) -> dict[str, float]:
    rows = [pair["global"] for pair in pairs] if tile_id is None else [pair["tiles"][tile_id] for pair in pairs]
    keys = [key for key, value in rows[0].items() if key != "bounds" and isinstance(value, (int, float))]
    return {key: _median([float(row[key]) for row in rows]) for key in keys}


def _comparison(boundary: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    fields: dict[str, float] = {}
    for name in _RESIDUAL_FIELDS:
        fields[f"{name}_over_prefix_median"] = _safe_ratio(boundary[name], baseline[name])
    for name in _DELTA_FIELDS:
        fields[f"{name}_minus_prefix_median"] = float(boundary[name]) - float(baseline[name])
    fields["cycle_consistent_fraction_minus_prefix_median"] = float(boundary["cycle_consistent_fraction"]) - float(
        baseline["cycle_consistent_fraction"]
    )
    return fields


def measure_boundary_content_continuity(
    video: torch.Tensor,
    prefix_t: int,
    *,
    pre_steps: int = 3,
    tile_rows: int = 4,
    tile_cols: int = 4,
    lowpass_kernel: int = 5,
    correspondence_radius: int = 3,
    min_similarity: float = 0.35,
    min_margin: float = 0.02,
) -> dict[str, Any]:
    """Measure local structural continuity at the exact-prefix -> suffix boundary.

    The diagnostic is observation-only. It compares the boundary pair against
    the immediately preceding exact-prefix transitions and publishes fixed-grid
    regional receipts plus conservative local correspondence support.
    """

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("boundary-content diagnostic expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("boundary-content diagnostic input contains NaN or Inf")
    prefix_t = int(prefix_t)
    temporal = int(video.shape[2])
    if prefix_t < 2 or prefix_t >= temporal:
        raise ValueError("boundary-content diagnostic requires at least two prefix frames and one suffix frame")
    if lowpass_kernel < 1 or lowpass_kernel % 2 == 0:
        raise ValueError("boundary-content low-pass kernel must be a positive odd integer")
    if correspondence_radius < 1:
        raise ValueError("boundary-content correspondence radius must be positive")
    pre_steps = min(max(1, int(pre_steps)), prefix_t - 1)
    height, width = map(int, video.shape[-2:])
    tiles = _tile_bounds(height, width, int(tile_rows), int(tile_cols))

    boundary_pair = _pair_metrics(
        video[:, :, prefix_t - 1],
        video[:, :, prefix_t],
        lowpass_kernel=lowpass_kernel,
        radius=int(correspondence_radius),
        min_similarity=float(min_similarity),
        min_margin=float(min_margin),
        tiles=tiles,
    )
    pre_pairs = []
    start = prefix_t - pre_steps - 1
    for left_index in range(start, prefix_t - 1):
        pre_pairs.append(
            _pair_metrics(
                video[:, :, left_index],
                video[:, :, left_index + 1],
                lowpass_kernel=lowpass_kernel,
                radius=int(correspondence_radius),
                min_similarity=float(min_similarity),
                min_margin=float(min_margin),
                tiles=tiles,
            )
        )

    global_baseline = _baseline_summary(pre_pairs)
    global_boundary = boundary_pair["global"]
    tile_fields: dict[str, Any] = {}
    for tile_id, bounds in tiles.items():
        baseline = _baseline_summary(pre_pairs, tile_id=tile_id)
        boundary = boundary_pair["tiles"][tile_id]
        tile_fields[tile_id] = {
            "bounds": list(bounds),
            "boundary": {key: value for key, value in boundary.items() if key != "bounds"},
            "prefix_median": baseline,
            "boundary_vs_prefix": _comparison(boundary, baseline),
        }

    ranked = sorted(
        tile_fields,
        key=lambda tile_id: float(
            tile_fields[tile_id]["boundary_vs_prefix"]["centered_lowpass_rms_over_prefix_median"]
        ),
        reverse=True,
    )
    return {
        "policy": BOUNDARY_CONTENT_DIAGNOSTIC_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "prefix_t": prefix_t,
        "pre_steps": pre_steps,
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "lowpass_kernel": int(lowpass_kernel),
        "correspondence_radius": int(correspondence_radius),
        "min_similarity": float(min_similarity),
        "min_margin": float(min_margin),
        "global_boundary": global_boundary,
        "global_prefix_median": global_baseline,
        "global_boundary_vs_prefix": _comparison(global_boundary, global_baseline),
        "tiles": tile_fields,
        "tiles_by_centered_structural_change": ranked,
    }


def _centered_region(value: torch.Tensor) -> torch.Tensor:
    return value.float() - value.float().mean(dim=(-2, -1), keepdim=True)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    denominator = left_flat.square().sum().sqrt() * right_flat.square().sum().sqrt()
    if float(denominator.detach().to(device="cpu").item()) <= _EPS:
        return 1.0 if _rms(left_flat - right_flat) <= _EPS else 0.0
    return _finite(torch.dot(left_flat, right_flat) / denominator)


def _predictor_region(
    baseline_deltas: list[torch.Tensor],
    actual_delta: torch.Tensor,
    bounds: tuple[int, int, int, int],
) -> dict[str, float]:
    y0, y1, x0, x1 = bounds
    baseline = [_centered_region(delta[..., y0:y1, x0:x1]) for delta in baseline_deltas]
    actual = _centered_region(actual_delta[..., y0:y1, x0:x1])
    median_delta = torch.stack(baseline, dim=0).median(dim=0).values
    prediction_error = actual - median_delta
    baseline_rms = [_rms(delta) for delta in baseline]
    baseline_dispersion = _median([_rms(delta - median_delta) for delta in baseline])
    actual_rms = _rms(actual)
    median_rms = _rms(median_delta)
    prediction_error_rms = _rms(prediction_error)
    denominator = median_delta.float().square().sum()
    projection_gain = (
        _finite((actual.float() * median_delta.float()).sum() / denominator)
        if float(denominator.detach().to(device="cpu").item()) > _EPS
        else 0.0
    )
    return {
        "actual_centered_lowpass_delta_rms": actual_rms,
        "prefix_median_centered_lowpass_delta_rms": _median(baseline_rms),
        "predictor_centered_lowpass_delta_rms": median_rms,
        "prediction_error_rms": prediction_error_rms,
        "prediction_error_over_prefix_dispersion": _safe_ratio(prediction_error_rms, baseline_dispersion),
        "prediction_error_over_actual_delta": _safe_ratio(prediction_error_rms, actual_rms),
        "prefix_dispersion_rms": baseline_dispersion,
        "actual_vs_predictor_cosine": _cosine(actual, median_delta),
        "actual_projection_gain_on_predictor": projection_gain,
    }


def measure_provider_boundary_temporal_predictor(
    video: torch.Tensor,
    prefix_t: int,
    *,
    pre_steps: int = 3,
    tile_rows: int = 4,
    tile_cols: int = 4,
    lowpass_kernel: int = 5,
) -> dict[str, Any]:
    """Measure whether the provider's first suffix delta follows recent prefix dynamics."""

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("provider-boundary predictor expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("provider-boundary predictor input contains NaN or Inf")
    prefix_t = int(prefix_t)
    temporal = int(video.shape[2])
    pre_steps = int(pre_steps)
    if pre_steps < 2:
        raise ValueError("provider-boundary predictor requires at least two baseline transitions")
    if prefix_t < pre_steps + 1 or prefix_t >= temporal:
        raise ValueError("provider-boundary predictor lacks prefix history or a first suffix frame")
    if lowpass_kernel < 1 or lowpass_kernel % 2 == 0:
        raise ValueError("provider-boundary predictor low-pass kernel must be a positive odd integer")

    height, width = map(int, video.shape[-2:])
    tiles = _tile_bounds(height, width, int(tile_rows), int(tile_cols))
    low = [_lowpass(video[:, :, index], int(lowpass_kernel)) for index in range(prefix_t - pre_steps - 1, prefix_t + 1)]
    baseline_deltas = [right - left for left, right in pairwise(low[: pre_steps + 1])]
    actual_delta = low[-1] - low[-2]

    global_fields = _predictor_region(
        baseline_deltas,
        actual_delta,
        (0, height, 0, width),
    )
    tile_fields = {
        tile_id: {
            "bounds": list(bounds),
            **_predictor_region(baseline_deltas, actual_delta, bounds),
        }
        for tile_id, bounds in tiles.items()
    }
    by_error_ratio = sorted(
        tile_fields,
        key=lambda tile_id: float(tile_fields[tile_id]["prediction_error_over_prefix_dispersion"]),
        reverse=True,
    )
    by_error_rms = sorted(
        tile_fields,
        key=lambda tile_id: float(tile_fields[tile_id]["prediction_error_rms"]),
        reverse=True,
    )
    return {
        "policy": PROVIDER_BOUNDARY_PREDICTOR_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "predictor": "elementwise_median_centered_lowpass_delta_v1",
        "prefix_t": prefix_t,
        "pre_steps": pre_steps,
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "lowpass_kernel": int(lowpass_kernel),
        "global": global_fields,
        "tiles": tile_fields,
        "tiles_by_prediction_error_ratio": by_error_ratio,
        "tiles_by_prediction_error_rms": by_error_rms,
    }


def _calibration_region_summary(
    history_rows: list[dict[str, float]],
    boundary: dict[str, float],
) -> dict[str, float]:
    if not history_rows:
        raise ValueError("provider-boundary calibration requires historical predictor rows")
    historical_errors = [row["prediction_error_rms"] for row in history_rows]
    historical_error_ratios = [row["prediction_error_over_prefix_dispersion"] for row in history_rows]
    historical_cosines = [row["actual_vs_predictor_cosine"] for row in history_rows]
    historical_gains = [row["actual_projection_gain_on_predictor"] for row in history_rows]
    error_median = _median(historical_errors)
    error_max = max(historical_errors)
    ratio_median = _median(historical_error_ratios)
    ratio_max = max(historical_error_ratios)
    cosine_median = _median(historical_cosines)
    cosine_min = min(historical_cosines)
    gain_median = _median(historical_gains)
    return {
        "boundary_prediction_error_rms": float(boundary["prediction_error_rms"]),
        "historical_prediction_error_rms_median": error_median,
        "historical_prediction_error_rms_max": error_max,
        "boundary_error_over_historical_median": _safe_ratio(boundary["prediction_error_rms"], error_median),
        "boundary_error_over_historical_max": _safe_ratio(boundary["prediction_error_rms"], error_max),
        "boundary_prediction_error_over_prefix_dispersion": float(boundary["prediction_error_over_prefix_dispersion"]),
        "historical_error_over_prefix_dispersion_median": ratio_median,
        "historical_error_over_prefix_dispersion_max": ratio_max,
        "boundary_dispersion_ratio_over_historical_median": _safe_ratio(
            boundary["prediction_error_over_prefix_dispersion"],
            ratio_median,
        ),
        "boundary_dispersion_ratio_over_historical_max": _safe_ratio(
            boundary["prediction_error_over_prefix_dispersion"],
            ratio_max,
        ),
        "boundary_actual_vs_predictor_cosine": float(boundary["actual_vs_predictor_cosine"]),
        "historical_actual_vs_predictor_cosine_median": cosine_median,
        "historical_actual_vs_predictor_cosine_min": cosine_min,
        "boundary_cosine_minus_historical_median": float(boundary["actual_vs_predictor_cosine"]) - cosine_median,
        "boundary_projection_gain_on_predictor": float(boundary["actual_projection_gain_on_predictor"]),
        "historical_projection_gain_median": gain_median,
        "boundary_projection_gain_minus_historical_median": float(boundary["actual_projection_gain_on_predictor"])
        - gain_median,
    }


def measure_provider_boundary_temporal_calibration(
    video: torch.Tensor,
    prefix_t: int,
    *,
    pre_steps: int = 3,
    calibration_targets: int = 5,
    tile_rows: int = 4,
    tile_cols: int = 4,
    lowpass_kernel: int = 5,
) -> dict[str, Any]:
    """Calibrate the first-suffix predictor against recent held-out prefix transitions."""

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("provider-boundary calibration expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("provider-boundary calibration input contains NaN or Inf")
    prefix_t = int(prefix_t)
    temporal = int(video.shape[2])
    pre_steps = int(pre_steps)
    calibration_targets = int(calibration_targets)
    if pre_steps < 2:
        raise ValueError("provider-boundary calibration requires at least two predictor history transitions")
    if calibration_targets < 1:
        raise ValueError("provider-boundary calibration requires at least one historical target")
    if prefix_t < pre_steps + 2 or prefix_t >= temporal:
        raise ValueError("provider-boundary calibration lacks prefix history or a first suffix frame")
    if lowpass_kernel < 1 or lowpass_kernel % 2 == 0:
        raise ValueError("provider-boundary calibration low-pass kernel must be a positive odd integer")

    height, width = map(int, video.shape[-2:])
    tiles = _tile_bounds(height, width, int(tile_rows), int(tile_cols))
    low = [_lowpass(video[:, :, index], int(lowpass_kernel)) for index in range(prefix_t + 1)]
    deltas = [right - left for left, right in pairwise(low)]
    boundary_index = prefix_t - 1
    boundary_history = deltas[boundary_index - pre_steps : boundary_index]
    boundary_actual = deltas[boundary_index]

    first_target = max(pre_steps, boundary_index - calibration_targets)
    target_indices = list(range(first_target, boundary_index))
    if not target_indices:
        raise ValueError("provider-boundary calibration found no held-out prefix transitions")

    bounds_by_id: dict[str, tuple[int, int, int, int]] = {"global": (0, height, 0, width), **tiles}
    regions: dict[str, Any] = {}
    for region_id, bounds in bounds_by_id.items():
        boundary_row = _predictor_region(boundary_history, boundary_actual, bounds)
        historical_rows = [
            _predictor_region(
                deltas[target_index - pre_steps : target_index],
                deltas[target_index],
                bounds,
            )
            for target_index in target_indices
        ]
        fields = _calibration_region_summary(historical_rows, boundary_row)
        if region_id != "global":
            fields = {"bounds": list(bounds), **fields}
        regions[region_id] = fields

    tile_fields = {tile_id: regions[tile_id] for tile_id in tiles}
    by_max_error = sorted(
        tile_fields,
        key=lambda tile_id: float(tile_fields[tile_id]["boundary_error_over_historical_max"]),
        reverse=True,
    )
    by_max_dispersion = sorted(
        tile_fields,
        key=lambda tile_id: float(tile_fields[tile_id]["boundary_dispersion_ratio_over_historical_max"]),
        reverse=True,
    )
    return {
        "policy": PROVIDER_BOUNDARY_CALIBRATION_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "predictor": "elementwise_median_centered_lowpass_delta_v1",
        "calibration": "rolling_held_out_prefix_transitions_v1",
        "prefix_t": prefix_t,
        "pre_steps": pre_steps,
        "requested_calibration_targets": calibration_targets,
        "calibration_target_indices": target_indices,
        "calibration_target_count": len(target_indices),
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "lowpass_kernel": int(lowpass_kernel),
        "global": regions["global"],
        "tiles": tile_fields,
        "tiles_by_boundary_error_over_historical_max": by_max_error,
        "tiles_by_boundary_dispersion_ratio_over_historical_max": by_max_dispersion,
    }


def measure_provider_boundary_stabilization_shadow(
    video: torch.Tensor,
    prefix_t: int,
    *,
    calibration_receipt: dict[str, Any],
    provider_content_receipt: dict[str, Any],
    pre_steps: int = 3,
    tile_rows: int = 4,
    tile_cols: int = 4,
    lowpass_kernel: int = 5,
) -> dict[str, Any]:
    """Measure a non-mutating first-suffix residual-envelope clamp candidate.

    The shadow candidate is deliberately not a production correction. A tile is
    eligible only when both its absolute predictor error and its
    dispersion-normalized predictor error exceed their respective recent
    held-out maxima. The candidate then shrinks only the prediction residual
    enough to return both ratios to the tighter historical envelope.
    """

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("provider-boundary shadow expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("provider-boundary shadow input contains NaN or Inf")
    prefix_t = int(prefix_t)
    temporal = int(video.shape[2])
    if prefix_t < int(pre_steps) + 1 or prefix_t >= temporal:
        raise ValueError("provider-boundary shadow lacks prefix history or a first suffix frame")
    if calibration_receipt.get("policy") != PROVIDER_BOUNDARY_CALIBRATION_POLICY:
        raise ValueError("provider-boundary shadow calibration policy mismatch")
    if provider_content_receipt.get("policy") != BOUNDARY_CONTENT_DIAGNOSTIC_POLICY:
        raise ValueError("provider-boundary shadow content policy mismatch")
    if int(calibration_receipt.get("prefix_t", -1)) != prefix_t:
        raise ValueError("provider-boundary shadow calibration prefix drifted")
    if int(provider_content_receipt.get("prefix_t", -1)) != prefix_t:
        raise ValueError("provider-boundary shadow content prefix drifted")
    if int(calibration_receipt.get("pre_steps", -1)) != int(pre_steps):
        raise ValueError("provider-boundary shadow predictor history drifted")
    if int(calibration_receipt.get("tile_rows", -1)) != int(tile_rows):
        raise ValueError("provider-boundary shadow tile-row geometry drifted")
    if int(calibration_receipt.get("tile_cols", -1)) != int(tile_cols):
        raise ValueError("provider-boundary shadow tile-column geometry drifted")
    if int(calibration_receipt.get("lowpass_kernel", -1)) != int(lowpass_kernel):
        raise ValueError("provider-boundary shadow low-pass kernel drifted")

    height, width = map(int, video.shape[-2:])
    tiles = _tile_bounds(height, width, int(tile_rows), int(tile_cols))
    calibration_tiles = calibration_receipt.get("tiles")
    content_tiles = provider_content_receipt.get("tiles")
    if not isinstance(calibration_tiles, dict) or set(calibration_tiles) != set(tiles):
        raise ValueError("provider-boundary shadow calibration tile set drifted")
    if not isinstance(content_tiles, dict) or set(content_tiles) != set(tiles):
        raise ValueError("provider-boundary shadow content tile set drifted")

    low = [_lowpass(video[:, :, index], int(lowpass_kernel)) for index in range(prefix_t + 1)]
    deltas = [right - left for left, right in pairwise(low)]
    boundary_index = prefix_t - 1
    boundary_history = deltas[boundary_index - int(pre_steps) : boundary_index]
    boundary_actual = deltas[boundary_index]

    candidate = video.clone()
    correction = torch.zeros_like(candidate[:, :, prefix_t], dtype=torch.float32)
    tile_fields: dict[str, Any] = {}
    eligible_tiles: list[str] = []

    for tile_id, bounds in tiles.items():
        y0, y1, x0, x1 = bounds
        calibration = calibration_tiles[tile_id]
        error_ratio = float(calibration["boundary_error_over_historical_max"])
        dispersion_ratio = float(calibration["boundary_dispersion_ratio_over_historical_max"])
        boundary_error_rms = float(calibration["boundary_prediction_error_rms"])
        historical_error_max = float(calibration["historical_prediction_error_rms_max"])
        min_signal_rms = 1e-6
        eligible = (
            boundary_error_rms > min_signal_rms
            and historical_error_max > min_signal_rms
            and error_ratio > 1.0
            and dispersion_ratio > 1.0
        )
        residual_scale = (
            min(
                1.0,
                1.0 / max(error_ratio, _EPS),
                1.0 / max(dispersion_ratio, _EPS),
            )
            if eligible
            else 1.0
        )

        baseline = [_centered_region(delta[..., y0:y1, x0:x1]) for delta in boundary_history]
        actual = _centered_region(boundary_actual[..., y0:y1, x0:x1])
        predictor = torch.stack(baseline, dim=0).median(dim=0).values
        prediction_residual = actual - predictor
        tile_correction = prediction_residual * (1.0 - residual_scale)
        if eligible:
            eligible_tiles.append(tile_id)
            correction[..., y0:y1, x0:x1] = tile_correction

        tile_fields[tile_id] = {
            "bounds": list(bounds),
            "eligible": bool(eligible),
            "boundary_prediction_error_rms": boundary_error_rms,
            "historical_prediction_error_rms_max": historical_error_max,
            "min_signal_rms": min_signal_rms,
            "boundary_error_over_historical_max": error_ratio,
            "boundary_dispersion_ratio_over_historical_max": dispersion_ratio,
            "residual_scale": float(residual_scale),
            "predicted_error_over_historical_max_after": error_ratio * residual_scale,
            "predicted_dispersion_ratio_over_historical_max_after": dispersion_ratio * residual_scale,
            "prediction_residual_rms": _rms(prediction_residual),
            "shadow_correction_rms": _rms(tile_correction) if eligible else 0.0,
            "shadow_correction_abs_max": (
                _finite(tile_correction.abs().max()) if eligible and tile_correction.numel() else 0.0
            ),
        }

    candidate[:, :, prefix_t] = candidate[:, :, prefix_t] - correction.to(candidate)
    candidate_content = measure_boundary_content_continuity(
        candidate,
        prefix_t,
        pre_steps=int(pre_steps),
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        lowpass_kernel=int(lowpass_kernel),
    )

    for tile_id in tiles:
        before = content_tiles[tile_id]["boundary"]
        after = candidate_content["tiles"][tile_id]["boundary"]
        tile_fields[tile_id].update(
            {
                "provider_centered_lowpass_rms_before": float(before["centered_lowpass_rms"]),
                "shadow_centered_lowpass_rms_after": float(after["centered_lowpass_rms"]),
                "shadow_centered_lowpass_rms_ratio": _safe_ratio(
                    after["centered_lowpass_rms"],
                    before["centered_lowpass_rms"],
                ),
                "provider_gradient_rms_before": float(before["gradient_rms"]),
                "shadow_gradient_rms_after": float(after["gradient_rms"]),
                "shadow_gradient_rms_ratio": _safe_ratio(after["gradient_rms"], before["gradient_rms"]),
                "provider_ncc_before": float(before["ncc"]),
                "shadow_ncc_after": float(after["ncc"]),
                "shadow_ncc_delta": float(after["ncc"]) - float(before["ncc"]),
            }
        )

    provider_global = provider_content_receipt["global_boundary"]
    candidate_global = candidate_content["global_boundary"]
    ranked = sorted(
        tile_fields,
        key=lambda tile_id: float(tile_fields[tile_id]["shadow_centered_lowpass_rms_ratio"]),
    )
    return {
        "policy": PROVIDER_BOUNDARY_STABILIZATION_SHADOW_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "production_applied": False,
        "output_mutated": False,
        "candidate": "heldout_max_prediction_residual_shrink_v1",
        "eligibility_rule": "finite_signal_and_absolute_and_dispersion_error_exceed_heldout_max",
        "min_signal_rms": 1e-6,
        "support": "hard_fixed_tile_shadow_only_v1",
        "prefix_t": prefix_t,
        "pre_steps": int(pre_steps),
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "lowpass_kernel": int(lowpass_kernel),
        "eligible_tiles": eligible_tiles,
        "eligible_tile_count": len(eligible_tiles),
        "correction_rms": _rms(correction),
        "correction_abs_max": _finite(correction.abs().max()) if correction.numel() else 0.0,
        "global": {
            "provider_centered_lowpass_rms_before": float(provider_global["centered_lowpass_rms"]),
            "shadow_centered_lowpass_rms_after": float(candidate_global["centered_lowpass_rms"]),
            "shadow_centered_lowpass_rms_ratio": _safe_ratio(
                candidate_global["centered_lowpass_rms"],
                provider_global["centered_lowpass_rms"],
            ),
            "provider_gradient_rms_before": float(provider_global["gradient_rms"]),
            "shadow_gradient_rms_after": float(candidate_global["gradient_rms"]),
            "shadow_gradient_rms_ratio": _safe_ratio(
                candidate_global["gradient_rms"],
                provider_global["gradient_rms"],
            ),
            "provider_ncc_before": float(provider_global["ncc"]),
            "shadow_ncc_after": float(candidate_global["ncc"]),
            "shadow_ncc_delta": float(candidate_global["ncc"]) - float(provider_global["ncc"]),
        },
        "tiles": tile_fields,
        "tiles_by_shadow_centered_lowpass_ratio": ranked,
    }


def _raised_cosine_frontier_support(
    height: int,
    width: int,
    *,
    bounds: tuple[int, int, int, int],
    tile_id: str,
    eligible_tiles: set[str],
    tile_rows: int,
    tile_cols: int,
    feather_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Build an inside-only smooth support mask for one selected tile."""

    y0, y1, x0, x1 = bounds
    local_h = y1 - y0
    local_w = x1 - x0
    support = torch.ones((local_h, local_w), dtype=torch.float32, device=device)
    row_text, col_text = tile_id[1:].split("c", 1)
    row = int(row_text)
    col = int(col_text)

    def ramp(length: int) -> torch.Tensor:
        if length <= 1:
            return torch.zeros((length,), dtype=torch.float32, device=device)
        coordinate = torch.linspace(0.0, 1.0, length, dtype=torch.float32, device=device)
        return 0.5 - 0.5 * torch.cos(math.pi * coordinate)

    feather_y = min(int(feather_width), local_h)
    feather_x = min(int(feather_width), local_w)
    if row > 0 and f"r{row - 1}c{col}" not in eligible_tiles:
        support[:feather_y] *= ramp(feather_y).view(feather_y, 1)
    if row + 1 < int(tile_rows) and f"r{row + 1}c{col}" not in eligible_tiles:
        support[-feather_y:] *= ramp(feather_y).flip(0).view(feather_y, 1)
    if col > 0 and f"r{row}c{col - 1}" not in eligible_tiles:
        support[:, :feather_x] *= ramp(feather_x).view(1, feather_x)
    if col + 1 < int(tile_cols) and f"r{row}c{col + 1}" not in eligible_tiles:
        support[:, -feather_x:] *= ramp(feather_x).flip(0).view(1, feather_x)

    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    mask[y0:y1, x0:x1] = support
    return mask


def _selected_frontier_jump(
    correction: torch.Tensor,
    *,
    tiles: dict[str, tuple[int, int, int, int]],
    eligible_tiles: set[str],
    tile_rows: int,
    tile_cols: int,
) -> dict[str, float]:
    """Measure direct correction jumps across selected/ineligible tile frontiers."""

    samples: list[torch.Tensor] = []
    for tile_id in sorted(eligible_tiles):
        y0, y1, x0, x1 = tiles[tile_id]
        row_text, col_text = tile_id[1:].split("c", 1)
        row = int(row_text)
        col = int(col_text)
        if row > 0 and f"r{row - 1}c{col}" not in eligible_tiles:
            samples.append(correction[..., y0, x0:x1] - correction[..., y0 - 1, x0:x1])
        if row + 1 < int(tile_rows) and f"r{row + 1}c{col}" not in eligible_tiles:
            samples.append(correction[..., y1 - 1, x0:x1] - correction[..., y1, x0:x1])
        if col > 0 and f"r{row}c{col - 1}" not in eligible_tiles:
            samples.append(correction[..., y0:y1, x0] - correction[..., y0:y1, x0 - 1])
        if col + 1 < int(tile_cols) and f"r{row}c{col + 1}" not in eligible_tiles:
            samples.append(correction[..., y0:y1, x1 - 1] - correction[..., y0:y1, x1])
    if not samples:
        return {"rms": 0.0, "abs_max": 0.0}
    flattened = torch.cat([sample.float().reshape(-1) for sample in samples], dim=0)
    return {
        "rms": _rms(flattened),
        "abs_max": _finite(flattened.abs().max()) if flattened.numel() else 0.0,
    }


def measure_provider_boundary_soft_support_shadow(
    video: torch.Tensor,
    prefix_t: int,
    *,
    calibration_receipt: dict[str, Any],
    provider_content_receipt: dict[str, Any],
    hard_shadow_receipt: dict[str, Any],
    pre_steps: int = 3,
    tile_rows: int = 4,
    tile_cols: int = 4,
    lowpass_kernel: int = 5,
) -> dict[str, Any]:
    """Compare the hard shadow with an inside-only raised-cosine support shadow."""

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("provider-boundary soft-support shadow expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("provider-boundary soft-support shadow input contains NaN or Inf")
    prefix_t = int(prefix_t)
    temporal = int(video.shape[2])
    if prefix_t < int(pre_steps) + 1 or prefix_t >= temporal:
        raise ValueError("provider-boundary soft-support shadow lacks prefix history or a first suffix frame")
    if calibration_receipt.get("policy") != PROVIDER_BOUNDARY_CALIBRATION_POLICY:
        raise ValueError("provider-boundary soft-support calibration policy mismatch")
    if provider_content_receipt.get("policy") != BOUNDARY_CONTENT_DIAGNOSTIC_POLICY:
        raise ValueError("provider-boundary soft-support content policy mismatch")
    if hard_shadow_receipt.get("policy") != PROVIDER_BOUNDARY_STABILIZATION_SHADOW_POLICY:
        raise ValueError("provider-boundary soft-support hard-shadow policy mismatch")
    if hard_shadow_receipt.get("output_mutated") is not False:
        raise ValueError("provider-boundary soft-support requires a non-mutating hard shadow")
    if int(calibration_receipt.get("prefix_t", -1)) != prefix_t:
        raise ValueError("provider-boundary soft-support calibration prefix drifted")
    if int(provider_content_receipt.get("prefix_t", -1)) != prefix_t:
        raise ValueError("provider-boundary soft-support content prefix drifted")
    if int(hard_shadow_receipt.get("prefix_t", -1)) != prefix_t:
        raise ValueError("provider-boundary soft-support hard-shadow prefix drifted")

    height, width = map(int, video.shape[-2:])
    tiles = _tile_bounds(height, width, int(tile_rows), int(tile_cols))
    calibration_tiles = calibration_receipt.get("tiles")
    content_tiles = provider_content_receipt.get("tiles")
    hard_tiles = hard_shadow_receipt.get("tiles")
    if not isinstance(calibration_tiles, dict) or set(calibration_tiles) != set(tiles):
        raise ValueError("provider-boundary soft-support calibration tile set drifted")
    if not isinstance(content_tiles, dict) or set(content_tiles) != set(tiles):
        raise ValueError("provider-boundary soft-support content tile set drifted")
    if not isinstance(hard_tiles, dict) or set(hard_tiles) != set(tiles):
        raise ValueError("provider-boundary soft-support hard-shadow tile set drifted")

    eligible_tiles = set(hard_shadow_receipt.get("eligible_tiles", []))
    if not eligible_tiles <= set(tiles):
        raise ValueError("provider-boundary soft-support eligible tile set drifted")
    feather_width = int(lowpass_kernel) // 2 + 1

    low = [_lowpass(video[:, :, index], int(lowpass_kernel)) for index in range(prefix_t + 1)]
    deltas = [right - left for left, right in pairwise(low)]
    boundary_index = prefix_t - 1
    boundary_history = deltas[boundary_index - int(pre_steps) : boundary_index]
    boundary_actual = deltas[boundary_index]

    hard_correction = torch.zeros_like(video[:, :, prefix_t], dtype=torch.float32)
    soft_correction = torch.zeros_like(hard_correction)
    support_union = torch.zeros((height, width), dtype=torch.float32, device=video.device)
    tile_fields: dict[str, Any] = {}

    for tile_id, bounds in tiles.items():
        y0, y1, x0, x1 = bounds
        hard_tile = hard_tiles[tile_id]
        selected = tile_id in eligible_tiles
        residual_scale = float(hard_tile["residual_scale"])
        baseline = [_centered_region(delta[..., y0:y1, x0:x1]) for delta in boundary_history]
        actual = _centered_region(boundary_actual[..., y0:y1, x0:x1])
        predictor = torch.stack(baseline, dim=0).median(dim=0).values
        prediction_residual = actual - predictor
        tile_correction = prediction_residual * (1.0 - residual_scale)

        support = torch.zeros((height, width), dtype=torch.float32, device=video.device)
        if selected:
            hard_correction[..., y0:y1, x0:x1] = tile_correction
            support = _raised_cosine_frontier_support(
                height,
                width,
                bounds=bounds,
                tile_id=tile_id,
                eligible_tiles=eligible_tiles,
                tile_rows=int(tile_rows),
                tile_cols=int(tile_cols),
                feather_width=feather_width,
                device=video.device,
            )
            soft_correction += torch.zeros_like(soft_correction).add(
                support.view(1, 1, height, width)
                * torch.nn.functional.pad(
                    tile_correction,
                    (x0, width - x1, y0, height - y1),
                )
            )
            support_union = torch.maximum(support_union, support)

        local_support = support[y0:y1, x0:x1]
        tile_fields[tile_id] = {
            "bounds": list(bounds),
            "eligible": bool(selected),
            "residual_scale": residual_scale,
            "support_min": _finite(local_support.min()) if selected and local_support.numel() else 0.0,
            "support_mean": _finite(local_support.mean()) if selected and local_support.numel() else 0.0,
            "support_max": _finite(local_support.max()) if selected and local_support.numel() else 0.0,
            "support_nonzero_fraction": (
                _finite((local_support > 0).float().mean()) if selected and local_support.numel() else 0.0
            ),
            "hard_direct_correction_rms": _rms(tile_correction) if selected else 0.0,
            "soft_direct_correction_rms": (_rms(soft_correction[..., y0:y1, x0:x1]) if selected else 0.0),
        }

    hard_frontier = _selected_frontier_jump(
        hard_correction,
        tiles=tiles,
        eligible_tiles=eligible_tiles,
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
    )
    soft_frontier = _selected_frontier_jump(
        soft_correction,
        tiles=tiles,
        eligible_tiles=eligible_tiles,
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
    )

    candidate = video.clone()
    candidate[:, :, prefix_t] = candidate[:, :, prefix_t] - soft_correction.to(candidate)
    candidate_content = measure_boundary_content_continuity(
        candidate,
        prefix_t,
        pre_steps=int(pre_steps),
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        lowpass_kernel=int(lowpass_kernel),
    )
    candidate_calibration = measure_provider_boundary_temporal_calibration(
        candidate,
        prefix_t,
        pre_steps=int(pre_steps),
        calibration_targets=int(calibration_receipt.get("requested_calibration_targets", 5)),
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        lowpass_kernel=int(lowpass_kernel),
    )

    for tile_id in tiles:
        before = content_tiles[tile_id]["boundary"]
        after = candidate_content["tiles"][tile_id]["boundary"]
        calibrated_after = candidate_calibration["tiles"][tile_id]
        tile_fields[tile_id].update(
            {
                "provider_centered_lowpass_rms_before": float(before["centered_lowpass_rms"]),
                "soft_centered_lowpass_rms_after": float(after["centered_lowpass_rms"]),
                "soft_centered_lowpass_rms_ratio": _safe_ratio(
                    after["centered_lowpass_rms"],
                    before["centered_lowpass_rms"],
                ),
                "provider_gradient_rms_before": float(before["gradient_rms"]),
                "soft_gradient_rms_after": float(after["gradient_rms"]),
                "soft_gradient_rms_ratio": _safe_ratio(after["gradient_rms"], before["gradient_rms"]),
                "provider_ncc_before": float(before["ncc"]),
                "soft_ncc_after": float(after["ncc"]),
                "soft_ncc_delta": float(after["ncc"]) - float(before["ncc"]),
                "soft_boundary_error_over_historical_max_after": float(
                    calibrated_after["boundary_error_over_historical_max"]
                ),
                "soft_boundary_dispersion_ratio_over_historical_max_after": float(
                    calibrated_after["boundary_dispersion_ratio_over_historical_max"]
                ),
            }
        )

    provider_global = provider_content_receipt["global_boundary"]
    candidate_global = candidate_content["global_boundary"]
    calibrated_global = candidate_calibration["global"]
    ranked = sorted(
        tile_fields,
        key=lambda tile_id: float(tile_fields[tile_id]["soft_centered_lowpass_rms_ratio"]),
    )
    return {
        "policy": PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "production_applied": False,
        "output_mutated": False,
        "candidate": "heldout_max_prediction_residual_shrink_v1",
        "eligibility_rule": "inherit_hard_shadow_heldout_max_selection_v1",
        "support": "inside_raised_cosine_selected_frontier_v1",
        "support_owner": "diagnostic_shadow_clone_only",
        "prefix_t": prefix_t,
        "pre_steps": int(pre_steps),
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "lowpass_kernel": int(lowpass_kernel),
        "feather_width": feather_width,
        "eligible_tiles": sorted(eligible_tiles),
        "eligible_tile_count": len(eligible_tiles),
        "hard_frontier_edge_jump_rms": hard_frontier["rms"],
        "hard_frontier_edge_jump_abs_max": hard_frontier["abs_max"],
        "soft_frontier_edge_jump_rms": soft_frontier["rms"],
        "soft_frontier_edge_jump_abs_max": soft_frontier["abs_max"],
        "soft_over_hard_frontier_edge_jump_rms": _safe_ratio(soft_frontier["rms"], hard_frontier["rms"]),
        "soft_over_hard_frontier_edge_jump_abs_max": _safe_ratio(
            soft_frontier["abs_max"],
            hard_frontier["abs_max"],
        ),
        "support_union_mean": _finite(support_union.mean()),
        "support_union_nonzero_fraction": _finite((support_union > 0).float().mean()),
        "soft_correction_rms": _rms(soft_correction),
        "soft_correction_abs_max": _finite(soft_correction.abs().max()) if soft_correction.numel() else 0.0,
        "global": {
            "provider_centered_lowpass_rms_before": float(provider_global["centered_lowpass_rms"]),
            "soft_centered_lowpass_rms_after": float(candidate_global["centered_lowpass_rms"]),
            "soft_centered_lowpass_rms_ratio": _safe_ratio(
                candidate_global["centered_lowpass_rms"],
                provider_global["centered_lowpass_rms"],
            ),
            "provider_gradient_rms_before": float(provider_global["gradient_rms"]),
            "soft_gradient_rms_after": float(candidate_global["gradient_rms"]),
            "soft_gradient_rms_ratio": _safe_ratio(
                candidate_global["gradient_rms"],
                provider_global["gradient_rms"],
            ),
            "provider_ncc_before": float(provider_global["ncc"]),
            "soft_ncc_after": float(candidate_global["ncc"]),
            "soft_ncc_delta": float(candidate_global["ncc"]) - float(provider_global["ncc"]),
            "soft_boundary_error_over_historical_max_after": float(
                calibrated_global["boundary_error_over_historical_max"]
            ),
            "soft_boundary_dispersion_ratio_over_historical_max_after": float(
                calibrated_global["boundary_dispersion_ratio_over_historical_max"]
            ),
        },
        "tiles": tile_fields,
        "tiles_by_soft_centered_lowpass_ratio": ranked,
    }



def measure_provider_boundary_post_high_shadow(
    video: torch.Tensor,
    prefix_t: int,
    *,
    post_high_content_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Re-evaluate the bounded provider hypothesis on the existing post-high clean state.

    This diagnostic deliberately performs no sampler/model/provider/VAE work and never returns
    a mutated tensor. Hardware validation demonstrated that a pre-high shadow improvement is
    not sufficient evidence that the same perturbation remains beneficial after target-high.
    """

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("post-high provider-boundary shadow expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("post-high provider-boundary shadow input contains NaN or Inf")
    prefix_t = int(prefix_t)
    if post_high_content_receipt.get("policy") != BOUNDARY_CONTENT_DIAGNOSTIC_POLICY:
        raise ValueError("post-high provider-boundary shadow content policy mismatch")
    if int(post_high_content_receipt.get("prefix_t", -1)) != prefix_t:
        raise ValueError("post-high provider-boundary shadow prefix drifted")
    if post_high_content_receipt.get("diagnostic_only") is not True:
        raise ValueError("post-high provider-boundary shadow requires diagnostic content evidence")

    calibration = measure_provider_boundary_temporal_calibration(video, prefix_t)
    hard_shadow = measure_provider_boundary_stabilization_shadow(
        video,
        prefix_t,
        calibration_receipt=calibration,
        provider_content_receipt=post_high_content_receipt,
    )
    soft_shadow = measure_provider_boundary_soft_support_shadow(
        video,
        prefix_t,
        calibration_receipt=calibration,
        provider_content_receipt=post_high_content_receipt,
        hard_shadow_receipt=hard_shadow,
    )

    if hard_shadow.get("output_mutated") is not False or soft_shadow.get("output_mutated") is not False:
        raise RuntimeError("post-high provider-boundary shadow attempted to mutate output")
    if hard_shadow.get("eligible_tiles") != soft_shadow.get("eligible_tiles"):
        raise RuntimeError("post-high provider-boundary shadow eligibility drifted")

    return {
        "policy": PROVIDER_BOUNDARY_POST_HIGH_SHADOW_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "production_application_permitted": False,
        "output_mutated": False,
        "reason": "post_high_validation_required_before_any_provider_boundary_mutation",
        "source_stage": "post_high_internal_clean",
        "prefix_t": prefix_t,
        "calibration": calibration,
        "hard_shadow": hard_shadow,
        "soft_shadow": soft_shadow,
        "eligible_tiles": list(soft_shadow.get("eligible_tiles", [])),
        "eligible_tile_count": int(soft_shadow.get("eligible_tile_count", 0)),
        "soft_correction_rms": float(soft_shadow.get("soft_correction_rms", 0.0)),
        "soft_correction_abs_max": float(soft_shadow.get("soft_correction_abs_max", 0.0)),
    }


def apply_provider_boundary_soft_support_stabilization(
    video: torch.Tensor,
    prefix_t: int,
    *,
    soft_shadow_receipt: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply exactly the measured soft-support candidate to the first suffix token.

    This is a bounded opt-in production candidate. Eligibility, residual scale,
    geometry, and support are inherited byte-for-byte from the preceding shadow
    receipt; no additional threshold is introduced here.
    """

    if video.ndim != 5 or not video.is_floating_point():
        raise ValueError("provider-boundary stabilization expects floating BxCxTxHxW video")
    if not bool(torch.isfinite(video).all().item()):
        raise RuntimeError("provider-boundary stabilization input contains NaN or Inf")
    if soft_shadow_receipt.get("policy") != PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY:
        raise ValueError("provider-boundary stabilization shadow policy mismatch")
    if soft_shadow_receipt.get("diagnostic_only") is not True:
        raise ValueError("provider-boundary stabilization requires diagnostic shadow evidence")
    if soft_shadow_receipt.get("production_applied") is not False:
        raise ValueError("provider-boundary stabilization shadow already claims production mutation")
    if soft_shadow_receipt.get("output_mutated") is not False:
        raise ValueError("provider-boundary stabilization shadow mutated its input")

    prefix_t = int(prefix_t)
    temporal = int(video.shape[2])
    pre_steps = int(soft_shadow_receipt.get("pre_steps", 0))
    tile_rows = int(soft_shadow_receipt.get("tile_rows", 0))
    tile_cols = int(soft_shadow_receipt.get("tile_cols", 0))
    lowpass_kernel = int(soft_shadow_receipt.get("lowpass_kernel", 0))
    feather_width = int(soft_shadow_receipt.get("feather_width", 0))
    if prefix_t != int(soft_shadow_receipt.get("prefix_t", -1)):
        raise ValueError("provider-boundary stabilization prefix drifted")
    if prefix_t < pre_steps + 1 or prefix_t >= temporal:
        raise ValueError("provider-boundary stabilization lacks prefix history or first suffix")
    if pre_steps != 3 or tile_rows != 4 or tile_cols != 4 or lowpass_kernel != 5 or feather_width != 3:
        raise ValueError("provider-boundary stabilization measured geometry drifted")

    height, width = map(int, video.shape[-2:])
    tiles = _tile_bounds(height, width, tile_rows, tile_cols)
    soft_tiles = soft_shadow_receipt.get("tiles")
    if not isinstance(soft_tiles, dict) or set(soft_tiles) != set(tiles):
        raise ValueError("provider-boundary stabilization shadow tile set drifted")
    eligible_tiles = set(soft_shadow_receipt.get("eligible_tiles", []))
    if not eligible_tiles <= set(tiles):
        raise ValueError("provider-boundary stabilization eligible tile set drifted")

    low = [_lowpass(video[:, :, index], lowpass_kernel) for index in range(prefix_t + 1)]
    deltas = [right - left for left, right in pairwise(low)]
    boundary_index = prefix_t - 1
    boundary_history = deltas[boundary_index - pre_steps : boundary_index]
    boundary_actual = deltas[boundary_index]

    correction = torch.zeros_like(video[:, :, prefix_t], dtype=torch.float32)
    for tile_id in sorted(eligible_tiles):
        y0, y1, x0, x1 = tiles[tile_id]
        tile = soft_tiles[tile_id]
        if tile.get("eligible") is not True:
            raise ValueError(f"provider-boundary stabilization tile {tile_id} eligibility drifted")
        residual_scale = float(tile["residual_scale"])
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError(f"provider-boundary stabilization tile {tile_id} residual scale is invalid")
        baseline = [_centered_region(delta[..., y0:y1, x0:x1]) for delta in boundary_history]
        actual = _centered_region(boundary_actual[..., y0:y1, x0:x1])
        predictor = torch.stack(baseline, dim=0).median(dim=0).values
        prediction_residual = actual - predictor
        tile_correction = prediction_residual * (1.0 - residual_scale)
        support = _raised_cosine_frontier_support(
            height,
            width,
            bounds=tiles[tile_id],
            tile_id=tile_id,
            eligible_tiles=eligible_tiles,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            feather_width=feather_width,
            device=video.device,
        )
        correction += support.view(1, 1, height, width) * torch.nn.functional.pad(
            tile_correction,
            (x0, width - x1, y0, height - y1),
        )

    measured_rms = _rms(correction)
    measured_abs_max = _finite(correction.abs().max()) if correction.numel() else 0.0
    expected_rms = float(soft_shadow_receipt.get("soft_correction_rms", float("nan")))
    expected_abs_max = float(soft_shadow_receipt.get("soft_correction_abs_max", float("nan")))
    if not (
        math.isfinite(expected_rms)
        and math.isfinite(expected_abs_max)
        and math.isclose(measured_rms, expected_rms, rel_tol=1e-6, abs_tol=1e-8)
        and math.isclose(measured_abs_max, expected_abs_max, rel_tol=1e-6, abs_tol=1e-8)
    ):
        raise RuntimeError("provider-boundary stabilization no longer reproduces its measured shadow")

    stabilized = video.clone()
    stabilized[:, :, prefix_t] = stabilized[:, :, prefix_t] - correction.to(stabilized)
    if not torch.equal(stabilized[:, :, :prefix_t], video[:, :, :prefix_t]):
        raise RuntimeError("provider-boundary stabilization modified provider prefix")
    if prefix_t + 1 < temporal and not torch.equal(
        stabilized[:, :, prefix_t + 1 :],
        video[:, :, prefix_t + 1 :],
    ):
        raise RuntimeError("provider-boundary stabilization extrapolated into later suffix tokens")

    return stabilized, {
        "policy": PROVIDER_BOUNDARY_STABILIZATION_POLICY,
        "source_shadow_policy": PROVIDER_BOUNDARY_SOFT_SUPPORT_SHADOW_POLICY,
        "source_shadow_candidate": str(soft_shadow_receipt.get("candidate", "")),
        "support": str(soft_shadow_receipt.get("support", "")),
        "support_owner": "production_first_suffix_only",
        "prefix_t": prefix_t,
        "pre_steps": pre_steps,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "lowpass_kernel": lowpass_kernel,
        "feather_width": feather_width,
        "eligible_tiles": sorted(eligible_tiles),
        "eligible_tile_count": len(eligible_tiles),
        "applied": bool(eligible_tiles and measured_rms > 0.0),
        "corrected_tokens": 1 if eligible_tiles and measured_rms > 0.0 else 0,
        "authoritative_prefix_modified": False,
        "later_suffix_extrapolated": False,
        "correction_rms": measured_rms,
        "correction_abs_max": measured_abs_max,
        "hard_frontier_edge_jump_rms": float(soft_shadow_receipt["hard_frontier_edge_jump_rms"]),
        "hard_frontier_edge_jump_abs_max": float(soft_shadow_receipt["hard_frontier_edge_jump_abs_max"]),
        "soft_frontier_edge_jump_rms": float(soft_shadow_receipt["soft_frontier_edge_jump_rms"]),
        "soft_frontier_edge_jump_abs_max": float(soft_shadow_receipt["soft_frontier_edge_jump_abs_max"]),
        "soft_centered_lowpass_rms_ratio": float(soft_shadow_receipt["global"]["soft_centered_lowpass_rms_ratio"]),
        "soft_gradient_rms_ratio": float(soft_shadow_receipt["global"]["soft_gradient_rms_ratio"]),
        "soft_ncc_delta": float(soft_shadow_receipt["global"]["soft_ncc_delta"]),
    }


def compare_boundary_content_stages(
    pre_high: dict[str, Any],
    post_high: dict[str, Any],
) -> dict[str, Any]:
    """Compare diagnostic receipts without turning them into a production gate."""

    if pre_high.get("policy") != BOUNDARY_CONTENT_DIAGNOSTIC_POLICY:
        raise ValueError("pre-high boundary-content policy mismatch")
    if post_high.get("policy") != BOUNDARY_CONTENT_DIAGNOSTIC_POLICY:
        raise ValueError("post-high boundary-content policy mismatch")
    if set(pre_high.get("tiles", {})) != set(post_high.get("tiles", {})):
        raise ValueError("boundary-content stage tile geometry drifted")

    pre_global = pre_high["global_boundary"]
    post_global = post_high["global_boundary"]
    tile_delta: dict[str, Any] = {}
    for tile_id in pre_high["tiles"]:
        pre = pre_high["tiles"][tile_id]["boundary"]
        post = post_high["tiles"][tile_id]["boundary"]
        tile_delta[tile_id] = {
            "centered_lowpass_rms_post_over_pre": _safe_ratio(
                post["centered_lowpass_rms"],
                pre["centered_lowpass_rms"],
            ),
            "gradient_rms_post_over_pre": _safe_ratio(post["gradient_rms"], pre["gradient_rms"]),
            "ncc_post_minus_pre": float(post["ncc"]) - float(pre["ncc"]),
            "cycle_support_post_minus_pre": (
                float(post["cycle_support_fraction"]) - float(pre["cycle_support_fraction"])
            ),
        }

    ranked = sorted(
        tile_delta,
        key=lambda tile_id: float(tile_delta[tile_id]["centered_lowpass_rms_post_over_pre"]),
        reverse=True,
    )
    return {
        "policy": BOUNDARY_CONTENT_DIAGNOSTIC_POLICY,
        "diagnostic_only": True,
        "production_gate": False,
        "global": {
            "centered_lowpass_rms_post_over_pre": _safe_ratio(
                post_global["centered_lowpass_rms"],
                pre_global["centered_lowpass_rms"],
            ),
            "gradient_rms_post_over_pre": _safe_ratio(
                post_global["gradient_rms"],
                pre_global["gradient_rms"],
            ),
            "ncc_post_minus_pre": float(post_global["ncc"]) - float(pre_global["ncc"]),
            "cycle_support_post_minus_pre": (
                float(post_global["cycle_support_fraction"]) - float(pre_global["cycle_support_fraction"])
            ),
        },
        "tiles": tile_delta,
        "tiles_by_post_high_structural_amplification": ranked,
    }
