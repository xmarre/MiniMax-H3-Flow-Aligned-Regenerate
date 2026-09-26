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
