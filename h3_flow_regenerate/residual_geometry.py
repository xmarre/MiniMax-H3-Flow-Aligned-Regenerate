from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from .frame_gauge import DEFAULT_POLICY, _coords, _gather_exact, _prepare, _sample

RESIDUAL_GEOMETRY_POLICY_VERSION = "paired_prefix_residual_geometry_v1"
RESIDUAL_GEOMETRY_MODES = ("off", "measure")
_TILE_IDS = ("TL", "TM", "TR", "ML", "C", "MR", "BL", "BM", "BR")


@dataclass(frozen=True, slots=True)
class ResidualGeometryPolicy:
    search_radius: float = 0.5
    coarse_step: float = 0.125
    fine_radius: float = 0.125
    fine_step: float = 0.03125
    min_tile_axis: int = 8
    min_textured_channels: int = 8
    min_ncc: float = 0.75
    min_runner_margin: float = 0.05
    runner_separation: float = 0.25
    zero_loss_floor: float = 1e-8
    uncertainty_fraction: float = 0.05
    uncertainty_floor: float = 0.03125
    max_uncertainty_half_width: float = 0.125
    max_condition: float = 100.0
    min_edge_signal: float = 0.125
    min_holdout_improvement: float = 0.25
    max_holdout_rms: float = 0.125
    max_tile_error: float = 0.25
    max_cross_axis: float = 0.125
    max_band_edge_disagreement: float = 0.125
    min_forward_scale: float = 0.985
    max_forward_scale: float = 1.015
    max_residual_intercept: float = 0.125
    max_residual_displacement: float = 0.5
    max_total_displacement: float = 2.0
    max_invalid_fraction: float = 0.08
    max_cpu_scratch_bytes: int = 64 * 1024 * 1024


DEFAULT_RESIDUAL_POLICY = ResidualGeometryPolicy()


def normalize_residual_geometry_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    if mode not in RESIDUAL_GEOMETRY_MODES:
        raise ValueError(
            f"frame_gauge_residual_mode must be one of {RESIDUAL_GEOMETRY_MODES}, got {value!r}"
        )
    return mode


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _tile_bounds(height: int, width: int, margin: int) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    y0, y3 = margin, height - margin
    x0, x3 = margin, width - margin
    y1 = y0 + (y3 - y0) // 3
    y2 = y0 + 2 * (y3 - y0) // 3
    x1 = x0 + (x3 - x0) // 3
    x2 = x0 + 2 * (x3 - x0) // 3
    ys = ((y0, y1), (y1, y2), (y2, y3))
    xs = ((x0, x1), (x1, x2), (x2, x3))
    return tuple(
        (_TILE_IDS[row * 3 + col], (ys[row][0], ys[row][1], xs[col][0], xs[col][1]))
        for row in range(3)
        for col in range(3)
    )


def _local_channels(prepared, region: tuple[int, int, int, int], policy: ResidualGeometryPolicy):
    y0, y1, x0, x1 = region
    valid = prepared.valid_channels.clone()
    learned = prepared.learned[:, valid, y0:y1, x0:x1]
    exact = prepared.exact[:, valid, y0:y1, x0:x1]
    if learned.shape[-2] < 2 or learned.shape[-1] < 2:
        return valid & False, 0.0, 0.0
    work = torch.cat((learned, exact), dim=0)
    gx = (work[..., 1:] - work[..., :-1]).square().mean(dim=(0, 2, 3)).sqrt()
    gy = (work[..., 1:, :] - work[..., :-1, :]).square().mean(dim=(0, 2, 3)).sqrt()
    texture = torch.sqrt(gx.square() + gy.square())
    threshold = max(1e-6, 0.01 * float(torch.median(texture[texture > 0]).item())) if bool((texture > 0).any()) else 1e-6
    keep_local = texture > threshold
    keep = torch.zeros_like(valid)
    keep[valid] = keep_local
    return keep, float(torch.median(gx).item()), float(torch.median(gy).item())


def _metrics(
    prepared,
    frames: tuple[int, ...],
    coords: tuple[torch.Tensor, torch.Tensor],
    channels: torch.Tensor,
    dx: float,
    dy: float,
) -> tuple[float, float, float]:
    yy, xx = coords
    learned = prepared.learned[list(frames)][:, channels]
    exact = prepared.exact[list(frames)][:, channels]
    shifted = _sample(learned, yy, xx, dx, dy)
    target = _gather_exact(exact, yy, xx)
    residual = shifted - target
    absolute = residual.abs()
    huber = torch.where(absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5)
    rms = float(residual.square().mean().sqrt().item())
    sc = shifted - shifted.mean(dim=-1, keepdim=True)
    tc = target - target.mean(dim=-1, keepdim=True)
    denominator = sc.square().sum(dim=-1).sqrt() * tc.square().sum(dim=-1).sqrt()
    valid = denominator > 1e-12
    ncc = float((sc[valid] * tc[valid]).sum(dim=-1).div(denominator[valid]).mean().item()) if bool(valid.any()) else -1.0
    return float(huber.mean().item()), rms, ncc


def _sample_field(
    value: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    sy = yy - dy
    sx = xx - dx
    y0 = torch.floor(sy).to(torch.int64)
    x0 = torch.floor(sx).to(torch.int64)
    y1 = (y0 + 1).clamp(max=value.shape[-2] - 1)
    x1 = (x0 + 1).clamp(max=value.shape[-1] - 1)
    y0 = y0.clamp(min=0)
    x0 = x0.clamp(min=0)
    wy = (sy - torch.floor(sy)).view(1, 1, -1)
    wx = (sx - torch.floor(sx)).view(1, 1, -1)
    flat = value.reshape(value.shape[0], value.shape[1], -1)
    width = value.shape[-1]

    def gather(y, x):
        index = (y * width + x).view(1, 1, -1).expand(value.shape[0], value.shape[1], -1)
        return torch.gather(flat, 2, index)

    top = gather(y0, x0) * (1.0 - wx) + gather(y0, x1) * wx
    bottom = gather(y1, x0) * (1.0 - wx) + gather(y1, x1) * wx
    return top * (1.0 - wy) + bottom * wy


def _field_metrics(
    prepared,
    frames: tuple[int, ...],
    coords: tuple[torch.Tensor, torch.Tensor],
    channels: torch.Tensor,
    *,
    rigid_dx: float,
    rigid_dy: float,
    a: float,
    b: float,
) -> tuple[float, float]:
    yy, xx = coords
    cx = (prepared.width - 1) / 2.0
    ux = a * (xx - cx) + b
    dx = torch.full_like(xx, float(rigid_dx)) + ux
    dy = torch.full_like(yy, float(rigid_dy))
    learned = prepared.learned[list(frames)][:, channels]
    exact = prepared.exact[list(frames)][:, channels]
    shifted = _sample_field(learned, yy, xx, dx, dy)
    target = _gather_exact(exact, yy, xx)
    residual = shifted - target
    rms = float(residual.square().mean().sqrt().item())
    sc = shifted - shifted.mean(dim=-1, keepdim=True)
    tc = target - target.mean(dim=-1, keepdim=True)
    denominator = sc.square().sum(dim=-1).sqrt() * tc.square().sum(dim=-1).sqrt()
    valid = denominator > 1e-12
    ncc = float((sc[valid] * tc[valid]).sum(dim=-1).div(denominator[valid]).mean().item()) if bool(valid.any()) else -1.0
    return rms, ncc


def _gradient_center(prepared, frames, coords, channels) -> tuple[float, float]:
    yy, xx = coords
    exact = prepared.exact[list(frames)][:, channels]
    gy = torch.zeros_like(exact)
    gx = torch.zeros_like(exact)
    gx[..., 1:-1] = 0.5 * (exact[..., 2:] - exact[..., :-2])
    gy[..., 1:-1, :] = 0.5 * (exact[..., 2:, :] - exact[..., :-2, :])
    energy = gx.square() + gy.square()
    gathered = _gather_exact(energy, yy, xx).mean(dim=(0, 1))
    total = float(gathered.sum().item())
    if not math.isfinite(total) or total <= 1e-12:
        return float(xx.mean().item()), float(yy.mean().item())
    return (
        float((xx * gathered).sum().item() / total),
        float((yy * gathered).sum().item() / total),
    )


def _candidate_values(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 8) for index in range(count + 1)]


def _measure_observation(
    prepared,
    frame: int,
    tile_id: str,
    region: tuple[int, int, int, int],
    *,
    rigid_dx: float,
    rigid_dy: float,
    policy: ResidualGeometryPolicy,
) -> dict[str, Any]:
    coords = _coords(prepared, region=region, min_axis=policy.min_tile_axis)
    y0, y1, x0, x1 = region
    base: dict[str, Any] = {
        "tile_id": tile_id,
        "frame_index": int(prepared.frame_indices[frame]),
        "bounds": [y0, y1, x0, x1],
        "geometric_center": [(x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0],
    }
    if coords is None:
        base.update(status="unavailable", reason="insufficient_axis_support")
        return base
    channels, texture_x, texture_y = _local_channels(prepared, region, policy)
    valid_channels = int(channels.sum().item())
    yy, xx = coords
    base.update(
        support_y=int(torch.unique(yy).numel()),
        support_x=int(torch.unique(xx).numel()),
        valid_pixels=int(yy.numel()),
        valid_channels=valid_channels,
        texture_x=texture_x,
        texture_y=texture_y,
    )
    if valid_channels < policy.min_textured_channels:
        base.update(status="unavailable", reason="insufficient_textured_channels")
        return base
    ex, ey = _gradient_center(prepared, (frame,), coords, channels)
    base["effective_center"] = [ex, ey]
    base["support_left"] = int((xx < ex).sum().item())
    base["support_right"] = int((xx > ex).sum().item())
    if base["support_left"] == 0 or base["support_right"] == 0:
        base.update(status="unavailable", reason="one_sided_effective_center")
        return base

    evaluated: dict[tuple[float, float], tuple[float, float, float, float, float]] = {}

    def evaluate(ux: float, uy: float):
        key = (round(float(ux), 8), round(float(uy), 8))
        if key not in evaluated:
            loss, rms, ncc = _metrics(
                prepared,
                (frame,),
                coords,
                channels,
                float(rigid_dx) + key[0],
                float(rigid_dy) + key[1],
            )
            evaluated[key] = key[0], key[1], loss, rms, ncc
        return evaluated[key]

    coarse_values = _candidate_values(-policy.search_radius, policy.search_radius, policy.coarse_step)
    for uy in coarse_values:
        for ux in coarse_values:
            evaluate(ux, uy)
    coarse_best = min(evaluated.values(), key=lambda item: (item[2], item[0] ** 2 + item[1] ** 2, item[1], item[0]))
    fine_x = _candidate_values(
        max(-policy.search_radius, coarse_best[0] - policy.fine_radius),
        min(policy.search_radius, coarse_best[0] + policy.fine_radius),
        policy.fine_step,
    )
    fine_y = _candidate_values(
        max(-policy.search_radius, coarse_best[1] - policy.fine_radius),
        min(policy.search_radius, coarse_best[1] + policy.fine_radius),
        policy.fine_step,
    )
    for uy in fine_y:
        for ux in fine_x:
            evaluate(ux, uy)
    best = min(evaluated.values(), key=lambda item: (item[2], item[0] ** 2 + item[1] ** 2, item[1], item[0]))
    zero = evaluate(0.0, 0.0)
    separated = [
        item for item in evaluated.values()
        if math.hypot(item[0] - best[0], item[1] - best[1]) >= policy.runner_separation - 1e-12
    ]
    runner = min(separated, key=lambda item: item[2]) if separated else None
    denominator = max(zero[2], policy.zero_loss_floor)
    runner_margin = math.inf if runner is None else (runner[2] - best[2]) / denominator
    uncertainty_limit = best[2] + policy.uncertainty_fraction * denominator
    near = [item for item in evaluated.values() if item[2] <= uncertainty_limit + 1e-15]
    ux_min = min(item[0] for item in near)
    ux_max = max(item[0] for item in near)
    uy_min = min(item[1] for item in near)
    uy_max = max(item[1] for item in near)
    ux_half = max((ux_max - ux_min) / 2.0, policy.uncertainty_floor)
    uy_half = max((uy_max - uy_min) / 2.0, policy.uncertainty_floor)
    saturated = (
        abs(best[0]) >= policy.search_radius - 1e-12
        or abs(best[1]) >= policy.search_radius - 1e-12
    )
    base.update(
        zero_loss=float(zero[2]),
        zero_rms=float(zero[3]),
        best_loss=float(best[2]),
        best_rms=float(best[3]),
        ncc=float(best[4]),
        ux=float(best[0]),
        uy=float(best[1]),
        runner_separation=(None if runner is None else math.hypot(runner[0] - best[0], runner[1] - best[1])),
        runner_margin_ratio=(None if runner is None else float(runner_margin)),
        uncertainty_x=[float(ux_min), float(ux_max)],
        uncertainty_y=[float(uy_min), float(uy_max)],
        uncertainty_half_width_x=float(ux_half),
        uncertainty_half_width_y=float(uy_half),
        saturated=bool(saturated),
        search_count=len(evaluated),
    )
    if not all(_finite(value) for value in (best[2], best[3], best[4], zero[2])):
        base.update(status="rejected", reason="nonfinite_score")
    elif zero[2] <= policy.zero_loss_floor:
        base.update(status="identity", reason="rigid_residual_below_floor")
    elif saturated:
        base.update(status="rejected", reason="search_saturated")
    elif best[4] < policy.min_ncc:
        base.update(status="rejected", reason="ncc")
    elif runner is not None and runner_margin < policy.min_runner_margin:
        base.update(status="rejected", reason="ambiguous_runner")
    else:
        base.update(status="accepted", reason="accepted")
    return base


def _lstsq(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor | None, int, float]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("residual geometry least-squares inputs are malformed")
    if x.shape[0] < x.shape[1]:
        return None, 0, math.inf
    singular = torch.linalg.svdvals(x)
    rank = int(torch.linalg.matrix_rank(x).item())
    condition = math.inf if float(singular[-1].item()) <= 1e-15 else float((singular[0] / singular[-1]).item())
    if rank != x.shape[1] or not math.isfinite(condition):
        return None, rank, condition
    solution = torch.linalg.lstsq(x, y).solution
    return solution, rank, condition


def _fit_horizontal(observations: list[dict[str, Any]], width: int) -> dict[str, Any]:
    if len(observations) < 2:
        return {"status": "rejected", "reason": "insufficient_observations", "rank": 0, "condition": math.inf}
    cx = (width - 1) / 2.0
    scale = max(cx, 1.0)
    x = torch.tensor(
        [[(float(obs["effective_center"][0]) - cx) / scale, 1.0] for obs in observations],
        dtype=torch.float64,
    )
    y = torch.tensor([[float(obs["ux"])] for obs in observations], dtype=torch.float64)
    solution, rank, condition = _lstsq(x, y)
    if solution is None:
        return {"status": "rejected", "reason": "rank_or_condition", "rank": rank, "condition": condition}
    a = float(solution[0, 0].item()) / scale
    b = float(solution[1, 0].item())
    return {"status": "accepted", "rank": rank, "condition": condition, "a": a, "b": b}


def _fit_constant(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        return {"status": "rejected", "reason": "insufficient_observations"}
    bx = sum(float(obs["ux"]) for obs in observations) / len(observations)
    by = sum(float(obs["uy"]) for obs in observations) / len(observations)
    return {"status": "accepted", "bx": bx, "by": by}


def _predict_horizontal(fit: dict[str, Any], obs: dict[str, Any], width: int) -> tuple[float, float]:
    cx = (width - 1) / 2.0
    return float(fit["a"]) * (float(obs["effective_center"][0]) - cx) + float(fit["b"]), 0.0


def _displacement_errors(fit: dict[str, Any], observations: list[dict[str, Any]], width: int) -> dict[str, float]:
    if not observations:
        return {"rms": math.inf, "max": math.inf}
    errors = []
    for obs in observations:
        px, py = _predict_horizontal(fit, obs, width)
        errors.append(math.hypot(px - float(obs["ux"]), py - float(obs["uy"])))
    return {
        "rms": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "max": max(errors),
    }


def _constant_errors(fit: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, float]:
    if not observations:
        return {"rms": math.inf, "max": math.inf}
    errors = [
        math.hypot(float(fit["bx"]) - float(obs["ux"]), float(fit["by"]) - float(obs["uy"]))
        for obs in observations
    ]
    return {"rms": math.sqrt(sum(value * value for value in errors) / len(errors)), "max": max(errors)}


def _deletion_envelope(observations: list[dict[str, Any]], width: int) -> dict[str, Any]:
    fits = []
    labels = []
    full = _fit_horizontal(observations, width)
    if full.get("status") == "accepted":
        fits.append(full)
        labels.append("full")
    frames = sorted({int(obs["frame_index"]) for obs in observations})
    tiles = sorted({str(obs["tile_id"]) for obs in observations})
    for frame in frames:
        fit = _fit_horizontal([obs for obs in observations if int(obs["frame_index"]) != frame], width)
        if fit.get("status") == "accepted":
            fits.append(fit)
            labels.append(f"drop_frame_{frame}")
    for tile in tiles:
        fit = _fit_horizontal([obs for obs in observations if str(obs["tile_id"]) != tile], width)
        if fit.get("status") == "accepted":
            fits.append(fit)
            labels.append(f"drop_tile_{tile}")
    for index, obs in enumerate(observations):
        for side, value in (("lo", obs["uncertainty_x"][0]), ("hi", obs["uncertainty_x"][1])):
            perturbed = [dict(item) for item in observations]
            perturbed[index]["ux"] = float(value)
            fit = _fit_horizontal(perturbed, width)
            if fit.get("status") == "accepted":
                fits.append(fit)
                labels.append(f"sensitivity_{index}_{side}")
    if not fits:
        return {"status": "rejected", "reason": "no_stable_deletion_fit"}
    a_values = [float(fit["a"]) for fit in fits]
    b_values = [float(fit["b"]) for fit in fits]
    return {
        "status": "accepted",
        "fit_count": len(fits),
        "labels": labels,
        "a": [min(a_values), max(a_values)],
        "b": [min(b_values), max(b_values)],
    }


def _model_receipts(
    prepared,
    observations: list[dict[str, Any]],
    rigid_dx: float,
    rigid_dy: float,
    policy: ResidualGeometryPolicy,
) -> dict[str, Any]:
    accepted = [
        obs for obs in observations
        if obs.get("status") == "accepted"
        and float(obs.get("uncertainty_half_width_x", math.inf)) <= policy.max_uncertainty_half_width
        and float(obs.get("uncertainty_half_width_y", math.inf)) <= policy.max_uncertainty_half_width
    ]
    fit_frames = {prepared.frame_indices[index] for index in prepared.fit}
    holdout_frames = {prepared.frame_indices[index] for index in prepared.validation}
    fit_obs = [obs for obs in accepted if int(obs["frame_index"]) in fit_frames]
    holdout_obs = [obs for obs in accepted if int(obs["frame_index"]) in holdout_frames]
    horizontal = _fit_horizontal(fit_obs, prepared.width)
    constant = _fit_constant(fit_obs)
    result: dict[str, Any] = {
        "selected_model": "horizontal" if horizontal.get("status") == "accepted" else "residual_constant",
        "horizontal": horizontal,
        "residual_constant": constant,
        "fit_observations": len(fit_obs),
        "holdout_observations": len(holdout_obs),
        "eligible": False,
        "eligibility_failures": [],
    }
    failures: list[str] = result["eligibility_failures"]
    required_tiles = {"TL", "TM", "TR", "BL", "BM", "BR"}
    for phase, frames in (("fit", fit_frames), ("holdout", holdout_frames)):
        usable_frames = 0
        for frame in frames:
            ids = {str(obs["tile_id"]) for obs in accepted if int(obs["frame_index"]) == frame}
            if required_tiles.issubset(ids):
                usable_frames += 1
        result[f"{phase}_globally_supported_frames"] = usable_frames
        if usable_frames < 2:
            failures.append(f"insufficient_{phase}_global_support")
    if horizontal.get("status") != "accepted" or constant.get("status") != "accepted":
        failures.append("model_fit_rejected")
        return result
    if int(horizontal["rank"]) != 2 or float(horizontal["condition"]) > policy.max_condition:
        failures.append("horizontal_rank_or_condition")
    holdout_h = _displacement_errors(horizontal, holdout_obs, prepared.width)
    holdout_c = _constant_errors(constant, holdout_obs)
    improvement = (
        (holdout_c["rms"] - holdout_h["rms"]) / max(holdout_c["rms"], policy.zero_loss_floor)
        if math.isfinite(holdout_c["rms"]) else -math.inf
    )
    result["holdout"] = {
        "horizontal_rms": holdout_h["rms"],
        "horizontal_max": holdout_h["max"],
        "constant_rms": holdout_c["rms"],
        "constant_max": holdout_c["max"],
        "improvement_ratio": improvement,
    }
    if improvement < policy.min_holdout_improvement:
        failures.append("holdout_improvement")
    if holdout_h["rms"] > policy.max_holdout_rms:
        failures.append("holdout_rms")
    if holdout_h["max"] > policy.max_tile_error:
        failures.append("holdout_tile_error")

    envelope = _deletion_envelope(fit_obs, prepared.width)
    result["deletion_sensitivity_envelope"] = envelope
    if envelope.get("status") != "accepted":
        failures.append("deletion_instability")
        return result
    a_lo, a_hi = map(float, envelope["a"])
    b_lo, b_hi = map(float, envelope["b"])
    a = float(horizontal["a"])
    b = float(horizontal["b"])
    edge_signal = abs(a) * max(prepared.width - 1, 1)
    edge_uncertainty = 0.5 * abs(a_hi - a_lo) * max(prepared.width - 1, 1)
    result["edge_signal"] = edge_signal
    result["edge_uncertainty"] = edge_uncertainty
    if a_lo <= 0.0 <= a_hi:
        failures.append("zero_slope_in_envelope")
    if edge_signal < policy.min_edge_signal or edge_signal < 2.0 * edge_uncertainty:
        failures.append("insufficient_edge_signal")

    top_fit = _fit_horizontal([obs for obs in fit_obs if str(obs["tile_id"]).startswith("T")], prepared.width)
    bottom_fit = _fit_horizontal([obs for obs in fit_obs if str(obs["tile_id"]).startswith("B")], prepared.width)
    result["band_fits"] = {"upper": top_fit, "lower": bottom_fit}
    if top_fit.get("status") != "accepted" or bottom_fit.get("status") != "accepted":
        failures.append("band_fit_missing")
    else:
        signs = [math.copysign(1.0, float(top_fit["a"])), math.copysign(1.0, float(bottom_fit["a"])), math.copysign(1.0, a)]
        if not (signs[0] == signs[1] == signs[2]):
            failures.append("band_slope_sign")
        cx = (prepared.width - 1) / 2.0
        for x in (0.0, float(prepared.width - 1)):
            upper = float(top_fit["a"]) * (x - cx) + float(top_fit["b"])
            lower = float(bottom_fit["a"]) * (x - cx) + float(bottom_fit["b"])
            if abs(upper - lower) > policy.max_band_edge_disagreement:
                failures.append("band_edge_disagreement")
                break

    heldout_fits = {}
    sign = 1 if a > 0 else -1
    for frame in sorted(holdout_frames):
        fit = _fit_horizontal([obs for obs in holdout_obs if int(obs["frame_index"]) == frame], prepared.width)
        heldout_fits[str(frame)] = fit
        if fit.get("status") != "accepted" or (1 if float(fit.get("a", 0.0)) > 0 else -1) != sign:
            failures.append("holdout_slope_sign")
    result["heldout_frame_fits"] = heldout_fits

    max_uy = max(
        (
            abs(float(obs["uy"])) + float(obs["uncertainty_half_width_y"])
            for obs in accepted
        ),
        default=math.inf,
    )
    result["max_cross_axis_with_uncertainty"] = max_uy
    if max_uy > policy.max_cross_axis:
        failures.append("cross_axis_residual")

    sx = 1.0 / (1.0 - a) if abs(1.0 - a) > 1e-12 else math.inf
    cx = (prepared.width - 1) / 2.0
    tx_residual = (b - a * cx) / (1.0 - a) if math.isfinite(sx) else math.inf
    inverse = [
        [1.0 - a, 0.0, a * cx - b - float(rigid_dx)],
        [0.0, 1.0, -float(rigid_dy)],
        [0.0, 0.0, 1.0],
    ]
    forward = [
        [sx, 0.0, (float(rigid_dx) + b - a * cx) * sx],
        [0.0, 1.0, float(rigid_dy)],
        [0.0, 0.0, 1.0],
    ]
    result["transform"] = {
        "a": a,
        "b": b,
        "forward_sx": sx,
        "tx_residual": tx_residual,
        "inverse_matrix": inverse,
        "forward_matrix": forward,
        "sign_convention": "sample L at r-d-u(r); positive rigid dx/content and positive residual a expands forward x geometry",
        "units": "target_latent_cells",
    }
    if not policy.min_forward_scale <= sx <= policy.max_forward_scale:
        failures.append("forward_scale_bound")
    if max(abs(b_lo), abs(b_hi)) > policy.max_residual_intercept:
        failures.append("residual_intercept_bound")
    corner_residual = max(abs(a_variant * (x - cx) + b_variant) for a_variant in (a_lo, a_hi) for b_variant in (b_lo, b_hi) for x in (0.0, float(prepared.width - 1)))
    result["max_corner_residual"] = corner_residual
    if corner_residual > policy.max_residual_displacement:
        failures.append("residual_displacement_bound")
    total_x = max(abs(float(rigid_dx) + a_variant * (x - cx) + b_variant) for a_variant in (a_lo, a_hi) for b_variant in (b_lo, b_hi) for x in (0.0, float(prepared.width - 1)))
    if total_x > policy.max_total_displacement or abs(float(rigid_dy)) > policy.max_total_displacement:
        failures.append("total_displacement_bound")

    direct = []
    by_key = {(int(obs["frame_index"]), str(obs["tile_id"])): obs for obs in holdout_obs}
    bounds_by_tile = dict(_tile_bounds(prepared.height, prepared.width, DEFAULT_POLICY.margin))
    frame_to_index = {value: index for index, value in enumerate(prepared.frame_indices)}
    for (frame_id, tile_id), obs in by_key.items():
        frame = frame_to_index[frame_id]
        region = bounds_by_tile[tile_id]
        coords = _coords(prepared, region=region, min_axis=policy.min_tile_axis)
        channels, _, _ = _local_channels(prepared, region, policy)
        if coords is None or int(channels.sum().item()) < policy.min_textured_channels:
            continue
        _, rigid_rms, rigid_ncc = _metrics(prepared, (frame,), coords, channels, rigid_dx, rigid_dy)
        candidate_rms, candidate_ncc = _field_metrics(
            prepared,
            (frame,),
            coords,
            channels,
            rigid_dx=rigid_dx,
            rigid_dy=rigid_dy,
            a=a,
            b=b,
        )
        direct.append(
            {
                "frame_index": frame_id,
                "tile_id": tile_id,
                "rigid_rms": rigid_rms,
                "candidate_rms": candidate_rms,
                "rigid_ncc": rigid_ncc,
                "candidate_ncc": candidate_ncc,
                "nondegrading": candidate_rms <= rigid_rms + 1e-9 and candidate_ncc + 1e-9 >= rigid_ncc,
            }
        )
    result["direct_feature_checks"] = direct
    if not direct or not all(item["nondegrading"] for item in direct):
        failures.append("direct_feature_degradation")

    parity_checks = []
    for py in (0, 1):
        for px in (0, 1):
            coords = _coords(prepared, parity=(py, px), min_axis=8)
            if coords is None:
                parity_checks.append({"phase": f"{py}{px}", "status": "unavailable"})
                continue
            channels = prepared.valid_channels
            _, rigid_rms, rigid_ncc = _metrics(prepared, prepared.validation, coords, channels, rigid_dx, rigid_dy)
            candidate_rms, candidate_ncc = _field_metrics(
                prepared,
                prepared.validation,
                coords,
                channels,
                rigid_dx=rigid_dx,
                rigid_dy=rigid_dy,
                a=a,
                b=b,
            )
            parity_checks.append(
                {
                    "phase": f"{py}{px}",
                    "status": "accepted",
                    "rigid_rms": rigid_rms,
                    "candidate_rms": candidate_rms,
                    "rigid_ncc": rigid_ncc,
                    "candidate_ncc": candidate_ncc,
                    "nondegrading": candidate_rms <= rigid_rms + 1e-9 and candidate_ncc + 1e-9 >= rigid_ncc,
                }
            )
    result["parity_checks"] = parity_checks
    informative_parity = [item for item in parity_checks if item.get("status") == "accepted"]
    if len(informative_parity) != 4 or not all(item["nondegrading"] for item in informative_parity):
        failures.append("parity_phase_check")

    result["eligible"] = not failures
    return result


def measure_residual_geometry(
    learned: torch.Tensor,
    exact: torch.Tensor,
    *,
    rigid_dx: float,
    rigid_dy: float,
    policy: ResidualGeometryPolicy = DEFAULT_RESIDUAL_POLICY,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not _finite(rigid_dx) or not _finite(rigid_dy):
        raise ValueError("residual geometry requires finite accepted rigid displacement")
    prepared_or_reason = _prepare(learned, exact, DEFAULT_POLICY)
    if isinstance(prepared_or_reason, str):
        return {
            "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
            "status": "rejected",
            "reason": prepared_or_reason,
            "eligible": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    prepared = prepared_or_reason
    selected_channels = int(prepared.valid_channels.sum().item())
    estimated_scratch = (
        2
        * len(prepared.frame_indices)
        * selected_channels
        * prepared.height
        * prepared.width
        * 8
    )
    if estimated_scratch > policy.max_cpu_scratch_bytes:
        return {
            "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
            "status": "rejected",
            "reason": "cpu_scratch_preflight",
            "estimated_cpu_scratch_bytes": estimated_scratch,
            "max_cpu_scratch_bytes": policy.max_cpu_scratch_bytes,
            "eligible": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    observations = []
    bounds = _tile_bounds(prepared.height, prepared.width, DEFAULT_POLICY.margin)
    fit_positions = set(prepared.fit)
    for frame in range(len(prepared.frame_indices)):
        for tile_id, region in bounds:
            observation = _measure_observation(
                prepared,
                frame,
                tile_id,
                region,
                rigid_dx=rigid_dx,
                rigid_dy=rigid_dy,
                policy=policy,
            )
            observation["phase"] = "fit" if frame in fit_positions else "holdout"
            observations.append(observation)
    models = _model_receipts(prepared, observations, rigid_dx, rigid_dy, policy)
    search_count = sum(int(item.get("search_count", 0)) for item in observations)
    accepted_count = sum(item.get("status") == "accepted" for item in observations)
    unavailable_count = sum(item.get("status") == "unavailable" for item in observations)
    rejected_count = sum(item.get("status") == "rejected" for item in observations)
    identity_count = sum(item.get("status") == "identity" for item in observations)
    return {
        "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
        "status": "measured",
        "reason": "measured",
        "rigid_dx": float(rigid_dx),
        "rigid_dy": float(rigid_dy),
        "fit_indices": [int(prepared.frame_indices[index]) for index in prepared.fit],
        "holdout_indices": [int(prepared.frame_indices[index]) for index in prepared.validation],
        "tile_bounds": {tile_id: list(region) for tile_id, region in bounds},
        "observations": observations,
        "models": models,
        "selected_model": models["selected_model"],
        "eligible": bool(models["eligible"]),
        "counts": {
            "observations": len(observations),
            "accepted": accepted_count,
            "unavailable": unavailable_count,
            "rejected": rejected_count,
            "identity": identity_count,
            "search_scores": search_count,
        },
        "support": {
            "height": prepared.height,
            "width": prepared.width,
            "stride": prepared.stride,
            "valid_channels": selected_channels,
            "margin": DEFAULT_POLICY.margin,
            "max_support_axis": DEFAULT_POLICY.max_support_axis,
        },
        "resource": {
            "estimated_cpu_scratch_bytes": estimated_scratch,
            "max_cpu_scratch_bytes": policy.max_cpu_scratch_bytes,
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


__all__ = [
    "DEFAULT_RESIDUAL_POLICY",
    "RESIDUAL_GEOMETRY_MODES",
    "RESIDUAL_GEOMETRY_POLICY_VERSION",
    "ResidualGeometryPolicy",
    "measure_residual_geometry",
    "normalize_residual_geometry_mode",
]
