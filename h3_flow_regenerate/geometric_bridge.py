"""Motion-aware geometric seam diagnostics for progressive H3 Continuum.

Transforms use output->input sampling coordinates about image centre, in latent
pixels. The bridge uses the source-grid clean handoff sequence to determine
whether the low-resolution trajectory introduced a persistent framing offset,
then requires the same residual to be independently visible on the learned
target-grid boundary before correcting it. The authoritative prefix is never
warped. No gradients are used.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

IDENTITY = (1.0, 1.0, 0.0, 0.0)
MOTION_BASELINE_TRANSITIONS = 6
PERSISTENCE_TRANSITIONS = 4
SOURCE_CLEAN_CONTEXT_ATTR = "_h3_flow_source_clean_video"
_SCALE_FLOOR = 0.005
_TRANSLATION_FLOOR = 0.25
_MAX_SCALE = math.log(1.03)
_MIN_DOMAIN_EVIDENCE = 1.25
_COMBINED_EVIDENCE_THRESHOLD = 2.5
_CORROBORATION_RATIO_MIN = 0.35
_CORROBORATION_RATIO_MAX = 2.85


def warp_frame(frame: torch.Tensor, transform: tuple[float, ...]) -> torch.Tensor:
    """Resample BxCxHxW with one centred output->input affine transform."""
    if frame.ndim != 4 or not frame.is_floating_point():
        raise ValueError("geometric warp requires floating BxCxHxW")
    sx, sy, tx, ty = (float(value) for value in transform)
    if not all(math.isfinite(v) for v in (sx, sy, tx, ty)) or min(sx, sy) <= 0:
        raise ValueError("geometric warp requires finite positive scales")
    if (sx, sy, tx, ty) == IDENTITY:
        return frame
    h, w = frame.shape[-2:]
    with torch.autocast(device_type=frame.device.type, enabled=False):
        theta = frame.new_tensor([[sx, 0.0, 2.0 * tx / w], [0.0, sy, 2.0 * ty / h]], dtype=torch.float32)
        grid = F.affine_grid(theta[None].expand(frame.shape[0], -1, -1), frame.shape, align_corners=False)
        return F.grid_sample(
            frame.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(frame)


def invert_transform(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = (float(value) for value in transform)
    if not all(math.isfinite(v) for v in (sx, sy, tx, ty)) or min(sx, sy) <= 0:
        raise ValueError("cannot invert invalid geometric transform")
    return (1.0 / sx, 1.0 / sy, -tx / sx, -ty / sy)


def compose_transform(first: tuple[float, ...], second: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Return first∘second for centred output->input diagonal affines."""
    sx1, sy1, tx1, ty1 = first
    sx2, sy2, tx2, ty2 = second
    return (sx1 * sx2, sy1 * sy2, sx1 * tx2 + tx1, sy1 * ty2 + ty1)


def residual_transform(observed: tuple[float, ...], expected: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Return C such that C∘expected == observed."""
    return compose_transform(observed, invert_transform(expected))


def _signed_components(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    return (math.log(transform[0]), math.log(transform[1]), transform[2], transform[3])


def _features(video: torch.Tensor):
    b, c, t, h, w = video.shape
    x = video.detach().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    x = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    ratio = min(1.0, 48.0 / max(h, w))
    x = F.interpolate(x, size=(round(h * ratio), round(w * ratio)), mode="area")
    x = x.cpu()
    x = x - x.mean((-2, -1), keepdim=True)
    rms = x.square().mean((-2, -1), keepdim=True).sqrt()
    return x / rms.clamp_min(1e-4), rms


def _errors(reference, moving, transform, original_hw):
    h, w = original_hw
    fh, fw = moving.shape[-2:]
    sx, sy, tx, ty = transform
    xx = np.clip(
        sx * (np.arange(fw, dtype=np.float32) - (fw - 1) / 2) + (fw - 1) / 2 + tx * fw / w,
        0,
        fw - 1,
    )
    yy = np.clip(
        sy * (np.arange(fh, dtype=np.float32) - (fh - 1) / 2) + (fh - 1) / 2 + ty * fh / h,
        0,
        fh - 1,
    )
    x0, y0 = xx.astype(np.int64), yy.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, fw - 1), np.minimum(y0 + 1, fh - 1)
    wx, wy = xx - x0, yy - y0
    top = moving[..., y0[:, None], x0] * (1 - wx) + moving[..., y0[:, None], x1] * wx
    bottom = moving[..., y1[:, None], x0] * (1 - wx) + moving[..., y1[:, None], x1] * wx
    warped = top * (1 - wy[:, None]) + bottom * wy[:, None]
    a, z = reference[..., 3:-3, 3:-3], warped[..., 3:-3, 3:-3]
    a = a - a.mean((-2, -1), keepdims=True)
    z = z - z.mean((-2, -1), keepdims=True)
    a = a / np.maximum(np.sqrt((a * a).mean((-2, -1), keepdims=True)), 1e-4)
    z = z / np.maximum(np.sqrt((z * z).mean((-2, -1), keepdims=True)), 1e-4)
    return np.minimum((a - z) ** 2, 4).mean((1, 2, 3))


def _fit(a, z, hw):
    theta = list(IDENTITY)
    loss = float(_errors(a, z, theta, hw).mean())
    bounds = (0.06, 0.06, 3.0, 3.0)
    for ds, dt in ((0.02, 1.0), (0.01, 0.5), (0.005, 0.25), (0.0025, 0.125)):
        for _ in range(3):
            changed = False
            for axis, step in enumerate((ds, ds, dt, dt)):
                best, best_loss = theta, loss
                for sign in (-1, 1):
                    candidate = theta.copy()
                    candidate[axis] += sign * step
                    if abs(candidate[axis] - IDENTITY[axis]) > bounds[axis] + 1e-8:
                        continue
                    value = float(_errors(a, z, candidate, hw).mean())
                    if value < best_loss - 1e-8:
                        best, best_loss = candidate, value
                if best is not theta:
                    changed = True
                theta, loss = best, best_loss
            if not changed:
                break
    return tuple(theta), loss


def _transform_fields(theta, hw):
    sx, sy, tx, ty = theta
    h, w = hw
    return {
        "scale_x": sx,
        "scale_y": sy,
        "translate_x": tx,
        "translate_y": ty,
        "translate_x_normalized": 2 * tx / w,
        "translate_y_normalized": 2 * ty / h,
    }


def _estimate_registration(reference: torch.Tensor, moving: torch.Tensor, *, comparison=None) -> dict:
    report = {
        "accepted": False,
        "reason": "invalid_geometry",
        "confidence": 0.0,
        "transform": IDENTITY,
        "identity_error": None,
        "aligned_error": None,
        "improvement": 0.0,
    }
    if reference.ndim != 5 or reference.shape != moving.shape or min(reference.shape) < 1:
        return report
    hw = tuple(reference.shape[-2:])
    report.update(_transform_fields(IDENTITY, hw))
    if min(hw) < 16 or min(hw) * min(1.0, 48 / max(hw)) < 12 or reference.shape[0] != 1:
        report["reason"] = "unsupported_geometry"
        return report
    if not reference.is_floating_point() or not moving.is_floating_point():
        return report
    reference, moving = reference[:, :, -3:], moving[:, :, -3:]
    if not bool(torch.isfinite(reference).all() and torch.isfinite(moving).all()):
        report["reason"] = "nonfinite_input"
        return report
    a, ar = _features(reference)
    z, zr = _features(moving)
    if not bool(torch.isfinite(a).all() and torch.isfinite(z).all()):
        report["reason"] = "nonfinite_features"
        return report
    informative = (ar > 1e-4) & (zr > 1e-4)
    if float(informative.float().mean()) < 0.5:
        report["reason"] = "insufficient_structure"
        return report
    gx = (a[..., 1:] - a[..., :-1]).square().mean()
    gy = (a[..., 1:, :] - a[..., :-1, :]).square().mean()
    if min(float(gx), float(gy)) < 1e-3:
        report["reason"] = "insufficient_axis_structure"
        return report
    a_np, z_np = a.numpy(), z.numpy()
    identity = float(_errors(a_np, z_np, IDENTITY, hw).mean())
    theta, error = _fit(a_np, z_np, hw)
    improvement = max(0.0, (identity - error) / max(identity, 1e-8))
    separations = []
    for axis, step in enumerate((0.01, 0.01, 0.5, 0.5)):
        losses = []
        for sign in (-1, 1):
            probe = list(theta)
            probe[axis] += sign * step
            losses.append(float(_errors(a_np, z_np, probe, hw).mean()) - error)
        separations.append(min(losses))
    report.update(
        _transform_fields(theta, hw),
        transform=theta,
        identity_error=identity,
        aligned_error=error,
        improvement=improvement,
        paired_tokens=a_np.shape[0],
        axis_objective_separation=separations,
    )
    if comparison is not None:
        comparison = tuple(float(value) for value in comparison)
        comparison_error = float(_errors(a_np, z_np, comparison, hw).mean())
        axis_gains = []
        for axis in range(4):
            probe = list(theta)
            probe[axis] = comparison[axis]
            axis_gains.append(float(_errors(a_np, z_np, probe, hw).mean()) - error)
        report.update(
            comparison_transform=comparison,
            comparison_error=comparison_error,
            comparison_improvement=max(0.0, (comparison_error - error) / max(comparison_error, 1e-8)),
            residual_axis_gains=axis_gains,
        )
    report["reason"] = "measured"
    return report


@torch.no_grad()
def register_prefix(reference: torch.Tensor, moving: torch.Tensor) -> dict:
    """Same-time paired-prefix registration with conservative acceptance gating."""
    report = _estimate_registration(reference, moving)
    if report["reason"] != "measured":
        return report
    identity = report["identity_error"]
    error = report["aligned_error"]
    theta = report["transform"]
    if identity < 1e-4 or report["improvement"] < 0.15:
        report["reason"] = "identity_or_no_material_improvement"
        return report
    if error > 0.25:
        report["reason"] = "poor_alignment"
        return report
    h, w = reference.shape[-2:]
    safety = (_MAX_SCALE, _MAX_SCALE, min(1.5, w * 0.025), min(1.5, h * 0.025))
    signed = _signed_components(theta)
    if any(abs(value) > limit + 1e-8 for value, limit in zip(signed, safety, strict=True)):
        report["reason"] = "outside_safety_bounds"
        return report
    if min(report["axis_objective_separation"]) < max(1e-4, 0.005 * identity):
        report["reason"] = "ambiguous_optimum"
        return report
    reference3, moving3 = reference[:, :, -3:], moving[:, :, -3:]
    a, _ = _features(reference3)
    z, _ = _features(moving3)
    a_np, z_np = a.numpy(), z.numpy()
    fits = [_fit(a_np[i : i + 1], z_np[i : i + 1], (h, w))[0] for i in range(a_np.shape[0])]
    spread = [max(abs(f[i] - theta[i]) for f in fits) for i in range(4)]
    report.update(per_token_transforms=fits, parameter_spread=spread)
    if a_np.shape[0] < 2:
        report["reason"] = "insufficient_paired_tokens"
        return report
    if any(s > limit for s, limit in zip(spread, (0.0075, 0.0075, 0.375, 0.375), strict=True)):
        report["reason"] = "unstable_prefix_estimates"
        return report
    report.update(accepted=True, reason="stable_paired_prefix", confidence=report["improvement"] * (1 - error))
    return report


def register_pair(reference: torch.Tensor, moving: torch.Tensor, *, comparison=None) -> dict:
    """Diagnostic one-transition fit; it never authorizes correction alone."""
    report = _estimate_registration(reference, moving, comparison=comparison)
    if report["reason"] == "measured":
        report["reason"] = "single_transition_diagnostic"
    return report


def _theil_sen_predict(samples: list[tuple[int, float]], next_index: int) -> tuple[float, float, float, float]:
    indices = np.asarray([item[0] for item in samples], dtype=np.float64)
    values = np.asarray([item[1] for item in samples], dtype=np.float64)
    slopes = [
        (values[j] - values[i]) / (indices[j] - indices[i])
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
        if indices[j] != indices[i]
    ]
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(values - slope * indices))
    predicted = intercept + slope * next_index
    residuals = values - (intercept + slope * indices)
    dispersion = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    return float(predicted), slope, dispersion, intercept


def _motion_model(video: torch.Tensor, prefix_t: int | None = None) -> dict:
    p = int(video.shape[2] if prefix_t is None else prefix_t)
    transitions = []
    start = max(1, p - MOTION_BASELINE_TRANSITIONS)
    for i in range(start, p):
        item = register_pair(video[:, :, i - 1 : i], video[:, :, i : i + 1])
        item["transition_index"] = i
        transitions.append(item)
    usable = [
        item
        for item in transitions
        if item.get("aligned_error") is not None
        and math.isfinite(float(item["aligned_error"]))
        and item["aligned_error"] <= 0.30
    ]
    report = {
        "transitions": transitions,
        "usable_transitions": len(usable),
        "accepted": False,
        "reason": "insufficient_motion_history",
        "next_transition_index": p,
    }
    if len(usable) < 3 or usable[-1]["transition_index"] - usable[0]["transition_index"] < 2:
        return report
    params = [[], [], [], []]
    for item in usable:
        index = int(item["transition_index"])
        sx, sy, tx, ty = item["transform"]
        params[0].append((index, math.log(sx)))
        params[1].append((index, math.log(sy)))
        params[2].append((index, tx))
        params[3].append((index, ty))
    fits = [_theil_sen_predict(values, p) for values in params]
    expected = (math.exp(fits[0][0]), math.exp(fits[1][0]), fits[2][0], fits[3][0])
    report.update(
        accepted=True,
        reason="robust_recent_motion",
        expected_transform=expected,
        trend=(fits[0][1], fits[1][1], fits[2][1], fits[3][1]),
        dispersion=(fits[0][2], fits[1][2], fits[2][2], fits[3][2]),
        intercept=(fits[0][3], fits[1][3], fits[2][3], fits[3][3]),
    )
    return report


def _predict_motion_transform(motion: dict, offset: int) -> tuple[float, float, float, float] | None:
    if not motion.get("accepted"):
        return None
    base = motion["expected_transform"]
    trend = motion["trend"]
    return (
        math.exp(math.log(base[0]) + trend[0] * offset),
        math.exp(math.log(base[1]) + trend[1] * offset),
        base[2] + trend[2] * offset,
        base[3] + trend[3] * offset,
    )


def _motion_residual_candidate(boundary: dict, motion: dict, hw) -> dict:
    """Build a provisional per-axis residual candidate without standalone sigma gating.

    A domain can contribute a moderate but coherent signal. Authorization is
    deferred until source persistence and cross-grid evidence are available.
    """
    report = {
        "accepted": False,
        "provisional": False,
        "reason": "motion_model_unavailable",
        "transform": IDENTITY,
        "axis_applied": [False] * 4,
        "axis_evidence_score": [0.0] * 4,
    }
    if not motion.get("accepted") or boundary.get("aligned_error") is None:
        return report
    expected = motion["expected_transform"]
    observed = boundary["transform"]
    correction = residual_transform(observed, expected)
    comparison_error = boundary.get("comparison_error")
    comparison_improvement = boundary.get("comparison_improvement", 0.0)
    if comparison_error is None or boundary["aligned_error"] > 0.25:
        report.update(reason="boundary_alignment_unusable", raw_transform=correction)
        return report
    if comparison_error <= boundary["aligned_error"] + 1e-12:
        report.update(reason="boundary_not_better_than_motion_model", raw_transform=correction)
        return report

    dispersions = tuple(float(value) for value in motion["dispersion"])
    axis_gains = tuple(float(value) for value in boundary.get("residual_axis_gains", [0.0] * 4))
    h, w = hw
    signed = _signed_components(correction)
    magnitudes = tuple(abs(value) for value in signed)
    floors = (_SCALE_FLOOR, _SCALE_FLOOR, _TRANSLATION_FLOOR, _TRANSLATION_FLOOR)
    evidence_units = tuple(max(floors[i], dispersions[i]) for i in range(4))
    evidence_scores = tuple(magnitudes[i] / evidence_units[i] for i in range(4))
    standalone_thresholds = tuple(max(floors[i], 2.5 * dispersions[i]) for i in range(4))
    gain_floor = max(1e-4, 0.002 * comparison_error)
    safety = (_MAX_SCALE, _MAX_SCALE, min(1.5, w * 0.025), min(1.5, h * 0.025))
    axis_applied = [
        magnitudes[i] >= floors[i] and axis_gains[i] >= gain_floor and magnitudes[i] <= safety[i] + 1e-8
        for i in range(4)
    ]
    applied = (
        correction[0] if axis_applied[0] else 1.0,
        correction[1] if axis_applied[1] else 1.0,
        correction[2] if axis_applied[2] else 0.0,
        correction[3] if axis_applied[3] else 0.0,
    )
    report.update(
        raw_transform=correction,
        raw_signed_residual=signed,
        transform=applied,
        axis_applied=axis_applied,
        axis_magnitude=magnitudes,
        axis_estimator_floor=floors,
        axis_motion_dispersion=dispersions,
        axis_evidence_unit=evidence_units,
        axis_evidence_score=evidence_scores,
        axis_standalone_2p5_dispersion_threshold=standalone_thresholds,
        axis_standalone_significant=[magnitudes[i] >= standalone_thresholds[i] for i in range(4)],
        axis_objective_gain=axis_gains,
        axis_objective_gain_floor=gain_floor,
        comparison_improvement=comparison_improvement,
    )
    if not any(axis_applied):
        report["reason"] = "motion_residual_not_provisional"
        return report
    report.update(
        accepted=True,
        provisional=True,
        reason="provisional_motion_residual",
        confidence=comparison_improvement,
    )
    return report


def _suffix_persistence(video: torch.Tensor, prefix_t: int, motion: dict, candidate: dict) -> dict:
    """Verify per axis that a boundary residual behaves like a constant framing offset."""
    report = {
        "accepted": False,
        "reason": "candidate_unavailable",
        "transitions": [],
        "usable_transitions": 0,
        "axis_persistent": [False] * 4,
    }
    if not candidate.get("accepted") or not motion.get("accepted"):
        return report
    available = int(video.shape[2]) - int(prefix_t) - 1
    if available < 2:
        report["reason"] = "insufficient_suffix_persistence_evidence"
        return report
    transitions = []
    usable = []
    for offset in range(min(PERSISTENCE_TRANSITIONS, available)):
        expected = _predict_motion_transform(motion, offset + 1)
        index = prefix_t + offset
        item = register_pair(video[:, :, index : index + 1], video[:, :, index + 1 : index + 2], comparison=expected)
        item["suffix_transition_offset"] = offset + 1
        if item.get("aligned_error") is not None and math.isfinite(float(item["aligned_error"])):
            item["motion_residual_transform"] = residual_transform(item["transform"], expected)
            item["motion_residual_signed"] = _signed_components(item["motion_residual_transform"])
            if item["aligned_error"] <= 0.30:
                usable.append(item)
        transitions.append(item)
    report.update(transitions=transitions, usable_transitions=len(usable))
    if len(usable) < 2:
        report["reason"] = "insufficient_suffix_persistence_evidence"
        return report

    boundary_signed = candidate["raw_signed_residual"]
    selected_axes = candidate["axis_applied"]
    axis_reports = []
    axis_persistent = [False] * 4
    for axis in range(4):
        if not selected_axes[axis]:
            axis_reports.append({"selected": False, "persistent_offset_supported": False})
            continue
        values = [float(item["motion_residual_signed"][axis]) for item in usable]
        boundary = float(boundary_signed[axis])
        floor = _SCALE_FLOOR if axis < 2 else _TRANSLATION_FLOOR
        limit = max(floor, 0.60 * abs(boundary))
        median_abs = float(np.median([abs(value) for value in values]))
        opposite_max = max((abs(value) for value in values if value * boundary < 0), default=0.0)
        axis_ok = median_abs <= limit and opposite_max < limit
        axis_persistent[axis] = axis_ok
        axis_reports.append(
            {
                "selected": True,
                "boundary_signed_residual": boundary,
                "followup_signed_residuals": values,
                "median_followup_abs_residual": median_abs,
                "followup_residual_limit": limit,
                "opposite_recovery_max": opposite_max,
                "opposite_recovery_limit": limit,
                "persistent_offset_supported": axis_ok,
            }
        )
    report.update(axis=axis_reports, axis_persistent=axis_persistent)
    if not any(axis_persistent):
        report["reason"] = "suffix_recovery_or_drift_detected"
        return report
    report.update(accepted=True, reason="persistent_axes_supported")
    return report


def _project_source_transform(candidate: dict, source_hw, target_hw) -> tuple[float, float, float, float]:
    sh, sw = source_hw
    th, tw = target_hw
    sx, sy, tx, ty = candidate["transform"]
    return (sx, sy, tx * tw / sw, ty * th / sh)


def _corroborate_target_candidate(
    source_candidate: dict,
    target_candidate: dict,
    persistence: dict,
    source_hw,
    target_hw,
) -> dict:
    """Authorize only persistent, same-axis cross-grid evidence with combined support.

    The quadrature score is an engineering concordance score, not a p-value or a
    claim that source and target measurements are statistically independent.
    Each domain must independently contribute at least `_MIN_DOMAIN_EVIDENCE`.
    """
    report = {
        "accepted": False,
        "reason": "candidate_unavailable",
        "transform": IDENTITY,
        "axis_applied": [False] * 4,
        "combined_evidence_threshold": _COMBINED_EVIDENCE_THRESHOLD,
        "minimum_domain_evidence": _MIN_DOMAIN_EVIDENCE,
    }
    if not source_candidate.get("accepted"):
        report["reason"] = "source_candidate_unavailable"
        return report
    projected = _project_source_transform(source_candidate, source_hw, target_hw)
    report["projected_source_transform"] = projected
    if not target_candidate.get("accepted"):
        report["reason"] = "target_boundary_not_provisional"
        return report
    if not persistence.get("accepted"):
        report["reason"] = "source_persistence_unavailable"
        return report

    source_signed = _signed_components(projected)
    target_signed = _signed_components(target_candidate["transform"])
    source_axes = source_candidate["axis_applied"]
    target_axes = target_candidate["axis_applied"]
    persistent_axes = persistence.get("axis_persistent", [False] * 4)
    source_scores = source_candidate.get("axis_evidence_score", [0.0] * 4)
    target_scores = target_candidate.get("axis_evidence_score", [0.0] * 4)

    sign_agreement = []
    magnitude_ratio = []
    combined_scores = []
    domain_floor_pass = []
    axis_applied = []
    axis_reasons = []
    for axis in range(4):
        selected = bool(source_axes[axis] and target_axes[axis] and persistent_axes[axis])
        source_value = float(source_signed[axis])
        target_value = float(target_signed[axis])
        same_sign = selected and source_value * target_value > 0
        ratio = abs(target_value) / abs(source_value) if selected and abs(source_value) > 1e-12 else None
        magnitude_ok = ratio is not None and _CORROBORATION_RATIO_MIN <= ratio <= _CORROBORATION_RATIO_MAX
        source_score = float(source_scores[axis])
        target_score = float(target_scores[axis])
        domain_ok = selected and min(source_score, target_score) >= _MIN_DOMAIN_EVIDENCE
        combined = math.hypot(source_score, target_score) if selected else 0.0
        combined_ok = combined >= _COMBINED_EVIDENCE_THRESHOLD
        authorized = bool(same_sign and magnitude_ok and domain_ok and combined_ok)

        if not selected:
            axis_reason = "axis_not_shared_or_persistent"
        elif not same_sign:
            axis_reason = "residual_sign_disagrees"
        elif not magnitude_ok:
            axis_reason = "residual_magnitude_disagrees"
        elif not domain_ok:
            axis_reason = "domain_evidence_too_weak"
        elif not combined_ok:
            axis_reason = "combined_evidence_insufficient"
        else:
            axis_reason = "combined_evidence_authorized"

        sign_agreement.append(bool(same_sign))
        magnitude_ratio.append(ratio)
        combined_scores.append(combined)
        domain_floor_pass.append(bool(domain_ok))
        axis_applied.append(authorized)
        axis_reasons.append(axis_reason)

    target_transform = target_candidate["transform"]
    applied = (
        target_transform[0] if axis_applied[0] else 1.0,
        target_transform[1] if axis_applied[1] else 1.0,
        target_transform[2] if axis_applied[2] else 0.0,
        target_transform[3] if axis_applied[3] else 0.0,
    )
    report.update(
        target_transform=target_transform,
        source_signed_residual=source_signed,
        target_signed_residual=target_signed,
        source_evidence_score=list(source_scores),
        target_evidence_score=list(target_scores),
        combined_evidence_score=combined_scores,
        domain_evidence_floor_pass=domain_floor_pass,
        persistence_axis=persistent_axes,
        sign_agreement=sign_agreement,
        magnitude_ratio=magnitude_ratio,
        axis_reason=axis_reasons,
        axis_applied=axis_applied,
        transform=applied,
    )
    if not any(axis_applied):
        if any(reason == "combined_evidence_insufficient" for reason in axis_reasons):
            report["reason"] = "combined_evidence_insufficient"
        elif any(reason == "domain_evidence_too_weak" for reason in axis_reasons):
            report["reason"] = "domain_evidence_too_weak"
        else:
            report["reason"] = "cross_grid_residual_not_corroborated"
        return report
    report.update(accepted=True, reason="combined_cross_grid_evidence_authorized")
    return report


def _warp_video_suffix(video: torch.Tensor, prefix_t: int, transform: tuple[float, ...]) -> torch.Tensor:
    corrected = video.clone()
    suffix = video[:, :, prefix_t:]
    if suffix.shape[2] == 0 or transform == IDENTITY:
        return corrected
    b, c, t, h, w = suffix.shape
    flat = suffix.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = warp_frame(flat, transform)
    corrected[:, :, prefix_t:] = flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return corrected


def _dc(a, b):
    return float((a.float().mean((-2, -1)) - b.float().mean((-2, -1))).square().mean().sqrt())


def _valid_source_context(source, learned, prefix_t: int, source_hw) -> tuple[bool, str]:
    if not isinstance(source, torch.Tensor):
        return False, "source_trajectory_context_unavailable"
    if source.ndim != 5 or source.shape[:3] != learned.shape[:3]:
        return False, "source_trajectory_context_shape_mismatch"
    if tuple(source.shape[-2:]) != tuple(source_hw):
        return False, "source_trajectory_context_geometry_mismatch"
    if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
        return False, "source_trajectory_context_invalid"
    if not 0 < prefix_t < source.shape[2]:
        return False, "source_trajectory_context_prefix_mismatch"
    return True, "source_trajectory_context_valid"


@torch.no_grad()
def geometric_seam_bridge(learned, exact, *, source_hw, requested):
    """Correct only a persistent residual authorized by combined cross-grid evidence."""
    started = time.perf_counter()
    p = int(exact.shape[2])
    h, w = learned.shape[-2:]
    ry, rx = h / source_hw[0], w / source_hw[1]

    prefix_registration = register_prefix(exact, learned[:, :, :p])
    target_motion = _motion_model(exact)
    target_expected = target_motion.get("expected_transform") if target_motion.get("accepted") else None
    target_boundary = register_pair(exact[:, :, -1:], learned[:, :, p : p + 1], comparison=target_expected)
    target_candidate = _motion_residual_candidate(target_boundary, target_motion, (h, w))

    source = getattr(learned, SOURCE_CLEAN_CONTEXT_ATTR, None)
    source_valid, source_reason = _valid_source_context(source, learned, p, source_hw)
    low_report = {"available": source_valid, "reason": source_reason, "source_hw": tuple(source_hw)}
    source_candidate = {
        "accepted": False,
        "provisional": False,
        "reason": source_reason,
        "transform": IDENTITY,
        "axis_applied": [False] * 4,
        "axis_evidence_score": [0.0] * 4,
    }
    persistence = {
        "accepted": False,
        "reason": source_reason,
        "transitions": [],
        "usable_transitions": 0,
        "axis_persistent": [False] * 4,
    }
    corroboration = {
        "accepted": False,
        "reason": source_reason,
        "transform": IDENTITY,
        "axis_applied": [False] * 4,
        "combined_evidence_threshold": _COMBINED_EVIDENCE_THRESHOLD,
        "minimum_domain_evidence": _MIN_DOMAIN_EVIDENCE,
    }
    if source_valid:
        source_motion = _motion_model(source, p)
        source_expected = source_motion.get("expected_transform") if source_motion.get("accepted") else None
        source_boundary = register_pair(
            source[:, :, p - 1 : p],
            source[:, :, p : p + 1],
            comparison=source_expected,
        )
        source_candidate = _motion_residual_candidate(source_boundary, source_motion, tuple(source_hw))
        persistence = _suffix_persistence(source, p, source_motion, source_candidate)
        if source_candidate.get("accepted") and persistence.get("accepted"):
            corroboration = _corroborate_target_candidate(
                source_candidate,
                target_candidate,
                persistence,
                tuple(source_hw),
                (h, w),
            )
        low_report.update(
            natural_motion_model=source_motion,
            boundary_before=source_boundary,
            motion_residual=source_candidate,
            suffix_persistence=persistence,
        )

    candidate_ready = bool(corroboration.get("accepted"))
    accepted = bool(requested and candidate_ready)
    if not requested:
        reason = "disabled"
    elif not source_valid:
        reason = source_reason
    elif not source_candidate.get("accepted"):
        reason = source_candidate["reason"]
    elif not persistence.get("accepted"):
        reason = persistence["reason"]
    elif not target_candidate.get("accepted"):
        reason = "target_boundary_not_provisional"
    else:
        reason = corroboration["reason"]

    report = {
        "source_hw": tuple(source_hw),
        "target_hw": (h, w),
        "grid_scale_x": rx,
        "grid_scale_y": ry,
        "grid_anisotropy": abs(rx / ry - 1),
        "prefix_registration": prefix_registration,
        "transfer_bias_candidate": prefix_registration["accepted"],
        "target_natural_motion_model": target_motion,
        "boundary_before": target_boundary,
        "target_motion_residual": target_candidate,
        "low_grid_trajectory": low_report,
        "motion_residual": source_candidate,
        "suffix_persistence": persistence,
        "cross_grid_corroboration": corroboration,
        "requested": bool(requested),
        "accepted": accepted,
        "reason": reason,
        "policy": "persistent_low_grid_motion_residual_combined_cross_grid_evidence",
        "persistent_bias_candidate": bool(any(persistence.get("axis_persistent", [False] * 4))),
        "corroborated_bias_candidate": candidate_ready,
        "tokens_corrected": 0,
        "applied_transforms": [],
        "out_of_bounds_fraction": 0.0,
    }

    corrected = learned
    aligned_last = learned[:, :, p - 1]
    if accepted:
        theta = corroboration["transform"]
        corrected = _warp_video_suffix(learned, p, theta)
        sx, sy, tx, ty = theta
        xx = sx * (torch.arange(w) - (w - 1) / 2) + (w - 1) / 2 + tx
        yy = sy * (torch.arange(h) - (h - 1) / 2) + (h - 1) / 2 + ty
        inside = ((xx >= 0) & (xx <= w - 1))[None, :] & ((yy >= 0) & (yy <= h - 1))[:, None]
        report.update(
            tokens_corrected=int(learned.shape[2] - p),
            applied_transforms=[theta],
            out_of_bounds_fraction=1 - float(inside.float().mean()),
        )

    report.update(
        pre_alignment_dc_rms=_dc(exact[:, :, -1], learned[:, :, p - 1]),
        post_geometry_dc_rms=_dc(exact[:, :, -1], aligned_last),
        boundary_after_geometry=register_pair(
            exact[:, :, -1:],
            corrected[:, :, p : p + 1],
            comparison=target_expected,
        ),
        learned_prefix_unchanged=torch.equal(corrected[:, :, :p], learned[:, :, :p]),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    return corrected, aligned_last, report
