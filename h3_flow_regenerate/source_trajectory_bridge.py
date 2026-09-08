"""Source-grid trajectory correction for Mixed-Grid Continuum seams.

The Mixed-Grid low stage can generate a genuine low-resolution suffix whose
first temporal tokens drift geometrically from the recent protected-prefix
motion. This module detects that drift on the clean source-grid trajectory and,
when source evidence and measured temporal evolution are strong enough,
corrects only the affected early suffix tokens before the learned 3D upscaler.

Transforms use output-to-input sampling coordinates about image centre, in
latent pixels. The authoritative prefix is never modified. No extra model or
VAE call is performed.
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
_SCALE_FLOOR = 0.005
_TRANSLATION_FLOOR = 0.25
_MAX_SCALE = math.log(1.03)
_COMBINED_EVIDENCE_THRESHOLD = 2.5
# The previous two-domain gate used a quadrature threshold of 2.5. Before the
# learned upscaler there is intentionally no independent target-domain sample,
# so require the equal-contribution per-domain equivalent instead of inventing
# a new tuned threshold. Safe temporal-state evidence and post-warp reduction
# remain mandatory as separate gates.
_MIN_SOURCE_EVIDENCE = _COMBINED_EVIDENCE_THRESHOLD / math.sqrt(2.0)


def invert_transform(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = (float(value) for value in transform)
    if not all(math.isfinite(value) for value in (sx, sy, tx, ty)) or min(sx, sy) <= 0:
        raise ValueError("cannot invert invalid geometric transform")
    return (1.0 / sx, 1.0 / sy, -tx / sx, -ty / sy)


def compose_transform(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> tuple[float, float, float, float]:
    sx1, sy1, tx1, ty1 = first
    sx2, sy2, tx2, ty2 = second
    return (sx1 * sx2, sy1 * sy2, sx1 * tx2 + tx1, sy1 * ty2 + ty1)


def residual_transform(
    observed: tuple[float, ...],
    expected: tuple[float, ...],
) -> tuple[float, float, float, float]:
    return compose_transform(observed, invert_transform(expected))


def _signed_components(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    return (math.log(transform[0]), math.log(transform[1]), transform[2], transform[3])


def _features(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, c, t, h, w = video.shape
    features = video.detach().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    features = F.avg_pool2d(F.pad(features, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    ratio = min(1.0, 48.0 / max(h, w))
    features = F.interpolate(
        features,
        size=(round(h * ratio), round(w * ratio)),
        mode="area",
    )
    features = features.cpu()
    features = features - features.mean((-2, -1), keepdim=True)
    rms = features.square().mean((-2, -1), keepdim=True).sqrt()
    return features / rms.clamp_min(1e-4), rms


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
    ref_crop = reference[..., 3:-3, 3:-3]
    warped_crop = warped[..., 3:-3, 3:-3]
    ref_crop = ref_crop - ref_crop.mean((-2, -1), keepdims=True)
    warped_crop = warped_crop - warped_crop.mean((-2, -1), keepdims=True)
    ref_crop = ref_crop / np.maximum(
        np.sqrt((ref_crop * ref_crop).mean((-2, -1), keepdims=True)),
        1e-4,
    )
    warped_crop = warped_crop / np.maximum(
        np.sqrt((warped_crop * warped_crop).mean((-2, -1), keepdims=True)),
        1e-4,
    )
    return np.minimum((ref_crop - warped_crop) ** 2, 4).mean((1, 2, 3))


def _fit(a, z, hw):
    theta = list(IDENTITY)
    loss = float(_errors(a, z, theta, hw).mean())
    bounds = (0.06, 0.06, 3.0, 3.0)
    for scale_step, translation_step in ((0.02, 1.0), (0.01, 0.5), (0.005, 0.25), (0.0025, 0.125)):
        for _ in range(3):
            changed = False
            for axis, step in enumerate((scale_step, scale_step, translation_step, translation_step)):
                best = theta
                best_loss = loss
                for sign in (-1, 1):
                    candidate = theta.copy()
                    candidate[axis] += sign * step
                    if abs(candidate[axis] - IDENTITY[axis]) > bounds[axis] + 1e-8:
                        continue
                    value = float(_errors(a, z, candidate, hw).mean())
                    if value < best_loss - 1e-8:
                        best = candidate
                        best_loss = value
                if best is not theta:
                    changed = True
                theta, loss = best, best_loss
            if not changed:
                break
    return tuple(theta), loss


def _estimate_registration(
    reference: torch.Tensor,
    moving: torch.Tensor,
    *,
    comparison: tuple[float, ...] | None = None,
) -> dict:
    report = {
        "accepted": False,
        "reason": "invalid_geometry",
        "transform": IDENTITY,
        "identity_error": None,
        "aligned_error": None,
        "improvement": 0.0,
    }
    if reference.ndim != 5 or reference.shape != moving.shape or reference.shape[0] != 1:
        return report
    hw = tuple(reference.shape[-2:])
    if min(hw) < 16 or not reference.is_floating_point() or not moving.is_floating_point():
        report["reason"] = "unsupported_geometry"
        return report
    reference = reference[:, :, -3:]
    moving = moving[:, :, -3:]
    if not bool(torch.isfinite(reference).all() and torch.isfinite(moving).all()):
        report["reason"] = "nonfinite_input"
        return report
    ref_features, ref_rms = _features(reference)
    moving_features, moving_rms = _features(moving)
    informative = (ref_rms > 1e-4) & (moving_rms > 1e-4)
    if float(informative.float().mean()) < 0.5:
        report["reason"] = "insufficient_structure"
        return report
    grad_x = (ref_features[..., 1:] - ref_features[..., :-1]).square().mean()
    grad_y = (ref_features[..., 1:, :] - ref_features[..., :-1, :]).square().mean()
    if min(float(grad_x), float(grad_y)) < 1e-3:
        report["reason"] = "insufficient_axis_structure"
        return report
    ref_np = ref_features.numpy()
    moving_np = moving_features.numpy()
    identity = float(_errors(ref_np, moving_np, IDENTITY, hw).mean())
    theta, error = _fit(ref_np, moving_np, hw)
    improvement = max(0.0, (identity - error) / max(identity, 1e-8))
    report.update(
        transform=theta,
        identity_error=identity,
        aligned_error=error,
        improvement=improvement,
        paired_tokens=ref_np.shape[0],
    )
    if comparison is not None:
        comparison = tuple(float(value) for value in comparison)
        comparison_error = float(_errors(ref_np, moving_np, comparison, hw).mean())
        axis_gains = []
        for axis in range(4):
            probe = list(theta)
            probe[axis] = comparison[axis]
            axis_gains.append(float(_errors(ref_np, moving_np, probe, hw).mean()) - error)
        report.update(
            comparison_transform=comparison,
            comparison_error=comparison_error,
            comparison_improvement=max(
                0.0,
                (comparison_error - error) / max(comparison_error, 1e-8),
            ),
            residual_axis_gains=axis_gains,
        )
    report["reason"] = "measured"
    return report


def register_pair(
    reference: torch.Tensor,
    moving: torch.Tensor,
    *,
    comparison: tuple[float, ...] | None = None,
) -> dict:
    report = _estimate_registration(reference, moving, comparison=comparison)
    if report["reason"] == "measured":
        report["reason"] = "single_transition_diagnostic"
    return report


def _theil_sen_predict(
    samples: list[tuple[int, float]],
    next_index: int,
) -> tuple[float, float, float, float]:
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


def _motion_model(video: torch.Tensor, prefix_t: int) -> dict:
    p = int(prefix_t)
    transitions = []
    start = max(1, p - MOTION_BASELINE_TRANSITIONS)
    for index in range(start, p):
        item = register_pair(video[:, :, index - 1 : index], video[:, :, index : index + 1])
        item["transition_index"] = index
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
    params: list[list[tuple[int, float]]] = [[], [], [], []]
    for item in usable:
        index = int(item["transition_index"])
        sx, sy, tx, ty = item["transform"]
        values = (math.log(sx), math.log(sy), tx, ty)
        for axis, value in enumerate(values):
            params[axis].append((index, value))
    fits = [_theil_sen_predict(values, p) for values in params]
    expected = (
        math.exp(fits[0][0]),
        math.exp(fits[1][0]),
        fits[2][0],
        fits[3][0],
    )
    report.update(
        accepted=True,
        reason="robust_recent_motion",
        expected_transform=expected,
        trend=tuple(item[1] for item in fits),
        dispersion=tuple(item[2] for item in fits),
        intercept=tuple(item[3] for item in fits),
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
    if (
        comparison_error is None
        or boundary["aligned_error"] > 0.25
        or comparison_error <= boundary["aligned_error"] + 1e-12
    ):
        report.update(reason="boundary_alignment_unusable", raw_transform=correction)
        return report

    dispersions = tuple(float(value) for value in motion["dispersion"])
    axis_gains = tuple(float(value) for value in boundary.get("residual_axis_gains", [0.0] * 4))
    h, w = hw
    signed = _signed_components(correction)
    magnitudes = tuple(abs(value) for value in signed)
    floors = (_SCALE_FLOOR, _SCALE_FLOOR, _TRANSLATION_FLOOR, _TRANSLATION_FLOOR)
    evidence_units = tuple(max(floors[index], dispersions[index]) for index in range(4))
    evidence_scores = tuple(magnitudes[index] / evidence_units[index] for index in range(4))
    gain_floor = max(1e-4, 0.002 * comparison_error)
    safety = (_MAX_SCALE, _MAX_SCALE, min(1.5, w * 0.025), min(1.5, h * 0.025))
    axis_applied = [
        magnitudes[index] >= floors[index]
        and axis_gains[index] >= gain_floor
        and magnitudes[index] <= safety[index] + 1e-8
        for index in range(4)
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
        axis_objective_gain=axis_gains,
        axis_objective_gain_floor=gain_floor,
    )
    if not any(axis_applied):
        report["reason"] = "motion_residual_not_provisional"
        return report
    report.update(
        accepted=True,
        provisional=True,
        reason="provisional_motion_residual",
    )
    return report


def _temporal_axis_state(
    boundary_signed_residual: float,
    followup_signed_residuals: list[float],
    *,
    estimator_floor: float,
    suffix_length: int,
) -> dict:
    boundary = float(boundary_signed_residual)
    floor = float(estimator_floor)
    report = {
        "accepted": False,
        "mode": "ambiguous",
        "reason": "invalid_temporal_state",
        "boundary_signed_residual": boundary,
        "followup_signed_residuals": [float(value) for value in followup_signed_residuals],
        "cumulative_signed_state": [],
        "normalized_raw_envelope": [],
        "monotonic_envelope": [],
        "estimator_floor": floor,
        "recovery_suffix_token": None,
        "active_tokens": 0,
        "applied_envelope": {"kind": "none", "active_tokens": 0},
    }
    if suffix_length < 1 or floor <= 0 or not math.isfinite(boundary) or abs(boundary) < floor:
        return report
    if any(not math.isfinite(float(value)) for value in followup_signed_residuals):
        report["reason"] = "nonfinite_followup_residual"
        return report

    states = [boundary]
    for residual in followup_signed_residuals:
        states.append(states[-1] + float(residual))

    raw = [1.0]
    recovery_index = None
    best_abs = abs(boundary)
    unsafe_reason = None
    for index, state in enumerate(states[1:], start=1):
        if recovery_index is not None:
            raw.append(0.0)
            continue
        if state * boundary <= 0 or abs(state) <= floor:
            recovery_index = index
            raw.append(0.0)
            continue

        state_abs = abs(state)
        if state_abs > abs(boundary) + floor:
            unsafe_reason = "temporal_state_diverged"
        elif state_abs > best_abs + floor:
            unsafe_reason = "temporal_state_reversed_away_from_recovery"
        best_abs = min(best_abs, state_abs)
        raw.append(max(0.0, min(1.0, state / boundary)))

    monotonic = [raw[0]]
    for value in raw[1:]:
        monotonic.append(min(monotonic[-1], value))
    report.update(
        cumulative_signed_state=states,
        normalized_raw_envelope=raw,
        monotonic_envelope=monotonic,
        recovery_suffix_token=recovery_index,
    )

    if unsafe_reason is not None:
        report["reason"] = unsafe_reason
        return report

    if recovery_index is not None:
        weights = monotonic[: recovery_index + 1]
        active_tokens = min(
            suffix_length,
            max(
                (index + 1 for index, weight in enumerate(weights) if weight > 0.0),
                default=0,
            ),
        )
        report.update(
            accepted=True,
            mode="recovering",
            reason="observed_recovery",
            active_tokens=active_tokens,
            applied_envelope={
                "kind": "measured_monotonic",
                "weights": weights,
                "active_tokens": active_tokens,
                "terminal_zero_observed": True,
            },
        )
        return report

    if len(followup_signed_residuals) < 2:
        report["reason"] = "insufficient_temporal_evidence"
        return report

    if all(abs(state - boundary) <= floor for state in states[1:]):
        report.update(
            accepted=True,
            mode="persistent",
            reason="persistent_state_within_estimator_floor",
            active_tokens=suffix_length,
            applied_envelope={
                "kind": "constant",
                "value": 1.0,
                "active_tokens": suffix_length,
            },
        )
        return report

    report["reason"] = "unresolved_temporal_drift"
    return report


def _temporal_profile(video: torch.Tensor, prefix_t: int, motion: dict, candidate: dict) -> dict:
    report = {
        "accepted": False,
        "reason": "candidate_unavailable",
        "transitions": [],
        "usable_transitions": 0,
        "contiguous_usable_transitions": 0,
        "axis_accepted": [False] * 4,
        "axis_mode": ["inactive"] * 4,
        "axis": [],
    }
    if not candidate.get("accepted") or not motion.get("accepted"):
        return report
    available = int(video.shape[2]) - prefix_t - 1
    if available < 1:
        report["reason"] = "insufficient_suffix_temporal_evidence"
        return report

    transitions = []
    contiguous = []
    contiguous_open = True
    usable_count = 0
    for offset in range(min(PERSISTENCE_TRANSITIONS, available)):
        expected = _predict_motion_transform(motion, offset + 1)
        index = prefix_t + offset
        item = register_pair(
            video[:, :, index : index + 1],
            video[:, :, index + 1 : index + 2],
            comparison=expected,
        )
        item["suffix_transition_offset"] = offset + 1
        usable = (
            item.get("aligned_error") is not None
            and math.isfinite(float(item["aligned_error"]))
            and item["aligned_error"] <= 0.30
        )
        if usable:
            item["motion_residual_transform"] = residual_transform(item["transform"], expected)
            item["motion_residual_signed"] = _signed_components(item["motion_residual_transform"])
            usable_count += 1
        item["temporal_state_usable"] = usable
        if contiguous_open and usable:
            contiguous.append(item)
        elif not usable:
            contiguous_open = False
        transitions.append(item)

    floors = candidate["axis_estimator_floor"]
    boundary_signed = candidate["raw_signed_residual"]
    scores = candidate["axis_evidence_score"]
    suffix_length = int(video.shape[2]) - prefix_t
    axis_reports = []
    axis_accepted = []
    axis_modes = []
    for axis in range(4):
        provisional = bool(candidate["axis_applied"][axis])
        evidence = float(scores[axis])
        selected = provisional and evidence >= _MIN_SOURCE_EVIDENCE
        if not selected:
            axis_reports.append(
                {
                    "selected": False,
                    "accepted": False,
                    "mode": "inactive",
                    "reason": "source_evidence_below_authorization" if provisional else "axis_not_provisional",
                    "source_evidence_score": evidence,
                    "minimum_source_evidence": _MIN_SOURCE_EVIDENCE,
                }
            )
            axis_accepted.append(False)
            axis_modes.append("inactive")
            continue

        values = [float(item["motion_residual_signed"][axis]) for item in contiguous]
        axis_report = _temporal_axis_state(
            float(boundary_signed[axis]),
            values,
            estimator_floor=float(floors[axis]),
            suffix_length=suffix_length,
        )
        axis_report.update(
            selected=True,
            source_evidence_score=evidence,
            minimum_source_evidence=_MIN_SOURCE_EVIDENCE,
        )
        if len(contiguous) < len(transitions) and not axis_report.get("accepted"):
            axis_report["reason"] = "noncontiguous_temporal_evidence"
        axis_reports.append(axis_report)
        axis_accepted.append(bool(axis_report["accepted"]))
        axis_modes.append(axis_report["mode"])

    report.update(
        transitions=transitions,
        usable_transitions=usable_count,
        contiguous_usable_transitions=len(contiguous),
        axis=axis_reports,
        axis_accepted=axis_accepted,
        axis_mode=axis_modes,
    )
    if not any(axis_accepted):
        report["reason"] = "no_source_axis_temporally_authorized"
        return report
    report.update(accepted=True, reason="source_evidence_and_temporal_state_authorized")
    return report


def _axis_weight(axis_report: dict, token_index: int) -> float:
    if not axis_report.get("accepted"):
        return 0.0
    policy = axis_report.get("applied_envelope", {})
    if policy.get("kind") == "constant":
        if token_index < int(policy.get("active_tokens", 0)):
            return float(policy.get("value", 1.0))
        return 0.0
    if policy.get("kind") == "measured_monotonic":
        weights = policy.get("weights", [])
        if token_index < len(weights):
            return float(weights[token_index])
    return 0.0


def _effective_transform(
    base_transform: tuple[float, ...],
    axis_reports: list[dict],
    token_index: int,
) -> tuple[float, float, float, float]:
    weights = [_axis_weight(axis_reports[axis], token_index) for axis in range(4)]
    return (
        math.exp(weights[0] * math.log(base_transform[0])),
        math.exp(weights[1] * math.log(base_transform[1])),
        weights[2] * base_transform[2],
        weights[3] * base_transform[3],
    )


def _warp_suffix(
    video: torch.Tensor,
    prefix_t: int,
    base_transform: tuple[float, ...],
    profile: dict,
) -> tuple[torch.Tensor, dict]:
    suffix = video[:, :, prefix_t:]
    axis_reports = profile["axis"]
    active_t = max(
        (int(axis.get("active_tokens", 0)) for axis in axis_reports if axis.get("accepted")),
        default=0,
    )
    active_t = min(active_t, int(suffix.shape[2]))
    metadata = {
        "tokens_corrected": 0,
        "effective_active_temporal_length": 0,
        "applied_transforms": [],
        "applied_transform_sequence_length": 0,
        "applied_transforms_compact": False,
        "out_of_bounds_fraction": 0.0,
    }
    if active_t <= 0:
        return video, metadata

    transforms = [_effective_transform(base_transform, axis_reports, token) for token in range(active_t)]
    non_identity = [token for token, transform in enumerate(transforms) if transform != IDENTITY]
    if not non_identity:
        return video, metadata
    active_t = non_identity[-1] + 1
    transforms = transforms[:active_t]

    corrected = video.clone()
    b, c, _, h, w = suffix.shape
    active = suffix[:, :, :active_t]
    flat = active.permute(0, 2, 1, 3, 4).reshape(b * active_t, c, h, w)
    with torch.autocast(device_type=video.device.type, enabled=False):
        theta = torch.zeros((active_t, 2, 3), device=video.device, dtype=torch.float32)
        for token, (sx, sy, tx, ty) in enumerate(transforms):
            theta[token, 0, 0] = sx
            theta[token, 0, 2] = 2.0 * tx / w
            theta[token, 1, 1] = sy
            theta[token, 1, 2] = 2.0 * ty / h
        theta = theta.repeat(b, 1, 1)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        warped = F.grid_sample(
            flat.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(video.dtype)
        outside = (grid[..., 0].abs() > 1.0) | (grid[..., 1].abs() > 1.0)
        out_of_bounds = float(outside.float().mean())
    corrected[:, :, prefix_t : prefix_t + active_t] = warped.reshape(b, active_t, c, h, w).permute(
        0,
        2,
        1,
        3,
        4,
    )
    same_transform = all(transform == transforms[0] for transform in transforms[1:])
    metadata.update(
        tokens_corrected=active_t,
        effective_active_temporal_length=active_t,
        applied_transforms=[transforms[0]] if same_transform else transforms,
        applied_transform_sequence_length=active_t,
        applied_transforms_compact=bool(same_transform and active_t > 1),
        out_of_bounds_fraction=out_of_bounds,
    )
    return corrected, metadata


def disabled_source_trajectory_bridge_metrics(*, prefix_t: int, requested: bool = False) -> dict:
    p = int(prefix_t)
    if p < 1:
        raise ValueError("source trajectory bridge prefix length must be positive")
    return {
        "source_trajectory_bridge_version": 1,
        "source_trajectory_bridge_requested": bool(requested),
        "source_trajectory_bridge_enabled": False,
        "source_trajectory_bridge_accepted": False,
        "source_trajectory_bridge_reason": "disabled" if not requested else "not_applied",
        "source_trajectory_bridge_prefix_t": p,
        "source_trajectory_bridge_tokens_corrected": 0,
        "source_trajectory_bridge_effective_active_temporal_length": 0,
        "source_trajectory_bridge_applied_transforms": [],
        "source_trajectory_bridge_applied_transform_sequence_length": 0,
        "source_trajectory_bridge_applied_transforms_compact": False,
        "source_trajectory_bridge_out_of_bounds_fraction": 0.0,
        "source_trajectory_bridge_minimum_evidence": _MIN_SOURCE_EVIDENCE,
    }


@torch.no_grad()
def apply_source_trajectory_bridge(
    video: torch.Tensor,
    prefix_t: int,
    *,
    requested: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Close a measured source-grid boundary residual before learned upscaling."""
    if not torch.is_tensor(video) or video.ndim != 5 or not video.is_floating_point():
        raise TypeError("source trajectory bridge expects a floating BxCxTxHxW tensor")
    p = int(prefix_t)
    temporal = int(video.shape[2])
    if not 1 <= p < temporal:
        raise ValueError("source trajectory bridge requires a non-empty prefix and suffix")
    if not bool(torch.isfinite(video).all()):
        raise RuntimeError("source trajectory bridge input contains NaN or Inf")
    if not isinstance(requested, bool):
        raise TypeError("source trajectory bridge requested flag must be boolean")
    if not requested:
        return video, disabled_source_trajectory_bridge_metrics(prefix_t=p)
    if p < 3 or p >= temporal - 1:
        metrics = disabled_source_trajectory_bridge_metrics(prefix_t=p, requested=True)
        metrics["source_trajectory_bridge_reason"] = "insufficient_source_motion_window"
        return video, metrics

    started = time.perf_counter()
    motion = _motion_model(video, p)
    expected = motion.get("expected_transform") if motion.get("accepted") else None
    boundary = register_pair(
        video[:, :, p - 1 : p],
        video[:, :, p : p + 1],
        comparison=expected,
    )
    candidate = _motion_residual_candidate(boundary, motion, tuple(video.shape[-2:]))
    profile = _temporal_profile(video, p, motion, candidate)
    authorized = profile.get("axis_accepted", [False] * 4)
    candidate_transform = candidate.get("transform", IDENTITY)
    base_transform = (
        candidate_transform[0] if authorized[0] else 1.0,
        candidate_transform[1] if authorized[1] else 1.0,
        candidate_transform[2] if authorized[2] else 0.0,
        candidate_transform[3] if authorized[3] else 0.0,
    )
    metrics = disabled_source_trajectory_bridge_metrics(prefix_t=p, requested=True)
    metrics.update(
        source_trajectory_natural_motion_model=motion,
        source_trajectory_boundary_before=boundary,
        source_trajectory_motion_residual=candidate,
        source_trajectory_temporal_profile=profile,
        source_trajectory_axis_authorized=authorized,
        source_trajectory_base_transform=base_transform,
    )
    if not motion.get("accepted"):
        metrics["source_trajectory_bridge_reason"] = motion["reason"]
        return video, metrics
    if not candidate.get("accepted"):
        metrics["source_trajectory_bridge_reason"] = candidate["reason"]
        return video, metrics
    if not profile.get("accepted"):
        metrics["source_trajectory_bridge_reason"] = profile["reason"]
        return video, metrics

    corrected, warp = _warp_suffix(video, p, base_transform, profile)
    if corrected is video:
        metrics["source_trajectory_bridge_reason"] = "no_nonidentity_source_transform"
        return video, metrics

    after = register_pair(
        corrected[:, :, p - 1 : p],
        corrected[:, :, p : p + 1],
        comparison=expected,
    )
    before_signed = candidate.get("raw_signed_residual", _signed_components(base_transform))
    after_signed = None
    if after.get("aligned_error") is not None and expected is not None:
        after_signed = _signed_components(residual_transform(after["transform"], expected))

    reductions = []
    verification_ok = after_signed is not None
    floors = candidate.get(
        "axis_estimator_floor",
        (_SCALE_FLOOR, _SCALE_FLOOR, _TRANSLATION_FLOOR, _TRANSLATION_FLOOR),
    )
    for axis in range(4):
        if not authorized[axis]:
            reductions.append(None)
            continue
        if after_signed is None:
            verification_ok = False
            reductions.append(None)
            continue
        before_abs = abs(float(before_signed[axis]))
        after_abs = abs(float(after_signed[axis]))
        reductions.append(after_abs / max(before_abs, 1e-12))
        if not after_abs < before_abs:
            verification_ok = False
        if float(after_signed[axis]) * float(before_signed[axis]) < 0 and after_abs > float(floors[axis]):
            verification_ok = False

    metrics.update(
        source_trajectory_boundary_after=after,
        source_trajectory_post_signed_residual=after_signed,
        source_trajectory_residual_reduction_ratio=reductions,
        source_trajectory_bridge_tokens_corrected=int(warp["tokens_corrected"]),
        source_trajectory_bridge_effective_active_temporal_length=int(
            warp["effective_active_temporal_length"]
        ),
        source_trajectory_bridge_applied_transforms=warp["applied_transforms"],
        source_trajectory_bridge_applied_transform_sequence_length=int(
            warp["applied_transform_sequence_length"]
        ),
        source_trajectory_bridge_applied_transforms_compact=bool(warp["applied_transforms_compact"]),
        source_trajectory_bridge_out_of_bounds_fraction=float(warp["out_of_bounds_fraction"]),
        source_trajectory_bridge_elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    if not torch.equal(corrected[:, :, :p], video[:, :, :p]):
        raise RuntimeError("source trajectory bridge modified protected source prefix")
    active = int(warp["tokens_corrected"])
    if not torch.equal(corrected[:, :, p + active :], video[:, :, p + active :]):
        raise RuntimeError("source trajectory bridge modified recovered later suffix")
    if not verification_ok:
        metrics["source_trajectory_bridge_reason"] = "post_warp_residual_verification_failed"
        return video, metrics

    metrics.update(
        source_trajectory_bridge_enabled=True,
        source_trajectory_bridge_accepted=True,
        source_trajectory_bridge_reason="source_trajectory_residual_closed_before_upscale",
    )
    return corrected, metrics
