"""Persistent target-grid suffix rebase for Mixed-Grid Continuum seams.

Runs 00390/00391 separated two independent handoff defects:

* a persistent latent tone/DC domain offset between the authoritative exact
  prefix and learned-upscaler suffix; and
* a persistent spatial frame-coordinate jump that remains after the tone/DC
  offset is removed.

The tone fix from v4 is retained unchanged. Geometry v5 no longer asks the
learned/discarded upscaler prefix to define expected motion. Instead it learns a
robust recent motion law from the caller-owned exact target-grid prefix, holds
out the latest exact-prefix transition to validate that law, then measures the
exact-prefix -> tone-corrected suffix boundary against the validated next-motion
prediction. A bounded residual transform may be applied to the complete suffix.

The bridge never modifies the protected prefix, transplants a raw structural
residual, adds RNG/model/VAE work, or introduces a finite correction horizon.
"""

from __future__ import annotations

import math
from statistics import median

import torch
import torch.nn.functional as F

from .source_trajectory_bridge import register_pair

_VERSION = 5
_MODE = "persistent_suffix_rebase_v2_authoritative_motion"
_OVERLAP_FRAMES = 4
_MOTION_FRAMES = 8
_SCALE_FLOOR = 0.005
_TRANSLATION_FLOOR = 0.25
_MAX_LOG_SCALE = math.log(1.03)
_MIN_GEOMETRY_IMPROVEMENT = 0.01
_MAX_INTERNAL_GEOMETRY_REGRESSION = 0.03
_MIN_TONE_DC_IMPROVEMENT = 0.20
_MAX_TONE_TOTAL_REGRESSION = 0.01


def _rms(value: torch.Tensor) -> float:
    result = float(value.float().square().mean().sqrt().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("representation bridge produced a non-finite RMS")
    return result


def _validate_pair(learned: torch.Tensor, exact_prefix: torch.Tensor) -> int:
    if not torch.is_tensor(learned) or not torch.is_tensor(exact_prefix):
        raise TypeError("representation bridge expects torch.Tensor inputs")
    if learned.ndim != 5 or exact_prefix.ndim != 5:
        raise ValueError("representation bridge expects BxCxTxHxW tensors")
    if not learned.is_floating_point() or not exact_prefix.is_floating_point():
        raise TypeError("representation bridge expects floating-point tensors")
    if learned.shape[:2] != exact_prefix.shape[:2] or learned.shape[-2:] != exact_prefix.shape[-2:]:
        raise ValueError("representation bridge learned/exact geometry differs")
    prefix_t = int(exact_prefix.shape[2])
    if prefix_t < 1 or prefix_t >= int(learned.shape[2]):
        raise ValueError("representation bridge requires a non-empty prefix and suffix")
    if not bool(torch.isfinite(learned).all().item()) or not bool(torch.isfinite(exact_prefix).all().item()):
        raise RuntimeError("representation bridge inputs contain NaN or Inf")
    return prefix_t


def disabled_suffix_representation_bridge_metrics(*, prefix_t: int, requested: bool = False) -> dict:
    p = int(prefix_t)
    if p < 1:
        raise ValueError("representation bridge prefix length must be positive")
    return {
        "suffix_representation_bridge_version": _VERSION,
        "suffix_representation_bridge_requested": bool(requested),
        "suffix_representation_bridge_enabled": False,
        "suffix_representation_bridge_accepted": False,
        "suffix_representation_bridge_reason": "disabled" if not requested else "not_applied",
        "suffix_representation_bridge_mode": _MODE,
        "suffix_representation_bridge_prefix_t": p,
        "suffix_representation_bridge_corrected_tokens": 0,
        "suffix_representation_bridge_delta_rms": 0.0,
        "suffix_representation_bridge_dc_rms": 0.0,
        "suffix_representation_bridge_structural_rms": 0.0,
        "suffix_representation_bridge_centered_error_before": 0.0,
        "suffix_representation_bridge_centered_error_after": 0.0,
        "suffix_representation_bridge_centered_error_ratio": 1.0,
        "suffix_representation_bridge_raw_structural_residual_transplanted": False,
        "suffix_representation_bridge_overlap_frames": 0,
        "suffix_representation_bridge_consensus_frames": 0,
        "suffix_representation_bridge_validation_frames": 0,
        "suffix_representation_bridge_improving_frames": 0,
        "suffix_representation_bridge_median_frame_improvement": 0.0,
        "suffix_representation_bridge_worst_frame_regression": 0.0,
        "suffix_representation_bridge_transform": (1.0, 1.0, 0.0, 0.0),
        "suffix_representation_bridge_identity_error": None,
        "suffix_representation_bridge_aligned_error": None,
        "suffix_representation_bridge_improvement": 0.0,
        "suffix_representation_bridge_out_of_bounds_fraction": 0.0,
        "suffix_representation_bridge_tone_bias_accepted": False,
        "suffix_representation_bridge_tone_reason": "disabled" if not requested else "not_applied",
        "suffix_representation_bridge_tone_bias_rms": 0.0,
        "suffix_representation_bridge_tone_validation_rms_before": None,
        "suffix_representation_bridge_tone_validation_rms_after": None,
        "suffix_representation_bridge_tone_validation_dc_rms_before": None,
        "suffix_representation_bridge_tone_validation_dc_rms_after": None,
        "suffix_representation_bridge_tone_validation_dc_improvement": 0.0,
        "suffix_representation_bridge_geometry_accepted": False,
        "suffix_representation_bridge_geometry_reason": "disabled" if not requested else "not_applied",
        "suffix_representation_bridge_geometry_training_transitions": 0,
        "suffix_representation_bridge_geometry_holdout_identity_error": None,
        "suffix_representation_bridge_geometry_holdout_aligned_error": None,
        "suffix_representation_bridge_geometry_holdout_prediction_error": None,
        "suffix_representation_bridge_geometry_holdout_prediction_gain": 0.0,
        "suffix_representation_bridge_geometry_expected_transform": (1.0, 1.0, 0.0, 0.0),
        "suffix_representation_bridge_geometry_observed_transform": (1.0, 1.0, 0.0, 0.0),
        "suffix_representation_bridge_geometry_residual_transform": (1.0, 1.0, 0.0, 0.0),
        "suffix_representation_bridge_geometry_boundary_error_before": None,
        "suffix_representation_bridge_geometry_boundary_error_after": None,
        "suffix_representation_bridge_geometry_internal_validation_transitions": 0,
        "suffix_representation_bridge_geometry_worst_internal_regression": 0.0,
    }


def _signed_transform(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = (float(value) for value in transform)
    if sx <= 0.0 or sy <= 0.0 or not all(math.isfinite(value) for value in (sx, sy, tx, ty)):
        raise ValueError("representation bridge received an invalid registration transform")
    return math.log(sx), math.log(sy), tx, ty


def _from_signed(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return math.exp(values[0]), math.exp(values[1]), values[2], values[3]


def _invert_transform(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = (float(value) for value in transform)
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("cannot invert invalid transform")
    return 1.0 / sx, 1.0 / sy, -tx / sx, -ty / sy


def _compose_transform(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> tuple[float, float, float, float]:
    sx1, sy1, tx1, ty1 = map(float, first)
    sx2, sy2, tx2, ty2 = map(float, second)
    return sx1 * sx2, sy1 * sy2, sx1 * tx2 + tx1, sy1 * ty2 + ty1


def _residual_transform(
    observed: tuple[float, ...],
    expected: tuple[float, ...],
) -> tuple[float, float, float, float]:
    return _compose_transform(observed, _invert_transform(expected))


def _warp_video_constant(
    video: torch.Tensor,
    transform: tuple[float, float, float, float],
) -> tuple[torch.Tensor, float]:
    b, c, t, h, w = video.shape
    if t == 0:
        return video.clone(), 0.0
    sx, sy, tx, ty = transform
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    with torch.autocast(device_type=video.device.type, enabled=False):
        theta = torch.zeros((b * t, 2, 3), device=video.device, dtype=torch.float32)
        theta[:, 0, 0] = float(sx)
        theta[:, 0, 2] = 2.0 * float(tx) / float(w)
        theta[:, 1, 1] = float(sy)
        theta[:, 1, 2] = 2.0 * float(ty) / float(h)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        warped = F.grid_sample(
            flat.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(video.dtype)
        outside = float(((grid[..., 0].abs() > 1.0) | (grid[..., 1].abs() > 1.0)).float().mean().item())
    return warped.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4), outside


def _registration_transform(item: dict) -> tuple[float, float, float, float] | None:
    transform = item.get("transform")
    aligned = item.get("aligned_error")
    if transform is None or aligned is None or not math.isfinite(float(aligned)):
        return None
    try:
        return tuple(float(value) for value in transform)
    except (TypeError, ValueError):
        return None


def _theil_sen_predict(samples: list[tuple[int, float]], target_index: int) -> float:
    if len(samples) < 2:
        raise ValueError("motion predictor requires at least two samples")
    slopes = [
        (samples[j][1] - samples[i][1]) / (samples[j][0] - samples[i][0])
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
        if samples[j][0] != samples[i][0]
    ]
    slope = float(median(slopes)) if slopes else 0.0
    intercept = float(median([value - slope * index for index, value in samples]))
    return intercept + slope * int(target_index)


def _predict_transform(samples: list[tuple[int, tuple[float, float, float, float]]], target_index: int):
    signed = [(_index, _signed_transform(transform)) for _index, transform in samples]
    values = tuple(
        _theil_sen_predict([(index, vector[axis]) for index, vector in signed], target_index)
        for axis in range(4)
    )
    return _from_signed(values)


def _persistent_tone_bias(learned_prefix: torch.Tensor, exact_prefix: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
    overlap = min(_OVERLAP_FRAMES, int(exact_prefix.shape[2]))
    report = {
        "accepted": False,
        "reason": "insufficient_overlap",
        "overlap_frames": overlap,
        "bias_rms": 0.0,
        "validation_rms_before": None,
        "validation_rms_after": None,
        "validation_dc_rms_before": None,
        "validation_dc_rms_after": None,
        "validation_dc_improvement": 0.0,
    }
    if overlap < 3:
        return None, report

    learned = learned_prefix[:, :, -overlap:].float()
    exact = exact_prefix[:, :, -overlap:].float()
    frame_bias = (exact - learned).mean(dim=(-2, -1))
    train = frame_bias[:, :, :-1]
    bias = train.median(dim=2).values[:, :, None, None, None]
    held_learned = learned[:, :, -1:]
    held_exact = exact[:, :, -1:]
    before_delta = held_exact - held_learned
    after_delta = held_exact - (held_learned + bias)
    before = _rms(before_delta)
    after = _rms(after_delta)
    before_dc = _rms(before_delta.mean(dim=(-2, -1), keepdim=True))
    after_dc = _rms(after_delta.mean(dim=(-2, -1), keepdim=True))
    dc_improvement = (before_dc - after_dc) / max(before_dc, 1e-12)
    bias_rms = _rms(bias)
    report.update(
        bias_rms=bias_rms,
        validation_rms_before=before,
        validation_rms_after=after,
        validation_dc_rms_before=before_dc,
        validation_dc_rms_after=after_dc,
        validation_dc_improvement=max(0.0, dc_improvement),
    )
    if bias_rms <= 1e-6 or before_dc <= 1e-6:
        report["reason"] = "overlap_tone_already_matched"
        return None, report
    if dc_improvement < _MIN_TONE_DC_IMPROVEMENT:
        report["reason"] = "heldout_tone_gain_too_small"
        return None, report
    if after > before * (1.0 + _MAX_TONE_TOTAL_REGRESSION):
        report["reason"] = "heldout_tone_total_regression"
        return None, report
    report.update(accepted=True, reason="heldout_persistent_channel_bias_authorized")
    return bias, report


def _authoritative_motion_prediction(exact_prefix: torch.Tensor) -> tuple[tuple[float, float, float, float] | None, dict]:
    identity = (1.0, 1.0, 0.0, 0.0)
    prefix_t = int(exact_prefix.shape[2])
    report = {
        "accepted": False,
        "reason": "insufficient_authoritative_motion_history",
        "training_transitions": 0,
        "holdout_identity_error": None,
        "holdout_aligned_error": None,
        "holdout_prediction_error": None,
        "holdout_prediction_gain": 0.0,
        "expected_transform": identity,
    }
    if prefix_t < 5:
        return None, report

    start = max(1, prefix_t - _MOTION_FRAMES + 1)
    holdout_index = prefix_t - 1
    samples: list[tuple[int, tuple[float, float, float, float]]] = []
    for index in range(start, holdout_index):
        item = register_pair(
            exact_prefix[:, :, index - 1 : index],
            exact_prefix[:, :, index : index + 1],
        )
        transform = _registration_transform(item)
        identity_error = item.get("identity_error")
        if transform is None or identity_error is None or not math.isfinite(float(identity_error)):
            continue
        samples.append((index, transform))

    report["training_transitions"] = len(samples)
    if len(samples) < 3 or samples[-1][0] - samples[0][0] < 2:
        return None, report

    holdout_prediction = _predict_transform(samples, holdout_index)
    holdout = register_pair(
        exact_prefix[:, :, holdout_index - 1 : holdout_index],
        exact_prefix[:, :, holdout_index : holdout_index + 1],
        comparison=holdout_prediction,
    )
    identity_error = holdout.get("identity_error")
    aligned_error = holdout.get("aligned_error")
    prediction_error = holdout.get("comparison_error")
    if any(value is None or not math.isfinite(float(value)) for value in (identity_error, aligned_error, prediction_error)):
        report["reason"] = "authoritative_motion_holdout_unavailable"
        return None, report

    identity_error = float(identity_error)
    aligned_error = float(aligned_error)
    prediction_error = float(prediction_error)
    prediction_gain = (identity_error - prediction_error) / max(identity_error, 1e-12)
    report.update(
        holdout_identity_error=identity_error,
        holdout_aligned_error=aligned_error,
        holdout_prediction_error=prediction_error,
        holdout_prediction_gain=prediction_gain,
    )
    if prediction_error > identity_error + 1e-8:
        report["reason"] = "authoritative_motion_holdout_regression"
        return None, report

    expected = _predict_transform(samples + [(holdout_index, _registration_transform(holdout) or identity)], prefix_t)
    report.update(accepted=True, reason="authoritative_motion_holdout_validated", expected_transform=expected)
    return expected, report


def _persistent_geometry_rebase(
    exact_prefix: torch.Tensor,
    suffix: torch.Tensor,
) -> tuple[tuple[float, float, float, float] | None, dict]:
    identity = (1.0, 1.0, 0.0, 0.0)
    report = {
        "accepted": False,
        "reason": "insufficient_authoritative_motion_history",
        "training_transitions": 0,
        "holdout_identity_error": None,
        "holdout_aligned_error": None,
        "holdout_prediction_error": None,
        "holdout_prediction_gain": 0.0,
        "expected_transform": identity,
        "observed_transform": identity,
        "residual_transform": identity,
        "boundary_error_before": None,
        "boundary_error_after": None,
        "improvement": 0.0,
        "validation_transitions": 0,
        "improving_transitions": 0,
        "median_transition_improvement": 0.0,
        "worst_internal_regression": 0.0,
    }
    if int(suffix.shape[2]) < 2 or min(suffix.shape[-2:]) < 16:
        return None, report

    expected, motion = _authoritative_motion_prediction(exact_prefix)
    report.update(
        reason=motion["reason"],
        training_transitions=int(motion["training_transitions"]),
        holdout_identity_error=motion["holdout_identity_error"],
        holdout_aligned_error=motion["holdout_aligned_error"],
        holdout_prediction_error=motion["holdout_prediction_error"],
        holdout_prediction_gain=float(motion["holdout_prediction_gain"]),
        expected_transform=motion["expected_transform"],
    )
    if expected is None:
        return None, report

    observed_item = register_pair(
        exact_prefix[:, :, -1:],
        suffix[:, :, :1],
        comparison=expected,
    )
    observed = _registration_transform(observed_item)
    comparison_error = observed_item.get("comparison_error")
    if observed is None or comparison_error is None or not math.isfinite(float(comparison_error)):
        report["reason"] = "authoritative_boundary_registration_unavailable"
        return None, report

    residual = _residual_transform(observed, expected)
    signed = _signed_transform(residual)
    h, w = map(int, suffix.shape[-2:])
    floors = (_SCALE_FLOOR, _SCALE_FLOOR, _TRANSLATION_FLOOR, _TRANSLATION_FLOOR)
    safety = (_MAX_LOG_SCALE, _MAX_LOG_SCALE, min(1.5, 0.025 * w), min(1.5, 0.025 * h))
    active = [abs(signed[axis]) >= floors[axis] for axis in range(4)]
    report.update(observed_transform=observed, residual_transform=residual)
    if not any(active):
        report["reason"] = "authoritative_boundary_motion_consistent"
        return None, report
    if any(active[axis] and abs(signed[axis]) > safety[axis] + 1e-8 for axis in range(4)):
        report["reason"] = "authoritative_boundary_residual_exceeds_safety_bound"
        return None, report

    filtered = tuple(signed[axis] if active[axis] else 0.0 for axis in range(4))
    correction = _from_signed(filtered)
    warped_suffix, _ = _warp_video_constant(suffix, correction)
    after_item = register_pair(
        exact_prefix[:, :, -1:],
        warped_suffix[:, :, :1],
        comparison=expected,
    )
    before_error = float(comparison_error)
    after_error_value = after_item.get("comparison_error")
    if after_error_value is None or not math.isfinite(float(after_error_value)):
        report["reason"] = "corrected_authoritative_boundary_registration_unavailable"
        return None, report
    after_error = float(after_error_value)
    improvement = (before_error - after_error) / max(before_error, 1e-12)
    report.update(
        residual_transform=correction,
        boundary_error_before=before_error,
        boundary_error_after=after_error,
        improvement=max(0.0, improvement),
    )
    if improvement < _MIN_GEOMETRY_IMPROVEMENT:
        report["reason"] = "authoritative_boundary_geometry_gain_too_small"
        return None, report

    transition_improvements: list[float] = []
    validation_count = min(3, int(suffix.shape[2]) - 1)
    for offset in range(validation_count):
        before_pair = register_pair(suffix[:, :, offset : offset + 1], suffix[:, :, offset + 1 : offset + 2])
        after_pair = register_pair(
            warped_suffix[:, :, offset : offset + 1],
            warped_suffix[:, :, offset + 1 : offset + 2],
        )
        before_aligned = before_pair.get("aligned_error")
        after_aligned = after_pair.get("aligned_error")
        if before_aligned is None or after_aligned is None:
            continue
        before_aligned = float(before_aligned)
        after_aligned = float(after_aligned)
        if not math.isfinite(before_aligned) or not math.isfinite(after_aligned) or before_aligned <= 0.0:
            continue
        transition_improvements.append((before_aligned - after_aligned) / before_aligned)

    report["validation_transitions"] = len(transition_improvements)
    if len(transition_improvements) < 2:
        report["reason"] = "insufficient_internal_geometry_validation"
        return None, report
    report["improving_transitions"] = sum(value >= 0.0 for value in transition_improvements)
    report["median_transition_improvement"] = float(median(transition_improvements))
    worst_regression = max(0.0, -min(transition_improvements))
    report["worst_internal_regression"] = worst_regression
    if worst_regression > _MAX_INTERNAL_GEOMETRY_REGRESSION:
        report["reason"] = "persistent_geometry_internal_regression"
        return None, report

    report.update(accepted=True, reason="authoritative_motion_suffix_rebase_authorized")
    return correction, report


@torch.no_grad()
def apply_suffix_representation_bridge(
    learned_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    requested: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Rebase a generated suffix persistently from directly observed overlap."""

    prefix_t = _validate_pair(learned_clean_video, exact_prefix)
    if not isinstance(requested, bool):
        raise TypeError("representation bridge requested flag must be boolean")
    if not requested:
        return learned_clean_video, disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t)

    exact = exact_prefix.to(device=learned_clean_video.device, dtype=learned_clean_video.dtype)
    learned_prefix = learned_clean_video[:, :, :prefix_t]
    learned_last = learned_prefix[:, :, -1:].float()
    exact_last = exact[:, :, -1:].float()
    delta = exact_last - learned_last
    dc = delta.mean(dim=(-2, -1), keepdim=True)
    structural = delta - dc
    before_centered = _rms(structural)

    metrics = disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t, requested=True)
    overlap = min(_OVERLAP_FRAMES, prefix_t)
    metrics.update(
        suffix_representation_bridge_delta_rms=_rms(delta),
        suffix_representation_bridge_dc_rms=_rms(dc),
        suffix_representation_bridge_structural_rms=before_centered,
        suffix_representation_bridge_centered_error_before=before_centered,
        suffix_representation_bridge_centered_error_after=before_centered,
        suffix_representation_bridge_centered_error_ratio=1.0,
        suffix_representation_bridge_overlap_frames=overlap,
    )

    tone_bias, tone = _persistent_tone_bias(learned_prefix, exact)
    suffix = learned_clean_video[:, :, prefix_t:]
    tone_corrected_suffix = suffix
    if tone_bias is not None:
        tone_corrected_suffix = suffix.float().add(tone_bias).to(learned_clean_video.dtype)

    geometry_transform, geometry = _persistent_geometry_rebase(exact, tone_corrected_suffix)
    corrected_suffix = tone_corrected_suffix
    out_of_bounds = 0.0
    if geometry_transform is not None:
        corrected_suffix, out_of_bounds = _warp_video_constant(corrected_suffix, geometry_transform)

    tone_accepted = bool(tone["accepted"])
    geometry_accepted = bool(geometry["accepted"])
    accepted = tone_accepted or geometry_accepted
    if accepted:
        corrected = learned_clean_video.clone()
        corrected[:, :, prefix_t:] = corrected_suffix
        if not torch.equal(corrected[:, :, :prefix_t], learned_clean_video[:, :, :prefix_t]):
            raise RuntimeError("representation bridge modified prefix values")
        if not bool(torch.isfinite(corrected).all().item()):
            raise RuntimeError("representation bridge produced NaN or Inf")
    else:
        corrected = learned_clean_video

    if tone_accepted and geometry_accepted:
        reason = "persistent_tone_and_authoritative_geometry_rebase_authorized"
    elif tone_accepted:
        reason = "persistent_tone_rebase_authorized"
    elif geometry_accepted:
        reason = "authoritative_geometry_rebase_authorized"
    else:
        reason = f"tone={tone['reason']};geometry={geometry['reason']}"

    validation_frames = 1 if tone["validation_rms_before"] is not None else 0
    improving_frames = int(tone_accepted)
    metrics.update(
        suffix_representation_bridge_enabled=accepted,
        suffix_representation_bridge_accepted=accepted,
        suffix_representation_bridge_reason=reason,
        suffix_representation_bridge_corrected_tokens=int(suffix.shape[2]) if accepted else 0,
        suffix_representation_bridge_consensus_frames=int(geometry.get("improving_transitions", 0)),
        suffix_representation_bridge_validation_frames=validation_frames,
        suffix_representation_bridge_improving_frames=improving_frames,
        suffix_representation_bridge_median_frame_improvement=float(tone["validation_dc_improvement"]),
        suffix_representation_bridge_worst_frame_regression=max(
            0.0,
            float(geometry.get("worst_internal_regression", 0.0)),
        ),
        suffix_representation_bridge_transform=geometry.get("residual_transform", (1.0, 1.0, 0.0, 0.0)),
        suffix_representation_bridge_identity_error=geometry.get("boundary_error_before"),
        suffix_representation_bridge_aligned_error=geometry.get("boundary_error_after"),
        suffix_representation_bridge_improvement=float(geometry.get("improvement", 0.0)),
        suffix_representation_bridge_out_of_bounds_fraction=out_of_bounds,
        suffix_representation_bridge_tone_bias_accepted=tone_accepted,
        suffix_representation_bridge_tone_reason=tone["reason"],
        suffix_representation_bridge_tone_bias_rms=float(tone["bias_rms"]),
        suffix_representation_bridge_tone_validation_rms_before=tone["validation_rms_before"],
        suffix_representation_bridge_tone_validation_rms_after=tone["validation_rms_after"],
        suffix_representation_bridge_tone_validation_dc_rms_before=tone["validation_dc_rms_before"],
        suffix_representation_bridge_tone_validation_dc_rms_after=tone["validation_dc_rms_after"],
        suffix_representation_bridge_tone_validation_dc_improvement=float(tone["validation_dc_improvement"]),
        suffix_representation_bridge_geometry_accepted=geometry_accepted,
        suffix_representation_bridge_geometry_reason=geometry["reason"],
        suffix_representation_bridge_geometry_training_transitions=int(geometry["training_transitions"]),
        suffix_representation_bridge_geometry_holdout_identity_error=geometry["holdout_identity_error"],
        suffix_representation_bridge_geometry_holdout_aligned_error=geometry["holdout_aligned_error"],
        suffix_representation_bridge_geometry_holdout_prediction_error=geometry["holdout_prediction_error"],
        suffix_representation_bridge_geometry_holdout_prediction_gain=float(geometry["holdout_prediction_gain"]),
        suffix_representation_bridge_geometry_expected_transform=geometry["expected_transform"],
        suffix_representation_bridge_geometry_observed_transform=geometry["observed_transform"],
        suffix_representation_bridge_geometry_residual_transform=geometry["residual_transform"],
        suffix_representation_bridge_geometry_boundary_error_before=geometry["boundary_error_before"],
        suffix_representation_bridge_geometry_boundary_error_after=geometry["boundary_error_after"],
        suffix_representation_bridge_geometry_internal_validation_transitions=int(geometry["validation_transitions"]),
        suffix_representation_bridge_geometry_worst_internal_regression=float(geometry["worst_internal_regression"]),
    )
    return corrected, metrics
