"""Bounded post-high video boundary residual correction.

This module is intentionally independent of H3 sampling. It consumes already
generated post-high video latents and can only translate a short generated-side
temporal support after the authoritative exact prefix.
"""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Any

import torch

from .frame_gauge import translate_video_cells
from .seam_diagnostics import measure_translation_trajectory

POST_HIGH_VIDEO_RESIDUAL_POLICY = "post_high_vertical_residual_release_v1"
POST_HIGH_VIDEO_RELEASE_WEIGHTS = (
    1.0,
    0.8535533905932737,
    0.5,
    0.14644660940672627,
)
POST_HIGH_VIDEO_MIN_RESPONSE = 6.0
POST_HIGH_VIDEO_MIN_BOUNDARY_ERROR_CELLS = 0.05
POST_HIGH_VIDEO_MIN_HIGH_DELTA_CELLS = 0.025
POST_HIGH_VIDEO_MIN_HIGH_DELTA_FRACTION = 0.40
POST_HIGH_VIDEO_MAX_PRE_HIGH_BOUNDARY_ERROR_CELLS = 0.125
POST_HIGH_VIDEO_MAX_ROI_DISAGREEMENT_CELLS = 0.20
POST_HIGH_VIDEO_MAX_CORRECTION_CELLS = 0.18
POST_HIGH_VIDEO_MAX_RELEASE_STEP_CELLS = 0.0625
POST_HIGH_VIDEO_MAX_ROI_REGRESSION_CELLS = 0.03125
POST_HIGH_VIDEO_MIN_MEAN_IMPROVEMENT = 0.25
POST_HIGH_VIDEO_MAX_X_PERTURBATION_CELLS = 0.0625
POST_HIGH_VIDEO_MAX_SUCCESSOR_PERTURBATION_CELLS = 0.0625
POST_HIGH_VIDEO_MIN_SUCCESSOR_RESPONSE = 3.0
POST_HIGH_VIDEO_ROIS = (("upper45", 0.45), ("full", 1.0))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _measure(video: torch.Tensor, prefix_t: int) -> dict[str, dict[str, Any]]:
    return {
        name: measure_translation_trajectory(
            video,
            prefix_t,
            forward_steps=5,
            backward_steps=3,
            roi_fraction=fraction,
            max_shift=4,
        )
        for name, fraction in POST_HIGH_VIDEO_ROIS
    }


def _first_pair(receipt: dict[str, Any], axis: str) -> float:
    values = receipt[f"pairwise_d{axis}"]
    if not values:
        raise ValueError("post-high residual repair requires a generated-side first pair")
    return float(values[0])


def _first_response(receipt: dict[str, Any]) -> float:
    values = receipt["pairwise_response"]
    if not values:
        raise ValueError("post-high residual repair requires a first-pair response")
    return float(values[0])


def _first_clipped(receipt: dict[str, Any]) -> bool:
    values = receipt["pairwise_clipped"]
    if not values:
        raise ValueError("post-high residual repair requires a first-pair clip flag")
    return bool(values[0])


def _prefix_motion(receipt: dict[str, Any], axis: str) -> float:
    return float(receipt[f"pre_pairwise_median_d{axis}"])


def plan_post_high_vertical_residual(
    pre_high: dict[str, dict[str, Any]],
    post_high: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return a fail-closed vertical-only correction plan from matched receipts."""

    receipt: dict[str, Any] = {
        "policy": POST_HIGH_VIDEO_RESIDUAL_POLICY,
        "eligible": False,
        "reason": "not_evaluated",
        "horizontal_application_enabled": False,
        "operator_dx_cells": 0.0,
        "release_weights": list(POST_HIGH_VIDEO_RELEASE_WEIGHTS),
    }
    if set(pre_high) != {"upper45", "full"} or set(post_high) != {"upper45", "full"}:
        receipt["reason"] = "missing_required_roi"
        return receipt

    boundary_errors: dict[str, float] = {}
    high_deltas: dict[str, float] = {}
    observations: dict[str, Any] = {}
    for roi, _fraction in POST_HIGH_VIDEO_ROIS:
        before = pre_high[roi]
        after = post_high[roi]
        if _first_clipped(before) or _first_clipped(after):
            receipt["reason"] = f"{roi}_first_pair_clipped"
            return receipt
        before_response = _first_response(before)
        after_response = _first_response(after)
        if min(before_response, after_response) < POST_HIGH_VIDEO_MIN_RESPONSE:
            receipt["reason"] = f"{roi}_low_response"
            return receipt

        pre_high_dy = _first_pair(before, "y")
        post_high_dy = _first_pair(after, "y")
        prefix_dy = _prefix_motion(after, "y")
        boundary_error = post_high_dy - prefix_dy
        pre_high_boundary_error = pre_high_dy - prefix_dy
        high_delta = post_high_dy - pre_high_dy
        boundary_errors[roi] = boundary_error
        high_deltas[roi] = high_delta
        observations[roi] = {
            "pre_high_first_dx": _first_pair(before, "x"),
            "pre_high_first_dy": pre_high_dy,
            "post_high_first_dx": _first_pair(after, "x"),
            "post_high_first_dy": post_high_dy,
            "prefix_median_dx": _prefix_motion(after, "x"),
            "prefix_median_dy": prefix_dy,
            "boundary_error_dy": boundary_error,
            "pre_high_boundary_error_dy": pre_high_boundary_error,
            "target_high_delta_dy": high_delta,
            "target_high_fraction_of_boundary_error": abs(high_delta) / max(abs(boundary_error), 1e-12),
            "pre_high_response": before_response,
            "post_high_response": after_response,
        }

    receipt["observations"] = observations
    boundary_signs = {_sign(v) for v in boundary_errors.values()}
    high_delta_signs = {_sign(v) for v in high_deltas.values()}
    if 0 in boundary_signs or len(boundary_signs) != 1:
        receipt["reason"] = "vertical_boundary_error_not_coherent"
        return receipt
    if 0 in high_delta_signs or len(high_delta_signs) != 1:
        receipt["reason"] = "target_high_vertical_delta_not_coherent"
        return receipt
    if next(iter(boundary_signs)) != next(iter(high_delta_signs)):
        receipt["reason"] = "boundary_error_not_explained_by_target_high_delta"
        return receipt
    if min(abs(v) for v in boundary_errors.values()) < POST_HIGH_VIDEO_MIN_BOUNDARY_ERROR_CELLS:
        receipt["reason"] = "vertical_boundary_error_below_floor"
        return receipt
    if min(abs(v) for v in high_deltas.values()) < POST_HIGH_VIDEO_MIN_HIGH_DELTA_CELLS:
        receipt["reason"] = "target_high_vertical_delta_below_floor"
        return receipt
    if any(
        abs(float(observations[roi]["pre_high_boundary_error_dy"])) > POST_HIGH_VIDEO_MAX_PRE_HIGH_BOUNDARY_ERROR_CELLS
        for roi, _fraction in POST_HIGH_VIDEO_ROIS
    ):
        receipt["reason"] = "pre_high_boundary_not_clean"
        return receipt
    if any(
        float(observations[roi]["target_high_fraction_of_boundary_error"]) < POST_HIGH_VIDEO_MIN_HIGH_DELTA_FRACTION
        for roi, _fraction in POST_HIGH_VIDEO_ROIS
    ):
        receipt["reason"] = "target_high_delta_not_dominant_enough"
        return receipt
    if max(boundary_errors.values()) - min(boundary_errors.values()) > POST_HIGH_VIDEO_MAX_ROI_DISAGREEMENT_CELLS:
        receipt["reason"] = "vertical_roi_disagreement_over_bound"
        return receipt

    # Correct what target-high added, not the entire observed boundary motion.
    # The pre-high exact-restored state is already the validated near-clean
    # reference; the prefix-motion error is used only as the acceptance target.
    signed_consensus = float(statistics.median(high_deltas.values()))
    magnitude = abs(signed_consensus)
    if magnitude > POST_HIGH_VIDEO_MAX_CORRECTION_CELLS:
        receipt["reason"] = "vertical_correction_over_bound"
        receipt["requested_correction_cells"] = signed_consensus
        return receipt

    extended_weights = (*POST_HIGH_VIDEO_RELEASE_WEIGHTS, 0.0)
    max_weight_step = max(abs(float(right) - float(left)) for left, right in itertools.pairwise(extended_weights))
    max_induced_step = magnitude * max_weight_step
    if max_induced_step > POST_HIGH_VIDEO_MAX_RELEASE_STEP_CELLS:
        receipt["reason"] = "temporal_release_step_over_bound"
        receipt["requested_correction_cells"] = signed_consensus
        receipt["max_induced_step_cells"] = max_induced_step
        return receipt

    receipt.update(
        eligible=True,
        reason="coherent_target_high_vertical_residual",
        signed_consensus_error_cells=signed_consensus,
        correction_magnitude_cells=magnitude,
        before_mean_abs_error_cells=float(statistics.mean(abs(v) for v in boundary_errors.values())),
        max_weight_step=max_weight_step,
        max_induced_step_cells=max_induced_step,
    )
    return receipt


def apply_weighted_vertical_translation(
    video: torch.Tensor,
    *,
    prefix_t: int,
    dy: float,
    weights: tuple[float, ...] = POST_HIGH_VIDEO_RELEASE_WEIGHTS,
) -> torch.Tensor:
    """Translate only the first generated tokens with a bounded temporal release."""

    if not torch.is_tensor(video) or video.ndim != 5 or not video.is_floating_point():
        raise TypeError("post-high residual repair expects floating BxCxTxHxW video")
    if not 0 < int(prefix_t) < int(video.shape[2]):
        raise ValueError("post-high residual repair requires a non-empty prefix and suffix")
    if not math.isfinite(float(dy)):
        raise ValueError("post-high residual repair dy must be finite")
    if not weights:
        return video
    output = video.clone()
    support = min(len(weights), int(video.shape[2]) - int(prefix_t))
    for offset in range(support):
        weight = float(weights[offset])
        if weight == 0.0:
            continue
        index = int(prefix_t) + offset
        translated = translate_video_cells(
            video[:, :, index : index + 1],
            dx=0.0,
            dy=float(dy) * weight,
            start_frame=0,
            batch_frames=1,
        )
        output[:, :, index : index + 1] = translated.video
    if not torch.equal(output[:, :, :prefix_t], video[:, :, :prefix_t]):
        raise RuntimeError("post-high residual repair modified the authoritative prefix")
    return output


def score_post_high_vertical_candidate(
    original: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    before_abs: dict[str, float] = {}
    after_abs: dict[str, float] = {}
    x_perturbation: dict[str, float] = {}
    successor_x_perturbation: dict[str, float] = {}
    successor_y_perturbation: dict[str, float] = {}
    successor_min_response: dict[str, float] = {}
    successor_clipped: dict[str, bool] = {}
    for roi, _fraction in POST_HIGH_VIDEO_ROIS:
        target_dy = _prefix_motion(original[roi], "y")
        before_abs[roi] = abs(_first_pair(original[roi], "y") - target_dy)
        after_abs[roi] = abs(_first_pair(candidate[roi], "y") - target_dy)
        x_perturbation[roi] = abs(_first_pair(candidate[roi], "x") - _first_pair(original[roi], "x"))

        original_dx = [float(value) for value in original[roi]["pairwise_dx"]]
        original_dy = [float(value) for value in original[roi]["pairwise_dy"]]
        candidate_dx = [float(value) for value in candidate[roi]["pairwise_dx"]]
        candidate_dy = [float(value) for value in candidate[roi]["pairwise_dy"]]
        count = min(len(original_dx), len(original_dy), len(candidate_dx), len(candidate_dy))
        if count < len(POST_HIGH_VIDEO_RELEASE_WEIGHTS) + 1:
            raise ValueError("post-high residual repair lost the temporal release frontier")
        successor_x_perturbation[roi] = max(
            abs(candidate_dx[index] - original_dx[index]) for index in range(1, len(POST_HIGH_VIDEO_RELEASE_WEIGHTS) + 1)
        )
        successor_y_perturbation[roi] = max(
            abs(candidate_dy[index] - original_dy[index]) for index in range(1, len(POST_HIGH_VIDEO_RELEASE_WEIGHTS) + 1)
        )
        responses = [float(value) for value in candidate[roi]["pairwise_response"]]
        clipped = [bool(value) for value in candidate[roi]["pairwise_clipped"]]
        successor_min_response[roi] = min(responses[1 : len(POST_HIGH_VIDEO_RELEASE_WEIGHTS) + 1])
        successor_clipped[roi] = any(clipped[1 : len(POST_HIGH_VIDEO_RELEASE_WEIGHTS) + 1])

    before_mean = float(statistics.mean(before_abs.values()))
    after_mean = float(statistics.mean(after_abs.values()))
    improvement = 1.0 - after_mean / max(before_mean, 1e-12)
    return after_mean, {
        "before_abs_error_cells": before_abs,
        "after_abs_error_cells": after_abs,
        "before_mean_abs_error_cells": before_mean,
        "after_mean_abs_error_cells": after_mean,
        "mean_improvement_ratio": improvement,
        "x_first_pair_perturbation_cells": x_perturbation,
        "successor_max_x_perturbation_cells": successor_x_perturbation,
        "successor_max_y_perturbation_cells": successor_y_perturbation,
        "successor_min_response": successor_min_response,
        "successor_clipped": successor_clipped,
    }


def validate_post_high_vertical_candidate(
    original: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Apply the production acceptance gates to a measured candidate."""

    _score, score_fields = score_post_high_vertical_candidate(original, candidate)
    no_material_roi_regression = all(
        float(score_fields["after_abs_error_cells"][roi])
        <= float(score_fields["before_abs_error_cells"][roi]) + POST_HIGH_VIDEO_MAX_ROI_REGRESSION_CELLS
        for roi, _fraction in POST_HIGH_VIDEO_ROIS
    )
    x_stable = all(
        float(score_fields["x_first_pair_perturbation_cells"][roi]) <= POST_HIGH_VIDEO_MAX_X_PERTURBATION_CELLS
        for roi, _fraction in POST_HIGH_VIDEO_ROIS
    )
    temporal_release_stable = all(
        float(score_fields["successor_max_x_perturbation_cells"][roi])
        <= POST_HIGH_VIDEO_MAX_SUCCESSOR_PERTURBATION_CELLS
        and float(score_fields["successor_max_y_perturbation_cells"][roi])
        <= POST_HIGH_VIDEO_MAX_SUCCESSOR_PERTURBATION_CELLS
        and float(score_fields["successor_min_response"][roi]) >= POST_HIGH_VIDEO_MIN_SUCCESSOR_RESPONSE
        and not bool(score_fields["successor_clipped"][roi])
        for roi, _fraction in POST_HIGH_VIDEO_ROIS
    )
    improved = float(score_fields["mean_improvement_ratio"]) >= POST_HIGH_VIDEO_MIN_MEAN_IMPROVEMENT
    accepted = bool(no_material_roi_regression and x_stable and temporal_release_stable and improved)
    return accepted, {
        **score_fields,
        "no_material_roi_regression": no_material_roi_regression,
        "horizontal_phase_stable": x_stable,
        "temporal_release_stable": temporal_release_stable,
        "minimum_mean_improvement": POST_HIGH_VIDEO_MIN_MEAN_IMPROVEMENT,
    }


def repair_post_high_vertical_residual(
    pre_high: dict[str, dict[str, Any]],
    final_video: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Measure, trial both signs, validate, and optionally return a corrected video."""

    post_high = _measure(final_video, prefix_t)
    plan = plan_post_high_vertical_residual(pre_high, post_high)
    receipt: dict[str, Any] = {
        **plan,
        "applied": False,
        "output_mutated": False,
        "authoritative_prefix_modified": False,
        "candidate_sign_trials": [],
        "pre_high": pre_high,
        "post_high_before": post_high,
    }
    if not bool(plan.get("eligible")):
        return final_video, receipt

    magnitude = float(plan["correction_magnitude_cells"])
    trials: list[tuple[float, torch.Tensor, dict[str, dict[str, Any]], float, dict[str, Any]]] = []
    for dy in (-magnitude, magnitude):
        candidate_video = apply_weighted_vertical_translation(
            final_video,
            prefix_t=prefix_t,
            dy=dy,
        )
        candidate_receipts = _measure(candidate_video, prefix_t)
        score, score_fields = score_post_high_vertical_candidate(post_high, candidate_receipts)
        trials.append((dy, candidate_video, candidate_receipts, score, score_fields))
        receipt["candidate_sign_trials"].append(
            {
                "dy": dy,
                **score_fields,
            }
        )

    dy, candidate_video, candidate_receipts, _score, _score_fields = min(trials, key=lambda item: item[3])
    accepted, validation = validate_post_high_vertical_candidate(post_high, candidate_receipts)
    receipt.update(
        selected_dy_cells=float(dy),
        selected_score=validation,
        post_high_after=candidate_receipts,
        accepted=accepted,
    )
    if not accepted:
        receipt["reason"] = "candidate_validation_failed"
        return final_video, receipt

    receipt.update(
        applied=True,
        output_mutated=True,
        reason="accepted_bounded_vertical_release",
        corrected_tokens=min(len(POST_HIGH_VIDEO_RELEASE_WEIGHTS), int(final_video.shape[2]) - int(prefix_t)),
    )
    return candidate_video, receipt


__all__ = [
    "POST_HIGH_VIDEO_RELEASE_WEIGHTS",
    "POST_HIGH_VIDEO_RESIDUAL_POLICY",
    "apply_weighted_vertical_translation",
    "plan_post_high_vertical_residual",
    "repair_post_high_vertical_residual",
    "score_post_high_vertical_candidate",
    "validate_post_high_vertical_candidate",
]
