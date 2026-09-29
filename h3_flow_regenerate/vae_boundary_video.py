"""Decoder-window-aware post-high video boundary correction.

MiniMax H3's video decoder consumes overlapping seven-latent temporal windows
(five-token stride plus two-token overlap).  For phase-aligned Continuum
prefixes, the first retained decoded frame is the first unblended frame of the
window that contains the exact-prefix boundary.  A correction that fades inside
that same latent window can therefore create or preserve a decoded camera jump.

This module keeps one measured rigid Y correction constant across every
generated latent token consumed by the boundary decoder window, then releases
it only after that window.  It never modifies the authoritative exact prefix,
audio, sampler state, or model execution.
"""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Any

import torch

from .frame_gauge import translate_video_cells
from .seam_diagnostics import measure_translation_trajectory

VAE_WINDOW_VIDEO_POLICY = "h3_vae_boundary_window_vertical_plateau_release_v1"

# Pinned MiniMax-H3 video-VAE temporal contract.
H3_VAE_RATIO_T = 4
H3_VAE_CLIP_FRAMES = 17
H3_VAE_FRAME_PRE_PADDING = 3
H3_VAE_CHUNK_TOKENS = 5
H3_VAE_TOKEN_OVERLAP = 2
H3_VAE_WINDOW_TOKENS = H3_VAE_CHUNK_TOKENS + H3_VAE_TOKEN_OVERLAP
H3_VAE_FRAME_OVERLAP = 5

# The correction is constant through the boundary decoder window.  The four
# values below are used only after that window, so the first zero-correction
# token is outside the window that renders the physical continuation boundary.
VAE_WINDOW_RELEASE_TAIL = (
    0.8535533905932737,
    0.5,
    0.14644660940672627,
)

VAE_WINDOW_MIN_RESPONSE = 6.0
VAE_WINDOW_MIN_HIGH_DELTA_CELLS = 0.025
VAE_WINDOW_MAX_PRE_HIGH_BOUNDARY_ERROR_CELLS = 0.125
VAE_WINDOW_MAX_ROI_DELTA_DISAGREEMENT_CELLS = 0.20
VAE_WINDOW_MAX_CORRECTION_CELLS = 0.18
VAE_WINDOW_MAX_RELEASE_STEP_CELLS = 0.0625
VAE_WINDOW_MAX_ROI_REGRESSION_CELLS = 0.03125
VAE_WINDOW_MIN_MEAN_IMPROVEMENT = 0.25
VAE_WINDOW_MAX_X_PERTURBATION_CELLS = 0.0625
VAE_WINDOW_MAX_PLATEAU_TRANSITION_PERTURBATION_CELLS = 0.0625
VAE_WINDOW_MAX_RELEASE_TRANSITION_PERTURBATION_CELLS = 0.08
VAE_WINDOW_MIN_SUCCESSOR_RESPONSE = 3.0
VAE_WINDOW_ROIS = (("upper45", 0.45), ("full", 1.0))


def h3_vae_boundary_window(prefix_t: int, temporal: int) -> dict[str, Any]:
    """Return the native decoder window that renders a phase-aligned boundary."""

    prefix_t = int(prefix_t)
    temporal = int(temporal)
    if prefix_t < H3_VAE_TOKEN_OVERLAP or prefix_t >= temporal:
        raise ValueError("H3 VAE boundary window requires a non-empty phase-aligned prefix and suffix")
    if (prefix_t - H3_VAE_TOKEN_OVERLAP) % H3_VAE_CHUNK_TOKENS:
        raise ValueError("H3 VAE boundary prefix is not on the native 5k+2 latent phase")

    chunk_index = (prefix_t - H3_VAE_TOKEN_OVERLAP) // H3_VAE_CHUNK_TOKENS
    window_start = chunk_index * H3_VAE_CHUNK_TOKENS
    window_stop = window_start + H3_VAE_WINDOW_TOKENS
    if window_stop > temporal:
        raise ValueError("H3 VAE boundary window lacks the complete native right-context window")
    if not window_start <= prefix_t < window_stop:
        raise RuntimeError("H3 VAE boundary window arithmetic drifted")

    generated_window_tokens = window_stop - prefix_t
    if generated_window_tokens != H3_VAE_CHUNK_TOKENS:
        raise RuntimeError("phase-aligned H3 boundary must expose exactly five generated window tokens")

    decoded_trim_frames = chunk_index * H3_VAE_CLIP_FRAMES + H3_VAE_FRAME_OVERLAP
    chunk_output_start = chunk_index * H3_VAE_CLIP_FRAMES
    first_retained_local_frame = decoded_trim_frames - chunk_output_start
    if first_retained_local_frame != H3_VAE_FRAME_OVERLAP:
        raise RuntimeError("H3 VAE first-retained-frame phase drifted")

    weights = (1.0,) * generated_window_tokens + VAE_WINDOW_RELEASE_TAIL
    support_stop = prefix_t + len(weights)
    if support_stop >= temporal:
        raise ValueError("H3 VAE boundary correction lacks an untouched release-frontier token")

    return {
        "policy": VAE_WINDOW_VIDEO_POLICY,
        "prefix_t": prefix_t,
        "temporal": temporal,
        "chunk_index": chunk_index,
        "window_start_t": window_start,
        "window_stop_t": window_stop,
        "window_tokens": H3_VAE_WINDOW_TOKENS,
        "chunk_stride_tokens": H3_VAE_CHUNK_TOKENS,
        "token_overlap": H3_VAE_TOKEN_OVERLAP,
        "generated_window_start_t": prefix_t,
        "generated_window_stop_t": window_stop,
        "generated_window_tokens": generated_window_tokens,
        "plateau_tokens": generated_window_tokens,
        "release_tail_tokens": len(VAE_WINDOW_RELEASE_TAIL),
        "support_tokens": len(weights),
        "support_stop_t": support_stop,
        "release_frontier_t": support_stop,
        "weights": list(weights),
        "decoded_trim_frames": decoded_trim_frames,
        "decoder_chunk_output_start_frame": chunk_output_start,
        "first_retained_local_frame": first_retained_local_frame,
        "decoder_internal_overlap_frames": H3_VAE_FRAME_OVERLAP,
        "first_retained_after_internal_overlap": True,
    }


def measure_vae_window_video_trajectory(
    video: torch.Tensor,
    prefix_t: int,
) -> dict[str, dict[str, Any]]:
    """Measure through the boundary window and the complete release frontier."""

    window = h3_vae_boundary_window(prefix_t, int(video.shape[2]))
    forward_steps = int(window["support_tokens"]) + 1
    return {
        name: measure_translation_trajectory(
            video,
            prefix_t,
            forward_steps=forward_steps,
            backward_steps=3,
            roi_fraction=fraction,
            max_shift=4,
        )
        for name, fraction in VAE_WINDOW_ROIS
    }


def _first_pair(receipt: dict[str, Any], axis: str) -> float:
    values = receipt[f"pairwise_d{axis}"]
    if not values:
        raise ValueError("VAE-window repair requires a generated-side first pair")
    return float(values[0])


def _first_response(receipt: dict[str, Any]) -> float:
    values = receipt["pairwise_response"]
    if not values:
        raise ValueError("VAE-window repair requires a first-pair response")
    return float(values[0])


def _first_clipped(receipt: dict[str, Any]) -> bool:
    values = receipt["pairwise_clipped"]
    if not values:
        raise ValueError("VAE-window repair requires a first-pair clip flag")
    return bool(values[0])


def _prefix_motion(receipt: dict[str, Any], axis: str) -> float:
    return float(receipt[f"pre_pairwise_median_d{axis}"])


def plan_vae_window_vertical_residual(
    pre_high: dict[str, dict[str, Any]],
    post_high: dict[str, dict[str, Any]],
    *,
    prefix_t: int,
    temporal: int,
) -> dict[str, Any]:
    """Plan a vertical correction for only the target-high-added boundary motion."""

    window = h3_vae_boundary_window(prefix_t, temporal)
    receipt: dict[str, Any] = {
        **window,
        "eligible": False,
        "reason": "not_evaluated",
        "horizontal_application_enabled": False,
        "operator_dx_cells": 0.0,
    }
    if set(pre_high) != {"upper45", "full"} or set(post_high) != {"upper45", "full"}:
        receipt["reason"] = "missing_required_roi"
        return receipt

    required_pairs = int(window["support_tokens"]) + 1
    if any(
        min(
            len(post_high[roi].get("pairwise_dx", [])),
            len(post_high[roi].get("pairwise_dy", [])),
            len(post_high[roi].get("pairwise_response", [])),
            len(post_high[roi].get("pairwise_clipped", [])),
        )
        < required_pairs
        for roi, _fraction in VAE_WINDOW_ROIS
    ):
        receipt["reason"] = "insufficient_release_frontier"
        return receipt

    high_deltas: dict[str, float] = {}
    observations: dict[str, Any] = {}
    for roi, _fraction in VAE_WINDOW_ROIS:
        before = pre_high[roi]
        after = post_high[roi]
        if _first_clipped(before) or _first_clipped(after):
            receipt["reason"] = f"{roi}_first_pair_clipped"
            return receipt
        before_response = _first_response(before)
        after_response = _first_response(after)
        if min(before_response, after_response) < VAE_WINDOW_MIN_RESPONSE:
            receipt["reason"] = f"{roi}_low_response"
            return receipt

        pre_high_dy = _first_pair(before, "y")
        post_high_dy = _first_pair(after, "y")
        prefix_dy = _prefix_motion(after, "y")
        high_delta = post_high_dy - pre_high_dy
        pre_high_boundary_error = pre_high_dy - prefix_dy
        high_deltas[roi] = high_delta
        observations[roi] = {
            "pre_high_first_dx": _first_pair(before, "x"),
            "pre_high_first_dy": pre_high_dy,
            "post_high_first_dx": _first_pair(after, "x"),
            "post_high_first_dy": post_high_dy,
            "prefix_median_dx": _prefix_motion(after, "x"),
            "prefix_median_dy": prefix_dy,
            "pre_high_boundary_error_dy": pre_high_boundary_error,
            "target_high_delta_dy": high_delta,
        }

    receipt["observations"] = observations
    signs = {1 if value > 0 else -1 if value < 0 else 0 for value in high_deltas.values()}
    if 0 in signs or len(signs) != 1:
        receipt["reason"] = "target_high_vertical_delta_not_coherent"
        return receipt
    if min(abs(value) for value in high_deltas.values()) < VAE_WINDOW_MIN_HIGH_DELTA_CELLS:
        receipt["reason"] = "target_high_vertical_delta_below_floor"
        return receipt
    if max(high_deltas.values()) - min(high_deltas.values()) > VAE_WINDOW_MAX_ROI_DELTA_DISAGREEMENT_CELLS:
        receipt["reason"] = "target_high_vertical_delta_roi_disagreement_over_bound"
        return receipt
    if any(
        abs(float(observations[roi]["pre_high_boundary_error_dy"])) > VAE_WINDOW_MAX_PRE_HIGH_BOUNDARY_ERROR_CELLS
        for roi, _fraction in VAE_WINDOW_ROIS
    ):
        receipt["reason"] = "pre_high_boundary_not_clean"
        return receipt

    signed_consensus = float(statistics.median(high_deltas.values()))
    magnitude = abs(signed_consensus)
    if magnitude > VAE_WINDOW_MAX_CORRECTION_CELLS:
        receipt["reason"] = "vertical_correction_over_bound"
        receipt["requested_correction_cells"] = signed_consensus
        return receipt

    weights = [float(value) for value in window["weights"]]
    extended = (*weights, 0.0)
    max_weight_step = max(abs(right - left) for left, right in itertools.pairwise(extended))
    max_induced_step = magnitude * max_weight_step
    if max_induced_step > VAE_WINDOW_MAX_RELEASE_STEP_CELLS:
        receipt["reason"] = "post_window_release_step_over_bound"
        receipt["requested_correction_cells"] = signed_consensus
        receipt["max_induced_step_cells"] = max_induced_step
        return receipt

    receipt.update(
        eligible=True,
        reason="coherent_target_high_vertical_residual_inside_vae_boundary_window",
        signed_consensus_error_cells=signed_consensus,
        correction_magnitude_cells=magnitude,
        max_weight_step=max_weight_step,
        max_induced_step_cells=max_induced_step,
        target_high_delta_cells=high_deltas,
    )
    return receipt


def apply_vae_window_vertical_translation(
    video: torch.Tensor,
    *,
    prefix_t: int,
    dy: float,
) -> torch.Tensor:
    """Translate the complete generated boundary window, then release afterward."""

    if not torch.is_tensor(video) or video.ndim != 5 or not video.is_floating_point():
        raise TypeError("VAE-window repair expects floating BxCxTxHxW video")
    if not math.isfinite(float(dy)):
        raise ValueError("VAE-window repair dy must be finite")
    window = h3_vae_boundary_window(prefix_t, int(video.shape[2]))
    weights = tuple(float(value) for value in window["weights"])

    output = video.clone()
    for offset, weight in enumerate(weights):
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
        raise RuntimeError("VAE-window repair modified the authoritative prefix")
    support_stop = int(window["support_stop_t"])
    if not torch.equal(output[:, :, support_stop:], video[:, :, support_stop:]):
        raise RuntimeError("VAE-window repair modified suffix outside bounded support")
    return output


def score_vae_window_vertical_candidate(
    pre_high: dict[str, dict[str, Any]],
    original: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    prefix_t: int,
    temporal: int,
) -> dict[str, Any]:
    """Score boundary restoration and all plateau/release transitions."""

    window = h3_vae_boundary_window(prefix_t, temporal)
    support = int(window["support_tokens"])
    plateau = int(window["plateau_tokens"])

    before_abs: dict[str, float] = {}
    after_abs: dict[str, float] = {}
    x_perturbation: dict[str, float] = {}
    plateau_x: dict[str, float] = {}
    plateau_y: dict[str, float] = {}
    release_x: dict[str, float] = {}
    release_y: dict[str, float] = {}
    release_min_response: dict[str, float] = {}
    release_clipped: dict[str, bool] = {}

    for roi, _fraction in VAE_WINDOW_ROIS:
        target_dy = _first_pair(pre_high[roi], "y")
        before_abs[roi] = abs(_first_pair(original[roi], "y") - target_dy)
        after_abs[roi] = abs(_first_pair(candidate[roi], "y") - target_dy)
        x_perturbation[roi] = abs(_first_pair(candidate[roi], "x") - _first_pair(original[roi], "x"))

        original_dx = [float(value) for value in original[roi]["pairwise_dx"]]
        original_dy = [float(value) for value in original[roi]["pairwise_dy"]]
        candidate_dx = [float(value) for value in candidate[roi]["pairwise_dx"]]
        candidate_dy = [float(value) for value in candidate[roi]["pairwise_dy"]]
        responses = [float(value) for value in candidate[roi]["pairwise_response"]]
        clipped = [bool(value) for value in candidate[roi]["pairwise_clipped"]]
        if (
            min(
                len(original_dx),
                len(original_dy),
                len(candidate_dx),
                len(candidate_dy),
                len(responses),
                len(clipped),
            )
            < support + 1
        ):
            raise ValueError("VAE-window candidate lost the complete release frontier")

        # Pair index 0 is prefix -> suffix0.  Pair indices 1..plateau-1
        # are wholly inside the constant-translation plateau.
        plateau_range = range(1, plateau)
        plateau_x[roi] = max((abs(candidate_dx[i] - original_dx[i]) for i in plateau_range), default=0.0)
        plateau_y[roi] = max((abs(candidate_dy[i] - original_dy[i]) for i in plateau_range), default=0.0)

        # Pair indices plateau..support cover plateau -> release and the
        # final release -> untouched transition.
        release_range = range(plateau, support + 1)
        release_x[roi] = max(abs(candidate_dx[i] - original_dx[i]) for i in release_range)
        release_y[roi] = max(abs(candidate_dy[i] - original_dy[i]) for i in release_range)
        release_min_response[roi] = min(responses[i] for i in release_range)
        release_clipped[roi] = any(clipped[i] for i in release_range)

    before_mean = float(statistics.mean(before_abs.values()))
    after_mean = float(statistics.mean(after_abs.values()))
    improvement = 1.0 - after_mean / max(before_mean, 1e-12)
    return {
        "before_abs_error_to_pre_high_cells": before_abs,
        "after_abs_error_to_pre_high_cells": after_abs,
        "before_mean_abs_error_cells": before_mean,
        "after_mean_abs_error_cells": after_mean,
        "mean_improvement_ratio": improvement,
        "x_first_pair_perturbation_cells": x_perturbation,
        "plateau_max_x_transition_perturbation_cells": plateau_x,
        "plateau_max_y_transition_perturbation_cells": plateau_y,
        "release_max_x_transition_perturbation_cells": release_x,
        "release_max_y_transition_perturbation_cells": release_y,
        "release_min_response": release_min_response,
        "release_clipped": release_clipped,
    }


def validate_vae_window_vertical_candidate(
    pre_high: dict[str, dict[str, Any]],
    original: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    prefix_t: int,
    temporal: int,
) -> tuple[bool, dict[str, Any]]:
    """Apply fail-closed gates to a measured decoder-window-aware candidate."""

    score = score_vae_window_vertical_candidate(
        pre_high,
        original,
        candidate,
        prefix_t=prefix_t,
        temporal=temporal,
    )
    no_roi_regression = all(
        float(score["after_abs_error_to_pre_high_cells"][roi])
        <= float(score["before_abs_error_to_pre_high_cells"][roi]) + VAE_WINDOW_MAX_ROI_REGRESSION_CELLS
        for roi, _fraction in VAE_WINDOW_ROIS
    )
    x_stable = all(
        float(score["x_first_pair_perturbation_cells"][roi]) <= VAE_WINDOW_MAX_X_PERTURBATION_CELLS
        for roi, _fraction in VAE_WINDOW_ROIS
    )
    plateau_stable = all(
        float(score["plateau_max_x_transition_perturbation_cells"][roi])
        <= VAE_WINDOW_MAX_PLATEAU_TRANSITION_PERTURBATION_CELLS
        and float(score["plateau_max_y_transition_perturbation_cells"][roi])
        <= VAE_WINDOW_MAX_PLATEAU_TRANSITION_PERTURBATION_CELLS
        for roi, _fraction in VAE_WINDOW_ROIS
    )
    release_stable = all(
        float(score["release_max_x_transition_perturbation_cells"][roi])
        <= VAE_WINDOW_MAX_RELEASE_TRANSITION_PERTURBATION_CELLS
        and float(score["release_max_y_transition_perturbation_cells"][roi])
        <= VAE_WINDOW_MAX_RELEASE_TRANSITION_PERTURBATION_CELLS
        and float(score["release_min_response"][roi]) >= VAE_WINDOW_MIN_SUCCESSOR_RESPONSE
        and not bool(score["release_clipped"][roi])
        for roi, _fraction in VAE_WINDOW_ROIS
    )
    improved = float(score["mean_improvement_ratio"]) >= VAE_WINDOW_MIN_MEAN_IMPROVEMENT
    accepted = bool(no_roi_regression and x_stable and plateau_stable and release_stable and improved)
    return accepted, {
        **score,
        "no_material_roi_regression": no_roi_regression,
        "horizontal_phase_stable": x_stable,
        "vae_window_plateau_stable": plateau_stable,
        "post_window_release_stable": release_stable,
        "minimum_mean_improvement": VAE_WINDOW_MIN_MEAN_IMPROVEMENT,
    }


def repair_vae_window_vertical_residual(
    pre_high: dict[str, dict[str, Any]],
    final_video: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Trial both vertical signs and return only a structurally accepted candidate."""

    temporal = int(final_video.shape[2])
    post_high = measure_vae_window_video_trajectory(final_video, prefix_t)
    plan = plan_vae_window_vertical_residual(
        pre_high,
        post_high,
        prefix_t=prefix_t,
        temporal=temporal,
    )
    receipt: dict[str, Any] = {
        **plan,
        "applied": False,
        "accepted": False,
        "output_mutated": False,
        "authoritative_prefix_modified": False,
        "candidate_sign_trials": [],
        "pre_high": pre_high,
        "post_high_before": post_high,
    }
    if not bool(plan.get("eligible")):
        return final_video, receipt

    magnitude = float(plan["correction_magnitude_cells"])
    trials: list[tuple[float, torch.Tensor, dict[str, dict[str, Any]], dict[str, Any]]] = []
    for dy in (-magnitude, magnitude):
        candidate_video = apply_vae_window_vertical_translation(
            final_video,
            prefix_t=prefix_t,
            dy=dy,
        )
        candidate_receipts = measure_vae_window_video_trajectory(candidate_video, prefix_t)
        score = score_vae_window_vertical_candidate(
            pre_high,
            post_high,
            candidate_receipts,
            prefix_t=prefix_t,
            temporal=temporal,
        )
        trials.append((dy, candidate_video, candidate_receipts, score))
        receipt["candidate_sign_trials"].append({"dy": dy, **score})

    dy, candidate_video, candidate_receipts, score = min(
        trials,
        key=lambda item: float(item[3]["after_mean_abs_error_cells"]),
    )
    accepted, validation = validate_vae_window_vertical_candidate(
        pre_high,
        post_high,
        candidate_receipts,
        prefix_t=prefix_t,
        temporal=temporal,
    )
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
        reason="accepted_vae_window_coherent_vertical_plateau_release",
        corrected_tokens=int(plan["support_tokens"]),
    )
    return candidate_video, receipt


__all__ = [
    "H3_VAE_CHUNK_TOKENS",
    "H3_VAE_FRAME_OVERLAP",
    "H3_VAE_TOKEN_OVERLAP",
    "H3_VAE_WINDOW_TOKENS",
    "VAE_WINDOW_RELEASE_TAIL",
    "VAE_WINDOW_VIDEO_POLICY",
    "apply_vae_window_vertical_translation",
    "h3_vae_boundary_window",
    "measure_vae_window_video_trajectory",
    "plan_vae_window_vertical_residual",
    "repair_vae_window_vertical_residual",
    "score_vae_window_vertical_candidate",
    "validate_vae_window_vertical_candidate",
]
