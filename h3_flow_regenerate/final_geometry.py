"""Closed-loop final latent geometry for mixed-grid Continuum handoffs.

The initial mixed-grid bridge authorizes spatial axes from independent source and
target evidence and derives a temporal envelope from the genuine source-grid
continuation. High-resolution refinement can nevertheless reintroduce part of an
authorized boundary residual. This module measures the *returned* target-grid
latent, reuses only the already-authorized axes/envelope, and closes any residual
that still passes the original target-domain evidence/safety gates.

No model evaluation, VAE pass, conditioning change, or protected-prefix warp is
performed here. Rejected/no-op paths return the original tensor object.
"""

from __future__ import annotations

import copy
import math
import time

import torch

from .geometric_bridge import (
    _MIN_DOMAIN_EVIDENCE,
    _motion_residual_candidate,
    _signed_components,
    _warp_video_suffix,
    register_pair,
    residual_transform,
)


def _inactive_axis(axis: dict) -> dict:
    result = dict(axis)
    result.update(accepted=False, active_tokens=0)
    return result


def _filtered_temporal_profile(profile: dict, axis_enabled: list[bool]) -> dict:
    result = copy.deepcopy(profile)
    axes = list(result.get("axis") or [])
    if len(axes) != 4:
        return {"accepted": False, "axis": []}
    result["axis"] = [axis if axis_enabled[index] else _inactive_axis(axis) for index, axis in enumerate(axes)]
    result["axis_accepted"] = [bool(axis_enabled[index] and axes[index].get("accepted")) for index in range(4)]
    result["accepted"] = any(result["axis_accepted"])
    return result


def _current_signed_residual(boundary: dict, motion: dict) -> tuple[float, float, float, float] | None:
    expected = motion.get("expected_transform")
    observed = boundary.get("transform")
    if not motion.get("accepted") or expected is None or observed is None or boundary.get("aligned_error") is None:
        return None
    try:
        return _signed_components(residual_transform(tuple(observed), tuple(expected)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _axis_reduction_is_safe(
    before: tuple[float, ...],
    after: tuple[float, ...],
    floors: list[float],
    axis_enabled: list[bool],
) -> tuple[bool, list[float | None]]:
    ratios: list[float | None] = [None] * 4
    for axis in range(4):
        if not axis_enabled[axis]:
            continue
        before_abs = abs(float(before[axis]))
        after_abs = abs(float(after[axis]))
        ratios[axis] = after_abs / max(before_abs, 1e-12)
        if not math.isfinite(after_abs) or after_abs >= before_abs - 1e-9:
            return False, ratios
        # A sign flip is acceptable only when the residual has actually reached
        # estimator resolution; otherwise the projection has overshot materially.
        if float(before[axis]) * float(after[axis]) < 0 and after_abs > float(floors[axis]) + 1e-9:
            return False, ratios
    return True, ratios


@torch.no_grad()
def close_final_mixed_grid_residual(
    video: torch.Tensor,
    prefix_t: int,
    initial_geometry: dict,
) -> tuple[torch.Tensor, dict]:
    """Close an authorized residual on the final returned target-grid latent.

    Spatial authorization and temporal shape are never re-inferred here. They
    come from the initial cross-grid bridge. Only the target-grid residual
    magnitude is re-measured after high-resolution refinement. The correction is
    retained only when the measured residual decreases on every applied axis.
    """

    started = time.perf_counter()
    p = int(prefix_t)
    report = {
        "requested": bool(initial_geometry.get("requested")),
        "accepted": False,
        "reason": "initial_geometry_not_authorized",
        "policy": "final_target_residual_closure_from_initial_cross_grid_authorization",
        "prefix_t": p,
        "axis_applied": [False] * 4,
        "tokens_corrected": 0,
        "effective_active_temporal_length": 0,
        "applied_transforms": [],
        "applied_transform_sequence_length": 0,
        "applied_transforms_compact": False,
        "out_of_bounds_fraction": 0.0,
    }
    if video.ndim != 5 or not video.is_floating_point() or not 0 < p < int(video.shape[2]):
        report["reason"] = "invalid_final_video_geometry"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report
    if not bool(torch.isfinite(video).all()):
        report["reason"] = "nonfinite_final_video"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report
    if not initial_geometry.get("accepted"):
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    corroboration = initial_geometry.get("cross_grid_corroboration") or {}
    temporal = initial_geometry.get("temporal_profile") or {}
    initial_axes = list(corroboration.get("axis_authorized") or [False] * 4)
    temporal_axes = list(temporal.get("axis_accepted") or [False] * 4)
    if len(initial_axes) != 4 or len(temporal_axes) != 4:
        report["reason"] = "initial_geometry_contract_invalid"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report
    allowed = [bool(initial_axes[i] and temporal_axes[i]) for i in range(4)]
    if not any(allowed):
        report["reason"] = "no_initial_temporally_authorized_axis"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    motion = initial_geometry.get("target_natural_motion_model") or {}
    expected = motion.get("expected_transform")
    if not motion.get("accepted") or expected is None:
        report["reason"] = "target_motion_model_unavailable"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    exact = video[:, :, :p]
    boundary_before = register_pair(
        exact[:, :, -1:],
        video[:, :, p : p + 1],
        comparison=tuple(float(value) for value in expected),
    )
    current = _motion_residual_candidate(boundary_before, motion, tuple(video.shape[-2:]))
    report.update(boundary_before=boundary_before, current_target_motion_residual=current)
    if not current.get("accepted"):
        report["reason"] = "final_target_residual_not_provisional"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    source_signed = list(corroboration.get("source_signed_residual") or [0.0] * 4)
    current_signed = list(current.get("raw_signed_residual") or _signed_components(current["raw_transform"]))
    current_axes = list(current.get("axis_applied") or [False] * 4)
    current_scores = list(current.get("axis_evidence_score") or [0.0] * 4)
    floors = [float(value) for value in current.get("axis_estimator_floor", [0.005, 0.005, 0.25, 0.25])]
    if not all(len(values) == 4 for values in (source_signed, current_signed, current_axes, current_scores, floors)):
        report["reason"] = "final_target_residual_contract_invalid"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    axis_enabled: list[bool] = []
    axis_reason: list[str] = []
    for axis in range(4):
        if not allowed[axis]:
            enabled = False
            reason = "axis_not_initially_authorized"
        elif not current_axes[axis]:
            enabled = False
            reason = "final_target_axis_not_provisional"
        elif float(current_scores[axis]) < _MIN_DOMAIN_EVIDENCE:
            enabled = False
            reason = "final_target_evidence_too_weak"
        elif float(source_signed[axis]) * float(current_signed[axis]) <= 0:
            enabled = False
            reason = "final_target_residual_sign_changed"
        else:
            enabled = True
            reason = "final_target_residual_authorized"
        axis_enabled.append(enabled)
        axis_reason.append(reason)

    report.update(
        initial_axis_authorized=allowed,
        final_target_evidence_score=current_scores,
        final_target_signed_residual=current_signed,
        axis_reason=axis_reason,
    )
    if not any(axis_enabled):
        report["reason"] = "no_final_residual_axis_survived"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    raw = tuple(float(value) for value in current["raw_transform"])
    base = (
        raw[0] if axis_enabled[0] else 1.0,
        raw[1] if axis_enabled[1] else 1.0,
        raw[2] if axis_enabled[2] else 0.0,
        raw[3] if axis_enabled[3] else 0.0,
    )
    filtered = _filtered_temporal_profile(temporal, axis_enabled)
    if not filtered.get("accepted"):
        report["reason"] = "final_temporal_profile_unavailable"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report

    corrected, warp_report = _warp_video_suffix(video, p, base, filtered)
    report.update(base_target_transform=base, axis_applied=axis_enabled, **warp_report)
    if corrected is video:
        report["reason"] = "no_nonidentity_final_transform"
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return video, report
    if not torch.equal(corrected[:, :, :p], video[:, :, :p]):
        raise RuntimeError("final mixed-grid geometry modified the authoritative prefix")

    boundary_after = register_pair(
        exact[:, :, -1:],
        corrected[:, :, p : p + 1],
        comparison=tuple(float(value) for value in expected),
    )
    after_signed = _current_signed_residual(boundary_after, motion)
    if after_signed is None:
        report.update(
            reason="final_projection_verification_unavailable",
            boundary_after=boundary_after,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return video, report

    safe, reduction = _axis_reduction_is_safe(tuple(current_signed), after_signed, floors, axis_enabled)
    report.update(
        boundary_after=boundary_after,
        final_target_signed_residual_after=list(after_signed),
        residual_magnitude_ratio_after=reduction,
        prefix_unchanged=True,
    )
    if not safe:
        report.update(
            accepted=False,
            reason="final_projection_did_not_reduce_authorized_residual",
            tokens_corrected=0,
            effective_active_temporal_length=0,
            applied_transforms=[],
            applied_transform_sequence_length=0,
            applied_transforms_compact=False,
            out_of_bounds_fraction=0.0,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return video, report

    report.update(
        accepted=True,
        reason="final_authorized_residual_closed",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    return corrected, report
