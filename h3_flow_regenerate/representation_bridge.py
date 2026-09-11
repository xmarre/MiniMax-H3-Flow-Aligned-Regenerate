"""Exact-prefix representation reconciliation for Mixed-Grid Continuum.

The learned 3D upscaler sees a resized copy of the authoritative prefix and
returns its own target-grid reconstruction of those same prefix frames. That
overlap is useful calibration data, but its raw latent difference must not be
added to a generated frame: when the two representations are spatially
misregistered, the difference is an edge/chroma residual and transplanting it
creates visible colour ghosts.

The bridge therefore estimates only a low-dimensional, colour-insensitive
spatial registration from recent same-time overlap frames. Version 3 fits one
joint transform over the overlap and validates that transform independently on
the constituent frames. This avoids rejecting a real global offset merely
because noisy single-frame local optima disagree in sign, while still requiring
direct multi-frame evidence before a suffix warp is authorized.

When accepted, one constant affine transform is applied to the entire generated
suffix. The prefix remains bit-exact. No temporal fade, local residual
injection, amplitude normalization, random field, model call, or VAE call is
introduced.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .source_trajectory_bridge import register_pair

_VERSION = 3
_OVERLAP_FRAMES = 4
_SCALE_FLOOR = 0.005
_TRANSLATION_FLOOR = 0.25
_MAX_LOG_SCALE = math.log(1.03)
_MIN_AGGREGATE_IMPROVEMENT = 0.01
_MAX_SINGLE_FRAME_REGRESSION = 0.03


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
        "suffix_representation_bridge_mode": "joint_overlap_geometry_v2",
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
    }


def _signed_transform(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = (float(value) for value in transform)
    if sx <= 0.0 or sy <= 0.0 or not all(math.isfinite(value) for value in (sx, sy, tx, ty)):
        raise ValueError("representation bridge received an invalid registration transform")
    return math.log(sx), math.log(sy), tx, ty


def _transform_from_signed(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return math.exp(values[0]), math.exp(values[1]), values[2], values[3]


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


def _centered_error(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = left.float() - right.float()
    delta = delta - delta.mean(dim=(-2, -1), keepdim=True)
    return _rms(delta)


def _registration_values(item: dict, *, comparison: bool = False) -> tuple[float, float] | None:
    identity = item.get("identity_error")
    aligned = item.get("comparison_error" if comparison else "aligned_error")
    if identity is None or aligned is None:
        return None
    identity = float(identity)
    aligned = float(aligned)
    if not math.isfinite(identity) or not math.isfinite(aligned) or identity <= 0.0:
        return None
    return identity, aligned


def _candidate_transform(
    learned_prefix: torch.Tensor,
    exact_prefix: torch.Tensor,
) -> tuple[tuple[float, float, float, float] | None, dict]:
    overlap = min(_OVERLAP_FRAMES, int(exact_prefix.shape[2]))
    report = {
        "overlap_frames": overlap,
        "consensus_frames": 0,
        "validation_frames": 0,
        "improving_frames": 0,
        "median_frame_improvement": 0.0,
        "worst_frame_regression": 0.0,
        "transform": (1.0, 1.0, 0.0, 0.0),
        "identity_error": None,
        "aligned_error": None,
        "improvement": 0.0,
        "reason": "insufficient_overlap_geometry",
    }
    if learned_prefix.shape[0] != 1 or overlap < 3 or min(learned_prefix.shape[-2:]) < 16:
        return None, report

    learned_recent = learned_prefix[:, :, -overlap:]
    exact_recent = exact_prefix[:, :, -overlap:]

    # Fit one transform directly against the recent overlap as a joint objective.
    # The previous v2 path first fit each frame independently and then required
    # majority agreement on each transform axis. 00389 demonstrated that this
    # can reject the entire candidate even when the user-visible defect is a
    # coherent boundary shift: local frame optima are noisy because latent
    # representation differences and motion perturb individual fits.
    joint = register_pair(exact_recent, learned_recent)
    joint_values = _registration_values(joint)
    raw_transform = joint.get("transform")
    if joint_values is None or raw_transform is None or not joint_values[1] < joint_values[0]:
        report["reason"] = "joint_overlap_geometry_unavailable"
        return None, report

    signed = _signed_transform(tuple(float(value) for value in raw_transform))
    floors = (_SCALE_FLOOR, _SCALE_FLOOR, _TRANSLATION_FLOOR, _TRANSLATION_FLOOR)
    h, w = map(int, learned_prefix.shape[-2:])
    safety = (_MAX_LOG_SCALE, _MAX_LOG_SCALE, min(1.5, 0.025 * w), min(1.5, 0.025 * h))
    active = [abs(signed[axis]) >= floors[axis] for axis in range(4)]
    if not any(active):
        report["reason"] = "overlap_geometry_already_matched"
        return None, report
    if any(active[axis] and abs(signed[axis]) > safety[axis] + 1e-8 for axis in range(4)):
        report["reason"] = "overlap_geometry_exceeds_safety_bound"
        return None, report

    filtered_signed = tuple(signed[axis] if active[axis] else 0.0 for axis in range(4))
    transform = _transform_from_signed(filtered_signed)

    # Re-score the filtered transform on the joint overlap. This is the actual
    # transform that would be applied, so it must improve the aggregate target.
    aggregate = register_pair(exact_recent, learned_recent, comparison=transform)
    aggregate_values = _registration_values(aggregate, comparison=True)
    if aggregate_values is None:
        report["reason"] = "overlap_geometry_objective_unavailable"
        return None, report
    identity, aligned = aggregate_values
    improvement = (identity - aligned) / max(identity, 1e-12)
    report.update(
        transform=transform,
        identity_error=identity,
        aligned_error=aligned,
        improvement=max(0.0, improvement),
    )
    if improvement < _MIN_AGGREGATE_IMPROVEMENT:
        report["reason"] = "overlap_geometry_joint_gain_too_small"
        return None, report

    # Cross-check the one joint transform on the individual overlap frames.
    # We do not ask their independently fitted transforms to agree; instead we
    # ask the proposed single transform itself to improve a strong majority of
    # the observed same-time frame pairs and to avoid a large holdout regression.
    frame_improvements: list[float] = []
    for index in range(overlap):
        item = register_pair(
            exact_recent[:, :, index : index + 1],
            learned_recent[:, :, index : index + 1],
            comparison=transform,
        )
        values = _registration_values(item, comparison=True)
        if values is None:
            continue
        frame_identity, frame_aligned = values
        frame_improvements.append((frame_identity - frame_aligned) / max(frame_identity, 1e-12))

    validation_frames = len(frame_improvements)
    improving_frames = sum(value > 0.0 for value in frame_improvements)
    report["validation_frames"] = validation_frames
    report["improving_frames"] = improving_frames
    report["consensus_frames"] = improving_frames
    if frame_improvements:
        ordered = sorted(frame_improvements)
        mid = len(ordered) // 2
        median_improvement = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
        worst_regression = max(0.0, -min(frame_improvements))
        report["median_frame_improvement"] = float(median_improvement)
        report["worst_frame_regression"] = float(worst_regression)
    else:
        worst_regression = math.inf

    required_support = max(3, math.ceil(0.75 * validation_frames)) if validation_frames else 3
    if validation_frames < 3:
        report["reason"] = "insufficient_joint_overlap_validation_frames"
        return None, report
    if improving_frames < required_support:
        report["reason"] = "joint_overlap_geometry_insufficient_frame_support"
        return None, report
    if worst_regression > _MAX_SINGLE_FRAME_REGRESSION:
        report["reason"] = "joint_overlap_geometry_holdout_regression"
        return None, report

    report["reason"] = "joint_overlap_geometry_authorized"
    return transform, report


@torch.no_grad()
def apply_suffix_representation_bridge(
    learned_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    requested: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Align suffix geometry from same-frame overlap without residual injection."""

    prefix_t = _validate_pair(learned_clean_video, exact_prefix)
    if not isinstance(requested, bool):
        raise TypeError("representation bridge requested flag must be boolean")
    if not requested:
        return learned_clean_video, disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t)

    learned_prefix = learned_clean_video[:, :, :prefix_t]
    exact = exact_prefix.to(device=learned_clean_video.device, dtype=learned_clean_video.dtype)
    learned_last = learned_prefix[:, :, -1].float()
    exact_last = exact[:, :, -1].float()
    delta = exact_last - learned_last
    dc = delta.mean(dim=(-2, -1), keepdim=True)
    structural = delta - dc
    before_error = _rms(structural)

    metrics = disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t, requested=True)
    metrics.update(
        suffix_representation_bridge_delta_rms=_rms(delta),
        suffix_representation_bridge_dc_rms=_rms(dc),
        suffix_representation_bridge_structural_rms=_rms(structural),
        suffix_representation_bridge_centered_error_before=before_error,
    )

    transform, registration = _candidate_transform(learned_prefix, exact)
    metrics.update(
        suffix_representation_bridge_overlap_frames=registration["overlap_frames"],
        suffix_representation_bridge_consensus_frames=registration["consensus_frames"],
        suffix_representation_bridge_validation_frames=registration["validation_frames"],
        suffix_representation_bridge_improving_frames=registration["improving_frames"],
        suffix_representation_bridge_median_frame_improvement=registration["median_frame_improvement"],
        suffix_representation_bridge_worst_frame_regression=registration["worst_frame_regression"],
        suffix_representation_bridge_transform=registration["transform"],
        suffix_representation_bridge_identity_error=registration["identity_error"],
        suffix_representation_bridge_aligned_error=registration["aligned_error"],
        suffix_representation_bridge_improvement=registration["improvement"],
    )
    if transform is None:
        metrics.update(
            suffix_representation_bridge_reason=registration["reason"],
            suffix_representation_bridge_centered_error_after=before_error,
            suffix_representation_bridge_centered_error_ratio=1.0,
        )
        return learned_clean_video, metrics

    suffix = learned_clean_video[:, :, prefix_t:]
    warped_suffix, out_of_bounds = _warp_video_constant(suffix, transform)
    corrected = learned_clean_video.clone()
    corrected[:, :, prefix_t:] = warped_suffix
    if not bool(torch.isfinite(corrected).all().item()):
        raise RuntimeError("representation bridge produced NaN or Inf")
    if not torch.equal(corrected[:, :, :prefix_t], learned_clean_video[:, :, :prefix_t]):
        raise RuntimeError("representation bridge modified prefix values")

    warped_last, _ = _warp_video_constant(learned_prefix[:, :, -1:], transform)
    after_error = _centered_error(exact[:, :, -1:], warped_last)
    ratio = after_error / max(before_error, 1e-12)
    metrics.update(
        suffix_representation_bridge_enabled=True,
        suffix_representation_bridge_accepted=True,
        suffix_representation_bridge_reason="joint_overlap_geometry_applied",
        suffix_representation_bridge_corrected_tokens=int(suffix.shape[2]),
        suffix_representation_bridge_centered_error_after=after_error,
        suffix_representation_bridge_centered_error_ratio=ratio,
        suffix_representation_bridge_out_of_bounds_fraction=out_of_bounds,
    )
    return corrected, metrics
