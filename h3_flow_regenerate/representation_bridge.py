"""Exact-prefix representation reconciliation for Mixed-Grid Continuum.

The learned 3D upscaler sees a resized copy of the authoritative prefix and
returns its own target-grid reconstruction of those same prefix frames. That
overlap is useful calibration data, but its raw latent difference must not be
added to a generated frame: when the two representations are spatially
misregistered, the difference is an edge/chroma residual and transplanting it
creates visible colour ghosts.

This bridge therefore estimates only a low-dimensional, colour-insensitive
spatial registration from recent overlap frames and, when that estimate is
stable and safe, applies one constant affine transform to the entire generated
suffix. The prefix remains bit-exact. No temporal fade, local residual
injection, amplitude normalization, random field, model call, or VAE call is
introduced.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .source_trajectory_bridge import register_pair

_VERSION = 2
_OVERLAP_FRAMES = 4
_SCALE_FLOOR = 0.005
_TRANSLATION_FLOOR = 0.25
_MAX_LOG_SCALE = math.log(1.03)


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
        "suffix_representation_bridge_mode": "constant_overlap_geometry_v1",
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


def _candidate_transform(
    learned_prefix: torch.Tensor,
    exact_prefix: torch.Tensor,
) -> tuple[tuple[float, float, float, float] | None, dict]:
    overlap = min(_OVERLAP_FRAMES, int(exact_prefix.shape[2]))
    report = {
        "overlap_frames": overlap,
        "consensus_frames": 0,
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
    candidates: list[tuple[float, float, float, float]] = []
    for index in range(overlap):
        item = register_pair(
            exact_recent[:, :, index : index + 1],
            learned_recent[:, :, index : index + 1],
        )
        identity = item.get("identity_error")
        aligned = item.get("aligned_error")
        transform = item.get("transform")
        if (
            identity is None
            or aligned is None
            or transform is None
            or not math.isfinite(float(identity))
            or not math.isfinite(float(aligned))
            or float(aligned) >= float(identity)
        ):
            continue
        candidates.append(tuple(float(value) for value in transform))
    report["consensus_frames"] = len(candidates)
    if len(candidates) < 3:
        report["reason"] = "insufficient_consistent_registration_frames"
        return None, report

    signed = torch.tensor([_signed_transform(value) for value in candidates], dtype=torch.float64)
    median = tuple(float(value) for value in signed.median(dim=0).values.tolist())
    floors = (_SCALE_FLOOR, _SCALE_FLOOR, _TRANSLATION_FLOOR, _TRANSLATION_FLOOR)
    h, w = map(int, learned_prefix.shape[-2:])
    safety = (_MAX_LOG_SCALE, _MAX_LOG_SCALE, min(1.5, 0.025 * w), min(1.5, 0.025 * h))
    active = [abs(median[axis]) >= floors[axis] for axis in range(4)]
    if not any(active):
        report["reason"] = "overlap_geometry_already_matched"
        return None, report
    if any(active[axis] and abs(median[axis]) > safety[axis] + 1e-8 for axis in range(4)):
        report["reason"] = "overlap_geometry_exceeds_safety_bound"
        return None, report

    required = len(candidates) // 2 + 1
    signed_rows = signed.tolist()
    for axis in range(4):
        if not active[axis]:
            continue
        sign = 1.0 if median[axis] > 0 else -1.0
        support = sum(abs(float(row[axis])) >= floors[axis] and float(row[axis]) * sign > 0.0 for row in signed_rows)
        if support < required:
            report["reason"] = "overlap_geometry_direction_not_consistent"
            return None, report

    filtered_signed = tuple(median[axis] if active[axis] else 0.0 for axis in range(4))
    transform = _transform_from_signed(filtered_signed)
    aggregate = register_pair(exact_recent, learned_recent, comparison=transform)
    identity = aggregate.get("identity_error")
    aligned = aggregate.get("comparison_error")
    if identity is None or aligned is None or not math.isfinite(float(identity)) or not math.isfinite(float(aligned)):
        report["reason"] = "overlap_geometry_objective_unavailable"
        return None, report
    improvement = max(0.0, (float(identity) - float(aligned)) / max(float(identity), 1e-12))
    report.update(
        transform=transform,
        identity_error=float(identity),
        aligned_error=float(aligned),
        improvement=improvement,
    )
    if not float(aligned) < float(identity):
        report["reason"] = "overlap_geometry_did_not_improve_registration"
        return None, report
    report["reason"] = "constant_overlap_geometry_authorized"
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
        suffix_representation_bridge_reason="constant_overlap_geometry_applied",
        suffix_representation_bridge_corrected_tokens=int(suffix.shape[2]),
        suffix_representation_bridge_centered_error_after=after_error,
        suffix_representation_bridge_centered_error_ratio=ratio,
        suffix_representation_bridge_out_of_bounds_fraction=out_of_bounds,
    )
    return corrected, metrics
