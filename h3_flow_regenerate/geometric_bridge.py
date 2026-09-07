"""Motion-aware geometric seam diagnostics for progressive H3 Continuum.

Transforms use output->input sampling coordinates about image centre, in latent
pixels. The bridge estimates the natural transform between recent exact-prefix
tokens, compares it with the protected-prefix -> learned-suffix boundary, and
corrects only the statistically supported residual. The authoritative prefix is
never warped. No gradients are used.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

IDENTITY = (1.0, 1.0, 0.0, 0.0)
MOTION_BASELINE_TRANSITIONS = 6
_SCALE_FLOOR = 0.005
_TRANSLATION_FLOOR = 0.25


def warp_frame(frame: torch.Tensor, transform: tuple[float, ...]) -> torch.Tensor:
    """One bilinear resampling, border extension, pixel-centre coordinates."""
    if frame.ndim != 4 or not frame.is_floating_point():
        raise ValueError("geometric warp requires floating BxCxHxW")
    sx, sy, tx, ty = transform
    if not all(math.isfinite(v) for v in transform) or min(sx, sy) <= 0:
        raise ValueError("geometric warp requires finite positive scales")
    if tuple(transform) == IDENTITY:
        return frame
    h, w = frame.shape[-2:]
    with torch.autocast(device_type=frame.device.type, enabled=False):
        theta = frame.new_tensor([[sx, 0, 2 * tx / w], [0, sy, 2 * ty / h]], dtype=torch.float32)
        grid = F.affine_grid(theta[None].expand(frame.shape[0], -1, -1), frame.shape, align_corners=False)
        return F.grid_sample(frame.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(frame)


def invert_transform(transform: tuple[float, ...]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = transform
    if min(sx, sy) <= 0 or not all(math.isfinite(v) for v in transform):
        raise ValueError("cannot invert invalid geometric transform")
    return (1.0 / sx, 1.0 / sy, -tx / sx, -ty / sy)


def compose_transform(first: tuple[float, ...], second: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Return first∘second for output->input centred affine transforms."""
    sx1, sy1, tx1, ty1 = first
    sx2, sy2, tx2, ty2 = second
    return (sx1 * sx2, sy1 * sy2, sx1 * tx2 + tx1, sy1 * ty2 + ty1)


def residual_transform(observed: tuple[float, ...], expected: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Sampling transform C such that C∘expected == observed."""
    return compose_transform(observed, invert_transform(expected))


def _features(video):
    b, c, t, h, w = video.shape
    x = video.detach().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    x = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    ratio = min(1.0, 48 / max(h, w))
    x = F.interpolate(x, size=(round(h * ratio), round(w * ratio)), mode="area")
    x = x.cpu()
    x = x - x.mean((-2, -1), keepdim=True)
    rms = x.square().mean((-2, -1), keepdim=True).sqrt()
    return x / rms.clamp_min(1e-4), rms


def _errors(reference, moving, transform, original_hw):
    h, w = original_hw
    fh, fw = moving.shape[-2:]
    sx, sy, tx, ty = transform
    xx = np.clip(sx * (np.arange(fw, dtype=np.float32) - (fw - 1) / 2) + (fw - 1) / 2 + tx * fw / w, 0, fw - 1)
    yy = np.clip(sy * (np.arange(fh, dtype=np.float32) - (fh - 1) / 2) + (fh - 1) / 2 + ty * fh / h, 0, fh - 1)
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
    return dict(
        scale_x=sx,
        scale_y=sy,
        translate_x=tx,
        translate_y=ty,
        translate_x_normalized=2 * tx / w,
        translate_y_normalized=2 * ty / h,
    )


def _estimate_registration(reference: torch.Tensor, moving: torch.Tensor, *, comparison=None) -> dict:
    report = dict(
        accepted=False,
        reason="invalid_geometry",
        confidence=0.0,
        transform=IDENTITY,
        identity_error=None,
        aligned_error=None,
        improvement=0.0,
    )
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
    a, z = a.numpy(), z.numpy()
    identity = float(_errors(a, z, IDENTITY, hw).mean())
    theta, error = _fit(a, z, hw)
    improvement = max(0.0, (identity - error) / max(identity, 1e-8))
    separations = []
    for axis, step in enumerate((0.01, 0.01, 0.5, 0.5)):
        losses = []
        for sign in (-1, 1):
            probe = list(theta)
            probe[axis] += sign * step
            losses.append(float(_errors(a, z, probe, hw).mean()) - error)
        separations.append(min(losses))
    report.update(
        _transform_fields(theta, hw),
        transform=theta,
        identity_error=identity,
        aligned_error=error,
        improvement=improvement,
        paired_tokens=a.shape[0],
        axis_objective_separation=separations,
    )
    if comparison is not None:
        comparison = tuple(float(v) for v in comparison)
        comparison_error = float(_errors(a, z, comparison, hw).mean())
        axis_gains = []
        for axis in range(4):
            probe = list(theta)
            probe[axis] = comparison[axis]
            axis_gains.append(float(_errors(a, z, probe, hw).mean()) - error)
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
    hw = tuple(reference.shape[-2:])
    if any(
        abs(theta[i] - IDENTITY[i]) > limit + 1e-8
        for i, limit in enumerate((0.03, 0.03, min(1.5, hw[1] * 0.025), min(1.5, hw[0] * 0.025)))
    ):
        report["reason"] = "outside_safety_bounds"
        return report
    if min(report["axis_objective_separation"]) < max(1e-4, 0.005 * identity):
        report["reason"] = "ambiguous_optimum"
        return report
    reference3, moving3 = reference[:, :, -3:], moving[:, :, -3:]
    a, _ = _features(reference3)
    z, _ = _features(moving3)
    a, z = a.numpy(), z.numpy()
    fits = [_fit(a[i : i + 1], z[i : i + 1], hw)[0] for i in range(a.shape[0])]
    spread = [max(abs(f[i] - theta[i]) for f in fits) for i in range(4)]
    report.update(per_token_transforms=fits, parameter_spread=spread)
    if a.shape[0] < 2:
        report["reason"] = "insufficient_paired_tokens"
        return report
    if any(s > limit for s, limit in zip(spread, (0.0075, 0.0075, 0.375, 0.375), strict=True)):
        report["reason"] = "unstable_prefix_estimates"
        return report
    report.update(accepted=True, reason="stable_paired_prefix", confidence=report["improvement"] * (1 - error))
    return report


def register_pair(reference: torch.Tensor, moving: torch.Tensor, *, comparison=None) -> dict:
    """Diagnostic one-transition fit. It never authorizes a bridge by itself."""
    report = _estimate_registration(reference, moving, comparison=comparison)
    if report["reason"] == "measured":
        report["reason"] = "single_transition_diagnostic"
    return report


def _theil_sen_next(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    slopes = [(arr[j] - arr[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    slope = float(np.median(slopes)) if slopes else 0.0
    predicted = float(np.median([arr[i] + slope * (n - i) for i in range(n)]))
    residuals = np.asarray([arr[i] - (predicted + slope * (i - n)) for i in range(n)])
    dispersion = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    return predicted, slope, dispersion


def _motion_model(exact: torch.Tensor) -> dict:
    p = exact.shape[2]
    transitions = []
    start = max(1, p - MOTION_BASELINE_TRANSITIONS)
    for i in range(start, p):
        transitions.append(register_pair(exact[:, :, i - 1 : i], exact[:, :, i : i + 1]))
    usable = [r for r in transitions if r.get("aligned_error") is not None and r["aligned_error"] <= 0.30]
    report = {
        "transitions": transitions,
        "usable_transitions": len(usable),
        "accepted": False,
        "reason": "insufficient_motion_history",
    }
    if len(usable) < 3:
        return report
    params = [[], [], [], []]
    for item in usable:
        sx, sy, tx, ty = item["transform"]
        params[0].append(math.log(sx))
        params[1].append(math.log(sy))
        params[2].append(tx)
        params[3].append(ty)
    predictions, slopes, dispersions = zip(*(_theil_sen_next(v) for v in params), strict=True)
    expected = (math.exp(predictions[0]), math.exp(predictions[1]), predictions[2], predictions[3])
    report.update(
        accepted=True,
        reason="robust_recent_motion",
        expected_transform=expected,
        trend=(slopes[0], slopes[1], slopes[2], slopes[3]),
        dispersion=(dispersions[0], dispersions[1], dispersions[2], dispersions[3]),
    )
    return report


def _motion_residual_candidate(boundary: dict, motion: dict, hw) -> dict:
    report = {
        "accepted": False,
        "reason": "motion_model_unavailable",
        "transform": IDENTITY,
        "axis_applied": [False] * 4,
    }
    if not motion.get("accepted") or boundary.get("aligned_error") is None:
        return report
    expected = motion["expected_transform"]
    observed = boundary["transform"]
    correction = residual_transform(observed, expected)
    comparison_error = boundary.get("comparison_error")
    comparison_improvement = boundary.get("comparison_improvement", 0.0)
    if comparison_error is None or comparison_improvement < 0.05 or boundary["aligned_error"] > 0.25:
        report.update(reason="boundary_not_better_than_motion_model", raw_transform=correction)
        return report
    dispersions = motion["dispersion"]
    axis_gains = boundary.get("residual_axis_gains", [0.0] * 4)
    h, w = hw
    magnitudes = (abs(math.log(correction[0])), abs(math.log(correction[1])), abs(correction[2]), abs(correction[3]))
    significance = (
        max(_SCALE_FLOOR, 2.5 * dispersions[0]),
        max(_SCALE_FLOOR, 2.5 * dispersions[1]),
        max(_TRANSLATION_FLOOR, 2.5 * dispersions[2]),
        max(_TRANSLATION_FLOOR, 2.5 * dispersions[3]),
    )
    gain_floor = max(1e-4, 0.002 * comparison_error)
    safety = (math.log(1.03), math.log(1.03), min(1.5, w * 0.025), min(1.5, h * 0.025))
    axis_applied = [
        magnitudes[i] >= significance[i] and axis_gains[i] >= gain_floor and magnitudes[i] <= safety[i] + 1e-8
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
        transform=applied,
        axis_applied=axis_applied,
        axis_magnitude=magnitudes,
        axis_significance_threshold=significance,
        axis_objective_gain=axis_gains,
        axis_objective_gain_floor=gain_floor,
        comparison_improvement=comparison_improvement,
    )
    if not any(axis_applied):
        report["reason"] = "motion_residual_not_significant"
        return report
    report.update(accepted=True, reason="significant_motion_residual", confidence=comparison_improvement)
    return report


def _dc(a, b):
    return float((a.float().mean((-2, -1)) - b.float().mean((-2, -1))).square().mean().sqrt())


@torch.no_grad()
def geometric_seam_bridge(learned, exact, *, source_hw, requested):
    """Correct a persistent suffix framing residual relative to recent exact motion.

    Paired-prefix registration remains diagnostic; it no longer drives correction.
    The bridge applies one confidence-gated residual transform to the full learned
    suffix, avoiding the delayed wobble created by a short decay-to-identity.
    """
    started = time.perf_counter()
    p = exact.shape[2]
    h, w = learned.shape[-2:]
    ry, rx = h / source_hw[0], w / source_hw[1]
    prefix_registration = register_prefix(exact, learned[:, :, :p])
    motion = _motion_model(exact)
    comparison = motion.get("expected_transform") if motion.get("accepted") else None
    boundary = register_pair(exact[:, :, -1:], learned[:, :, p : p + 1], comparison=comparison)
    candidate = _motion_residual_candidate(boundary, motion, (h, w))
    accepted = bool(requested and candidate["accepted"])
    report = dict(
        source_hw=source_hw,
        target_hw=(h, w),
        grid_scale_x=rx,
        grid_scale_y=ry,
        grid_anisotropy=abs(rx / ry - 1),
        prefix_registration=prefix_registration,
        transfer_bias_candidate=prefix_registration["accepted"],
        natural_motion_model=motion,
        boundary_before=boundary,
        motion_residual=candidate,
        requested=bool(requested),
        accepted=accepted,
        reason=candidate["reason"] if requested else "disabled",
        policy="persistent_motion_residual",
        persistent_bias_candidate=candidate["accepted"],
        tokens_corrected=0,
        applied_transforms=[],
        out_of_bounds_fraction=0.0,
    )
    corrected = learned
    aligned_last = learned[:, :, p - 1]
    if accepted:
        theta = candidate["transform"]
        corrected = learned.clone()
        for k in range(p, learned.shape[2]):
            corrected[:, :, k] = warp_frame(learned[:, :, k], theta)
        sx, sy, tx, ty = theta
        xx = sx * (torch.arange(w) - (w - 1) / 2) + (w - 1) / 2 + tx
        yy = sy * (torch.arange(h) - (h - 1) / 2) + (h - 1) / 2 + ty
        inside = ((xx >= 0) & (xx <= w - 1))[None, :] & ((yy >= 0) & (yy <= h - 1))[:, None]
        report.update(
            tokens_corrected=int(learned.shape[2] - p),
            applied_transforms=[theta],
            out_of_bounds_fraction=1 - float(inside.float().mean()),
        )
    if prefix_registration["accepted"]:
        aligned_last = warp_frame(aligned_last, prefix_registration["transform"])
    report.update(
        pre_alignment_dc_rms=_dc(exact[:, :, -1], learned[:, :, p - 1]),
        post_geometry_dc_rms=_dc(exact[:, :, -1], aligned_last),
        boundary_after_geometry=register_pair(
            exact[:, :, -1:], corrected[:, :, p : p + 1], comparison=comparison
        ),
        learned_prefix_unchanged=torch.equal(corrected[:, :, :p], learned[:, :, :p]),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    return corrected, aligned_last, report
