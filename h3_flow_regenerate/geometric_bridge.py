"""Conservative, DC-insensitive same-frame registration for learned H3 latents.

Transform convention: output->input about the image centre, in latent pixels:
input_x = sx * (output_x - (W-1)/2) + (W-1)/2 + tx (and likewise y).
Thus positive tx samples to the right and moves visible content LEFT. These are
sampling parameters, not a forward optical-flow displacement. No gradients used.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

IDENTITY = (1.0, 1.0, 0.0, 0.0)
BRIDGE_WEIGHTS = (1.0, 0.5, 0.25)


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
    # Explicit float32 and autocast exclusion also support CPU half/bfloat16.
    with torch.autocast(device_type=frame.device.type, enabled=False):
        theta = frame.new_tensor([[sx, 0, 2 * tx / w], [0, sy, 2 * ty / h]], dtype=torch.float32)
        grid = F.affine_grid(theta[None].expand(frame.shape[0], -1, -1), frame.shape, align_corners=False)
        return F.grid_sample(frame.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(frame)


def _features(video):
    # Bound search memory/compute; all channels participate, never select one.
    b, c, t, h, w = video.shape
    x = video.detach().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    x = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    ratio = min(1.0, 48 / max(h, w))
    x = F.interpolate(x, size=(round(h * ratio), round(w * ratio)), mode="area")
    # A small CPU search avoids hundreds of GPU launches/synchronizations.
    x = x.cpu()
    x = x - x.mean((-2, -1), keepdim=True)
    rms = x.square().mean((-2, -1), keepdim=True).sqrt()
    return x / rms.clamp_min(1e-4), rms


def _errors(reference, moving, transform, original_hw):
    # NumPy elementwise operations avoid PyTorch's process-global CPU thread
    # pool overhead on these tiny candidates. No thread settings are mutated.
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
    # Fixed interior domain for every candidate. Padding cannot win the fit.
    a, z = reference[..., 3:-3, 3:-3], warped[..., 3:-3, 3:-3]
    a = a - a.mean((-2, -1), keepdims=True)
    z = z - z.mean((-2, -1), keepdims=True)
    a = a / np.maximum(np.sqrt((a * a).mean((-2, -1), keepdims=True)), 1e-4)
    z = z / np.maximum(np.sqrt((z * z).mean((-2, -1), keepdims=True)), 1e-4)
    # Winsorized residual limits isolated local detail influence.
    return np.minimum((a - z) ** 2, 4).mean((1, 2, 3))


def _fit(a, z, hw):
    theta = list(IDENTITY)
    loss = float(_errors(a, z, theta, hw).mean())
    # Search extends beyond application bounds: an optimum outside is rejected.
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


@torch.no_grad()
def register_prefix(reference: torch.Tensor, moving: torch.Tensor) -> dict:
    """Joint last-three-token fit, with independent temporal consistency checks.

    Confidence is a deterministic gate score, not a calibrated probability.
    Consecutive-time diagnostics use this same estimator but NEVER drive a warp.
    """
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
    # Reject one-dimensional ramps/stripes: both geometric axes must be observable.
    gx = (a[..., 1:] - a[..., :-1]).square().mean()
    gy = (a[..., 1:, :] - a[..., :-1, :]).square().mean()
    if min(float(gx), float(gy)) < 1e-3:
        report["reason"] = "insufficient_axis_structure"
        return report
    a, z = a.numpy(), z.numpy()
    identity = float(_errors(a, z, IDENTITY, hw).mean())
    theta, error = _fit(a, z, hw)
    improvement = max(0.0, (identity - error) / max(identity, 1e-8))
    report.update(
        _transform_fields(theta, hw),
        transform=theta,
        identity_error=identity,
        aligned_error=error,
        improvement=improvement,
        paired_tokens=a.shape[0],
    )
    if identity < 1e-4 or improvement < 0.15:
        report["reason"] = "identity_or_no_material_improvement"
        return report
    if error > 0.25:
        report["reason"] = "poor_alignment"
        return report
    if any(
        abs(theta[i] - IDENTITY[i]) > limit + 1e-8
        for i, limit in enumerate((0.03, 0.03, min(1.5, hw[1] * 0.025), min(1.5, hw[0] * 0.025)))
    ):
        report["reason"] = "outside_safety_bounds"
        return report
    # A narrow well-defined optimum is required along every fitted axis.
    # This rejects ambiguous scale/translation trade-offs in weak representations.
    separations = []
    for axis, step in enumerate((0.01, 0.01, 0.5, 0.5)):
        losses = []
        for sign in (-1, 1):
            probe = list(theta)
            probe[axis] += sign * step
            losses.append(float(_errors(a, z, probe, hw).mean()) - error)
        separations.append(min(losses))
    report["axis_objective_separation"] = separations
    if min(separations) < max(1e-4, 0.005 * identity):
        report["reason"] = "ambiguous_optimum"
        return report
    fits = [_fit(a[i : i + 1], z[i : i + 1], hw)[0] for i in range(a.shape[0])]
    spread = [max(abs(f[i] - theta[i]) for f in fits) for i in range(4)]
    report.update(per_token_transforms=fits, parameter_spread=spread)
    if a.shape[0] < 2:
        report["reason"] = "insufficient_paired_tokens"
        return report
    if any(s > limit for s, limit in zip(spread, (0.0075, 0.0075, 0.375, 0.375), strict=True)):
        report["reason"] = "unstable_prefix_estimates"
        return report
    report.update(accepted=True, reason="stable_paired_prefix", confidence=improvement * (1 - error))
    return report


def _dc(a, b):
    return float((a.float().mean((-2, -1)) - b.float().mean((-2, -1))).square().mean().sqrt())


@torch.no_grad()
def geometric_seam_bridge(learned, exact, *, source_hw, requested):
    """Return original object on rejection/off; never alter the learned prefix.

    Short provisional seam correction only. Stable prefix bias does not establish
    persistence in generated content; full-suffix warping awaits real-media evidence.
    """
    started = time.perf_counter()
    p = exact.shape[2]
    registration = register_prefix(exact, learned[:, :, :p])
    h, w = learned.shape[-2:]
    ry, rx = h / source_hw[0], w / source_hw[1]
    accepted = requested and registration["accepted"]
    report = dict(
        source_hw=source_hw,
        target_hw=(h, w),
        grid_scale_x=rx,
        grid_scale_y=ry,
        grid_anisotropy=abs(rx / ry - 1),
        prefix_registration=registration,
        requested=requested,
        accepted=accepted,
        reason=registration["reason"] if requested else "disabled",
        policy="transient_provisional",
        persistent_bias_candidate=registration["accepted"],
        tokens_corrected=0,
        applied_transforms=[],
        out_of_bounds_fraction=0.0,
        boundary_before=register_prefix(exact[:, :, -1:], learned[:, :, p : p + 1]),
        natural_motion_baseline=[
            register_prefix(exact[:, :, i - 1 : i], exact[:, :, i : i + 1]) for i in range(max(1, p - 2), p)
        ],
    )
    corrected, aligned_last = learned, learned[:, :, p - 1]
    if accepted:
        corrected = learned.clone()
        theta = registration["transform"]
        aligned_last = warp_frame(aligned_last, theta)
        transforms = []
        for k, weight in enumerate(BRIDGE_WEIGHTS[: learned.shape[2] - p]):
            transform = (
                math.exp(weight * math.log(theta[0])),
                math.exp(weight * math.log(theta[1])),
                weight * theta[2],
                weight * theta[3],
            )
            corrected[:, :, p + k] = warp_frame(learned[:, :, p + k], transform)
            transforms.append(transform)
        sx, sy, tx, ty = theta
        xx = sx * (torch.arange(w) - (w - 1) / 2) + (w - 1) / 2 + tx
        yy = sy * (torch.arange(h) - (h - 1) / 2) + (h - 1) / 2 + ty
        inside = ((xx >= 0) & (xx <= w - 1))[None, :] & ((yy >= 0) & (yy <= h - 1))[:, None]
        report.update(
            tokens_corrected=len(transforms),
            applied_transforms=transforms,
            out_of_bounds_fraction=1 - float(inside.float().mean()),
        )
    report.update(
        pre_alignment_dc_rms=_dc(exact[:, :, -1], learned[:, :, p - 1]),
        post_geometry_dc_rms=_dc(exact[:, :, -1], aligned_last),
        boundary_after_geometry=register_prefix(exact[:, :, -1:], corrected[:, :, p : p + 1]),
        learned_prefix_unchanged=torch.equal(corrected[:, :, :p], learned[:, :, :p]),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
    return corrected, aligned_last, report
