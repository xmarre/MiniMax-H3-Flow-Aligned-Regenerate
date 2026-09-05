"""Non-mutating latent seam diagnostics for progressive exact-prefix handoffs."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

_EPS = 1e-12
_DEFAULT_LOWPASS_KERNEL = 5


def _finite_rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        raise ValueError("seam diagnostic RMS requires a non-empty tensor")
    result = float(value.float().square().mean().sqrt().detach().to(device="cpu").item())
    if not math.isfinite(result):
        raise RuntimeError("seam diagnostic produced a non-finite RMS value")
    return result


def _effective_lowpass_kernel(video: torch.Tensor, requested: int = _DEFAULT_LOWPASS_KERNEL) -> int:
    if video.ndim != 5:
        raise ValueError("seam diagnostic low-pass expects BxCxTxHxW")
    if requested < 1 or requested % 2 == 0:
        raise ValueError("seam diagnostic low-pass kernel must be a positive odd integer")
    height, width = map(int, video.shape[-2:])
    if height < 1 or width < 1:
        raise ValueError("seam diagnostic spatial axes must be non-empty")
    radius = min(requested // 2, (height - 1) // 2, (width - 1) // 2)
    return 2 * max(radius, 0) + 1


def _spatial_lowpass(video: torch.Tensor, kernel: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("seam diagnostic low-pass expects BxCxTxHxW")
    if kernel == 1:
        return video.float()
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("seam diagnostic low-pass kernel must be a positive odd integer")
    batch, channels, frames, height, width = video.shape
    radius = kernel // 2
    work = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).float()
    work = F.pad(work, (radius, radius, radius, radius), mode="replicate")
    work = F.avg_pool2d(work, kernel_size=kernel, stride=1)
    return work.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _delta_metrics(left: torch.Tensor, right: torch.Tensor, *, lowpass_kernel: int) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 5:
        raise ValueError("seam diagnostic comparison tensors must be matching BxCxTxHxW values")
    raw = _finite_rms(left.float() - right.float())
    left_low = _spatial_lowpass(left, lowpass_kernel)
    right_low = _spatial_lowpass(right, lowpass_kernel)
    lowpass = _finite_rms(left_low - right_low)
    left_mean = left.float().mean(dim=(-2, -1))
    right_mean = right.float().mean(dim=(-2, -1))
    spatial_mean = _finite_rms(left_mean - right_mean)
    return {
        "rms": raw,
        "lowpass_rms": lowpass,
        "spatial_mean_rms": spatial_mean,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    result = float(numerator) / max(float(denominator), _EPS)
    if not math.isfinite(result):
        raise RuntimeError("seam diagnostic produced a non-finite ratio")
    return result


def recover_conditional_clean_for_diagnostics(
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    """Invert ``x_t = (1-sigma) * x0 + sigma * noise`` for diagnostics only.

    The returned float32 tensor is never fed back into sampling. The caller must
    supply the exact deterministic noise used by the handoff constructor.
    """
    if state.shape != noise.shape or state.ndim != 5:
        raise ValueError("conditional-clean recovery expects matching BxCxTxHxW state/noise tensors")
    sigma = float(sigma)
    if not math.isfinite(sigma) or not 0.0 <= sigma < 1.0:
        raise ValueError("conditional-clean recovery requires finite sigma in [0, 1)")
    with torch.no_grad():
        return (state.float() - sigma * noise.float()) / (1.0 - sigma)


def measure_exact_prefix_splice(
    upscaled_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    corrected_clean_video: torch.Tensor | None = None,
    requested_lowpass_kernel: int = _DEFAULT_LOWPASS_KERNEL,
) -> dict[str, float | int]:
    """Measure the learned-upscaler seam before and after exact-prefix restoration.

    ``upscaled_clean_video`` is the clean target-grid result produced by the
    learned 3D upscaler before Flow discards its prefix. ``exact_prefix`` is the
    authoritative target-grid prefix that will replace that learned prefix.
    Only scalar diagnostics are returned; neither input is mutated.
    """
    if upscaled_clean_video.ndim != 5 or exact_prefix.ndim != 5:
        raise ValueError("exact-prefix splice diagnostics expect BxCxTxHxW tensors")
    if upscaled_clean_video.shape[:2] != exact_prefix.shape[:2]:
        raise ValueError("exact-prefix splice batch/channel geometry differs")
    if upscaled_clean_video.shape[-2:] != exact_prefix.shape[-2:]:
        raise ValueError("exact-prefix splice spatial geometry differs")
    if corrected_clean_video is not None:
        if corrected_clean_video.ndim != 5 or corrected_clean_video.shape != upscaled_clean_video.shape:
            raise ValueError("corrected exact-prefix splice tensor geometry differs")
        if not corrected_clean_video.is_floating_point():
            raise TypeError("corrected exact-prefix splice tensor must be floating-point")
        if not bool(torch.isfinite(corrected_clean_video).all().item()):
            raise RuntimeError("corrected exact-prefix splice tensor contains NaN or Inf values")
    prefix_t = int(exact_prefix.shape[2])
    temporal = int(upscaled_clean_video.shape[2])
    if prefix_t < 1 or prefix_t >= temporal:
        raise ValueError("exact-prefix splice requires a non-empty prefix shorter than the video")

    lowpass_kernel = _effective_lowpass_kernel(upscaled_clean_video, requested_lowpass_kernel)
    upscaled_prefix = upscaled_clean_video[:, :, :prefix_t]
    learned_last = upscaled_clean_video[:, :, prefix_t - 1 : prefix_t]
    suffix_first = upscaled_clean_video[:, :, prefix_t : prefix_t + 1]
    exact_last = exact_prefix[:, :, -1:]

    with torch.no_grad():
        fields: dict[str, float | int] = {
            "splice_diagnostic_version": 1,
            "splice_prefix_temporal_length": prefix_t,
            "splice_lowpass_kernel": lowpass_kernel,
        }

        full_restore = _delta_metrics(exact_prefix, upscaled_prefix, lowpass_kernel=lowpass_kernel)
        fields.update(
            prefix_restore_rms_all=full_restore["rms"],
            prefix_restore_lowpass_rms_all=full_restore["lowpass_rms"],
            prefix_restore_spatial_mean_rms_all=full_restore["spatial_mean_rms"],
        )
        for tail in (1, 2, 4):
            if prefix_t < tail:
                continue
            tail_restore = _delta_metrics(
                exact_prefix[:, :, -tail:],
                upscaled_prefix[:, :, -tail:],
                lowpass_kernel=lowpass_kernel,
            )
            fields[f"prefix_restore_rms_tail_{tail}"] = tail_restore["rms"]
            fields[f"prefix_restore_lowpass_rms_tail_{tail}"] = tail_restore["lowpass_rms"]
            fields[f"prefix_restore_spatial_mean_rms_tail_{tail}"] = tail_restore["spatial_mean_rms"]

        native = _delta_metrics(suffix_first, learned_last, lowpass_kernel=lowpass_kernel)
        restored = _delta_metrics(suffix_first, exact_last, lowpass_kernel=lowpass_kernel)
        corrected_first = (
            suffix_first if corrected_clean_video is None else corrected_clean_video[:, :, prefix_t : prefix_t + 1]
        )
        corrected = _delta_metrics(corrected_first, exact_last, lowpass_kernel=lowpass_kernel)
        fields.update(
            upscaler_native_seam_rms=native["rms"],
            exact_restored_seam_rms=restored["rms"],
            seam_rms_amplification=_safe_ratio(restored["rms"], native["rms"]),
            upscaler_native_seam_lowpass_rms=native["lowpass_rms"],
            exact_restored_seam_lowpass_rms=restored["lowpass_rms"],
            seam_lowpass_amplification=_safe_ratio(restored["lowpass_rms"], native["lowpass_rms"]),
            upscaler_native_seam_spatial_mean_rms=native["spatial_mean_rms"],
            exact_restored_seam_spatial_mean_rms=restored["spatial_mean_rms"],
            seam_spatial_mean_amplification=_safe_ratio(restored["spatial_mean_rms"], native["spatial_mean_rms"]),
            corrected_exact_seam_rms=corrected["rms"],
            corrected_exact_seam_lowpass_rms=corrected["lowpass_rms"],
            corrected_exact_seam_spatial_mean_rms=corrected["spatial_mean_rms"],
            corrected_over_native_seam_rms_ratio=_safe_ratio(corrected["rms"], native["rms"]),
            corrected_over_native_seam_lowpass_ratio=_safe_ratio(corrected["lowpass_rms"], native["lowpass_rms"]),
            corrected_over_native_seam_spatial_mean_ratio=_safe_ratio(
                corrected["spatial_mean_rms"], native["spatial_mean_rms"]
            ),
            corrected_over_uncorrected_exact_seam_rms_ratio=_safe_ratio(corrected["rms"], restored["rms"]),
            corrected_over_uncorrected_exact_seam_lowpass_ratio=_safe_ratio(
                corrected["lowpass_rms"], restored["lowpass_rms"]
            ),
            corrected_over_uncorrected_exact_seam_spatial_mean_ratio=_safe_ratio(
                corrected["spatial_mean_rms"], restored["spatial_mean_rms"]
            ),
        )
    return fields


def measure_video_boundary(
    video: torch.Tensor,
    prefix_t: int,
    *,
    requested_lowpass_kernel: int = _DEFAULT_LOWPASS_KERNEL,
) -> dict[str, float | int]:
    """Measure the final latent discontinuity at one protected/generated boundary."""
    if video.ndim != 5:
        raise ValueError("video-boundary diagnostics expect BxCxTxHxW")
    prefix_t = int(prefix_t)
    if prefix_t < 1 or prefix_t >= int(video.shape[2]):
        raise ValueError("video-boundary diagnostics require a non-empty prefix shorter than the video")
    lowpass_kernel = _effective_lowpass_kernel(video, requested_lowpass_kernel)
    left = video[:, :, prefix_t - 1 : prefix_t]
    right = video[:, :, prefix_t : prefix_t + 1]
    with torch.no_grad():
        values = _delta_metrics(right, left, lowpass_kernel=lowpass_kernel)
    return {
        "lowpass_kernel": lowpass_kernel,
        "seam_rms": values["rms"],
        "seam_lowpass_rms": values["lowpass_rms"],
        "seam_spatial_mean_rms": values["spatial_mean_rms"],
    }
