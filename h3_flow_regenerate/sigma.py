from __future__ import annotations

import math

import torch

H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0


def _positive_shift(shift: float) -> float:
    value = float(shift)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("flow shift must be finite and positive")
    return value


def flow_shift(t: torch.Tensor | float, shift: float) -> torch.Tensor | float:
    shift = _positive_shift(shift)
    return shift * t / (1.0 + (shift - 1.0) * t)


def inverse_flow_shift(sigma: torch.Tensor | float, shift: float) -> torch.Tensor | float:
    shift = _positive_shift(shift)
    return sigma / (shift + (1.0 - shift) * sigma)


def remap_shift(sigma: torch.Tensor | float, from_shift: float, to_shift: float) -> torch.Tensor | float:
    return flow_shift(inverse_flow_shift(sigma, from_shift), to_shift)


def audio_sigma(
    video_sigma: torch.Tensor | float,
    *,
    video_shift: float = H3_VIDEO_SHIFT,
    audio_shift: float = H3_AUDIO_SHIFT,
) -> torch.Tensor | float:
    return remap_shift(video_sigma, video_shift, audio_shift)


def normalized_coordinate(video_sigma: torch.Tensor | float, *, video_shift: float = H3_VIDEO_SHIFT):
    """Return H3's unshifted full-trajectory coordinate in [0, 1]."""
    return inverse_flow_shift(video_sigma, video_shift)


def resolution_shift_factor(source_area: int, target_area: int, strength: float = 1.0) -> float:
    if source_area <= 0 or target_area <= 0:
        raise ValueError("source and target areas must be positive")
    if not math.isfinite(float(strength)):
        raise ValueError("resolution shift strength must be finite")
    # SD3 Eq. 23 uses alpha=sqrt(m/n). Strength interpolates in log-space,
    # preserving alpha=1 at strength=0 and composition at strength=1.
    return math.exp(0.5 * float(strength) * math.log(target_area / source_area))


def resolution_aware_sigmas(
    sigmas: torch.Tensor,
    *,
    source_area: int,
    target_area: int,
    mode: str = "resolution_aware",
    strength: float = 1.0,
    calibrated_factor: float = 1.0,
    video_shift: float = H3_VIDEO_SHIFT,
) -> torch.Tensor:
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("sigmas must be a one-dimensional tensor with at least two entries")
    if not bool(torch.isfinite(sigmas).all()) or bool((sigmas < 0).any()):
        raise ValueError("sigmas must be finite and non-negative")
    if bool((sigmas[1:] > sigmas[:-1]).any()):
        raise ValueError("sigmas must be monotonically non-increasing")
    if mode == "off":
        return sigmas.clone()
    if mode == "resolution_aware":
        factor = resolution_shift_factor(source_area, target_area, strength)
    elif mode == "calibrated":
        factor = _positive_shift(calibrated_factor)
    else:
        raise ValueError(f"unsupported resolution shift mode {mode!r}")
    base = inverse_flow_shift(sigmas, video_shift)
    mapped = flow_shift(flow_shift(base, factor), video_shift).to(sigmas)
    mapped[0] = sigmas.new_tensor(1.0) if float(sigmas[0]) == 1.0 else mapped[0]
    mapped[-1] = sigmas.new_tensor(0.0) if float(sigmas[-1]) == 0.0 else mapped[-1]
    if not bool(torch.isfinite(mapped).all()) or bool((mapped[1:] > mapped[:-1]).any()):
        raise RuntimeError("resolution-aware mapping broke the sigma schedule invariant")
    return mapped


def interpolate_coordinate(a: float, b: float, target: float) -> float:
    if not all(math.isfinite(v) for v in (a, b, target)):
        raise ValueError("interpolation coordinates must be finite")
    if a == b:
        raise ValueError("cannot interpolate across a zero-width coordinate interval")
    weight = (target - a) / (b - a)
    return min(1.0, max(0.0, weight))
