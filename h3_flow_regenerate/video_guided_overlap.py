"""Bounded sampler-lifetime video overlap for exact-prefix H3 continuation.

The caller-owned carried video prefix remains authoritative at the framework
boundary. During the target-grid high sampler lifetime this helper can feather
only the tail of a canonical exact video prefix onto the normal denoise mask.
That gives the model a short generated temporal runway before the nominal chunk
boundary while the final exact-mask canonicalization still restores every
caller-owned prefix latent byte-for-byte.

This is deliberately not an RGB crossfade. It is the sampling-side half of an
overlap-add experiment, modeled after the released audio overlap contract.
"""

from __future__ import annotations

from typing import Any

import torch

from .geometry import unpack_streams

_MASK_QUANTIZATION_LEVELS = 256.0


def validate_video_guided_overlap_tokens(
    value: int,
    *,
    source: str = "video guided overlap",
) -> int:
    """Validate a non-negative H3 video-latent overlap request."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{source} must be a non-negative integer, got {value!r}")
    return int(value)


def apply_video_guided_overlap_mask(
    denoise_mask: torch.Tensor | None,
    latent_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None,
    *,
    tokens: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Feather the tail of a canonical exact video prefix for sampling only.

    Native H3 continuation uses a contiguous zero-valued carried video prefix
    followed by a one-valued generated suffix. For an applied width N the last N
    protected video latent tokens receive a monotonic 1/(N+1)..N/(N+1)
    denoise-strength ramp, rounded upward to Core H3's 1/256 mask grid.

    Audio is untouched. The caller-owned exact mask is not mutated and remains
    authoritative for the final output restore. The applied width uses the
    available carried prefix, including the whole prefix when requested.
    Non-canonical partially protected layouts retain their existing rejection.
    """

    tokens = validate_video_guided_overlap_tokens(tokens)
    report: dict[str, Any] = {
        "requested_tokens": tokens,
        "applied_tokens": 0,
        "width_limited_by_prefix": False,
        "applied": False,
        "reason": "disabled" if tokens <= 0 else "pending",
        "video_prefix_tokens": 0,
        "video_total_tokens": 0,
        "hard_prefix_tokens": 0,
        "ramp_values": [],
        "mask_grid": "ceil_1_over_256",
        "original_exact_output_restore": True,
        "audio_mask_modified": False,
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
    }
    if tokens <= 0:
        return denoise_mask, report
    if denoise_mask is None:
        report["reason"] = "no_denoise_mask"
        return denoise_mask, report
    if latent_shapes is None or len(latent_shapes) != 2:
        raise ValueError("video guided overlap requires native packed H3 video/audio shapes")
    if denoise_mask.ndim != 3 or int(denoise_mask.shape[1]) != 1:
        raise ValueError("video guided overlap requires packed Bx1xN denoise mask")

    runtime_mask = denoise_mask.clone()
    video_mask, audio_mask = unpack_streams(runtime_mask, latent_shapes)
    exact_video_mask, exact_audio_mask = unpack_streams(denoise_mask, latent_shapes)
    if video_mask.ndim != 5:
        raise ValueError("video guided overlap requires native BxCxTxHxW video mask")
    if tuple(video_mask.shape) != tuple(exact_video_mask.shape):
        raise RuntimeError("video guided overlap lost exact video-mask geometry")

    temporal_min = exact_video_mask.amin(dim=(0, 1, 3, 4))
    temporal_max = exact_video_mask.amax(dim=(0, 1, 3, 4))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8
    temporal = int(video_mask.shape[2])
    report["video_total_tokens"] = temporal

    if bool(exact_one.all().item()):
        report["reason"] = "no_exact_video_prefix"
        return denoise_mask, report
    if bool(exact_zero.all().item()):
        report.update(
            reason="no_generated_video_suffix",
            video_prefix_tokens=temporal,
            hard_prefix_tokens=temporal,
        )
        return denoise_mask, report

    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    report["video_prefix_tokens"] = prefix
    if prefix <= 0:
        raise ValueError("video guided overlap found protected video without a leading exact prefix")
    if not bool(exact_one[prefix:].all().item()):
        raise ValueError(
            "video guided overlap requires a contiguous exact video prefix followed by a fully generated suffix"
        )
    overlap_tokens = min(tokens, prefix)
    raw_ramp = torch.arange(1, overlap_tokens + 1, device=video_mask.device, dtype=torch.float32) / float(
        overlap_tokens + 1
    )
    ramp = torch.ceil(raw_ramp * _MASK_QUANTIZATION_LEVELS) / _MASK_QUANTIZATION_LEVELS
    ramp = ramp.to(dtype=video_mask.dtype)
    start = prefix - overlap_tokens
    stop = prefix
    video_mask[:, :, start:stop, :, :] = ramp.view(1, 1, -1, 1, 1)

    if not torch.equal(audio_mask, exact_audio_mask.to(device=audio_mask.device, dtype=audio_mask.dtype)):
        raise RuntimeError("video guided overlap modified audio sampler-mask values")

    report.update(
        applied=True,
        applied_tokens=overlap_tokens,
        width_limited_by_prefix=tokens > prefix,
        reason="exact_video_prefix_tail",
        hard_prefix_tokens=start,
        ramp_values=[float(value) for value in ramp.detach().to(device="cpu", dtype=torch.float32).tolist()],
        ramp_start_token=start,
        ramp_stop_token=stop,
    )
    return runtime_mask, report


__all__ = [
    "apply_video_guided_overlap_mask",
    "validate_video_guided_overlap_tokens",
]
