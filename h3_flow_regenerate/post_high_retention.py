"""Bounded post-high retention for exact-prefix continuation boundaries.

The 00692 hardware evidence localizes the remaining decoded video impulse to the
native target-high stage: the exact-restored pre-high boundary is smooth and the
same boundary is discontinuous again after high refinement. Audio exhibits a
parallel feature-space drift across target-high even when scalar latent RMS looks
nearly continuous.

This module never samples. It blends only already-computed clean-domain boundary
support from the pre-high reference into the post-high result and leaves the
caller-owned exact prefix plus all suffix values outside the declared support
untouched.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch

POST_HIGH_VIDEO_RETENTION_WEIGHTS = (1.0, 0.75, 0.5, 0.25)
# Continuum's context-safe PT214 diagnostic excludes 30 native 40-Hz audio
# latents at each decoder edge. Use the same conservative support length while
# returning the retained pre-high audio state continuously to native target-high.
POST_HIGH_AUDIO_RETENTION_TICKS = 30


def linear_return_weights(support: int) -> tuple[float, ...]:
    """Return a monotonic 1 -> 1/support taper with a matching final exit step."""

    support = int(support)
    if support < 2:
        raise ValueError("post-high retention support must be at least two")
    return tuple(1.0 - float(offset) / float(support) for offset in range(support))


def exact_audio_prefix_ticks(exact_audio_mask: torch.Tensor) -> int:
    """Return contiguous protected-prefix length for native BxCx2xT audio masks."""

    if not torch.is_tensor(exact_audio_mask) or exact_audio_mask.ndim != 4:
        raise ValueError("post-high audio retention requires a native BxCx2xT mask")
    if int(exact_audio_mask.shape[2]) != 2:
        raise ValueError("post-high audio retention requires native H3 audio geometry")

    temporal_min = exact_audio_mask.amin(dim=(0, 1, 2))
    temporal_max = exact_audio_mask.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1.0e-8
    exact_one = temporal_min >= 1.0 - 1.0e-8
    temporal = int(exact_audio_mask.shape[-1])

    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    if prefix <= 0 or prefix >= temporal or not bool(exact_one[prefix:].all().item()):
        raise ValueError("post-high audio retention requires a contiguous exact prefix followed by generated suffix")
    return prefix


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().to(dtype=torch.float32).square().mean().sqrt().item())


def _correction_report(
    before: torch.Tensor,
    after: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: tuple[float, ...],
) -> dict[str, Any]:
    correction = after.detach().to(dtype=torch.float32) - before.detach().to(dtype=torch.float32)
    native_delta = reference.detach().to(device=before.device, dtype=torch.float32) - before.detach().to(
        dtype=torch.float32
    )
    per_token = [_rms(correction[..., index]) for index in range(int(correction.shape[-1]))]
    reference_delta = [_rms(native_delta[..., index]) for index in range(int(native_delta.shape[-1]))]
    adjacent = []
    if int(correction.shape[-1]) > 1:
        adjacent.extend(
            _rms(correction[..., index] - correction[..., index - 1]) for index in range(1, int(correction.shape[-1]))
        )
    adjacent.append(_rms(-correction[..., -1]))
    return {
        "weights": list(weights),
        "support": len(weights),
        "correction_rms_by_token": per_token,
        "reference_delta_rms_by_token": reference_delta,
        "first_correction_rms": per_token[0],
        "last_correction_rms": per_token[-1],
        "max_adjacent_correction_step_rms": max(adjacent),
        "exit_correction_step_rms": adjacent[-1],
    }


def apply_post_high_boundary_retention(
    final_video: torch.Tensor,
    final_audio: torch.Tensor,
    *,
    video_prefix_t: int,
    video_reference_suffix: torch.Tensor,
    audio_prefix_t: int,
    audio_reference_suffix: torch.Tensor,
    video_weights: tuple[float, ...] = POST_HIGH_VIDEO_RETENTION_WEIGHTS,
    audio_support_ticks: int = POST_HIGH_AUDIO_RETENTION_TICKS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Retain bounded pre-high AV boundary state inside the generated suffix.

    The video support is four H3 video-latent frames. Audio uses a linear 30-tick
    return matching the conservative decoder-context margin used by Continuum's
    PT214 diagnostic. The first generated token/tick is exactly the pre-high
    reference; later values smoothly hand ownership back to native target-high.
    """

    if final_video.ndim != 5 or final_audio.ndim != 4 or int(final_audio.shape[2]) != 2:
        raise ValueError("post-high retention requires native H3 video/audio tensors")
    if not bool(torch.isfinite(final_video).all().item()) or not bool(torch.isfinite(final_audio).all().item()):
        raise ValueError("post-high retention requires finite final tensors")

    video_prefix_t = int(video_prefix_t)
    audio_prefix_t = int(audio_prefix_t)
    video_weights = tuple(float(weight) for weight in video_weights)
    audio_weights = linear_return_weights(audio_support_ticks)
    if not video_weights or video_weights[0] != 1.0:
        raise ValueError("post-high video retention must begin with full pre-high ownership")
    if any(not 0.0 < weight <= 1.0 for weight in video_weights):
        raise ValueError("post-high video retention weights must be inside (0, 1]")
    if any(right >= left for left, right in pairwise(video_weights)):
        raise ValueError("post-high video retention weights must decrease strictly")

    video_support = len(video_weights)
    audio_support = len(audio_weights)
    if video_prefix_t <= 0 or video_prefix_t + video_support > int(final_video.shape[2]):
        raise ValueError("post-high video retention support exceeds generated suffix")
    if audio_prefix_t <= 0 or audio_prefix_t + audio_support > int(final_audio.shape[-1]):
        raise ValueError("post-high audio retention support exceeds generated suffix")

    expected_video_shape = (
        int(final_video.shape[0]),
        int(final_video.shape[1]),
        video_support,
        int(final_video.shape[3]),
        int(final_video.shape[4]),
    )
    expected_audio_shape = (
        int(final_audio.shape[0]),
        int(final_audio.shape[1]),
        int(final_audio.shape[2]),
        audio_support,
    )
    if tuple(video_reference_suffix.shape) != expected_video_shape:
        raise ValueError(
            "post-high video reference geometry mismatch: "
            f"{tuple(video_reference_suffix.shape)} != {expected_video_shape}"
        )
    if tuple(audio_reference_suffix.shape) != expected_audio_shape:
        raise ValueError(
            "post-high audio reference geometry mismatch: "
            f"{tuple(audio_reference_suffix.shape)} != {expected_audio_shape}"
        )
    if not bool(torch.isfinite(video_reference_suffix).all().item()) or not bool(
        torch.isfinite(audio_reference_suffix).all().item()
    ):
        raise ValueError("post-high retention references must be finite")

    corrected_video = final_video.clone()
    corrected_audio = final_audio.clone()
    video_before = corrected_video[:, :, video_prefix_t : video_prefix_t + video_support].clone()
    audio_before = corrected_audio[..., audio_prefix_t : audio_prefix_t + audio_support].clone()
    video_reference = video_reference_suffix.to(device=corrected_video.device, dtype=corrected_video.dtype)
    audio_reference = audio_reference_suffix.to(device=corrected_audio.device, dtype=corrected_audio.dtype)

    for offset, weight in enumerate(video_weights):
        corrected_video[:, :, video_prefix_t + offset].lerp_(video_reference[:, :, offset], weight)
    for offset, weight in enumerate(audio_weights):
        corrected_audio[..., audio_prefix_t + offset].lerp_(audio_reference[..., offset], weight)

    if not torch.equal(corrected_video[:, :, :video_prefix_t], final_video[:, :, :video_prefix_t]):
        raise RuntimeError("post-high retention modified the authoritative video prefix")
    if not torch.equal(
        corrected_video[:, :, video_prefix_t + video_support :],
        final_video[:, :, video_prefix_t + video_support :],
    ):
        raise RuntimeError("post-high retention modified video outside bounded support")
    if not torch.equal(corrected_audio[..., :audio_prefix_t], final_audio[..., :audio_prefix_t]):
        raise RuntimeError("post-high retention modified the authoritative audio prefix")
    if not torch.equal(
        corrected_audio[..., audio_prefix_t + audio_support :],
        final_audio[..., audio_prefix_t + audio_support :],
    ):
        raise RuntimeError("post-high retention modified audio outside bounded support")

    video_after = corrected_video[:, :, video_prefix_t : video_prefix_t + video_support]
    audio_after = corrected_audio[..., audio_prefix_t : audio_prefix_t + audio_support]
    video_report = _correction_report(
        video_before.movedim(2, -1),
        video_after.movedim(2, -1),
        video_reference.movedim(2, -1),
        weights=video_weights,
    )
    audio_report = _correction_report(
        audio_before,
        audio_after,
        audio_reference,
        weights=audio_weights,
    )

    return (
        corrected_video,
        corrected_audio,
        {
            "policy": "partitioned_post_high_boundary_retention_v1",
            "applied": True,
            "video": video_report,
            "audio": audio_report,
            "video_prefix_t": video_prefix_t,
            "audio_prefix_ticks": audio_prefix_t,
            "authoritative_video_prefix_modified": False,
            "authoritative_audio_prefix_modified": False,
            "video_outside_support_modified": False,
            "audio_outside_support_modified": False,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
        },
    )
