"""One-token high-stage video guard for partitioned H3 continuation.

The low/probe handoff already owns the first generated target-grid clean token.
This module promotes exactly that token to protected high-stage context while
leaving caller prefix ownership, audio ownership, and every later suffix token
unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from .geometry import pack_streams, unpack_streams

HIGH_STAGE_VIDEO_GUARD_POLICY = "first_generated_token_exact_high_context_v1"
HIGH_STAGE_VIDEO_GUARD_TOKENS = 1
HIGH_STAGE_VIDEO_GUARD_MAX_ENDPOINT_DELTA = 2e-5


def build_high_stage_video_guard(
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor,
    shapes: list[tuple[int, ...]],
    *,
    guard_video_caller: torch.Tensor,
    prefix_t: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return high-stage latent/mask with one generated video token protected."""

    if len(shapes) != 2:
        raise ValueError("high-stage video guard requires packed H3 video/audio shapes")
    video, audio = unpack_streams(latent_image, shapes)
    video_mask, audio_mask = unpack_streams(denoise_mask, shapes)
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("high-stage video guard requires Bx24xTxHxW video")
    if tuple(guard_video_caller.shape) != tuple(video.shape):
        raise ValueError("high-stage video guard source geometry differs from caller video")
    prefix_t = int(prefix_t)
    if not 0 < prefix_t < int(video.shape[2]):
        raise ValueError("high-stage video guard requires a non-empty exact prefix and generated suffix")

    original_prefix = video[:, :, :prefix_t]
    original_guard = video[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS]
    original_later_suffix = video[:, :, prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS :]
    guard_source = guard_video_caller[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS]
    if guard_source.shape[2] != HIGH_STAGE_VIDEO_GUARD_TOKENS:
        raise RuntimeError("high-stage video guard source does not contain the requested token")

    original_guard_mask = video_mask[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS]
    if not bool(torch.all(original_guard_mask == 1).item()):
        raise RuntimeError("high-stage video guard token was not generated in the caller mask")

    guarded_video = video.clone()
    guarded_video[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS] = guard_source.to(guarded_video)
    guarded_video_mask = video_mask.clone()
    guarded_video_mask[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS] = 0

    guarded_latent, guarded_shapes = pack_streams((guarded_video, audio))
    guarded_mask, guarded_mask_shapes = pack_streams((guarded_video_mask, audio_mask))
    if guarded_shapes != shapes or guarded_mask_shapes != shapes:
        raise RuntimeError("high-stage video guard changed packed AV geometry")

    checked_video, checked_audio = unpack_streams(guarded_latent, shapes)
    checked_video_mask, checked_audio_mask = unpack_streams(guarded_mask, shapes)
    prefix_exact = torch.equal(checked_video[:, :, :prefix_t], original_prefix)
    audio_exact = torch.equal(checked_audio, audio)
    audio_mask_exact = torch.equal(checked_audio_mask, audio_mask)
    later_suffix_exact = torch.equal(
        checked_video[:, :, prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS :],
        original_later_suffix,
    )
    mask_outside_guard_exact = torch.equal(
        torch.cat(
            (
                checked_video_mask[:, :, :prefix_t],
                checked_video_mask[:, :, prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS :],
            ),
            dim=2,
        ),
        torch.cat(
            (
                video_mask[:, :, :prefix_t],
                video_mask[:, :, prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS :],
            ),
            dim=2,
        ),
    )
    guard_mask_exact_zero = bool(
        torch.all(checked_video_mask[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS] == 0).item()
    )
    guard_source_exact = torch.equal(
        checked_video[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS],
        guard_source.to(checked_video),
    )
    if not all(
        (
            prefix_exact,
            audio_exact,
            audio_mask_exact,
            later_suffix_exact,
            mask_outside_guard_exact,
            guard_mask_exact_zero,
            guard_source_exact,
        )
    ):
        raise RuntimeError("high-stage video guard violated its bounded ownership contract")

    changed_from_caller = not torch.equal(
        original_guard,
        checked_video[:, :, prefix_t : prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS],
    )
    return (
        guarded_latent,
        guarded_mask,
    {
            "policy": HIGH_STAGE_VIDEO_GUARD_POLICY,
            "enabled": True,
            "applied": True,
            "guard_tokens": HIGH_STAGE_VIDEO_GUARD_TOKENS,
            "guard_token_index": prefix_t,
            "caller_exact_prefix_tokens": prefix_t,
            "high_protected_video_tokens": prefix_t + HIGH_STAGE_VIDEO_GUARD_TOKENS,
            "guard_source": "exact_restored_pre_high_first_suffix_clean",
            "guard_source_exact": guard_source_exact,
            "guard_differs_from_caller_latent": changed_from_caller,
            "caller_prefix_modified": False,
            "caller_prefix_exact": prefix_exact,
            "audio_modified": False,
            "audio_exact": audio_exact,
            "audio_mask_modified": False,
            "audio_mask_exact": audio_mask_exact,
            "later_suffix_modified": False,
            "later_suffix_exact": later_suffix_exact,
            "mask_outside_guard_modified": False,
            "mask_outside_guard_exact": mask_outside_guard_exact,
            "original_guard_mask_exact_one": True,
            "high_guard_mask_exact_zero": guard_mask_exact_zero,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
        }
        )


def disabled_high_stage_video_guard(*, prefix_t: int, reason: str) -> dict[str, Any]:
    return {
        "policy": HIGH_STAGE_VIDEO_GUARD_POLICY,
        "enabled": False,
        "applied": False,
        "guard_tokens": 0,
        "guard_token_index": int(prefix_t),
        "caller_exact_prefix_tokens": int(prefix_t),
        "high_protected_video_tokens": int(prefix_t),
        "reason": str(reason),
        "caller_prefix_modified": False,
        "audio_modified": False,
        "audio_mask_modified": False,
        "later_suffix_modified": False,
        "mask_outside_guard_modified": False,
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
    }


__all__ = [
    "HIGH_STAGE_VIDEO_GUARD_MAX_ENDPOINT_DELTA",
    "HIGH_STAGE_VIDEO_GUARD_POLICY",
    "HIGH_STAGE_VIDEO_GUARD_TOKENS",
    "build_high_stage_video_guard",
    "disabled_high_stage_video_guard",
]
