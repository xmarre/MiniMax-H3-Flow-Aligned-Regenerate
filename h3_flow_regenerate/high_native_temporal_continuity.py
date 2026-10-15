"""Conservative native masked-diffusion transition for the first generated H3 tokens.

This changes the *model input's sigma* and its timestep labels through Core's
own inpainting contract. It does not copy, warp, interpolate, or drop frames.
Opt-in only; trained-weight visual quality is not yet established.
"""

from __future__ import annotations

import torch

from .geometry import pack_streams, unpack_streams

HIGH_NATIVE_TEMPORAL_CONTINUITY_KEY = "h3_flow_high_native_temporal_continuity_v1"
HIGH_NATIVE_TEMPORAL_CONTINUITY_OFF = "off"
HIGH_NATIVE_TEMPORAL_CONTINUITY_TWO_TOKEN = "native_two_token_v1"
HIGH_NATIVE_TEMPORAL_CONTINUITY_OPTIONS = (
    HIGH_NATIVE_TEMPORAL_CONTINUITY_OFF,
    HIGH_NATIVE_TEMPORAL_CONTINUITY_TWO_TOKEN,
)
# Multiples of 1/256: Core's H3 mask pooling must not change these values.
NATIVE_TWO_TOKEN_WEIGHTS = (0.5, 0.75)


def high_native_temporal_clean_latent(latent_image, clean_video, shapes, prefix_t):
    """Supply the actual learned clean at the graded-mask tokens, not blank suffix."""
    if prefix_t < 1 or prefix_t + len(NATIVE_TWO_TOKEN_WEIGHTS) > shapes[0][2]:
        raise ValueError("native high transition needs protected prefix and two generated tokens")
    original_video, audio = unpack_streams(latent_image, shapes)
    if tuple(original_video.shape) != tuple(clean_video.shape):
        raise ValueError("native high clean/source geometry mismatch")
    replaced = original_video.clone()
    replaced[:, :, prefix_t : prefix_t + 2] = clean_video[:, :, prefix_t : prefix_t + 2].to(replaced)
    packed = pack_streams((replaced, audio))[0]
    original_packed_audio = unpack_streams(latent_image, shapes)[1]
    if not torch.equal(unpack_streams(packed, shapes)[1], original_packed_audio):
        raise RuntimeError("native high transition modified protected audio")
    return packed


def verify_high_native_temporal_entry(
    target_raw, high_noise, high_clean_internal, shapes, prefix_t, sigma, noise_scale
):
    """Require actual target-high conditional state reconstruction at both graded rows."""
    if not 0 < float(sigma) < 1 or not float(noise_scale) > 0:
        raise RuntimeError("native high transition requires valid sigma and noise scale")
    start, stop = prefix_t, prefix_t + 2
    raw, _ = unpack_streams(target_raw, shapes)
    noisy, _ = unpack_streams(high_noise, shapes)
    clean, _ = unpack_streams(high_clean_internal, shapes)
    # OUTER_SAMPLE receives the high noisy state on CUDA while ComfyUI can
    # still hold its private high latent_image on CPU. _noise_argument handles
    # the same split. Move ONLY the two graded tokens, never the whole latent.
    state = raw[:, :, start:stop].float()
    noise_rows = noisy[:, :, start:stop].to(device=state.device, dtype=state.dtype)
    clean_rows = clean[:, :, start:stop].to(device=state.device, dtype=state.dtype)
    rebuilt = float(sigma) * float(noise_scale) * noise_rows + (1.0 - float(sigma)) * clean_rows
    error = float((rebuilt - state).abs().max().item())
    bound = 1e-3 * max(1.0, float(state.abs().max().item()))
    if not bool(torch.isfinite(rebuilt).all().item()) or error > bound:
        raise RuntimeError("native high transition did not reconstruct the high noisy entry state")
    return error


def high_native_temporal_mask(mask, shapes, prefix_t):
    """Modify only two generated VIDEO tokens; exact prefix and AUDIO remain untouched."""
    if prefix_t < 1 or prefix_t + len(NATIVE_TWO_TOKEN_WEIGHTS) > shapes[0][2]:
        raise ValueError("native high temporal transition requires two generated tokens")
    video, audio = unpack_streams(mask, shapes)
    if not bool((video[:, :, :prefix_t] == 0).all()):
        raise ValueError("native high temporal transition requires exact zero VIDEO prefix")
    if not bool((video[:, :, prefix_t:] == 1).all()):
        raise ValueError("native high temporal transition requires generated VIDEO mask of one")
    updated = video.clone()
    for i, weight in enumerate(NATIVE_TWO_TOKEN_WEIGHTS):
        updated[:, :, prefix_t + i] = weight
    result = pack_streams((updated, audio))[0]
    actual_video, actual_audio = unpack_streams(result, shapes)
    if not torch.equal(actual_audio, audio):
        raise RuntimeError("native high temporal transition changed audio masking")
    if not torch.equal(actual_video[:, :, :prefix_t], video[:, :, :prefix_t]):
        raise RuntimeError("native high temporal transition changed protected prefix")
    if not torch.equal(actual_video[:, :, prefix_t + 2 :], video[:, :, prefix_t + 2 :]):
        raise RuntimeError("native high temporal transition changed later suffix")
    # Exact k/256 fractions ensure H3's token-grid ceil() is an identity.
    if not torch.equal((actual_video * 256).ceil() / 256, actual_video):
        raise RuntimeError("native high graded mask would be quantized by Core")
    return result


__all__ = [
    "HIGH_NATIVE_TEMPORAL_CONTINUITY_KEY",
    "HIGH_NATIVE_TEMPORAL_CONTINUITY_OFF",
    "HIGH_NATIVE_TEMPORAL_CONTINUITY_OPTIONS",
    "HIGH_NATIVE_TEMPORAL_CONTINUITY_TWO_TOKEN",
    "NATIVE_TWO_TOKEN_WEIGHTS",
    "high_native_temporal_clean_latent",
    "high_native_temporal_mask",
    "verify_high_native_temporal_entry",
]
