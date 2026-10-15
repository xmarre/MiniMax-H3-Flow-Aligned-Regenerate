"""Candidate native graded denoise transition for the first generated target-high tokens.

Target-high starts the generated suffix at the handoff sigma with an exact
binary prefix mask. This opt-in policy gives the first one or two generated
video tokens a fractional Core denoise mask ``m`` for the whole high stage.
Core's native masked-diffusion semantics then apply consistently to those
rows: H3 labels them at ``t = 1 - m * sigma``, the model input is
``m * x + (1 - m) * inject(clean)``, and each prediction is restored as
``m * out + (1 - m) * clean``. The rows are denoised at a reduced effective
sigma rather than frozen.

Core reads ``clean`` from the sampler's ``latent_image``. For generated tokens
the caller's latent is not the handoff clean state, so the high stage receives
a private ``latent_image`` whose transition tokens hold the exact-prefix
restored pre-high clean state. The sampler noise is re-derived against it, so
the high entry state is unchanged. The caller's latent, the exact prefix, the
protected noise and audio are unchanged.

The transition rows finish as a blend of refined output and the pre-high clean
state, so they can lose detail where the two differ. Rendered quality is not
validated. This policy is a candidate and is off unless explicitly requested.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from .geometry import pack_streams, unpack_streams

HIGH_BOUNDARY_TRANSITION_KEY = "h3_flow_high_boundary_transition_v1"
HIGH_BOUNDARY_TRANSITION_POLICY = "native_graded_high_boundary_transition_v1"
# Core quantizes H3 label masks to ceil(m * 256) / 256 and applies an extra blend
# for any remainder, so every weight must be exactly representable in 1/256 steps.
HIGH_BOUNDARY_TRANSITION_WEIGHTS = {1: (0.5,), 2: (0.5, 0.75)}


@dataclasses.dataclass(frozen=True)
class HighBoundaryTransitionRequest:
    tokens: int

    def __post_init__(self):
        if self.tokens not in HIGH_BOUNDARY_TRANSITION_WEIGHTS:
            raise ValueError("high boundary transition supports one or two generated tokens")

    @property
    def weights(self) -> tuple[float, ...]:
        return HIGH_BOUNDARY_TRANSITION_WEIGHTS[self.tokens]


def resolve_high_boundary_transition(model_options) -> HighBoundaryTransitionRequest | None:
    request = (model_options or {}).get(HIGH_BOUNDARY_TRANSITION_KEY)
    if request is None:
        return None
    if not isinstance(request, HighBoundaryTransitionRequest):
        raise TypeError("high boundary transition option must be a HighBoundaryTransitionRequest")
    return request


def prepare_high_boundary_transition(
    request: HighBoundaryTransitionRequest,
    *,
    latent_image: torch.Tensor,
    target_raw: torch.Tensor,
    target_noise: torch.Tensor,
    denoise_mask: torch.Tensor,
    preserved_noise: torch.Tensor,
    transition_clean: torch.Tensor,
    prefix_t: int,
    sigma: float,
    noise_scale: float,
    shapes,
    process_latent_in,
    noise_argument,
    merge_preserved_noise,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return the high-stage latent image, sampler noise and graded mask, with a receipt."""

    weights = request.weights
    tokens = len(weights)
    if any(not 0.0 < w < 1.0 or w * 256 != int(w * 256) for w in weights):
        raise RuntimeError("high boundary transition weights must lie strictly inside (0, 1) in 1/256 steps")
    video_mask, audio_mask = unpack_streams(denoise_mask, shapes)
    temporal = int(video_mask.shape[2])
    if not 0 < prefix_t < temporal - tokens:
        raise RuntimeError("high boundary transition needs generated tokens after the exact prefix")
    if not bool((video_mask[:, :, :prefix_t] == 0).all()) or not bool((video_mask[:, :, prefix_t:] == 1).all()):
        raise RuntimeError("high boundary transition requires the exact binary target-high video mask")
    latent_video, latent_audio = unpack_streams(latent_image, shapes)
    if not torch.equal(unpack_streams(process_latent_in(latent_image), shapes)[0], latent_video):
        raise RuntimeError("high boundary transition requires an identity video latent-in transform")
    expected = (*latent_video.shape[:2], tokens, *latent_video.shape[3:])
    if tuple(transition_clean.shape) != expected or not bool(torch.isfinite(transition_clean).all()):
        raise RuntimeError("high boundary transition clean tokens do not match the target-high geometry")

    high_video = latent_video.clone()
    high_video[:, :, prefix_t : prefix_t + tokens] = transition_clean.to(high_video)
    high_latent = pack_streams((high_video, latent_audio))[0]
    high_noise = merge_preserved_noise(
        noise_argument(target_raw, sigma, process_latent_in(high_latent)), preserved_noise, denoise_mask
    )

    old_video, old_audio = unpack_streams(target_noise, shapes)
    new_video, new_audio = unpack_streams(high_noise, shapes)
    stop = prefix_t + tokens
    if not (
        torch.equal(new_audio, old_audio)
        and torch.equal(new_video[:, :, :prefix_t], old_video[:, :, :prefix_t])
        and torch.equal(new_video[:, :, stop:], old_video[:, :, stop:])
    ):
        raise RuntimeError("high boundary transition changed sampler noise outside its transition tokens")
    state_video = unpack_streams(target_raw, shapes)[0][:, :, prefix_t:stop].float()
    clean = transition_clean.to(state_video)
    implied = new_video[:, :, prefix_t:stop].float()
    rebuilt = sigma * noise_scale * implied + (1.0 - sigma) * clean
    reconstruction_error = float((rebuilt - state_video).abs().max())
    if reconstruction_error > 1e-3 * max(1.0, float(state_video.abs().max())):
        raise RuntimeError("high boundary transition could not reproduce the high entry state")

    graded_video = video_mask.clone()
    for offset, weight in enumerate(weights):
        graded_video[:, :, prefix_t + offset] = weight
    graded = pack_streams((graded_video, audio_mask))[0]
    receipt = {
        "policy": HIGH_BOUNDARY_TRANSITION_POLICY,
        "applied": True,
        "candidate": True,
        "rendered_quality_validated": False,
        "prefix_t": int(prefix_t),
        "transition_tokens": list(range(prefix_t, stop)),
        "mask_weights": list(weights),
        "effective_sigma": [float(sigma) * w for w in weights],
        "entry_sigma": float(sigma),
        "clean_source": "exact_prefix_restored_pre_high_clean",
        "entry_state_reconstruction_max_abs_error": reconstruction_error,
        "implied_noise_std_per_token": implied.std(dim=(0, 1, 3, 4)).tolist(),
        "caller_latent_modified": False,
        "authoritative_prefix_modified": False,
        "audio_modified": False,
        "extra_h3_nfe": 0,
        "extra_vae_calls": 0,
    }
    return high_latent, high_noise, graded, receipt


class H3HighBoundaryTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": False}),
                "transition_tokens": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2,
                        "tooltip": "Generated tokens after the exact prefix that use a graded target-high denoise "
                        "mask: 1 uses 0.5; 2 uses 0.5 then 0.75.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/experimental"
    DESCRIPTION = (
        "Candidate for uniform-source continuations. The first generated tokens refine at a reduced native "
        "masked-diffusion sigma during target-high, anchored to the pre-high clean state. They can lose detail. "
        "Off by default; adds no model or VAE evaluation."
    )

    def patch(self, model, enabled, transition_tokens):
        if not enabled:
            return (model,)
        patched = model.clone()
        patched.model_options[HIGH_BOUNDARY_TRANSITION_KEY] = HighBoundaryTransitionRequest(int(transition_tokens))
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3HighBoundaryTransition": H3HighBoundaryTransition}
NODE_DISPLAY_NAME_MAPPINGS = {"H3HighBoundaryTransition": "MiniMax H3 High Boundary Transition [Candidate]"}

__all__ = [
    "HIGH_BOUNDARY_TRANSITION_KEY",
    "HIGH_BOUNDARY_TRANSITION_POLICY",
    "HIGH_BOUNDARY_TRANSITION_WEIGHTS",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "H3HighBoundaryTransition",
    "HighBoundaryTransitionRequest",
    "prepare_high_boundary_transition",
    "resolve_high_boundary_transition",
]
