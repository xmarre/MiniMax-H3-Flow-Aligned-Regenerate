"""Input-identity witness for controlled uniform-source low-sampler A/B.

Captures only hashes/metadata during an explicitly requested boundary witness.
No sampler or model mutation is performed. Matching receipts is a necessary,
not sufficient, condition for causal paired continuation comparisons.
"""

from __future__ import annotations

import torch

from .geometry import unpack_streams
from .partitioned_stage import tensor_sha256

POLICY = "h3_uniform_source_low_input_pairing_v1"

# The protected source prefix is the ONLY low-sampler input this experiment
# may vary. All entries below must be identical across the two arms.
PAIRED_FIELDS = (
    "authoritative_target_prefix_sha256",
    "source_video_initial_noise_sha256",
    "source_suffix_initial_noise_sha256",
    "audio_initial_noise_sha256",
    "source_suffix_initial_latent_sha256",
    "initial_audio_latent_sha256",
    "source_video_mask_sha256",
    "audio_mask_sha256",
    "low_sigmas_sha256",
    "conditioning_signature",
    "sampler",
    "seed",
    "source_hw",
    "target_hw",
    "prefix_t",
    "source_temporal",
    "video_dtype",
    "audio_dtype",
)


def make_low_input_pairing_receipt(
    *,
    authoritative_target_prefix: torch.Tensor,
    source_prefix: torch.Tensor,
    low_noise: torch.Tensor,
    low_latent_image: torch.Tensor,
    low_mask: torch.Tensor,
    low_shapes: list[tuple[int, ...]],
    low_sigmas: torch.Tensor,
    prefix_t: int,
    source_policy: str,
    sampler: str,
    seed: int,
    conditioning_signature: str,
) -> dict:
    if len(low_shapes) != 2 or type(prefix_t) is not int or prefix_t < 2:
        raise ValueError("low input witness requires H3 AV shapes and native prefix length")
    if not all(torch.is_tensor(x) for x in (low_noise, low_latent_image, low_mask, low_sigmas)):
        raise TypeError("low input witness requires concrete sampler tensors")
    initial_video, initial_audio = unpack_streams(low_latent_image, low_shapes)
    noise_video, noise_audio = unpack_streams(low_noise, low_shapes)
    mask_video, mask_audio = unpack_streams(low_mask, low_shapes)
    if (
        initial_video.shape != noise_video.shape
        or initial_audio.shape != noise_audio.shape
        or mask_video.shape != initial_video.shape
        or mask_audio.shape != initial_audio.shape
        or source_prefix.shape != initial_video[:, :, :prefix_t].shape
        or authoritative_target_prefix.shape[:3] != source_prefix.shape[:3]
        or source_prefix.shape[-2:] == authoritative_target_prefix.shape[-2:]
        or initial_video.shape[2] <= prefix_t
        or (initial_video.shape[2] - 2) % 5 != 0
        or (prefix_t - 2) % 5 != 0
    ):
        raise ValueError("low input witness shape or native temporal phase mismatch")
    if not torch.equal(
        initial_video[:, :, :prefix_t].contiguous().view(torch.uint8),
        source_prefix.contiguous().view(torch.uint8),
    ):
        raise ValueError("low input witness source prefix differs from actual low sampler input")
    return {
        "policy": POLICY,
        "source_policy": str(source_policy),
        "authoritative_target_prefix_sha256": tensor_sha256(authoritative_target_prefix),
        "source_prefix_sha256": tensor_sha256(source_prefix),
        "source_video_initial_noise_sha256": tensor_sha256(noise_video),
        "source_suffix_initial_noise_sha256": tensor_sha256(noise_video[:, :, prefix_t:]),
        "audio_initial_noise_sha256": tensor_sha256(noise_audio),
        "source_suffix_initial_latent_sha256": tensor_sha256(initial_video[:, :, prefix_t:]),
        "initial_audio_latent_sha256": tensor_sha256(initial_audio),
        "source_video_mask_sha256": tensor_sha256(mask_video),
        "audio_mask_sha256": tensor_sha256(mask_audio),
        "low_sigmas_sha256": tensor_sha256(low_sigmas),
        "conditioning_signature": str(conditioning_signature),
        "sampler": str(sampler),
        "seed": int(seed),
        "source_hw": list(source_prefix.shape[-2:]),
        "target_hw": list(authoritative_target_prefix.shape[-2:]),
        "prefix_t": prefix_t,
        "source_temporal": int(initial_video.shape[2]),
        "video_dtype": str(initial_video.dtype),
        "audio_dtype": str(initial_audio.dtype),
        "checkpoint_identity_verified": False,
        "rng_state_verified": False,
        "runtime_model_cache_state_verified": False,
        "causal_pair_qualified": False,
        "note": "Matching input hashes do not prove model weights, RNG history or stateful backend equivalence.",
    }


def compare_low_input_pairing_receipts(left: dict, right: dict) -> dict:
    """Fail closed on any known mismatched starting input; no causal verdict."""
    if left.get("policy") != POLICY or right.get("policy") != POLICY:
        raise ValueError("low pairing requires two version-matched low input receipts")
    missing = [key for key in PAIRED_FIELDS if key not in left or key not in right]
    if missing:
        raise ValueError(f"low input receipt missing required comparison fields: {missing}")
    mismatches = [key for key in PAIRED_FIELDS if left[key] != right[key]]
    source_changed = left.get("source_prefix_sha256") != right.get("source_prefix_sha256")
    modes_differ = left.get("source_policy") != right.get("source_policy")
    return {
        "policy": POLICY,
        "matched_known_low_inputs": not mismatches,
        "mismatched_fields": mismatches,
        "source_prefix_intervention_observed": source_changed,
        "different_source_policies": modes_differ,
        "input_pair_eligible": not mismatches and source_changed and modes_differ,
        "causal_pair_qualified": False,
        "unverified_controls": [
            "full model/checkpoint weight identity",
            "initial and intermediate sampler/backend RNG state",
            "Spectrum/VDN/cache and runtime execution history",
            "source-prefix selection as the sole modified model input",
        ],
    }
