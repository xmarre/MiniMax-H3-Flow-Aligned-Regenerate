"""Production guided overlap for exact-prefix MiniMax-H3 audio continuation.

The conservative Target Input continuation path keeps caller-owned protected
video/audio values exact at the framework boundary. During sampler lifetime,
this module can expose only the tail of a canonical carried audio prefix as a
short monotonic denoise-strength ramp. The video mask is never changed, and the
original exact mask remains authoritative for the final output restore.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from .geometry import unpack_streams

AUDIO_GUIDED_OVERLAP_ENV = "H3_FLOW_AUDIO_GUIDED_OVERLAP_TICKS"
DEFAULT_AUDIO_GUIDED_OVERLAP_TICKS = 4
MAX_AUDIO_GUIDED_OVERLAP_TICKS = 16
_MASK_QUANTIZATION_LEVELS = 256.0


def validate_audio_guided_overlap_ticks(value: int, *, source: str = "audio guided overlap") -> int:
    """Validate an explicit overlap width without consulting process environment."""

    if type(value) is not int or not 0 <= value <= MAX_AUDIO_GUIDED_OVERLAP_TICKS:
        raise ValueError(f"{source} must be an integer in [0, {MAX_AUDIO_GUIDED_OVERLAP_TICKS}], got {value!r}")
    return int(value)


def configured_audio_guided_overlap_ticks() -> int:
    """Return the production overlap width in 40-Hz audio latent ticks."""

    raw_value = os.environ.get(AUDIO_GUIDED_OVERLAP_ENV)
    if raw_value is None or not str(raw_value).strip():
        return DEFAULT_AUDIO_GUIDED_OVERLAP_TICKS
    try:
        ticks = int(str(raw_value).strip())
    except ValueError as exc:
        message = f"{AUDIO_GUIDED_OVERLAP_ENV} must be an integer in [0, {MAX_AUDIO_GUIDED_OVERLAP_TICKS}]"
        raise ValueError(message) from exc
    return validate_audio_guided_overlap_ticks(ticks, source=AUDIO_GUIDED_OVERLAP_ENV)


def measure_audio_latent_boundary(
    packed_latent: torch.Tensor,
    latent_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None,
    exact_denoise_mask: torch.Tensor | None,
    *,
    windows: tuple[int, ...] = (4, 20),
) -> dict[str, Any]:
    """Measure latent amplitude immediately around the original exact audio boundary.

    This is diagnostics-only: it never modifies the latent or mask. Windows are
    expressed in native 40-Hz MiniMax-H3 audio latent ticks. The boundary is
    derived from the caller-owned exact mask rather than any sampler-lifetime
    guided-overlap mask, so 4-tick and 0-tick A/B runs are measured at the same
    physical carried-prefix -> generated-suffix boundary.
    """

    report: dict[str, Any] = {
        "available": False,
        "reason": "pending",
        "audio_prefix_ticks": 0,
        "audio_total_ticks": 0,
        "windows": {},
    }
    if exact_denoise_mask is None:
        report["reason"] = "no_denoise_mask"
        return report
    if latent_shapes is None or len(latent_shapes) != 2:
        raise ValueError("audio latent boundary diagnostics require native packed H3 AV shapes")
    if packed_latent.ndim != 3 or exact_denoise_mask.ndim != 3:
        raise ValueError("audio latent boundary diagnostics require packed BxCxN tensors")

    latent_audio = unpack_streams(packed_latent, latent_shapes)[1]
    mask_audio = unpack_streams(exact_denoise_mask, latent_shapes)[1]
    if latent_audio.ndim != 4 or int(latent_audio.shape[2]) != 2:
        raise ValueError("audio latent boundary diagnostics require native BxCx2xT audio latents")
    if mask_audio.shape != latent_audio.shape:
        raise ValueError("audio latent boundary diagnostics require mask/audio geometry parity")

    temporal_min = mask_audio.amin(dim=(0, 1, 2))
    temporal_max = mask_audio.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8
    temporal = int(latent_audio.shape[-1])
    report["audio_total_ticks"] = temporal

    if bool(exact_one.all().item()):
        report["reason"] = "no_exact_audio_prefix"
        return report
    if bool(exact_zero.all().item()):
        report["reason"] = "no_generated_audio_suffix"
        report["audio_prefix_ticks"] = temporal
        return report

    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    report["audio_prefix_ticks"] = prefix
    if prefix <= 0 or not bool(exact_one[prefix:].all().item()):
        raise ValueError(
            "audio latent boundary diagnostics require a contiguous exact audio prefix "
            "followed by a fully generated suffix"
        )

    audio = latent_audio.detach().to(dtype=torch.float32)
    for requested in windows:
        width = int(requested)
        if width <= 0:
            raise ValueError("audio latent boundary diagnostic windows must be positive")
        usable = min(width, prefix, temporal - prefix)
        if usable <= 0:
            continue
        pre = audio[..., prefix - usable : prefix]
        post = audio[..., prefix : prefix + usable]
        pre_rms = float(pre.square().mean().sqrt().item())
        post_rms = float(post.square().mean().sqrt().item())
        ratio = post_rms / max(pre_rms, 1e-12)
        report["windows"][str(width)] = {
            "requested_ticks": width,
            "used_ticks": usable,
            "duration_ms_at_40hz": usable * 25.0,
            "pre_rms": pre_rms,
            "post_rms": post_rms,
            "post_over_pre_rms_ratio": ratio,
            "post_over_pre_db": 20.0 * math.log10(max(ratio, 1e-12)),
            "pre_mean_abs": float(pre.abs().mean().item()),
            "post_mean_abs": float(post.abs().mean().item()),
        }

    report["available"] = bool(report["windows"])
    report["reason"] = "measured" if report["available"] else "insufficient_boundary_context"
    return report


def compare_audio_latent_stages(
    reference_audio: torch.Tensor,
    candidate_audio: torch.Tensor,
    exact_audio_mask: torch.Tensor,
    *,
    windows: tuple[int, ...] = (4, 20),
) -> dict[str, Any]:
    """Compare two clean-domain H3 audio latents at the exact continuation boundary.

    Inputs are unpacked native BxCx2xT audio tensors. The report is diagnostics-only
    and intentionally bounded: aggregate and per-channel RMS changes plus temporal
    first-difference energy for the first generated ticks. It never mutates either
    stage.
    """

    if reference_audio.shape != candidate_audio.shape:
        raise ValueError("audio stage comparison requires identical reference/candidate geometry")
    if reference_audio.ndim != 4 or int(reference_audio.shape[2]) != 2:
        raise ValueError("audio stage comparison requires native BxCx2xT audio latents")
    if exact_audio_mask.shape != reference_audio.shape:
        raise ValueError("audio stage comparison requires mask/audio geometry parity")
    if not bool(torch.isfinite(reference_audio).all().item()) or not bool(torch.isfinite(candidate_audio).all().item()):
        raise ValueError("audio stage comparison requires finite latents")

    temporal_min = exact_audio_mask.amin(dim=(0, 1, 2))
    temporal_max = exact_audio_mask.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8
    temporal = int(reference_audio.shape[-1])
    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    if prefix <= 0 or prefix >= temporal or not bool(exact_one[prefix:].all().item()):
        raise ValueError("audio stage comparison requires a contiguous exact prefix followed by generated suffix")

    reference = reference_audio.detach().to(dtype=torch.float32)
    candidate = candidate_audio.detach().to(device=reference.device, dtype=torch.float32)
    prefix_delta = candidate[..., :prefix] - reference[..., :prefix]
    report: dict[str, Any] = {
        "available": True,
        "audio_prefix_ticks": prefix,
        "audio_total_ticks": temporal,
        "exact_prefix_max_abs_delta": float(prefix_delta.abs().max().item()),
        "exact_prefix_rms_delta": float(prefix_delta.square().mean().sqrt().item()),
        "windows": {},
    }

    for requested in windows:
        width = int(requested)
        if width <= 0:
            raise ValueError("audio stage comparison windows must be positive")
        usable = min(width, temporal - prefix)
        if usable <= 0:
            continue
        ref = reference[..., prefix : prefix + usable]
        cand = candidate[..., prefix : prefix + usable]
        delta = cand - ref
        ref_rms = float(ref.square().mean().sqrt().item())
        cand_rms = float(cand.square().mean().sqrt().item())
        delta_rms = float(delta.square().mean().sqrt().item())

        # BxCx2xT -> C summaries, bounded to the native H3 channel count.
        channel_dims = (0, 2, 3)
        ref_ch = ref.square().mean(dim=channel_dims).sqrt()
        cand_ch = cand.square().mean(dim=channel_dims).sqrt()
        channel_ratio = cand_ch / ref_ch.clamp_min(1e-12)
        channel_db = 20.0 * torch.log10(channel_ratio.clamp_min(1e-12))

        flat_ref = ref.reshape(-1)
        flat_cand = cand.reshape(-1)
        denom = float(flat_ref.norm().item() * flat_cand.norm().item())
        cosine = float(torch.dot(flat_ref, flat_cand).item() / denom) if denom > 1e-20 else None

        if usable > 1:
            ref_diff_rms = float((ref[..., 1:] - ref[..., :-1]).square().mean().sqrt().item())
            cand_diff_rms = float((cand[..., 1:] - cand[..., :-1]).square().mean().sqrt().item())
        else:
            ref_diff_rms = None
            cand_diff_rms = None

        report["windows"][str(width)] = {
            "requested_ticks": width,
            "used_ticks": usable,
            "duration_ms_at_40hz": usable * 25.0,
            "reference_rms": ref_rms,
            "candidate_rms": cand_rms,
            "candidate_over_reference_rms_db": 20.0 * math.log10(max(cand_rms / max(ref_rms, 1e-12), 1e-12)),
            "delta_rms": delta_rms,
            "delta_over_reference_rms": delta_rms / max(ref_rms, 1e-12),
            "cosine_similarity": cosine,
            "reference_first_difference_rms": ref_diff_rms,
            "candidate_first_difference_rms": cand_diff_rms,
            "per_channel_candidate_over_reference_db": [
                float(value) for value in channel_db.detach().to(device="cpu", dtype=torch.float32).tolist()
            ],
        }

    report["available"] = bool(report["windows"])
    return report



def apply_audio_exact_restore_suffix_bridge(
    result: torch.Tensor,
    latent_image: torch.Tensor,
    exact_denoise_mask: torch.Tensor | None,
    latent_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None,
    *,
    enabled: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Preserve the sampler's first audio-suffix transition across exact restore.

    Sampler-owned guided overlap deliberately makes the tail of the carried
    audio prefix fractional during sampling. The final framework boundary then
    restores the caller-owned exact prefix. Without a matching suffix update,
    the first generated audio tick keeps the relation it learned against the
    temporary sampler prefix rather than the restored exact prefix.

    This bridge transfers only that one-step relation:

        corrected_suffix0 - exact_prefix_last
        == sampled_suffix0 - sampled_prefix_last

    It changes one generated audio tick in place and leaves video, the complete
    protected audio prefix, and every later generated audio tick untouched.
    There is no model call, sampler lifetime, VAE call, resampling, or decoded
    audio crossfade.
    """

    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "pending",
        "audio_prefix_ticks": 0,
        "corrected_ticks": 0,
        "delta_rms": 0.0,
        "delta_max_abs": 0.0,
        "relation_error_rms": 0.0,
        "protected_prefix_modified": False,
        "later_suffix_modified": False,
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_vae_calls": 0,
    }
    if not enabled:
        return result, report
    if exact_denoise_mask is None:
        report["reason"] = "no_exact_denoise_mask"
        return result, report
    if latent_shapes is None or len(latent_shapes) != 2:
        raise ValueError("audio exact-restore suffix bridge requires native packed H3 AV shapes")
    if result.shape != latent_image.shape or result.shape != exact_denoise_mask.shape:
        raise ValueError("audio exact-restore suffix bridge requires matching packed sampler tensors")

    result_audio = unpack_streams(result, latent_shapes)[1]
    reference_audio = unpack_streams(latent_image, latent_shapes)[1]
    mask_audio = unpack_streams(exact_denoise_mask, latent_shapes)[1]
    if result_audio.ndim != 4 or int(result_audio.shape[2]) != 2:
        raise ValueError("audio exact-restore suffix bridge requires native BxCx2xT audio latents")
    if tuple(reference_audio.shape) != tuple(result_audio.shape) or tuple(mask_audio.shape) != tuple(result_audio.shape):
        raise ValueError("audio exact-restore suffix bridge audio geometry drifted")

    temporal_min = mask_audio.amin(dim=(0, 1, 2))
    temporal_max = mask_audio.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8
    temporal = int(result_audio.shape[-1])
    if bool(exact_one.all().item()):
        report["reason"] = "no_exact_audio_prefix"
        return result, report
    if bool(exact_zero.all().item()):
        report.update(reason="no_generated_audio_suffix", audio_prefix_ticks=temporal)
        return result, report

    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    report["audio_prefix_ticks"] = prefix
    if prefix <= 0 or prefix >= temporal or not bool(exact_one[prefix:].all().item()):
        raise ValueError(
            "audio exact-restore suffix bridge requires a contiguous exact audio prefix "
            "followed by a fully generated suffix"
        )

    sampled_last = result_audio[..., prefix - 1].detach().to(torch.float32)
    exact_last = reference_audio[..., prefix - 1].detach().to(torch.float32)
    suffix_before = result_audio[..., prefix].detach().to(torch.float32)
    delta = exact_last - sampled_last
    delta_rms = float(delta.square().mean().sqrt().item())
    delta_max = float(delta.abs().max().item())
    report["delta_rms"] = delta_rms
    report["delta_max_abs"] = delta_max
    if delta_max == 0.0:
        report["reason"] = "sampler_prefix_already_exact"
        return result, report

    sampled_relation = suffix_before - sampled_last
    result_audio[..., prefix].copy_((suffix_before + delta).to(dtype=result_audio.dtype))
    corrected_relation = result_audio[..., prefix].detach().to(torch.float32) - exact_last
    relation_error = corrected_relation - sampled_relation
    report.update(
        {
            "applied": True,
            "reason": "exact_restore_transition_transfer",
            "corrected_ticks": 1,
            "relation_error_rms": float(relation_error.square().mean().sqrt().item()),
        }
    )
    return result, report


def apply_audio_guided_overlap_mask(
    denoise_mask: torch.Tensor | None,
    latent_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...] | None,
    *,
    ticks: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Feather only the tail of a canonical exact audio prefix.

    The input is Comfy's packed, broadcast-to-latent denoise mask. Native
    continuation uses a contiguous 0-valued carried prefix followed by a
    1-valued generated suffix. For ``ticks=N`` the last N protected audio
    ticks become a monotonic ``1/(N+1) .. N/(N+1)`` ramp for the sampler
    lifetime. Values are rounded upward onto Core MiniMax-H3's 1/256 mask
    grid, matching ``MiniMaxH3._token_grid_masks``.

    Video is byte-for-byte untouched. The original mask remains caller-owned
    and is required for final exact-prefix canonicalization.

    All-one audio (ordinary first chunk), all-protected audio, and exact
    prefixes too short to leave at least one fully protected tick are expected
    no-ops. Any other partially protected but non-canonical audio layout is
    rejected rather than silently applying a heuristic to unknown semantics.
    """

    ticks = int(ticks)
    report: dict[str, Any] = {
        "requested_ticks": ticks,
        "applied": False,
        "reason": "disabled" if ticks <= 0 else "pending",
        "audio_prefix_ticks": 0,
        "ramp_values": [],
        "mask_grid": "ceil_1_over_256",
        "original_exact_output_restore": True,
    }
    if ticks <= 0:
        return denoise_mask, report
    if denoise_mask is None:
        report["reason"] = "no_denoise_mask"
        return denoise_mask, report
    if latent_shapes is None or len(latent_shapes) != 2:
        raise ValueError("audio guided overlap requires native packed H3 video/audio shapes")
    if denoise_mask.ndim != 3 or denoise_mask.shape[1] != 1:
        raise ValueError("audio guided overlap requires packed Bx1xN denoise mask")

    runtime_mask = denoise_mask.clone()
    streams = unpack_streams(runtime_mask, latent_shapes)
    audio_mask = streams[1]
    if audio_mask.ndim != 4 or int(audio_mask.shape[2]) != 2:
        raise ValueError("audio guided overlap requires native BxCx2xT audio mask")

    temporal_min = audio_mask.amin(dim=(0, 1, 2))
    temporal_max = audio_mask.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8

    if bool(exact_one.all().item()):
        report["reason"] = "no_exact_audio_prefix"
        return denoise_mask, report

    temporal = int(audio_mask.shape[-1])
    if bool(exact_zero.all().item()):
        report.update(
            {
                "reason": "no_generated_audio_suffix",
                "audio_prefix_ticks": temporal,
            }
        )
        return denoise_mask, report

    prefix = 0
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1
    report["audio_prefix_ticks"] = prefix
    if prefix <= 0:
        raise ValueError("audio guided overlap found protected audio without a leading exact prefix")
    if not bool(exact_one[prefix:].all().item()):
        raise ValueError(
            "audio guided overlap requires a contiguous exact audio prefix followed by a fully generated suffix"
        )
    if ticks >= prefix:
        report["reason"] = "exact_audio_prefix_too_short"
        return denoise_mask, report

    raw_ramp = torch.arange(1, ticks + 1, device=audio_mask.device, dtype=torch.float32) / float(ticks + 1)
    ramp = torch.ceil(raw_ramp * _MASK_QUANTIZATION_LEVELS) / _MASK_QUANTIZATION_LEVELS
    ramp = ramp.to(dtype=audio_mask.dtype)
    audio_mask[..., prefix - ticks : prefix] = ramp.view(1, 1, 1, -1)
    report.update(
        {
            "applied": True,
            "reason": "exact_audio_prefix_tail",
            "audio_prefix_ticks": prefix,
            "ramp_values": [float(value) for value in ramp.detach().to(device="cpu", dtype=torch.float32).tolist()],
            "ramp_start_tick": prefix - ticks,
            "ramp_stop_tick": prefix,
        }
    )
    return runtime_mask, report
