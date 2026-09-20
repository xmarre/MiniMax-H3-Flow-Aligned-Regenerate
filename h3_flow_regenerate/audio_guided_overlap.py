"""Production guided overlap for exact-prefix MiniMax-H3 audio continuation.

The conservative Target Input continuation path keeps caller-owned protected
video/audio values exact at the framework boundary. During sampler lifetime,
this module can expose only the tail of a canonical carried audio prefix as a
short monotonic denoise-strength ramp. The video mask is never changed, and the
original exact mask remains authoritative for the final output restore.
"""

from __future__ import annotations

import math
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
    denoise_mask: torch.Tensor,
    latent_shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    *,
    window_ticks: int = 8,
) -> dict[str, Any]:
    """Measure final latent energy immediately across an exact audio boundary."""

    if type(window_ticks) is not int or window_ticks <= 0:
        raise ValueError("audio latent boundary window_ticks must be a positive integer")
    if latent_shapes is None or len(latent_shapes) != 2:
        raise ValueError("audio latent boundary diagnostic requires native packed H3 shapes")

    _video, audio = unpack_streams(packed_latent, latent_shapes)
    _video_mask, audio_mask = unpack_streams(denoise_mask, latent_shapes)
    if audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2):
        raise ValueError("audio latent boundary diagnostic requires [1,32,2,T] H3 audio latents")
    if tuple(audio_mask.shape) != tuple(audio.shape):
        raise ValueError("audio latent boundary mask does not match audio latent geometry")

    temporal_min = audio_mask.amin(dim=(0, 1, 2))
    temporal_max = audio_mask.amax(dim=(0, 1, 2))
    exact_zero = temporal_max <= 1e-8
    exact_one = temporal_min >= 1.0 - 1e-8
    prefix = 0
    temporal = int(audio.shape[-1])
    while prefix < temporal and bool(exact_zero[prefix].item()):
        prefix += 1

    report: dict[str, Any] = {
        "available": False,
        "audio_prefix_ticks": prefix,
        "window_ticks": 0,
        "protected_tail_rms": None,
        "generated_head_rms": None,
        "generated_over_protected_ratio": None,
        "generated_over_protected_db": None,
        "reason": "pending",
    }
    if prefix <= 0:
        report["reason"] = "no_exact_audio_prefix"
        return report
    if prefix >= temporal:
        report["reason"] = "no_generated_audio_suffix"
        return report
    if not bool(exact_one[prefix:].all().item()):
        report["reason"] = "noncanonical_partial_audio_mask"
        return report

    count = min(int(window_ticks), prefix, temporal - prefix)
    if count <= 0:
        report["reason"] = "empty_boundary_window"
        return report

    tail = audio[..., prefix - count : prefix].float()
    head = audio[..., prefix : prefix + count].float()
    tail_rms = float(torch.sqrt(torch.mean(tail.square())).item())
    head_rms = float(torch.sqrt(torch.mean(head.square())).item())
    eps = 1e-12
    ratio = (head_rms + eps) / (tail_rms + eps)
    report.update(
        {
            "available": True,
            "window_ticks": count,
            "protected_tail_rms": tail_rms,
            "generated_head_rms": head_rms,
            "generated_over_protected_ratio": ratio,
            "generated_over_protected_db": 20.0 * math.log10(ratio),
            "reason": "exact_audio_boundary",
        }
    )
    return report


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
