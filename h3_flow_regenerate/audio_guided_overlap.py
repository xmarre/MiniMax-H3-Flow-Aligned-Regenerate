"""Production guided overlap for exact-prefix MiniMax-H3 audio continuation.

The conservative Target Input continuation path keeps caller-owned protected
video/audio values exact at the framework boundary. During sampler lifetime,
this module can expose only the tail of a canonical carried audio prefix as a
short monotonic denoise-strength ramp. The video mask is never changed, and the
original exact mask remains authoritative for the final output restore.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .geometry import unpack_streams

AUDIO_GUIDED_OVERLAP_ENV = "H3_FLOW_AUDIO_GUIDED_OVERLAP_TICKS"
DEFAULT_AUDIO_GUIDED_OVERLAP_TICKS = 4
MAX_AUDIO_GUIDED_OVERLAP_TICKS = 16
_MASK_QUANTIZATION_LEVELS = 256.0


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
    if not 0 <= ticks <= MAX_AUDIO_GUIDED_OVERLAP_TICKS:
        raise ValueError(f"{AUDIO_GUIDED_OVERLAP_ENV} must be in [0, {MAX_AUDIO_GUIDED_OVERLAP_TICKS}], got {ticks}")
    return ticks


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

    All-one audio (ordinary first chunk) is an expected no-op. Any other
    partially protected but non-canonical audio layout is rejected rather than
    silently applying a heuristic to unknown semantics.
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

    prefix = 0
    temporal = int(audio_mask.shape[-1])
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
        raise ValueError(f"audio guided overlap width {ticks} must be smaller than exact audio prefix {prefix}")

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
