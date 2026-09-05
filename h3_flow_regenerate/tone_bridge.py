"""Experimental suffix-only DC bridge for mixed-grid exact-prefix handoffs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _validate_video_pair(upscaled_clean_video: torch.Tensor, exact_prefix: torch.Tensor) -> int:
    if not isinstance(upscaled_clean_video, torch.Tensor) or not isinstance(exact_prefix, torch.Tensor):
        raise TypeError("suffix DC bridge expects torch.Tensor inputs")
    if upscaled_clean_video.ndim != 5 or exact_prefix.ndim != 5:
        raise ValueError("suffix DC bridge expects BxCxTxHxW tensors")
    if not upscaled_clean_video.is_floating_point() or not exact_prefix.is_floating_point():
        raise TypeError("suffix DC bridge expects floating-point tensors")
    if upscaled_clean_video.shape[:2] != exact_prefix.shape[:2]:
        raise ValueError("suffix DC bridge batch/channel geometry differs")
    if upscaled_clean_video.shape[-2:] != exact_prefix.shape[-2:]:
        raise ValueError("suffix DC bridge spatial geometry differs")
    prefix_t = int(exact_prefix.shape[2])
    if prefix_t < 1 or prefix_t >= int(upscaled_clean_video.shape[2]):
        raise ValueError("suffix DC bridge requires a non-empty prefix shorter than the video")
    if not bool(torch.isfinite(upscaled_clean_video).all().item()):
        raise RuntimeError("suffix DC bridge learned clean video contains NaN or Inf values")
    if not bool(torch.isfinite(exact_prefix).all().item()):
        raise RuntimeError("suffix DC bridge exact prefix contains NaN or Inf values")
    return prefix_t


def disabled_suffix_dc_bridge_metrics(*, prefix_t: int) -> dict[str, float | int | bool]:
    prefix_t = int(prefix_t)
    if prefix_t < 1:
        raise ValueError("suffix DC bridge prefix length must be positive")
    return {
        "suffix_dc_bridge_version": 1,
        "suffix_dc_bridge_enabled": False,
        "suffix_dc_bridge_prefix_t": prefix_t,
        "suffix_dc_bridge_corrected_tokens": 0,
        "suffix_dc_bridge_first_weight": 0.0,
        "suffix_dc_bridge_last_weight": 0.0,
        "suffix_dc_bridge_delta_rms": 0.0,
        "suffix_dc_bridge_delta_abs_mean": 0.0,
        "suffix_dc_bridge_delta_abs_max": 0.0,
    }


def apply_suffix_dc_bridge(
    upscaled_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    weights: Sequence[float] = (1.0,),
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Translate the learned native boundary offset onto an authoritative prefix.

    The authoritative prefix is never modified. For each selected suffix token,
    only a per-batch/per-channel spatially constant offset is added. With weight
    1.0 on suffix token 0, its channel-wise spatial-mean offset from the exact
    prefix equals the learned upscaler's original native boundary offset.
    """

    prefix_t = _validate_video_pair(upscaled_clean_video, exact_prefix)
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)) or not weights:
        raise ValueError("suffix DC bridge weights must be a non-empty sequence")
    normalized_weights = tuple(float(weight) for weight in weights)
    if any(not math.isfinite(weight) or weight < 0.0 or weight > 1.0 for weight in normalized_weights):
        raise ValueError("suffix DC bridge weights must be finite values inside [0, 1]")

    suffix_t = int(upscaled_clean_video.shape[2]) - prefix_t
    corrected_tokens = min(len(normalized_weights), suffix_t)
    exact_last = exact_prefix[:, :, -1].to(device=upscaled_clean_video.device, dtype=torch.float32)
    learned_last = upscaled_clean_video[:, :, prefix_t - 1].float()
    delta = exact_last.mean(dim=(-2, -1), keepdim=True) - learned_last.mean(dim=(-2, -1), keepdim=True)
    if not bool(torch.isfinite(delta).all().item()):
        raise RuntimeError("suffix DC bridge produced a non-finite channel offset")

    corrected = upscaled_clean_video.clone()
    for offset in range(corrected_tokens):
        weight = normalized_weights[offset]
        if weight == 0.0:
            continue
        corrected[:, :, prefix_t + offset].add_((delta * weight).to(dtype=corrected.dtype))
    if not bool(torch.isfinite(corrected).all().item()):
        raise RuntimeError("suffix DC bridge produced NaN or Inf values")

    summary = (
        torch.stack(
            (
                delta.square().mean().sqrt(),
                delta.abs().mean(),
                delta.abs().max(),
            )
        )
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    delta_rms, delta_abs_mean, delta_abs_max = map(float, summary.tolist())
    return corrected, {
        "suffix_dc_bridge_version": 1,
        "suffix_dc_bridge_enabled": True,
        "suffix_dc_bridge_prefix_t": prefix_t,
        "suffix_dc_bridge_corrected_tokens": corrected_tokens,
        "suffix_dc_bridge_first_weight": normalized_weights[0] if corrected_tokens else 0.0,
        "suffix_dc_bridge_last_weight": normalized_weights[corrected_tokens - 1] if corrected_tokens else 0.0,
        "suffix_dc_bridge_delta_rms": delta_rms,
        "suffix_dc_bridge_delta_abs_mean": delta_abs_mean,
        "suffix_dc_bridge_delta_abs_max": delta_abs_max,
    }


def map_clean_bridge_to_conditional_state(
    state: torch.Tensor,
    clean_before: torch.Tensor,
    clean_after: torch.Tensor,
    *,
    sigma: float,
    prefix_t: int,
    corrected_tokens: int,
) -> torch.Tensor:
    """Map a clean-space suffix bridge onto an already re-noised state exactly.

    ``conditional_renoise_target`` is affine:
    ``x_sigma = (1-sigma) * x0 + sigma * noise``. Therefore adding
    ``(1-sigma) * (x0_after - x0_before)`` to the same suffix rows is
    algebraically identical to applying the clean bridge immediately before
    conditional re-noising, while leaving the deterministic noise untouched.
    """

    if state.ndim != 5 or clean_before.ndim != 5 or clean_after.ndim != 5:
        raise ValueError("conditional bridge mapping expects BxCxTxHxW tensors")
    if state.shape != clean_before.shape or state.shape != clean_after.shape:
        raise ValueError("conditional bridge mapping tensor geometry differs")
    if not state.is_floating_point() or not clean_before.is_floating_point() or not clean_after.is_floating_point():
        raise TypeError("conditional bridge mapping expects floating-point tensors")
    sigma = float(sigma)
    if not math.isfinite(sigma) or not 0.0 <= sigma < 1.0:
        raise ValueError("conditional bridge mapping requires finite sigma in [0, 1)")
    prefix_t = int(prefix_t)
    corrected_tokens = int(corrected_tokens)
    temporal = int(state.shape[2])
    if prefix_t < 1 or prefix_t >= temporal:
        raise ValueError("conditional bridge mapping requires a valid prefix boundary")
    if corrected_tokens < 1 or prefix_t + corrected_tokens > temporal:
        raise ValueError("conditional bridge mapping corrected-token range is invalid")
    if not torch.equal(clean_before[:, :, :prefix_t], clean_after[:, :, :prefix_t]):
        raise RuntimeError("suffix DC bridge attempted to alter the authoritative prefix")
    if not torch.equal(
        clean_before[:, :, prefix_t + corrected_tokens :], clean_after[:, :, prefix_t + corrected_tokens :]
    ):
        raise RuntimeError("suffix DC bridge attempted to alter later suffix tokens")

    result = state.clone()
    clean_delta = (
        clean_after[:, :, prefix_t : prefix_t + corrected_tokens].float()
        - clean_before[:, :, prefix_t : prefix_t + corrected_tokens].float()
    )
    result[:, :, prefix_t : prefix_t + corrected_tokens].add_(
        ((1.0 - sigma) * clean_delta).to(device=result.device, dtype=result.dtype)
    )
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("conditional bridge mapping produced NaN or Inf values")
    return result
