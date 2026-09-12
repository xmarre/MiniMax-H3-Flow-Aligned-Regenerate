from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .geometry import resize_spatial_5d, validate_video

HANDOFF_STATE_POLICY_LEGACY = "legacy_renoise"
HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1 = "velocity_bicubic_v1"
HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1 = "endpoint_residual_bicubic_v1"
HANDOFF_STATE_POLICIES = frozenset(
    {
        HANDOFF_STATE_POLICY_LEGACY,
        HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
        HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,
    }
)


def resolve_handoff_state_policy(policy: str | None) -> str:
    """Resolve serialized/omitted state policy without changing legacy graphs."""

    if policy is None:
        return HANDOFF_STATE_POLICY_LEGACY
    if not isinstance(policy, str):
        raise TypeError("handoff_state_policy must be a string or None")
    if policy not in HANDOFF_STATE_POLICIES:
        raise ValueError(f"unsupported handoff_state_policy {policy!r}")
    return policy


def _transport_compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    if tensors and all(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def _resize_displacement(
    displacement: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Apply the fixed framewise bicubic state lift from the design contract."""

    source_h, source_w = map(int, displacement.shape[-2:])
    if target_h < source_h or target_w < source_w:
        raise ValueError("state transport must not shrink either video axis")
    if target_h == source_h and target_w == source_w:
        return displacement.clone()

    if displacement.dtype != torch.float64:
        return resize_spatial_5d(displacement, target_h, target_w, mode="bicubic")

    b, c, t, h, w = displacement.shape
    work = displacement.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = F.interpolate(
        work,
        size=(int(target_h), int(target_w)),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    return out.reshape(b, t, c, int(target_h), int(target_w)).permute(0, 2, 1, 3, 4)


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def _lowpass_rms(value: torch.Tensor, *, kernel: int = 5) -> float:
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("state-transport diagnostic lowpass kernel must be positive and odd")
    b, c, t, h, w = value.shape
    work = value.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    pad = kernel // 2
    filtered = F.avg_pool2d(F.pad(work, (pad, pad, pad, pad), mode="replicate"), kernel_size=kernel, stride=1)
    return _rms(filtered)


def _spatial_mean_rms(value: torch.Tensor) -> float:
    return _rms(value.float().mean(dim=(-2, -1), keepdim=True))


def _validate_transport_inputs(
    source_state_video: torch.Tensor,
    source_clean_video: torch.Tensor,
    target_clean_video: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype, int, int, int, int]:
    source_state_video = validate_video(source_state_video)
    source_clean_video = validate_video(source_clean_video)
    target_clean_video = validate_video(target_clean_video)

    if source_state_video.shape != source_clean_video.shape:
        raise ValueError("source sampler state and accepted clean prediction must have identical video geometry")
    if source_state_video.device != source_clean_video.device:
        raise ValueError("source sampler state and accepted clean prediction must share a device")
    if source_state_video.shape[:3] != target_clean_video.shape[:3]:
        raise ValueError("state transport cannot change batch, channel, or temporal geometry")
    if target_clean_video.device != source_state_video.device:
        raise ValueError("source and target state-transport videos must share a device")

    temporal = int(source_state_video.shape[2])
    if type(prefix_t) is not int or not 0 < prefix_t < temporal:
        raise ValueError("state transport requires a nonempty exact prefix and generated suffix")

    source_h, source_w = map(int, source_state_video.shape[-2:])
    target_h, target_w = map(int, target_clean_video.shape[-2:])
    if target_h < source_h or target_w < source_w:
        raise ValueError("state transport target must not shrink either video axis")
    if target_h == source_h and target_w == source_w:
        raise ValueError("state transport comparison is only defined for changed spatial geometry")

    for name, tensor in (
        ("source sampler state", source_state_video),
        ("source accepted clean", source_clean_video),
        ("target corrected clean", target_clean_video),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or Inf")

    compute_dtype = _transport_compute_dtype(source_state_video, source_clean_video, target_clean_video)
    return (
        source_state_video,
        source_clean_video,
        target_clean_video,
        compute_dtype,
        source_h,
        source_w,
        target_h,
        target_w,
    )


def measure_state_transport_comparison_v1(
    source_state_video: torch.Tensor,
    source_clean_video: torch.Tensor,
    target_clean_video: torch.Tensor,
    *,
    prefix_t: int,
    sigma: float,
) -> dict[str, Any]:
    """Record bounded scalar evidence distinguishing tangent and endpoint lifts."""

    if not math.isfinite(float(sigma)) or not 0.0 < float(sigma) < 1.0:
        raise ValueError("state transport comparison requires finite sigma inside (0, 1)")
    (
        source_state_video,
        source_clean_video,
        target_clean_video,
        compute_dtype,
        _source_h,
        _source_w,
        target_h,
        target_w,
    ) = _validate_transport_inputs(
        source_state_video,
        source_clean_video,
        target_clean_video,
        prefix_t=prefix_t,
    )
    source_state = source_state_video[:, :, prefix_t:].to(dtype=compute_dtype)
    source_clean = source_clean_video[:, :, prefix_t:].to(dtype=compute_dtype)
    target_clean = target_clean_video[:, :, prefix_t:].to(dtype=compute_dtype)

    lifted_source_clean = _resize_displacement(source_clean, target_h, target_w).to(dtype=compute_dtype)
    clean_reanchor_delta = target_clean - lifted_source_clean
    displacement = source_state - source_clean
    lifted_displacement = _resize_displacement(displacement, target_h, target_w).to(dtype=compute_dtype)
    a = 1.0 - float(sigma)
    endpoint_residual = source_state - a * source_clean
    lifted_endpoint_residual = _resize_displacement(endpoint_residual, target_h, target_w).to(dtype=compute_dtype)

    clean_delta_rms = _rms(clean_reanchor_delta)
    return {
        "state_transport_comparison_version": 1,
        "state_transport_comparison_sigma": float(sigma),
        "state_transport_clean_reanchor_delta_rms": clean_delta_rms,
        "state_transport_clean_reanchor_delta_lowpass_rms": _lowpass_rms(clean_reanchor_delta),
        "state_transport_clean_reanchor_delta_spatial_mean_rms": _spatial_mean_rms(clean_reanchor_delta),
        "state_transport_velocity_endpoint_expected_delta_rms": float(sigma) * clean_delta_rms,
        "state_transport_source_displacement_lowpass_rms": _lowpass_rms(displacement),
        "state_transport_source_displacement_spatial_mean_rms": _spatial_mean_rms(displacement),
        "state_transport_lifted_displacement_lowpass_rms": _lowpass_rms(lifted_displacement),
        "state_transport_lifted_displacement_spatial_mean_rms": _spatial_mean_rms(lifted_displacement),
        "state_transport_source_endpoint_residual_rms": _rms(endpoint_residual),
        "state_transport_source_endpoint_residual_lowpass_rms": _lowpass_rms(endpoint_residual),
        "state_transport_lifted_endpoint_residual_rms": _rms(lifted_endpoint_residual),
        "state_transport_lifted_endpoint_residual_lowpass_rms": _lowpass_rms(lifted_endpoint_residual),
    }


def transport_velocity_bicubic_v1(
    source_state_video: torch.Tensor,
    source_clean_video: torch.Tensor,
    target_clean_video: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Re-anchor the generated suffix while preserving source flow displacement."""

    (
        source_state_video,
        source_clean_video,
        target_clean_video,
        compute_dtype,
        source_h,
        source_w,
        target_h,
        target_w,
    ) = _validate_transport_inputs(
        source_state_video,
        source_clean_video,
        target_clean_video,
        prefix_t=prefix_t,
    )
    source_suffix_state = source_state_video[:, :, prefix_t:].to(dtype=compute_dtype)
    source_suffix_clean = source_clean_video[:, :, prefix_t:].to(dtype=compute_dtype)
    target_suffix_clean = target_clean_video[:, :, prefix_t:].to(dtype=compute_dtype)

    displacement = source_suffix_state - source_suffix_clean
    source_roundtrip_error = source_suffix_clean + displacement - source_suffix_state
    lifted_displacement = _resize_displacement(displacement, target_h, target_w)
    if lifted_displacement.dtype != compute_dtype:
        lifted_displacement = lifted_displacement.to(dtype=compute_dtype)
    if not bool(torch.isfinite(lifted_displacement).all().item()):
        raise RuntimeError("lifted state displacement contains NaN or Inf")

    target_video = target_clean_video.clone()
    reconstructed_suffix = target_suffix_clean + lifted_displacement
    if not bool(torch.isfinite(reconstructed_suffix).all().item()):
        raise RuntimeError("reconstructed target sampler state contains NaN or Inf")
    target_video[:, :, prefix_t:] = reconstructed_suffix.to(dtype=target_video.dtype)

    realized_suffix = target_video[:, :, prefix_t:].to(dtype=compute_dtype)
    closure_error = realized_suffix - target_suffix_clean - lifted_displacement
    source_rms = _rms(displacement)
    lifted_rms = _rms(lifted_displacement)
    metrics = {
        "handoff_state_policy": HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
        "state_transport_applied": True,
        "state_transport_residual_kind": "flow_displacement",
        "state_transport_clean_coefficient": 1.0,
        "state_transport_prefix_t": prefix_t,
        "state_transport_source_hw": (source_h, source_w),
        "state_transport_target_hw": (target_h, target_w),
        "state_transport_compute_dtype": str(compute_dtype),
        "state_transport_source_displacement_rms": source_rms,
        "state_transport_lifted_displacement_rms": lifted_rms,
        "state_transport_lifted_over_source_rms_ratio": lifted_rms / max(source_rms, 1e-30),
        "state_transport_source_roundtrip_max_abs": float(source_roundtrip_error.abs().max().item()),
        "state_transport_source_roundtrip_rms": _rms(source_roundtrip_error),
        "state_transport_target_closure_max_abs": float(closure_error.abs().max().item()),
        "state_transport_target_closure_rms": _rms(closure_error),
        "state_transport_added_rng": False,
        "state_transport_temporal_mixing": False,
        "state_transport_amplitude_normalization": False,
    }
    return target_video, metrics


def transport_endpoint_residual_bicubic_v1(
    source_state_video: torch.Tensor,
    source_clean_video: torch.Tensor,
    target_clean_video: torch.Tensor,
    *,
    prefix_t: int,
    sigma: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transport the effective endpoint field instead of the source tangent."""

    if not math.isfinite(float(sigma)) or not 0.0 < float(sigma) < 1.0:
        raise ValueError("endpoint_residual_bicubic_v1 requires finite sigma inside (0, 1)")
    (
        source_state_video,
        source_clean_video,
        target_clean_video,
        compute_dtype,
        source_h,
        source_w,
        target_h,
        target_w,
    ) = _validate_transport_inputs(
        source_state_video,
        source_clean_video,
        target_clean_video,
        prefix_t=prefix_t,
    )
    source_suffix_state = source_state_video[:, :, prefix_t:].to(dtype=compute_dtype)
    source_suffix_clean = source_clean_video[:, :, prefix_t:].to(dtype=compute_dtype)
    target_suffix_clean = target_clean_video[:, :, prefix_t:].to(dtype=compute_dtype)
    a = 1.0 - float(sigma)

    endpoint_residual = source_suffix_state - a * source_suffix_clean
    source_roundtrip_error = a * source_suffix_clean + endpoint_residual - source_suffix_state
    lifted_endpoint_residual = _resize_displacement(endpoint_residual, target_h, target_w)
    if lifted_endpoint_residual.dtype != compute_dtype:
        lifted_endpoint_residual = lifted_endpoint_residual.to(dtype=compute_dtype)
    if not bool(torch.isfinite(lifted_endpoint_residual).all().item()):
        raise RuntimeError("lifted endpoint residual contains NaN or Inf")

    target_video = target_clean_video.clone()
    reconstructed_suffix = a * target_suffix_clean + lifted_endpoint_residual
    if not bool(torch.isfinite(reconstructed_suffix).all().item()):
        raise RuntimeError("reconstructed endpoint target sampler state contains NaN or Inf")
    target_video[:, :, prefix_t:] = reconstructed_suffix.to(dtype=target_video.dtype)

    realized_suffix = target_video[:, :, prefix_t:].to(dtype=compute_dtype)
    closure_error = realized_suffix - a * target_suffix_clean - lifted_endpoint_residual
    source_rms = _rms(endpoint_residual)
    lifted_rms = _rms(lifted_endpoint_residual)
    metrics = {
        "handoff_state_policy": HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,
        "state_transport_applied": True,
        "state_transport_residual_kind": "effective_endpoint",
        "state_transport_clean_coefficient": a,
        "state_transport_prefix_t": prefix_t,
        "state_transport_source_hw": (source_h, source_w),
        "state_transport_target_hw": (target_h, target_w),
        "state_transport_compute_dtype": str(compute_dtype),
        "state_transport_source_endpoint_residual_rms": source_rms,
        "state_transport_lifted_endpoint_residual_rms": lifted_rms,
        "state_transport_lifted_over_source_rms_ratio": lifted_rms / max(source_rms, 1e-30),
        "state_transport_source_roundtrip_max_abs": float(source_roundtrip_error.abs().max().item()),
        "state_transport_source_roundtrip_rms": _rms(source_roundtrip_error),
        "state_transport_target_closure_max_abs": float(closure_error.abs().max().item()),
        "state_transport_target_closure_rms": _rms(closure_error),
        "state_transport_added_rng": False,
        "state_transport_temporal_mixing": False,
        "state_transport_amplitude_normalization": False,
    }
    return target_video, metrics
