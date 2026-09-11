from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .geometry import resize_spatial_5d, validate_video

HANDOFF_STATE_POLICY_LEGACY = "legacy_renoise"
HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1 = "velocity_bicubic_v1"
HANDOFF_STATE_POLICIES = frozenset(
    {
        HANDOFF_STATE_POLICY_LEGACY,
        HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
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
    # Keep algebraic FP64 fixtures genuinely FP64. Production half/BF16/FP32
    # inputs use FP32 for subtraction/interpolation/addition, then cast back.
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
        # geometry.resize_spatial_5d already fixes align_corners=False and sets
        # antialias=False for non-shrinking geometry.
        return resize_spatial_5d(displacement, target_h, target_w, mode="bicubic")

    # resize_spatial_5d deliberately computes production dtypes in FP32. Keep a
    # narrow FP64 oracle path here rather than changing geometry semantics.
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


def transport_velocity_bicubic_v1(
    source_state_video: torch.Tensor,
    source_clean_video: torch.Tensor,
    target_clean_video: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Re-anchor the generated suffix while preserving source flow displacement.

    For generated suffix frames this implements exactly

        X_target = C_target + U(X_source - C_source)

    where U is a fixed framewise bicubic spatial lift. Protected target-prefix
    rows are copied from ``target_clean_video`` only as staging values; caller
    ownership is restored by the runtime before high-stage sampling.
    """

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
        raise ValueError("velocity_bicubic_v1 is only defined for a changed spatial geometry")

    for name, tensor in (
        ("source sampler state", source_state_video),
        ("source accepted clean", source_clean_video),
        ("target corrected clean", target_clean_video),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or Inf")

    compute_dtype = _transport_compute_dtype(source_state_video, source_clean_video, target_clean_video)
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
    source_rms = float(displacement.square().mean().sqrt().item())
    lifted_rms = float(lifted_displacement.square().mean().sqrt().item())
    metrics = {
        "handoff_state_policy": HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
        "state_transport_applied": True,
        "state_transport_prefix_t": prefix_t,
        "state_transport_source_hw": (source_h, source_w),
        "state_transport_target_hw": (target_h, target_w),
        "state_transport_compute_dtype": str(compute_dtype),
        "state_transport_source_displacement_rms": source_rms,
        "state_transport_lifted_displacement_rms": lifted_rms,
        "state_transport_lifted_over_source_rms_ratio": lifted_rms / max(source_rms, 1e-30),
        "state_transport_source_roundtrip_max_abs": float(source_roundtrip_error.abs().max().item()),
        "state_transport_source_roundtrip_rms": float(source_roundtrip_error.square().mean().sqrt().item()),
        "state_transport_target_closure_max_abs": float(closure_error.abs().max().item()),
        "state_transport_target_closure_rms": float(closure_error.square().mean().sqrt().item()),
        "state_transport_added_rng": False,
        "state_transport_temporal_mixing": False,
        "state_transport_amplitude_normalization": False,
    }
    return target_video, metrics
