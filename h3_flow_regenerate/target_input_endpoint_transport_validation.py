"""Validation overlay for flow-endpoint-preserving target-input handoff transport.

PR #39 tested a full local-velocity re-anchor across the learned 3D resolution
change. Media validation showed that construction drives both first- and
last-high clean predictions into a new global corruption mode. The failed
construction was:

    X_target = C_target + U(X_source - C_source)

For ComfyUI's CONST flow path, a state at sigma is parameterized as

    X_sigma = sigma * N + (1 - sigma) * C

for a clean endpoint C and a noise/path endpoint N. If the clean endpoint is
replaced by the learned target-grid clean C_target while preserving the source
path endpoint through a spatial lift U, the corresponding same-sigma state is

    X_target = U(X_source) + (1 - sigma) * (C_target - U(C_source))

or equivalently

    N_scaled_source = X_source - (1 - sigma) * C_source
    X_target = U(N_scaled_source) + (1 - sigma) * C_target

This overlay tests that endpoint-preserving construction only. The existing
learned upscaler still executes exactly once through the wrapped production
handoff. The legacy target random field is generated only so the already-built
learned clean target can be recovered, then discarded. No H3 evaluation,
sampler invocation, temporal mixing, or amplitude normalization is added.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from . import runtime as _runtime
from .geometry import pack_streams, resize_spatial_5d, unpack_streams
from .handoff import ProgressiveTargetInputConfig, deterministic_video_noise
from .seam_diagnostics import recover_conditional_clean_for_diagnostics

_STATE_ATTR = "_h3_target_input_endpoint_transport_validation_state_v1"
_WRAPPER_MARK = "_h3_target_input_endpoint_transport_validation_v1"
_POLICY = "same_sigma_endpoint_bicubic_v1"


@dataclass(slots=True)
class _TransportRecord:
    enabled: bool
    binding: Any
    transport_calls: int = 0


class _TransportState:
    def __init__(self) -> None:
        self.tls = threading.local()


def _state() -> _TransportState:
    state = getattr(_runtime, _STATE_ATTR, None)
    if state is None:
        state = _TransportState()
        setattr(_runtime, _STATE_ATTR, state)
    return state


def _active() -> _TransportRecord | None:
    return getattr(_state().tls, "record", None)


def _eligible(config: Any, binding: Any, denoise_mask: torch.Tensor | None) -> bool:
    guidance = getattr(binding, "guidance", None)
    return (
        isinstance(config, ProgressiveTargetInputConfig)
        and config.transfer_mode == "learned_3d"
        and config.exact_prefix_mode == "fallback"
        and denoise_mask is None
        and guidance is not None
        and getattr(guidance, "mode", "off") != "off"
    )


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    if tensors and all(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().to(device="cpu", dtype=torch.float64).item())


def _resize_field(field: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if field.dtype != torch.float64:
        return resize_spatial_5d(field, target_h, target_w, mode="bicubic")

    batch, channels, temporal, height, width = field.shape
    work = field.permute(0, 2, 1, 3, 4).reshape(batch * temporal, channels, height, width)
    out = F.interpolate(
        work,
        size=(int(target_h), int(target_w)),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    return out.reshape(batch, temporal, channels, int(target_h), int(target_w)).permute(0, 2, 1, 3, 4)


def _transport_video_state(
    source_state: torch.Tensor,
    source_clean: torch.Tensor,
    target_clean: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if source_state.ndim != 5 or source_clean.ndim != 5 or target_clean.ndim != 5:
        raise ValueError("endpoint transport requires BxCxTxHxW video tensors")
    if source_state.shape != source_clean.shape:
        raise ValueError("source sampler state and source clean prediction must have identical geometry")
    if source_state.shape[:3] != target_clean.shape[:3]:
        raise ValueError("endpoint transport cannot change batch, channel, or temporal geometry")
    if source_state.device != source_clean.device or source_state.device != target_clean.device:
        raise ValueError("endpoint-transport tensors must share a device")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in (source_state, source_clean, target_clean)):
        raise ValueError("endpoint transport input contains NaN or Inf")
    sigma = float(sigma)
    if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
        raise ValueError("endpoint transport requires finite sigma strictly inside (0, 1)")

    source_h, source_w = map(int, source_state.shape[-2:])
    target_h, target_w = map(int, target_clean.shape[-2:])
    if target_h < source_h or target_w < source_w:
        raise ValueError("endpoint transport target must not shrink either video axis")
    if target_h == source_h and target_w == source_w:
        raise ValueError("endpoint transport requires changed spatial geometry")

    compute_dtype = _compute_dtype(source_state, source_clean, target_clean)
    source_state_work = source_state.to(dtype=compute_dtype)
    source_clean_work = source_clean.to(dtype=compute_dtype)
    target_clean_work = target_clean.to(dtype=compute_dtype)
    clean_weight = 1.0 - sigma

    source_scaled_endpoint = source_state_work - clean_weight * source_clean_work
    lifted_scaled_endpoint = _resize_field(source_scaled_endpoint, target_h, target_w)
    if lifted_scaled_endpoint.dtype != compute_dtype:
        lifted_scaled_endpoint = lifted_scaled_endpoint.to(dtype=compute_dtype)
    if not bool(torch.isfinite(lifted_scaled_endpoint).all().item()):
        raise RuntimeError("lifted source endpoint contains NaN or Inf")

    transported_work = lifted_scaled_endpoint + clean_weight * target_clean_work
    if not bool(torch.isfinite(transported_work).all().item()):
        raise RuntimeError("transported target sampler state contains NaN or Inf")
    transported = transported_work.to(dtype=target_clean.dtype)

    source_roundtrip = source_scaled_endpoint + clean_weight * source_clean_work - source_state_work
    realized = transported.to(dtype=compute_dtype)
    closure = realized - lifted_scaled_endpoint - clean_weight * target_clean_work

    lifted_source_state = _resize_field(source_state_work, target_h, target_w).to(dtype=compute_dtype)
    lifted_source_clean = _resize_field(source_clean_work, target_h, target_w).to(dtype=compute_dtype)
    anchor_delta = target_clean_work - lifted_source_clean
    direct_form = lifted_source_state + clean_weight * anchor_delta
    direct_form_error = realized - direct_form

    lifted_velocity = _resize_field(source_state_work - source_clean_work, target_h, target_w).to(dtype=compute_dtype)
    failed_velocity_state = target_clean_work + lifted_velocity
    failed_velocity_overshoot = failed_velocity_state - realized
    expected_overshoot = sigma * anchor_delta
    overshoot_identity_error = failed_velocity_overshoot - expected_overshoot

    endpoint_rms = _rms(source_scaled_endpoint)
    lifted_endpoint_rms = _rms(lifted_scaled_endpoint)
    metrics = {
        "policy": _POLICY,
        "source_hw": (source_h, source_w),
        "target_hw": (target_h, target_w),
        "compute_dtype": str(compute_dtype),
        "sigma": sigma,
        "clean_weight": clean_weight,
        "source_scaled_endpoint_rms": endpoint_rms,
        "lifted_scaled_endpoint_rms": lifted_endpoint_rms,
        "lifted_over_source_endpoint_rms_ratio": lifted_endpoint_rms / max(endpoint_rms, 1e-30),
        "clean_anchor_delta_rms": _rms(anchor_delta),
        "failed_velocity_overshoot_rms": _rms(failed_velocity_overshoot),
        "failed_velocity_overshoot_identity_max_abs": float(overshoot_identity_error.abs().max().item()),
        "failed_velocity_overshoot_identity_rms": _rms(overshoot_identity_error),
        "source_roundtrip_max_abs": float(source_roundtrip.abs().max().item()),
        "source_roundtrip_rms": _rms(source_roundtrip),
        "target_closure_max_abs": float(closure.abs().max().item()),
        "target_closure_rms": _rms(closure),
        "direct_form_max_abs": float(direct_form_error.abs().max().item()),
        "direct_form_rms": _rms(direct_form_error),
        "temporal_mixing": False,
        "amplitude_normalization": False,
    }
    return transported, metrics


def _build_handoff_state_transport_wrapper(
    *,
    source_packed_state: torch.Tensor,
    source_x0_packed: torch.Tensor,
    source_shapes: list[tuple[int, ...]],
    sigma: float,
    target_h: int,
    target_w: int,
    seed: int,
    transfer_mode: str = "bicubic",
    learned_upscaler: Any | None = None,
    transfer_metrics: dict[str, Any] | None = None,
):
    target_packed, target_shapes = _ORIGINAL_BUILD_HANDOFF_STATE(
        source_packed_state=source_packed_state,
        source_x0_packed=source_x0_packed,
        source_shapes=source_shapes,
        sigma=sigma,
        target_h=target_h,
        target_w=target_w,
        seed=seed,
        transfer_mode=transfer_mode,
        learned_upscaler=learned_upscaler,
        transfer_metrics=transfer_metrics,
    )
    record = _active()
    if record is None or not record.enabled or transfer_mode != "learned_3d":
        return target_packed, target_shapes

    normalized_source = [tuple(int(dim) for dim in shape) for shape in source_shapes]
    normalized_target = [tuple(int(dim) for dim in shape) for shape in target_shapes]
    source_state_video, source_audio = unpack_streams(source_packed_state, normalized_source)
    source_clean_video, _source_clean_audio = unpack_streams(source_x0_packed, normalized_source)
    legacy_target_video, legacy_target_audio = unpack_streams(target_packed, normalized_target)

    if not torch.equal(source_audio, legacy_target_audio):
        raise RuntimeError("legacy learned_3d handoff did not preserve carried audio state exactly")

    legacy_noise = deterministic_video_noise(
        tuple(int(dim) for dim in legacy_target_video.shape),
        seed=int(seed),
        device=legacy_target_video.device,
        dtype=legacy_target_video.dtype,
    )
    target_clean = recover_conditional_clean_for_diagnostics(
        legacy_target_video,
        legacy_noise,
        sigma=float(sigma),
    )
    transported_video, facts = _transport_video_state(
        source_state_video,
        source_clean_video,
        target_clean,
        sigma=float(sigma),
    )
    transported_packed, transported_shapes = pack_streams((transported_video, source_audio.clone()))
    normalized_transported = [tuple(int(dim) for dim in shape) for shape in transported_shapes]
    if normalized_transported != normalized_target:
        raise RuntimeError(
            "same-sigma endpoint transport changed packed target geometry: "
            f"{normalized_transported} vs {normalized_target}"
        )

    record.transport_calls += 1
    record.binding.metrics.increment("target_input_endpoint_transport_validation_runs")
    record.binding.metrics.event(
        "target_input_endpoint_transport_validation",
        applied=True,
        transfer_mode=transfer_mode,
        exact_prefix_mode="fallback",
        unprotected_only=True,
        target_audio_preserved_exactly=True,
        final_target_rng_contribution=False,
        legacy_target_rng_generated_then_discarded=True,
        extra_h3_evaluations=0,
        extra_upscaler_calls=0,
        production_status="validation_only",
        legacy_to_transport_rms=_rms(legacy_target_video.float() - transported_video.float()),
        **facts,
    )
    return transported_packed, transported_shapes


setattr(_build_handoff_state_transport_wrapper, _WRAPPER_MARK, True)


def _run_progressive_transport_wrapper(
    executor,
    guider,
    binding,
    config,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
):
    state = _state()
    previous = getattr(state.tls, "record", None)
    record = _TransportRecord(
        enabled=_eligible(config, binding, denoise_mask),
        binding=binding,
    )
    state.tls.record = record
    try:
        result = _ORIGINAL_RUN_PROGRESSIVE(
            executor,
            guider,
            binding,
            config,
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes,
        )
        if record.enabled and record.transport_calls != 1:
            raise RuntimeError(
                f"same-sigma endpoint transport expected exactly one learned handoff; observed {record.transport_calls}"
            )
        return result
    finally:
        state.tls.record = previous


setattr(_run_progressive_transport_wrapper, _WRAPPER_MARK, True)

_ORIGINAL_BUILD_HANDOFF_STATE = _runtime.build_handoff_state
_ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive


def install_target_input_endpoint_transport_validation() -> None:
    global _ORIGINAL_BUILD_HANDOFF_STATE, _ORIGINAL_RUN_PROGRESSIVE

    if not getattr(_runtime.build_handoff_state, _WRAPPER_MARK, False):
        _ORIGINAL_BUILD_HANDOFF_STATE = _runtime.build_handoff_state
        _runtime.build_handoff_state = _build_handoff_state_transport_wrapper
    if not getattr(_runtime._run_progressive, _WRAPPER_MARK, False):
        _ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive
        _runtime._run_progressive = _run_progressive_transport_wrapper


__all__ = ["install_target_input_endpoint_transport_validation"]
