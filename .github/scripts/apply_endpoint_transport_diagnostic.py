from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


STATE_TRANSPORT = r'''from __future__ import annotations

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
'''


HIGH_STAGE_DIAGNOSTICS = r'''from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch

from .geometry import unpack_streams
from .metrics import H3FlowMetrics
from .seam_diagnostics import measure_video_boundary

HIGH_STAGE_DIAGNOSTIC_KEY = "h3_flow_mixed_grid_high_boundary_v1"

_CALL_PROVENANCE_FIELDS = (
    "logical_step",
    "sigma",
    "coordinate",
    "actual",
    "provenance",
    "solver_phase",
    "solver_outer_step",
    "spectrum_step_id",
)


@dataclass(slots=True)
class HighStageDiagnosticHolder:
    """Invocation-local provenance shared through Comfy model-options cloning."""

    active: bool = False
    call_index: int = 0
    last_call: dict[str, Any] | None = None
    call_history: list[dict[str, Any]] = field(default_factory=list)


def _holder(contract: dict[str, Any]) -> HighStageDiagnosticHolder:
    holder = contract.get("holder")
    if not isinstance(holder, HighStageDiagnosticHolder):
        raise RuntimeError("high-stage diagnostic contract lost its shared provenance holder")
    return holder


def make_high_stage_diagnostic_contract(
    *,
    prefix_t: int,
    shapes: list[tuple[int, ...]],
    phases: tuple[tuple[int, str], ...],
    sampler: str,
) -> dict[str, Any]:
    if prefix_t <= 0:
        raise ValueError("high-stage boundary diagnostics require a positive exact-prefix length")
    if len(shapes) != 2:
        raise ValueError("high-stage boundary diagnostics require packed video/audio shapes")
    return {
        "api": 2,
        "prefix_t": int(prefix_t),
        "shapes": tuple(tuple(int(value) for value in shape) for shape in shapes),
        "phases": tuple((int(outer), str(phase)) for outer, phase in phases),
        "sampler": str(sampler),
        "holder": HighStageDiagnosticHolder(),
    }


@contextlib.contextmanager
def high_stage_diagnostic_context(guider: Any, contract: dict[str, Any]):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("high-stage boundary diagnostics require mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("high-stage boundary diagnostics require mutable transformer options")
    if HIGH_STAGE_DIAGNOSTIC_KEY in transformer:
        raise RuntimeError("nested high-stage boundary diagnostics are unsupported")
    holder = _holder(contract)
    if holder.active:
        raise RuntimeError("high-stage diagnostic holder is already owned by another invocation")
    holder.active = True
    transformer[HIGH_STAGE_DIAGNOSTIC_KEY] = contract
    try:
        yield contract
    finally:
        transformer.pop(HIGH_STAGE_DIAGNOSTIC_KEY, None)
        holder.active = False


def next_call_fields(
    contract: dict[str, Any],
    *,
    sigma: float,
    coordinate: float,
    actual: bool,
    solver_phase: str | None,
    solver_outer_step: int | None,
    spectrum_step_id: int | None,
) -> dict[str, Any]:
    holder = _holder(contract)
    call_index = int(holder.call_index)
    phases = tuple(contract.get("phases") or ())
    if call_index < len(phases):
        fallback_outer, fallback_phase = phases[call_index]
    else:
        fallback_outer, fallback_phase = call_index, "unclassified"
    phase = fallback_phase if solver_phase is None else str(solver_phase)
    outer = fallback_outer if solver_outer_step is None else int(solver_outer_step)
    fields = {
        "logical_step": call_index,
        "sigma": float(sigma),
        "coordinate": float(coordinate),
        "actual": bool(actual),
        "provenance": "actual" if actual else "forecast",
        "solver_phase": phase,
        "solver_outer_step": int(outer),
        "spectrum_step_id": None if spectrum_step_id is None else int(spectrum_step_id),
        "prefix_t": int(contract["prefix_t"]),
        "sampler": str(contract["sampler"]),
    }
    holder.call_index = call_index + 1
    stored = fields.copy()
    holder.last_call = stored
    holder.call_history.append(stored)
    return fields


def _boundary_fields(video: torch.Tensor, prefix_t: int, *, prefix: str = "") -> dict[str, Any]:
    boundary = measure_video_boundary(video, prefix_t)
    return {
        f"{prefix}seam_lowpass_kernel": boundary["lowpass_kernel"],
        f"{prefix}seam_rms": boundary["seam_rms"],
        f"{prefix}seam_lowpass_rms": boundary["seam_lowpass_rms"],
        f"{prefix}seam_spatial_mean_rms": boundary["seam_spatial_mean_rms"],
    }


def packed_boundary_fields(
    packed: torch.Tensor,
    contract: dict[str, Any],
    *,
    field_prefix: str = "",
) -> dict[str, Any]:
    shapes = [tuple(shape) for shape in contract["shapes"]]
    video, _audio = unpack_streams(packed, shapes)
    return _boundary_fields(video, int(contract["prefix_t"]), prefix=field_prefix)


def _prefixed_call_fields(call: dict[str, Any] | None, *, prefix: str) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {f"{prefix}{name}": None for name in _CALL_PROVENANCE_FIELDS}
    return {f"{prefix}{name}": call.get(name) for name in _CALL_PROVENANCE_FIELDS}


def _state_source_call(holder: HighStageDiagnosticHolder, current_call: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(current_call, dict):
        return None
    current_outer = current_call.get("solver_outer_step")
    if current_outer is None:
        return None
    current_outer = int(current_outer)
    if len(holder.call_history) < 2:
        return None
    for candidate in reversed(holder.call_history[:-1]):
        candidate_outer = candidate.get("solver_outer_step")
        if candidate_outer is not None and int(candidate_outer) < current_outer:
            return candidate
    return None


def record_packed_boundary(
    metrics: H3FlowMetrics,
    kind: str,
    packed: torch.Tensor,
    contract: dict[str, Any],
    fields: dict[str, Any],
    *,
    field_prefix: str = "",
) -> None:
    metrics.event(kind, **fields, **packed_boundary_fields(packed, contract, field_prefix=field_prefix))


def record_video_boundary(
    metrics: H3FlowMetrics,
    kind: str,
    video: torch.Tensor,
    contract: dict[str, Any],
    fields: dict[str, Any],
    *,
    field_prefix: str = "",
) -> None:
    metrics.event(
        kind,
        **fields,
        **_boundary_fields(video, int(contract["prefix_t"]), prefix=field_prefix),
    )


def record_callback_boundary(
    metrics: H3FlowMetrics,
    *,
    step: int,
    global_step: int,
    x0: torch.Tensor,
    x: torch.Tensor,
    contract: dict[str, Any],
) -> None:
    holder = _holder(contract)
    current_call = holder.last_call
    fields = dict(current_call) if isinstance(current_call, dict) else {}
    state_source = _state_source_call(holder, current_call)
    state_after_previous_outer = state_source is not None
    completed_outer = None if state_source is None else int(state_source["solver_outer_step"])
    fields.update(
        {
            "callback_step": int(step),
            "global_step": int(global_step),
            "event_call_fields_semantics": "current_callback_x0_prediction",
            "state_semantics": (
                "pre_current_solver_update_post_previous_outer_update"
                if state_after_previous_outer
                else "high_stage_input_before_first_solver_update"
            ),
            "state_after_previous_solver_step": state_after_previous_outer,
            "completed_solver_step": completed_outer,
            "state_source_semantics": (
                "last_model_call_of_previous_solver_outer" if state_after_previous_outer else "no_previous_solver_outer"
            ),
            "x0_semantics": "sampler_callback_denoised_after_model_wrappers",
            **_prefixed_call_fields(current_call, prefix="x0_call_"),
            **_prefixed_call_fields(state_source, prefix="state_source_call_"),
        }
    )
    shapes = [tuple(shape) for shape in contract["shapes"]]
    state_video, _ = unpack_streams(x, shapes)
    x0_video, _ = unpack_streams(x0, shapes)
    metrics.event(
        "mixed_grid_high_step_boundary",
        **fields,
        **_boundary_fields(state_video, int(contract["prefix_t"]), prefix="state_"),
        **_boundary_fields(x0_video, int(contract["prefix_t"]), prefix="x0_"),
    )
'''

Path("h3_flow_regenerate/state_transport.py").write_text(STATE_TRANSPORT, encoding="utf-8")
Path("h3_flow_regenerate/high_stage_diagnostics.py").write_text(HIGH_STAGE_DIAGNOSTICS, encoding="utf-8")

# Config accepts the temporary endpoint-residual comparator only on the same
# Mixed-Grid learned-transfer path. The legacy default remains unchanged.
replace_once(
    "h3_flow_regenerate/handoff.py",
    "from .state_transport import (\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    resolve_handoff_state_policy,\n)",
    "from .state_transport import (\n    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    resolve_handoff_state_policy,\n)",
)
replace_once(
    "h3_flow_regenerate/handoff.py",
    "        if state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1 and self.transfer_mode != \"learned_3d\":\n            raise ValueError(\"velocity_bicubic_v1 requires learned_3d clean transfer\")",
    "        if state_policy in {\n            HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n            HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,\n        } and self.transfer_mode != \"learned_3d\":\n            raise ValueError(f\"{state_policy} requires learned_3d clean transfer\")",
)

# Expose the endpoint comparator only in the temporary diagnostic overlay.
replace_once(
    "h3_flow_regenerate/target_sparse_node.py",
    "            [\"legacy_renoise\", \"velocity_bicubic_v1\"],",
    "            [\"legacy_renoise\", \"velocity_bicubic_v1\", \"endpoint_residual_bicubic_v1\"],",
)
replace_once(
    "h3_flow_regenerate/target_sparse_node.py",
    "                    \"velocity_bicubic_v1 re-anchors the learned corrected clean suffix while transporting \"\n                    \"the source sampler displacement X-C with a fixed framewise bicubic lift. This is an \"\n                    \"experimental controlled candidate until matched CUDA/media validation is accepted.\"",
    "                    \"velocity_bicubic_v1 transports the source sampler displacement X-C. \"\n                    \"endpoint_residual_bicubic_v1 is the temporary matched comparator that instead preserves \"\n                    \"the inferred endpoint field X-(1-sigma)C. Both use the same fixed framewise bicubic lift; \"\n                    \"legacy_renoise remains the default and production rollback.\"",
)

# Runtime: both explicit transport candidates retain the accepted probe clean,
# run the learned provider once, and differ only in the state invariant.
replace_once(
    "h3_flow_regenerate/runtime.py",
    "from .high_stage_diagnostics import (\n    HIGH_STAGE_DIAGNOSTIC_KEY,\n    high_stage_diagnostic_context,\n    make_high_stage_diagnostic_contract,\n    next_call_fields,\n    record_callback_boundary,\n    record_packed_boundary,\n    record_video_boundary,\n)",
    "from .high_stage_diagnostics import (\n    HIGH_STAGE_DIAGNOSTIC_KEY,\n    high_stage_diagnostic_context,\n    make_high_stage_diagnostic_contract,\n    next_call_fields,\n    packed_boundary_fields,\n    record_callback_boundary,\n    record_packed_boundary,\n    record_video_boundary,\n)",
)
replace_once(
    "h3_flow_regenerate/runtime.py",
    "from .state_transport import (\n    HANDOFF_STATE_POLICY_LEGACY,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    resolve_handoff_state_policy,\n    transport_velocity_bicubic_v1,\n)",
    "from .state_transport import (\n    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,\n    HANDOFF_STATE_POLICY_LEGACY,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    measure_state_transport_comparison_v1,\n    resolve_handoff_state_policy,\n    transport_endpoint_residual_bicubic_v1,\n    transport_velocity_bicubic_v1,\n)",
)
replace_once(
    "h3_flow_regenerate/runtime.py",
    "    started = time.perf_counter()\n    result = executor(x, timestep, model_options, seed)\n    if binding is None:\n        return result\n    transformer = (model_options or {}).get(\"transformer_options\") or {}\n    probe_context = transformer.get(PROBE_CONTEXT_KEY)\n    stage = str(transformer.get(FLOW_STAGE_KEY, \"single\"))",
    "    transformer = (model_options or {}).get(\"transformer_options\") or {}\n    stage = str(transformer.get(FLOW_STAGE_KEY, \"single\"))\n    pre_high_diag_contract = transformer.get(HIGH_STAGE_DIAGNOSTIC_KEY)\n    high_model_input_boundary = None\n    if binding is not None and stage == \"high\" and isinstance(pre_high_diag_contract, dict):\n        high_model_input_boundary = packed_boundary_fields(\n            x,\n            pre_high_diag_contract,\n            field_prefix=\"model_input_\",\n        )\n    started = time.perf_counter()\n    result = executor(x, timestep, model_options, seed)\n    if binding is None:\n        return result\n    probe_context = transformer.get(PROBE_CONTEXT_KEY)",
)
replace_once(
    "h3_flow_regenerate/runtime.py",
    "        high_diag_fields = next_call_fields(\n            high_diag_contract,\n            sigma=sigma,\n            coordinate=coordinate,\n            actual=actual,\n            solver_phase=diagnostic_phase,\n            solver_outer_step=diagnostic_outer,\n            spectrum_step_id=spectrum_step_id,\n        )\n        record_packed_boundary(\n            binding.metrics,\n            \"mixed_grid_high_prediction_boundary\",\n            result,\n            high_diag_contract,\n            high_diag_fields,\n        )",
    "        high_diag_fields = next_call_fields(\n            high_diag_contract,\n            sigma=sigma,\n            coordinate=coordinate,\n            actual=actual,\n            solver_phase=diagnostic_phase,\n            solver_outer_step=diagnostic_outer,\n            spectrum_step_id=spectrum_step_id,\n        )\n        if high_model_input_boundary is None:\n            raise RuntimeError(\"high-stage diagnostics lost the pre-executor model-input boundary\")\n        binding.metrics.event(\n            \"mixed_grid_high_model_input_boundary\",\n            **high_diag_fields,\n            model_input_semantics=\"post_native_inpaint_pre_prediction_executor\",\n            captured_before_executor=True,\n            **high_model_input_boundary,\n        )\n        record_packed_boundary(\n            binding.metrics,\n            \"mixed_grid_high_prediction_boundary\",\n            result,\n            high_diag_contract,\n            high_diag_fields,\n        )",
)
replace_once(
    "h3_flow_regenerate/runtime.py",
    "            source_x0.detach().clone()\n            if mixed_plan is not None and effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1\n            else None",
    "            source_x0.detach().clone()\n            if mixed_plan is not None and effective_state_policy != HANDOFF_STATE_POLICY_LEGACY\n            else None",
)
# Three policy branches currently name velocity explicitly: provider preparation,
# clean-bridge telemetry and target-state construction.
runtime_path = Path("h3_flow_regenerate/runtime.py")
runtime_text = runtime_path.read_text(encoding="utf-8")
needle = "if mixed_plan is not None and effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:"
if runtime_text.count(needle) != 1:
    raise RuntimeError(f"runtime provider branch count changed: {runtime_text.count(needle)}")
runtime_text = runtime_text.replace(
    needle,
    "if mixed_plan is not None and effective_state_policy != HANDOFF_STATE_POLICY_LEGACY:",
    1,
)
needle = "if effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:"
if runtime_text.count(needle) != 2:
    raise RuntimeError(f"runtime state-policy branch count changed: {runtime_text.count(needle)}")
runtime_text = runtime_text.replace(needle, "if effective_state_policy != HANDOFF_STATE_POLICY_LEGACY:", 1)
old_transport = '''            if effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:\n                target_video, state_transport_metrics = transport_velocity_bicubic_v1(\n                    source_state_video,\n                    accepted_clean_video,\n                    corrected_clean,\n                    prefix_t=mixed_plan.prefix_t,\n                )\n                binding.metrics.increment("mixed_grid_state_transport_runs")\n                binding.metrics.event("mixed_grid_state_transport", **state_transport_metrics)'''
new_transport = '''            if effective_state_policy != HANDOFF_STATE_POLICY_LEGACY:\n                comparison_metrics = measure_state_transport_comparison_v1(\n                    source_state_video,\n                    accepted_clean_video,\n                    corrected_clean,\n                    prefix_t=mixed_plan.prefix_t,\n                    sigma=sigma,\n                )\n                if effective_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:\n                    target_video, transport_metrics = transport_velocity_bicubic_v1(\n                        source_state_video,\n                        accepted_clean_video,\n                        corrected_clean,\n                        prefix_t=mixed_plan.prefix_t,\n                    )\n                elif effective_state_policy == HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1:\n                    target_video, transport_metrics = transport_endpoint_residual_bicubic_v1(\n                        source_state_video,\n                        accepted_clean_video,\n                        corrected_clean,\n                        prefix_t=mixed_plan.prefix_t,\n                        sigma=sigma,\n                    )\n                else:\n                    raise RuntimeError(f"unreviewed Mixed-Grid handoff state policy {effective_state_policy!r}")\n                state_transport_metrics = {**comparison_metrics, **transport_metrics}\n                binding.metrics.increment("mixed_grid_state_transport_runs")\n                binding.metrics.event("mixed_grid_state_transport", **state_transport_metrics)'''
if old_transport not in runtime_text:
    raise RuntimeError("runtime transport branch did not match expected source")
runtime_text = runtime_text.replace(old_transport, new_transport, 1)
runtime_text = runtime_text.replace(
    '"clean_reanchor_velocity_bicubic_v1" if dc_enabled else "disabled"',
    'f"clean_reanchor_{effective_state_policy}" if dc_enabled else "disabled"',
    1,
)
runtime_text = runtime_text.replace(
    '"clean_reanchor_velocity_bicubic_v1"\n                    if representation_metrics["suffix_representation_bridge_accepted"]',
    'f"clean_reanchor_{effective_state_policy}"\n                    if representation_metrics["suffix_representation_bridge_accepted"]',
    1,
)
runtime_path.write_text(runtime_text, encoding="utf-8")

# Unit expectations for the corrected shared provenance holder and the added
# pre-prediction model-input scalar event.
replace_once(
    "tests/test_high_stage_diagnostics.py",
    "    assert contract[\"call_index\"] == 2\n    assert contract[\"last_call\"] == forecast\n    assert contract[\"call_history\"] == [actual, forecast]",
    "    holder = contract[\"holder\"]\n    assert holder.call_index == 2\n    assert holder.last_call == forecast\n    assert holder.call_history == [actual, forecast]",
)
replace_once(
    "tests/test_high_stage_diagnostics.py",
    "    diagnostic_kinds = [event.kind for event in diagnostic_metrics.events]\n    assert diagnostic_kinds == [\n        \"model_call\",\n        \"mixed_grid_high_prediction_boundary\",\n        \"mixed_grid_high_guided_boundary\",\n        \"guidance\",\n    ]\n    prediction = diagnostic_metrics.events[1].fields\n    guided = diagnostic_metrics.events[2].fields",
    "    diagnostic_kinds = [event.kind for event in diagnostic_metrics.events]\n    assert diagnostic_kinds == [\n        \"model_call\",\n        \"mixed_grid_high_model_input_boundary\",\n        \"mixed_grid_high_prediction_boundary\",\n        \"mixed_grid_high_guided_boundary\",\n        \"guidance\",\n    ]\n    model_input = diagnostic_metrics.events[1].fields\n    prediction = diagnostic_metrics.events[2].fields\n    guided = diagnostic_metrics.events[3].fields\n    assert model_input[\"model_input_semantics\"] == \"post_native_inpaint_pre_prediction_executor\"\n    assert model_input[\"captured_before_executor\"] is True\n    assert model_input[\"provenance\"] == \"actual\"",
)

# Runtime state-transport fixture covers both serious residual policies with the
# same low/probe/high cardinality and no added random draw.
replace_once(
    "tests/test_state_transport_runtime.py",
    "import torch\nfrom test_handoff import FakeLearnedProvider",
    "import pytest\nimport torch\nfrom test_handoff import FakeLearnedProvider",
)
replace_once(
    "tests/test_state_transport_runtime.py",
    "from h3_flow_regenerate.state_transport import HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1\n\n\ndef test_mixed_grid_velocity_transport_preserves_state_noise_audio_and_sampler_boundaries(monkeypatch):",
    "from h3_flow_regenerate.state_transport import (\n    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n)\n\n\n@pytest.mark.parametrize(\n    \"policy\",\n    [HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1, HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1],\n)\ndef test_mixed_grid_transport_preserves_state_noise_audio_and_sampler_boundaries(monkeypatch, policy):",
)
replace_once(
    "tests/test_state_transport_runtime.py",
    "        handoff_state_policy=HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,",
    "        handoff_state_policy=policy,",
)
replace_once(
    "tests/test_state_transport_runtime.py",
    "            lifted_displacement = resize_spatial_5d(realized_source_displacement, 8, 12, mode=\"bicubic\")\n            expected_suffix = target_clean_video[:, :, 2:] + lifted_displacement[:, :, 2:]",
    "            if policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1:\n                lifted_displacement = resize_spatial_5d(realized_source_displacement, 8, 12, mode=\"bicubic\")\n                expected_suffix = target_clean_video[:, :, 2:] + lifted_displacement[:, :, 2:]\n            else:\n                endpoint_residual = source_state_video - (1.0 - sigma) * source_clean_video\n                lifted_endpoint = resize_spatial_5d(endpoint_residual, 8, 12, mode=\"bicubic\")\n                expected_suffix = (1.0 - sigma) * target_clean_video[:, :, 2:] + lifted_endpoint[:, :, 2:]",
)
replace_once(
    "tests/test_state_transport_runtime.py",
    "    # The only deterministic transfer-independent draw is the existing private\n    # low-grid source noise. velocity_bicubic_v1 adds no target-grid random field.",
    "    # The only deterministic transfer-independent draw is the existing private\n    # low-grid source noise. Neither transport candidate adds a target-grid random field.",
)
replace_once(
    "tests/test_state_transport_runtime.py",
    "    assert transport[\"handoff_state_policy\"] == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1",
    "    assert transport[\"handoff_state_policy\"] == policy",
)
replace_once(
    "tests/test_state_transport_runtime.py",
    "    assert complete[0].fields[\"handoff_state_policy\"] == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1",
    "    assert complete[0].fields[\"handoff_state_policy\"] == policy",
)

# UI/config tests keep legacy as the default while exposing one temporary endpoint comparator.
replace_once(
    "tests/test_state_transport_config.py",
    "from h3_flow_regenerate.state_transport import (\n    HANDOFF_STATE_POLICY_LEGACY,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    resolve_handoff_state_policy,\n)",
    "from h3_flow_regenerate.state_transport import (\n    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,\n    HANDOFF_STATE_POLICY_LEGACY,\n    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n    resolve_handoff_state_policy,\n)",
)
replace_once(
    "tests/test_state_transport_config.py",
    "    assert state_policy[0] == [HANDOFF_STATE_POLICY_LEGACY, HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1]",
    "    assert state_policy[0] == [\n        HANDOFF_STATE_POLICY_LEGACY,\n        HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,\n        HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,\n    ]",
)

# Add focused pure-policy and native clone-semantics tests.
Path("tests/test_endpoint_state_transport.py").write_text(
    r'''from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d
from h3_flow_regenerate.state_transport import (
    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,
    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    measure_state_transport_comparison_v1,
    resolve_handoff_state_policy,
    transport_endpoint_residual_bicubic_v1,
    transport_velocity_bicubic_v1,
)


def test_endpoint_policy_resolves_without_changing_legacy_default():
    assert (
        resolve_handoff_state_policy(HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1)
        == HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1
    )


def test_endpoint_and_velocity_are_distinct_for_nonlinear_clean_reanchor():
    sigma = 0.8
    prefix_t = 1
    source_clean = torch.full((1, 24, 3, 4, 4), 0.5, dtype=torch.float32)
    source_state = source_clean + 0.75
    target_clean = torch.full((1, 24, 3, 6, 8), 2.0, dtype=torch.float32)

    velocity, _ = transport_velocity_bicubic_v1(source_state, source_clean, target_clean, prefix_t=prefix_t)
    endpoint, metrics = transport_endpoint_residual_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
        sigma=sigma,
    )
    lifted_source_clean = resize_spatial_5d(source_clean, 6, 8, mode="bicubic")
    expected_delta = sigma * (target_clean[:, :, prefix_t:] - lifted_source_clean[:, :, prefix_t:])
    torch.testing.assert_close(
        velocity[:, :, prefix_t:] - endpoint[:, :, prefix_t:],
        expected_delta,
        rtol=2e-5,
        atol=2e-5,
    )
    assert metrics["state_transport_clean_coefficient"] == pytest.approx(1.0 - sigma)
    assert metrics["state_transport_residual_kind"] == "effective_endpoint"


def test_endpoint_linear_clean_transfer_reduces_to_direct_state_lift():
    sigma = 0.8780487775802612
    prefix_t = 1
    source_clean = torch.linspace(-0.5, 0.7, steps=1 * 24 * 4 * 4 * 6).reshape(1, 24, 4, 4, 6)
    source_state = source_clean + torch.linspace(-0.2, 0.3, steps=source_clean.numel()).reshape_as(source_clean)
    target_clean = resize_spatial_5d(source_clean, 8, 10, mode="bicubic")
    endpoint, _ = transport_endpoint_residual_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
        sigma=sigma,
    )
    direct = resize_spatial_5d(source_state, 8, 10, mode="bicubic")
    torch.testing.assert_close(endpoint[:, :, prefix_t:], direct[:, :, prefix_t:], rtol=2e-5, atol=2e-5)


def test_endpoint_rejects_sigma_endpoints_and_comparison_records_policy_gap():
    source_clean = torch.zeros((1, 24, 3, 4, 4), dtype=torch.float32)
    source_state = torch.ones_like(source_clean)
    target_clean = torch.full((1, 24, 3, 6, 8), 2.0)
    for sigma in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="sigma inside"):
            transport_endpoint_residual_bicubic_v1(
                source_state,
                source_clean,
                target_clean,
                prefix_t=1,
                sigma=sigma,
            )

    metrics = measure_state_transport_comparison_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=1,
        sigma=0.8,
    )
    assert metrics["state_transport_comparison_version"] == 1
    assert metrics["state_transport_clean_reanchor_delta_rms"] == pytest.approx(2.0, rel=1e-6)
    assert metrics["state_transport_velocity_endpoint_expected_delta_rms"] == pytest.approx(1.6, rel=1e-6)
    for key, value in metrics.items():
        if key.endswith("_rms"):
            assert torch.isfinite(torch.tensor(value))
''',
    encoding="utf-8",
)

Path("tests/test_high_stage_diagnostics_native.py").write_text(
    r'''from __future__ import annotations

import ast
import os
from pathlib import Path
from types import ModuleType

import pytest

from h3_flow_regenerate.high_stage_diagnostics import HIGH_STAGE_DIAGNOSTIC_KEY, make_high_stage_diagnostic_contract


@pytest.fixture(scope="module")
def native_copy_nested_dicts():
    root = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "comfy"))
    path = root / "comfy/patcher_extension.py"
    if not path.is_file():
        if os.environ.get("COMFYUI_ROOT"):
            raise FileNotFoundError(path)
        pytest.skip("native clone oracle runs in source-contract CI with COMFYUI_ROOT")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "copy_nested_dicts"]
    if len(nodes) != 1:
        raise AssertionError(f"pinned ComfyUI copy_nested_dicts changed: found {len(nodes)} definitions")
    module = ModuleType("native_copy_nested_dicts_oracle")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.copy_nested_dicts


def test_non_dict_diagnostic_holder_keeps_identity_through_pinned_comfy_clone(native_copy_nested_dicts):
    contract = make_high_stage_diagnostic_contract(
        prefix_t=1,
        shapes=[(1, 24, 3, 4, 4), (1, 32, 2, 3)],
        phases=((0, "single"),),
        sampler="sample_res_multistep",
    )
    holder = contract["holder"]
    options = {"transformer_options": {HIGH_STAGE_DIAGNOSTIC_KEY: contract}}
    cloned = native_copy_nested_dicts(options)
    cloned_contract = cloned["transformer_options"][HIGH_STAGE_DIAGNOSTIC_KEY]
    assert cloned_contract is not contract
    assert cloned_contract["holder"] is holder
''',
    encoding="utf-8",
)

# Exercise the actual pinned clone semantic in source-contract CI.
replace_once(
    ".github/workflows/ci.yml",
    "python -m pytest tests/test_mixed_grid.py tests/test_state_transport_native.py tests/test_decode_context.py -q",
    "python -m pytest tests/test_mixed_grid.py tests/test_state_transport_native.py tests/test_high_stage_diagnostics_native.py tests/test_decode_context.py -q",
)

# Temporary helper files remove themselves from the resulting implementation commit.
Path(".github/scripts/apply_endpoint_transport_diagnostic.py").unlink()
Path(".github/workflows/apply-endpoint-transport-diagnostic.yml").unlink()
