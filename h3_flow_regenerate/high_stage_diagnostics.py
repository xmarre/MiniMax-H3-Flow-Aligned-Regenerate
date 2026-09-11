from __future__ import annotations

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
