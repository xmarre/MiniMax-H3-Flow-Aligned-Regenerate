from __future__ import annotations

import contextlib
from typing import Any

import torch

from .geometry import unpack_streams
from .metrics import H3FlowMetrics
from .seam_diagnostics import measure_video_boundary

HIGH_STAGE_DIAGNOSTIC_KEY = "h3_flow_mixed_grid_high_boundary_v1"


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
        "api": 1,
        "prefix_t": int(prefix_t),
        "shapes": tuple(tuple(int(value) for value in shape) for shape in shapes),
        "phases": tuple((int(outer), str(phase)) for outer, phase in phases),
        "sampler": str(sampler),
        "call_index": 0,
        "last_call": None,
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
    transformer[HIGH_STAGE_DIAGNOSTIC_KEY] = contract
    try:
        yield contract
    finally:
        transformer.pop(HIGH_STAGE_DIAGNOSTIC_KEY, None)


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
    call_index = int(contract.get("call_index", 0))
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
    contract["call_index"] = call_index + 1
    contract["last_call"] = fields.copy()
    return fields


def _boundary_fields(video: torch.Tensor, prefix_t: int, *, prefix: str = "") -> dict[str, Any]:
    boundary = measure_video_boundary(video, prefix_t)
    return {
        f"{prefix}seam_lowpass_kernel": boundary["lowpass_kernel"],
        f"{prefix}seam_rms": boundary["seam_rms"],
        f"{prefix}seam_lowpass_rms": boundary["seam_lowpass_rms"],
        f"{prefix}seam_spatial_mean_rms": boundary["seam_spatial_mean_rms"],
    }


def record_packed_boundary(
    metrics: H3FlowMetrics,
    kind: str,
    packed: torch.Tensor,
    contract: dict[str, Any],
    fields: dict[str, Any],
    *,
    field_prefix: str = "",
) -> None:
    shapes = [tuple(shape) for shape in contract["shapes"]]
    video, _audio = unpack_streams(packed, shapes)
    metrics.event(
        kind,
        **fields,
        **_boundary_fields(video, int(contract["prefix_t"]), prefix=field_prefix),
    )


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
    call_fields = contract.get("last_call")
    fields = dict(call_fields) if isinstance(call_fields, dict) else {}
    fields.update(
        {
            "callback_step": int(step),
            "global_step": int(global_step),
            "state_semantics": "pre_current_solver_update_post_previous_solver_update",
            "state_after_previous_solver_step": bool(step > 0),
            "completed_solver_step": int(step - 1) if step > 0 else None,
            "x0_semantics": "sampler_callback_denoised",
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
