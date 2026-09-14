"""Decode-only checkpoint capture for progressive H3 handoff diagnosis.

This module is intentionally diagnostic. It observes the existing progressive
Target Input runtime without adding H3 evaluations or invoking the latent
upscaler a second time. Captured video latents are copied to CPU only after a
successful progressive invocation completes.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import runtime as _runtime
from .geometry import resize_video, unpack_streams
from .guidance import GuidanceState, time_matched_reference_info
from .handoff import ProgressiveTargetInputConfig, deterministic_video_noise
from .seam_diagnostics import recover_conditional_clean_for_diagnostics

_STATE_ATTR = "_h3_handoff_checkpoint_diagnostic_state_v2"
_WRAPPER_MARK = "_h3_handoff_checkpoint_diagnostic_v2"
_MAX_COMPLETE_CAPTURES = 8


@dataclass(slots=True)
class _CaptureRecord:
    capture_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_ns: int = field(default_factory=time.time_ns)
    completed_ns: int | None = None
    source_shapes: list[tuple[int, ...]] | None = None
    target_shapes: list[tuple[int, ...]] | None = None
    handoff_sigma: float | None = None
    low_last_sigma: float | None = None
    first_high_sigma: float | None = None
    low_model_calls: int = 0
    low_last_actual_hint: bool | None = None
    first_high_actual_hint: bool | None = None
    low_last_model_clean_video: torch.Tensor | None = None
    exact_probe_clean_video: torch.Tensor | None = None
    learned_transfer_clean_video: torch.Tensor | None = None
    first_high_clean_video: torch.Tensor | None = None
    final_high_video: torch.Tensor | None = None
    first_high_model_raw_video: torch.Tensor | None = None
    first_high_pre_guidance_video: torch.Tensor | None = None
    first_high_direction_only_video: torch.Tensor | None = None
    first_high_guidance_reference_video: torch.Tensor | None = None
    first_high_input_state_rms: float | None = None
    first_high_contract: dict[str, Any] | None = None
    first_high_guidance: dict[str, Any] | None = None

    def missing(self) -> list[str]:
        required = {
            "low_last_model_clean": self.low_last_model_clean_video,
            "exact_probe_clean": self.exact_probe_clean_video,
            "learned_transfer_clean": self.learned_transfer_clean_video,
            "first_high_clean": self.first_high_clean_video,
            "final_high": self.final_high_video,
            "first_high_model_raw": self.first_high_model_raw_video,
            "first_high_pre_guidance": self.first_high_pre_guidance_video,
            "first_high_direction_only": self.first_high_direction_only_video,
            "first_high_guidance_reference": self.first_high_guidance_reference_video,
        }
        return [name for name, value in required.items() if value is None]

    def move_to_cpu(self) -> None:
        for name in (
            "low_last_model_clean_video",
            "exact_probe_clean_video",
            "learned_transfer_clean_video",
            "first_high_clean_video",
            "final_high_video",
            "first_high_model_raw_video",
            "first_high_pre_guidance_video",
            "first_high_direction_only_video",
            "first_high_guidance_reference_video",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.detach().to(device="cpu").contiguous().clone())


class _DiagnosticState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tls = threading.local()
        self.complete: deque[_CaptureRecord] = deque(maxlen=_MAX_COMPLETE_CAPTURES)


def _state() -> _DiagnosticState:
    state = getattr(_runtime, _STATE_ATTR, None)
    if state is None:
        state = _DiagnosticState()
        setattr(_runtime, _STATE_ATTR, state)
    return state


def _active_record() -> _CaptureRecord | None:
    return getattr(_state().tls, "record", None)


def _active_stage() -> str | None:
    value = getattr(_state().tls, "flow_stage", None)
    return str(value) if value is not None else None


def _copy_video_from_packed(value: torch.Tensor, shapes: list[tuple[int, ...]]) -> torch.Tensor:
    video, _audio = unpack_streams(value, shapes)
    return video.detach().clone()


def _normalize_shapes(shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return [tuple(int(dim) for dim in shape) for shape in shapes]


def _eligible(config: Any) -> bool:
    return (
        isinstance(config, ProgressiveTargetInputConfig)
        and config.transfer_mode == "learned_3d"
        and config.exact_prefix_mode == "fallback"
    )


def _capture_actual_hint(model_options: dict[str, Any] | None) -> bool | None:
    transformer = (model_options or {}).get("transformer_options") or {}
    value = transformer.get(_runtime.SPECTRUM_ACTUAL_KEY)
    return bool(value) if isinstance(value, bool) else None


def _tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().to(device="cpu", dtype=torch.float64).item())


def _delta_rms(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return _tensor_rms(right.float() - left.float())


def _first_high_contract_snapshot(model_options: dict[str, Any] | None) -> dict[str, Any]:
    transformer = (model_options or {}).get("transformer_options") or {}
    refinement = transformer.get("h3_refinement")
    if isinstance(refinement, dict):
        refinement = {
            key: refinement.get(key)
            for key in ("api", "active", "min_actual_prefix_steps", "sigma_reference", "source")
            if key in refinement
        }
    else:
        refinement = None
    return {
        "flow_stage": transformer.get(_runtime.FLOW_STAGE_KEY),
        "h3_refinement": refinement,
        "spectrum_actual_hint": transformer.get(_runtime.SPECTRUM_ACTUAL_KEY),
        "spectrum_phase_hint": transformer.get(_runtime.SPECTRUM_PHASE_KEY),
        "spectrum_outer_step_hint": transformer.get(_runtime.SPECTRUM_OUTER_STEP_KEY),
        "exact_prefix_bridge_present": _runtime.EXACT_PREFIX_BRIDGE_KEY in transformer,
        "mixed_grid_present": _runtime.MIXED_GRID_KEY in transformer,
        "target_sparse_present": _runtime.TARGET_SPARSE_CONTRACT_KEY in transformer,
        "probe_context_present": _runtime.PROBE_CONTEXT_KEY in transformer,
    }


class _RawExecutorCaptureProxy:
    """Transparent executor proxy that snapshots the model result before Flow post-processing."""

    def __init__(self, executor):
        self._executor = executor
        self.class_obj = executor.class_obj

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def __call__(self, x, timestep, model_options=None, seed=None):
        result = self._executor(x, timestep, model_options, seed)
        record = _active_record()
        transformer = (model_options or {}).get("transformer_options") or {}
        stage = str(transformer.get(_runtime.FLOW_STAGE_KEY, "single"))
        if record is None or stage != "high" or record.first_high_model_raw_video is not None:
            return result
        inner_model = getattr(self.class_obj, "inner_model", None)
        shapes = getattr(inner_model, "latent_shapes", None)
        if not isinstance(shapes, list) or len(shapes) != 2:
            return result
        normalized = _normalize_shapes(shapes)
        record.first_high_model_raw_video = _copy_video_from_packed(result, normalized)
        input_video = _copy_video_from_packed(x, normalized)
        record.first_high_input_state_rms = _tensor_rms(input_video)
        record.first_high_contract = _first_high_contract_snapshot(model_options)
        return result


def _flow_predict_capture_wrapper(executor, x, timestep, model_options=None, seed=None):
    transformer = (model_options or {}).get("transformer_options") or {}
    stage = str(transformer.get(_runtime.FLOW_STAGE_KEY, "single"))
    actual_hint = _capture_actual_hint(model_options)
    diagnostic_state = _state()
    previous_stage = getattr(diagnostic_state.tls, "flow_stage", None)
    diagnostic_state.tls.flow_stage = stage
    try:
        observed_executor = _RawExecutorCaptureProxy(executor) if _active_record() is not None else executor
        result = _ORIGINAL_FLOW_PREDICT(observed_executor, x, timestep, model_options, seed)
    finally:
        diagnostic_state.tls.flow_stage = previous_stage

    record = _active_record()
    if record is None or stage not in {"low", "high"}:
        return result
    inner_model = getattr(executor.class_obj, "inner_model", None)
    shapes = getattr(inner_model, "latent_shapes", None)
    if not isinstance(shapes, list) or len(shapes) != 2:
        return result
    shapes = _normalize_shapes(shapes)
    sigma = float(timestep.detach().reshape(-1)[0].item())
    video = _copy_video_from_packed(result, shapes)
    if stage == "low":
        record.low_last_model_clean_video = video
        record.low_last_sigma = sigma
        record.low_last_actual_hint = actual_hint
        record.low_model_calls += 1
        record.source_shapes = shapes
    elif record.first_high_clean_video is None:
        record.first_high_clean_video = video
        record.first_high_sigma = sigma
        record.first_high_actual_hint = actual_hint
        record.target_shapes = shapes
    return result


setattr(_flow_predict_capture_wrapper, _WRAPPER_MARK, True)


def _apply_guidance_capture_wrapper(
    high_x0: torch.Tensor,
    *,
    run,
    coordinate: float,
    config,
    state,
    high_state: torch.Tensor | None = None,
    sigma: float | None = None,
):
    result = _ORIGINAL_APPLY_GUIDANCE(
        high_x0,
        run=run,
        coordinate=coordinate,
        config=config,
        state=state,
        high_state=high_state,
        sigma=sigma,
    )
    record = _active_record()
    if record is None or _active_stage() != "high" or record.first_high_pre_guidance_video is not None:
        return result

    direction_config = replace(
        config,
        mode="direction",
        acceleration_weight=0.0,
        temporal_weight=0.0,
        consistency_weight=0.0,
    )
    direction_only = _ORIGINAL_APPLY_GUIDANCE(
        high_x0,
        run=run,
        coordinate=coordinate,
        config=direction_config,
        state=GuidanceState(),
        high_state=high_state,
        sigma=sigma,
    )
    source_ref, reference_coordinate, reference_clamped = time_matched_reference_info(run, float(coordinate))
    source_ref = source_ref.to(device=high_x0.device, dtype=high_x0.dtype)
    reference = resize_video(source_ref, high_x0.shape[-2], high_x0.shape[-1], mode=config.transfer_mode)

    record.first_high_pre_guidance_video = high_x0.detach().clone()
    record.first_high_direction_only_video = direction_only.detach().clone()
    record.first_high_guidance_reference_video = reference.detach().clone()
    record.first_high_guidance = {
        "mode": str(config.mode),
        "direction_weight": float(config.direction_weight),
        "temporal_weight": float(config.temporal_weight),
        "acceleration_weight": float(config.acceleration_weight),
        "consistency_weight": float(config.consistency_weight),
        "cutoff": float(config.cutoff),
        "max_correction_rms_ratio": float(config.max_correction_rms_ratio),
        "coordinate": float(coordinate),
        "reference_coordinate": float(reference_coordinate),
        "reference_clamped": bool(reference_clamped),
        "schedule": state.last_schedule,
        "correction_rms": state.last_correction_rms,
        "baseline_rms": state.last_baseline_rms,
        "correction_rms_ratio": state.last_correction_rms_ratio,
        "clamp_scale": state.last_clamp_scale,
        "direction_rms_ratio": state.last_direction_rms_ratio,
        "temporal_rms_ratio": state.last_temporal_rms_ratio,
        "temporal_confidence_mean": state.last_temporal_confidence_mean,
        "temporal_valid_fraction": state.last_temporal_valid_fraction,
        "temporal_disocclusion_fraction": state.last_temporal_disocclusion_fraction,
        "temporal_similarity_mean": state.last_temporal_similarity_mean,
        "temporal_margin_mean": state.last_temporal_margin_mean,
        "temporal_cache_hit": state.last_temporal_cache_hit,
    }
    return result


setattr(_apply_guidance_capture_wrapper, _WRAPPER_MARK, True)


def _build_handoff_state_capture_wrapper(
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
    record = _active_record()
    if record is not None and transfer_mode == "learned_3d":
        record.source_shapes = _normalize_shapes(source_shapes)
        record.handoff_sigma = float(sigma)
        record.exact_probe_clean_video = _copy_video_from_packed(source_x0_packed, record.source_shapes)

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

    if record is not None and transfer_mode == "learned_3d":
        normalized_target_shapes = _normalize_shapes(target_shapes)
        target_video, _target_audio = unpack_streams(target_packed, normalized_target_shapes)
        noise = deterministic_video_noise(
            tuple(int(dim) for dim in target_video.shape),
            seed=int(seed),
            device=target_video.device,
            dtype=target_video.dtype,
        )
        learned_clean = recover_conditional_clean_for_diagnostics(target_video, noise, sigma=float(sigma))
        record.learned_transfer_clean_video = learned_clean.detach().clone()
        record.target_shapes = normalized_target_shapes
    return target_packed, target_shapes


setattr(_build_handoff_state_capture_wrapper, _WRAPPER_MARK, True)


def _run_progressive_capture_wrapper(
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
    if not _eligible(config):
        return _ORIGINAL_RUN_PROGRESSIVE(
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

    state = _state()
    previous = getattr(state.tls, "record", None)
    record = _CaptureRecord()
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
        if record.target_shapes is not None and record.learned_transfer_clean_video is not None:
            record.final_high_video = _copy_video_from_packed(result, record.target_shapes)
        missing = record.missing()
        if not missing:
            record.completed_ns = time.time_ns()
            record.move_to_cpu()
            with state.lock:
                state.complete.append(record)
        return result
    finally:
        state.tls.record = previous


setattr(_run_progressive_capture_wrapper, _WRAPPER_MARK, True)

_ORIGINAL_FLOW_PREDICT = _comfy_compat.flow_predict_wrapper
_ORIGINAL_APPLY_GUIDANCE = _runtime.apply_guidance
_ORIGINAL_BUILD_HANDOFF_STATE = _runtime.build_handoff_state
_ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive


def install_handoff_checkpoint_diagnostic() -> None:
    """Install bounded observers into the already-loaded Flow runtime."""

    global _ORIGINAL_FLOW_PREDICT, _ORIGINAL_APPLY_GUIDANCE, _ORIGINAL_BUILD_HANDOFF_STATE, _ORIGINAL_RUN_PROGRESSIVE

    if not getattr(_comfy_compat.flow_predict_wrapper, _WRAPPER_MARK, False):
        _ORIGINAL_FLOW_PREDICT = _comfy_compat.flow_predict_wrapper
        _comfy_compat.flow_predict_wrapper = _flow_predict_capture_wrapper
        _runtime.flow_predict_wrapper = _flow_predict_capture_wrapper
    if not getattr(_runtime.apply_guidance, _WRAPPER_MARK, False):
        _ORIGINAL_APPLY_GUIDANCE = _runtime.apply_guidance
        _runtime.apply_guidance = _apply_guidance_capture_wrapper
    if not getattr(_runtime.build_handoff_state, _WRAPPER_MARK, False):
        _ORIGINAL_BUILD_HANDOFF_STATE = _runtime.build_handoff_state
        _runtime.build_handoff_state = _build_handoff_state_capture_wrapper
    if not getattr(_runtime._run_progressive, _WRAPPER_MARK, False):
        _ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive
        _runtime._run_progressive = _run_progressive_capture_wrapper


def _pop_latest_complete() -> _CaptureRecord:
    state = _state()
    with state.lock:
        if not state.complete:
            raise RuntimeError(
                "no complete learned_3d Target Input first-high checkpoint capture is available; "
                "run the sampler with direction+temporal guidance after enabling this diagnostic overlay"
            )
        return state.complete.pop()


def _validate_model_video_latent_contract(model: Any) -> None:
    base = getattr(model, "model", None)
    if base is None or base.__class__.__name__ != "MiniMaxH3":
        raise TypeError("checkpoint diagnostic requires the same native MiniMax H3 MODEL used by the sampler")
    latent_format = getattr(base, "latent_format", None)
    if latent_format is None or latent_format.__class__.__name__ != "MiniMaxH3AV":
        raise TypeError("checkpoint diagnostic requires ComfyUI's MiniMaxH3AV latent format")
    if float(getattr(latent_format, "scale_factor", float("nan"))) != 1.0:
        raise RuntimeError("MiniMaxH3 video latent scaling changed; diagnostic conversion must be re-audited")


def _latent(video: torch.Tensor | None, name: str) -> dict[str, torch.Tensor]:
    if video is None:
        raise RuntimeError(f"checkpoint capture is missing {name}")
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise RuntimeError(f"checkpoint {name} is not a Bx24xTxHxW MiniMax-H3 video latent")
    return {"samples": video}


class H3HandoffCheckpointDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Extract decode-ready checkpoints from the latest standard learned_3d Target Input progressive handoff, "
        "including a first-high split at raw model output, pre-guidance, direction-only counterfactual, and the "
        "existing full-guidance output. It adds no H3 evaluation or latent-upscaler call. Connect the sampled "
        "LATENT as trigger so extraction runs after sampling."
    )
    RETURN_TYPES = (
        "LATENT",
        "LATENT",
        "LATENT",
        "LATENT",
        "LATENT",
        "STRING",
        "LATENT",
        "LATENT",
        "LATENT",
        "LATENT",
    )
    RETURN_NAMES = (
        "low_last_model_clean",
        "exact_probe_clean",
        "learned_transfer_clean",
        "first_high_clean",
        "final_high",
        "report",
        "first_high_model_raw",
        "first_high_pre_guidance",
        "first_high_direction_only",
        "first_high_guidance_reference",
    )
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "trigger": ("LATENT",)}}

    def extract(self, model, trigger):
        _ = trigger
        _validate_model_video_latent_contract(model)
        record = _pop_latest_complete()
        report = {
            "capture_id": record.capture_id,
            "capture_started_ns": record.started_ns,
            "capture_completed_ns": record.completed_ns,
            "path": "target_input_fallback_config__unprotected_progressive_learned_3d",
            "source_shapes": record.source_shapes,
            "target_shapes": record.target_shapes,
            "handoff_sigma": record.handoff_sigma,
            "low_last_sigma": record.low_last_sigma,
            "first_high_sigma": record.first_high_sigma,
            "low_model_calls": record.low_model_calls,
            "low_last_actual_hint": record.low_last_actual_hint,
            "first_high_actual_hint": record.first_high_actual_hint,
            "first_high_input_state_rms": record.first_high_input_state_rms,
            "first_high_contract": record.first_high_contract,
            "first_high_guidance": record.first_high_guidance,
            "first_high_delta_rms": {
                "model_raw_to_pre_guidance": _delta_rms(
                    record.first_high_model_raw_video, record.first_high_pre_guidance_video
                ),
                "pre_guidance_to_direction_only": _delta_rms(
                    record.first_high_pre_guidance_video, record.first_high_direction_only_video
                ),
                "direction_only_to_full_guidance": _delta_rms(
                    record.first_high_direction_only_video, record.first_high_clean_video
                ),
                "pre_guidance_to_full_guidance": _delta_rms(
                    record.first_high_pre_guidance_video, record.first_high_clean_video
                ),
            },
            "no_extra_h3_evaluations": True,
            "no_extra_upscaler_calls": True,
            "direction_only_counterfactual_adds_h3_evaluations": False,
            "learned_transfer_recovery": "inverse_conditional_renoise_from_existing_target_state",
            "video_latent_domain": "MiniMaxH3AV scale_factor=1.0; internal/external video slice identical",
            "checkpoint_meaning": {
                "low_last_model_clean": "last low-stage model clean prediction before the exact handoff probe",
                "exact_probe_clean": "exact one-call low-grid probe clean prediction used for transfer",
                "learned_transfer_clean": "learned_3d clean target-grid output recovered before high-stage denoising",
                "first_high_model_raw": "first high-stage underlying model output before Flow suffix bridge/guidance",
                "first_high_pre_guidance": "first high-stage output immediately before Flow guidance",
                "first_high_direction_only": "pre-guidance tensor with direction-only Flow guidance",
                "first_high_clean": "actual first high-stage output after configured Flow guidance",
                "first_high_guidance_reference": "time-matched low-stage trajectory reference resized to the high grid",
                "final_high": "final target-grid sampler endpoint returned to the caller",
            },
        }
        return (
            _latent(record.low_last_model_clean_video, "low_last_model_clean"),
            _latent(record.exact_probe_clean_video, "exact_probe_clean"),
            _latent(record.learned_transfer_clean_video, "learned_transfer_clean"),
            _latent(record.first_high_clean_video, "first_high_clean"),
            _latent(record.final_high_video, "final_high"),
            json.dumps(report, indent=2, sort_keys=True),
            _latent(record.first_high_model_raw_video, "first_high_model_raw"),
            _latent(record.first_high_pre_guidance_video, "first_high_pre_guidance"),
            _latent(record.first_high_direction_only_video, "first_high_direction_only"),
            _latent(record.first_high_guidance_reference_video, "first_high_guidance_reference"),
        )


install_handoff_checkpoint_diagnostic()

NODE_CLASS_MAPPINGS = {"H3HandoffCheckpointDiagnostic": H3HandoffCheckpointDiagnostic}
NODE_DISPLAY_NAME_MAPPINGS = {"H3HandoffCheckpointDiagnostic": "MiniMax H3 Progressive Handoff Checkpoint Diagnostic"}
