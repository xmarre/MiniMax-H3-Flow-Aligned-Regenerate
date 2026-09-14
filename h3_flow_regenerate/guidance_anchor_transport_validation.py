"""Validation overlay for learned-3D-aware progressive Flow guidance.

The learned 3D handoff establishes a target-grid clean representation at the
handoff coordinate. The production guidance path historically discarded that
representation and independently bicubic-resized absolute low-grid trajectory
states. This validation overlay instead transports trajectory *deltas* around
the learned target anchor:

    target_ref(c) = target_anchor + resize(source_ref(c) - source_anchor)

At the handoff coordinate the transported reference is therefore exactly the
learned target anchor. No H3 evaluation and no additional learned-upscaler call
is added. This module is intentionally stacked on the checkpoint diagnostic PR
so decoded media can validate the representation contract before production
extraction.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from . import runtime as _runtime
from .contracts import TrajectoryRun, TrajectorySample
from .geometry import geometry_from_video, resize_video, unpack_streams
from .guidance import time_matched_reference_info
from .handoff import ProgressiveTargetInputConfig, deterministic_video_noise
from .seam_diagnostics import recover_conditional_clean_for_diagnostics

_STATE_ATTR = "_h3_guidance_anchor_transport_validation_state_v1"
_WRAPPER_MARK = "_h3_guidance_anchor_transport_validation_v1"
_MAX_COMPLETE = 8
_MATCH_ATOL = 1e-5


@dataclass(slots=True)
class _ValidationRecord:
    started_ns: int = field(default_factory=time.time_ns)
    completed_ns: int | None = None
    enabled: bool = False
    source_anchor: torch.Tensor | None = None
    target_anchor: torch.Tensor | None = None
    target_shapes: list[tuple[int, ...]] | None = None
    handoff_sigma: float | None = None
    anchor_coordinate: float | None = None
    anchor_source_match_max_abs: float | None = None
    anchor_source_match_rms: float | None = None
    transformed_run: TrajectoryRun | None = None
    first_reference: torch.Tensor | None = None
    first_reference_to_target_max_abs: float | None = None
    first_reference_to_target_rms: float | None = None
    last_pre_guidance: torch.Tensor | None = None
    last_full_guidance: torch.Tensor | None = None
    guidance_calls: int = 0
    source_sample_count: int = 0
    target_sample_count: int = 0

    def ready(self) -> bool:
        return (
            self.enabled
            and self.source_anchor is not None
            and self.target_anchor is not None
            and self.first_reference is not None
            and self.last_pre_guidance is not None
            and self.last_full_guidance is not None
            and self.guidance_calls > 0
        )

    def move_outputs_to_cpu(self) -> None:
        for name in (
            "source_anchor",
            "target_anchor",
            "first_reference",
            "last_pre_guidance",
            "last_full_guidance",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.detach().to(device="cpu").contiguous().clone())
        self.transformed_run = None


class _ValidationState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.tls = threading.local()
        self.complete: deque[_ValidationRecord] = deque(maxlen=_MAX_COMPLETE)


def _state() -> _ValidationState:
    state = getattr(_runtime, _STATE_ATTR, None)
    if state is None:
        state = _ValidationState()
        setattr(_runtime, _STATE_ATTR, state)
    return state


def _active() -> _ValidationRecord | None:
    return getattr(_state().tls, "record", None)


def _normalize_shapes(shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return [tuple(int(dim) for dim in shape) for shape in shapes]


def _tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().to(device="cpu", dtype=torch.float64).item())


def _eligible(config: Any, binding: Any) -> bool:
    guidance = getattr(binding, "guidance", None)
    return (
        isinstance(config, ProgressiveTargetInputConfig)
        and config.transfer_mode == "learned_3d"
        and config.exact_prefix_mode == "fallback"
        and guidance is not None
        and getattr(guidance, "mode", "off") != "off"
    )


def _find_exact_probe(run: TrajectoryRun) -> TrajectorySample:
    probes = [sample for sample in run.samples if sample.provenance == "actual" and sample.phase == "handoff_probe"]
    if len(probes) != 1:
        raise RuntimeError(
            "learned-anchor guidance validation requires exactly one actual handoff_probe trajectory sample; "
            f"observed {len(probes)}"
        )
    return probes[0]


def _transport_run_to_target(
    run: TrajectoryRun,
    *,
    captured_source_anchor: torch.Tensor,
    target_anchor: torch.Tensor,
) -> tuple[TrajectoryRun, dict[str, float | int]]:
    if not run.complete:
        raise RuntimeError("learned-anchor guidance validation requires a committed low-stage trajectory")
    if target_anchor.ndim != 5 or int(target_anchor.shape[1]) != 24:
        raise ValueError("learned target anchor must be Bx24xTxHxW")
    probe = _find_exact_probe(run)
    source_anchor = probe.video_x0.to(device=target_anchor.device, dtype=target_anchor.dtype)
    captured = captured_source_anchor.to(device=target_anchor.device, dtype=target_anchor.dtype)
    if source_anchor.shape != captured.shape:
        raise RuntimeError("captured exact-probe anchor and committed handoff_probe geometry differ")
    mismatch = source_anchor.float() - captured.float()
    match_max_abs = float(mismatch.abs().max().detach().to(device="cpu", dtype=torch.float64).item())
    match_rms = _tensor_rms(mismatch)
    if match_max_abs > _MATCH_ATOL:
        raise RuntimeError(
            "committed handoff_probe does not match the clean state used by learned_3d transfer; "
            f"max_abs={match_max_abs:.6g} exceeds {_MATCH_ATOL:.6g}"
        )
    if source_anchor.shape[:3] != target_anchor.shape[:3]:
        raise RuntimeError("source and target guidance anchors differ in batch/channel/time geometry")

    target_h, target_w = map(int, target_anchor.shape[-2:])
    mapped_samples: list[TrajectorySample] = []
    for sample in run.samples:
        source = sample.video_x0.to(device=target_anchor.device, dtype=target_anchor.dtype)
        if source.shape != source_anchor.shape:
            raise RuntimeError("trajectory sample geometry differs from the learned guidance source anchor")
        if sample is probe:
            mapped = target_anchor.detach().clone()
        else:
            delta = source - source_anchor
            mapped = target_anchor + resize_video(delta, target_h, target_w, mode="bicubic")
        if not bool(torch.isfinite(mapped).all().item()):
            raise RuntimeError("learned-anchor trajectory transport produced NaN or Inf")
        mapped_samples.append(replace(sample, video_x0=mapped))

    transformed = replace(
        run,
        geometry=geometry_from_video(target_anchor),
        layout_signature=(
            f"learned_anchor_transport_v1|source={tuple(source_anchor.shape)}|target={tuple(target_anchor.shape)}"
        ),
        samples=tuple(mapped_samples),
    )
    return transformed, {
        "anchor_coordinate": float(probe.coordinate),
        "anchor_source_match_max_abs": match_max_abs,
        "anchor_source_match_rms": match_rms,
        "source_sample_count": len(run.samples),
        "target_sample_count": len(mapped_samples),
    }


def _build_handoff_state_anchor_wrapper(
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

    normalized_source = _normalize_shapes(source_shapes)
    normalized_target = _normalize_shapes(target_shapes)
    source_video, _source_audio = unpack_streams(source_x0_packed, normalized_source)
    target_video, _target_audio = unpack_streams(target_packed, normalized_target)
    noise = deterministic_video_noise(
        tuple(int(dim) for dim in target_video.shape),
        seed=int(seed),
        device=target_video.device,
        dtype=target_video.dtype,
    )
    target_clean = recover_conditional_clean_for_diagnostics(target_video, noise, sigma=float(sigma))
    record.source_anchor = source_video.detach().clone()
    record.target_anchor = target_clean.detach().clone()
    record.target_shapes = normalized_target
    record.handoff_sigma = float(sigma)
    return target_packed, target_shapes


setattr(_build_handoff_state_anchor_wrapper, _WRAPPER_MARK, True)


def _apply_guidance_anchor_wrapper(
    high_x0: torch.Tensor,
    *,
    run: TrajectoryRun,
    coordinate: float,
    config,
    state,
    high_state: torch.Tensor | None = None,
    sigma: float | None = None,
):
    record = _active()
    if record is None or not record.enabled:
        return _ORIGINAL_APPLY_GUIDANCE(
            high_x0,
            run=run,
            coordinate=coordinate,
            config=config,
            state=state,
            high_state=high_state,
            sigma=sigma,
        )
    if record.source_anchor is None or record.target_anchor is None:
        raise RuntimeError(
            "learned-anchor guidance reached the high stage before the learned transfer anchor was captured"
        )
    if tuple(high_x0.shape) != tuple(record.target_anchor.shape):
        raise RuntimeError(
            "learned-anchor guidance high prediction geometry differs from learned target anchor: "
            f"{tuple(high_x0.shape)} vs {tuple(record.target_anchor.shape)}"
        )

    if record.transformed_run is None:
        transformed, facts = _transport_run_to_target(
            run,
            captured_source_anchor=record.source_anchor,
            target_anchor=record.target_anchor,
        )
        record.transformed_run = transformed
        record.anchor_coordinate = float(facts["anchor_coordinate"])
        record.anchor_source_match_max_abs = float(facts["anchor_source_match_max_abs"])
        record.anchor_source_match_rms = float(facts["anchor_source_match_rms"])
        record.source_sample_count = int(facts["source_sample_count"])
        record.target_sample_count = int(facts["target_sample_count"])

    transformed_run = record.transformed_run
    assert transformed_run is not None
    if (
        record.guidance_calls == 0
        and record.anchor_coordinate is not None
        and abs(float(coordinate) - record.anchor_coordinate) > 1e-6
    ):
        raise RuntimeError(
            "first high-stage guidance coordinate does not match the exact handoff probe anchor: "
            f"high={float(coordinate):.9f} anchor={record.anchor_coordinate:.9f}"
        )

    reference, _resolved, _clamped = time_matched_reference_info(transformed_run, float(coordinate))
    reference = reference.to(device=high_x0.device, dtype=high_x0.dtype)
    if record.guidance_calls == 0:
        target = record.target_anchor.to(device=high_x0.device, dtype=high_x0.dtype)
        delta = reference.float() - target.float()
        max_abs = float(delta.abs().max().detach().to(device="cpu", dtype=torch.float64).item())
        rms = _tensor_rms(delta)
        if max_abs > _MATCH_ATOL:
            raise RuntimeError(
                "learned-anchor first high guidance reference is not the learned transfer target at the handoff; "
                f"max_abs={max_abs:.6g}"
            )
        record.first_reference = reference.detach().clone()
        record.first_reference_to_target_max_abs = max_abs
        record.first_reference_to_target_rms = rms

    record.last_pre_guidance = high_x0.detach().clone()
    result = _ORIGINAL_APPLY_GUIDANCE(
        high_x0,
        run=transformed_run,
        coordinate=coordinate,
        config=config,
        state=state,
        high_state=high_state,
        sigma=sigma,
    )
    record.last_full_guidance = result.detach().clone()
    record.guidance_calls += 1
    return result


setattr(_apply_guidance_anchor_wrapper, _WRAPPER_MARK, True)


def _run_progressive_anchor_wrapper(
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
    enabled = _eligible(config, binding)
    state = _state()
    previous = getattr(state.tls, "record", None)
    record = _ValidationRecord(enabled=enabled)
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
        if record.ready():
            record.completed_ns = time.time_ns()
            record.move_outputs_to_cpu()
            with state.lock:
                state.complete.append(record)
        return result
    finally:
        state.tls.record = previous


setattr(_run_progressive_anchor_wrapper, _WRAPPER_MARK, True)

_ORIGINAL_BUILD_HANDOFF_STATE = _runtime.build_handoff_state
_ORIGINAL_APPLY_GUIDANCE = _runtime.apply_guidance
_ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive


def install_guidance_anchor_transport_validation() -> None:
    global _ORIGINAL_BUILD_HANDOFF_STATE, _ORIGINAL_APPLY_GUIDANCE, _ORIGINAL_RUN_PROGRESSIVE

    if not getattr(_runtime.build_handoff_state, _WRAPPER_MARK, False):
        _ORIGINAL_BUILD_HANDOFF_STATE = _runtime.build_handoff_state
        _runtime.build_handoff_state = _build_handoff_state_anchor_wrapper
    if not getattr(_runtime.apply_guidance, _WRAPPER_MARK, False):
        _ORIGINAL_APPLY_GUIDANCE = _runtime.apply_guidance
        _runtime.apply_guidance = _apply_guidance_anchor_wrapper
    if not getattr(_runtime._run_progressive, _WRAPPER_MARK, False):
        _ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive
        _runtime._run_progressive = _run_progressive_anchor_wrapper


def _pop_latest() -> _ValidationRecord:
    state = _state()
    with state.lock:
        if not state.complete:
            raise RuntimeError(
                "no complete learned-anchor guidance validation capture is available; run an unprotected learned_3d "
                "Target Input progressive sample with Flow guidance enabled"
            )
        return state.complete.pop()


def _latent(video: torch.Tensor | None, name: str) -> dict[str, torch.Tensor]:
    if video is None:
        raise RuntimeError(f"learned-anchor validation capture is missing {name}")
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise RuntimeError(f"{name} is not a Bx24xTxHxW MiniMax-H3 video latent")
    return {"samples": video}


class H3GuidanceAnchorTransportValidation:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Validation-only learned_3d guidance representation transport. It anchors the low-stage trajectory in the "
        "existing learned target-grid handoff representation, adds no H3/upscaler evaluation, and exposes the "
        "first transported guidance reference plus the final high-stage pre/full-guidance checkpoints."
    )
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = (
        "guidance_anchor_target",
        "first_high_anchor_reference",
        "last_high_pre_guidance",
        "last_high_full_guidance",
        "report",
    )
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("LATENT",)}}

    def extract(self, trigger):
        _ = trigger
        record = _pop_latest()
        report = {
            "schema": "learned_anchor_guidance_validation_v1",
            "started_ns": record.started_ns,
            "completed_ns": record.completed_ns,
            "handoff_sigma": record.handoff_sigma,
            "anchor_coordinate": record.anchor_coordinate,
            "anchor_source_match_max_abs": record.anchor_source_match_max_abs,
            "anchor_source_match_rms": record.anchor_source_match_rms,
            "first_reference_to_target_max_abs": record.first_reference_to_target_max_abs,
            "first_reference_to_target_rms": record.first_reference_to_target_rms,
            "guidance_calls": record.guidance_calls,
            "source_sample_count": record.source_sample_count,
            "target_sample_count": record.target_sample_count,
            "transport": "target_anchor + bicubic(source_trajectory - exact_probe_anchor)",
            "first_high_invariant": "transported reference equals learned target anchor at handoff coordinate",
            "no_extra_h3_evaluations": True,
            "no_extra_upscaler_calls": True,
            "production_status": "validation_only_not_for_merge",
        }
        return (
            _latent(record.target_anchor, "guidance_anchor_target"),
            _latent(record.first_reference, "first_high_anchor_reference"),
            _latent(record.last_pre_guidance, "last_high_pre_guidance"),
            _latent(record.last_full_guidance, "last_high_full_guidance"),
            json.dumps(report, indent=2, sort_keys=True),
        )


install_guidance_anchor_transport_validation()

NODE_CLASS_MAPPINGS = {"H3GuidanceAnchorTransportValidation": H3GuidanceAnchorTransportValidation}
NODE_DISPLAY_NAME_MAPPINGS = {"H3GuidanceAnchorTransportValidation": "MiniMax H3 Learned-Anchor Guidance Validation"}
