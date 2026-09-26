"""Low/probe/high scheduler for partitioned exact-prefix continuation.

The released Flow runtime remains untouched.  This module is selected only by the
experimental partitioned node and publishes a new stage contract rather than
reinterpreting the retired Mixed-Grid mode.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from typing import Any

import torch

from .audio_guided_overlap import compare_audio_latent_stages, measure_audio_latent_boundary
from .boundary_content_diagnostics import (
    compare_boundary_content_stages,
    measure_boundary_content_continuity,
    measure_provider_boundary_temporal_calibration,
    measure_provider_boundary_temporal_predictor,
)
from .contracts import H3FlowTrajectory
from .frame_gauge import (
    FRAME_GAUGE_POLICY_VERSION,
    GUIDANCE_REFERENCE_POLICY,
    LEARNED_VIDEO_POLICY,
    estimate_paired_prefix_translation,
    translate_video_cells,
)
from .geometry import (
    pack_streams,
    resize_spatial_5d_h3_patch_lattice,
    resize_video,
    unpack_streams,
)
from .guidance import RegisteredGuidanceReference, time_matched_reference_info
from .handoff import (
    CleanVideoPostprocessResult,
    ProgressiveTargetInputConfig,
    build_handoff_state,
    deterministic_video_noise,
)
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_KEY,
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API,
    normalize_audio_handoff_source,
    normalize_audio_position_domain,
    normalize_av_handoff_source,
    normalize_guidance_trajectory_source,
    normalize_low_probe_execution_source,
    normalize_prefix_transformer_context,
    normalize_vdn_linear_diagnostic,
    resolve_partitioned_audio_guided_overlap_mode,
    resolve_partitioned_audio_guided_overlap_ticks,
)
from .partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStageRuntime,
    build_partitioned_stage_plan,
    tensor_sha256,
)
from .partitioned_transformer import VDN_PARTITIONED_SEQUENCE_API
from .representation_bridge import (
    apply_suffix_representation_bridge,
    disabled_suffix_representation_bridge_metrics,
)
from .residual_evidence import export_residual_geometry_evidence
from .residual_geometry import (
    RESIDUAL_GEOMETRY_POLICY_VERSION,
    measure_residual_geometry,
    normalize_residual_geometry_mode,
)
from .runtime import (
    FLOW_STAGE_KEY,
    PROBE_CONTEXT_KEY,
    _begin_capture,
    _conditioning_signature,
    _finish_capture,
    _flow_stage_contract,
    _has_exact_video_protection,
    _high_stage_contract,
    _interop_identity,
    _make_probe_sampler,
    _merge_preserved_noise,
    _noise_argument,
    _process_latent_in,
    _raw_sampler_state,
    _reset_guider_conds,
    _resize_packed_latent_image,
    _resize_packed_mask,
    _validate_progressive_sampler_state,
)
from .seam_diagnostics import (
    measure_exact_prefix_splice,
    measure_translation_trajectory,
    measure_video_boundary,
    project_translation_trajectory_to_grid,
    recover_conditional_clean_for_diagnostics,
)
from .sigma import H3_VIDEO_SHIFT, normalized_coordinate
from .tone_bridge import (
    apply_suffix_dc_bridge,
    disabled_suffix_dc_bridge_metrics,
    map_clean_bridge_to_conditional_state,
)

PARTITIONED_PROGRESSIVE_KEY = "h3_flow_partitioned_progressive_v1"
SOL_RUNTIME_KEY = "sol_h3_runtime_v1"
FRAME_GAUGE_GUIDANCE_SUPPORTED_SAMPLERS = frozenset({"sample_res_multistep"})
FRAME_GAUGE_BOUNDARY_MIN_ERROR_CELLS = 0.125
FRAME_GAUGE_BOUNDARY_MIN_IMPROVEMENT = 0.25
FRAME_GAUGE_BOUNDARY_MIN_RESPONSE = 3.0
PARTITIONED_EXACT_OVERLAP_POLICY = "partitioned_exact_overlap_structural_plus_dc_v1"
FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS = frozenset(
    {
        "boundary_upper45_not_improved",
        "boundary_full_not_improved",
        "boundary_upper45_insufficient_improvement",
        "boundary_full_insufficient_improvement",
    }
)
PARTITIONED_SOL_REQUIRED_METADATA = {
    "api": 1,
    "owner": "comfyui_sol_h3",
    "exact": True,
    "backend": "sol",
    "attention_ownership": "sol",
    "kernel_contract": "sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4",
    "history_policy": "attention_backend_history_v1",
}


class PartitionedPreflightUnsupported(RuntimeError):
    """A condition detected before sampling that must use the exact target fallback."""


def _validate_audio_handoff_shadow_configuration(
    audio_handoff_source: str,
    *,
    prefix_transformer_context: str,
    vdn_linear_diagnostic: str,
    audio_position_domain: str,
    audio_guided_overlap_mode: str,
    audio_guided_overlap_ticks: int,
) -> None:
    source = normalize_audio_handoff_source(audio_handoff_source)
    if source == PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN:
        return
    mismatches = []
    if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        mismatches.append("prefix_transformer_context='exact_target_partitioned'")
    if vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
        mismatches.append("vdn_linear_diagnostic='normal'")
    if audio_position_domain != PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
        mismatches.append("audio_position_domain='source_carrier'")
    if audio_guided_overlap_mode != PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP:
        mismatches.append("audio_guided_overlap_mode='model_timestep_only'")
    if int(audio_guided_overlap_ticks) != 4:
        mismatches.append("audio_guided_overlap_ticks=4")
    if mismatches:
        raise PartitionedPreflightUnsupported(
            "source_carrier_uniform_shadow audio handoff requires " + ", ".join(mismatches)
        )


def _validate_av_handoff_shadow_configuration(
    av_handoff_source: str,
    *,
    audio_handoff_source: str,
    prefix_transformer_context: str,
    vdn_linear_diagnostic: str,
    audio_position_domain: str,
    audio_guided_overlap_mode: str,
    audio_guided_overlap_ticks: int,
) -> None:
    source = normalize_av_handoff_source(av_handoff_source)
    if source == PARTITIONED_AV_HANDOFF_SOURCE_MAIN:
        return
    mismatches = []
    if audio_handoff_source != PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN:
        mismatches.append("audio_handoff_source='main_partitioned'")
    if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        mismatches.append("prefix_transformer_context='exact_target_partitioned'")
    if vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
        mismatches.append("vdn_linear_diagnostic='normal'")
    if audio_position_domain != PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
        mismatches.append("audio_position_domain='source_carrier'")
    if audio_guided_overlap_mode != PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP:
        mismatches.append("audio_guided_overlap_mode='model_timestep_only'")
    if int(audio_guided_overlap_ticks) != 4:
        mismatches.append("audio_guided_overlap_ticks=4")
    if mismatches:
        raise PartitionedPreflightUnsupported(
            "source_carrier_uniform_shadow AV handoff requires " + ", ".join(mismatches)
        )


def _validate_guidance_trajectory_shadow_configuration(
    guidance_trajectory_source: str,
    *,
    av_handoff_source: str,
) -> None:
    source = normalize_guidance_trajectory_source(guidance_trajectory_source)
    if source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN:
        return
    if av_handoff_source != PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
        raise PartitionedPreflightUnsupported(
            "source_carrier_uniform_shadow guidance trajectory requires "
            "av_handoff_source='source_carrier_uniform_shadow'"
        )


def _validate_low_probe_execution_source_configuration(
    low_probe_execution_source: str,
    *,
    prefix_transformer_context: str,
    vdn_linear_diagnostic: str,
    audio_position_domain: str,
    audio_handoff_source: str,
    av_handoff_source: str,
    guidance_trajectory_source: str,
    audio_guided_overlap_mode: str,
    audio_guided_overlap_ticks: int,
) -> None:
    source = normalize_low_probe_execution_source(low_probe_execution_source)
    if source == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW:
        return
    mismatches = []
    if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        mismatches.append("prefix_transformer_context='exact_target_partitioned'")
    if vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
        mismatches.append("vdn_linear_diagnostic='normal'")
    if audio_position_domain != PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
        mismatches.append("audio_position_domain='source_carrier'")
    if audio_handoff_source != PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN:
        mismatches.append("audio_handoff_source='main_partitioned'")
    if av_handoff_source != PARTITIONED_AV_HANDOFF_SOURCE_MAIN:
        mismatches.append("av_handoff_source='main_partitioned'")
    if guidance_trajectory_source != PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN:
        mismatches.append("guidance_trajectory_source='main_exact_partitioned'")
    if audio_guided_overlap_mode not in (
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    ):
        mismatches.append("audio_guided_overlap_mode in {'sampler_mask', 'sampler_mask_exact_timestep'}")
    # Overlap width is a bounded runtime control, not a structural requirement
    # of source-uniform execution. The node-level validator constrains it to 0..16.
    if mismatches:
        raise PartitionedPreflightUnsupported(
            "source_carrier_uniform_only low/probe execution requires " + ", ".join(mismatches)
        )


@contextlib.contextmanager
def _source_uniform_primary_execution_controls(transformer: dict[str, Any]):
    """Make the selected source-uniform pair the only low/probe execution."""

    if not isinstance(transformer, dict):
        raise RuntimeError("source-uniform primary execution requires mutable transformer options")
    keys = (
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
        PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
        PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY,
        PARTITIONED_AV_HANDOFF_SOURCE_KEY,
        PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
        PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
    )
    missing = object()
    previous = {key: transformer.get(key, missing) for key in keys}
    transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] = PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
    for key in keys[1:]:
        transformer.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                transformer.pop(key, None)
            else:
                transformer[key] = value


def _resolve_audio_diagnostic_masks(
    runtime_mask: torch.Tensor,
    exact_mask: torch.Tensor | None,
    target_shapes: list[tuple[int, ...]],
    source_shapes: list[tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep sampler overlap ownership separate from exact-boundary diagnostics."""

    diagnostic_target_mask = runtime_mask if exact_mask is None else exact_mask
    if tuple(diagnostic_target_mask.shape) != tuple(runtime_mask.shape):
        raise RuntimeError("partitioned exact-prefix diagnostic mask geometry drifted from sampler mask")
    diagnostic_low_mask = _resize_packed_mask(
        diagnostic_target_mask,
        target_shapes,
        source_shapes,
    )
    return diagnostic_target_mask, diagnostic_low_mask


def _cuda_allocator_checkpoint(metrics, stage: str) -> None:
    """Record CUDA allocator/headroom state without changing allocation policy."""

    if not torch.cuda.is_available():
        return
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        metrics.event(
            "partitioned_allocator_checkpoint",
            stage=str(stage),
            allocated_mib=torch.cuda.memory_allocated(device) / (1 << 20),
            reserved_mib=torch.cuda.memory_reserved(device) / (1 << 20),
            free_mib=int(free_bytes) / (1 << 20),
            total_mib=int(total_bytes) / (1 << 20),
        )
    except Exception as exc:
        metrics.event(
            "partitioned_allocator_checkpoint",
            stage=str(stage),
            unavailable=True,
            error=type(exc).__name__,
        )


@contextlib.contextmanager
def _source_uniform_audio_shadow_controls(transformer: dict[str, Any]):
    """Temporarily select the already-released uniform source-grid low/probe arm."""

    if not isinstance(transformer, dict):
        raise RuntimeError("audio handoff shadow requires mutable transformer options")
    missing = object()
    previous_prefix = transformer.get(PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY, missing)
    previous_position = transformer.get(PARTITIONED_AUDIO_POSITION_DOMAIN_KEY, missing)
    transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] = PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
    transformer.pop(PARTITIONED_AUDIO_POSITION_DOMAIN_KEY, None)
    try:
        yield
    finally:
        if previous_prefix is missing:
            transformer.pop(PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY, None)
        else:
            transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] = previous_prefix
        if previous_position is missing:
            transformer.pop(PARTITIONED_AUDIO_POSITION_DOMAIN_KEY, None)
        else:
            transformer[PARTITIONED_AUDIO_POSITION_DOMAIN_KEY] = previous_position


def _splice_source_uniform_shadow_audio_state(
    main_state: torch.Tensor,
    shadow_state: torch.Tensor,
    shapes: list[tuple[int, ...]],
    denoise_mask: torch.Tensor,
    metrics,
) -> torch.Tensor:
    """Return main video + shadow audio while proving all other state is unchanged."""

    main_video, main_audio = unpack_streams(main_state, shapes)
    shadow_video, shadow_audio = unpack_streams(shadow_state, shapes)
    _mask_video, audio_mask = unpack_streams(denoise_mask, shapes)
    if tuple(main_video.shape) != tuple(shadow_video.shape) or tuple(main_audio.shape) != tuple(shadow_audio.shape):
        raise RuntimeError("audio handoff shadow state geometry drifted")
    if tuple(audio_mask.shape) != tuple(main_audio.shape):
        raise RuntimeError("audio handoff shadow mask geometry drifted")
    protected = audio_mask == 0
    generated = ~protected
    if not bool(protected.any().item()) or not bool(generated.any().item()):
        raise RuntimeError("audio handoff shadow requires protected and generated audio regions")
    if not torch.equal(main_audio[protected], shadow_audio[protected]):
        raise RuntimeError("source-uniform audio shadow changed the caller-owned protected audio prefix")
    generated_delta = shadow_audio[generated].to(torch.float32) - main_audio[generated].to(torch.float32)
    changed_elements = int(torch.count_nonzero(generated_delta).item())
    if changed_elements <= 0:
        raise RuntimeError("source-uniform audio shadow produced no distinct generated audio state")

    main_video_digest = tensor_sha256(main_video)
    hybrid, hybrid_shapes = pack_streams((main_video, shadow_audio.clone()))
    if list(hybrid_shapes) != list(shapes):
        raise RuntimeError("audio handoff shadow changed packed AV geometry")
    hybrid_video, hybrid_audio = unpack_streams(hybrid, shapes)
    if tensor_sha256(hybrid_video) != main_video_digest or not torch.equal(hybrid_video, main_video):
        raise RuntimeError("audio handoff shadow mutated the main exact-partitioned video state")
    if not torch.equal(hybrid_audio, shadow_audio):
        raise RuntimeError("audio handoff shadow did not preserve the selected shadow audio state")

    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_audio_handoff_shadow_splice",
            source=PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW,
            main_video_digest=main_video_digest,
            main_audio_digest=tensor_sha256(main_audio),
            shadow_audio_digest=tensor_sha256(shadow_audio),
            shadow_video_discarded=True,
            main_video_preserved=True,
            protected_audio_prefix_exact=True,
            generated_audio_changed_elements=changed_elements,
            generated_audio_rms_delta=float(generated_delta.square().mean().sqrt().item()),
            fail_closed=True,
        )
    return hybrid


def _select_source_uniform_shadow_clean_video(
    main_clean: torch.Tensor,
    shadow_clean: torch.Tensor,
    shapes: list[tuple[int, ...]],
    *,
    prefix_t: int,
    exact_prefix_source: torch.Tensor,
    metrics,
) -> torch.Tensor:
    """Select only shadow clean video while keeping exact prefix and main clean audio."""

    main_video, main_audio = unpack_streams(main_clean, shapes)
    shadow_video, shadow_audio = unpack_streams(shadow_clean, shapes)
    if tuple(main_video.shape) != tuple(shadow_video.shape) or tuple(main_audio.shape) != tuple(shadow_audio.shape):
        raise RuntimeError("AV handoff shadow clean-state geometry drifted")
    if not 0 < int(prefix_t) < int(main_video.shape[2]):
        raise RuntimeError("AV handoff shadow requires a non-empty protected video prefix and generated suffix")
    expected_prefix_shape = tuple(main_video[:, :, : int(prefix_t)].shape)
    if tuple(exact_prefix_source.shape) != expected_prefix_shape:
        raise RuntimeError("AV handoff shadow exact source-prefix geometry drifted")

    generated_main = main_video[:, :, int(prefix_t) :]
    generated_shadow = shadow_video[:, :, int(prefix_t) :]
    generated_delta = generated_shadow.to(torch.float32) - generated_main.to(torch.float32)
    changed_elements = int(torch.count_nonzero(generated_delta).item())
    if changed_elements <= 0:
        raise RuntimeError("source-uniform AV shadow produced no distinct generated clean-video state")

    selected_video = shadow_video.clone()
    selected_video[:, :, : int(prefix_t)] = exact_prefix_source.to(selected_video)
    selected, selected_shapes = pack_streams((selected_video, main_audio.clone()))
    if list(selected_shapes) != list(shapes):
        raise RuntimeError("AV handoff shadow changed packed clean-state geometry")
    check_video, check_audio = unpack_streams(selected, shapes)
    if not torch.equal(check_video[:, :, : int(prefix_t)], exact_prefix_source.to(check_video)):
        raise RuntimeError("AV handoff shadow failed to restore the exact source-grid video prefix")
    if not torch.equal(check_video[:, :, int(prefix_t) :], generated_shadow):
        raise RuntimeError("AV handoff shadow failed to preserve the selected generated clean-video suffix")
    if not torch.equal(check_audio, main_audio):
        raise RuntimeError("AV handoff shadow mutated the main clean-audio probe state")

    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_av_handoff_shadow_clean_video",
            source=PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
            main_clean_video_digest=tensor_sha256(main_video),
            shadow_clean_video_digest=tensor_sha256(shadow_video),
            selected_clean_video_digest=tensor_sha256(check_video),
            shadow_clean_audio_discarded=True,
            main_clean_audio_preserved=True,
            exact_source_prefix_restored=True,
            generated_video_changed_elements=changed_elements,
            generated_video_rms_delta=float(generated_delta.square().mean().sqrt().item()),
            fail_closed=True,
        )
    return selected


@contextlib.contextmanager
def _partitioned_stage_contract(guider: Any, plan, metrics):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable transformer options")
    if PARTITIONED_STAGE_KEY in transformer:
        raise RuntimeError("nested partitioned exact-prefix stage is unsupported")
    if "h3_flow_mixed_grid_v1" in transformer or "h3_flow_mixed_grid_attention_measure_v1" in transformer:
        raise RuntimeError("partitioned exact-prefix refuses deprecated Mixed-Grid stage state")

    # This must remain a non-dict leaf. ComfyUI recursively copies nested option
    # dictionaries between model calls, while scalar/object leaves retain their
    # identity. The runtime object therefore owns provider identity for exactly
    # this low/probe sampler-stage lifetime.
    linear_mode = normalize_vdn_linear_diagnostic(
        transformer.get(PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY, PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL)
    )
    prefix_context = normalize_prefix_transformer_context(
        transformer.get(
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        )
    )
    audio_position_domain = normalize_audio_position_domain(
        transformer.get(
            PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
            PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
        )
    )
    transformer[PARTITIONED_STAGE_KEY] = PartitionedStageRuntime(
        plan=plan,
        metrics=metrics,
        vdn_linear_diagnostic=linear_mode,
        prefix_transformer_context=prefix_context,
        audio_position_domain=audio_position_domain,
    )
    try:
        yield
    finally:
        transformer.pop(PARTITIONED_STAGE_KEY, None)


@contextlib.contextmanager
def _source_uniform_audio_shadow_sampler_contract(guider: Any, plan, metrics):
    """Run the shadow sampler at an explicit quiescent OUTER_SAMPLE boundary.

    The shadow is an independent low-grid sampler lifetime, but it must not publish
    h3_flow_stage=low. VDN-H3 deliberately retains low-stage scratch until the
    following probe; #61 runs this shadow after the main probe has already released
    that scratch. Leaving the Flow stage marker absent makes VDN treat this separate
    sampler as an outer-complete boundary and drain/trim its retained CUDA scratch
    before the target-grid high stage starts.
    """

    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("audio handoff shadow requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("audio handoff shadow requires mutable transformer options")
    if FLOW_STAGE_KEY in transformer:
        raise RuntimeError("audio handoff shadow requires a quiescent Flow stage boundary")
    with _partitioned_stage_contract(guider, plan, metrics):
        if FLOW_STAGE_KEY in transformer:
            raise RuntimeError("audio handoff shadow unexpectedly acquired a Flow stage marker")
        yield


@contextlib.contextmanager
def _source_uniform_av_shadow_stage_contract(guider: Any, plan, metrics, stage: str):
    """Publish a real low->probe pair so VDN releases retained shadow scratch at probe."""

    stage = str(stage)
    if stage not in {"low", "probe"}:
        raise ValueError("AV handoff shadow stage must be low or probe")
    with contextlib.ExitStack() as stack:
        stack.enter_context(_flow_stage_contract(guider, stage))
        if stage == "probe":
            stack.enter_context(_high_stage_contract(guider))
        stack.enter_context(_partitioned_stage_contract(guider, plan, metrics))
        transformer = guider.model_options["transformer_options"]
        if transformer.get(FLOW_STAGE_KEY) != stage:
            raise RuntimeError("AV handoff shadow Flow stage marker drifted")
        yield


@contextlib.contextmanager
def _isolated_shadow_trajectory_capture(
    binding,
    guider,
    sampler,
    sigmas,
    latent_shapes,
    *,
    enabled: bool,
):
    """Capture a shadow low+probe trajectory without mutating the shared trajectory handle."""

    if not enabled:
        yield None
        return
    if binding.active_capture is not None:
        raise RuntimeError("shadow guidance trajectory capture requires a quiescent capture boundary")
    if binding.active_guidance_run is not None:
        raise RuntimeError("shadow guidance trajectory capture must complete before target-high guidance")
    main_trajectory = binding.trajectory
    if main_trajectory is None:
        raise RuntimeError("shadow guidance trajectory capture requires the shared H3 flow trajectory")
    if not binding.capture_enabled:
        raise RuntimeError("shadow guidance trajectory capture requires Flow capture to be enabled")
    main_captured_run_id = binding.captured_run_id
    shadow_trajectory = H3FlowTrajectory(storage=main_trajectory.storage, max_runs=1)
    holder: dict[str, Any] = {}
    binding.trajectory = shadow_trajectory
    try:
        _begin_capture(binding, guider, sampler, sigmas, latent_shapes)
        if binding.active_capture is None:
            raise RuntimeError("shadow guidance trajectory capture did not start")
        try:
            yield holder
        except BaseException as exc:
            if binding.active_capture is not None:
                _finish_capture(binding, error=exc)
            raise
        else:
            run = _finish_capture(binding)
            if run is None or not run.complete:
                raise RuntimeError("shadow guidance trajectory capture did not commit")
            if not run.exact_samples():
                raise RuntimeError("shadow guidance trajectory capture produced no exact anchors")
            holder["run"] = run
    finally:
        try:
            if binding.active_capture is not None:
                _finish_capture(
                    binding,
                    error=RuntimeError("shadow guidance trajectory capture escaped its bounded lifetime"),
                )
        finally:
            binding.trajectory = main_trajectory
            binding.captured_run_id = main_captured_run_id


def _validate_partitioned_vdn_compat(
    patcher: Any,
    *,
    required_linear_diagnostic: str = PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
) -> None:
    object_patches = getattr(patcher, "object_patches", None)
    if not isinstance(object_patches, dict):
        raise PartitionedPreflightUnsupported("VDN object-patch ownership is unavailable")
    matched = 0
    for key, owner in object_patches.items():
        if not key.startswith("diffusion_model.blocks.") or not key.endswith(".attn.forward"):
            continue
        if not getattr(owner, "_vdn_forward", False):
            continue
        matched += 1
        if int(getattr(owner, "_vdn_external_sequence_api", 0)) != VDN_PARTITIONED_SEQUENCE_API:
            raise PartitionedPreflightUnsupported(
                f"VDN partitioned external-sequence API {VDN_PARTITIONED_SEQUENCE_API} is unavailable"
            )
        if (
            required_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL
            and int(getattr(owner, "_vdn_partitioned_linear_diagnostic_api", 0))
            != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API
        ):
            raise PartitionedPreflightUnsupported(
                "partitioned VDN linear diagnostic was requested but the installed VDN bridge "
                f"does not publish diagnostic API v{VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API}"
            )
        capability_modes = (
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
        )
        if required_linear_diagnostic in capability_modes:
            supported = tuple(getattr(owner, "_vdn_partitioned_linear_diagnostic_modes", ()))
            if required_linear_diagnostic not in supported:
                raise PartitionedPreflightUnsupported(
                    f"partitioned VDN diagnostic {required_linear_diagnostic!r} was requested but "
                    "the installed VDN bridge does not publish that diagnostic capability"
                )
    if matched == 0:
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires active VDN-H3 ownership")


def _verify_partitioned_vdn_linear_diagnostic(
    metrics,
    mode: str,
    *,
    bypass_calls_before: int,
    bypass_video_rows_before: int,
    suppression_calls_before: int = 0,
    suppressed_taps_before: int = 0,
    suppressed_rows_before: int = 0,
    raw_measure_calls_before: int = 0,
    raw_measure_prefix_frames_before: int = 0,
) -> None:
    """Fail closed when a requested VDN linear diagnostic did not execute in low/probe."""

    mode = normalize_vdn_linear_diagnostic(mode)
    if mode == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
        return
    counters = getattr(metrics, "counters", {})
    event = getattr(metrics, "event", None)

    if mode == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS:
        calls_after = int(counters.get("partitioned_vdn_linear_bypass_calls", 0))
        rows_after = int(counters.get("partitioned_vdn_linear_bypass_video_rows", 0))
        delta_calls = calls_after - int(bypass_calls_before)
        delta_rows = rows_after - int(bypass_video_rows_before)
        if delta_calls <= 0:
            raise RuntimeError(
                "partitioned VDN linear bypass was requested but zero bypass calls were observed; "
                "refusing to accept this run as a diagnostic sample"
            )
        if delta_rows <= 0:
            raise RuntimeError(
                "partitioned VDN linear bypass executed without reporting any bypassed video rows; "
                "refusing to accept this run as a diagnostic sample"
            )
        if callable(event):
            event(
                "partitioned_vdn_linear_diagnostic_verified",
                mode=mode,
                bypass_calls=delta_calls,
                bypass_video_rows=delta_rows,
                fail_closed=True,
            )
        return

    if mode == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE:
        calls_after = int(counters.get("partitioned_vdn_raw_token_measure_calls", 0))
        frames_after = int(counters.get("partitioned_vdn_raw_token_measure_prefix_frames", 0))
        delta_calls = calls_after - int(raw_measure_calls_before)
        delta_frames = frames_after - int(raw_measure_prefix_frames_before)
        if delta_calls <= 0 or delta_frames <= 0:
            raise RuntimeError(
                "partitioned VDN raw-token measure diagnostic was requested but no verified "
                "raw-token measure execution was observed; refusing this diagnostic sample"
            )
        if callable(event):
            event(
                "partitioned_vdn_linear_diagnostic_verified",
                mode=mode,
                raw_token_measure_calls=delta_calls,
                raw_token_measure_prefix_frames=delta_frames,
                fail_closed=True,
            )
        return

    calls_after = int(counters.get("partitioned_vdn_cross_grid_temporal_suppression_calls", 0))
    taps_after = int(counters.get("partitioned_vdn_cross_grid_temporal_suppressed_taps", 0))
    rows_after = int(counters.get("partitioned_vdn_cross_grid_temporal_suppressed_rows", 0))
    delta_calls = calls_after - int(suppression_calls_before)
    delta_taps = taps_after - int(suppressed_taps_before)
    delta_rows = rows_after - int(suppressed_rows_before)
    if delta_calls <= 0 or delta_taps <= 0 or delta_rows <= 0:
        raise RuntimeError(
            "partitioned VDN cross-grid temporal diagnostic was requested but no verified "
            "cross-grid short-conv suppression was observed; refusing this diagnostic sample"
        )
    if callable(event):
        event(
            "partitioned_vdn_linear_diagnostic_verified",
            mode=mode,
            cross_grid_temporal_suppression_calls=delta_calls,
            suppressed_taps=delta_taps,
            suppressed_rows=delta_rows,
            fail_closed=True,
        )


def _verify_prefix_transformer_context_diagnostic(
    metrics,
    mode: str,
    *,
    calls_before: int,
    prefix_frames_before: int,
) -> None:
    """Fail closed when the source-carrier structural A/B did not execute."""

    mode = normalize_prefix_transformer_context(mode)
    if mode == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        return
    counters = getattr(metrics, "counters", {})
    calls_after = int(counters.get("partitioned_source_carrier_uniform_transformer_calls", 0))
    frames_after = int(counters.get("partitioned_source_carrier_uniform_prefix_frames", 0))
    delta_calls = calls_after - int(calls_before)
    delta_frames = frames_after - int(prefix_frames_before)
    if delta_calls <= 0 or delta_frames <= 0:
        raise RuntimeError(
            "source-carrier uniform transformer diagnostic was requested but no verified "
            "low/probe execution was observed; refusing this diagnostic sample"
        )
    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_prefix_transformer_context_verified",
            mode=mode,
            transformer_calls=delta_calls,
            prefix_frames=delta_frames,
            heterogeneous_partition_contract_published=False,
            fail_closed=True,
        )


def _validate_audio_position_candidate_configuration(
    audio_position_domain: str,
    prefix_transformer_context: str,
    vdn_linear_diagnostic: str,
) -> None:
    mode = normalize_audio_position_domain(audio_position_domain)
    if mode == PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY:
        return
    if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        raise PartitionedPreflightUnsupported(
            "source_carrier audio-position candidate requires prefix_transformer_context='exact_target_partitioned'"
        )
    if vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
        raise PartitionedPreflightUnsupported(
            "source_carrier audio-position candidate requires vdn_linear_diagnostic='normal'"
        )


def _verify_audio_position_domain_diagnostic(
    metrics,
    mode: str,
    *,
    block0_calls_before: int,
    wrapper_entries_before: int,
    model_timestep_calls_before: int,
) -> None:
    mode = normalize_audio_position_domain(mode)
    if mode == PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY:
        return
    counters = getattr(metrics, "counters", {})
    block0_after = int(counters.get("partitioned_audio_position_source_carrier_block0_calls", 0))
    wrapper_after = int(counters.get("partitioned_audio_position_candidate_wrapper_entries", 0))
    model_timestep_after = int(counters.get("partitioned_audio_model_timestep_override_calls", 0))
    block0_delta = block0_after - int(block0_calls_before)
    wrapper_delta = wrapper_after - int(wrapper_entries_before)
    model_timestep_delta = model_timestep_after - int(model_timestep_calls_before)
    if block0_delta <= 0:
        raise RuntimeError(
            "source-carrier audio-position candidate was requested but no actual low/probe "
            "block-zero execution was observed; refusing this diagnostic sample"
        )
    if wrapper_delta < block0_delta:
        raise RuntimeError("source-carrier audio-position candidate block execution exceeded wrapper-entry accounting")
    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_audio_position_domain_verified",
            mode=mode,
            wrapper_entries=wrapper_delta,
            actual_block0_calls=block0_delta,
            model_timestep_override_calls=model_timestep_delta,
            target_audio_rows_only=True,
            sampler_mask_mutated=False,
            fail_closed=True,
        )


def _validate_partitioned_sol_compat(guider: Any) -> None:
    """Require the Sol request owner before any partitioned sampler lifetime.

    VDN API-4 delegates partitioned sparse attention to Sol's request-owned
    backend.  Sol intentionally refuses those calls outside its native
    OUTER_SAMPLE lifecycle, so missing or stale Sol metadata is a preflight
    fallback condition rather than a mid-sampler runtime failure.
    """
    options = getattr(guider, "model_options", None)
    transformer = options.get("transformer_options") if isinstance(options, dict) else None
    metadata = transformer.get(SOL_RUNTIME_KEY) if isinstance(transformer, dict) else None
    if not isinstance(metadata, dict):
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix requires active Sol-H3 native runtime ownership"
        )

    mismatches = [
        f"{name}={metadata.get(name)!r}"
        for name, expected in PARTITIONED_SOL_REQUIRED_METADATA.items()
        if metadata.get(name) != expected
    ]
    if mismatches:
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix requires compatible Sol-H3 native runtime metadata: " + ", ".join(mismatches)
        )


def _preflight(
    guider: Any,
    config: ProgressiveTargetInputConfig,
    noise: torch.Tensor,
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor | None,
    latent_shapes: list[tuple[int, ...]],
    *,
    required_vdn_linear_diagnostic: str = PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
):
    if len(latent_shapes) != 2:
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires native packed H3 AV latents")
    if config.transfer_mode != "learned_3d":
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires learned_3d handoff transfer")
    if config.suffix_dc_bridge or config.suffix_geometric_bridge:
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix owns its DC bridge locally and does not inherit generic seam-repair flags"
        )
    if denoise_mask is None or not _has_exact_video_protection(denoise_mask, latent_shapes):
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires an exact protected video prefix")
    target_h, target_w = map(int, latent_shapes[0][-2:])
    if target_h % 2 or target_w % 2:
        raise PartitionedPreflightUnsupported("target H3 geometry is not patch-safe")

    # These two owners are structural prerequisites for the heterogeneous
    # attention path. Validate them before any sampler lifetime is committed so
    # unsupported saved workflows use the released exact target-grid fallback.
    _validate_partitioned_vdn_compat(
        guider.model_patcher,
        required_linear_diagnostic=required_vdn_linear_diagnostic,
    )
    _validate_partitioned_sol_compat(guider)

    try:
        source_h, source_w = config.resolve_source(target_h, target_w)
        internal = _process_latent_in(guider.model_patcher.model, latent_image, latent_shapes)
        stage_plan = build_partitioned_stage_plan(
            denoise_mask,
            latent_shapes,
            internal,
            noise,
            source_h=source_h,
            source_w=source_w,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PartitionedPreflightUnsupported(str(exc)) from exc
    return source_h, source_w, stage_plan


def _recover_partitioned_transfer_clean(
    target_video: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    diagnostic_noise = deterministic_video_noise(
        tuple(target_video.shape),
        seed=int(seed),
        device=target_video.device,
        dtype=target_video.dtype,
    )
    return recover_conditional_clean_for_diagnostics(
        target_video,
        diagnostic_noise,
        sigma=float(sigma),
    )


def _apply_partitioned_suffix_dc_bridge(
    target_video: torch.Tensor,
    learned_clean: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    sigma: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | bool]]:
    """Preserve the learned handoff's native DC relation across exact-prefix restore.

    The learned 3D transfer produces a coherent learned prefix/suffix pair, but
    partitioned continuation must replace that prefix with caller-owned exact
    target-grid context.  When enabled, transfer the resulting per-channel
    spatial-mean offset onto only the first generated suffix token and map that
    clean-space change onto the already re-noised conditional state.  The exact
    prefix and every later suffix token remain untouched by the bridge itself.
    """

    prefix_t = int(exact_prefix.shape[2])
    if not enabled:
        return (
            target_video.clone(),
            learned_clean.clone(),
            disabled_suffix_dc_bridge_metrics(prefix_t=prefix_t),
        )

    corrected_clean, bridge_metrics = apply_suffix_dc_bridge(
        learned_clean,
        exact_prefix,
        weights=(1.0,),
    )
    corrected_tokens = int(bridge_metrics["suffix_dc_bridge_corrected_tokens"])
    mapped_state = map_clean_bridge_to_conditional_state(
        target_video,
        learned_clean,
        corrected_clean,
        sigma=float(sigma),
        prefix_t=prefix_t,
        corrected_tokens=corrected_tokens,
    )
    return mapped_state, corrected_clean, bridge_metrics


def _partitioned_exact_overlap_fallback_eligibility(
    transaction: dict[str, Any],
) -> tuple[bool, str]:
    """Authorize structural overlap repair only after an unambiguous rigid boundary veto.

    The rigid-v2 transaction remains authoritative.  This fallback is not a
    weaker registration threshold: it is eligible only when video registration
    accepted, all four boundary-motion witnesses were measurable in both ROIs,
    and the proposed rigid translation was rejected solely because it did not
    preserve enough native boundary motion.  Ambiguous, clipped, invalid-area,
    guidance, or registration failures remain exact baseline fallbacks.
    """

    if transaction.get("result") != "rejected":
        return False, "frame_gauge_not_rejected"
    reason = str(transaction.get("reason", ""))
    if reason not in FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS:
        return False, "frame_gauge_rejection_not_structural_overlap_eligible"
    video_registration = transaction.get("video_registration")
    if not isinstance(video_registration, dict) or video_registration.get("status") != "accepted":
        return False, "video_registration_not_accepted"
    boundary = transaction.get("boundary_motion")
    if (
        not isinstance(boundary, dict)
        or boundary.get("status") != "rejected"
        or str(boundary.get("reason", "")) != reason
        or boundary.get("policy") != "native_boundary_motion_preservation_v2"
    ):
        return False, "boundary_receipt_inconsistent"
    checks = boundary.get("checks")
    if not isinstance(checks, dict) or set(checks) != {"upper45", "full"}:
        return False, "boundary_receipt_incomplete"
    for roi in ("upper45", "full"):
        check = checks.get(roi)
        if not isinstance(check, dict):
            return False, f"boundary_{roi}_receipt_missing"
        for variant in ("native", "transformed_native", "exact_restored", "candidate"):
            receipt = check.get(variant)
            if not isinstance(receipt, dict):
                return False, f"boundary_{roi}_{variant}_missing"
            try:
                response = float(receipt.get("response"))
            except (TypeError, ValueError):
                return False, f"boundary_{roi}_{variant}_response_invalid"
            if not math.isfinite(response) or response < FRAME_GAUGE_BOUNDARY_MIN_RESPONSE:
                return False, f"boundary_{roi}_{variant}_ambiguous"
            if bool(receipt.get("clipped")):
                return False, f"boundary_{roi}_{variant}_clipped"
    return True, reason


def _apply_partitioned_exact_overlap_bridge(
    target_video: torch.Tensor,
    learned_clean: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, float | int | bool]]:
    """Reconcile the measured exact/learned overlap before authoritative restore.

    The existing representation primitive transfers only the zero-spatial-mean
    overlap residual onto the first generated suffix token.  The existing DC
    bridge then transfers the spatial mean.  Their sum is exactly the measured
    same-time residual E_p-L_p, so the immediate exact-prefix -> suffix clean
    transition equals the learned provider's native L_p -> L_s transition
    algebraically.  No later suffix token is extrapolated.
    """

    prefix_t = int(exact_prefix.shape[2])
    structured_clean, representation_metrics = apply_suffix_representation_bridge(
        learned_clean,
        exact_prefix,
        requested=True,
    )
    corrected_clean, dc_metrics = apply_suffix_dc_bridge(
        structured_clean,
        exact_prefix,
        weights=(1.0,),
    )
    corrected_tokens = max(
        int(representation_metrics["suffix_representation_bridge_corrected_tokens"]),
        int(dc_metrics["suffix_dc_bridge_corrected_tokens"]),
    )
    if corrected_tokens < 1:
        return target_video.clone(), corrected_clean, representation_metrics, dc_metrics
    mapped_state = map_clean_bridge_to_conditional_state(
        target_video,
        learned_clean,
        corrected_clean,
        sigma=float(sigma),
        prefix_t=prefix_t,
        corrected_tokens=corrected_tokens,
    )
    return mapped_state, corrected_clean, representation_metrics, dc_metrics


def _prepare_registered_guidance_reference(
    *,
    run: Any,
    guidance: Any,
    exact_prefix: torch.Tensor,
    target_h: int,
    target_w: int,
    prefix_t: int,
    split_coordinate: float,
    high_sigmas: torch.Tensor,
    video_shift: float,
    residual_mode: str = "off",
    residual_witnesses: dict[str, torch.Tensor] | None = None,
) -> tuple[RegisteredGuidanceReference | None, dict[str, Any], str | None]:
    """Register the active target-grid Flow reference without mutating its trajectory."""

    if guidance.mode == "downsample_consistency":
        return (
            None,
            {"status": "rejected", "reason": "unsupported_guidance_operator"},
            ("unsupported_guidance_operator"),
        )

    sampler = str(getattr(run, "sampler", ""))
    if sampler not in FRAME_GAUGE_GUIDANCE_SUPPORTED_SAMPLERS:
        return (
            None,
            {
                "status": "rejected",
                "reason": "unsupported_sampler_contract",
                "sampler": sampler,
                "supported_samplers": tuple(sorted(FRAME_GAUGE_GUIDANCE_SUPPORTED_SAMPLERS)),
            },
            "unsupported_sampler_contract",
        )

    coordinate_tolerance = 1e-7
    exact_probe = any(
        sample.phase == "handoff_probe"
        and math.isclose(
            float(sample.coordinate),
            float(split_coordinate),
            rel_tol=0.0,
            abs_tol=coordinate_tolerance,
        )
        for sample in run.exact_samples()
    )
    source_ref, reference_coordinate, reference_clamped = time_matched_reference_info(
        run,
        split_coordinate,
    )
    if (
        not exact_probe
        or reference_clamped
        or not math.isclose(
            float(reference_coordinate),
            float(split_coordinate),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        return (
            None,
            {
                "status": "rejected",
                "reason": "missing_exact_probe_endpoint",
                "reference_coordinate": float(reference_coordinate),
                "split_coordinate": float(split_coordinate),
            },
            "missing_exact_probe_endpoint",
        )

    high_coordinates = [
        float(normalized_coordinate(float(value), video_shift=video_shift))
        for value in high_sigmas[:-1].detach().to(device="cpu", dtype=torch.float64).tolist()
    ]
    if any(value > float(reference_coordinate) + coordinate_tolerance for value in high_coordinates):
        return (
            None,
            {
                "status": "rejected",
                "reason": "high_schedule_exceeds_probe_endpoint",
                "reference_coordinate": float(reference_coordinate),
                "high_coordinate_max": max(high_coordinates),
            },
            "high_schedule_exceeds_probe_endpoint",
        )
    resolved_high_coordinates = []
    for value in high_coordinates:
        _, resolved, _ = time_matched_reference_info(run, value)
        resolved_high_coordinates.append(float(resolved))
    if any(
        not math.isclose(
            value,
            float(reference_coordinate),
            rel_tol=0.0,
            abs_tol=coordinate_tolerance,
        )
        for value in resolved_high_coordinates
    ):
        return (
            None,
            {
                "status": "rejected",
                "reason": "high_schedule_changes_reference_identity",
                "reference_coordinate": float(reference_coordinate),
                "resolved_high_coordinates": tuple(resolved_high_coordinates),
            },
            "high_schedule_changes_reference_identity",
        )

    source_ref = source_ref.to(device=exact_prefix.device, dtype=exact_prefix.dtype)
    if int(source_ref.shape[2]) < prefix_t:
        return (
            None,
            {
                "status": "rejected",
                "reason": "guidance_target_geometry_mismatch",
            },
            "guidance_target_geometry_mismatch",
        )

    # Registration only needs the bounded authoritative prefix. Resize that
    # first so rejected candidates do not pay for a full trajectory transfer.
    # If registration succeeds, resize the suffix exactly once and concatenate
    # the already-resized prefix. This preserves the numerical transfer while
    # bounding rejected-path work to the prefix needed for registration.
    prefix_resize_started = time.perf_counter()
    target_prefix = resize_video(
        source_ref[:, :, :prefix_t],
        target_h,
        target_w,
        mode=guidance.transfer_mode,
    )
    prefix_resize_elapsed_ms = (time.perf_counter() - prefix_resize_started) * 1000.0
    if tuple(target_prefix.shape) != tuple(exact_prefix.shape):
        return (
            None,
            {
                "status": "rejected",
                "reason": "guidance_target_geometry_mismatch",
                "prefix_resize_elapsed_ms": prefix_resize_elapsed_ms,
            },
            "guidance_target_geometry_mismatch",
        )

    estimate = estimate_paired_prefix_translation(
        target_prefix,
        exact_prefix,
        policy=GUIDANCE_REFERENCE_POLICY,
    )
    estimate_fields = estimate.telemetry()
    estimate_fields["prefix_resize_elapsed_ms"] = prefix_resize_elapsed_ms
    if estimate.rejected:
        return None, estimate_fields, f"guidance_{estimate.reason}"

    # Identity is an exact no-resample path. The estimator may return a
    # sub-grid optimum inside the identity tolerance; that value remains
    # diagnostic only and must not become a spatial interpolation.
    applied_dx = 0.0 if estimate.identity else float(estimate.dx)
    applied_dy = 0.0 if estimate.identity else float(estimate.dy)
    residual_mode = normalize_residual_geometry_mode(residual_mode)
    if residual_mode == "measure" and estimate.accepted:
        guidance_residual = measure_residual_geometry(
            target_prefix,
            exact_prefix,
            rigid_dx=applied_dx,
            rigid_dy=applied_dy,
        )
    elif residual_mode == "measure":
        guidance_residual = {
            "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
            "status": "not_evaluated",
            "reason": "rigid_not_accepted",
            "eligible": False,
        }
    else:
        guidance_residual = {
            "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
            "status": "off",
            "reason": "disabled",
            "eligible": False,
        }

    temporal_radius = int(guidance.temporal_search_radius)
    if guidance.mode == "direction+temporal":
        spatial_scale = max(
            target_h / int(run.geometry.latent_h),
            target_w / int(run.geometry.latent_w),
        )
        temporal_radius = math.ceil(int(guidance.temporal_search_radius) * spatial_scale)
        if temporal_radius > 8:
            fields = dict(estimate_fields)
            fields.update(
                status="rejected",
                reason="target_temporal_radius_over_bound",
                target_temporal_radius=temporal_radius,
            )
            return None, fields, "target_temporal_radius_over_bound"

    suffix_resize_started = time.perf_counter()
    if prefix_t < int(source_ref.shape[2]):
        target_suffix = resize_video(
            source_ref[:, :, prefix_t:],
            target_h,
            target_w,
            mode=guidance.transfer_mode,
        )
        target_ref = torch.cat((target_prefix, target_suffix), dim=2)
    else:
        target_ref = target_prefix
    suffix_resize_elapsed_ms = (time.perf_counter() - suffix_resize_started) * 1000.0
    if tuple(target_ref.shape) != (
        int(exact_prefix.shape[0]),
        int(exact_prefix.shape[1]),
        int(run.geometry.latent_t),
        int(target_h),
        int(target_w),
    ):
        fields = dict(estimate_fields)
        fields.update(
            status="rejected",
            reason="guidance_target_geometry_mismatch",
            suffix_resize_elapsed_ms=suffix_resize_elapsed_ms,
        )
        return None, fields, "guidance_target_geometry_mismatch"

    if residual_mode == "measure" and residual_witnesses is not None:
        witness_start = max(0, prefix_t - 6)
        witness_end = min(int(target_ref.shape[2]), prefix_t + 4)
        residual_witnesses["guidance_native_bounded"] = target_ref[:, :, witness_start:witness_end].detach().clone()

    translation_started = time.perf_counter()
    application = translate_video_cells(
        target_ref,
        dx=applied_dx,
        dy=applied_dy,
        start_frame=prefix_t,
        batch_frames=4,
    )
    translation_elapsed_ms = (time.perf_counter() - translation_started) * 1000.0
    if residual_mode == "measure" and residual_witnesses is not None:
        witness_start = max(0, prefix_t - 6)
        witness_end = min(int(application.video.shape[2]), prefix_t + 4)
        residual_witnesses["guidance_aligned_bounded"] = (
            application.video[:, :, witness_start:witness_end].detach().clone()
        )

    if application.invalid_fraction > 0.08:
        fields = dict(estimate_fields)
        fields.update(
            status="rejected",
            reason="guidance_invalid_area_over_bound",
            invalid_fraction=application.invalid_fraction,
        )
        return None, fields, "guidance_invalid_area_over_bound"

    # application.video can alias target_ref for an identity registration.
    # Always clone before restoring the caller-owned prefix so the captured
    # trajectory and its time-matched tensor remain read-only.
    registered_video = application.video.clone()
    registered_video[:, :, :prefix_t] = exact_prefix.to(registered_video)
    cache_key = (
        f"{run.run_id}:{run.session_id}:{run.chunk_id}:handoff_probe:"
        f"{prefix_t}:{target_h}x{target_w}:"
        f"{applied_dx:.8f}:{applied_dy:.8f}:"
        f"{float(reference_coordinate):.12f}:"
        f"translation_validity_v1:{FRAME_GAUGE_POLICY_VERSION}"
    )
    registered = RegisteredGuidanceReference(
        video=registered_video,
        validity=application.valid_mask.detach(),
        prefix_t=prefix_t,
        run_id=str(run.run_id),
        reference_coordinate=float(reference_coordinate),
        dx=applied_dx,
        dy=applied_dy,
        cache_key=cache_key,
        temporal_search_radius=temporal_radius,
    )
    fields = dict(estimate_fields)
    fields.update(
        estimated_dx=float(estimate.dx),
        estimated_dy=float(estimate.dy),
        dx=applied_dx,
        dy=applied_dy,
        identity_no_resample=bool(estimate.identity),
        reference_coordinate=float(reference_coordinate),
        reference_clamped=False,
        target_temporal_radius=temporal_radius,
        cache_key=cache_key,
        trajectory_mutated=False,
        exact_prefix_restored=True,
        invalid_fraction=float(application.invalid_fraction),
        residual_geometry=guidance_residual,
        prefix_resize_elapsed_ms=prefix_resize_elapsed_ms,
        suffix_resize_elapsed_ms=suffix_resize_elapsed_ms,
        translation_elapsed_ms=translation_elapsed_ms,
    )
    return registered, fields, None


def _frame_gauge_boundary_motion_check(
    learned_clean: torch.Tensor,
    exact_prefix: torch.Tensor,
    aligned_clean: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[bool, dict[str, Any], str]:
    """Verify that the proposed rigid shift repairs the actual splice motion.

    Prefix-wide residual fit can be diluted by learned synthesis differences.
    This gate instead asks whether replacing the provider prefix with the exact
    prefix introduces a boundary-motion error relative to the provider's own
    native transition in each corresponding coordinate domain, and whether the
    proposed suffix translation removes a substantial fraction of that error
    in both the upper-region and full-frame measurements.
    """

    if not 0 < int(prefix_t) < int(learned_clean.shape[2]):
        raise RuntimeError("frame-gauge boundary check requires a non-empty prefix and suffix")
    if tuple(exact_prefix.shape) != tuple(learned_clean[:, :, : int(prefix_t)].shape):
        raise RuntimeError("frame-gauge boundary check exact-prefix geometry drifted")

    learned_last = learned_clean[:, :, int(prefix_t) - 1]
    exact_last = exact_prefix[:, :, -1].to(learned_clean)
    native_first = learned_clean[:, :, int(prefix_t)]
    if tuple(aligned_clean.shape) == tuple(learned_clean.shape):
        aligned_last = aligned_clean[:, :, int(prefix_t) - 1]
        aligned_first = aligned_clean[:, :, int(prefix_t)]
    else:
        compact_shape = (
            int(learned_clean.shape[0]),
            int(learned_clean.shape[1]),
            2,
            int(learned_clean.shape[-2]),
            int(learned_clean.shape[-1]),
        )
        if tuple(aligned_clean.shape) != compact_shape:
            raise RuntimeError("frame-gauge boundary check geometry drifted")
        aligned_last = aligned_clean[:, :, 0]
        aligned_first = aligned_clean[:, :, 1]

    def shift(left: torch.Tensor, right: torch.Tensor, roi_fraction: float) -> dict[str, Any]:
        pair = torch.stack((left, right), dim=2)
        fields = measure_translation_trajectory(
            pair,
            1,
            forward_steps=1,
            backward_steps=0,
            roi_fraction=roi_fraction,
            max_shift=4,
        )
        return {
            "dx": float(fields["pairwise_dx"][0]),
            "dy": float(fields["pairwise_dy"][0]),
            "response": float(fields["pairwise_response"][0]),
            "clipped": bool(fields["pairwise_clipped"][0]),
        }

    checks: dict[str, Any] = {}
    for name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
        native = shift(learned_last, native_first, roi_fraction)
        # A cropped/windowed phase estimate is not invariant to jointly warping
        # both frames. Compare the candidate against the same transformed
        # provider transition, so only exact-prefix replacement contributes to
        # its error. The aligned prefix remains a witness, never output state.
        transformed_native = shift(aligned_last, aligned_first, roi_fraction)
        restored = shift(exact_last, native_first, roi_fraction)
        candidate = shift(exact_last, aligned_first, roi_fraction)
        before_error = math.hypot(
            restored["dx"] - native["dx"],
            restored["dy"] - native["dy"],
        )
        after_error = math.hypot(
            candidate["dx"] - transformed_native["dx"],
            candidate["dy"] - transformed_native["dy"],
        )
        improvement = (before_error - after_error) / max(before_error, 1e-12)
        checks[name] = {
            "roi_fraction": roi_fraction,
            "native": native,
            "transformed_native": transformed_native,
            "exact_restored": restored,
            "candidate": candidate,
            "before_error_cells": before_error,
            "after_error_cells": after_error,
            "error_improvement_ratio": improvement,
            "informative": bool(before_error >= FRAME_GAUGE_BOUNDARY_MIN_ERROR_CELLS),
        }

    fields: dict[str, Any] = {
        "policy": "native_boundary_motion_preservation_v2",
        "min_error_cells": FRAME_GAUGE_BOUNDARY_MIN_ERROR_CELLS,
        "min_improvement_ratio": FRAME_GAUGE_BOUNDARY_MIN_IMPROVEMENT,
        "min_response": FRAME_GAUGE_BOUNDARY_MIN_RESPONSE,
        "checks": checks,
    }

    for name in ("upper45", "full"):
        check = checks[name]
        for variant in ("native", "transformed_native", "exact_restored", "candidate"):
            receipt = check[variant]
            if receipt["clipped"] or receipt["response"] < FRAME_GAUGE_BOUNDARY_MIN_RESPONSE:
                fields["status"] = "rejected"
                fields["reason"] = f"boundary_{name}_{variant}_ambiguous"
                return False, fields, str(fields["reason"])
        # A low-amplitude boundary receipt is not evidence against the prefix
        # estimator: smooth/periodic synthetic fields can make phase correlation
        # under-report an otherwise well-conditioned rigid shift. It remains a
        # non-degradation veto, while a clearly measurable boundary error must
        # improve by the stronger minimum ratio.
        if check["after_error_cells"] > check["before_error_cells"]:
            fields["status"] = "rejected"
            fields["reason"] = f"boundary_{name}_not_improved"
            return False, fields, str(fields["reason"])
        if check["informative"] and check["error_improvement_ratio"] < FRAME_GAUGE_BOUNDARY_MIN_IMPROVEMENT:
            fields["status"] = "rejected"
            fields["reason"] = f"boundary_{name}_insufficient_improvement"
            return False, fields, str(fields["reason"])

    fields["status"] = "accepted"
    fields["reason"] = "accepted"
    return True, fields, "accepted"


def _regional_boundary_motion_receipts(
    learned_clean: torch.Tensor,
    exact_prefix: torch.Tensor,
    aligned_witness: torch.Tensor,
    corrected_clean: torch.Tensor,
    *,
    prefix_t: int,
    tile_bounds: dict[str, list[int]],
) -> dict[str, Any]:
    """Measure the rigid-v2 boundary transaction on each disjoint residual tile."""

    if prefix_t < 1 or prefix_t >= int(learned_clean.shape[2]):
        raise RuntimeError("regional boundary receipt requires a prefix/suffix boundary")
    exact_last = exact_prefix[:, :, -1].to(learned_clean)

    def shift(left: torch.Tensor, right: torch.Tensor, bounds: list[int]) -> dict[str, Any]:
        y0, y1, x0, x1 = (int(value) for value in bounds)
        pair = torch.stack(
            (
                left[:, :, y0:y1, x0:x1],
                right[:, :, y0:y1, x0:x1],
            ),
            dim=2,
        )
        fields = measure_translation_trajectory(
            pair,
            1,
            forward_steps=1,
            backward_steps=0,
            roi_fraction=1.0,
            max_shift=2,
        )
        return {
            "dx": float(fields["pairwise_dx"][0]),
            "dy": float(fields["pairwise_dy"][0]),
            "response": float(fields["pairwise_response"][0]),
            "clipped": bool(fields["pairwise_clipped"][0]),
        }

    learned_last = learned_clean[:, :, prefix_t - 1]
    learned_first = learned_clean[:, :, prefix_t]
    aligned_last = aligned_witness[:, :, prefix_t - 1]
    aligned_first = aligned_witness[:, :, prefix_t]
    corrected_first = corrected_clean[:, :, prefix_t]
    receipts: dict[str, Any] = {}
    for tile_id, bounds in tile_bounds.items():
        native = shift(learned_last, learned_first, bounds)
        exact_unregistered = shift(exact_last, learned_first, bounds)
        transformed_native = shift(aligned_last, aligned_first, bounds)
        exact_rigid_pre_dc = shift(exact_last, aligned_first, bounds)
        exact_rigid_post_dc = shift(exact_last, corrected_first, bounds)
        receipts[str(tile_id)] = {
            "bounds": list(bounds),
            "native": native,
            "exact_unregistered": exact_unregistered,
            "transformed_native": transformed_native,
            "exact_rigid_pre_dc": exact_rigid_pre_dc,
            "exact_rigid_post_dc": exact_rigid_post_dc,
            "pre_dc_native_error": math.hypot(
                exact_rigid_pre_dc["dx"] - transformed_native["dx"],
                exact_rigid_pre_dc["dy"] - transformed_native["dy"],
            ),
            "post_dc_native_error": math.hypot(
                exact_rigid_post_dc["dx"] - transformed_native["dx"],
                exact_rigid_post_dc["dy"] - transformed_native["dy"],
            ),
        }
    return {
        "policy": "paired_prefix_residual_boundary_regions_v1",
        "coordinate_units": "target_latent_cells",
        "comparison": (
            "exact-restored boundary versus rigid-transformed native motion; "
            "pre-DC and actual post-DC are reported separately"
        ),
        "tiles": receipts,
    }


def _frame_gauge_clean_postprocess(
    learned_clean: torch.Tensor,
    *,
    exact_prefix: torch.Tensor,
    guidance_run: Any,
    guidance: Any,
    target_h: int,
    target_w: int,
    prefix_t: int,
    split_coordinate: float,
    high_sigmas: torch.Tensor,
    video_shift: float,
    residual_mode: str = "off",
) -> tuple[
    CleanVideoPostprocessResult,
    RegisteredGuidanceReference | None,
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    """Run the all-or-nothing clean-domain frame-gauge registration transaction."""

    started = time.perf_counter()
    residual_mode = normalize_residual_geometry_mode(residual_mode)
    exact_prefix = exact_prefix.to(
        device=learned_clean.device,
        dtype=learned_clean.dtype,
    )
    video_estimate = estimate_paired_prefix_translation(
        learned_clean[:, :, :prefix_t],
        exact_prefix,
        policy=LEARNED_VIDEO_POLICY,
    )
    guidance_active = guidance is not None and guidance.mode != "off"
    residual_witnesses: dict[str, torch.Tensor] = {}
    residual_geometry: dict[str, Any] = {
        "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
        "requested_mode": residual_mode,
        "measured": False,
        "measurement_status": "off" if residual_mode == "off" else "not_evaluated",
        "reason": "disabled" if residual_mode == "off" else "rigid_not_accepted",
        "selected_model": "none",
        "decision": "not_evaluated",
        "applied": False,
        "final_path": "baseline",
        "video": {
            "status": "off" if residual_mode == "off" else "not_evaluated",
            "reason": "disabled" if residual_mode == "off" else "rigid_not_accepted",
        },
        "guidance": (
            {"status": "off", "reason": "guidance_off"}
            if not guidance_active
            else {
                "status": "off" if residual_mode == "off" else "not_evaluated",
                "reason": "disabled" if residual_mode == "off" else "rigid_not_accepted",
            }
        ),
    }
    transaction: dict[str, Any] = {
        "policy_version": FRAME_GAUGE_POLICY_VERSION,
        "video_registration": video_estimate.telemetry(),
        "guidance_registration": (
            {"status": "not_evaluated", "reason": "pending_video_acceptance"}
            if guidance_active
            else {"status": "off", "reason": "guidance_off"}
        ),
        "boundary_motion": {"status": "not_evaluated", "reason": "pending_video_acceptance"},
        "result": "baseline",
        "reason": video_estimate.reason,
        "dc_applied_in_clean_hook": False,
        "spatial_warp_applied": False,
        "residual_geometry": residual_geometry,
    }
    if not video_estimate.accepted:
        if guidance_active:
            transaction["guidance_registration"] = {
                "status": "not_evaluated",
                "reason": "video_registration_not_accepted",
            }
        transaction["boundary_motion"] = {
            "status": "not_evaluated",
            "reason": "video_registration_not_accepted",
        }
        transaction["result"] = video_estimate.status
        transaction["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        result = CleanVideoPostprocessResult(
            clean_video=learned_clean,
            protected_prefix_t=prefix_t,
            metadata=transaction,
        )
        return result, None, {}, transaction

    # The boundary gate only consumes the last learned-prefix frame and the
    # first suffix frame. Translate that two-frame witness first; do not clone
    # and warp the complete trajectory until every fail-closed gate (including
    # guidance registration) has accepted.
    boundary_translation_started = time.perf_counter()
    boundary_application = translate_video_cells(
        learned_clean[:, :, prefix_t - 1 : prefix_t + 1],
        dx=float(video_estimate.dx),
        dy=float(video_estimate.dy),
        start_frame=0,
        batch_frames=2,
    )
    boundary_translation_elapsed_ms = (time.perf_counter() - boundary_translation_started) * 1000.0
    transaction["boundary_translation_elapsed_ms"] = boundary_translation_elapsed_ms
    if boundary_application.invalid_fraction > 0.08:
        transaction.update(
            result="rejected",
            reason="video_invalid_area_over_bound",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        transaction["boundary_motion"] = {
            "status": "not_evaluated",
            "reason": "video_invalid_area_over_bound",
        }
        if guidance_active:
            transaction["guidance_registration"] = {
                "status": "not_evaluated",
                "reason": "video_invalid_area_over_bound",
            }
        result = CleanVideoPostprocessResult(
            clean_video=learned_clean,
            protected_prefix_t=prefix_t,
            metadata=transaction,
        )
        return result, None, {}, transaction

    boundary_ok, boundary_fields, boundary_reason = _frame_gauge_boundary_motion_check(
        learned_clean,
        exact_prefix,
        boundary_application.video,
        prefix_t=prefix_t,
    )
    transaction["boundary_motion"] = boundary_fields
    if not boundary_ok:
        transaction.update(
            result="rejected",
            reason=boundary_reason,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        if guidance_active:
            transaction["guidance_registration"] = {
                "status": "not_evaluated",
                "reason": "boundary_motion_rejected",
            }
        result = CleanVideoPostprocessResult(
            clean_video=learned_clean,
            protected_prefix_t=prefix_t,
            metadata=transaction,
        )
        witnesses = {"learned_boundary_pair": learned_clean[:, :, prefix_t - 1 : prefix_t + 1].detach().clone()}
        return result, None, witnesses, transaction

    registered_reference = None
    if guidance_active:
        guidance_registration_started = time.perf_counter()
        registered_reference, guidance_fields, guidance_error = _prepare_registered_guidance_reference(
            run=guidance_run,
            guidance=guidance,
            exact_prefix=exact_prefix,
            target_h=target_h,
            target_w=target_w,
            prefix_t=prefix_t,
            split_coordinate=split_coordinate,
            high_sigmas=high_sigmas,
            video_shift=video_shift,
            residual_mode=residual_mode,
            residual_witnesses=residual_witnesses,
        )
        transaction["guidance_registration_elapsed_ms"] = (time.perf_counter() - guidance_registration_started) * 1000.0
        transaction["guidance_registration"] = guidance_fields
        if guidance_error is not None:
            transaction["result"] = "rejected"
            transaction["reason"] = guidance_error
            transaction["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
            result = CleanVideoPostprocessResult(
                clean_video=learned_clean,
                protected_prefix_t=prefix_t,
                metadata=transaction,
            )
            return result, None, {}, transaction

    if residual_mode == "measure":
        video_residual = measure_residual_geometry(
            learned_clean[:, :, :prefix_t],
            exact_prefix,
            rigid_dx=float(video_estimate.dx),
            rigid_dy=float(video_estimate.dy),
        )
        guidance_residual = (
            transaction["guidance_registration"].get(
                "residual_geometry",
                {
                    "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
                    "status": "not_evaluated",
                    "reason": "guidance_residual_missing",
                    "eligible": False,
                },
            )
            if guidance_active
            else {"status": "off", "reason": "guidance_off"}
        )
        residual_geometry.update(
            measured=True,
            measurement_status="measured",
            reason="measurement_only",
            selected_model=str(video_residual.get("selected_model", "none")),
            decision="not_evaluated",
            applied=False,
            final_path="rigid_v2",
            video=video_residual,
            guidance=guidance_residual,
        )
    else:
        residual_geometry.update(final_path="rigid_v2")

    # Only now materialize the translated learned trajectory. The final clean
    # state needs the translated suffix and one translated provider-prefix
    # witness for the DC bridge. Measurement mode additionally keeps the last
    # six prefix frames aligned for its bounded evidence bundle.
    aligned_start_frame = max(0, prefix_t - 6) if residual_mode == "measure" else prefix_t - 1
    aligned_translation_started = time.perf_counter()
    aligned_application = translate_video_cells(
        learned_clean,
        dx=float(video_estimate.dx),
        dy=float(video_estimate.dy),
        start_frame=aligned_start_frame,
        batch_frames=4,
    )
    aligned_translation_elapsed_ms = (time.perf_counter() - aligned_translation_started) * 1000.0
    transaction["aligned_translation_elapsed_ms"] = aligned_translation_elapsed_ms
    transaction["aligned_translation_start_frame"] = aligned_start_frame
    if aligned_application.invalid_fraction > 0.08:
        raise RuntimeError("frame-gauge accepted boundary witness but full translation exceeded invalid-area bound")
    aligned_full = aligned_application.video

    diagnostic_end = min(
        int(learned_clean.shape[2]),
        prefix_t + 4,
    )
    selected_prefix_frames = min(prefix_t, 6)
    registration_pair_f64_bytes = (
        2
        * selected_prefix_frames
        * int(learned_clean.shape[1])
        * int(learned_clean.shape[-2])
        * int(learned_clean.shape[-1])
        * 8
    )
    registration_pair_f32_bytes = registration_pair_f64_bytes // 2
    translation_output_bytes = learned_clean.numel() * learned_clean.element_size()
    translation_batch_f32_bytes = (
        int(learned_clean.shape[0])
        * int(learned_clean.shape[1])
        * min(4, int(learned_clean.shape[2]))
        * int(learned_clean.shape[-2])
        * int(learned_clean.shape[-1])
        * 4
    )
    translation_grid_bytes = int(learned_clean.shape[-2]) * int(learned_clean.shape[-1]) * 2 * 4
    aligned_witness = aligned_full[:, :, :diagnostic_end].detach().clone()
    corrected_clean, dc_metrics = apply_suffix_dc_bridge(
        aligned_full,
        exact_prefix,
        weights=(1.0,),
        clone_output=False,
    )
    # The shifted prefix exists only as a disposable registration/DC witness.
    # Restore the provider's original learned prefix before the clean state is
    # re-noised; scheduler ownership will replace it with authoritative E later.
    corrected_clean[:, :, :prefix_t] = learned_clean[:, :, :prefix_t]
    if not torch.equal(
        corrected_clean[:, :, :prefix_t],
        learned_clean[:, :, :prefix_t],
    ):
        raise RuntimeError("frame-gauge clean transaction altered learned prefix ownership")

    if residual_mode == "measure":
        video_residual = residual_geometry.get("video", {})
        tile_bounds = video_residual.get("tile_bounds")
        if isinstance(tile_bounds, dict):
            residual_geometry["boundary_regional"] = _regional_boundary_motion_receipts(
                learned_clean,
                exact_prefix,
                aligned_witness,
                corrected_clean,
                prefix_t=prefix_t,
                tile_bounds=tile_bounds,
            )
        else:
            residual_geometry["boundary_regional"] = {
                "policy": "paired_prefix_residual_boundary_regions_v1",
                "status": "not_evaluated",
                "reason": "regional_measurement_unavailable",
            }

    video_registration = dict(transaction["video_registration"])
    video_registration["invalid_fraction"] = float(aligned_application.invalid_fraction)
    transaction["video_registration"] = video_registration
    transaction.update(
        result="accepted",
        reason="accepted",
        dc_applied_in_clean_hook=True,
        spatial_warp_applied=True,
        invalid_fraction=aligned_application.invalid_fraction,
        dc_policy="existing_one_token_spatial_mean_v1",
        dc_order="after_spatial_registration_before_conditional_renoise",
        dc_metrics=dc_metrics,
        workspace_upper_bound_bytes=(
            registration_pair_f64_bytes
            + registration_pair_f32_bytes
            + translation_output_bytes
            + translation_batch_f32_bytes
            + translation_grid_bytes
        ),
        workspace_components={
            "registration_pair_f64_bytes": registration_pair_f64_bytes,
            "registration_pair_f32_bytes": registration_pair_f32_bytes,
            "translation_output_bytes": translation_output_bytes,
            "translation_batch_f32_bytes": translation_batch_f32_bytes,
            "translation_grid_bytes": translation_grid_bytes,
        },
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    witnesses = {
        "learned_native": (learned_clean[:, :, :diagnostic_end].detach().clone()),
        "paired_prefix_aligned_witness": aligned_witness,
        "corrected_clean": (corrected_clean[:, :, :diagnostic_end].detach().clone()),
    }
    witnesses.update(residual_witnesses)
    result = CleanVideoPostprocessResult(
        clean_video=corrected_clean,
        protected_prefix_t=prefix_t,
        metadata=transaction,
    )
    return result, registered_reference, witnesses, transaction


def _bounded_residual_stage_slice(
    video: torch.Tensor,
    *,
    prefix_t: int,
    temporal_offset: int = 0,
) -> tuple[torch.Tensor, int, int, int]:
    local_prefix = int(prefix_t) - int(temporal_offset)
    if not 0 <= local_prefix <= int(video.shape[2]):
        raise RuntimeError("residual stage witness does not contain the declared prefix boundary")
    local_start = max(0, local_prefix - 6)
    local_stop = min(int(video.shape[2]), local_prefix + 4)
    return (
        video[:, :, local_start:local_stop].detach(),
        int(temporal_offset) + local_start,
        int(temporal_offset) + local_stop,
        local_prefix - local_start,
    )


def _emit_residual_geometry_stage(
    metrics,
    *,
    stage: str,
    video: torch.Tensor,
    prefix_t: int,
    session_id: str,
    chunk_id: str,
    domain: str,
    owner_before: str,
    owner_after: str,
    temporal_relation: str,
    applied_transform: str,
    provenance: str,
    temporal_offset: int = 0,
) -> dict[str, Any]:
    bounded, start, stop, local_prefix = _bounded_residual_stage_slice(
        video,
        prefix_t=prefix_t,
        temporal_offset=temporal_offset,
    )
    boundary = measure_video_boundary(bounded, local_prefix) if 0 < local_prefix < int(bounded.shape[2]) else {}
    fields: dict[str, Any] = {
        "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
        "stage": stage,
        "session_id": str(session_id),
        "chunk_id": str(chunk_id),
        "domain": domain,
        "owner_before": owner_before,
        "owner_after": owner_after,
        "temporal_relation": temporal_relation,
        "temporal_start": start,
        "temporal_stop": stop,
        "prefix_boundary_index": int(prefix_t),
        "dtype": str(video.dtype),
        "device": str(video.device),
        "height": int(video.shape[-2]),
        "width": int(video.shape[-1]),
        "normalization": "none_stage_receipt",
        "downsample": "none",
        "applied_transform": applied_transform,
        "provenance": provenance,
        "tensor_sha256": tensor_sha256(bounded),
        "bounded_prefix_frames": min(int(prefix_t), 6),
        "bounded_suffix_frames": max(0, stop - int(prefix_t)),
        "valid_support": "full_tensor_receipt",
        "caller_domain_comparison_allowed": domain == "model_internal_clean",
        **boundary,
    }
    metrics.event("partitioned_residual_geometry_stage", **fields)
    return fields


def _measure_partitioned_transfer_splice(
    target_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> dict[str, Any]:
    """Measure the learned-transfer seam immediately before exact-prefix restore.

    target_video is the target-grid conditional state returned by
    build_handoff_state. Reconstructing the clean learned state with the same
    deterministic handoff noise lets us compare two clean-domain boundaries:
    the learned upscaler's own prefix-last -> suffix-first boundary and the
    authoritative exact-prefix-last -> learned suffix-first boundary created
    when Flow discards the learned prefix.

    The tensors are diagnostics-only and never feed back into sampling.
    """

    diagnostic_started = time.perf_counter()
    learned_clean = _recover_partitioned_transfer_clean(
        target_video,
        sigma=float(sigma),
        seed=int(seed),
    )
    exact = exact_prefix.to(device=learned_clean.device, dtype=learned_clean.dtype)
    fields = measure_exact_prefix_splice(learned_clean, exact)
    fields.update(
        splice_diagnostic_elapsed_ms=(time.perf_counter() - diagnostic_started) * 1000.0,
        splice_recovery="inverse_conditional_renoise",
        splice_clean_source="inverse_recovered",
        splice_scope="learned_clean_before_exact_prefix_restore",
    )
    return fields


def run_partitioned_progressive(
    executor,
    guider,
    binding,
    config: ProgressiveTargetInputConfig,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
    exact_denoise_mask=None,
):
    """Execute exact-prefix low/probe/high continuation with a physical two-grid H3 stage."""
    chunk_started = time.perf_counter()
    if not isinstance(latent_shapes, list):
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires mutable latent-shape metadata")
    initial_model_options = getattr(guider, "model_options", None)
    initial_transformer = (
        initial_model_options.get("transformer_options") if isinstance(initial_model_options, dict) else None
    )
    if not isinstance(initial_transformer, dict):
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires mutable transformer options")
    vdn_linear_diagnostic = normalize_vdn_linear_diagnostic(
        initial_transformer.get(
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        )
    )
    prefix_transformer_context = normalize_prefix_transformer_context(
        initial_transformer.get(
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        )
    )
    audio_position_domain = normalize_audio_position_domain(
        initial_transformer.get(
            PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
            PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
        )
    )
    audio_handoff_source = normalize_audio_handoff_source(
        initial_transformer.get(
            PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY,
            PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        )
    )
    av_handoff_source = normalize_av_handoff_source(
        initial_transformer.get(
            PARTITIONED_AV_HANDOFF_SOURCE_KEY,
            PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        )
    )
    guidance_trajectory_source = normalize_guidance_trajectory_source(
        initial_transformer.get(
            PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
            PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        )
    )
    low_probe_execution_source = normalize_low_probe_execution_source(
        initial_transformer.get(
            PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
            PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
        )
    )
    audio_guided_overlap_mode, _audio_mode_source = resolve_partitioned_audio_guided_overlap_mode(initial_model_options)
    audio_guided_overlap_ticks, _audio_ticks_source = resolve_partitioned_audio_guided_overlap_ticks(
        initial_model_options
    )
    _validate_low_probe_execution_source_configuration(
        low_probe_execution_source,
        prefix_transformer_context=prefix_transformer_context,
        vdn_linear_diagnostic=vdn_linear_diagnostic,
        audio_position_domain=audio_position_domain,
        audio_handoff_source=audio_handoff_source,
        av_handoff_source=av_handoff_source,
        guidance_trajectory_source=guidance_trajectory_source,
        audio_guided_overlap_mode=audio_guided_overlap_mode,
        audio_guided_overlap_ticks=audio_guided_overlap_ticks,
    )
    if low_probe_execution_source != PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY:
        # In the collapsed source-only diagnostic, the shadow selectors above are
        # guard values proving the exact #64 control tuple; no shadow sampler
        # lifetime executes. Validate shadow-specific overlap requirements only
        # when those shadow lifetimes actually remain in the execution plan.
        _validate_audio_handoff_shadow_configuration(
            audio_handoff_source,
            prefix_transformer_context=prefix_transformer_context,
            vdn_linear_diagnostic=vdn_linear_diagnostic,
            audio_position_domain=audio_position_domain,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
        )
        _validate_av_handoff_shadow_configuration(
            av_handoff_source,
            audio_handoff_source=audio_handoff_source,
            prefix_transformer_context=prefix_transformer_context,
            vdn_linear_diagnostic=vdn_linear_diagnostic,
            audio_position_domain=audio_position_domain,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
        )
        _validate_guidance_trajectory_shadow_configuration(
            guidance_trajectory_source,
            av_handoff_source=av_handoff_source,
        )
    if guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW:
        if binding.guidance is None or binding.guidance.mode == "off":
            raise PartitionedPreflightUnsupported(
                "source_carrier_uniform_shadow guidance trajectory requires active Flow guidance"
            )
        if binding.trajectory is None or not binding.capture_enabled:
            raise PartitionedPreflightUnsupported(
                "source_carrier_uniform_shadow guidance trajectory requires enabled Flow trajectory capture"
            )
    if low_probe_execution_source == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY:
        sampler_calls_before = int(binding.metrics.counters.get("progressive_sampler_invocations", 0))
        history_boundaries_before = int(binding.metrics.counters.get("progressive_history_boundaries", 0))
        source_calls_before = int(
            binding.metrics.counters.get("partitioned_source_carrier_uniform_transformer_calls", 0)
        )
        partitioned_calls_before = int(binding.metrics.counters.get("partitioned_transformer_calls", 0))
        binding.metrics.event(
            "partitioned_low_probe_execution_plan",
            source=low_probe_execution_source,
            skipped_main_exact_partitioned_low_probe=True,
            raw_audio_owner="source_carrier_uniform_primary",
            clean_video_owner="source_carrier_uniform_primary_probe",
            guidance_trajectory_owner="source_carrier_uniform_primary",
            ui_audio_handoff_source=audio_handoff_source,
            ui_av_handoff_source=av_handoff_source,
            ui_guidance_trajectory_source=guidance_trajectory_source,
            duplicate_shadow_lifetimes_expected=0,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
            exact_target_prefix_restore_unchanged=True,
            learned_transfer_unchanged=True,
            target_high_unchanged=True,
            diagnostic_only=True,
        )
        _cuda_allocator_checkpoint(binding.metrics, "source_uniform_primary_entry")
        with _source_uniform_primary_execution_controls(initial_transformer):
            result = run_partitioned_progressive(
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
                exact_denoise_mask=exact_denoise_mask,
            )
        _cuda_allocator_checkpoint(binding.metrics, "source_uniform_primary_exit")
        sampler_delta = int(binding.metrics.counters.get("progressive_sampler_invocations", 0)) - sampler_calls_before
        history_delta = (
            int(binding.metrics.counters.get("progressive_history_boundaries", 0)) - history_boundaries_before
        )
        source_call_delta = (
            int(binding.metrics.counters.get("partitioned_source_carrier_uniform_transformer_calls", 0))
            - source_calls_before
        )
        partitioned_call_delta = (
            int(binding.metrics.counters.get("partitioned_transformer_calls", 0)) - partitioned_calls_before
        )
        if sampler_delta != 3 or history_delta != 2:
            raise RuntimeError(
                "source-uniform primary execution did not produce exactly low/probe/high sampler lifetimes"
            )
        if source_call_delta <= 0 or partitioned_call_delta != 0:
            raise RuntimeError(
                "source-uniform primary execution did not isolate the uniform low/probe transformer path"
            )
        binding.metrics.event(
            "partitioned_low_probe_execution_complete",
            source=low_probe_execution_source,
            sampler_invocation_delta=sampler_delta,
            history_boundary_delta=history_delta,
            source_uniform_transformer_calls=source_call_delta,
            exact_partitioned_transformer_calls=partitioned_call_delta,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
            skipped_main_exact_partitioned_low_probe=True,
            duplicate_shadow_lifetimes_executed=0,
            production_shaped_single_path=True,
            diagnostic_only=True,
        )
        return result
    if (
        prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
        and vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL
    ):
        raise PartitionedPreflightUnsupported(
            "source_carrier_uniform tests the ordinary uniform-grid VDN/Sol path and therefore "
            "requires vdn_linear_diagnostic='normal'"
        )
    _validate_audio_position_candidate_configuration(
        audio_position_domain,
        prefix_transformer_context,
        vdn_linear_diagnostic,
    )
    source_h, source_w, stage_plan = _preflight(
        guider,
        config,
        noise,
        latent_image,
        denoise_mask,
        latent_shapes,
        required_vdn_linear_diagnostic=vdn_linear_diagnostic,
    )

    # All unsupported conditions above are checked before any sampler lifetime.
    # From this point onward a failure is a real candidate failure and must not
    # silently restart the chunk under a different numerical path.
    _validate_progressive_sampler_state(sampler)
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise RuntimeError("partitioned exact-prefix requires a full H3 sigma schedule")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
        float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError("partitioned exact-prefix requires a full 1-to-0 H3 sigma schedule")

    target_shapes = list(latent_shapes)
    target_h, target_w = map(int, target_shapes[0][-2:])
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable model options")
    transformer = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("partitioned exact-prefix requires mutable transformer options")
    current_conds = getattr(guider, "conds", None)
    if not isinstance(current_conds, dict):
        raise RuntimeError("partitioned exact-prefix requires ComfyUI guider conditioning state")
    conditioning_template = {
        key: [entry.copy() if isinstance(entry, dict) else copy.copy(entry) for entry in entries]
        for key, entries in current_conds.items()
    }
    video_shift = float(transformer.get("minimax_h3_sigma_shift_video", H3_VIDEO_SHIFT))
    selected_coordinate = config.resolve_coordinate(source_h, source_w, target_h, target_w)
    from .handoff import select_handoff_index

    index = select_handoff_index(
        sigmas,
        selected_coordinate,
        min_high_steps=config.min_high_steps,
        video_shift=video_shift,
    )
    sigma = float(sigmas[index].item())
    low_sigmas = sigmas[: index + 1]
    high_sigmas = sigmas[index:]

    source_shapes = list(target_shapes)
    source_shapes[0] = (*source_shapes[0][:-2], source_h, source_w)
    target_video_noise, target_audio_noise = unpack_streams(noise, target_shapes)
    source_video_noise = deterministic_video_noise(
        (*target_video_noise.shape[:-2], source_h, source_w),
        seed=int(seed or 0) + config.source_noise_offset,
        device=target_video_noise.device,
        dtype=target_video_noise.dtype,
    )
    low_noise = pack_streams((source_video_noise, target_audio_noise))[0]
    low_latent_image = _resize_packed_latent_image(latent_image, target_shapes, source_shapes)
    low_video, low_audio = unpack_streams(low_latent_image, source_shapes)
    physical_prefix_source = resize_spatial_5d_h3_patch_lattice(
        stage_plan.prefix.to(low_video),
        source_h,
        source_w,
    )
    if int(physical_prefix_source.shape[2]) != int(stage_plan.prefix_t):
        raise RuntimeError("H3 physical prefix resample changed temporal ownership")
    generic_prefix_source = low_video[:, :, : stage_plan.prefix_t]
    prefix_resample_delta = physical_prefix_source.to(torch.float32) - generic_prefix_source.to(torch.float32)
    prefix_resample_delta_rms = float(prefix_resample_delta.square().mean().sqrt().item())
    prefix_resample_delta_abs_max = float(prefix_resample_delta.abs().max().item())
    low_video = low_video.clone()
    low_video[:, :, : stage_plan.prefix_t] = physical_prefix_source
    low_latent_image = pack_streams((low_video, low_audio))[0]
    binding.metrics.event(
        "partitioned_prefix_source_resample",
        policy="h3_physical_patch_lattice_v1",
        prefix_source="authoritative_exact_target_prefix",
        prefix_t=int(stage_plan.prefix_t),
        target_hw=(int(target_h), int(target_w)),
        source_hw=(int(source_h), int(source_w)),
        generic_half_pixel_prefix_replaced=True,
        generic_vs_physical_delta_rms=prefix_resample_delta_rms,
        generic_vs_physical_delta_abs_max=prefix_resample_delta_abs_max,
        numerical_change_observed=prefix_resample_delta_abs_max > 0.0,
        extra_h3_nfe=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
    )
    del prefix_resample_delta
    del low_video, low_audio
    low_mask = _resize_packed_mask(denoise_mask, target_shapes, source_shapes)

    diagnostic_target_mask, diagnostic_low_mask = _resolve_audio_diagnostic_masks(
        denoise_mask,
        exact_denoise_mask,
        target_shapes,
        source_shapes,
    )

    binding.metrics.increment("progressive_partitioned_exact_prefix_runs")
    binding.metrics.event(
        "partitioned_stage_plan",
        index=index,
        sigma=sigma,
        coordinate=float(normalized_coordinate(sigma, video_shift=video_shift)),
        requested_coordinate=config.handoff_coordinate,
        selected_coordinate=selected_coordinate,
        selection=config.handoff_selection,
        input_mode="partitioned_exact_prefix",
        transfer_mode="learned_3d_suffix_context",
        prefix_temporal_length=stage_plan.prefix_t,
        suffix_temporal_length=stage_plan.temporal - stage_plan.prefix_t,
        prefix_target_hw=stage_plan.target_hw,
        suffix_source_hw=(source_h, source_w),
        source_shape=source_shapes[0],
        target_hw=(target_h, target_w),
        prefix_video_rows=stage_plan.prefix_rows,
        suffix_video_rows=stage_plan.suffix_rows,
        partitioned_video_rows=stage_plan.partitioned_rows,
        prefix_exact_latent_resized_for_transformer=(
            prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
        ),
        prefix_target_grid_rows_injected=(prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT),
        deprecated_mixed_grid_contract_active=False,
        vdn_linear_diagnostic=vdn_linear_diagnostic,
        prefix_transformer_context=prefix_transformer_context,
        audio_position_domain=audio_position_domain,
    )
    if audio_handoff_source != PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN:
        binding.metrics.event(
            "partitioned_audio_handoff_shadow_plan",
            source=audio_handoff_source,
            main_prefix_transformer_context=prefix_transformer_context,
            main_audio_position_domain=audio_position_domain,
            shadow_prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
            shadow_audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
            vdn_linear_diagnostic=vdn_linear_diagnostic,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
            video_owner="main_exact_partitioned",
            audio_handoff_owner="source_carrier_uniform_shadow",
            diagnostic_only=True,
        )
    if av_handoff_source != PARTITIONED_AV_HANDOFF_SOURCE_MAIN:
        binding.metrics.event(
            "partitioned_av_handoff_shadow_plan",
            source=av_handoff_source,
            main_prefix_transformer_context=prefix_transformer_context,
            main_audio_position_domain=audio_position_domain,
            shadow_prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
            shadow_audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
            vdn_linear_diagnostic=vdn_linear_diagnostic,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
            raw_audio_owner="source_carrier_uniform_shadow",
            clean_video_owner="source_carrier_uniform_shadow_probe",
            exact_video_prefix_owner="main_target_prefix",
            clean_audio_probe_owner="main_exact_partitioned",
            guidance_trajectory_owner=(
                "source_carrier_uniform_shadow"
                if guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW
                else "main_exact_partitioned"
            ),
            guidance_trajectory_source=guidance_trajectory_source,
            diagnostic_only=True,
        )

    sampler_invocation_count = 0
    history_boundary_count = 0
    bypass_calls_before = int(binding.metrics.counters.get("partitioned_vdn_linear_bypass_calls", 0))
    bypass_video_rows_before = int(binding.metrics.counters.get("partitioned_vdn_linear_bypass_video_rows", 0))
    suppression_calls_before = int(
        binding.metrics.counters.get("partitioned_vdn_cross_grid_temporal_suppression_calls", 0)
    )
    suppressed_taps_before = int(binding.metrics.counters.get("partitioned_vdn_cross_grid_temporal_suppressed_taps", 0))
    suppressed_rows_before = int(binding.metrics.counters.get("partitioned_vdn_cross_grid_temporal_suppressed_rows", 0))
    raw_measure_calls_before = int(binding.metrics.counters.get("partitioned_vdn_raw_token_measure_calls", 0))
    raw_measure_prefix_frames_before = int(
        binding.metrics.counters.get("partitioned_vdn_raw_token_measure_prefix_frames", 0)
    )
    source_carrier_calls_before = int(
        binding.metrics.counters.get("partitioned_source_carrier_uniform_transformer_calls", 0)
    )
    source_carrier_prefix_frames_before = int(
        binding.metrics.counters.get("partitioned_source_carrier_uniform_prefix_frames", 0)
    )
    audio_position_calls_before = int(
        binding.metrics.counters.get("partitioned_audio_position_source_carrier_block0_calls", 0)
    )
    audio_position_wrapper_entries_before = int(
        binding.metrics.counters.get("partitioned_audio_position_candidate_wrapper_entries", 0)
    )
    audio_model_timestep_calls_before = int(
        binding.metrics.counters.get("partitioned_audio_model_timestep_override_calls", 0)
    )
    candidate_low_mask_digest = (
        tensor_sha256(low_mask) if audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE else None
    )
    candidate_high_mask_digest = (
        tensor_sha256(denoise_mask) if audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE else None
    )
    diagnostic_audio_control = (
        PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY in model_options
        or PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY in model_options
    )

    def low_callback(step, x0, x, _total):
        if callback is None:
            return None
        x0 = _resize_packed_latent_image(x0, source_shapes, target_shapes)
        x = _resize_packed_latent_image(x, source_shapes, target_shapes)
        return callback(step, x0, x, len(sigmas) - 1)

    _begin_capture(binding, guider, sampler, low_sigmas, source_shapes)
    try:
        _reset_guider_conds(guider, template=conditioning_template)
        low_started = time.perf_counter()
        try:
            sampler_invocation_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            with _flow_stage_contract(guider, "low"), _partitioned_stage_contract(guider, stage_plan, binding.metrics):
                low_result = executor(
                    low_noise,
                    low_latent_image,
                    sampler,
                    low_sigmas,
                    low_mask,
                    low_callback,
                    disable_pbar,
                    seed,
                    latent_shapes=source_shapes,
                )
        finally:
            binding.metrics.event(
                "low_stage_wall",
                elapsed_ms=(time.perf_counter() - low_started) * 1000.0,
                partitioned_exact_prefix=True,
            )

        base_model = guider.model_patcher.model
        source_raw = _raw_sampler_state(base_model, low_result, source_shapes, sigma)
        source_latent_internal = _process_latent_in(base_model, low_latent_image, source_shapes)
        active = binding.active_capture
        if active is not None:
            active.phases = (*active.phases, (index, "handoff_probe"))
        probe_started = time.perf_counter()
        probe_noise = _noise_argument(base_model, source_raw, sigma, source_latent_internal)
        previous_probe = transformer.get(PROBE_CONTEXT_KEY)
        transformer[PROBE_CONTEXT_KEY] = {"outer_step": index}
        try:
            _reset_guider_conds(guider, template=conditioning_template)
            sampler_invocation_count += 1
            history_boundary_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            binding.metrics.increment("progressive_history_boundaries")
            with (
                _flow_stage_contract(guider, "probe"),
                _high_stage_contract(guider),
                _partitioned_stage_contract(guider, stage_plan, binding.metrics),
            ):
                source_x0 = executor(
                    probe_noise,
                    low_latent_image,
                    _make_probe_sampler(sampler),
                    sigmas[index : index + 1],
                    low_mask,
                    None,
                    disable_pbar,
                    seed,
                    latent_shapes=source_shapes,
                )
        finally:
            if previous_probe is None:
                transformer.pop(PROBE_CONTEXT_KEY, None)
            else:
                transformer[PROBE_CONTEXT_KEY] = previous_probe
            binding.metrics.event(
                "handoff_probe_wall",
                elapsed_ms=(time.perf_counter() - probe_started) * 1000.0,
                partitioned_exact_prefix=True,
            )
    except BaseException as exc:
        _finish_capture(binding, error=exc)
        raise

    committed_low_run = _finish_capture(binding)
    binding.metrics.increment("handoff_exact_probe_nfe")
    _cuda_allocator_checkpoint(binding.metrics, "after_primary_probe")
    try:
        _verify_prefix_transformer_context_diagnostic(
            binding.metrics,
            prefix_transformer_context,
            calls_before=source_carrier_calls_before,
            prefix_frames_before=source_carrier_prefix_frames_before,
        )
        _verify_audio_position_domain_diagnostic(
            binding.metrics,
            audio_position_domain,
            block0_calls_before=audio_position_calls_before,
            wrapper_entries_before=audio_position_wrapper_entries_before,
            model_timestep_calls_before=audio_model_timestep_calls_before,
        )
        if audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE and (
            tensor_sha256(low_mask) != candidate_low_mask_digest
            or tensor_sha256(denoise_mask) != candidate_high_mask_digest
        ):
            raise RuntimeError("source-carrier audio-position candidate mutated sampler masks during low/probe")
        if prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
            _verify_partitioned_vdn_linear_diagnostic(
                binding.metrics,
                vdn_linear_diagnostic,
                bypass_calls_before=bypass_calls_before,
                bypass_video_rows_before=bypass_video_rows_before,
                suppression_calls_before=suppression_calls_before,
                suppressed_taps_before=suppressed_taps_before,
                suppressed_rows_before=suppressed_rows_before,
                raw_measure_calls_before=raw_measure_calls_before,
                raw_measure_prefix_frames_before=raw_measure_prefix_frames_before,
            )

        shadow_source_raw = None
        if audio_handoff_source == PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW:
            shadow_calls_before = int(
                binding.metrics.counters.get("partitioned_source_carrier_uniform_transformer_calls", 0)
            )
            shadow_prefix_frames_before = int(
                binding.metrics.counters.get("partitioned_source_carrier_uniform_prefix_frames", 0)
            )
            shadow_audio_timestep_calls_before = int(
                binding.metrics.counters.get("partitioned_audio_model_timestep_override_calls", 0)
            )
            shadow_started = time.perf_counter()
            with _source_uniform_audio_shadow_controls(transformer):
                _reset_guider_conds(guider, template=conditioning_template)
                sampler_invocation_count += 1
                history_boundary_count += 1
                binding.metrics.increment("progressive_sampler_invocations")
                binding.metrics.increment("progressive_history_boundaries")
                binding.metrics.increment("partitioned_audio_handoff_shadow_low_invocations")
                with _source_uniform_audio_shadow_sampler_contract(
                    guider,
                    stage_plan,
                    binding.metrics,
                ):
                    shadow_low_result = executor(
                        low_noise,
                        low_latent_image,
                        sampler,
                        low_sigmas,
                        low_mask,
                        None,
                        disable_pbar,
                        seed,
                        latent_shapes=source_shapes,
                    )
                shadow_source_raw = _raw_sampler_state(
                    base_model,
                    shadow_low_result,
                    source_shapes,
                    sigma,
                )

            _verify_prefix_transformer_context_diagnostic(
                binding.metrics,
                PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
                calls_before=shadow_calls_before,
                prefix_frames_before=shadow_prefix_frames_before,
            )
            shadow_audio_timestep_calls = (
                int(binding.metrics.counters.get("partitioned_audio_model_timestep_override_calls", 0))
                - shadow_audio_timestep_calls_before
            )
            if shadow_audio_timestep_calls <= 0:
                raise RuntimeError("source-uniform audio handoff shadow observed no model-timestep audio guidance")
            binding.metrics.event(
                "partitioned_audio_handoff_shadow_execution",
                source=audio_handoff_source,
                elapsed_ms=(time.perf_counter() - shadow_started) * 1000.0,
                model_timestep_override_calls=shadow_audio_timestep_calls,
                main_pre_high_video_state_used=True,
                shadow_video_state_discarded=True,
                shadow_audio_state_selected=True,
                shadow_probe_executed=False,
                handoff_state_source="shadow_low_raw_audio",
                separate_sampler_lifetime=True,
                flow_stage_marker=None,
                vdn_quiescent_release_boundary="outer_complete",
                joint_high_video_can_depend_on_shadow_audio=True,
                fail_closed=True,
            )

        shadow_source_x0 = None
        shadow_guidance_run = None
        if av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
            if binding.active_capture is not None:
                raise RuntimeError("AV handoff shadow must begin after the main Flow trajectory is committed")
            if committed_low_run is None or binding.captured_run_id != committed_low_run.run_id:
                raise RuntimeError("AV handoff shadow lost main exact-partitioned trajectory ownership")
            main_captured_run_id = binding.captured_run_id
            main_trajectory_handle = binding.trajectory
            if main_trajectory_handle is None:
                raise RuntimeError("AV handoff shadow lost the shared main Flow trajectory handle")
            shadow_calls_before = int(
                binding.metrics.counters.get("partitioned_source_carrier_uniform_transformer_calls", 0)
            )
            shadow_prefix_frames_before = int(
                binding.metrics.counters.get("partitioned_source_carrier_uniform_prefix_frames", 0)
            )
            shadow_audio_timestep_calls_before = int(
                binding.metrics.counters.get("partitioned_audio_model_timestep_override_calls", 0)
            )
            shadow_started = time.perf_counter()
            shadow_event_start = len(binding.metrics.events)
            capture_shadow_guidance = guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW
            with (
                _isolated_shadow_trajectory_capture(
                    binding,
                    guider,
                    sampler,
                    low_sigmas,
                    source_shapes,
                    enabled=capture_shadow_guidance,
                ) as shadow_capture,
                _source_uniform_audio_shadow_controls(transformer),
            ):
                _reset_guider_conds(guider, template=conditioning_template)
                sampler_invocation_count += 1
                history_boundary_count += 1
                binding.metrics.increment("progressive_sampler_invocations")
                binding.metrics.increment("progressive_history_boundaries")
                binding.metrics.increment("partitioned_av_handoff_shadow_low_invocations")
                with _source_uniform_av_shadow_stage_contract(
                    guider,
                    stage_plan,
                    binding.metrics,
                    "low",
                ):
                    shadow_low_result = executor(
                        low_noise,
                        low_latent_image,
                        sampler,
                        low_sigmas,
                        low_mask,
                        None,
                        disable_pbar,
                        seed,
                        latent_shapes=source_shapes,
                    )
                shadow_source_raw = _raw_sampler_state(
                    base_model,
                    shadow_low_result,
                    source_shapes,
                    sigma,
                )
                shadow_probe_noise = _noise_argument(
                    base_model,
                    shadow_source_raw,
                    sigma,
                    source_latent_internal,
                )
                previous_shadow_probe = transformer.get(PROBE_CONTEXT_KEY)
                transformer[PROBE_CONTEXT_KEY] = {"outer_step": index}
                try:
                    _reset_guider_conds(guider, template=conditioning_template)
                    sampler_invocation_count += 1
                    history_boundary_count += 1
                    binding.metrics.increment("progressive_sampler_invocations")
                    binding.metrics.increment("progressive_history_boundaries")
                    binding.metrics.increment("partitioned_av_handoff_shadow_probe_invocations")
                    binding.metrics.increment("partitioned_av_handoff_shadow_probe_nfe")
                    with _source_uniform_av_shadow_stage_contract(
                        guider,
                        stage_plan,
                        binding.metrics,
                        "probe",
                    ):
                        shadow_probe_result = executor(
                            shadow_probe_noise,
                            low_latent_image,
                            _make_probe_sampler(sampler),
                            sigmas[index : index + 1],
                            low_mask,
                            None,
                            disable_pbar,
                            seed,
                            latent_shapes=source_shapes,
                        )
                finally:
                    if previous_shadow_probe is None:
                        transformer.pop(PROBE_CONTEXT_KEY, None)
                    else:
                        transformer[PROBE_CONTEXT_KEY] = previous_shadow_probe
                shadow_source_x0 = _process_latent_in(
                    base_model,
                    shadow_probe_result,
                    source_shapes,
                )

            if capture_shadow_guidance:
                if shadow_capture is None or "run" not in shadow_capture:
                    raise RuntimeError("source-uniform shadow guidance trajectory was not committed")
                shadow_guidance_run = shadow_capture["run"]
            if binding.active_capture is not None:
                raise RuntimeError("AV handoff shadow unexpectedly retained an active Flow trajectory capture")
            if binding.trajectory is not main_trajectory_handle:
                raise RuntimeError("AV handoff shadow failed to restore the shared main Flow trajectory handle")
            if binding.captured_run_id != main_captured_run_id:
                raise RuntimeError("AV handoff shadow replaced the main exact-partitioned Flow trajectory")
            if shadow_guidance_run is not None and any(
                run.run_id == shadow_guidance_run.run_id for run in main_trajectory_handle.runs
            ):
                raise RuntimeError("isolated shadow guidance trajectory leaked into the shared trajectory store")
            shadow_model_calls = [
                event for event in binding.metrics.events[shadow_event_start:] if event.kind == "model_call"
            ]
            shadow_low_calls = [event for event in shadow_model_calls if event.fields.get("stage") == "low"]
            shadow_probe_calls = [event for event in shadow_model_calls if event.fields.get("stage") == "probe"]
            if not shadow_low_calls:
                raise RuntimeError("source-uniform AV handoff shadow produced no low-stage H3 evaluations")
            if len(shadow_probe_calls) != 1 or not bool(shadow_probe_calls[0].fields.get("actual")):
                raise RuntimeError(
                    "source-uniform AV handoff shadow did not produce exactly one actual probe evaluation"
                )
            _verify_prefix_transformer_context_diagnostic(
                binding.metrics,
                PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
                calls_before=shadow_calls_before,
                prefix_frames_before=shadow_prefix_frames_before,
            )
            shadow_audio_timestep_calls = (
                int(binding.metrics.counters.get("partitioned_audio_model_timestep_override_calls", 0))
                - shadow_audio_timestep_calls_before
            )
            if shadow_audio_timestep_calls <= 0:
                raise RuntimeError("source-uniform AV handoff shadow observed no model-timestep audio guidance")
            _cuda_allocator_checkpoint(binding.metrics, "after_shadow_probe")
            binding.metrics.event(
                "partitioned_av_handoff_shadow_execution",
                source=av_handoff_source,
                elapsed_ms=(time.perf_counter() - shadow_started) * 1000.0,
                model_timestep_override_calls=shadow_audio_timestep_calls,
                shadow_low_model_calls=len(shadow_low_calls),
                shadow_probe_model_calls=len(shadow_probe_calls),
                shadow_probe_first_call_actual=bool(shadow_probe_calls[0].fields.get("actual")),
                shadow_low_executed=True,
                shadow_probe_executed=True,
                separate_sampler_lifetimes=2,
                vdn_low_stage_retained_until_probe=True,
                vdn_release_boundary="probe_to_high",
                guidance_trajectory_source=guidance_trajectory_source,
                main_guidance_trajectory_preserved=not capture_shadow_guidance,
                shadow_trajectory_captured=capture_shadow_guidance,
                shadow_trajectory_run_id=(shadow_guidance_run.run_id if shadow_guidance_run is not None else None),
                shadow_trajectory_exact_samples=(
                    len(shadow_guidance_run.exact_samples()) if shadow_guidance_run is not None else 0
                ),
                raw_audio_state_selected=True,
                clean_video_probe_selected=True,
                shadow_raw_video_discarded=True,
                shadow_clean_audio_discarded=True,
                fail_closed=True,
            )

        source_x0 = _process_latent_in(base_model, source_x0, source_shapes)

        # The learned 3D upscaler may use all prefix frames as transient temporal
        # context.  This resized copy never enters H3 attention and its upscaled
        # prefix output is discarded below in favor of the authoritative target
        # prefix captured before the low stage.
        clean_video, clean_audio = unpack_streams(source_x0, source_shapes)
        low_probe_clean_audio = clean_audio.detach().clone() if diagnostic_audio_control else None
        if diagnostic_audio_control:
            low_probe_audio_report = measure_audio_latent_boundary(
                source_x0,
                source_shapes,
                diagnostic_low_mask,
                windows=(4, 20),
            )
            binding.metrics.event(
                "partitioned_audio_stage_boundary",
                stage="low_probe_clean",
                domain="model_internal_clean",
                **low_probe_audio_report,
            )
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            source_native_trajectory = measure_translation_trajectory(
                clean_video,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="source_low_native",
                roi=roi_name,
                source_hw=(source_h, source_w),
                target_hw=(target_h, target_w),
                **source_native_trajectory,
                **project_translation_trajectory_to_grid(
                    source_native_trajectory,
                    source_hw=(source_h, source_w),
                    target_hw=(target_h, target_w),
                ),
            )
        exact_prefix_source = physical_prefix_source.to(clean_video)
        if av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
            if shadow_source_x0 is None or shadow_source_raw is None:
                raise RuntimeError("source-uniform AV handoff shadow lost its low/probe state")
            shadow_clean_video, _shadow_clean_audio = unpack_streams(shadow_source_x0, source_shapes)
            for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
                shadow_native_trajectory = measure_translation_trajectory(
                    shadow_clean_video,
                    stage_plan.prefix_t,
                    forward_steps=4,
                    backward_steps=3,
                    roi_fraction=roi_fraction,
                    max_shift=4,
                )
                binding.metrics.event(
                    "partitioned_multiframe_trajectory",
                    stage="source_uniform_shadow_clean_native",
                    roi=roi_name,
                    source_hw=(source_h, source_w),
                    target_hw=(target_h, target_w),
                    **shadow_native_trajectory,
                    **project_translation_trajectory_to_grid(
                        shadow_native_trajectory,
                        source_hw=(source_h, source_w),
                        target_hw=(target_h, target_w),
                    ),
                )
            source_x0 = _select_source_uniform_shadow_clean_video(
                source_x0,
                shadow_source_x0,
                source_shapes,
                prefix_t=stage_plan.prefix_t,
                exact_prefix_source=exact_prefix_source,
                metrics=binding.metrics,
            )
            clean_video, clean_audio = unpack_streams(source_x0, source_shapes)
        else:
            clean_video = clean_video.clone()
            clean_video[:, :, : stage_plan.prefix_t] = exact_prefix_source
            source_x0 = pack_streams((clean_video, clean_audio))[0]

        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            source_exact_trajectory = measure_translation_trajectory(
                clean_video,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="source_low_exact_context",
                roi=roi_name,
                source_hw=(source_h, source_w),
                target_hw=(target_h, target_w),
                **source_exact_trajectory,
                **project_translation_trajectory_to_grid(
                    source_exact_trajectory,
                    source_hw=(source_h, source_w),
                    target_hw=(target_h, target_w),
                ),
            )

        if audio_handoff_source == PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW:
            if shadow_source_raw is None:
                raise RuntimeError("source-uniform audio handoff shadow lost its sampler state")
            source_raw = _splice_source_uniform_shadow_audio_state(
                source_raw,
                shadow_source_raw,
                source_shapes,
                low_mask,
                binding.metrics,
            )
        elif av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
            source_raw = _splice_source_uniform_shadow_audio_state(
                source_raw,
                shadow_source_raw,
                source_shapes,
                low_mask,
                binding.metrics,
            )

        session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
        guidance_run = None
        if binding.guidance is not None and binding.guidance.mode != "off":
            if binding.trajectory is None:
                raise RuntimeError("partitioned exact-prefix Flow guidance requires an H3_FLOW_TRAJECTORY")
            expected_signature = binding.guidance_conditioning_signature or _conditioning_signature(guider)
            if guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW:
                guidance_run = shadow_guidance_run
                if guidance_run is None:
                    raise RuntimeError("source-uniform shadow guidance trajectory is unavailable")
                if guidance_run.chunk_id != str(chunk_id) or guidance_run.session_id != str(session_id):
                    raise RuntimeError("source-uniform shadow guidance trajectory identity drifted")
                if guidance_run.conditioning_signature != expected_signature:
                    raise RuntimeError("source-uniform shadow guidance trajectory conditioning drifted")
            else:
                guidance_run = binding.trajectory.select(
                    chunk_id=chunk_id,
                    session_id=session_id,
                    conditioning_signature=expected_signature,
                )
            if guidance_run.geometry.latent_t != int(target_shapes[0][2]):
                raise RuntimeError("partitioned low trajectory and target video temporal geometry differ")

        split_coordinate = float(normalized_coordinate(sigma, video_shift=video_shift))
        residual_mode = normalize_residual_geometry_mode(config.frame_gauge_residual_mode)
        boundary_content_diagnostic_enabled = bool(config.frame_gauge_repair)
        boundary_content_pre_high_receipt: dict[str, Any] | None = None
        pending_registered_reference = None
        frame_gauge_witnesses: dict[str, torch.Tensor] = {}
        residual_evidence_tensors: dict[str, torch.Tensor] = {}
        residual_stage_receipts: list[dict[str, Any]] = []
        frame_gauge_transaction: dict[str, Any] = {
            "policy_version": FRAME_GAUGE_POLICY_VERSION,
            "result": "off",
            "reason": "disabled",
            "video_registration": {
                "status": "off",
                "reason": "disabled",
            },
            "guidance_registration": {
                "status": "off",
                "reason": (
                    "guidance_off" if binding.guidance is None or binding.guidance.mode == "off" else "repair_disabled"
                ),
            },
            "dc_applied_in_clean_hook": False,
            "spatial_warp_applied": False,
            "residual_geometry": {
                "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
                "requested_mode": residual_mode,
                "measured": False,
                "measurement_status": "off" if residual_mode == "off" else "not_evaluated",
                "reason": "disabled" if residual_mode == "off" else "frame_gauge_repair_disabled",
                "selected_model": "none",
                "decision": "not_evaluated",
                "applied": False,
                "final_path": "baseline",
                "video": {
                    "status": "off" if residual_mode == "off" else "not_evaluated",
                    "reason": "disabled" if residual_mode == "off" else "frame_gauge_repair_disabled",
                },
                "guidance": (
                    {"status": "off", "reason": "guidance_off"}
                    if binding.guidance is None or binding.guidance.mode == "off"
                    else {
                        "status": "off" if residual_mode == "off" else "not_evaluated",
                        "reason": "disabled" if residual_mode == "off" else "frame_gauge_repair_disabled",
                    }
                ),
            },
        }

        clean_video_postprocess = None
        if config.frame_gauge_repair:

            def clean_video_postprocess(learned_clean):
                nonlocal pending_registered_reference
                nonlocal frame_gauge_witnesses
                nonlocal frame_gauge_transaction

                result, registered, witnesses, transaction = _frame_gauge_clean_postprocess(
                    learned_clean,
                    exact_prefix=stage_plan.prefix,
                    guidance_run=guidance_run,
                    guidance=binding.guidance,
                    target_h=target_h,
                    target_w=target_w,
                    prefix_t=stage_plan.prefix_t,
                    split_coordinate=split_coordinate,
                    high_sigmas=high_sigmas,
                    video_shift=video_shift,
                    residual_mode=config.frame_gauge_residual_mode,
                )
                pending_registered_reference = registered
                frame_gauge_witnesses = witnesses
                frame_gauge_transaction = transaction
                return result

        transfer_started = time.perf_counter()
        transfer_metrics: dict[str, Any] = {}
        target_raw, rebuilt_shapes = build_handoff_state(
            source_packed_state=source_raw,
            source_x0_packed=source_x0,
            source_shapes=source_shapes,
            sigma=sigma,
            target_h=target_h,
            target_w=target_w,
            seed=int(seed or 0) + config.seed_offset,
            transfer_mode="learned_3d",
            learned_upscaler=config.learned_upscaler,
            transfer_metrics=transfer_metrics,
            clean_video_postprocess=clean_video_postprocess,
        )
        if rebuilt_shapes != target_shapes:
            raise RuntimeError("partitioned exact-prefix handoff changed caller-visible AV geometry")
        target_video, target_audio = unpack_streams(
            target_raw,
            target_shapes,
        )
        if diagnostic_audio_control:
            _source_state_video, source_state_audio = unpack_streams(
                source_raw,
                source_shapes,
            )
            audio_copy_exact = torch.equal(
                source_state_audio,
                target_audio,
            )
            audio_copy_delta = target_audio.to(torch.float32) - source_state_audio.to(
                device=target_audio.device,
                dtype=torch.float32,
            )
            binding.metrics.event(
                "partitioned_audio_handoff_copy",
                exact=audio_copy_exact,
                max_abs_delta=float(audio_copy_delta.abs().max().item()),
                rms_delta=float(audio_copy_delta.square().mean().sqrt().item()),
                source_ticks=int(source_state_audio.shape[-1]),
                target_ticks=int(target_audio.shape[-1]),
                learned_video_transfer_audio_mutation=False,
            )
            if not audio_copy_exact:
                raise RuntimeError("learned video handoff mutated the carried H3 audio sampler state")

        diagnostic_seed = int(seed or 0) + config.seed_offset
        exact_prefix = stage_plan.prefix.to(
            device=target_video.device,
            dtype=target_video.dtype,
        )
        frame_gauge_accepted = frame_gauge_transaction.get("result") == "accepted"
        exact_overlap_fallback_requested, exact_overlap_fallback_trigger = (
            _partitioned_exact_overlap_fallback_eligibility(frame_gauge_transaction)
            if config.frame_gauge_repair and not frame_gauge_accepted
            else (False, "rigid_v2_selected" if frame_gauge_accepted else "frame_gauge_repair_disabled")
        )
        representation_metrics = disabled_suffix_representation_bridge_metrics(
            prefix_t=stage_plan.prefix_t,
            requested=False,
        )
        splice_started = time.perf_counter()
        aligned_witness = None
        if frame_gauge_accepted:
            required_witnesses = {
                "learned_native",
                "paired_prefix_aligned_witness",
                "corrected_clean",
            }
            if not required_witnesses.issubset(frame_gauge_witnesses):
                raise RuntimeError("accepted frame-gauge transaction lost required clean-domain witnesses")
            learned_clean = frame_gauge_witnesses["learned_native"]
            aligned_witness = frame_gauge_witnesses["paired_prefix_aligned_witness"]
            corrected_clean = frame_gauge_witnesses["corrected_clean"]
            dc_metrics = frame_gauge_transaction.get("dc_metrics")
            if not isinstance(dc_metrics, dict):
                raise RuntimeError("accepted frame-gauge transaction lost the existing DC bridge receipt")
            dc_metrics = dict(dc_metrics)
            splice_recovery = "actual_provider_clean_postprocess"
        else:
            if pending_registered_reference is not None:
                raise RuntimeError("rejected frame-gauge transaction published a guidance reference")
            learned_clean = _recover_partitioned_transfer_clean(
                target_video,
                sigma=sigma,
                seed=diagnostic_seed,
            )
            if exact_overlap_fallback_requested:
                learned_boundary_pair = frame_gauge_witnesses.get("learned_boundary_pair")
                if learned_boundary_pair is None:
                    raise RuntimeError("eligible exact-overlap fallback lost the actual provider boundary witness")
                expected_boundary_shape = (
                    int(learned_clean.shape[0]),
                    int(learned_clean.shape[1]),
                    2,
                    int(learned_clean.shape[-2]),
                    int(learned_clean.shape[-1]),
                )
                if tuple(learned_boundary_pair.shape) != expected_boundary_shape:
                    raise RuntimeError("eligible exact-overlap fallback provider boundary witness geometry drifted")
                learned_clean = learned_clean.clone()
                learned_clean[:, :, stage_plan.prefix_t - 1 : stage_plan.prefix_t + 1] = learned_boundary_pair.to(
                    learned_clean
                )
                target_video, corrected_clean, representation_metrics, dc_metrics = (
                    _apply_partitioned_exact_overlap_bridge(
                        target_video,
                        learned_clean,
                        exact_prefix,
                        sigma=sigma,
                    )
                )
            else:
                target_video, corrected_clean, dc_metrics = _apply_partitioned_suffix_dc_bridge(
                    target_video,
                    learned_clean,
                    exact_prefix,
                    sigma=sigma,
                    enabled=True,
                )
            splice_recovery = (
                "inverse_conditional_renoise_with_actual_provider_boundary_pair"
                if exact_overlap_fallback_requested
                else "inverse_conditional_renoise"
            )

        splice_diagnostics = measure_exact_prefix_splice(
            learned_clean,
            exact_prefix,
            corrected_clean_video=corrected_clean,
        )
        splice_diagnostics.update(
            splice_diagnostic_elapsed_ms=(time.perf_counter() - splice_started) * 1000.0,
            splice_recovery=splice_recovery,
            splice_clean_source=(
                "actual_provider"
                if frame_gauge_accepted
                else "actual_provider_boundary_pair_plus_inverse_recovered"
                if exact_overlap_fallback_requested
                else "inverse_recovered"
            ),
            splice_scope="learned_clean_before_exact_prefix_restore",
        )

        video_registration = frame_gauge_transaction.get(
            "video_registration",
            {},
        )
        guidance_registration = frame_gauge_transaction.get(
            "guidance_registration",
            {},
        )
        postprocess_report = transfer_metrics.get("clean_video_postprocess")
        if not isinstance(postprocess_report, dict):
            postprocess_report = {}
        residual_geometry_receipt = frame_gauge_transaction.get(
            "residual_geometry",
            {
                "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
                "requested_mode": residual_mode,
                "measured": False,
                "measurement_status": "not_evaluated",
                "reason": "missing_transaction_receipt",
                "decision": "not_evaluated",
                "applied": False,
                "final_path": "baseline",
            },
        )
        residual_telemetry_bytes = len(
            json.dumps(
                residual_geometry_receipt,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        if residual_telemetry_bytes > 128 * 1024:
            raise RuntimeError("residual geometry telemetry exceeded the 128 KiB transaction bound")
        binding.metrics.event(
            "partitioned_frame_gauge",
            mode="on" if config.frame_gauge_repair else "off",
            enabled=bool(config.frame_gauge_repair),
            eligible=bool(
                config.frame_gauge_repair
                and frame_gauge_transaction.get("result") in {"accepted", "identity", "rejected"}
            ),
            result=str(frame_gauge_transaction.get("result", "off")),
            reason=str(frame_gauge_transaction.get("reason", "unknown")),
            policy_version=FRAME_GAUGE_POLICY_VERSION,
            split_coordinate=split_coordinate,
            video_dx=float(video_registration.get("dx", 0.0)),
            video_dy=float(video_registration.get("dy", 0.0)),
            guidance_dx=float(guidance_registration.get("dx", 0.0)),
            guidance_dy=float(guidance_registration.get("dy", 0.0)),
            video_registration=video_registration,
            guidance_registration=guidance_registration,
            boundary_motion=frame_gauge_transaction.get(
                "boundary_motion",
                {"status": "off", "reason": "disabled"},
            ),
            spatial_warp_applied=bool(
                frame_gauge_transaction.get(
                    "spatial_warp_applied",
                    False,
                )
            ),
            dc_bridge_applied=True,
            dc_policy="existing_one_token_spatial_mean_v1",
            dc_order=(
                "after_spatial_registration_before_conditional_renoise"
                if frame_gauge_accepted
                else "historical_post_renoise_affine_mapping"
            ),
            deterministic_noise_seed=diagnostic_seed,
            deterministic_noise_seed_offset=int(config.seed_offset),
            mask_classification="exact_protected_video_prefix",
            exact_prefix_sha256=tensor_sha256(stage_plan.prefix),
            authoritative_prefix_modified=False,
            registration_domain=("actual_clean_target_video" if config.frame_gauge_repair else "off"),
            transform_domain=("actual_clean_target_video" if frame_gauge_accepted else "none"),
            transformed_states=(
                ["learned_suffix", "derived_guidance_suffix"]
                if frame_gauge_accepted and pending_registered_reference is not None
                else ["learned_suffix"]
                if frame_gauge_accepted
                else []
            ),
            provider_output_observed_before_noise=bool(config.frame_gauge_repair),
            provider_api_version=transfer_metrics.get("provider_api_version"),
            provider_kind=transfer_metrics.get("provider_kind"),
            provider_model_name=transfer_metrics.get("model_name"),
            actual_clean_dtype=postprocess_report.get("clean_dtype"),
            actual_clean_device=postprocess_report.get("clean_device"),
            guidance_mode=(binding.guidance.mode if binding.guidance is not None else "off"),
            registered_guidance_reference=bool(pending_registered_reference is not None),
            target_temporal_search_radius=guidance_registration.get("target_temporal_radius"),
            extra_h3_nfe=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            auto_strength_owner="dora_dynamic_lora_loader",
            auto_strength_receipt_status="unknown",
            auto_strength_resolved_off=None,
            auto_strength_validation_required=True,
            workspace_upper_bound_bytes=frame_gauge_transaction.get("workspace_upper_bound_bytes"),
            workspace_components=frame_gauge_transaction.get(
                "workspace_components",
                {},
            ),
            transaction_elapsed_ms=float(frame_gauge_transaction.get("elapsed_ms", 0.0)),
            boundary_translation_elapsed_ms=float(frame_gauge_transaction.get("boundary_translation_elapsed_ms", 0.0)),
            guidance_registration_elapsed_ms=float(
                frame_gauge_transaction.get("guidance_registration_elapsed_ms", 0.0)
            ),
            aligned_translation_elapsed_ms=float(frame_gauge_transaction.get("aligned_translation_elapsed_ms", 0.0)),
            aligned_translation_start_frame=frame_gauge_transaction.get("aligned_translation_start_frame"),
            residual_geometry=residual_geometry_receipt,
            residual_geometry_telemetry_bytes=residual_telemetry_bytes,
            exact_overlap_fallback_requested=bool(exact_overlap_fallback_requested),
            exact_overlap_fallback_trigger=str(exact_overlap_fallback_trigger),
            exact_overlap_fallback_applied=bool(
                representation_metrics.get("suffix_representation_bridge_accepted", False)
            ),
            exact_overlap_fallback_policy=PARTITIONED_EXACT_OVERLAP_POLICY,
            exact_overlap_fallback_source=(
                "actual_provider_boundary_pair" if exact_overlap_fallback_requested else "not_used"
            ),
            exact_overlap_fallback_transformed_states=(
                ["learned_suffix_first"]
                if representation_metrics.get("suffix_representation_bridge_accepted", False)
                else []
            ),
        )

        binding.metrics.event(
            "partitioned_exact_overlap_bridge",
            policy=PARTITIONED_EXACT_OVERLAP_POLICY,
            requested=bool(exact_overlap_fallback_requested),
            trigger=str(exact_overlap_fallback_trigger),
            applied=bool(representation_metrics.get("suffix_representation_bridge_accepted", False)),
            state_mapping=(
                "conditional_renoise_affine"
                if representation_metrics.get("suffix_representation_bridge_accepted", False)
                else "disabled_or_noop"
            ),
            authoritative_prefix_modified=False,
            later_suffix_extrapolated=False,
            extra_h3_nfe=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            **representation_metrics,
        )

        restored_clean = corrected_clean.clone()
        restored_clean[:, :, : stage_plan.prefix_t] = exact_prefix.to(restored_clean)

        if boundary_content_diagnostic_enabled:
            boundary_content_started = time.perf_counter()
            provider_receipt = measure_boundary_content_continuity(
                learned_clean,
                stage_plan.prefix_t,
            )
            binding.metrics.event(
                "partitioned_boundary_content_continuity",
                stage="provider_native",
                domain="model_internal_clean",
                owner_before="learned_provider_prefix",
                owner_after="learned_provider_suffix",
                elapsed_ms=(time.perf_counter() - boundary_content_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **provider_receipt,
            )
            predictor_started = time.perf_counter()
            provider_predictor_receipt = measure_provider_boundary_temporal_predictor(
                learned_clean,
                stage_plan.prefix_t,
            )
            binding.metrics.event(
                "partitioned_provider_boundary_predictor",
                domain="model_internal_clean",
                owner_before="learned_provider_prefix",
                owner_after="learned_provider_suffix",
                elapsed_ms=(time.perf_counter() - predictor_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **provider_predictor_receipt,
            )
            calibration_started = time.perf_counter()
            provider_calibration_receipt = measure_provider_boundary_temporal_calibration(
                learned_clean,
                stage_plan.prefix_t,
            )
            binding.metrics.event(
                "partitioned_provider_boundary_predictor_calibration",
                domain="model_internal_clean",
                owner_before="learned_provider_prefix",
                owner_after="learned_provider_suffix",
                elapsed_ms=(time.perf_counter() - calibration_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **provider_calibration_receipt,
            )
            boundary_content_started = time.perf_counter()
            boundary_content_pre_high_receipt = measure_boundary_content_continuity(
                restored_clean,
                stage_plan.prefix_t,
            )
            binding.metrics.event(
                "partitioned_boundary_content_continuity",
                stage="pre_high_exact_restored",
                domain="model_internal_clean",
                owner_before="authoritative_exact_prefix",
                owner_after="corrected_learned_suffix",
                elapsed_ms=(time.perf_counter() - boundary_content_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **boundary_content_pre_high_receipt,
            )
            binding.metrics.increment("partitioned_boundary_content_diagnostic_runs")

        if residual_mode == "measure" and frame_gauge_accepted:
            evidence_start = max(0, stage_plan.prefix_t - 6)
            evidence_stop = min(int(learned_clean.shape[2]), stage_plan.prefix_t + 4)
            residual_evidence_tensors["exact_prefix_last6"] = exact_prefix[
                :, :, evidence_start : stage_plan.prefix_t
            ].detach()
            residual_evidence_tensors["learned_native_prefix_suffix"] = learned_clean[
                :, :, evidence_start:evidence_stop
            ].detach()
            residual_evidence_tensors["learned_rigid_aligned_prefix_suffix"] = aligned_witness[
                :, :, evidence_start:evidence_stop
            ].detach()
            residual_evidence_tensors["pre_high_exact_restored_dc"] = restored_clean[
                :, :, evidence_start:evidence_stop
            ].detach()
            residual_stage_receipts.extend(
                [
                    _emit_residual_geometry_stage(
                        binding.metrics,
                        stage="learned_native_same_time",
                        video=learned_clean,
                        prefix_t=stage_plan.prefix_t,
                        session_id=session_id,
                        chunk_id=chunk_id,
                        domain="model_internal_clean",
                        owner_before="learned_provider_clean_L",
                        owner_after="authoritative_exact_prefix_E",
                        temporal_relation="same_time_prefix_calibration",
                        applied_transform="none",
                        provenance="actual_provider_clean_postprocess",
                    ),
                    _emit_residual_geometry_stage(
                        binding.metrics,
                        stage="learned_rigid_aligned_same_time",
                        video=aligned_witness,
                        prefix_t=stage_plan.prefix_t,
                        session_id=session_id,
                        chunk_id=chunk_id,
                        domain="model_internal_clean",
                        owner_before="rigid_aligned_learned_L",
                        owner_after="authoritative_exact_prefix_E",
                        temporal_relation="same_time_prefix_calibration_and_native_boundary",
                        applied_transform="paired_prefix_rigid_v2",
                        provenance="actual_provider_clean_postprocess",
                    ),
                    _emit_residual_geometry_stage(
                        binding.metrics,
                        stage="exact_restored_pre_high_dc",
                        video=restored_clean,
                        prefix_t=stage_plan.prefix_t,
                        session_id=session_id,
                        chunk_id=chunk_id,
                        domain="model_internal_clean",
                        owner_before="authoritative_exact_prefix_E",
                        owner_after="rigid_aligned_learned_suffix_plus_single_dc",
                        temporal_relation="adjacent_time_boundary",
                        applied_transform="paired_prefix_rigid_v2_then_one_token_dc",
                        provenance="actual_pre_high_clean",
                    ),
                ]
            )
            guidance_native = frame_gauge_witnesses.get("guidance_native_bounded")
            guidance_aligned = frame_gauge_witnesses.get("guidance_aligned_bounded")
            guidance_offset = max(0, stage_plan.prefix_t - 6)
            if guidance_native is not None:
                residual_evidence_tensors["guidance_native_prefix_suffix"] = guidance_native.detach()
                residual_stage_receipts.append(
                    _emit_residual_geometry_stage(
                        binding.metrics,
                        stage="guidance_native_same_time",
                        video=guidance_native,
                        prefix_t=stage_plan.prefix_t,
                        temporal_offset=guidance_offset,
                        session_id=session_id,
                        chunk_id=chunk_id,
                        domain="model_internal_clean",
                        owner_before="time_matched_guidance_G",
                        owner_after="authoritative_exact_prefix_E",
                        temporal_relation="same_time_prefix_calibration",
                        applied_transform="none",
                        provenance="independent_guidance_registration_input",
                    )
                )
            if guidance_aligned is not None:
                residual_evidence_tensors["guidance_rigid_aligned_prefix_suffix"] = guidance_aligned.detach()
                residual_stage_receipts.append(
                    _emit_residual_geometry_stage(
                        binding.metrics,
                        stage="guidance_rigid_aligned_same_time",
                        video=guidance_aligned,
                        prefix_t=stage_plan.prefix_t,
                        temporal_offset=guidance_offset,
                        session_id=session_id,
                        chunk_id=chunk_id,
                        domain="model_internal_clean",
                        owner_before="independently_rigid_aligned_guidance_G",
                        owner_after="authoritative_exact_prefix_E",
                        temporal_relation="same_time_prefix_calibration",
                        applied_transform="paired_prefix_rigid_v2_guidance_independent",
                        provenance="independent_guidance_registration_output",
                    )
                )

        for roi_name, roi_fraction in (
            ("upper45", 0.45),
            ("full", 1.0),
        ):
            native_trajectory = measure_translation_trajectory(
                learned_clean,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="learned_native",
                roi=roi_name,
                **native_trajectory,
            )
            if aligned_witness is not None:
                aligned_trajectory = measure_translation_trajectory(
                    aligned_witness,
                    stage_plan.prefix_t,
                    forward_steps=4,
                    backward_steps=3,
                    roi_fraction=roi_fraction,
                    max_shift=4,
                )
                binding.metrics.event(
                    "partitioned_multiframe_trajectory",
                    stage="paired_prefix_aligned_witness",
                    roi=roi_name,
                    **aligned_trajectory,
                )
                binding.metrics.event(
                    "partitioned_multiframe_trajectory",
                    stage="suffix_aligned_before_dc",
                    roi=roi_name,
                    **aligned_trajectory,
                )
            restored_trajectory = measure_translation_trajectory(
                restored_clean,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="exact_restored_pre_high",
                roi=roi_name,
                **restored_trajectory,
            )
        binding.metrics.increment("partitioned_splice_diagnostic_runs")
        binding.metrics.increment("partitioned_multiframe_trajectory_runs")

        target_video[:, :, : stage_plan.prefix_t] = stage_plan.prefix.to(target_video)
        target_raw = pack_streams((target_video, target_audio))[0]
        binding.metrics.event(
            "partitioned_transfer",
            learned_transfer_performed=True,
            upscaler_prefix_context_used=True,
            upscaler_prefix_output_discarded=True,
            authoritative_target_prefix_restored=True,
            target_prefix_resized_for_transformer=False,
            deprecated_mixed_grid_repairs_applied=False,
            frame_gauge_repair_enabled=bool(config.frame_gauge_repair),
            frame_gauge_result=str(frame_gauge_transaction.get("result", "baseline")),
            frame_gauge_reason=str(frame_gauge_transaction.get("reason", "unknown")),
            suffix_dc_bridge_state_mapping=(
                "pre_renoise_clean_operand" if frame_gauge_accepted else "conditional_renoise_affine"
            ),
            suffix_dc_bridge_policy="one_token_spatial_mean_v1",
            partitioned_exact_overlap_bridge={
                "policy": PARTITIONED_EXACT_OVERLAP_POLICY,
                "requested": bool(exact_overlap_fallback_requested),
                "trigger": str(exact_overlap_fallback_trigger),
                "applied": bool(representation_metrics.get("suffix_representation_bridge_accepted", False)),
                "state_mapping": (
                    "conditional_renoise_affine"
                    if representation_metrics.get("suffix_representation_bridge_accepted", False)
                    else "disabled_or_noop"
                ),
                "source": ("actual_provider_boundary_pair" if exact_overlap_fallback_requested else "not_used"),
                "authoritative_prefix_modified": False,
                "later_suffix_extrapolated": False,
                **representation_metrics,
            },
            **dc_metrics,
            provider_api_version=transfer_metrics.get("provider_api_version"),
            provider_kind=transfer_metrics.get("provider_kind"),
            model_name=transfer_metrics.get("model_name"),
            source_hw=transfer_metrics.get("source_hw"),
            target_hw=transfer_metrics.get("target_hw"),
            temporal_length=transfer_metrics.get("temporal_length"),
            learned_upscale_elapsed_ms=transfer_metrics.get("learned_upscale_elapsed_ms"),
            clean_video_postprocess=transfer_metrics.get("clean_video_postprocess"),
            **splice_diagnostics,
        )

        target_latent_internal = _process_latent_in(
            base_model,
            latent_image,
            target_shapes,
        )
        target_noise = _noise_argument(
            base_model,
            target_raw,
            sigma,
            target_latent_internal,
        )
        target_noise = _merge_preserved_noise(
            target_noise,
            noise,
            denoise_mask,
        )
        merged_video_noise, _merged_audio_noise = unpack_streams(
            target_noise,
            target_shapes,
        )
        original_video_noise, _original_audio_noise = unpack_streams(
            noise,
            target_shapes,
        )
        protected_video_noise_exact = torch.equal(
            merged_video_noise[:, :, : stage_plan.prefix_t],
            original_video_noise[:, :, : stage_plan.prefix_t].to(merged_video_noise),
        )
        if not protected_video_noise_exact:
            raise RuntimeError("partitioned handoff changed caller-owned protected video noise")
        binding.metrics.event(
            "handoff_transfer_wall",
            elapsed_ms=(time.perf_counter() - transfer_started) * 1000.0,
            protected_video_noise_exact=protected_video_noise_exact,
            partitioned_exact_prefix=True,
        )

        if guidance_run is not None:
            if frame_gauge_accepted:
                if pending_registered_reference is None:
                    raise RuntimeError("accepted frame-gauge transaction lost registered Flow guidance")
                binding.registered_guidance_reference = pending_registered_reference
            else:
                binding.registered_guidance_reference = None
            binding.active_guidance_run = guidance_run
            binding.guidance_state.reset()
            binding.metrics.event(
                "partitioned_guidance_trajectory_selection",
                source=guidance_trajectory_source,
                run_id=guidance_run.run_id,
                exact_samples=len(guidance_run.exact_samples()),
                source_hw=(
                    guidance_run.geometry.latent_h,
                    guidance_run.geometry.latent_w,
                ),
                target_hw=(target_h, target_w),
                main_captured_run_id=(committed_low_run.run_id if committed_low_run is not None else None),
                shared_trajectory_handle_preserved=True,
                audio_latent_trajectory_present=False,
                frame_gauge_registered=bool(binding.registered_guidance_reference is not None),
                guidance_state_reset_before_high=True,
                diagnostic_only=(guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW),
            )
        else:
            if pending_registered_reference is not None:
                raise RuntimeError("frame-gauge transaction published guidance while guidance is off")
            binding.registered_guidance_reference = None
            binding.guidance_state.reset()

        # Release full clean-domain registration witnesses before target-high.
        del restored_clean
        del corrected_clean
        del learned_clean
        if aligned_witness is not None:
            del aligned_witness
        frame_gauge_witnesses.clear()

        def high_callback(step, x0, x, _total):
            if callback is not None:
                return callback(index + step, x0, x, len(sigmas) - 1)
            return None

        _cuda_allocator_checkpoint(binding.metrics, "pre_high")
        _reset_guider_conds(guider, template=conditioning_template)
        high_started = time.perf_counter()
        high_event_start = len(binding.metrics.events)
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider):
            result = executor(
                target_noise,
                latent_image,
                sampler,
                high_sigmas,
                denoise_mask,
                high_callback,
                disable_pbar,
                seed,
                latent_shapes=target_shapes,
            )
        binding.metrics.event(
            "high_stage_wall",
            elapsed_ms=(time.perf_counter() - high_started) * 1000.0,
            partitioned_exact_prefix=True,
        )
        _cuda_allocator_checkpoint(binding.metrics, "post_high")
        high_model_calls = [event for event in binding.metrics.events[high_event_start:] if event.kind == "model_call"]
        if not high_model_calls:
            raise RuntimeError("partitioned exact-prefix high stage produced no H3 model evaluations")
        first_high_actual = bool(high_model_calls[0].fields.get("actual"))
        if not first_high_actual:
            raise RuntimeError("partitioned exact-prefix high stage did not begin with an exact H3 evaluation")

        final_video, final_audio = unpack_streams(result, target_shapes)
        final_internal = None
        final_internal_video = None
        final_internal_audio = None
        if (
            diagnostic_audio_control
            or (residual_mode == "measure" and frame_gauge_accepted)
            or boundary_content_diagnostic_enabled
        ):
            final_internal = _process_latent_in(base_model, result, target_shapes)
            final_internal_video, final_internal_audio = unpack_streams(final_internal, target_shapes)
        if residual_mode == "measure" and frame_gauge_accepted:
            if final_internal_video is None:
                raise RuntimeError("residual measurement lost the common-domain final video operand")
            final_internal_prefix = final_internal_video[:, :, : stage_plan.prefix_t]
            if (
                final_internal_prefix.shape != exact_prefix.shape
                or final_internal_prefix.dtype != exact_prefix.dtype
                or not torch.equal(
                    final_internal_prefix,
                    exact_prefix.to(device=final_internal_prefix.device),
                )
            ):
                raise RuntimeError(
                    "post-high latent-input conversion does not reproduce the authoritative internal exact prefix"
                )
            final_bounded, _final_start, _final_stop, _final_local_prefix = _bounded_residual_stage_slice(
                final_internal_video,
                prefix_t=stage_plan.prefix_t,
            )
            residual_evidence_tensors["final_post_high_internal_clean"] = final_bounded.detach()
            residual_stage_receipts.append(
                _emit_residual_geometry_stage(
                    binding.metrics,
                    stage="final_post_high_internal_clean",
                    video=final_internal_video,
                    prefix_t=stage_plan.prefix_t,
                    session_id=session_id,
                    chunk_id=chunk_id,
                    domain="model_internal_clean",
                    owner_before="authoritative_exact_prefix_E",
                    owner_after="post_high_generated_suffix",
                    temporal_relation="adjacent_time_boundary_and_next_three_suffix_pairs",
                    applied_transform="none_post_high_observation",
                    provenance="existing_post_high_output_via_model_latent_input_conversion",
                )
            )
            residual_stage_receipts.append(
                _emit_residual_geometry_stage(
                    binding.metrics,
                    stage="final_post_high_caller_domain",
                    video=final_video,
                    prefix_t=stage_plan.prefix_t,
                    session_id=session_id,
                    chunk_id=chunk_id,
                    domain="caller_output_latent",
                    owner_before="caller_owned_exact_prefix",
                    owner_after="returned_generated_suffix",
                    temporal_relation="adjacent_time_boundary_receipt_only",
                    applied_transform="none",
                    provenance="existing_post_high_output",
                )
            )
        if boundary_content_diagnostic_enabled:
            if final_internal_video is None or boundary_content_pre_high_receipt is None:
                raise RuntimeError("boundary-content diagnostic lost its common-domain operands")
            post_high_diagnostic_video = final_internal_video
            final_internal_prefix = post_high_diagnostic_video[:, :, : stage_plan.prefix_t]
            prefix_recanonicalized = not (
                final_internal_prefix.shape == exact_prefix.shape
                and final_internal_prefix.dtype == exact_prefix.dtype
                and torch.equal(
                    final_internal_prefix,
                    exact_prefix.to(device=final_internal_prefix.device),
                )
            )
            if prefix_recanonicalized:
                post_high_diagnostic_video = post_high_diagnostic_video.clone()
                post_high_diagnostic_video[:, :, : stage_plan.prefix_t] = exact_prefix.to(post_high_diagnostic_video)
            boundary_content_started = time.perf_counter()
            post_high_receipt = measure_boundary_content_continuity(
                post_high_diagnostic_video,
                stage_plan.prefix_t,
            )
            binding.metrics.event(
                "partitioned_boundary_content_continuity",
                stage="post_high_internal_clean",
                domain="model_internal_clean",
                owner_before="authoritative_exact_prefix",
                owner_after="post_high_generated_suffix",
                prefix_recanonicalized_for_diagnostic=prefix_recanonicalized,
                elapsed_ms=(time.perf_counter() - boundary_content_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **post_high_receipt,
            )
            binding.metrics.event(
                "partitioned_boundary_content_stage_delta",
                pre_stage="pre_high_exact_restored",
                post_stage="post_high_internal_clean",
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **compare_boundary_content_stages(
                    boundary_content_pre_high_receipt,
                    post_high_receipt,
                ),
            )

        if diagnostic_audio_control:
            if final_internal is None or final_internal_audio is None:
                raise RuntimeError("audio diagnostics lost the converted final sampler state")
            final_audio_report = measure_audio_latent_boundary(
                final_internal,
                target_shapes,
                diagnostic_target_mask,
                windows=(4, 20),
            )
            binding.metrics.event(
                "partitioned_audio_stage_boundary",
                stage="final_high_clean",
                domain="model_internal_clean",
                **final_audio_report,
            )
            if low_probe_clean_audio is None:
                raise RuntimeError("audio stage diagnostics lost the low/probe clean reference")
            _mask_video, exact_audio_mask = unpack_streams(diagnostic_target_mask, target_shapes)
            stage_delta = compare_audio_latent_stages(
                low_probe_clean_audio,
                final_internal_audio,
                exact_audio_mask,
                windows=(4, 20),
            )
            binding.metrics.event(
                "partitioned_audio_stage_delta",
                reference_stage="low_probe_clean",
                candidate_stage="final_high_clean",
                domain="model_internal_clean",
                **stage_delta,
            )
            if audio_handoff_source == PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW:
                binding.metrics.event(
                    "partitioned_audio_stage_delta_interpretation",
                    reference_stage="low_probe_clean",
                    candidate_stage="final_high_clean",
                    causal_pair=False,
                    reason="high stage initialized from source-uniform shadow raw audio state",
                )
            elif av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
                binding.metrics.event(
                    "partitioned_audio_stage_delta_interpretation",
                    reference_stage="low_probe_clean",
                    candidate_stage="final_high_clean",
                    causal_pair=False,
                    reason=(
                        "high stage initialized from source-uniform shadow raw audio and "
                        "shadow-probe clean video handoff state"
                        + (
                            " and guided by the isolated source-uniform shadow video trajectory"
                            if guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW
                            else ""
                        )
                    ),
                )
        original_video, original_audio = unpack_streams(latent_image, target_shapes)
        final_prefix = final_video[:, :, : stage_plan.prefix_t]
        original_prefix = original_video[:, :, : stage_plan.prefix_t]
        if (
            final_prefix.shape != original_prefix.shape
            or final_prefix.dtype != original_prefix.dtype
            or not torch.equal(
                final_prefix,
                original_prefix.to(device=final_prefix.device),
            )
        ):
            raise RuntimeError("partitioned exact-prefix high stage violated byte-exact original-prefix preservation")
        if audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
            if tensor_sha256(denoise_mask) != candidate_high_mask_digest:
                raise RuntimeError("source-carrier audio-position candidate mutated the high-stage sampler mask")
            _mask_video, audio_mask = unpack_streams(denoise_mask, target_shapes)
            protected_audio = audio_mask == 0
            if not bool(protected_audio.any().item()):
                raise RuntimeError("source-carrier audio-position candidate found no protected carried-audio prefix")
            audio_exact = final_audio.dtype == original_audio.dtype and torch.equal(
                final_audio[protected_audio],
                original_audio.to(device=final_audio.device)[protected_audio],
            )
            if not audio_exact:
                raise RuntimeError("source-carrier audio-position candidate violated exact carried-audio restoration")
            binding.metrics.event(
                "partitioned_audio_position_candidate_integrity",
                mode=audio_position_domain,
                low_mask_digest=candidate_low_mask_digest,
                high_mask_digest=candidate_high_mask_digest,
                final_exact_video_prefix=True,
                final_exact_audio_prefix=True,
                sampler_masks_unchanged=True,
            )
        boundary = measure_video_boundary(final_video, stage_plan.prefix_t)
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            final_trajectory = measure_translation_trajectory(
                final_video,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="final_post_high",
                roi=roi_name,
                **final_trajectory,
            )
        if not splice_diagnostics:
            raise RuntimeError(
                "partitioned exact-prefix splice diagnostics were not recorded before high-stage sampling"
            )

        residual_evidence_receipt: dict[str, Any] | None = None
        if residual_mode == "measure" and frame_gauge_accepted:
            video_registration = frame_gauge_transaction.get("video_registration", {})
            guidance_registration = frame_gauge_transaction.get("guidance_registration", {})
            residual_evidence_receipt = export_residual_geometry_evidence(
                residual_evidence_tensors,
                session_id=str(session_id),
                chunk_id=str(chunk_id),
                seed=int(seed or 0),
                sigma=float(sigma),
                metadata={
                    "policy": RESIDUAL_GEOMETRY_POLICY_VERSION,
                    "rigid_policy": FRAME_GAUGE_POLICY_VERSION,
                    "prefix_t": int(stage_plan.prefix_t),
                    "fit_indices": residual_geometry_receipt.get("video", {}).get("fit_indices", []),
                    "holdout_indices": residual_geometry_receipt.get("video", {}).get("holdout_indices", []),
                    "video_rigid_dx": float(video_registration.get("dx", 0.0)),
                    "video_rigid_dy": float(video_registration.get("dy", 0.0)),
                    "guidance_rigid_dx": float(guidance_registration.get("dx", 0.0)),
                    "guidance_rigid_dy": float(guidance_registration.get("dy", 0.0)),
                    "target_hw": [int(target_h), int(target_w)],
                    "dtype": str(exact_prefix.dtype),
                    "video_validity_policy": "translation_validity_v1",
                    "video_invalid_fraction": float(video_registration.get("invalid_fraction", 0.0)),
                    "guidance_invalid_fraction": float(guidance_registration.get("invalid_fraction", 0.0)),
                    "exact_prefix_sha256": tensor_sha256(exact_prefix),
                    "protected_noise_exact": bool(protected_video_noise_exact),
                    "dc_policy": "existing_one_token_spatial_mean_v1",
                    "dc_corrected_tokens": int(dc_metrics.get("suffix_dc_bridge_corrected_tokens", 0)),
                    "stage_receipts": residual_stage_receipts,
                    "extra_h3_nfe": 0,
                    "extra_sampler_lifetimes": 0,
                    "extra_history_boundaries": 0,
                    "extra_provider_calls": 0,
                    "extra_vae_calls": 0,
                    "horizontal_application_enabled": False,
                },
            )
            binding.metrics.event(
                "partitioned_residual_geometry_evidence",
                policy=RESIDUAL_GEOMETRY_POLICY_VERSION,
                requested_mode=residual_mode,
                decision="not_evaluated",
                applied=False,
                horizontal_application_enabled=False,
                session_id=str(session_id),
                chunk_id=str(chunk_id),
                **residual_evidence_receipt,
            )

        if final_internal is not None:
            del final_internal
        binding.metrics.event(
            "partitioned_exact_prefix_complete",
            final_prefix_exact=True,
            high_stage_first_call_actual=first_high_actual,
            total_chunk_sampler_elapsed_ms=(time.perf_counter() - chunk_started) * 1000.0,
            final_seam_lowpass_kernel=boundary["lowpass_kernel"],
            final_seam_rms=boundary["seam_rms"],
            final_seam_lowpass_rms=boundary["seam_lowpass_rms"],
            final_seam_spatial_mean_rms=boundary["seam_spatial_mean_rms"],
            final_over_transfer_native_seam_rms_ratio=boundary["seam_rms"]
            / max(float(splice_diagnostics["upscaler_native_seam_rms"]), 1e-12),
            final_over_transfer_native_seam_lowpass_ratio=boundary["seam_lowpass_rms"]
            / max(float(splice_diagnostics["upscaler_native_seam_lowpass_rms"]), 1e-12),
            final_over_transfer_native_seam_spatial_mean_ratio=boundary["seam_spatial_mean_rms"]
            / max(float(splice_diagnostics["upscaler_native_seam_spatial_mean_rms"]), 1e-12),
            final_over_transfer_exact_seam_rms_ratio=boundary["seam_rms"]
            / max(float(splice_diagnostics["exact_restored_seam_rms"]), 1e-12),
            final_over_transfer_exact_seam_lowpass_ratio=boundary["seam_lowpass_rms"]
            / max(float(splice_diagnostics["exact_restored_seam_lowpass_rms"]), 1e-12),
            final_over_transfer_exact_seam_spatial_mean_ratio=boundary["seam_spatial_mean_rms"]
            / max(float(splice_diagnostics["exact_restored_seam_spatial_mean_rms"]), 1e-12),
            deprecated_mixed_grid_contract_active=False,
        )
        binding.metrics.event(
            "handoff_complete",
            sigma=sigma,
            target_shape=target_shapes[0],
            audio_state_copied=True,
            separate_sampler_invocations=True,
            sampler_invocation_count=sampler_invocation_count,
            history_boundary_count=history_boundary_count,
            exact_probe_performed=True,
            high_stage_exact_prefix_requested=1,
            high_stage_first_call_actual=first_high_actual,
            high_stage_model_calls=len(high_model_calls),
            conditioning_rebuilt_for_high_grid=True,
            transfer_mode="learned_3d_suffix_context",
            input_mode="partitioned_exact_prefix",
        )
        if audio_handoff_source == PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW:
            binding.metrics.event(
                "partitioned_audio_handoff_shadow_complete",
                source=audio_handoff_source,
                final_exact_video_prefix=True,
                final_exact_audio_prefix=True,
                pre_high_main_video_state_preserved=True,
                final_video_independent_of_shadow_audio=False,
                shadow_audio_handoff_applied=True,
                high_stage_first_call_actual=first_high_actual,
                sampler_invocation_count=sampler_invocation_count,
                history_boundary_count=history_boundary_count,
                diagnostic_only=True,
            )
        if av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
            binding.metrics.event(
                "partitioned_av_handoff_shadow_complete",
                source=av_handoff_source,
                final_exact_video_prefix=True,
                final_exact_audio_prefix=True,
                guidance_trajectory_source=guidance_trajectory_source,
                main_guidance_trajectory_preserved=(
                    guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
                ),
                shadow_guidance_trajectory_applied=(
                    guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW
                ),
                shared_trajectory_handle_preserved=True,
                shadow_raw_audio_handoff_applied=True,
                shadow_clean_video_handoff_applied=True,
                final_video_independent_of_shadow_av=False,
                high_stage_first_call_actual=first_high_actual,
                sampler_invocation_count=sampler_invocation_count,
                history_boundary_count=history_boundary_count,
                diagnostic_only=True,
            )
        return result
    except BaseException as exc:
        if committed_low_run is not None and binding.trajectory is not None:
            invalid = binding.trajectory.invalidate(
                committed_low_run.run_id,
                f"partitioned exact-prefix continuation failed: {type(exc).__name__}: {exc}",
            )
            binding.metrics.event(
                "trajectory_invalidate",
                run_id=invalid.run_id,
                error=type(exc).__name__,
                partitioned_exact_prefix=True,
            )
        raise


__all__ = [
    "PARTITIONED_PROGRESSIVE_KEY",
    "PARTITIONED_SOL_REQUIRED_METADATA",
    "SOL_RUNTIME_KEY",
    "PartitionedPreflightUnsupported",
    "_isolated_shadow_trajectory_capture",
    "_resolve_audio_diagnostic_masks",
    "_select_source_uniform_shadow_clean_video",
    "_source_uniform_audio_shadow_controls",
    "_source_uniform_av_shadow_stage_contract",
    "_splice_source_uniform_shadow_audio_state",
    "_validate_audio_handoff_shadow_configuration",
    "_validate_audio_position_candidate_configuration",
    "_validate_av_handoff_shadow_configuration",
    "_validate_guidance_trajectory_shadow_configuration",
    "_verify_audio_position_domain_diagnostic",
    "run_partitioned_progressive",
]
