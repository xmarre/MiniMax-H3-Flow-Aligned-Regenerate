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
    measure_learned_transfer_residual_diagnostic,
    measure_provider_boundary_post_high_shadow,
    measure_provider_boundary_soft_support_shadow,
    measure_provider_boundary_stabilization_shadow,
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
    h3_patch_lattice_weight_square_sum,
    pack_streams,
    resize_spatial_5d_h3_patch_lattice,
    resize_video,
    unpack_streams,
)
from .guidance import HandoffGuidanceReference, RegisteredGuidanceReference, time_matched_reference_info
from .handoff import (
    H3_HANDOFF_NOISE_DENSE_DRIFT,
    H3_HANDOFF_NOISE_INDEPENDENT,
    H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
    H3_LATENT_UPSCALER_API_VERSION,
    H3_LATENT_UPSCALER_KIND,
    CleanVideoPostprocessResult,
    ProgressiveTargetInputConfig,
    build_handoff_state,
    deterministic_video_noise,
)
from .high_stage_boundary import (
    HIGH_BOUNDARY_REFERENCE_POLICY,
    HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
    HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS,
    high_boundary_contract,
)
from .partitioned_band import (
    TARGET_BAND_HANDOFF_POLICY,
    TARGET_BAND_RAW_CARRY_HANDOFF_POLICY,
    pack_target_band_video,
    target_band_padding_max_abs,
    target_band_source_view,
    target_band_target_preview,
)
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT,
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
    PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL,
    PARTITIONED_HANDOFF_TRANSFER_CONTROL_KEY,
    PARTITIONED_HANDOFF_TRANSFER_LEARNED,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_KEY,
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF,
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_API,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK,
    PARTITIONED_SPATIAL_STAGE_CONTROL_KEY,
    PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
    PARTITIONED_SPATIAL_STAGE_SAME_GRID,
    PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
    PARTITIONED_TARGET_BAND_CONTEXT_DOMAIN_UNIFORM,
    PARTITIONED_TARGET_BAND_CONTEXT_KEY,
    PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
    PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY,
    PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY,
    PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    PARTITIONED_VDN_TEMPORAL_CARRIER_API,
    PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
    PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
    PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API,
    normalize_audio_handoff_source,
    normalize_audio_position_domain,
    normalize_av_handoff_source,
    normalize_guidance_trajectory_source,
    normalize_handoff_transfer_control,
    normalize_low_probe_execution_source,
    normalize_partitioned_softmax_diagnostic,
    normalize_prefix_transformer_context,
    normalize_provider_boundary_stabilization,
    normalize_spatial_stage_control,
    normalize_target_band_handoff_state,
    normalize_vdn_linear_diagnostic,
    normalize_vdn_temporal_carrier_policy,
    resolve_partitioned_audio_guided_overlap_mode,
    resolve_partitioned_audio_guided_overlap_ticks,
    resolve_partitioned_suffix_dc_bridge,
    resolve_partitioned_target_band_context,
    resolve_partitioned_target_band_handoff_state,
    resolve_partitioned_target_band_tokens,
    resolve_partitioned_video_guided_overlap_tokens,
)
from .partitioned_prefix import PARTITIONED_NATIVE_CARRIER_SOURCE, PARTITIONED_NATIVE_CARRIER_TARGET
from .partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    TARGET_BAND_DOMAIN_BAND_CARRIER_POLICY,
    TARGET_BAND_DOMAIN_STREAM_API,
    TARGET_BAND_DOMAIN_UNIFORM_POLICY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
    PartitionedTargetBandGeometry,
    TargetBandDomainContext,
    build_partitioned_stage_plan,
    tensor_sha256,
)
from .partitioned_transformer import VDN_PARTITIONED_SEQUENCE_API
from .representation_bridge import disabled_suffix_representation_bridge_metrics
from .residual_evidence import BoundaryWindowEvidence, export_residual_geometry_evidence
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
    _process_latent_out,
    _raw_sampler_state,
    _reset_guider_conds,
    _resize_packed_latent_image,
    _resize_packed_mask,
    _validate_progressive_sampler_state,
    sampler_name,
)
from .seam_diagnostics import (
    measure_exact_prefix_splice,
    measure_target_band_overlap,
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
from .transfer_lattice import H3_TRANSFER_LATTICE, H3PatchLatticeTransferProvider, measure_paired_prefix_affine
from .vae_boundary_video import (
    VAE_WINDOW_VIDEO_POLICY,
    apply_vae_window_vertical_translation,
    h3_vae_boundary_window,
    measure_vae_window_video_trajectory,
    repair_vae_window_vertical_residual,
    validate_vae_window_vertical_candidate,
)

PARTITIONED_PROGRESSIVE_KEY = "h3_flow_partitioned_progressive_v1"
SOL_RUNTIME_KEY = "sol_h3_runtime_v1"
FRAME_GAUGE_GUIDANCE_SUPPORTED_SAMPLERS = frozenset({"sample_res_multistep"})
# Dense drift transport assumes residual - noise_scale * initial_noise is model
# drift. Stochastic samplers add fresh white noise to that difference; dense
# interpolation would remove about half of its variance and correlate it.
H3_DENSE_DRIFT_DETERMINISTIC_SAMPLERS = frozenset(
    {
        "sample_deis",
        "sample_dpmpp_2m",
        "sample_dpmpp_2m_cfg_pp",
        "sample_euler_cfg_pp",
        "sample_exp_heun_2_x0",
        "sample_gradient_estimation",
        "sample_gradient_estimation_cfg_pp",
        "sample_ipndm",
        "sample_ipndm_v",
        "sample_lms",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
        "sample_unipc",
        "sample_unipc_bh2",
    }
)
# Deterministic only while s_churn is zero.
H3_DENSE_DRIFT_CHURN_SAMPLERS = frozenset({"sample_dpm_2", "sample_euler", "sample_heun", "sample_heunpp2"})
FRAME_GAUGE_BOUNDARY_MIN_ERROR_CELLS = 0.125
FRAME_GAUGE_BOUNDARY_MIN_IMPROVEMENT = 0.25
FRAME_GAUGE_BOUNDARY_MIN_RESPONSE = 3.0
# The rigid estimator itself resolves translations on a 1/16-cell fine grid.
# Boundary-motion phase estimates can therefore move by less than one estimator
# quantum even when the held-out prefix registration is coherent. Treat at most
# one fine-search quantum as measurement-floor degradation, never as evidence
# strong enough to veto an otherwise strongly supported transaction.
FRAME_GAUGE_BOUNDARY_MAX_DEGRADATION_CELLS = 0.0625
PARTITIONED_EXACT_OVERLAP_POLICY = "partitioned_exact_overlap_dc_only_v5"
# Keep the existing name for integrations that import the support contract.
# Only the first generated token owns a channel-mean correction.
PARTITIONED_EXACT_OVERLAP_PRODUCTION_WEIGHTS = (1.0,)
PARTITIONED_VIDEO_BOUNDARY_REPAIR_CONTRACT = "dense_drift_handoff_plus_one_token_dc_v6"
PARTITIONED_AUDIO_BOUNDARY_REPAIR_CONTRACT = "released_sampler_overlap_exact_restore_v1"
PARTITIONED_HIGH_VIDEO_REFERENCE_ENABLED = False
PARTITIONED_HIGH_AUDIO_REFERENCE_ENABLED = False
PARTITIONED_HIGH_ATTENTION_POLICY = "exact_prefix_query_continuity_v1"
VDN_PARTITIONED_BOUNDARY_QUERY_API = 1
VDN_PARTITIONED_BOUNDARY_QUERY_POLICY = "boundary_suffix_local_group_dense_v1"
FRAME_GAUGE_HARDWARE_INVALIDATED_RIGID_REASON = "hardware_invalidated_global_rigid_application_00687"


class _BicubicSameSourceTransferProvider:
    """Provider-contract shim that replaces only the learned clean-video operator."""

    api_version = H3_LATENT_UPSCALER_API_VERSION
    kind = H3_LATENT_UPSCALER_KIND
    model_name = "diagnostic:bicubic_same_source_control"
    offload_after_upscale = False

    def __init__(self, template) -> None:
        self.device = str(getattr(template, "device", "cuda"))
        self.inference_device = self.device
        self.precision = str(getattr(template, "precision", "bf16"))
        self.calls = 0

    def upscale_clean_video(
        self,
        video: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        self.calls += 1
        return resize_video(video, int(target_h), int(target_w), mode="bicubic")


class _IdentitySameGridTransferProvider:
    """Provider-contract shim for a no-resize low/probe -> high handoff control."""

    api_version = H3_LATENT_UPSCALER_API_VERSION
    kind = H3_LATENT_UPSCALER_KIND
    model_name = "diagnostic:same_grid_target_identity"
    offload_after_upscale = False

    def __init__(self, template) -> None:
        self.device = str(getattr(template, "device", "cuda"))
        self.inference_device = self.device
        self.precision = str(getattr(template, "precision", "bf16"))
        self.calls = 0

    def upscale_clean_video(
        self,
        video: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        self.calls += 1
        if tuple(map(int, video.shape[-2:])) != (int(target_h), int(target_w)):
            raise RuntimeError("same-grid identity transfer received a spatial resize request")
        return video.clone()


def _cache_audio_decode_witness(
    binding,
    model_options,
    base_model,
    internal_state: torch.Tensor,
    shapes: list[tuple[int, ...]],
    *,
    stage: str,
) -> torch.Tensor:
    """Cache one output-domain audio stage witness without affecting sampling."""

    caller_state = _process_latent_out(base_model, internal_state, shapes)
    _caller_video, caller_audio = unpack_streams(caller_state, shapes)
    del _caller_video, caller_state
    witness = caller_audio.detach().to(device="cpu").clone()
    session_id, chunk_id = _interop_identity(model_options)
    registry = binding.audio_stage_witnesses
    resident_sessions = {key[0] for key in registry}
    if resident_sessions and session_id not in resident_sessions:
        registry.clear()
    witness_key = (str(session_id), str(chunk_id))
    registry[witness_key] = {
        "stage": str(stage),
        "domain": "caller_vae_latent",
        "audio": witness,
        "source_shapes": tuple(tuple(int(v) for v in shape) for shape in shapes),
    }
    while len(registry) > 16:
        registry.pop(next(iter(registry)))
    binding.metrics.event(
        "partitioned_audio_decode_witness_cached",
        session_id=str(session_id),
        chunk_id=str(chunk_id),
        stage=str(stage),
        domain="caller_vae_latent",
        audio_shape=tuple(int(v) for v in witness.shape),
        storage_device=str(witness.device),
        process_latent_out_applied=True,
        output_neutral=True,
        extra_h3_nfe=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
        extra_vae_calls=0,
    )
    return witness


FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS = frozenset(
    {
        "boundary_upper45_not_improved",
        "boundary_full_not_improved",
        "boundary_upper45_degraded_over_bound",
        "boundary_full_degraded_over_bound",
        "boundary_upper45_insufficient_improvement",
        "boundary_full_insufficient_improvement",
        "boundary_no_informative_roi_strong_improvement",
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
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    ):
        mismatches.append("audio_guided_overlap_mode in {'exact_mask', 'sampler_mask', 'sampler_mask_exact_timestep'}")
    # Overlap width is a non-negative runtime control. Its applied support uses
    # the available carried prefix and is not a structural source-uniform requirement.
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


def _partitioned_high_video_overlap_mask(
    runtime_mask: torch.Tensor,
    exact_mask: torch.Tensor | None,
    latent_shapes: list[tuple[int, ...]],
    model_options: dict[str, Any],
    metrics,
    *,
    prefix_t: int,
) -> torch.Tensor:
    """Keep target-high video context exact even when an old overlap width is requested.

    Run 01093 falsified the prefix-release interpretation of video overlap.  The
    old path repainted the tail of the caller-owned carried prefix during early
    high evaluations and restored those bytes only after sampling.  H3 therefore
    generated the suffix against context that was later discarded.  Wider
    overlap increases the amount of discarded context and can worsen the visible
    boundary.

    Preserve the public non-negative width for workflow compatibility and report
    the request, but do not mutate the exact-prefix video mask.  Audio mask
    ownership remains independent.
    """

    tokens, source = resolve_partitioned_video_guided_overlap_tokens(model_options)
    if tokens <= 0:
        return runtime_mask
    if exact_mask is None:
        raise RuntimeError("video guided overlap requires an authoritative exact output mask")
    if tuple(exact_mask.shape) != tuple(runtime_mask.shape):
        raise RuntimeError("video guided overlap exact/runtime mask geometry mismatch")
    if model_options.get("denoise_mask_function") is not None:
        raise RuntimeError(
            "retired video overlap cannot certify exact target-high video context while denoise_mask_function is owned"
        )
    if type(prefix_t) is not int or prefix_t <= 0 or prefix_t >= int(latent_shapes[0][2]):
        raise RuntimeError("retired video overlap received invalid exact-prefix ownership")

    runtime_video, _runtime_audio = unpack_streams(runtime_mask, latent_shapes)
    exact_video, _exact_audio = unpack_streams(exact_mask, latent_shapes)
    if not torch.equal(runtime_video, exact_video.to(device=runtime_video.device, dtype=runtime_video.dtype)):
        raise RuntimeError("target-high video mask already differs from authoritative exact-prefix context")

    metrics.event(
        "partitioned_video_guided_overlap",
        requested_tokens=tokens,
        applied_tokens=0,
        width_limited_by_prefix=tokens > prefix_t,
        applied=False,
        reason="retired_discarded_prefix_context_01093",
        video_prefix_tokens=prefix_t,
        video_total_tokens=int(latent_shapes[0][2]),
        hard_prefix_tokens=prefix_t,
        ramp_values=[],
        configuration_source=source,
        sampler_mask_modified=False,
        model_mask_modified=False,
        stage="high",
        low_probe_sampler_mask_unchanged=True,
        structural_preflight_mask_exact=True,
        high_sampler_video_mask_exact=True,
        high_model_video_context_exact=True,
        final_exact_prefix_restore=True,
        policy="partitioned_video_high_exact_context_v4",
        retired_policy="partitioned_video_high_sampler_overlap_exact_tail_v3",
        retired_closure_policy="video_prefix_release_then_exact_tail_v1",
        retired_prefix_release=True,
        hardware_verdict="falsified_01093_rendered_shift_shock_tone_and_overlap_worsening",
        partitioned_exact_prefix=True,
        extra_h3_nfe=0,
        extra_provider_calls=0,
        extra_vae_calls=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
    )
    return runtime_mask


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
def _partitioned_stage_contract(
    guider: Any,
    plan,
    metrics,
    *,
    target_band=None,
    attention_head_t=None,
    target_band_domain=None,
):
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
    softmax_mode = normalize_partitioned_softmax_diagnostic(
        transformer.get(PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY, PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL)
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
    temporal_carrier_policy = normalize_vdn_temporal_carrier_policy(
        transformer.get(
            PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
            PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
        )
    )
    temporal_carrier_spec = None
    if temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION:
        temporal_carrier_spec = _validate_partitioned_vdn_compat(
            guider.model_patcher,
            required_linear_diagnostic=linear_mode,
            required_temporal_carrier_policy=temporal_carrier_policy,
        )
    from .boundary_witness import WITNESS_DIRECTORY_OPTION, configured_boundary_witness

    witness_directory = options.get(WITNESS_DIRECTORY_OPTION, None)
    transformer[PARTITIONED_STAGE_KEY] = PartitionedStageRuntime(
        plan=plan,
        metrics=metrics,
        vdn_linear_diagnostic=linear_mode,
        softmax_diagnostic=softmax_mode,
        vdn_temporal_carrier_policy=temporal_carrier_policy,
        vdn_temporal_carrier_short_conv_spec=temporal_carrier_spec,
        boundary_witness=(
            None
            if transformer.get(FLOW_STAGE_KEY) == "high" or target_band is not None
            else configured_boundary_witness(metrics, directory=witness_directory)
        ),
        prefix_transformer_context=prefix_context,
        audio_position_domain=audio_position_domain,
        target_band=target_band,
        attention_head_t=attention_head_t,
        target_band_domain=target_band_domain,
    )
    if target_band_domain is not None and (target_band is None or attention_head_t is not None):
        transformer.pop(PARTITIONED_STAGE_KEY, None)
        raise RuntimeError("domain-uniform target-band context applies only to target-band low/probe stages")
    try:
        yield
        owner = transformer[PARTITIONED_STAGE_KEY]
        if (
            transformer.get(FLOW_STAGE_KEY) == "low"
            and owner.boundary_witness is not None
            and not owner.boundary_witness.completed
        ):
            raise RuntimeError("requested boundary witness was not completed by the actual low-stage VDN call")
        if owner.vdn_temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION:
            contract = owner.vdn_temporal_carrier_contract
            if not isinstance(contract, dict) or not isinstance(contract.get("numerical_digest"), str):
                raise RuntimeError("destination-grid temporal stencil produced no bound numerical-policy receipt")
            metrics.event(
                "partitioned_vdn_temporal_carrier_stage",
                stage=transformer.get(FLOW_STAGE_KEY),
                policy=owner.vdn_temporal_carrier_policy,
                numerical_digest=contract["numerical_digest"],
                short_conv_spec=owner.vdn_temporal_carrier_short_conv_spec,
                output_space_mutation=False,
            )
    finally:
        transformer.pop(PARTITIONED_STAGE_KEY, None)


@contextlib.contextmanager
def _partitioned_high_stage_contract(guider, plan, metrics, *, exact_prefix_attention=True, attention_head_t=None):
    """Keep exact-prefix query policy across the target-grid refinement boundary."""
    if attention_head_t is not None and (
        type(attention_head_t) is not int
        or not plan.prefix_t <= attention_head_t < plan.temporal
        or not exact_prefix_attention
    ):
        raise ValueError(
            "target-band dense-query head requires exact-prefix high attention and a valid temporal extent"
        )
    transformer = guider.model_options["transformer_options"]
    with _high_stage_contract(guider, source="h3_flow_partitioned_refinement"):
        metrics.event(
            "partitioned_high_attention_plan",
            policy=PARTITIONED_HIGH_ATTENTION_POLICY,
            exact_prefix_attention=bool(exact_prefix_attention),
            prefix_t=int(plan.prefix_t),
            attention_head_t=int(plan.prefix_t if attention_head_t is None else attention_head_t),
            temporal=int(plan.temporal),
            target_hw=plan.target_hw,
            protected_prefix_local_queries="dense" if exact_prefix_attention else "native",
            generated_local_queries=(
                "boundary_and_band_dense_then_native_sol_selection"
                if attention_head_t is not None
                else "boundary_dense_then_native_sol_selection"
            ),
            boundary_query_policy=VDN_PARTITIONED_BOUNDARY_QUERY_POLICY,
            startup_density_exemption=False,
            refinement_source="h3_flow_partitioned_refinement",
            high_linear_diagnostic="normal",
            high_softmax_diagnostic="normal",
            high_audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
            extra_logical_model_calls=0,
            extra_sampler_invocations=0,
        )
        if not exact_prefix_attention:
            yield
            return

        high_plan = PartitionedStagePlan(
            prefix=plan.prefix,
            prefix_noise=plan.prefix_noise,
            temporal=plan.temporal,
            source_h=plan.target_hw[0],
            source_w=plan.target_hw[1],
        )
        # These selectors describe low/probe interventions. High keeps its native
        # learned complement and target-audio positions. The free band's local
        # query groups retain their low/probe dense policy through attention_head_t.
        controls = {
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY: PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
            PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY: PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
            PARTITIONED_VDN_TEMPORAL_CARRIER_KEY: PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY: PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
            PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
        }
        previous = {key: transformer[key] for key in controls if key in transformer}
        transformer.update(controls)
        try:
            with _partitioned_stage_contract(guider, high_plan, metrics, attention_head_t=attention_head_t):
                yield
        finally:
            for key in controls:
                if key in previous:
                    transformer[key] = previous[key]
                else:
                    transformer.pop(key, None)


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
    required_temporal_carrier_policy: str = PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    required_softmax_diagnostic: str = PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    required_native_carrier: str = PARTITIONED_NATIVE_CARRIER_SOURCE,
    required_domain_stream: bool = False,
) -> str | None:
    object_patches = getattr(patcher, "object_patches", None)
    if not isinstance(object_patches, dict):
        raise PartitionedPreflightUnsupported("VDN object-patch ownership is unavailable")
    from .boundary_witness import WITNESS_DIRECTORY_OPTION, witness_requested

    model_options = getattr(patcher, "model_options", None)
    witness_directory = model_options.get(WITNESS_DIRECTORY_OPTION, None) if isinstance(model_options, dict) else None
    # Target-band capture uses scheduler-owned stage snapshots, not the VDN
    # short-convolution feature sink used by the reduced-grid carrier.
    boundary_witness_requested = required_native_carrier != PARTITIONED_NATIVE_CARRIER_TARGET and witness_requested(
        witness_directory
    )
    required_temporal_carrier_policy = normalize_vdn_temporal_carrier_policy(required_temporal_carrier_policy)
    required_softmax_diagnostic = normalize_partitioned_softmax_diagnostic(required_softmax_diagnostic)
    carrier_specs: set[str] = set()
    matched = 0
    for key, owner in object_patches.items():
        if not key.startswith("diffusion_model.blocks.") or not key.endswith(".attn.forward"):
            continue
        if not getattr(owner, "_vdn_forward", False):
            continue
        matched += 1
        if boundary_witness_requested and getattr(owner, "_vdn_partitioned_boundary_witness_api", 0) != 1:
            raise RuntimeError("requested boundary witness requires paired VDN witness API 1 before sampling")
        if int(getattr(owner, "_vdn_partitioned_boundary_query_api", 0)) != VDN_PARTITIONED_BOUNDARY_QUERY_API:
            raise PartitionedPreflightUnsupported(
                f"partitioned exact-prefix requires paired VDN boundary-query API v{VDN_PARTITIONED_BOUNDARY_QUERY_API}"
            )
        if getattr(owner, "_vdn_partitioned_boundary_query_policy", None) != VDN_PARTITIONED_BOUNDARY_QUERY_POLICY:
            raise PartitionedPreflightUnsupported(
                "installed VDN bridge does not advertise the required continuation "
                f"boundary-query policy {VDN_PARTITIONED_BOUNDARY_QUERY_POLICY!r}"
            )
        if int(getattr(owner, "_vdn_external_sequence_api", 0)) != VDN_PARTITIONED_SEQUENCE_API:
            raise PartitionedPreflightUnsupported(
                f"VDN partitioned external-sequence API {VDN_PARTITIONED_SEQUENCE_API} is unavailable"
            )
        if required_domain_stream and int(getattr(owner, "_vdn_partitioned_domain_stream_api", 0)) != (
            TARGET_BAND_DOMAIN_STREAM_API
        ):
            raise PartitionedPreflightUnsupported(
                "target_band_context='domain_uniform_v1' requires a VDN-H3-Plus release that accepts "
                f"domain-stream API v{TARGET_BAND_DOMAIN_STREAM_API}"
            )
        if required_native_carrier != PARTITIONED_NATIVE_CARRIER_SOURCE and required_native_carrier not in tuple(
            getattr(owner, "_vdn_partitioned_native_carrier_grids", ())
        ):
            raise PartitionedPreflightUnsupported(
                "progressive_target_band requires a VDN-H3-Plus release that accepts a "
                f"{required_native_carrier!r} native partition carrier"
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
        if required_softmax_diagnostic != PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
            if int(getattr(owner, "_vdn_partitioned_softmax_diagnostic_api", 0)) != PARTITIONED_SOFTMAX_DIAGNOSTIC_API:
                raise PartitionedPreflightUnsupported(
                    "partitioned softmax diagnostic requires paired VDN softmax diagnostic "
                    f"API v{PARTITIONED_SOFTMAX_DIAGNOSTIC_API}"
                )
            supported_softmax = tuple(getattr(owner, "_vdn_partitioned_softmax_diagnostic_modes", ()))
            if required_softmax_diagnostic not in supported_softmax:
                raise PartitionedPreflightUnsupported(
                    f"partitioned softmax diagnostic {required_softmax_diagnostic!r} was requested but "
                    "the installed VDN bridge does not advertise that mode"
                )
        if required_temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION:
            if int(getattr(owner, "_vdn_partitioned_temporal_carrier_api", 0)) != PARTITIONED_VDN_TEMPORAL_CARRIER_API:
                raise PartitionedPreflightUnsupported(
                    "destination-grid temporal stencil requires paired VDN temporal-carrier API "
                    f"v{PARTITIONED_VDN_TEMPORAL_CARRIER_API}"
                )
            policies = tuple(getattr(owner, "_vdn_partitioned_temporal_carrier_policies", ()))
            if required_temporal_carrier_policy not in policies:
                raise PartitionedPreflightUnsupported(
                    "installed VDN bridge does not advertise destination-grid temporal-stencil capability"
                )
            spec = getattr(owner, "_vdn_partitioned_temporal_carrier_short_conv_spec", None)
            if not isinstance(spec, str) or not spec:
                raise PartitionedPreflightUnsupported(
                    "installed VDN bridge did not publish its checkpoint short-conv specification"
                )
            carrier_specs.add(spec)
    if matched == 0:
        raise PartitionedPreflightUnsupported("partitioned exact-prefix requires active VDN-H3 ownership")
    if required_temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION:
        if len(carrier_specs) != 1:
            raise PartitionedPreflightUnsupported(
                "destination-grid temporal stencil requires one consistent VDN checkpoint short-conv specification"
            )
        return next(iter(carrier_specs))
    return None


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


def _verify_partitioned_softmax_diagnostic(
    metrics,
    mode: str,
    *,
    calls_before: int,
    q_rows_before: int,
    kv_rows_before: int,
) -> None:
    """Fail closed when the same-domain dense suffix discriminator did not execute."""

    mode = normalize_partitioned_softmax_diagnostic(mode)
    if mode == PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
        return
    counters = getattr(metrics, "counters", {})
    if mode == PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK:
        calls = int(counters.get("partitioned_vdn_target_sink_measure_calls", 0)) - int(calls_before)
        q_rows = int(counters.get("partitioned_vdn_target_sink_measure_q_rows", 0)) - int(q_rows_before)
        kv_rows = int(counters.get("partitioned_vdn_target_sink_measure_kv_rows", 0)) - int(kv_rows_before)
        if min(calls, q_rows, kv_rows) <= 0:
            raise RuntimeError(
                "target-query sink measure was requested but no completed biased target-query work was observed"
            )
        metrics.event(
            "partitioned_softmax_diagnostic_verified",
            mode=mode,
            target_query_calls=calls,
            target_query_q_rows=q_rows,
            target_query_kv_rows=kv_rows,
            reduced_query_measure_unchanged=True,
            global_query_measure_unchanged=True,
            linear_measure_unchanged=True,
            same_gathered_domain=True,
        )
        return
    calls = int(counters.get("partitioned_vdn_dense_suffix_same_domain_calls", 0)) - int(calls_before)
    q_rows = int(counters.get("partitioned_vdn_dense_suffix_same_domain_q_rows", 0)) - int(q_rows_before)
    kv_rows = int(counters.get("partitioned_vdn_dense_suffix_same_domain_kv_rows", 0)) - int(kv_rows_before)
    if min(calls, q_rows, kv_rows) <= 0:
        raise RuntimeError(
            "same-domain dense suffix diagnostic was requested but no verified suffix local-query "
            "dense work was observed; refusing this diagnostic sample"
        )
    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_softmax_diagnostic_verified",
            mode=mode,
            dense_suffix_calls=calls,
            dense_suffix_q_rows=q_rows,
            dense_suffix_kv_rows=kv_rows,
            same_gathered_domain=True,
            prefix_measure_unchanged=True,
            grouped_ownership_unchanged=True,
            fail_closed=True,
        )


def _verify_partitioned_vdn_temporal_carrier_policy(
    metrics,
    policy: str,
    *,
    calls_before: int,
    taps_before: int,
    carriers_before: int,
    rows_before: int,
    events_before: int,
) -> None:
    policy = normalize_vdn_temporal_carrier_policy(policy)
    if policy == PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
        return
    counters = getattr(metrics, "counters", {})
    calls = int(counters.get("partitioned_vdn_destination_grid_stencil_calls", 0)) - int(calls_before)
    taps = int(counters.get("partitioned_vdn_destination_grid_stencil_taps", 0)) - int(taps_before)
    carriers = int(counters.get("partitioned_vdn_destination_grid_stencil_carriers", 0)) - int(carriers_before)
    rows = int(counters.get("partitioned_vdn_destination_grid_stencil_rows", 0)) - int(rows_before)
    if min(calls, taps, carriers, rows) <= 0:
        raise RuntimeError(
            "destination-grid temporal stencil was requested but no verified cross-grid carrier work was observed"
        )
    events = getattr(metrics, "events", ())
    receipts = [
        event
        for event in events[int(events_before) :]
        if getattr(event, "kind", None) == "partitioned_vdn_temporal_carrier_stage"
        and getattr(event, "fields", {}).get("policy") == policy
    ]
    digests = {event.fields.get("numerical_digest") for event in receipts}
    if len(digests) != 1 or not all(isinstance(value, str) and len(value) == 64 for value in digests):
        raise RuntimeError("destination-grid temporal stencil numerical identity was not stable across low/probe")
    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_vdn_temporal_carrier_verified",
            policy=policy,
            numerical_digest=next(iter(digests)),
            calls=calls,
            cross_grid_taps=taps,
            mapped_carriers=carriers,
            mapped_carrier_rows=rows,
            fail_closed=True,
            output_space_mutation=False,
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
    # The audio-position policy rewrites block-0 RoPE positions only. These VDN
    # modes change the linear complement or the target-prefix key measure and
    # leave positions untouched, so each is verified by its own counters.
    compatible_vdn_modes = {
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    }
    if vdn_linear_diagnostic not in compatible_vdn_modes:
        raise PartitionedPreflightUnsupported(
            "source_carrier audio-position candidate requires vdn_linear_diagnostic="
            "'normal', 'bypass_partitioned_linear', 'suppress_cross_grid_temporal_taps' or 'raw_token_measure'"
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
            coherent_exact_model_mask_calls=int(counters.get("coherent_exact_audio_model_mask_calls", 0)),
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


def _validate_partitioned_sol_sink_measure(mode: str) -> None:
    if mode != PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK:
        return
    try:
        from sol_h3 import partitioned_request as sol_request
    except ImportError as exc:
        raise PartitionedPreflightUnsupported(
            "target-query sink measure requires paired Sol sink-measure API 1"
        ) from exc
    if getattr(sol_request, "PARTITIONED_SINK_MEASURE_API", 0) != 1:
        raise PartitionedPreflightUnsupported("target-query sink measure requires paired Sol sink-measure API 1")


# Spatial-stage controls whose continuation performs a real learned spatial transfer.
_LEARNED_TRANSFER_SPATIAL_STAGES = frozenset(
    {PARTITIONED_SPATIAL_STAGE_PROGRESSIVE, PARTITIONED_SPATIAL_STAGE_TARGET_BAND}
)


def _validate_target_band_configuration(
    *,
    handoff_transfer_control: str,
    vdn_temporal_carrier_policy: str,
    prefix_transformer_context: str,
    low_probe_execution_source: str,
    audio_handoff_source: str,
    av_handoff_source: str,
    guidance_trajectory_source: str,
    residual_mode: str,
    model_options: Any,
) -> None:
    """Reject selector combinations that progressive_target_band does not implement."""
    unsupported = []
    if handoff_transfer_control != PARTITIONED_HANDOFF_TRANSFER_LEARNED:
        unsupported.append("handoff_transfer_control must be 'learned_3d'")
    normalize_vdn_temporal_carrier_policy(vdn_temporal_carrier_policy)
    if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        unsupported.append(f"prefix_transformer_context must be {PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT!r}")
    if low_probe_execution_source != PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW:
        unsupported.append(
            f"low_probe_execution_source must be {PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW!r}"
        )
    if (
        audio_handoff_source != PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN
        or av_handoff_source != PARTITIONED_AV_HANDOFF_SOURCE_MAIN
        or guidance_trajectory_source != PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
    ):
        unsupported.append("audio, AV and guidance handoff sources must be the main partitioned sources")
    if residual_mode not in {"off", "measure"}:
        unsupported.append("frame_gauge_residual_mode must be 'off' or 'measure'")
    if unsupported:
        raise PartitionedPreflightUnsupported("progressive_target_band requires: " + "; ".join(unsupported))


def _target_band_low_stage_inputs(
    band: PartitionedTargetBandGeometry,
    *,
    noise: torch.Tensor,
    source_video_noise: torch.Tensor,
    latent_image: torch.Tensor,
    source_latent_image: torch.Tensor,
    denoise_mask: torch.Tensor,
    source_mask: torch.Tensor,
    target_shapes: list[tuple[int, ...]],
    source_shapes: list[tuple[int, ...]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the low/probe sampler noise, latent and mask in the target-band layout.

    Band frames take the target-grid sampler noise that same-grid control would
    use for those frames; tail frames take the reduced-grid noise that
    progressive_low_to_high would use. Padding has zero noise, zero latent and a
    zero (protected) mask. Stochastic samplers may write raw padding; inpainting
    masks it before model calls and the reduced-grid handoff view discards it.
    """
    head = band.head_t
    target_video_noise, target_audio_noise = unpack_streams(noise, target_shapes)
    band_noise = deterministic_video_noise(
        tuple(target_video_noise.shape),
        seed=int(seed),
        device=target_video_noise.device,
        dtype=target_video_noise.dtype,
    )
    low_noise = pack_streams(
        (
            pack_target_band_video(band_noise[:, :, :head], source_video_noise[:, :, head:], band),
            target_audio_noise,
        )
    )[0]

    target_latent_video, target_latent_audio = unpack_streams(latent_image, target_shapes)
    source_latent_video, _ = unpack_streams(source_latent_image, source_shapes)
    low_latent_image = pack_streams(
        (
            pack_target_band_video(target_latent_video[:, :, :head], source_latent_video[:, :, head:], band),
            target_latent_audio,
        )
    )[0]

    target_mask_video, target_mask_audio = unpack_streams(denoise_mask, target_shapes)
    source_mask_video, _ = unpack_streams(source_mask, source_shapes)
    if not bool((target_mask_video[:, :, band.protected_t : head] == 1).all().item()) or not bool(
        (source_mask_video[:, :, head:] == 1).all().item()
    ):
        raise PartitionedPreflightUnsupported("progressive_target_band requires fully generated band and tail tokens")
    low_mask = pack_streams(
        (
            pack_target_band_video(target_mask_video[:, :, :head], source_mask_video[:, :, head:], band),
            target_mask_audio,
        )
    )[0]
    return low_noise, low_latent_image, low_mask


def _target_band_preview_packed(
    packed: torch.Tensor,
    band: PartitionedTargetBandGeometry,
    target_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    video, audio = unpack_streams(packed, target_shapes)
    return pack_streams((target_band_target_preview(video, band), audio))[0]


def _target_band_source_views(
    raw: torch.Tensor,
    clean: torch.Tensor,
    band: PartitionedTargetBandGeometry,
    *,
    target_shapes: list[tuple[int, ...]],
    source_shapes: list[tuple[int, ...]],
    handoff_state: str = PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Split low/probe results into the band's clean prediction and reduced-grid views.

    The returned packed views are uniform reduced-grid tensors, so every
    downstream handoff consumer (learned transfer, residual transport,
    diagnostics) sees the same geometry as progressive_low_to_high. By default
    the band's raw sampler state is not carried: the high stage re-noises the
    band from its clean operand like every other generated token. The
    ``carry_raw_band`` control additionally returns that raw target-grid state.
    """
    handoff_state = normalize_target_band_handoff_state(handoff_state)
    raw_video, raw_audio = unpack_streams(raw, target_shapes)
    clean_video, clean_audio = unpack_streams(clean, target_shapes)
    clean_padding = target_band_padding_max_abs(clean_video, band)
    if clean_padding != 0.0:
        raise RuntimeError("target-band probe prediction wrote outside its reduced-grid storage windows")
    band_clean = clean_video[:, :, band.protected_t : band.head_t].detach().clone()
    band_raw = (
        raw_video[:, :, band.protected_t : band.head_t].detach().clone()
        if handoff_state == PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY
        else None
    )
    source_raw, raw_shapes = pack_streams((target_band_source_view(raw_video, band), raw_audio))
    source_clean, clean_shapes = pack_streams((target_band_source_view(clean_video, band), clean_audio))
    if raw_shapes != source_shapes or clean_shapes != source_shapes:
        raise RuntimeError("target-band reduced-grid view does not match the source geometry")
    receipt = {
        "target_band_tokens": int(band.band_t),
        "protected_prefix_t": int(band.protected_t),
        "head_t": int(band.head_t),
        "temporal": int(band.temporal),
        "source_hw": (int(band.source_h), int(band.source_w)),
        "target_hw": (int(band.target_h), int(band.target_w)),
        "low_probe_video_rows": int(band.partitioned_rows),
        "band_video_rows": int(band.band_t * band.target_rows),
        "tail_video_rows": int(band.suffix_rows),
        "clean_padding_max_abs": clean_padding,
        "raw_padding_max_abs": target_band_padding_max_abs(raw_video, band),
        "band_handoff_policy": (
            TARGET_BAND_RAW_CARRY_HANDOFF_POLICY if band_raw is not None else TARGET_BAND_HANDOFF_POLICY
        ),
        "band_handoff_state": handoff_state,
        "band_raw_state_carried": band_raw is not None,
        "tail_transfer": "learned_3d",
    }
    return source_raw, source_clean, band_clean, band_raw, receipt


def _apply_target_band_head_dc_bridge(
    target_video: torch.Tensor,
    spliced_clean: torch.Tensor,
    provider_native_clean: torch.Tensor | None,
    exact_prefix: torch.Tensor,
    band: PartitionedTargetBandGeometry,
    *,
    sigma: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the suffix DC bridge at the band/tail transfer boundary.

    The context is the target-grid head the high stage actually receives: the
    authoritative prefix followed by the band's own clean prediction. The offset
    is measured against the learned provider's own rendering of that head, so it
    is exactly the transfer's channel-mean error on the context and is applied
    only to the first transferred tail token.
    """
    if provider_native_clean is None or tuple(provider_native_clean.shape) != tuple(spliced_clean.shape):
        raise RuntimeError("target-band handoff lost the learned provider's native clean output")
    head_context = spliced_clean[:, :, : band.head_t].clone()
    head_context[:, :, : band.protected_t] = exact_prefix.to(head_context)
    mapped_state, corrected_native, dc_metrics = _apply_partitioned_suffix_dc_bridge(
        target_video,
        provider_native_clean,
        head_context,
        sigma=sigma,
        enabled=enabled,
    )
    corrected_clean = spliced_clean + (corrected_native - provider_native_clean.to(corrected_native)).to(spliced_clean)
    dc_metrics = dict(dc_metrics, suffix_dc_bridge_boundary="target_band_head")
    return mapped_state, corrected_clean, dc_metrics


def _emit_target_band_tail_boundary(
    metrics,
    video: torch.Tensor,
    band: PartitionedTargetBandGeometry,
    *,
    stage: str,
    measure_trajectory: bool,
    source_hw: tuple[int, int] | None = None,
    target_hw: tuple[int, int] | None = None,
) -> None:
    """Record motion and seam evidence at the band/tail boundary for one stage.

    FFT trajectory fitting is opt-in; inexpensive neighbouring seam receipts
    remain available in ordinary runs. Diagnostic only. Neighbouring seam
    values are reported beside the band/tail pair so a discontinuity can be told apart
    from ordinary frame-to-frame change.
    """
    boundary_t = int(band.head_t)
    for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)) if measure_trajectory else ():
        trajectory = measure_translation_trajectory(
            video,
            boundary_t,
            forward_steps=4,
            backward_steps=3,
            roi_fraction=roi_fraction,
            max_shift=4,
        )
        projection = {}
        if source_hw is not None and target_hw is not None:
            projection = project_translation_trajectory_to_grid(trajectory, source_hw=source_hw, target_hw=target_hw)
        metrics.event(
            "partitioned_target_band_tail_trajectory",
            stage=stage,
            roi=roi_name,
            boundary_t=boundary_t,
            grid="source" if source_hw is not None else "target",
            output_mutated=False,
            **trajectory,
            **projection,
        )
    seams = {}
    for label, offset in (("before", -1), ("boundary", 0), ("after", 1)):
        seam_t = boundary_t + offset
        if 1 <= seam_t < int(video.shape[2]):
            values = measure_video_boundary(video, seam_t)
            seams.update({f"{label}_{name}": value for name, value in values.items() if name != "lowpass_kernel"})
    metrics.event(
        "partitioned_target_band_tail_seam",
        stage=stage,
        boundary_t=boundary_t,
        grid="source" if source_hw is not None else "target",
        output_mutated=False,
        **seams,
    )


def _validate_partitioned_sol_native_carrier(required_native_carrier: str) -> None:
    """Require Sol history identity support for a non-source native carrier."""
    if required_native_carrier == PARTITIONED_NATIVE_CARRIER_SOURCE:
        return
    try:
        from sol_h3.partitioned_history import PARTITIONED_NATIVE_CARRIER_GRIDS
    except ImportError:
        PARTITIONED_NATIVE_CARRIER_GRIDS = ()
    if required_native_carrier not in tuple(PARTITIONED_NATIVE_CARRIER_GRIDS):
        raise PartitionedPreflightUnsupported(
            "progressive_target_band requires a Sol-H3 release whose partitioned history recognizes a "
            f"{required_native_carrier!r} native partition carrier"
        )


_DOMAIN_UNIFORM_COUNTERS = (
    "partitioned_domain_uniform_calls",
    "partitioned_domain_uniform_target_block_calls",
    "partitioned_domain_uniform_source_block_calls",
)


def _domain_uniform_counters(metrics) -> tuple[int, ...]:
    return tuple(int(metrics.counters.get(name, 0)) for name in _DOMAIN_UNIFORM_COUNTERS)


def _verify_domain_uniform_stage(metrics, stage: str, domain, before: tuple[int, ...]) -> None:
    """Fail closed unless a requested domain-uniform stage actually ran both streams."""
    calls, target_blocks, source_blocks = (
        after - prior for after, prior in zip(_domain_uniform_counters(metrics), before, strict=True)
    )
    if domain is None:
        if calls or target_blocks or source_blocks:
            raise RuntimeError("domain-uniform transformer executed without a requested domain context")
        return
    if calls <= 0 or target_blocks <= 0 or target_blocks != source_blocks or target_blocks % calls:
        raise RuntimeError(
            f"domain_uniform_v1 {stage} did not execute both uniform streams through every transformer block "
            f"(calls={calls}, target_blocks={target_blocks}, source_blocks={source_blocks})"
        )
    metrics.event(
        "partitioned_target_band_domain_stage",
        stage=stage,
        policy=domain.policy,
        model_calls=calls,
        target_stream_block_calls=target_blocks,
        source_stream_block_calls=source_blocks,
        blocks_per_call=target_blocks // calls,
        verified=True,
    )


def _validate_partitioned_sol_domain_stream() -> None:
    """Require Sol history identity support for domain-uniform stream replacements."""
    try:
        from sol_h3.partitioned_history import PARTITIONED_DOMAIN_STREAM_API
    except ImportError:
        PARTITIONED_DOMAIN_STREAM_API = 0
    if PARTITIONED_DOMAIN_STREAM_API != TARGET_BAND_DOMAIN_STREAM_API:
        raise PartitionedPreflightUnsupported(
            "target_band_context='domain_uniform_v1' requires a Sol-H3 release whose partitioned history "
            f"recognizes domain-stream API v{TARGET_BAND_DOMAIN_STREAM_API}"
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
    required_vdn_temporal_carrier_policy: str = PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    required_softmax_diagnostic: str = PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    spatial_stage_control: str = PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
    target_band_tokens: int = 0,
    target_band_context: str = PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
):
    """Validate a partitioned continuation before any sampler lifetime.

    Returns ``(source_h, source_w, stage_plan, target_band)``; ``target_band`` is
    ``None`` outside progressive_target_band.
    """
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
    native_carrier = (
        PARTITIONED_NATIVE_CARRIER_TARGET
        if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND
        else PARTITIONED_NATIVE_CARRIER_SOURCE
    )
    _validate_partitioned_vdn_compat(
        guider.model_patcher,
        required_linear_diagnostic=required_vdn_linear_diagnostic,
        required_temporal_carrier_policy=required_vdn_temporal_carrier_policy,
        required_softmax_diagnostic=required_softmax_diagnostic,
        required_native_carrier=native_carrier,
        required_domain_stream=target_band_context == PARTITIONED_TARGET_BAND_CONTEXT_DOMAIN_UNIFORM,
    )
    _validate_partitioned_sol_compat(guider)
    _validate_partitioned_sol_native_carrier(native_carrier)
    if target_band_context == PARTITIONED_TARGET_BAND_CONTEXT_DOMAIN_UNIFORM:
        _validate_partitioned_sol_domain_stream()
    if required_softmax_diagnostic == PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK:
        _validate_partitioned_sol_sink_measure(required_softmax_diagnostic)
        if required_vdn_linear_diagnostic == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE:
            raise PartitionedPreflightUnsupported("target-query sink measure cannot be combined with raw_token_measure")
        if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID:
            raise PartitionedPreflightUnsupported("target-query sink measure requires heterogeneous continuation grids")

    try:
        spatial_stage_control = normalize_spatial_stage_control(spatial_stage_control)
        if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID:
            source_h, source_w = target_h, target_w
        else:
            source_h, source_w = config.resolve_source(target_h, target_w)
        internal = _process_latent_in(guider.model_patcher.model, latent_image, latent_shapes)
        stage_plan = build_partitioned_stage_plan(
            denoise_mask,
            latent_shapes,
            internal,
            noise,
            source_h=source_h,
            source_w=source_w,
            allow_same_grid=spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID,
        )
        target_band = None
        if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND:
            target_band = PartitionedTargetBandGeometry(
                protected_t=int(stage_plan.prefix_t),
                band_t=int(target_band_tokens),
                temporal=int(stage_plan.temporal),
                source_h=int(source_h),
                source_w=int(source_w),
                target_h=target_h,
                target_w=target_w,
            )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PartitionedPreflightUnsupported(str(exc)) from exc
    return source_h, source_w, stage_plan, target_band


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


def _dense_drift_sampler_contract(sampler: Any) -> tuple[bool, str]:
    """Return whether the low-stage handoff state is a deterministic flow of its initial noise."""

    name = sampler_name(sampler)
    options = getattr(sampler, "extra_options", {}) or {}
    if name in H3_DENSE_DRIFT_CHURN_SAMPLERS:
        try:
            churn = float(options.get("s_churn", 0.0))
        except (TypeError, ValueError):
            return False, name
        return churn == 0.0, name
    return name in H3_DENSE_DRIFT_DETERMINISTIC_SAMPLERS, name


def _resolve_partitioned_transfer_clean(
    target_video: torch.Tensor,
    actual_handoff_clean: torch.Tensor | None,
    *,
    handoff_noise_mode: str,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, str, str]:
    """Resolve the clean tensor from the noise contract that actually built target_video."""

    if actual_handoff_clean is not None:
        if (
            tuple(actual_handoff_clean.shape) != tuple(target_video.shape)
            or actual_handoff_clean.device != target_video.device
            or actual_handoff_clean.dtype != target_video.dtype
        ):
            raise RuntimeError("captured handoff clean tensor does not match target video geometry/device/dtype")
        if not actual_handoff_clean.is_floating_point() or not bool(torch.isfinite(actual_handoff_clean).all().item()):
            raise RuntimeError("captured handoff clean tensor is not finite floating-point video")
        return actual_handoff_clean, "actual_clean_postprocess", "actual_clean_postprocess_no_inverse"

    if handoff_noise_mode in {H3_HANDOFF_NOISE_SOURCE_RESIDUAL, H3_HANDOFF_NOISE_DENSE_DRIFT}:
        raise RuntimeError(
            "source-residual handoff lost the actual clean postprocess tensor; "
            "refusing deterministic-noise inverse recovery"
        )
    if handoff_noise_mode != H3_HANDOFF_NOISE_INDEPENDENT:
        raise RuntimeError(f"unsupported partitioned handoff noise mode {handoff_noise_mode!r}")

    recovered = _recover_partitioned_transfer_clean(
        target_video,
        sigma=sigma,
        seed=seed,
    )
    return recovered, "inverse_recovered", "inverse_conditional_renoise"


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
    """Authorize the structural overlap bridge from validated rigid evidence.

    Three fail-closed arms are eligible:

    1. the historical rigid boundary-motion veto, where registration accepted
       but the proposed rigid transform failed the native-motion preservation
       gate;
    2. a guidance-only rejection after the video registration and boundary
       motion checks have already accepted; and
    3. the 00687 hardware-invalidated rigid arm, where the complete v4 rigid
       candidate passed numerically but production mutation is deliberately
       shadow-only because decoded media disproved the global warp itself.

    The latter two arms are not permission to resurrect the rigid transform or
    to extend guidance support to a sampler it does not support. The exact-overlap
    bridge is an independent representation correction: it restores the learned
    provider's measured native prefix->suffix transition after caller-owned exact
    prefix replacement. Ambiguous/clipped video evidence remains ineligible.
    """

    result = str(transaction.get("result", ""))
    reason = str(transaction.get("reason", ""))
    boundary = transaction.get("boundary_motion")
    guidance_registration = transaction.get("guidance_registration")
    guidance_only_rejection = bool(
        result == "rejected"
        and isinstance(boundary, dict)
        and boundary.get("status") == "accepted"
        and str(boundary.get("reason", "")) == "accepted"
        and isinstance(guidance_registration, dict)
        and guidance_registration.get("status") == "rejected"
        and str(guidance_registration.get("reason", "")) == reason
    )
    if result == "rejected":
        if guidance_only_rejection:
            expected_boundary_status = "accepted"
            eligibility_reason = f"guidance_rejected_after_video_boundary_acceptance:{reason}"
        else:
            if reason not in FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS:
                return False, "frame_gauge_rejection_not_structural_overlap_eligible"
            expected_boundary_status = "rejected"
            eligibility_reason = reason
    elif result == "shadow_only":
        if reason != FRAME_GAUGE_HARDWARE_INVALIDATED_RIGID_REASON:
            return False, "frame_gauge_shadow_not_hardware_invalidated_rigid"
        if transaction.get("candidate_accepted") is not True:
            return False, "frame_gauge_shadow_candidate_not_accepted"
        if transaction.get("production_mutation_allowed") is not False:
            return False, "frame_gauge_shadow_production_mutation_not_disabled"
        if transaction.get("spatial_warp_applied") is not False:
            return False, "frame_gauge_shadow_spatial_warp_was_applied"
        expected_boundary_status = "accepted"
        eligibility_reason = reason
    else:
        return False, "frame_gauge_not_structural_overlap_eligible"

    video_registration = transaction.get("video_registration")
    if not isinstance(video_registration, dict) or video_registration.get("status") != "accepted":
        return False, "video_registration_not_accepted"
    if (
        not isinstance(boundary, dict)
        or boundary.get("status") != expected_boundary_status
        or boundary.get("policy")
        not in {
            "native_boundary_motion_preservation_v2",
            "native_boundary_motion_consensus_v3",
            "native_boundary_motion_consensus_v4",
        }
    ):
        return False, "boundary_receipt_inconsistent"
    if result == "rejected" and not guidance_only_rejection and str(boundary.get("reason", "")) != reason:
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
    return True, eligibility_reason


def _apply_partitioned_exact_overlap_bridge(
    target_video: torch.Tensor,
    learned_clean: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    sigma: float,
    weights: tuple[float, ...] = (1.0,),
    dc_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, float | int | bool]]:
    """Preserve the provider's first-transition channel means after prefix restore.

    A spatially varying prefix residual is not a model or decoder symmetry.
    Do not transplant it into generated content or extrapolate it through later
    suffix tokens. Map only the first-token DC correction onto the existing
    conditional state, preserving its noise and all other tokens.
    """

    normalized_weights = tuple(float(weight) for weight in weights)
    if normalized_weights != (1.0,):
        raise RuntimeError(
            "production exact-overlap repair is one-token DC-only; "
            "multi-token or structural successor transport is retired"
        )
    prefix_t = int(exact_prefix.shape[2])
    representation_metrics = disabled_suffix_representation_bridge_metrics(
        prefix_t=prefix_t,
        requested=False,
        weights=(1.0,),
    )
    representation_metrics.update(
        suffix_representation_bridge_reason="structural_successor_transport_retired",
        suffix_representation_bridge_production_retired=True,
        suffix_representation_bridge_hardware_verdict="rendered_continuity_unqualified",
    )
    if not dc_enabled:
        return (
            target_video.clone(),
            learned_clean.clone(),
            representation_metrics,
            disabled_suffix_dc_bridge_metrics(prefix_t=prefix_t),
        )
    corrected_clean, dc_metrics = apply_suffix_dc_bridge(
        learned_clean,
        exact_prefix,
        weights=(1.0,),
    )
    corrected_tokens = int(dc_metrics["suffix_dc_bridge_corrected_tokens"])
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


def _prepare_handoff_guidance_reference(
    *,
    run: Any,
    provider_input: torch.Tensor,
    provider_output: torch.Tensor,
    prefix_t: int,
    split_coordinate: float,
    high_sigmas: torch.Tensor,
    video_shift: float,
) -> HandoffGuidanceReference:
    """Bind an already executed learned transfer to its exact guidance endpoint."""

    source_ref, resolved, clamped = time_matched_reference_info(run, split_coordinate)
    exact_probe = any(
        sample.provenance == "actual"
        and sample.phase == "handoff_probe"
        and math.isclose(float(sample.coordinate), split_coordinate, rel_tol=0.0, abs_tol=1e-7)
        for sample in run.samples
    )
    if clamped or not exact_probe or not math.isclose(resolved, split_coordinate, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError("learned handoff guidance lost its exact probe endpoint")
    if source_ref.shape != provider_input.shape or not 0 < prefix_t < source_ref.shape[2]:
        raise RuntimeError("learned handoff guidance source geometry drifted")
    if provider_output.shape[:3] != provider_input.shape[:3]:
        raise RuntimeError("learned handoff guidance target temporal geometry drifted")
    # PREDICT_NOISE capture precedes the sampler's output algebra. The exact
    # one-call probe subsequently multiplies by (1-sigma) and CONST inverse
    # scaling divides by the same value; that round-trip is mathematically
    # identity but is not required to be bit-exact in floating point. The
    # provider then consumes this returned probe after authoritative prefix
    # restoration. Bind guidance to that actual provider input/output pair while
    # retaining the captured run only for endpoint and trajectory provenance.
    captured_source = source_ref.to(device=provider_input.device, dtype=provider_input.dtype)
    captured_delta = provider_input.float() - captured_source.float()
    prefix_delta = captured_delta[:, :, :prefix_t]
    suffix_delta = captured_delta[:, :, prefix_t:]

    for sigma in high_sigmas[:-1].detach().to(device="cpu", dtype=torch.float64).tolist():
        coordinate = float(normalized_coordinate(sigma, video_shift=video_shift))
        _, endpoint, _ = time_matched_reference_info(run, coordinate)
        if coordinate > split_coordinate + 1e-7 or not math.isclose(
            endpoint, split_coordinate, rel_tol=0.0, abs_tol=1e-7
        ):
            raise RuntimeError("learned handoff guidance high schedule changes endpoint identity")

    owner = HandoffGuidanceReference(
        provider_input,
        provider_output,
        run_id=run.run_id,
        coordinate=split_coordinate,
        prefix_t=prefix_t,
    )
    source_delta_summary = (
        torch.stack(
            (
                prefix_delta.square().mean().sqrt(),
                prefix_delta.abs().max(),
                suffix_delta.square().mean().sqrt(),
                suffix_delta.abs().max(),
            )
        )
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    (
        owner.captured_source_prefix_delta_rms,
        owner.captured_source_prefix_delta_abs_max,
        owner.captured_source_suffix_delta_rms,
        owner.captured_source_suffix_delta_abs_max,
    ) = map(float, source_delta_summary.tolist())
    return owner


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
    in at least one informative ROI. Every ROI still has veto power for a
    material regression, but a degradation no larger than one 1/16-cell
    frame-gauge fine-search quantum is treated as phase-estimator resolution
    noise rather than as an exact zero-tolerance cliff. The full-frame and
    upper-region witnesses therefore remain symmetric safety checks without
    requiring both crops to clear the same strong-improvement threshold.
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
        error_delta = after_error - before_error
        checks[name] = {
            "roi_fraction": roi_fraction,
            "native": native,
            "transformed_native": transformed_native,
            "exact_restored": restored,
            "candidate": candidate,
            "before_error_cells": before_error,
            "after_error_cells": after_error,
            "error_improvement_ratio": improvement,
            "error_delta_cells": error_delta,
            "within_degradation_bound": bool(error_delta <= FRAME_GAUGE_BOUNDARY_MAX_DEGRADATION_CELLS + 1e-12),
            "informative": bool(before_error >= FRAME_GAUGE_BOUNDARY_MIN_ERROR_CELLS),
        }

    fields: dict[str, Any] = {
        "policy": "native_boundary_motion_consensus_v4",
        "min_error_cells": FRAME_GAUGE_BOUNDARY_MIN_ERROR_CELLS,
        "min_improvement_ratio": FRAME_GAUGE_BOUNDARY_MIN_IMPROVEMENT,
        "min_response": FRAME_GAUGE_BOUNDARY_MIN_RESPONSE,
        "max_degradation_cells": FRAME_GAUGE_BOUNDARY_MAX_DEGRADATION_CELLS,
        "degradation_bound_basis": "frame_gauge_fine_search_quantum_1_over_16_cell",
        "acceptance_rule": "all_rois_within_one_fine_search_tick_and_any_informative_roi_strong_v1",
        "checks": checks,
    }

    informative_rois: list[str] = []
    strong_improvement_rois: list[str] = []
    for name in ("upper45", "full"):
        check = checks[name]
        for variant in ("native", "transformed_native", "exact_restored", "candidate"):
            receipt = check[variant]
            if receipt["clipped"] or receipt["response"] < FRAME_GAUGE_BOUNDARY_MIN_RESPONSE:
                fields["status"] = "rejected"
                fields["reason"] = f"boundary_{name}_{variant}_ambiguous"
                return False, fields, str(fields["reason"])
        # Every ROI remains a material-degradation veto. The phase-estimator
        # boundary witness is continuous-valued while the authoritative rigid
        # registration is selected on a 1/16-cell fine grid, so an error increase
        # no larger than one fine-search quantum is treated as measurement-floor
        # disagreement. Anything larger still fails closed.
        if (
            check["after_error_cells"] - check["before_error_cells"]
            > FRAME_GAUGE_BOUNDARY_MAX_DEGRADATION_CELLS + 1e-12
        ):
            fields["status"] = "rejected"
            fields["reason"] = f"boundary_{name}_degraded_over_bound"
            return False, fields, str(fields["reason"])
        if check["informative"]:
            informative_rois.append(name)
            if check["error_improvement_ratio"] >= FRAME_GAUGE_BOUNDARY_MIN_IMPROVEMENT:
                strong_improvement_rois.append(name)

    fields["informative_rois"] = informative_rois
    fields["strong_improvement_rois"] = strong_improvement_rois
    if informative_rois and not strong_improvement_rois:
        fields["status"] = "rejected"
        fields["reason"] = "boundary_no_informative_roi_strong_improvement"
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
            # Guidance registration owns only the optional high-stage reference.
            # Preserve the independently validated provider boundary witness so a
            # guidance-only rejection cannot suppress exact-prefix representation
            # reconciliation at the learned low->high handoff.
            witnesses = {"learned_boundary_pair": learned_clean[:, :, prefix_t - 1 : prefix_t + 1].detach().clone()}
            return result, None, witnesses, transaction

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

    # Materialize the rigid candidate only as a bounded shadow witness. Run
    # 00687 proved that a fully accepted/applied whole-suffix rigid transaction
    # can still retain the user-visible frame shift. Do not let paired-prefix
    # calibration mutate production video or Flow guidance until a regional
    # boundary model is validated against decoded media.
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
        raise RuntimeError("frame-gauge accepted boundary witness but shadow translation exceeded invalid-area bound")
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
    candidate_corrected_clean, candidate_dc_metrics = apply_suffix_dc_bridge(
        aligned_full,
        exact_prefix,
        weights=(1.0,),
        clone_output=False,
    )
    # The shifted prefix is a disposable shadow witness only.
    candidate_corrected_clean[:, :, :prefix_t] = learned_clean[:, :, :prefix_t]
    if not torch.equal(
        candidate_corrected_clean[:, :, :prefix_t],
        learned_clean[:, :, :prefix_t],
    ):
        raise RuntimeError("frame-gauge shadow candidate altered learned prefix ownership")

    if residual_mode == "measure":
        video_residual = residual_geometry.get("video", {})
        tile_bounds = video_residual.get("tile_bounds")
        if isinstance(tile_bounds, dict):
            residual_geometry["boundary_regional"] = _regional_boundary_motion_receipts(
                learned_clean,
                exact_prefix,
                aligned_witness,
                candidate_corrected_clean,
                prefix_t=prefix_t,
                tile_bounds=tile_bounds,
            )
        else:
            residual_geometry["boundary_regional"] = {
                "policy": "paired_prefix_residual_boundary_regions_v1",
                "status": "not_evaluated",
                "reason": "regional_measurement_unavailable",
            }

    residual_geometry.update(
        final_path="production_baseline_rigid_shadow_only",
        rigid_candidate_evaluated=True,
        rigid_candidate_production_applied=False,
    )
    video_registration = dict(transaction["video_registration"])
    video_registration["invalid_fraction"] = float(aligned_application.invalid_fraction)
    transaction["video_registration"] = video_registration
    transaction.update(
        result="shadow_only",
        reason="hardware_invalidated_global_rigid_application_00687",
        candidate_accepted=True,
        production_mutation_allowed=False,
        candidate_spatial_warp_computed=True,
        candidate_guidance_reference_computed=bool(registered_reference is not None),
        candidate_dc_applied=True,
        candidate_dc_metrics=candidate_dc_metrics,
        dc_applied_in_clean_hook=False,
        spatial_warp_applied=False,
        invalid_fraction=aligned_application.invalid_fraction,
        dc_policy="existing_one_token_spatial_mean_v1",
        dc_order="production_baseline_after_shadow_candidate",
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
        "learned_boundary_pair": (learned_clean[:, :, prefix_t - 1 : prefix_t + 1].detach().clone()),
        "paired_prefix_aligned_witness": aligned_witness,
        "candidate_corrected_clean": (candidate_corrected_clean[:, :, :diagnostic_end].detach().clone()),
    }
    witnesses.update(residual_witnesses)
    result = CleanVideoPostprocessResult(
        clean_video=learned_clean,
        protected_prefix_t=prefix_t,
        metadata=transaction,
    )
    return result, None, witnesses, transaction


def _emit_bicubic_transfer_shadow_trajectory(
    metrics,
    clean_video: torch.Tensor,
    *,
    prefix_t: int,
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Measure a spatial-only transfer shadow without changing production state."""

    shadow = resize_video(clean_video, target_h, target_w, mode="bicubic")
    expected_shape = (*clean_video.shape[:-2], int(target_h), int(target_w))
    if tuple(shadow.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"bicubic transfer shadow returned shape {tuple(shadow.shape)}; expected {tuple(expected_shape)}"
        )
    for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
        trajectory = measure_translation_trajectory(
            shadow,
            prefix_t,
            forward_steps=4,
            backward_steps=3,
            roi_fraction=roi_fraction,
            max_shift=4,
        )
        metrics.event(
            "partitioned_multiframe_trajectory",
            stage="source_low_exact_context_bicubic_shadow",
            roi=roi_name,
            source_hw=(int(source_h), int(source_w)),
            target_hw=(int(target_h), int(target_w)),
            transfer_mode="bicubic",
            diagnostic_only=True,
            production_gate=False,
            extra_h3_nfe=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            **trajectory,
        )
    return shadow


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
    vdn_temporal_carrier_policy = normalize_vdn_temporal_carrier_policy(
        initial_transformer.get(
            PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
            PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
        )
    )
    softmax_diagnostic = normalize_partitioned_softmax_diagnostic(
        initial_transformer.get(
            PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
            PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
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
    if vdn_temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION:
        if vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
            raise PartitionedPreflightUnsupported(
                "destination-grid temporal stencil requires vdn_linear_diagnostic='normal'"
            )
        if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
            raise PartitionedPreflightUnsupported(
                "destination-grid temporal stencil requires prefix_transformer_context='exact_target_partitioned'"
            )
        if low_probe_execution_source == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY:
            raise PartitionedPreflightUnsupported(
                "destination-grid temporal stencil requires the exact-partitioned low/probe execution path"
            )
    if softmax_diagnostic != PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
        if prefix_transformer_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
            raise PartitionedPreflightUnsupported(
                "partitioned softmax diagnostic requires prefix_transformer_context='exact_target_partitioned'"
            )
        if low_probe_execution_source == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY:
            raise PartitionedPreflightUnsupported(
                "partitioned softmax diagnostic requires the exact-partitioned low/probe execution path"
            )
    provider_boundary_stabilization = normalize_provider_boundary_stabilization(
        initial_transformer.get(
            PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_KEY,
            PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF,
        )
    )
    handoff_transfer_control = normalize_handoff_transfer_control(
        initial_transformer.get(
            PARTITIONED_HANDOFF_TRANSFER_CONTROL_KEY,
            PARTITIONED_HANDOFF_TRANSFER_LEARNED,
        )
    )
    spatial_stage_control = normalize_spatial_stage_control(
        initial_transformer.get(
            PARTITIONED_SPATIAL_STAGE_CONTROL_KEY,
            PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
        )
    )
    suffix_dc_bridge_enabled = resolve_partitioned_suffix_dc_bridge(initial_transformer)
    band_mode = spatial_stage_control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND
    from .boundary_witness import WITNESS_DIRECTORY_OPTION, witness_requested

    band_witness_requested = band_mode and witness_requested(initial_model_options.get(WITNESS_DIRECTORY_OPTION))
    target_band_tokens = resolve_partitioned_target_band_tokens(initial_transformer) if band_mode else 0
    target_band_handoff_state = resolve_partitioned_target_band_handoff_state(initial_transformer)
    target_band_context = resolve_partitioned_target_band_context(initial_transformer)
    if not band_mode and (
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY in initial_transformer
        or PARTITIONED_TARGET_BAND_CONTEXT_KEY in initial_transformer
    ):
        raise PartitionedPreflightUnsupported(
            "target-band handoff-state and context selectors require spatial_stage_control='progressive_target_band'"
        )
    if (
        spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID
        and handoff_transfer_control != PARTITIONED_HANDOFF_TRANSFER_LEARNED
    ):
        raise PartitionedPreflightUnsupported(
            "same-grid spatial-stage control requires handoff_transfer_control='learned_3d'"
        )
    if (
        spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID
        and vdn_temporal_carrier_policy != PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE
    ):
        raise PartitionedPreflightUnsupported(
            "same-grid spatial-stage control requires vdn_temporal_carrier_policy='native_grid_then_map_v1'"
        )
    if band_mode:
        _validate_target_band_configuration(
            handoff_transfer_control=handoff_transfer_control,
            vdn_temporal_carrier_policy=vdn_temporal_carrier_policy,
            prefix_transformer_context=prefix_transformer_context,
            low_probe_execution_source=low_probe_execution_source,
            audio_handoff_source=audio_handoff_source,
            av_handoff_source=av_handoff_source,
            guidance_trajectory_source=guidance_trajectory_source,
            residual_mode=normalize_residual_geometry_mode(config.frame_gauge_residual_mode),
            model_options=initial_model_options,
        )
    domain_context_requested = band_mode and target_band_context == PARTITIONED_TARGET_BAND_CONTEXT_DOMAIN_UNIFORM
    if domain_context_requested:
        # Each domain stream is a uniform grid: there are no cross-grid temporal
        # taps, destination stencils or non-unit key measures for these
        # selectors to act on, so they cannot be combined with this context.
        unsupported = []
        if vdn_linear_diagnostic != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL:
            unsupported.append("vdn_linear_diagnostic='normal'")
        if softmax_diagnostic != PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
            unsupported.append("softmax_diagnostic='normal'")
        if vdn_temporal_carrier_policy != PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
            unsupported.append("vdn_temporal_carrier_policy='native_grid_then_map_v1'")
        if unsupported:
            raise PartitionedPreflightUnsupported(
                "target_band_context='domain_uniform_v1' requires " + "; ".join(unsupported)
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
            handoff_transfer_control=handoff_transfer_control,
            spatial_stage_control=spatial_stage_control,
            learned_transfer_unchanged=(
                handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_LEARNED
                and spatial_stage_control == PARTITIONED_SPATIAL_STAGE_PROGRESSIVE
            ),
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
    structural_denoise_mask = denoise_mask if exact_denoise_mask is None else exact_denoise_mask
    if tuple(structural_denoise_mask.shape) != tuple(denoise_mask.shape):
        raise PartitionedPreflightUnsupported(
            "partitioned exact-prefix structural mask geometry drifted from runtime sampler mask"
        )
    source_h, source_w, stage_plan, target_band = _preflight(
        guider,
        config,
        noise,
        latent_image,
        structural_denoise_mask,
        latent_shapes,
        required_vdn_linear_diagnostic=vdn_linear_diagnostic,
        required_vdn_temporal_carrier_policy=vdn_temporal_carrier_policy,
        required_softmax_diagnostic=softmax_diagnostic,
        spatial_stage_control=spatial_stage_control,
        target_band_tokens=target_band_tokens,
        target_band_context=target_band_context if band_mode else PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
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
    configured_source_h, configured_source_w = config.resolve_source(target_h, target_w)
    selected_coordinate = config.resolve_coordinate(
        configured_source_h,
        configured_source_w,
        target_h,
        target_w,
    )
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
    transfer_lattice_provider = None
    if (
        spatial_stage_control != PARTITIONED_SPATIAL_STAGE_SAME_GRID
        and handoff_transfer_control != PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL
    ):
        transfer_lattice_provider = H3PatchLatticeTransferProvider(config.learned_upscaler)
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
        policy=H3_TRANSFER_LATTICE,
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
    target_band_domain = None
    if domain_context_requested:
        head_t = int(stage_plan.prefix_t) + int(target_band_tokens)
        weight_square = h3_patch_lattice_weight_square_sum(target_h, target_w, source_h, source_w)
        target_band_domain = TargetBandDomainContext(
            policy=TARGET_BAND_DOMAIN_UNIFORM_POLICY,
            source_prefix=physical_prefix_source.detach().to(torch.float32).clone(),
            source_prefix_noise=source_video_noise[:, :, : stage_plan.prefix_t].detach().to(torch.float32).clone(),
            source_band_noise=source_video_noise[:, :, stage_plan.prefix_t : head_t].detach().to(torch.float32).clone(),
            band_noise_complement=(1.0 - weight_square).clamp_min(0.0).sqrt(),
            model_noise_scale=float(getattr(guider.model_patcher.model.model_sampling, "noise_scale", 1.0)),
        )
        binding.metrics.event(
            "partitioned_target_band_domain_plan",
            policy=TARGET_BAND_DOMAIN_UNIFORM_POLICY,
            band_carrier_policy=TARGET_BAND_DOMAIN_BAND_CARRIER_POLICY,
            protected_prefix_t=int(stage_plan.prefix_t),
            head_t=head_t,
            temporal=int(stage_plan.temporal),
            target_hw=(int(target_h), int(target_w)),
            source_hw=(int(source_h), int(source_w)),
            projected_noise_variance_min=float(weight_square.min().item()),
            projected_noise_variance_mean=float(weight_square.mean().item()),
            projected_noise_variance_max=float(weight_square.max().item()),
            model_noise_scale=float(target_band_domain.model_noise_scale),
            vdn_linear_diagnostic=vdn_linear_diagnostic,
            softmax_diagnostic=softmax_diagnostic,
            vdn_temporal_carrier_policy=vdn_temporal_carrier_policy,
            # Each stream is an ordinary uniform clip, so its audio rows use that
            # clip's native spatial endpoints; the mixed-grid audio-position
            # selector has no row to act on in this context.
            audio_position_domain_requested=audio_position_domain,
            stream_audio_positions="per_stream_native",
            target_stream_native_clip_length=(head_t % 5 == 2),
            high_stage_changed=False,
            handoff_changed=False,
            extra_sampler_lifetimes=0,
        )
        del weight_square
    del low_video, low_audio
    low_mask = _resize_packed_mask(denoise_mask, target_shapes, source_shapes)
    # Shapes of the low/probe sampler state. The target-band layout keeps that
    # state on the target grid; every later handoff consumer uses source views.
    low_shapes = source_shapes
    if target_band is not None:
        low_shapes = list(target_shapes)
        low_noise, low_latent_image, low_mask = _target_band_low_stage_inputs(
            target_band,
            noise=noise,
            source_video_noise=source_video_noise,
            latent_image=latent_image,
            source_latent_image=low_latent_image,
            denoise_mask=denoise_mask,
            source_mask=low_mask,
            target_shapes=target_shapes,
            source_shapes=source_shapes,
            seed=int(seed or 0) + config.source_noise_offset,
        )

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
        configured_progressive_source_hw=(configured_source_h, configured_source_w),
        spatial_stage_control=spatial_stage_control,
        same_grid_control_active=spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID,
        handoff_split_preserved_from_configured_source=True,
        prefix_video_rows=stage_plan.prefix_rows,
        suffix_video_rows=(target_band or stage_plan).suffix_rows,
        partitioned_video_rows=(target_band or stage_plan).partitioned_rows,
        target_band_tokens=int(target_band.band_t) if target_band is not None else 0,
        target_band_video_rows=(target_band.band_t * target_band.target_rows) if target_band is not None else 0,
        low_probe_sampler_shape=tuple(low_shapes[0]),
        native_carrier_grid="target" if target_band is not None else "source",
        prefix_exact_latent_resized_for_transformer=(
            prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
        ),
        prefix_target_grid_rows_injected=(prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT),
        deprecated_mixed_grid_contract_active=False,
        vdn_linear_diagnostic=vdn_linear_diagnostic,
        vdn_temporal_carrier_policy=vdn_temporal_carrier_policy,
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
    softmax_counter = (
        "partitioned_vdn_target_sink_measure"
        if softmax_diagnostic == PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK
        else "partitioned_vdn_dense_suffix_same_domain"
    )
    dense_suffix_calls_before = int(binding.metrics.counters.get(softmax_counter + "_calls", 0))
    dense_suffix_q_rows_before = int(binding.metrics.counters.get(softmax_counter + "_q_rows", 0))
    dense_suffix_kv_rows_before = int(binding.metrics.counters.get(softmax_counter + "_kv_rows", 0))
    temporal_carrier_calls_before = int(
        binding.metrics.counters.get("partitioned_vdn_destination_grid_stencil_calls", 0)
    )
    temporal_carrier_taps_before = int(binding.metrics.counters.get("partitioned_vdn_destination_grid_stencil_taps", 0))
    temporal_carrier_carriers_before = int(
        binding.metrics.counters.get("partitioned_vdn_destination_grid_stencil_carriers", 0)
    )
    temporal_carrier_rows_before = int(binding.metrics.counters.get("partitioned_vdn_destination_grid_stencil_rows", 0))
    temporal_carrier_events_before = len(binding.metrics.events)
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
        if target_band is not None:
            x0 = _target_band_preview_packed(x0, target_band, target_shapes)
            x = _target_band_preview_packed(x, target_band, target_shapes)
        else:
            x0 = _resize_packed_latent_image(x0, source_shapes, target_shapes)
            x = _resize_packed_latent_image(x, source_shapes, target_shapes)
        return callback(step, x0, x, len(sigmas) - 1)

    if target_band is None:
        _begin_capture(binding, guider, sampler, low_sigmas, source_shapes)
    else:
        # Guidance consumes a uniform reduced-grid low trajectory, as in
        # progressive_low_to_high; record the reduced-grid view of each prediction.
        _begin_capture(
            binding,
            guider,
            sampler,
            low_sigmas,
            source_shapes,
            sampler_shapes=target_shapes,
            video_view=lambda video: target_band_source_view(video, target_band),
        )
    try:
        _reset_guider_conds(guider, template=conditioning_template)
        low_started = time.perf_counter()
        try:
            sampler_invocation_count += 1
            binding.metrics.increment("progressive_sampler_invocations")
            domain_calls_before = _domain_uniform_counters(binding.metrics)
            with (
                _flow_stage_contract(guider, "low"),
                _partitioned_stage_contract(
                    guider,
                    stage_plan,
                    binding.metrics,
                    target_band=target_band,
                    target_band_domain=target_band_domain,
                ),
            ):
                low_result = executor(
                    low_noise,
                    low_latent_image,
                    sampler,
                    low_sigmas,
                    low_mask,
                    low_callback,
                    disable_pbar,
                    seed,
                    latent_shapes=low_shapes,
                )
            _verify_domain_uniform_stage(binding.metrics, "low", target_band_domain, domain_calls_before)
        finally:
            binding.metrics.event(
                "low_stage_wall",
                elapsed_ms=(time.perf_counter() - low_started) * 1000.0,
                partitioned_exact_prefix=True,
            )

        base_model = guider.model_patcher.model
        source_raw = _raw_sampler_state(base_model, low_result, low_shapes, sigma)
        source_latent_internal = _process_latent_in(base_model, low_latent_image, low_shapes)
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
            domain_calls_before = _domain_uniform_counters(binding.metrics)
            with (
                _flow_stage_contract(guider, "probe"),
                _high_stage_contract(guider),
                _partitioned_stage_contract(
                    guider,
                    stage_plan,
                    binding.metrics,
                    target_band=target_band,
                    target_band_domain=target_band_domain,
                ),
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
                    latent_shapes=low_shapes,
                )
            _verify_domain_uniform_stage(binding.metrics, "probe", target_band_domain, domain_calls_before)
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
    boundary_window_evidence = None
    handoff_guidance_reference = None
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
            _verify_partitioned_softmax_diagnostic(
                binding.metrics,
                softmax_diagnostic,
                calls_before=dense_suffix_calls_before,
                q_rows_before=dense_suffix_q_rows_before,
                kv_rows_before=dense_suffix_kv_rows_before,
            )
            _verify_partitioned_vdn_temporal_carrier_policy(
                binding.metrics,
                vdn_temporal_carrier_policy,
                calls_before=temporal_carrier_calls_before,
                taps_before=temporal_carrier_taps_before,
                carriers_before=temporal_carrier_carriers_before,
                rows_before=temporal_carrier_rows_before,
                events_before=temporal_carrier_events_before,
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

        source_x0 = _process_latent_in(base_model, source_x0, low_shapes)
        band_clean_video = None
        band_raw_video = None
        measure_band_trajectory = normalize_residual_geometry_mode(config.frame_gauge_residual_mode) == "measure"
        if target_band is not None:
            if measure_band_trajectory or band_witness_requested:
                try:
                    boundary_window_evidence = BoundaryWindowEvidence(
                        stage_plan.prefix, target_band.temporal, capture_full_video=True
                    )
                except ValueError as exc:
                    if band_witness_requested:
                        raise RuntimeError(f"requested target-band stage capture is unsupported: {exc}") from exc
                    binding.metrics.event(
                        "partitioned_boundary_window_evidence",
                        policy="native_boundary_decoder_window_evidence_v1",
                        status="unsupported",
                        reason=str(exc),
                        output_mutated=False,
                        diagnostic_only=True,
                    )
                else:
                    native_probe_video, _ = unpack_streams(source_x0, target_shapes)
                    boundary_window_evidence.capture("low_probe_native_carrier_clean", native_probe_video)
                    del native_probe_video
            source_raw, source_x0, band_clean_video, band_raw_video, band_low_receipt = _target_band_source_views(
                source_raw,
                source_x0,
                target_band,
                target_shapes=target_shapes,
                source_shapes=source_shapes,
                handoff_state=target_band_handoff_state,
            )
            binding.metrics.event("partitioned_target_band_low_state", **band_low_receipt)

        # The learned 3D upscaler may use all prefix frames as transient temporal
        # context.  This resized copy never enters H3 attention and its upscaled
        # prefix output is discarded below in favor of the authoritative target
        # prefix captured before the low stage.
        clean_video, clean_audio = unpack_streams(source_x0, source_shapes)
        # Retain the clean low/probe audio only for output-neutral stage
        # diagnostics. Production target-high no longer consumes this witness.
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

            # The low/probe tensor above is in MiniMax-H3's model-internal
            # packed domain. AudioVAE consumes the caller latent domain, and H3
            # process_latent_in/out also owns sampler audio scaling. Cache a
            # converted CPU copy for the downstream decode-matched audit only.
            _cache_audio_decode_witness(
                binding,
                model_options,
                base_model,
                source_x0,
                source_shapes,
                stage="low_probe_clean",
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
        if target_band is not None:
            _emit_target_band_tail_boundary(
                binding.metrics,
                clean_video,
                target_band,
                stage="source_low",
                measure_trajectory=measure_band_trajectory,
                source_hw=(source_h, source_w),
                target_hw=(target_h, target_w),
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

        bicubic_transfer_shadow: torch.Tensor | None = None
        if normalize_residual_geometry_mode(config.frame_gauge_residual_mode) == "measure":
            bicubic_transfer_shadow = _emit_bicubic_transfer_shadow_trajectory(
                binding.metrics,
                clean_video,
                prefix_t=stage_plan.prefix_t,
                source_h=source_h,
                source_w=source_w,
                target_h=target_h,
                target_w=target_w,
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
        if target_band is not None and boundary_window_evidence is not None:
            boundary_window_evidence.capture("source_probe_clean", clean_video)
        residual_mode = normalize_residual_geometry_mode(config.frame_gauge_residual_mode)
        boundary_content_diagnostic_enabled = bool(config.frame_gauge_repair)
        boundary_content_pre_high_receipt: dict[str, Any] | None = None
        pending_registered_reference = None
        frame_gauge_witnesses: dict[str, torch.Tensor] = {}
        actual_handoff_clean: torch.Tensor | None = None
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
        band_provider_native_clean: torch.Tensor | None = None
        if target_band is not None:
            # The band keeps its own target-grid prediction, so the protected
            # prefix is followed by identical-grid content exactly as in same-grid
            # control. The prefix-boundary frame gauge therefore has no transfer
            # boundary to register; the remaining transfer boundary is the
            # band/tail edge.
            frame_gauge_transaction.update(reason="target_band_identity_boundary")

            def clean_video_postprocess(learned_clean):
                nonlocal actual_handoff_clean
                nonlocal band_provider_native_clean

                band_provider_native_clean = learned_clean.detach().clone()
                if boundary_window_evidence is not None:
                    boundary_window_evidence.capture("provider_native_clean", learned_clean)
                # Measure before splicing while the provider operand is already live.
                # Do not retain a full extra video through the handoff for diagnostics.
                _emit_target_band_tail_boundary(
                    binding.metrics,
                    learned_clean,
                    target_band,
                    stage="provider_native",
                    measure_trajectory=measure_band_trajectory,
                )
                # Diagnostic only: the provider's rendering of the band comes from a
                # reduced-grid projection of a target-grid latent, not from a native
                # reduced-grid latent, so it is compared with the band but never mixed in.
                transferred_band = learned_clean[:, :, stage_plan.prefix_t : target_band.head_t]
                overlap = measure_target_band_overlap(band_clean_video.to(transferred_band), transferred_band)
                spliced = learned_clean.clone()
                spliced[:, :, stage_plan.prefix_t : target_band.head_t] = band_clean_video.to(spliced)
                actual_handoff_clean = spliced.detach().clone()
                binding.metrics.event(
                    "partitioned_target_band_overlap",
                    policy=TARGET_BAND_HANDOFF_POLICY,
                    target_band_tokens=int(target_band.band_t),
                    protected_prefix_t=int(stage_plan.prefix_t),
                    head_t=int(target_band.head_t),
                    domain="model_internal_clean",
                    output_mutated=False,
                    extra_h3_nfe=0,
                    extra_provider_calls=0,
                    **overlap,
                )
                if residual_mode == "measure":
                    native_head = learned_clean[:, :, : target_band.head_t].clone()
                    native_head[:, :, stage_plan.prefix_t :] = band_clean_video.to(native_head)
                    binding.metrics.event(
                        "partitioned_target_band_same_frame_affine",
                        **measure_paired_prefix_affine(
                            learned_clean[:, :, : target_band.head_t],
                            native_head,
                            prefix_t=int(target_band.head_t),
                            frames=int(target_band.band_t),
                        ),
                    )
                return CleanVideoPostprocessResult(
                    clean_video=spliced,
                    protected_prefix_t=int(stage_plan.prefix_t),
                    metadata={
                        "result": "target_band_native_splice",
                        "policy": TARGET_BAND_HANDOFF_POLICY,
                        "target_band_tokens": int(target_band.band_t),
                        "native_frames": [int(stage_plan.prefix_t), int(target_band.head_t)],
                    },
                )

        elif config.frame_gauge_repair:

            def clean_video_postprocess(learned_clean):
                nonlocal pending_registered_reference
                nonlocal frame_gauge_witnesses
                nonlocal frame_gauge_transaction
                nonlocal actual_handoff_clean

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
                actual_handoff_clean = result.clean_video.detach().clone()
                return result

        transfer_started = time.perf_counter()
        transfer_metrics: dict[str, Any] = {}
        deterministic_handoff_sampler, handoff_sampler = _dense_drift_sampler_contract(sampler)
        # Target-band continuation re-noises its learned tail with independent
        # Gaussian noise, as the uniform progressive handoff does. Transporting the
        # low-stage residual into the tail carries content-correlated structure
        # into the high-stage entry noise, which the high stage sharpens.
        handoff_noise_mode = (
            (H3_HANDOFF_NOISE_DENSE_DRIFT if deterministic_handoff_sampler else H3_HANDOFF_NOISE_SOURCE_RESIDUAL)
            if config.frame_gauge_repair
            and av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_MAIN
            and target_band is None
            else H3_HANDOFF_NOISE_INDEPENDENT
        )
        if handoff_noise_mode in {H3_HANDOFF_NOISE_SOURCE_RESIDUAL, H3_HANDOFF_NOISE_DENSE_DRIFT}:
            source_state_video, _ = unpack_streams(source_raw, source_shapes)
            source_clean_video, _ = unpack_streams(source_x0, source_shapes)
            source_effective_residual = (
                source_state_video.to(torch.float32) - (1.0 - float(sigma)) * source_clean_video.to(torch.float32)
            ) / float(sigma)
            transfer_start_t = target_band.head_t if target_band is not None else stage_plan.prefix_t
            residual_suffix = source_effective_residual[:, :, transfer_start_t:]
            model_noise_scale = float(getattr(base_model.model_sampling, "noise_scale", 1.0))
            if not math.isfinite(model_noise_scale) or model_noise_scale <= 0.0:
                raise ValueError("partitioned H3 handoff requires a finite positive model noise_scale")
            initial_suffix = source_video_noise[:, :, transfer_start_t:].to(
                device=residual_suffix.device,
                dtype=torch.float32,
            )
            initial_effective_suffix = initial_suffix * model_noise_scale
            residual_delta = residual_suffix - initial_effective_suffix
            residual_norm = float(residual_suffix.norm().item())
            initial_norm = float(initial_effective_suffix.norm().item())
            residual_cosine = (
                float(torch.dot(residual_suffix.reshape(-1), initial_effective_suffix.reshape(-1)).item())
                / (residual_norm * initial_norm)
                if residual_norm > 1e-20 and initial_norm > 1e-20
                else None
            )
            binding.metrics.event(
                "partitioned_handoff_residual_provenance",
                policy=handoff_noise_mode,
                source="same_sigma_source_state_minus_clean_probe",
                generated_suffix_only=True,
                transfer_start_t=int(transfer_start_t),
                prefix_t=int(stage_plan.prefix_t),
                residual_rms=float(residual_suffix.square().mean().sqrt().item()),
                initial_source_noise_rms=float(initial_suffix.square().mean().sqrt().item()),
                initial_effective_residual_rms=float(initial_effective_suffix.square().mean().sqrt().item()),
                residual_vs_initial_rms=float(residual_delta.square().mean().sqrt().item()),
                residual_vs_initial_cosine=residual_cosine,
                model_noise_scale=model_noise_scale,
                sampler=handoff_sampler,
                dense_drift_sampler_contract=(
                    "deterministic_initial_noise_flow"
                    if deterministic_handoff_sampler
                    else "stochastic_or_unverified_sampler_gaussian_refinement"
                ),
                residual_domain="effective_flow_residual_carried_state_units",
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
            )
            del source_effective_residual, residual_suffix, initial_suffix, initial_effective_suffix, residual_delta
        effective_upscaler = config.learned_upscaler
        spatial_transfer_control = None
        source_clean_sha256 = None
        if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID:
            spatial_transfer_control = _IdentitySameGridTransferProvider(config.learned_upscaler)
            effective_upscaler = spatial_transfer_control
        elif handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL:
            # Keep the exact learned-handoff plumbing (postprocess, residual/noise
            # transport, exact-prefix restoration and target-high stage) but replace
            # the checkpoint transform itself with deterministic spatial bicubic.
            spatial_transfer_control = _BicubicSameSourceTransferProvider(config.learned_upscaler)
            effective_upscaler = spatial_transfer_control
        else:
            effective_upscaler = transfer_lattice_provider
        if spatial_transfer_control is not None:
            source_clean_control, _source_clean_audio_control = unpack_streams(source_x0, source_shapes)
            source_clean_sha256 = tensor_sha256(source_clean_control)
            del source_clean_control, _source_clean_audio_control

        target_raw, rebuilt_shapes = build_handoff_state(
            source_packed_state=source_raw,
            source_x0_packed=source_x0,
            source_shapes=source_shapes,
            sigma=sigma,
            target_h=target_h,
            target_w=target_w,
            seed=int(seed or 0) + config.seed_offset,
            transfer_mode="learned_3d",
            learned_upscaler=effective_upscaler,
            transfer_metrics=transfer_metrics,
            clean_video_postprocess=clean_video_postprocess,
            noise_mode=handoff_noise_mode,
            initial_source_noise=source_video_noise if handoff_noise_mode == H3_HANDOFF_NOISE_DENSE_DRIFT else None,
            model_noise_scale=model_noise_scale if handoff_noise_mode == H3_HANDOFF_NOISE_DENSE_DRIFT else 1.0,
            run_same_grid_handoff=spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID,
        )
        if spatial_transfer_control is not None:
            if spatial_transfer_control.calls != 1:
                raise RuntimeError("diagnostic spatial transfer control did not execute exactly once")
            if transfer_metrics.get("model_name") != spatial_transfer_control.model_name:
                raise RuntimeError("diagnostic spatial transfer control lost its runtime identity")
            binding.metrics.increment("partitioned_handoff_spatial_control_calls")
            binding.metrics.event(
                "partitioned_handoff_transfer_control",
                mode=handoff_transfer_control,
                spatial_stage_control=spatial_stage_control,
                operator=spatial_transfer_control.model_name,
                source_clean_sha256=source_clean_sha256,
                source_hw=tuple(int(value) for value in source_shapes[0][-2:]),
                target_hw=(int(target_h), int(target_w)),
                configured_progressive_source_hw=(configured_source_h, configured_source_w),
                temporal_length=int(source_shapes[0][2]),
                spatial_control_calls=int(spatial_transfer_control.calls),
                actual_learned_checkpoint_provider_calls=0,
                clean_video_postprocess_preserved=clean_video_postprocess is not None,
                handoff_noise_policy=handoff_noise_mode,
                authoritative_prefix_restore_preserved=True,
                target_high_preserved=True,
                diagnostic_only=True,
                production_default_changed=False,
                extra_h3_nfe=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
            )
        handoff_noise_report = transfer_metrics.get("handoff_noise")
        if not isinstance(handoff_noise_report, dict):
            raise RuntimeError("partitioned handoff did not report its target-grid noise contract")
        if handoff_noise_report.get("policy") != handoff_noise_mode:
            raise RuntimeError("partitioned handoff noise policy drifted from the selected contract")
        binding.metrics.event(
            "partitioned_handoff_noise",
            frame_gauge_repair_enabled=bool(config.frame_gauge_repair),
            **handoff_noise_report,
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
                video_transfer_audio_mutation=False,
                handoff_transfer_control=handoff_transfer_control,
            )
            if not audio_copy_exact:
                raise RuntimeError("video handoff mutated the carried H3 audio sampler state")

        diagnostic_seed = int(seed or 0) + config.seed_offset
        exact_prefix = stage_plan.prefix.to(
            device=target_video.device,
            dtype=target_video.dtype,
        )
        frame_gauge_accepted = frame_gauge_transaction.get("result") == "accepted"
        frame_gauge_candidate_accepted = bool(
            frame_gauge_accepted or frame_gauge_transaction.get("candidate_accepted") is True
        )
        exact_overlap_fallback_requested, exact_overlap_fallback_trigger = (
            _partitioned_exact_overlap_fallback_eligibility(frame_gauge_transaction)
            if config.frame_gauge_repair and not frame_gauge_accepted
            else (
                False,
                "frame_gauge_selected" if frame_gauge_accepted else "frame_gauge_repair_disabled",
            )
        )
        if target_band is not None and (frame_gauge_accepted or exact_overlap_fallback_requested):
            raise RuntimeError("progressive_target_band must not select a prefix-boundary frame-gauge splice")
        representation_metrics = disabled_suffix_representation_bridge_metrics(
            prefix_t=stage_plan.prefix_t,
            requested=False,
        )
        splice_started = time.perf_counter()
        aligned_witness = None
        provider_native_clean: torch.Tensor | None = None
        provider_boundary_stabilization_receipt: dict[str, Any] = {
            "requested": provider_boundary_stabilization != PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF,
            "mode": provider_boundary_stabilization,
            "applied": False,
            "reason": (
                "off"
                if provider_boundary_stabilization == PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF
                else "not_evaluated"
            ),
            "authoritative_prefix_modified": False,
            "later_suffix_extrapolated": False,
            "corrected_tokens": 0,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
        }
        if frame_gauge_accepted:
            required_witnesses = {
                "learned_native",
                "paired_prefix_aligned_witness",
                "corrected_clean",
            }
            if not required_witnesses.issubset(frame_gauge_witnesses):
                raise RuntimeError("accepted frame-gauge transaction lost required clean-domain witnesses")
            learned_clean = frame_gauge_witnesses["learned_native"]
            provider_native_clean = learned_clean
            if provider_boundary_stabilization != PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF:
                provider_boundary_stabilization_receipt["reason"] = "frame_gauge_selected"
            aligned_witness = frame_gauge_witnesses["paired_prefix_aligned_witness"]
            corrected_clean = frame_gauge_witnesses["corrected_clean"]
            if not suffix_dc_bridge_enabled:
                raise RuntimeError("the accepted frame-gauge arm requires the partitioned suffix DC bridge")
            dc_metrics = frame_gauge_transaction.get("dc_metrics")
            if not isinstance(dc_metrics, dict):
                raise RuntimeError("accepted frame-gauge transaction lost the existing DC bridge receipt")
            dc_metrics = dict(dc_metrics)
            splice_recovery = "actual_provider_clean_postprocess"
            splice_clean_source = "actual_provider"
        else:
            if pending_registered_reference is not None:
                raise RuntimeError("rejected frame-gauge transaction published a guidance reference")
            learned_clean, splice_clean_source, splice_recovery = _resolve_partitioned_transfer_clean(
                target_video,
                actual_handoff_clean,
                handoff_noise_mode=handoff_noise_mode,
                sigma=sigma,
                seed=diagnostic_seed,
            )
            provider_native_clean = learned_clean
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
                provider_native_clean = learned_clean

                if provider_boundary_stabilization == PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT:
                    # Hardware validation disproved the pre-high-only promotion gate: the same
                    # provider/shadow state improved before target-high, then target-high
                    # amplified the selected tile and decoded-media validation regressed.
                    # Preserve the serialized selector for old workflows but fail closed.
                    # The same requested run now emits a post-high shadow below instead.
                    provider_boundary_stabilization_receipt.update(
                        reason="disabled_pending_post_high_validation",
                        exact_overlap_fallback_required=True,
                        historical_candidate_policy="partitioned_provider_boundary_soft_support_production_v1",
                        historical_candidate_mutation_disabled=True,
                        production_mutation_allowed=False,
                    )

                target_video, corrected_clean, representation_metrics, dc_metrics = (
                    _apply_partitioned_exact_overlap_bridge(
                        target_video,
                        learned_clean,
                        exact_prefix,
                        sigma=sigma,
                        weights=PARTITIONED_EXACT_OVERLAP_PRODUCTION_WEIGHTS,
                        dc_enabled=suffix_dc_bridge_enabled,
                    )
                )
            elif target_band is not None:
                if provider_boundary_stabilization != PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF:
                    provider_boundary_stabilization_receipt["reason"] = "target_band_identity_boundary"
                target_video, corrected_clean, dc_metrics = _apply_target_band_head_dc_bridge(
                    target_video,
                    learned_clean,
                    band_provider_native_clean,
                    exact_prefix,
                    target_band,
                    sigma=sigma,
                    enabled=suffix_dc_bridge_enabled,
                )
                band_provider_native_clean = None
            else:
                if provider_boundary_stabilization != PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF:
                    provider_boundary_stabilization_receipt["reason"] = "exact_overlap_fallback_not_selected"
                target_video, corrected_clean, dc_metrics = _apply_partitioned_suffix_dc_bridge(
                    target_video,
                    learned_clean,
                    exact_prefix,
                    sigma=sigma,
                    enabled=suffix_dc_bridge_enabled,
                )
            if exact_overlap_fallback_requested:
                if splice_clean_source == "inverse_recovered":
                    splice_recovery = "inverse_conditional_renoise_with_actual_provider_boundary_pair"
                    splice_clean_source = "actual_provider_boundary_pair_plus_inverse_recovered"
                else:
                    splice_recovery = "actual_clean_postprocess_with_verified_boundary_pair"

        if provider_native_clean is None:
            raise RuntimeError("partitioned provider-native clean witness was not established")
        exact_overlap_policy = PARTITIONED_EXACT_OVERLAP_POLICY
        binding.metrics.event(
            "partitioned_provider_boundary_stabilization",
            exact_overlap_fallback_requested=bool(exact_overlap_fallback_requested),
            exact_overlap_fallback_trigger=str(exact_overlap_fallback_trigger),
            **provider_boundary_stabilization_receipt,
        )

        if isinstance(effective_upscaler, H3PatchLatticeTransferProvider):
            binding.metrics.event(
                "partitioned_transfer_lattice",
                policy=H3_TRANSFER_LATTICE,
                prefix_projection_policy=H3_TRANSFER_LATTICE,
                source_hw=(source_h, source_w),
                target_hw=(target_h, target_w),
                resample_position="encoder_to_decoder",
                provider_calls=effective_upscaler.calls,
                transferred_prefix_output_discarded=True,
                exact_prefix_modified=False,
                extra_h3_nfe=0,
                extra_provider_calls=0,
            )
        if residual_mode == "measure" and target_band is None:
            binding.metrics.event(
                "partitioned_same_frame_prefix_affine",
                **measure_paired_prefix_affine(learned_clean, exact_prefix, prefix_t=stage_plan.prefix_t),
            )

        splice_diagnostics = measure_exact_prefix_splice(
            learned_clean,
            exact_prefix,
            corrected_clean_video=corrected_clean,
        )
        splice_diagnostics.update(
            splice_diagnostic_elapsed_ms=(time.perf_counter() - splice_started) * 1000.0,
            splice_recovery=splice_recovery,
            splice_clean_source=splice_clean_source,
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
        exact_overlap_applied = bool(
            exact_overlap_fallback_requested
            and (
                representation_metrics.get("suffix_representation_bridge_accepted", False)
                or float(dc_metrics.get("suffix_dc_bridge_delta_rms", 0.0)) > 0.0
            )
        )
        exact_overlap_corrected_tokens = (
            max(
                int(representation_metrics.get("suffix_representation_bridge_corrected_tokens", 0)),
                int(dc_metrics.get("suffix_dc_bridge_corrected_tokens", 0)),
            )
            if exact_overlap_applied
            else 0
        )
        exact_overlap_bounded_successor_support = exact_overlap_corrected_tokens > 1
        binding.metrics.event(
            "partitioned_frame_gauge",
            mode="on" if config.frame_gauge_repair else "off",
            enabled=bool(config.frame_gauge_repair),
            eligible=bool(
                config.frame_gauge_repair
                and frame_gauge_transaction.get("result") in {"accepted", "identity", "rejected", "shadow_only"}
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
            candidate_accepted=bool(frame_gauge_transaction.get("candidate_accepted", False)),
            candidate_spatial_warp_computed=bool(frame_gauge_transaction.get("candidate_spatial_warp_computed", False)),
            candidate_guidance_reference_computed=bool(
                frame_gauge_transaction.get("candidate_guidance_reference_computed", False)
            ),
            production_mutation_allowed=bool(frame_gauge_transaction.get("production_mutation_allowed", False)),
            hardware_invalidation=(
                "00687_visible_frame_shift_after_applied_rigid_v4"
                if frame_gauge_transaction.get("result") == "shadow_only"
                else None
            ),
            dc_bridge_applied=bool(dc_metrics.get("suffix_dc_bridge_enabled", False)),
            suffix_dc_bridge_requested=suffix_dc_bridge_enabled,
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
            exact_overlap_fallback_applied=exact_overlap_applied,
            exact_overlap_fallback_policy=exact_overlap_policy,
            exact_overlap_fallback_source=(
                "actual_provider_boundary_pair" if exact_overlap_fallback_requested else "not_used"
            ),
            exact_overlap_fallback_transformed_states=(
                [f"learned_suffix_{offset}" for offset in range(exact_overlap_corrected_tokens)]
                if exact_overlap_applied
                else []
            ),
        )

        if suffix_dc_bridge_enabled:
            exact_overlap_dc_weights = (
                PARTITIONED_EXACT_OVERLAP_PRODUCTION_WEIGHTS if exact_overlap_fallback_requested else (1.0,)
            )
            exact_overlap_dc_support_policy = (
                "bounded_linear_return_v2" if len(exact_overlap_dc_weights) > 1 else "first_suffix_only_v1"
            )
        else:
            exact_overlap_dc_weights = ()
            exact_overlap_dc_support_policy = "disabled"
        binding.metrics.event(
            "partitioned_exact_overlap_bridge",
            policy=exact_overlap_policy,
            requested=bool(exact_overlap_fallback_requested),
            trigger=str(exact_overlap_fallback_trigger),
            applied=exact_overlap_applied,
            state_mapping=("conditional_renoise_affine" if exact_overlap_applied else "disabled_or_noop"),
            authoritative_prefix_modified=False,
            later_suffix_extrapolated=exact_overlap_bounded_successor_support,
            suffix_support_policy=(
                "bounded_linear_return_v2" if exact_overlap_bounded_successor_support else "first_suffix_only_v1"
            ),
            suffix_support_tokens=exact_overlap_corrected_tokens,
            dc_support_policy=exact_overlap_dc_support_policy,
            dc_support_tokens=int(dc_metrics.get("suffix_dc_bridge_corrected_tokens", 0)),
            dc_temporal_weights=list(exact_overlap_dc_weights),
            suffix_outside_support_modified=False,
            structural_bridge_retired=True,
            structural_bridge_hardware_verdict="rendered_continuity_unqualified",
            structural_support_tokens=0,
            production_correction="one_token_per_channel_spatial_mean_only",
            extra_h3_nfe=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            **representation_metrics,
        )

        restored_clean = corrected_clean.clone()
        restored_clean[:, :, : stage_plan.prefix_t] = exact_prefix.to(restored_clean)

        if residual_mode == "measure" and target_band is None:
            try:
                boundary_window_evidence = BoundaryWindowEvidence(exact_prefix, int(target_video.shape[2]))
            except ValueError as exc:
                binding.metrics.event(
                    "partitioned_boundary_window_evidence",
                    policy="native_boundary_decoder_window_evidence_v1",
                    status="unsupported",
                    reason=str(exc),
                    output_mutated=False,
                    diagnostic_only=True,
                )
            else:
                boundary_window_evidence.capture("provider_native_clean", provider_native_clean)
                boundary_window_evidence.capture("pre_high_exact_restored", restored_clean)
        elif boundary_window_evidence is not None:
            boundary_window_evidence.capture("pre_high_exact_restored", restored_clean)

        high_video_reference_enabled = PARTITIONED_HIGH_VIDEO_REFERENCE_ENABLED
        high_audio_reference_enabled = PARTITIONED_HIGH_AUDIO_REFERENCE_ENABLED
        high_video_reference_suffix = None
        high_audio_reference = None

        bridge_support_tokens = len(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS)
        safe_high_prediction_bridge_topology = bool(
            config.frame_gauge_repair
            and prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT
            and av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_MAIN
            and guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
            and handoff_noise_mode in {H3_HANDOFF_NOISE_SOURCE_RESIDUAL, H3_HANDOFF_NOISE_DENSE_DRIFT}
            and representation_metrics.get("suffix_representation_bridge_accepted", False)
            and exact_overlap_corrected_tokens == 1
            and not high_video_reference_enabled
            and stage_plan.prefix_t + bridge_support_tokens <= int(target_video.shape[2])
        )
        bridge_mask_fully_generated = False
        if denoise_mask is not None and safe_high_prediction_bridge_topology:
            candidate_video_mask, _candidate_audio_mask = unpack_streams(
                denoise_mask,
                target_shapes,
            )
            bridge_mask_fully_generated = bool(
                torch.all(
                    candidate_video_mask[
                        :,
                        :,
                        stage_plan.prefix_t : stage_plan.prefix_t + bridge_support_tokens,
                    ]
                    == 1
                ).item()
            )
            del candidate_video_mask, _candidate_audio_mask
        # 00715 proved that the per-call model-prefix value residual is tiny
        # (~1.4e-3 RMS) and that exact first-transition rebasing does not remove
        # the decoded frame shift. Keep the machinery for historical evidence,
        # but do not mutate production predictions with the falsified actuator.
        high_prediction_gauge_bridge_enabled = False
        high_prediction_gauge_bridge_candidate_eligible = bool(
            safe_high_prediction_bridge_topology and bridge_mask_fully_generated
        )

        safe_vae_window_repair_topology = bool(
            config.frame_gauge_repair
            and prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT
            and av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_MAIN
            and guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
            and handoff_noise_mode in {H3_HANDOFF_NOISE_SOURCE_RESIDUAL, H3_HANDOFF_NOISE_DENSE_DRIFT}
            and representation_metrics.get("suffix_representation_bridge_accepted", False)
            and exact_overlap_corrected_tokens == 1
            and not high_video_reference_enabled
        )
        vae_window_plan = None
        vae_window_plan_reason = "safe_video_topology_not_active"
        if safe_vae_window_repair_topology:
            try:
                vae_window_plan = h3_vae_boundary_window(
                    stage_plan.prefix_t,
                    int(target_video.shape[2]),
                )
                vae_window_plan_reason = "native_boundary_window_resolved"
            except ValueError as exc:
                vae_window_plan_reason = f"unsupported_native_vae_phase:{exc}"
        # 00716 did not satisfy the cross-ROI gate and the rendered frame
        # shift remained. Retain the native-window plan as diagnostic provenance,
        # but do not permit this latent actuator to become production-active on
        # another resolution/scene while decoded-space localization is under test.
        vae_window_repair_candidate_eligible = vae_window_plan is not None
        vae_window_repair_armed = False
        if vae_window_repair_candidate_eligible:
            vae_window_plan_reason = "hardware_falsified_00716_candidate_disabled"
        high_boundary_context = exact_prefix
        high_boundary_prefix_witness = "authoritative_exact_tail_after_inpaint_restore"

        binding.metrics.event(
            "partitioned_high_boundary_reference_plan",
            policy=HIGH_BOUNDARY_REFERENCE_POLICY,
            enabled=bool(high_video_reference_enabled or high_audio_reference_enabled),
            video_enabled=high_video_reference_enabled,
            video_selection_reason="model_native_prediction_gauge_bridge_no_fixed_clean_reference",
            video_boundary_repair_contract=PARTITIONED_VIDEO_BOUNDARY_REPAIR_CONTRACT,
            high_prediction_gauge_bridge_policy=HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
            high_prediction_gauge_bridge_enabled=high_prediction_gauge_bridge_enabled,
            high_prediction_gauge_bridge_candidate_eligible=high_prediction_gauge_bridge_candidate_eligible,
            high_prediction_gauge_bridge_hardware_verdict="falsified_00715_no_decoded_frame_shift_improvement",
            vae_window_video_policy=VAE_WINDOW_VIDEO_POLICY,
            vae_window_video_repair_armed=vae_window_repair_armed,
            vae_window_video_repair_candidate_eligible=vae_window_repair_candidate_eligible,
            vae_window_video_hardware_verdict="falsified_00716_visible_frame_shift_remained",
            vae_window_video_plan_reason=vae_window_plan_reason,
            vae_window_video_plan=dict(vae_window_plan or {}),
            high_prediction_gauge_bridge_support_tokens=(
                bridge_support_tokens if high_prediction_gauge_bridge_enabled else 0
            ),
            high_prediction_gauge_bridge_weights=(
                list(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS) if high_prediction_gauge_bridge_enabled else []
            ),
            high_prediction_gauge_bridge_mask_fully_generated=bridge_mask_fully_generated,
            high_boundary_context_tokens=int(high_boundary_context.shape[2]),
            high_boundary_prefix_witness=high_boundary_prefix_witness,
            video_reference_domain="disabled",
            video_support_tokens=0,
            video_temporal_weights=[],
            audio_enabled=high_audio_reference_enabled,
            audio_selection_reason="released_sampler_overlap_exact_restore_only",
            audio_boundary_repair_contract=PARTITIONED_AUDIO_BOUNDARY_REPAIR_CONTRACT,
            audio_reference_domain="disabled",
            audio_support_ticks=0,
            audio_temporal_weights=[],
            authoritative_prefix_modified=False,
            extra_h3_nfe=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
        )

        if boundary_content_diagnostic_enabled:
            boundary_content_started = time.perf_counter()
            provider_receipt = measure_boundary_content_continuity(
                provider_native_clean,
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
            if residual_mode == "measure" and frame_gauge_candidate_accepted:
                if bicubic_transfer_shadow is None:
                    raise RuntimeError("residual measurement lost the bicubic transfer shadow")
                actual_provider_witness = frame_gauge_witnesses.get("learned_native")
                if actual_provider_witness is None:
                    raise RuntimeError("learned-transfer residual diagnostic lost the actual provider clean witness")
                shadow_witness = bicubic_transfer_shadow[:, :, : int(actual_provider_witness.shape[2])]
                if tuple(shadow_witness.shape) != tuple(actual_provider_witness.shape):
                    raise RuntimeError("learned-transfer residual witness geometry drifted")
                learned_residual_started = time.perf_counter()
                learned_residual_receipt = measure_learned_transfer_residual_diagnostic(
                    actual_provider_witness,
                    shadow_witness,
                    stage_plan.prefix_t,
                )
                binding.metrics.event(
                    "partitioned_learned_transfer_residual",
                    domain="model_internal_clean",
                    owner_before="source_low_exact_context_bicubic_shadow",
                    owner_after="actual_provider_clean_postprocess",
                    provenance="actual_provider_clean_minus_same_source_bicubic_shadow",
                    elapsed_ms=(time.perf_counter() - learned_residual_started) * 1000.0,
                    extra_h3_nfe=0,
                    extra_sampler_lifetimes=0,
                    extra_history_boundaries=0,
                    extra_provider_calls=0,
                    extra_vae_calls=0,
                    **learned_residual_receipt,
                )
                del shadow_witness
                del bicubic_transfer_shadow
                bicubic_transfer_shadow = None

            predictor_started = time.perf_counter()
            provider_predictor_receipt = measure_provider_boundary_temporal_predictor(
                provider_native_clean,
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
                provider_native_clean,
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
            stabilization_shadow_started = time.perf_counter()
            provider_stabilization_shadow_receipt = measure_provider_boundary_stabilization_shadow(
                provider_native_clean,
                stage_plan.prefix_t,
                calibration_receipt=provider_calibration_receipt,
                provider_content_receipt=provider_receipt,
            )
            binding.metrics.event(
                "partitioned_provider_boundary_stabilization_shadow",
                domain="model_internal_clean",
                owner_before="learned_provider_prefix",
                owner_after="learned_provider_suffix",
                elapsed_ms=(time.perf_counter() - stabilization_shadow_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **provider_stabilization_shadow_receipt,
            )
            soft_support_shadow_started = time.perf_counter()
            provider_soft_support_shadow_receipt = measure_provider_boundary_soft_support_shadow(
                provider_native_clean,
                stage_plan.prefix_t,
                calibration_receipt=provider_calibration_receipt,
                provider_content_receipt=provider_receipt,
                hard_shadow_receipt=provider_stabilization_shadow_receipt,
            )
            binding.metrics.event(
                "partitioned_provider_boundary_soft_support_shadow",
                domain="model_internal_clean",
                owner_before="learned_provider_prefix",
                owner_after="learned_provider_suffix",
                elapsed_ms=(time.perf_counter() - soft_support_shadow_started) * 1000.0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                **provider_soft_support_shadow_receipt,
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

        if residual_mode == "measure" and frame_gauge_candidate_accepted:
            measurement_learned_native = frame_gauge_witnesses.get("learned_native")
            if measurement_learned_native is None:
                raise RuntimeError("residual measurement lost the actual provider clean witness")
            if aligned_witness is None:
                aligned_witness = frame_gauge_witnesses.get("paired_prefix_aligned_witness")
            if aligned_witness is None:
                raise RuntimeError("residual measurement lost the rigid shadow witness")
            evidence_start = max(0, stage_plan.prefix_t - 6)
            evidence_stop = min(int(measurement_learned_native.shape[2]), stage_plan.prefix_t + 4)
            residual_evidence_tensors["exact_prefix_last6"] = exact_prefix[
                :, :, evidence_start : stage_plan.prefix_t
            ].detach()
            residual_evidence_tensors["learned_native_prefix_suffix"] = measurement_learned_native[
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
                        video=measurement_learned_native,
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
                        applied_transform=(
                            "paired_prefix_rigid_v2"
                            if frame_gauge_accepted
                            else "shadow_candidate_paired_prefix_rigid_v2"
                        ),
                        provenance="actual_provider_clean_postprocess_shadow_candidate",
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
                        applied_transform=(
                            "paired_prefix_rigid_v2_then_one_token_dc"
                            if frame_gauge_accepted
                            else "production_baseline_then_one_token_dc"
                        ),
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

        pre_high_vae_window_trajectories: dict[str, dict[str, Any]] = {}

        provider_native_trajectory_source = (
            frame_gauge_witnesses.get("learned_native") if frame_gauge_candidate_accepted else provider_native_clean
        )
        if provider_native_trajectory_source is None:
            provider_native_trajectory_source = provider_native_clean

        for roi_name, roi_fraction in (
            ("upper45", 0.45),
            ("full", 1.0),
        ):
            native_trajectory = measure_translation_trajectory(
                provider_native_trajectory_source,
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
        if target_band is not None:
            _emit_target_band_tail_boundary(
                binding.metrics,
                restored_clean,
                target_band,
                stage="pre_high",
                measure_trajectory=measure_band_trajectory,
            )
        binding.metrics.increment("partitioned_splice_diagnostic_runs")
        binding.metrics.increment("partitioned_multiframe_trajectory_runs")

        binding.metrics.event(
            "partitioned_video_vae_boundary_window_plan",
            policy=VAE_WINDOW_VIDEO_POLICY,
            armed=vae_window_repair_armed,
            reason=vae_window_plan_reason,
            safe_video_topology=safe_vae_window_repair_topology,
            high_prediction_gauge_bridge_enabled=high_prediction_gauge_bridge_enabled,
            high_prediction_gauge_bridge_candidate_eligible=high_prediction_gauge_bridge_candidate_eligible,
            plan=dict(vae_window_plan or {}),
            authoritative_prefix_modified=False,
            audio_modified=False,
            extra_h3_nfe=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
        )
        if vae_window_repair_armed:
            pre_high_vae_window_trajectories = measure_vae_window_video_trajectory(
                restored_clean,
                stage_plan.prefix_t,
            )
            for roi_name, trajectory in pre_high_vae_window_trajectories.items():
                binding.metrics.event(
                    "partitioned_video_vae_boundary_window_trajectory",
                    policy=VAE_WINDOW_VIDEO_POLICY,
                    stage="pre_high_exact_restored",
                    domain="model_internal_clean",
                    roi=roi_name,
                    diagnostic_only=True,
                    **trajectory,
                )

        target_video[:, :, : stage_plan.prefix_t] = stage_plan.prefix.to(target_video)
        band_handoff_receipt = None
        if target_band is not None:
            # Default: the band enters the high stage as its clean prediction
            # re-noised with the same independent noise as the tail. With
            # carry_raw_band the band resumes from its actual low/probe sampler
            # state at the handoff sigma, as same-grid control resumes every
            # generated token. Only the band frames are replaced; the protected
            # prefix, the tail entry and audio are the values assembled above.
            raw_carried = band_raw_video is not None
            if raw_carried:
                if tuple(band_raw_video.shape) != tuple(
                    target_video[:, :, stage_plan.prefix_t : target_band.head_t].shape
                ):
                    raise RuntimeError("target-band raw carry geometry drifted from the high-stage band")
                target_video[:, :, stage_plan.prefix_t : target_band.head_t] = band_raw_video.to(target_video)
            band_handoff_receipt = {
                "target_band_tokens": int(target_band.band_t),
                "target_band_handoff_policy": (
                    TARGET_BAND_RAW_CARRY_HANDOFF_POLICY if raw_carried else TARGET_BAND_HANDOFF_POLICY
                ),
                "target_band_handoff_state": target_band_handoff_state,
                "target_band_raw_state_carried": raw_carried,
                "target_band_raw_state_identity": (
                    bool(
                        torch.equal(
                            target_video[:, :, stage_plan.prefix_t : target_band.head_t],
                            band_raw_video.to(target_video),
                        )
                    )
                    if raw_carried
                    else None
                ),
                "target_band_transfer_start_t": int(target_band.head_t),
            }
            if raw_carried and not band_handoff_receipt["target_band_raw_state_identity"]:
                raise RuntimeError("target-band raw carry did not preserve the low/probe band state")
            band_raw_video = None
        target_raw = pack_streams((target_video, target_audio))[0]
        binding.metrics.event(
            "partitioned_transfer",
            handoff_transfer_control=handoff_transfer_control,
            spatial_stage_control=spatial_stage_control,
            learned_transfer_performed=(
                handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_LEARNED
                and spatial_stage_control in _LEARNED_TRANSFER_SPATIAL_STAGES
            ),
            spatial_transfer_control_applied=spatial_transfer_control is not None,
            same_grid_identity_transfer_applied=(spatial_stage_control == PARTITIONED_SPATIAL_STAGE_SAME_GRID),
            actual_learned_checkpoint_provider_invoked=(
                handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_LEARNED
                and spatial_stage_control in _LEARNED_TRANSFER_SPATIAL_STAGES
            ),
            upscaler_prefix_context_used=(
                handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_LEARNED
                and spatial_stage_control in _LEARNED_TRANSFER_SPATIAL_STAGES
            ),
            **(band_handoff_receipt or {}),
            transferred_prefix_output_discarded=True,
            authoritative_target_prefix_restored=True,
            target_prefix_resized_for_transformer=False,
            deprecated_mixed_grid_repairs_applied=False,
            frame_gauge_repair_enabled=bool(config.frame_gauge_repair),
            frame_gauge_result=str(frame_gauge_transaction.get("result", "baseline")),
            frame_gauge_reason=str(frame_gauge_transaction.get("reason", "unknown")),
            suffix_dc_bridge_requested=suffix_dc_bridge_enabled,
            suffix_dc_bridge_state_mapping=(
                "disabled"
                if not suffix_dc_bridge_enabled
                else "pre_renoise_clean_operand"
                if frame_gauge_accepted
                else "conditional_renoise_affine"
            ),
            suffix_dc_bridge_policy=(
                "disabled"
                if not suffix_dc_bridge_enabled
                else "successor_safe_linear_v2"
                if int(dc_metrics.get("suffix_dc_bridge_corrected_tokens", 0)) > 1
                else "one_token_spatial_mean_v1"
            ),
            provider_boundary_stabilization=dict(provider_boundary_stabilization_receipt),
            video_boundary_repair_contract=PARTITIONED_VIDEO_BOUNDARY_REPAIR_CONTRACT,
            video_high_clean_reference_enabled=high_video_reference_enabled,
            video_high_prediction_gauge_bridge_policy=HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
            video_high_prediction_gauge_bridge_enabled=high_prediction_gauge_bridge_enabled,
            video_high_prediction_gauge_bridge_support_tokens=(
                bridge_support_tokens if high_prediction_gauge_bridge_enabled else 0
            ),
            audio_boundary_repair_contract=PARTITIONED_AUDIO_BOUNDARY_REPAIR_CONTRACT,
            audio_high_clean_reference_enabled=high_audio_reference_enabled,
            audio_exact_restore_successor_bridge_enabled=False,
            partitioned_exact_overlap_bridge={
                "policy": exact_overlap_policy,
                "requested": bool(exact_overlap_fallback_requested),
                "trigger": str(exact_overlap_fallback_trigger),
                "applied": exact_overlap_applied,
                "state_mapping": ("conditional_renoise_affine" if exact_overlap_applied else "disabled_or_noop"),
                "source": ("actual_provider_boundary_pair" if exact_overlap_fallback_requested else "not_used"),
                "authoritative_prefix_modified": False,
                "later_suffix_extrapolated": exact_overlap_bounded_successor_support,
                "suffix_support_policy": (
                    "bounded_linear_return_v2" if exact_overlap_bounded_successor_support else "first_suffix_only_v1"
                ),
                "suffix_support_tokens": exact_overlap_corrected_tokens,
                "dc_support_policy": exact_overlap_dc_support_policy,
                "dc_support_tokens": int(dc_metrics.get("suffix_dc_bridge_corrected_tokens", 0)),
                "dc_temporal_weights": list(exact_overlap_dc_weights),
                "suffix_outside_support_modified": False,
                "structural_bridge_retired": True,
                "structural_bridge_hardware_verdict": "rendered_continuity_unqualified",
                "structural_support_tokens": 0,
                "production_correction": "one_token_per_channel_spatial_mean_only",
                **representation_metrics,
            },
            **dc_metrics,
            provider_api_version=transfer_metrics.get("provider_api_version"),
            provider_kind=transfer_metrics.get("provider_kind"),
            model_name=transfer_metrics.get("model_name"),
            source_hw=transfer_metrics.get("source_hw"),
            target_hw=transfer_metrics.get("target_hw"),
            temporal_length=transfer_metrics.get("temporal_length"),
            transfer_operator_elapsed_ms=transfer_metrics.get("learned_upscale_elapsed_ms"),
            learned_upscale_elapsed_ms=(
                transfer_metrics.get("learned_upscale_elapsed_ms")
                if handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_LEARNED
                and spatial_stage_control in _LEARNED_TRANSFER_SPATIAL_STAGES
                else None
            ),
            clean_video_postprocess=transfer_metrics.get("clean_video_postprocess"),
            handoff_noise=transfer_metrics.get("handoff_noise"),
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
            high_prediction_gauge_bridge_enabled=high_prediction_gauge_bridge_enabled,
            high_prediction_gauge_bridge_policy=HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
            high_prediction_gauge_bridge_support_tokens=(
                bridge_support_tokens if high_prediction_gauge_bridge_enabled else 0
            ),
            sampler_latent_mask_unchanged=True,
            sampler_entry_state_unchanged=True,
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

        if (
            guidance_run is not None
            and binding.guidance.mode in {"direction", "direction+temporal", "direction+acceleration"}
            and binding.registered_guidance_reference is None
            and guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
            and av_handoff_source == PARTITIONED_AV_HANDOFF_SOURCE_MAIN
            and handoff_transfer_control == PARTITIONED_HANDOFF_TRANSFER_LEARNED
            and spatial_stage_control in _LEARNED_TRANSFER_SPATIAL_STAGES
            and splice_clean_source in {"actual_clean_postprocess", "actual_provider"}
        ):
            # Under target-band control the high stage starts from the band's own
            # prediction and the learned tail (with its head-boundary DC correction), so
            # guidance binds to that actual clean operand rather than to the
            # provider's own rendering of the band.
            handoff_guidance_reference = _prepare_handoff_guidance_reference(
                run=guidance_run,
                provider_input=clean_video,
                provider_output=corrected_clean if target_band is not None else provider_native_clean,
                prefix_t=stage_plan.prefix_t,
                split_coordinate=split_coordinate,
                high_sigmas=high_sigmas,
                video_shift=video_shift,
            )
            source_retained_bytes = (
                handoff_guidance_reference.source_video.numel() * handoff_guidance_reference.source_video.element_size()
            )
            target_retained_bytes = (
                handoff_guidance_reference.video.numel() * handoff_guidance_reference.video.element_size()
            )
            binding.metrics.event(
                "partitioned_handoff_guidance_reference",
                run_id=guidance_run.run_id,
                reference_coordinate=split_coordinate,
                source="actual_learned_provider_pair",
                source_shape=tuple(handoff_guidance_reference.source_video.shape),
                target_shape=tuple(provider_native_clean.shape),
                source_retained_bytes=source_retained_bytes,
                target_retained_bytes=target_retained_bytes,
                retained_bytes=source_retained_bytes + target_retained_bytes,
                captured_source_prefix_delta_rms=handoff_guidance_reference.captured_source_prefix_delta_rms,
                captured_source_prefix_delta_abs_max=handoff_guidance_reference.captured_source_prefix_delta_abs_max,
                captured_source_suffix_delta_rms=handoff_guidance_reference.captured_source_suffix_delta_rms,
                captured_source_suffix_delta_abs_max=handoff_guidance_reference.captured_source_suffix_delta_abs_max,
                native_temporal_correspondence_source="actual_provider_input",
                native_temporal_correspondence_preserved=True,
                extra_provider_calls=0,
                extra_h3_nfe=0,
                extra_sampler_lifetimes=0,
            )

        # Release registration witnesses. Only the executed learned source/target
        # guidance pair remains owned for high; it is released on success or failure.
        del restored_clean
        del corrected_clean
        del learned_clean
        del provider_native_clean
        del provider_native_trajectory_source
        if residual_mode == "measure" and frame_gauge_candidate_accepted:
            del measurement_learned_native
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
        high_boundary_reference_calls_before = int(
            binding.metrics.counters.get("high_boundary_reference_anchor_calls", 0)
        )
        high_prediction_bridge_calls_before = int(binding.metrics.counters.get("high_prediction_gauge_bridge_calls", 0))
        high_denoise_mask = _partitioned_high_video_overlap_mask(
            denoise_mask,
            exact_denoise_mask,
            target_shapes,
            model_options,
            binding.metrics,
            prefix_t=stage_plan.prefix_t,
        )
        if boundary_window_evidence is not None:
            high_video_mask, _ = unpack_streams(high_denoise_mask, target_shapes)
            boundary_window_evidence.capture("initial_high_video_mask", high_video_mask)
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with (
            _flow_stage_contract(guider, "high"),
            _partitioned_high_stage_contract(
                guider,
                stage_plan,
                binding.metrics,
                exact_prefix_attention=prefix_transformer_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
                attention_head_t=target_band.head_t if target_band is not None else None,
            ),
            high_boundary_contract(
                binding,
                high_boundary_context,
                target_shapes,
                measure=residual_mode == "measure" or band_witness_requested,
                video_reference_suffix=high_video_reference_suffix,
                audio_reference=high_audio_reference,
                exact_denoise_mask=(diagnostic_target_mask if high_audio_reference is not None else None),
                prefix_witness=high_boundary_prefix_witness,
                prediction_gauge_bridge_weights=(
                    HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS if high_prediction_gauge_bridge_enabled else None
                ),
                # Guidance must use the same first-token channel-mean correction
                # as the handoff state, without a spatial residual transplant.
                guidance_reference_gauge_weights=None,
                guidance_reference_dc_metrics=dc_metrics,
                window_evidence=boundary_window_evidence,
                handoff_guidance_reference=handoff_guidance_reference,
            ),
        ):
            result = executor(
                target_noise,
                latent_image,
                sampler,
                high_sigmas,
                high_denoise_mask,
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
        high_boundary_reference_calls = (
            int(binding.metrics.counters.get("high_boundary_reference_anchor_calls", 0))
            - high_boundary_reference_calls_before
        )
        high_prediction_bridge_calls = (
            int(binding.metrics.counters.get("high_prediction_gauge_bridge_calls", 0))
            - high_prediction_bridge_calls_before
        )
        high_boundary_reference_expected = bool(high_video_reference_enabled or high_audio_reference_enabled)
        if high_boundary_reference_expected and high_boundary_reference_calls != len(high_model_calls):
            raise RuntimeError("target-high clean boundary reference did not cover every high-stage model prediction")
        if not high_boundary_reference_expected and high_boundary_reference_calls != 0:
            raise RuntimeError("target-high clean boundary reference executed outside its selected plan")
        if high_prediction_gauge_bridge_enabled and high_prediction_bridge_calls != len(high_model_calls):
            raise RuntimeError("target-high prediction gauge bridge did not cover every high-stage model prediction")
        if not high_prediction_gauge_bridge_enabled and high_prediction_bridge_calls != 0:
            raise RuntimeError("target-high prediction gauge bridge executed outside its selected plan")
        binding.metrics.event(
            "partitioned_high_boundary_reference_verified",
            policy=HIGH_BOUNDARY_REFERENCE_POLICY,
            expected=high_boundary_reference_expected,
            model_calls=len(high_model_calls),
            anchor_calls=high_boundary_reference_calls,
            all_model_calls_covered=(
                high_boundary_reference_calls == len(high_model_calls)
                if high_boundary_reference_expected
                else high_boundary_reference_calls == 0
            ),
            first_high_actual=first_high_actual,
            video_enabled=high_video_reference_enabled,
            audio_enabled=high_audio_reference_enabled,
            high_prediction_gauge_bridge_enabled=high_prediction_gauge_bridge_enabled,
            high_prediction_gauge_bridge_policy=HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
            high_prediction_gauge_bridge_calls=high_prediction_bridge_calls,
            high_prediction_gauge_bridge_all_model_calls_covered=(
                high_prediction_bridge_calls == len(high_model_calls)
                if high_prediction_gauge_bridge_enabled
                else high_prediction_bridge_calls == 0
            ),
            high_boundary_context_tokens=int(high_boundary_context.shape[2]),
            high_boundary_prefix_witness=high_boundary_prefix_witness,
            fail_closed=True,
            extra_h3_nfe=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
        )
        if high_prediction_gauge_bridge_enabled:
            binding.metrics.event(
                "partitioned_high_prediction_gauge_bridge_verified",
                policy=HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
                expected=True,
                model_calls=len(high_model_calls),
                bridge_calls=high_prediction_bridge_calls,
                all_model_calls_covered=high_prediction_bridge_calls == len(high_model_calls),
                support_tokens=bridge_support_tokens,
                temporal_weights=list(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS),
                sampler_mask_modified=False,
                sampler_entry_state_modified=False,
                fixed_clean_reference_used=False,
                extra_h3_nfe=0,
                extra_provider_calls=0,
                extra_vae_calls=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
            )

        final_video, final_audio = unpack_streams(result, target_shapes)
        final_internal = None
        final_internal_video = None
        final_internal_audio = None
        if (
            diagnostic_audio_control
            or (residual_mode == "measure" and frame_gauge_candidate_accepted)
            or boundary_window_evidence is not None
            or boundary_content_diagnostic_enabled
            or vae_window_repair_armed
        ):
            final_internal = _process_latent_in(base_model, result, target_shapes)
            final_internal_video, final_internal_audio = unpack_streams(final_internal, target_shapes)
        vae_window_video_repair: dict[str, Any] = {
            "policy": VAE_WINDOW_VIDEO_POLICY,
            "eligible": False,
            "accepted": False,
            "applied": False,
            "output_mutated": False,
            "armed": vae_window_repair_armed,
            "reason": vae_window_plan_reason if not vae_window_repair_armed else "not_evaluated",
            "safe_video_topology": safe_vae_window_repair_topology,
            "authoritative_prefix_modified": False,
            "audio_modified": False,
            "high_prediction_gauge_bridge_enabled": high_prediction_gauge_bridge_enabled,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
        }
        if vae_window_repair_armed:
            if final_internal_video is None or final_internal_audio is None:
                raise RuntimeError("VAE-window video repair lost the common-domain final state")
            if not pre_high_vae_window_trajectories:
                raise RuntimeError("VAE-window video repair lost the pre-high trajectory reference")

            _internal_candidate, repair_receipt = repair_vae_window_vertical_residual(
                pre_high_vae_window_trajectories,
                final_internal_video,
                prefix_t=stage_plan.prefix_t,
            )
            vae_window_video_repair = {
                **repair_receipt,
                "armed": True,
                "safe_video_topology": True,
                "audio_modified": False,
                "high_prediction_gauge_bridge_enabled": False,
                "extra_h3_nfe": 0,
                "extra_sampler_lifetimes": 0,
                "extra_history_boundaries": 0,
                "extra_provider_calls": 0,
                "extra_vae_calls": 0,
            }
            if bool(repair_receipt.get("applied")):
                proposed_dy = float(repair_receipt["selected_dy_cells"])
                corrected_caller_video = apply_vae_window_vertical_translation(
                    final_video,
                    prefix_t=stage_plan.prefix_t,
                    dy=proposed_dy,
                )
                if not torch.equal(
                    corrected_caller_video[:, :, : stage_plan.prefix_t],
                    final_video[:, :, : stage_plan.prefix_t],
                ):
                    raise RuntimeError("VAE-window video repair modified the caller-owned exact prefix")

                corrected_result, corrected_shapes = pack_streams((corrected_caller_video, final_audio))
                if corrected_shapes != target_shapes:
                    raise RuntimeError("VAE-window video repair changed packed stream geometry")
                _corrected_video_check, corrected_audio_check = unpack_streams(corrected_result, target_shapes)
                if not torch.equal(corrected_audio_check, final_audio):
                    raise RuntimeError("VAE-window video repair modified caller audio")

                recomputed_internal = _process_latent_in(base_model, corrected_result, target_shapes)
                recomputed_internal_video, recomputed_internal_audio = unpack_streams(
                    recomputed_internal,
                    target_shapes,
                )
                if not torch.equal(recomputed_internal_audio, final_internal_audio):
                    raise RuntimeError("VAE-window video repair changed internal audio")
                if not torch.equal(
                    recomputed_internal_video[:, :, : stage_plan.prefix_t],
                    exact_prefix.to(device=recomputed_internal_video.device),
                ):
                    raise RuntimeError("VAE-window video repair changed the internal exact prefix")

                canonical_receipts = measure_vae_window_video_trajectory(
                    recomputed_internal_video,
                    stage_plan.prefix_t,
                )
                canonical_accepted, canonical_validation = validate_vae_window_vertical_candidate(
                    pre_high_vae_window_trajectories,
                    repair_receipt["post_high_before"],
                    canonical_receipts,
                    prefix_t=stage_plan.prefix_t,
                    temporal=int(recomputed_internal_video.shape[2]),
                )
                vae_window_video_repair["caller_reentry_validation"] = canonical_validation
                vae_window_video_repair["canonical_candidate_accepted"] = canonical_accepted
                vae_window_video_repair["post_high_after"] = canonical_receipts
                if canonical_accepted:
                    final_video = corrected_caller_video
                    result = corrected_result
                    final_internal = recomputed_internal
                    final_internal_video = recomputed_internal_video
                    final_internal_audio = recomputed_internal_audio
                    vae_window_video_repair.update(
                        caller_audio_exact=True,
                        caller_prefix_exact=True,
                        internal_audio_exact=True,
                        internal_prefix_exact=True,
                    )
                else:
                    vae_window_video_repair.update(
                        applied=False,
                        accepted=False,
                        output_mutated=False,
                        reason="caller_reentry_validation_failed",
                    )

        binding.metrics.event(
            "partitioned_post_high_vae_window_video_repair",
            **vae_window_video_repair,
        )
        for roi_name, trajectory in vae_window_video_repair.get("post_high_before", {}).items():
            binding.metrics.event(
                "partitioned_video_vae_boundary_window_trajectory",
                policy=VAE_WINDOW_VIDEO_POLICY,
                stage="post_high_internal_uncorrected",
                domain="model_internal_clean",
                roi=roi_name,
                diagnostic_only=True,
                **trajectory,
            )
        after_stage = (
            "post_high_internal_corrected"
            if vae_window_video_repair.get("applied", False)
            else "post_high_internal_candidate_rejected"
        )
        for roi_name, trajectory in vae_window_video_repair.get("post_high_after", {}).items():
            binding.metrics.event(
                "partitioned_video_vae_boundary_window_trajectory",
                policy=VAE_WINDOW_VIDEO_POLICY,
                stage=after_stage,
                domain="model_internal_clean",
                roi=roi_name,
                diagnostic_only=not bool(vae_window_video_repair.get("applied", False)),
                **trajectory,
            )

        if boundary_window_evidence is not None:
            boundary_window_evidence.capture("final_post_high_internal_clean", final_internal_video)

        if residual_mode == "measure" and frame_gauge_candidate_accepted:
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
                    applied_transform=(
                        VAE_WINDOW_VIDEO_POLICY
                        if vae_window_video_repair.get("applied", False)
                        else "none_post_high_observation"
                    ),
                    provenance=(
                        "post_high_output_after_vae_window_vertical_plateau_release"
                        if vae_window_video_repair.get("applied", False)
                        else "existing_post_high_output_via_model_latent_input_conversion"
                    ),
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
                    applied_transform=(
                        VAE_WINDOW_VIDEO_POLICY if vae_window_video_repair.get("applied", False) else "none"
                    ),
                    provenance=(
                        "returned_post_high_output_after_vae_window_vertical_plateau_release"
                        if vae_window_video_repair.get("applied", False)
                        else "existing_post_high_output"
                    ),
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

            if provider_boundary_stabilization == PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT:
                post_high_shadow_started = time.perf_counter()
                post_high_shadow = measure_provider_boundary_post_high_shadow(
                    post_high_diagnostic_video,
                    stage_plan.prefix_t,
                    post_high_content_receipt=post_high_receipt,
                )
                binding.metrics.event(
                    "partitioned_provider_boundary_post_high_shadow",
                    mode=provider_boundary_stabilization,
                    domain="model_internal_clean",
                    owner_before="authoritative_exact_prefix",
                    owner_after="post_high_generated_suffix",
                    legacy_pre_high_mutation_disabled=True,
                    extra_h3_nfe=0,
                    extra_sampler_lifetimes=0,
                    extra_history_boundaries=0,
                    extra_provider_calls=0,
                    extra_vae_calls=0,
                    elapsed_ms=(time.perf_counter() - post_high_shadow_started) * 1000.0,
                    **post_high_shadow,
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
                source_sampler_masks_unchanged=True,
                high_stage_guard_mask_local_override=False,
                high_stage_prediction_gauge_bridge_output_only=high_prediction_gauge_bridge_enabled,
                post_high_vae_window_video_output_only=bool(vae_window_video_repair.get("applied", False)),
                high_stage_audio_mask_unchanged=True,
            )
        boundary = measure_video_boundary(final_video, stage_plan.prefix_t)
        if target_band is not None:
            _emit_target_band_tail_boundary(
                binding.metrics,
                final_video,
                target_band,
                stage="final_post_high",
                measure_trajectory=measure_band_trajectory,
            )
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
            if vae_window_repair_armed and vae_window_plan is not None:
                release_boundary_t = int(vae_window_plan["release_frontier_t"])
                if release_boundary_t >= int(final_video.shape[2]):
                    raise RuntimeError("VAE-window video repair has no suffix after its release frontier")
                release_trajectory = measure_translation_trajectory(
                    final_video,
                    release_boundary_t,
                    forward_steps=4,
                    backward_steps=3,
                    roi_fraction=roi_fraction,
                    max_shift=4,
                )
                binding.metrics.event(
                    "partitioned_multiframe_trajectory",
                    stage="final_post_high_vae_window_release",
                    roi=roi_name,
                    vae_window_video_policy=VAE_WINDOW_VIDEO_POLICY,
                    repair_applied=bool(vae_window_video_repair.get("applied", False)),
                    plateau_tokens=int(vae_window_plan["plateau_tokens"]),
                    support_tokens=int(vae_window_plan["support_tokens"]),
                    temporal_weights=list(vae_window_plan["weights"]),
                    original_boundary_t=stage_plan.prefix_t,
                    decoder_window_start_t=int(vae_window_plan["window_start_t"]),
                    decoder_window_stop_t=int(vae_window_plan["window_stop_t"]),
                    release_boundary_t=release_boundary_t,
                    diagnostic_only=True,
                    **release_trajectory,
                )
        if not splice_diagnostics:
            raise RuntimeError(
                "partitioned exact-prefix splice diagnostics were not recorded before high-stage sampling"
            )

        residual_evidence_receipt: dict[str, Any] | None = None
        if residual_mode == "measure" and frame_gauge_candidate_accepted:
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

        if boundary_window_evidence is not None:
            window_evidence_receipt = export_residual_geometry_evidence(
                boundary_window_evidence.tensors,
                session_id=str(session_id),
                chunk_id=str(chunk_id),
                seed=int(seed or 0),
                sigma=float(sigma),
                evidence_kind="h3_flow_native_boundary_decoder_window_evidence",
                metadata={
                    "policy": "native_boundary_decoder_window_evidence_v1",
                    "window": boundary_window_evidence.plan,
                    "domain": "model_internal_clean_except_sampler_input_and_mask",
                    "prediction_prefix": "native_model_prediction_not_recanonicalized",
                    "decoder_comparison_prefix": "replace_with_authoritative_prefix_bytes",
                    "sampler_input_domain": "predict_noise_input_after_sampler_inpaint",
                    "initial_mask_domain": "initial_high_sampler_and_model_video_denoise_mask",
                    "provider_clean_provenance": (
                        "actual_learned_provider_before_target_band_splice"
                        if target_band is not None
                        else splice_clean_source
                    ),
                    **(
                        {
                            "target_band_tokens": int(target_band.band_t),
                            "target_band_transfer_start_t": int(target_band.head_t),
                            "full_video_snapshots": True,
                            "full_video_temporal_start_t": 0,
                            "cpu_byte_budget": int(boundary_window_evidence.max_bytes),
                            "source_probe_clean_grid": [int(source_h), int(source_w)],
                            "low_probe_native_carrier_layout": "target_head_and_top_left_reduced_tail_with_padding",
                            "low_probe_native_carrier_decodable": False,
                        }
                        if target_band is not None
                        else {}
                    ),
                    "registration_result": str(frame_gauge_transaction.get("result")),
                    "registration_reason": str(frame_gauge_transaction.get("reason")),
                    "registration_acceptance_required": False,
                    "first_high_call_index": boundary_window_evidence.first_high_call_index,
                    "first_high_sigma": boundary_window_evidence.first_high_sigma,
                    "first_high_actual": boundary_window_evidence.first_high_call_index is not None,
                    "process_latent_out_required_before_vae": True,
                    "capture_boundary_witness_requested": bool(band_witness_requested),
                    "extra_h3_nfe": 0,
                    "extra_provider_calls": 0,
                    "extra_vae_calls": 0,
                },
            )
            binding.metrics.event(
                "partitioned_boundary_window_evidence",
                policy="native_boundary_decoder_window_evidence_v1",
                requested_mode=residual_mode,
                capture_boundary_witness_requested=bool(band_witness_requested),
                registration_acceptance_required=False,
                first_high_call_index=boundary_window_evidence.first_high_call_index,
                first_high_sigma=boundary_window_evidence.first_high_sigma,
                tensor_copy_wall_s=boundary_window_evidence.copy_wall_s,
                output_mutated=False,
                diagnostic_only=True,
                extra_h3_nfe=0,
                extra_provider_calls=0,
                extra_sampler_lifetimes=0,
                extra_history_boundaries=0,
                **window_evidence_receipt,
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
            video_boundary_repair_contract=PARTITIONED_VIDEO_BOUNDARY_REPAIR_CONTRACT,
            high_prediction_gauge_bridge_policy=HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
            high_prediction_gauge_bridge_enabled=high_prediction_gauge_bridge_enabled,
            high_prediction_gauge_bridge_support_tokens=(
                bridge_support_tokens if high_prediction_gauge_bridge_enabled else 0
            ),
            high_prediction_gauge_bridge_weights=(
                list(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS) if high_prediction_gauge_bridge_enabled else []
            ),
            high_prediction_gauge_bridge_release_boundary_t=(
                stage_plan.prefix_t + bridge_support_tokens if high_prediction_gauge_bridge_enabled else None
            ),
            high_prediction_gauge_bridge_hardware_verdict="falsified_00715_no_decoded_frame_shift_improvement",
            vae_window_video_policy=VAE_WINDOW_VIDEO_POLICY,
            vae_window_video_repair_armed=vae_window_repair_armed,
            vae_window_video_repair_applied=bool(vae_window_video_repair.get("applied", False)),
            vae_window_video_repair_reason=str(vae_window_video_repair.get("reason", "unknown")),
            vae_window_video_selected_dy_cells=vae_window_video_repair.get("selected_dy_cells"),
            vae_window_video_plan=dict(vae_window_plan or {}),
            high_boundary_context_tokens=int(high_boundary_context.shape[2]),
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
            high_stage_video_guard_requested=0,
            high_stage_prediction_gauge_bridge_requested=(
                bridge_support_tokens if high_prediction_gauge_bridge_enabled else 0
            ),
            post_high_vae_window_video_repair_requested=bool(vae_window_repair_armed),
            post_high_vae_window_video_repair_applied=bool(vae_window_video_repair.get("applied", False)),
            post_high_vae_window_video_policy=VAE_WINDOW_VIDEO_POLICY,
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
    finally:
        if boundary_window_evidence is not None:
            boundary_window_evidence.close()
        if handoff_guidance_reference is not None:
            handoff_guidance_reference.close()


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
