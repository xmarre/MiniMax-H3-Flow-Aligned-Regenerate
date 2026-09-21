"""Low/probe/high scheduler for partitioned exact-prefix continuation.

The released Flow runtime remains untouched.  This module is selected only by the
experimental partitioned node and publishes a new stage contract rather than
reinterpreting the retired Mixed-Grid mode.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from typing import Any

import torch

from .audio_guided_overlap import compare_audio_latent_stages, measure_audio_latent_boundary
from .contracts import H3FlowTrajectory
from .geometry import pack_streams, resize_spatial_5d, unpack_streams
from .handoff import ProgressiveTargetInputConfig, build_handoff_state, deterministic_video_noise
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
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

PARTITIONED_PROGRESSIVE_KEY = "h3_flow_partitioned_progressive_v1"
SOL_RUNTIME_KEY = "sol_h3_runtime_v1"
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


def _validate_fast_main_width16_candidate_configuration(
    *,
    low_probe_execution_source: str,
    prefix_transformer_context: str,
    vdn_linear_diagnostic: str,
    audio_position_domain: str,
    audio_handoff_source: str,
    av_handoff_source: str,
    guidance_trajectory_source: str,
    audio_guided_overlap_mode: str,
    audio_guided_overlap_ticks: int,
) -> bool:
    """Fail-close the production-shaped width-16 candidate to one exact-main path."""

    if (
        normalize_low_probe_execution_source(low_probe_execution_source)
        != PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW
        or audio_guided_overlap_mode != PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
        or int(audio_guided_overlap_ticks) != 16
    ):
        return False

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
    if mismatches:
        raise PartitionedPreflightUnsupported(
            "fast main width-16 candidate requires " + ", ".join(mismatches)
        )
    return True


def _verify_fast_main_width16_sampler_mask(
    runtime_mask: torch.Tensor,
    latent_shapes: list[tuple[int, ...]],
    *,
    ticks: int,
) -> dict[str, Any]:
    """Prove the single-path candidate received the canonical fractional audio ramp."""

    _video_mask, audio_mask = unpack_streams(runtime_mask, latent_shapes)
    temporal_min = audio_mask.amin(dim=(0, 1, 2))
    temporal_max = audio_mask.amax(dim=(0, 1, 2))
    fractional = (temporal_min > 1e-8) & (temporal_max < 1.0 - 1e-8)
    fractional_ticks = int(fractional.sum().item())
    if fractional_ticks != int(ticks):
        raise RuntimeError(
            "fast main width-16 candidate did not receive the requested fractional audio overlap "
            f"(expected {int(ticks)}, observed {fractional_ticks})"
        )
    observed = temporal_min[fractional].to(dtype=torch.float32)
    observed_max = temporal_max[fractional].to(dtype=torch.float32)
    if not torch.equal(observed, observed_max):
        raise RuntimeError("fast main width-16 audio overlap is not uniform across audio rows")
    raw = torch.arange(1, int(ticks) + 1, device=observed.device, dtype=torch.float32) / float(int(ticks) + 1)
    expected = torch.ceil(raw * 256.0) / 256.0
    if not torch.equal(observed, expected):
        raise RuntimeError("fast main width-16 audio overlap does not match the canonical quantized ramp")
    return {
        "sampler_mask_fractional_ticks": fractional_ticks,
        "sampler_mask_ramp_verified": True,
        "sampler_mask_ramp": [float(value) for value in observed.tolist()],
    }


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
    if av_handoff_source != PARTITIONED_AV_HANDOFF_SOURCE_SHADOW:
        mismatches.append("av_handoff_source='source_carrier_uniform_shadow'")
    if guidance_trajectory_source != PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW:
        mismatches.append("guidance_trajectory_source='source_carrier_uniform_shadow'")
    if audio_guided_overlap_mode != PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER:
        mismatches.append("audio_guided_overlap_mode='sampler_mask'")
    if int(audio_guided_overlap_ticks) != 16:
        mismatches.append("audio_guided_overlap_ticks=16")
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
            "partitioned exact-prefix does not inherit retired seam-repair heuristics"
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
    fast_main_width16_candidate = _validate_fast_main_width16_candidate_configuration(
        low_probe_execution_source=low_probe_execution_source,
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
    low_mask = _resize_packed_mask(denoise_mask, target_shapes, source_shapes)

    diagnostic_target_mask, diagnostic_low_mask = _resolve_audio_diagnostic_masks(
        denoise_mask,
        exact_denoise_mask,
        target_shapes,
        source_shapes,
    )

    fast_main_mask_report = None
    if fast_main_width16_candidate:
        fast_main_mask_report = _verify_fast_main_width16_sampler_mask(
            low_mask,
            source_shapes,
            ticks=audio_guided_overlap_ticks,
        )
        binding.metrics.event(
            "partitioned_fast_main_width16_plan",
            source="main_exact_partitioned",
            low_probe_execution_source=low_probe_execution_source,
            prefix_transformer_context=prefix_transformer_context,
            audio_position_domain=audio_position_domain,
            audio_handoff_source=audio_handoff_source,
            av_handoff_source=av_handoff_source,
            guidance_trajectory_source=guidance_trajectory_source,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
            expected_sampler_invocations=3,
            expected_history_boundaries=2,
            shadow_sampler_lifetimes_expected=0,
            production_shaped_single_path=True,
            diagnostic_only=True,
            **fast_main_mask_report,
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
    fast_main_shadow_audio_before = int(
        binding.metrics.counters.get("partitioned_audio_handoff_shadow_low_invocations", 0)
    )
    fast_main_shadow_av_low_before = int(
        binding.metrics.counters.get("partitioned_av_handoff_shadow_low_invocations", 0)
    )
    fast_main_shadow_av_probe_before = int(
        binding.metrics.counters.get("partitioned_av_handoff_shadow_probe_invocations", 0)
    )
    fast_main_partitioned_calls_before = int(binding.metrics.counters.get("partitioned_transformer_calls", 0))
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
        exact_prefix_source = resize_spatial_5d(
            stage_plan.prefix.to(clean_video),
            source_h,
            source_w,
            mode="bicubic",
        )
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
        )
        if rebuilt_shapes != target_shapes:
            raise RuntimeError("partitioned exact-prefix handoff changed caller-visible AV geometry")
        target_video, target_audio = unpack_streams(target_raw, target_shapes)
        if diagnostic_audio_control:
            _source_state_video, source_state_audio = unpack_streams(source_raw, source_shapes)
            audio_copy_exact = torch.equal(source_state_audio, target_audio)
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
        learned_clean = _recover_partitioned_transfer_clean(
            target_video,
            sigma=sigma,
            seed=diagnostic_seed,
        )
        splice_started = time.perf_counter()
        exact_prefix = stage_plan.prefix.to(
            device=learned_clean.device,
            dtype=learned_clean.dtype,
        )
        splice_diagnostics = measure_exact_prefix_splice(
            learned_clean,
            exact_prefix,
        )
        splice_diagnostics.update(
            splice_diagnostic_elapsed_ms=(time.perf_counter() - splice_started) * 1000.0,
            splice_recovery="inverse_conditional_renoise",
            splice_scope="learned_clean_before_exact_prefix_restore",
        )
        restored_clean = learned_clean.clone()
        restored_clean[:, :, : stage_plan.prefix_t] = exact_prefix
        for roi_name, roi_fraction in (("upper45", 0.45), ("full", 1.0)):
            native_trajectory = measure_translation_trajectory(
                learned_clean,
                stage_plan.prefix_t,
                forward_steps=4,
                backward_steps=3,
                roi_fraction=roi_fraction,
                max_shift=4,
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
                stage="learned_native",
                roi=roi_name,
                **native_trajectory,
            )
            binding.metrics.event(
                "partitioned_multiframe_trajectory",
                stage="exact_restored_pre_high",
                roi=roi_name,
                **restored_trajectory,
            )
        binding.metrics.increment("partitioned_splice_diagnostic_runs")
        binding.metrics.increment("partitioned_multiframe_trajectory_runs")
        del restored_clean, learned_clean
        target_video = target_video.clone()
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
            provider_api_version=transfer_metrics.get("provider_api_version"),
            provider_kind=transfer_metrics.get("provider_kind"),
            model_name=transfer_metrics.get("model_name"),
            source_hw=transfer_metrics.get("source_hw"),
            target_hw=transfer_metrics.get("target_hw"),
            temporal_length=transfer_metrics.get("temporal_length"),
            learned_upscale_elapsed_ms=transfer_metrics.get("learned_upscale_elapsed_ms"),
            **splice_diagnostics,
        )

        target_latent_internal = _process_latent_in(base_model, latent_image, target_shapes)
        target_noise = _noise_argument(base_model, target_raw, sigma, target_latent_internal)
        target_noise = _merge_preserved_noise(target_noise, noise, denoise_mask)
        binding.metrics.event(
            "handoff_transfer_wall",
            elapsed_ms=(time.perf_counter() - transfer_started) * 1000.0,
            partitioned_exact_prefix=True,
        )

        if binding.guidance is not None and binding.guidance.mode != "off":
            if binding.trajectory is None:
                raise RuntimeError("partitioned exact-prefix Flow guidance requires an H3_FLOW_TRAJECTORY")
            session_id, chunk_id = _interop_identity(getattr(guider, "model_options", None))
            expected_signature = binding.guidance_conditioning_signature or _conditioning_signature(guider)
            if guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW:
                run = shadow_guidance_run
                if run is None:
                    raise RuntimeError("source-uniform shadow guidance trajectory is unavailable")
                if run.chunk_id != str(chunk_id) or run.session_id != str(session_id):
                    raise RuntimeError("source-uniform shadow guidance trajectory identity drifted")
                if run.conditioning_signature != expected_signature:
                    raise RuntimeError("source-uniform shadow guidance trajectory conditioning drifted")
            else:
                run = binding.trajectory.select(
                    chunk_id=chunk_id,
                    session_id=session_id,
                    conditioning_signature=expected_signature,
                )
            if run.geometry.latent_t != int(target_shapes[0][2]):
                raise RuntimeError("partitioned low trajectory and target video temporal geometry differ")
            binding.active_guidance_run = run
            binding.metrics.event(
                "partitioned_guidance_trajectory_selection",
                source=guidance_trajectory_source,
                run_id=run.run_id,
                exact_samples=len(run.exact_samples()),
                source_hw=(run.geometry.latent_h, run.geometry.latent_w),
                target_hw=(target_h, target_w),
                main_captured_run_id=(committed_low_run.run_id if committed_low_run is not None else None),
                shared_trajectory_handle_preserved=True,
                audio_latent_trajectory_present=False,
                diagnostic_only=(guidance_trajectory_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW),
            )

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
        if diagnostic_audio_control:
            final_internal = _process_latent_in(base_model, result, target_shapes)
            _final_internal_video, final_internal_audio = unpack_streams(final_internal, target_shapes)
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
            del final_internal
        original_video, original_audio = unpack_streams(latent_image, target_shapes)
        if not torch.equal(
            final_video[:, :, : stage_plan.prefix_t],
            original_video[:, :, : stage_plan.prefix_t].to(final_video),
        ):
            raise RuntimeError("partitioned exact-prefix high stage violated exact original-prefix preservation")
        if audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
            if tensor_sha256(denoise_mask) != candidate_high_mask_digest:
                raise RuntimeError("source-carrier audio-position candidate mutated the high-stage sampler mask")
            _mask_video, audio_mask = unpack_streams(denoise_mask, target_shapes)
            protected_audio = audio_mask == 0
            if not bool(protected_audio.any().item()):
                raise RuntimeError("source-carrier audio-position candidate found no protected carried-audio prefix")
            audio_exact = torch.equal(
                final_audio[protected_audio],
                original_audio.to(final_audio)[protected_audio],
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
        if fast_main_width16_candidate:
            shadow_audio_delta = (
                int(binding.metrics.counters.get("partitioned_audio_handoff_shadow_low_invocations", 0))
                - fast_main_shadow_audio_before
            )
            shadow_av_low_delta = (
                int(binding.metrics.counters.get("partitioned_av_handoff_shadow_low_invocations", 0))
                - fast_main_shadow_av_low_before
            )
            shadow_av_probe_delta = (
                int(binding.metrics.counters.get("partitioned_av_handoff_shadow_probe_invocations", 0))
                - fast_main_shadow_av_probe_before
            )
            source_uniform_call_delta = (
                int(binding.metrics.counters.get("partitioned_source_carrier_uniform_transformer_calls", 0))
                - source_carrier_calls_before
            )
            exact_partitioned_call_delta = (
                int(binding.metrics.counters.get("partitioned_transformer_calls", 0))
                - fast_main_partitioned_calls_before
            )
            model_timestep_call_delta = (
                int(binding.metrics.counters.get("partitioned_audio_model_timestep_override_calls", 0))
                - audio_model_timestep_calls_before
            )
            if sampler_invocation_count != 3 or history_boundary_count != 2:
                raise RuntimeError(
                    "fast main width-16 candidate did not preserve exactly low/probe/high sampler lifetimes"
                )
            if shadow_audio_delta or shadow_av_low_delta or shadow_av_probe_delta:
                raise RuntimeError("fast main width-16 candidate unexpectedly executed a shadow sampler lifetime")
            if source_uniform_call_delta != 0:
                raise RuntimeError(
                    "fast main width-16 candidate unexpectedly executed source-uniform transformer calls"
                )
            if exact_partitioned_call_delta <= 0:
                raise RuntimeError("fast main width-16 candidate produced no exact-partitioned transformer calls")
            if model_timestep_call_delta != 0:
                raise RuntimeError("fast main width-16 sampler-mask candidate executed model-timestep audio guidance")
            binding.metrics.event(
                "partitioned_fast_main_width16_complete",
                source="main_exact_partitioned",
                sampler_invocation_count=sampler_invocation_count,
                history_boundary_count=history_boundary_count,
                shadow_audio_low_invocation_delta=shadow_audio_delta,
                shadow_av_low_invocation_delta=shadow_av_low_delta,
                shadow_av_probe_invocation_delta=shadow_av_probe_delta,
                source_uniform_transformer_call_delta=source_uniform_call_delta,
                exact_partitioned_transformer_call_delta=exact_partitioned_call_delta,
                model_timestep_override_call_delta=model_timestep_call_delta,
                raw_audio_owner="main_exact_partitioned",
                clean_video_owner="main_exact_partitioned_probe",
                guidance_trajectory_owner="main_exact_partitioned",
                production_shaped_single_path=True,
                final_exact_prefix_requested=True,
                diagnostic_only=True,
                **(fast_main_mask_report or {}),
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
    "_validate_fast_main_width16_candidate_configuration",
    "_validate_guidance_trajectory_shadow_configuration",
    "_verify_audio_position_domain_diagnostic",
    "_verify_fast_main_width16_sampler_mask",
    "run_partitioned_progressive",
]
